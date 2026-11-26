"""QyranEye online demo (Gradio): upload a short drone video, get the annotated video and the alarms.

Runs exactly the same pipeline as main.py (video_pipeline.py + fall_logic.py + config.yaml),
only on CPU when there is no GPU. Locally:
    .venv/bin/python demo/app.py            # then open http://127.0.0.1:7860
In the Hugging Face Space this file is app.py next to the project files (see demo/build_space.py).
"""
import copy
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "video_pipeline.py").exists() else HERE.parent      # Space: flat; repo: demo/..
sys.path.insert(0, str(ROOT))

import cv2                       # noqa: E402
import gradio as gr              # noqa: E402
import torch                     # noqa: E402
from ultralytics import YOLO     # noqa: E402

import video_pipeline as vp      # noqa: E402

MAX_SECONDS = 30                 # free cloud CPU: longer clips take too long (a fall needs >= 4 s after it)
EXAMPLES = HERE / "examples"
CFG = vp.load_config(ROOT / "config.yaml")
CFG.device = "mps" if torch.backends.mps.is_available() else "cpu"
CFG.show_preview = False
MODEL = YOLO(str(vp.resolve(CFG, CFG.model)))

VIEW_CHOICES = {"Tilted camera (45–60°)": "oblique", "Straight down (90°)": "nadir"}

LEGEND = """
**Colours in the video:** <span style="color:#1a9e1a">■ green</span> standing / walking ·
<span style="color:#d4a000">■ yellow FALL?</span> went down fast, waiting 4 s ·
<span style="color:#d11">■ red FALL</span> **alarm** · <span style="color:#1e7fd6">■ blue LYING</span> lying, no fall seen ·
**CHECK** lying still 10 s without an alarm — the operator should look.
"""


def probe(path):
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (n / fps) if fps > 0 else 0.0


def to_web_mp4(src, dst):
    """OpenCV writes mp4v on Linux, which browsers can't play: re-encode to H.264, max 720p."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale=-2:'min(720,ih)'",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", str(dst)], check=True)


def read_rows(path, cols):
    if not Path(path).exists():
        return []
    return [[r[c] for c in cols] for r in csv.DictReader(open(path, encoding="utf-8"))]


def summary(alarms, checks, view, seconds, took=None):
    lines = []
    if alarms:
        lines.append(f"## 🚨 Fall detected: {len(alarms)} alarm{'s' if len(alarms) > 1 else ''}")
        for a in alarms:
            lines.append(f"- person **#{a['track_id']}** went down at **{float(a['fall_start_s']):.1f} s**, "
                         f"alarm at **{float(a['alarm_s']):.1f} s** (lay still ≥ 4 s after falling)")
    else:
        lines.append("## ✅ No fall alarm")
    for c in checks:
        lines.append(f"- ⚠️ **CHECK** at {float(c['check_s']):.1f} s: person #{c['track_id']} has been lying still "
                     f"for 10 s without an alarm — an operator should look")
    lines.append(f"\nCamera view: **{view}** · analysed {seconds:.1f} s of video" +
                 (f" in {took:.0f} s" if took else ""))
    return "\n".join(lines)


def cleanup_old(max_age_s=3600):
    """Results are served from a temp folder: delete folders older than an hour (no uploads are kept)."""
    for d in Path(tempfile.gettempdir()).glob("qyraneye_*"):
        try:
            if time.time() - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def run_system(video, view, on_progress=lambda f, msg: None, max_seconds=MAX_SECONDS):
    """The real pipeline on one video -> (web mp4 path, alarms, checks, seconds analysed, seconds taken, note)."""
    work = Path(tempfile.mkdtemp(prefix="qyraneye_"))
    src = work / "input.mp4"
    dur = probe(video)
    if dur <= 0:
        raise gr.Error("Could not read this video. Please upload an .mp4 or .mov file.")
    note = ""
    if max_seconds and dur > max_seconds + 0.5:
        note = f"\n\n*Only the first {max_seconds} s of your {dur:.0f} s video were analysed (free cloud CPU).*"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-t", str(max_seconds), "-an",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(src)], check=True)
        dur = max_seconds
    else:
        shutil.copy(video, src)
    cfg = copy.copy(CFG)
    cfg.camera_view = view
    out = work / "out"
    out.mkdir()
    t0 = time.time()
    vp.process_video(src, cfg, MODEL, out,
                     on_frame=lambda t, *_: on_progress(min(0.99, t / max(dur, 0.1)), f"Analysing {t:.1f} / {dur:.1f} s"))
    took = time.time() - t0
    on_progress(1, "Preparing the video…")
    web = work / "result.mp4"
    to_web_mp4(out / "input_fall.mp4", web)
    alarms = list(csv.DictReader(open(out / "input_events.csv", encoding="utf-8")))
    checks = list(csv.DictReader(open(out / "input_checks.csv", encoding="utf-8")))
    return web, alarms, checks, dur, took, note


def analyse(video, view_label, progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a video first.")
    view = VIEW_CHOICES[view_label]
    cleanup_old()
    try:
        progress(0, desc="Detecting people…")
        web, alarms, checks, dur, took, note = run_system(video, view, lambda f, m: progress(f, desc=m))
        table = [["ALARM (fall)", a["track_id"], a["fall_start_s"], a["alarm_s"]] for a in alarms] + \
                [["CHECK (lying still)", c["track_id"], "", c["check_s"]] for c in checks]
        return str(web), summary(alarms, checks, view, dur, took) + note, table
    except gr.Error:
        raise
    except Exception as e:                                   # show a readable error instead of a crash
        raise gr.Error(f"Processing failed: {e}") from e


def load_example(name):
    meta = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    return str(EXAMPLES / f"{name}.mp4"), meta["summary"] + f"\n\n*{meta['description']}*  \nSource: {meta['credit']}"


def example_names():
    """(label shown, file name) pairs; label is in the example's json."""
    files = sorted(p.stem for p in EXAMPLES.glob("*.json")) if EXAMPLES.exists() else []
    return [(json.loads((EXAMPLES / f"{n}.json").read_text(encoding="utf-8")).get("title", n), n) for n in files]


ABOUT = """
### What QyranEye does
QyranEye watches **recorded drone video** and raises an alarm when a person **falls fast and then lies still**
for 4 seconds. A human operator then checks the alarm — the system finds candidates, a person decides.

**How:** person detector (YOLO26s) → tracking with drone-motion correction → body posture
(box shape; for people ≥ 80 px and a tilted camera, body keypoints on an enlarged crop) → speed of the
shoulders relative to the ankles (fast = fall, slow = lay down) → wait 4 s, check the person was really seen
lying and not moving → **alarm**. Lying still for 10 s without an alarm → **CHECK** for the operator.

### Honest limits (prototype)
- Tested on public datasets (Okutama-Action, NOMAD, MOD20) and **one** real fall video; our own drone filming
  (FIMI X8 SE, 5–20 m) is planned next. No accuracy claim for real falls yet.
- Lying down quickly on purpose looks like a fall. People hidden in grass, very small people (< 80 px) and the
  straight-down view are the weakest cases. Tell the system the camera view — a wrong view causes errors.
- This free demo runs on a cloud CPU: max 30 s per video, about 1–5 minutes. Uploaded videos are processed in a
  temporary folder and deleted; nothing is kept.

### Credits
Detector and keypoints: Ultralytics YOLO26 (AGPL-3.0). Example videos: Mixkit (free licence), Pexels (free licence),
NOMAD dataset (A. M. Russell Bernal et al., WACV 2024, CC BY 4.0).
"""

with gr.Blocks(title="QyranEye — fall detection from drone video", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("# 🚁 QyranEye\n**Finds people who fell and lie still in drone video — so rescuers check the right "
                "spot first.** Student project, prototype.")
    with gr.Tab("▶ Examples (instant)"):
        names = example_names()
        with gr.Row():
            ex_pick = gr.Radio(names, value=names[0][1] if names else None, label="Choose an example")
        with gr.Row():
            ex_video = gr.Video(label="Result", autoplay=True, height=460)
            ex_text = gr.Markdown()
        gr.Markdown(LEGEND)
        ex_pick.change(load_example, ex_pick, [ex_video, ex_text])
        if names:
            demo.load(load_example, ex_pick, [ex_video, ex_text])
    with gr.Tab("⬆ Try your own video"):
        gr.Markdown(f"Upload a **drone video up to {MAX_SECONDS} s** (longer videos: the first {MAX_SECONDS} s are used). "
                    "Best: 5–20 m altitude, people visible, camera not moving too fast. "
                    "Processing on the free cloud CPU takes about **1–5 minutes**. "
                    "A fall is confirmed only after the person lies still for 4 s, so keep at least 4 s after a fall.")
        with gr.Row():
            with gr.Column(scale=1):
                up = gr.Video(label="Your video (.mp4 / .mov)", sources=["upload"])
                view = gr.Radio(list(VIEW_CHOICES), value=list(VIEW_CHOICES)[0], label="Camera view")
                go = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=2):
                res_video = gr.Video(label="Result", height=460)
                res_text = gr.Markdown()
                res_table = gr.Dataframe(headers=["Event", "Person", "Went down, s", "At, s"], interactive=False)
        gr.Markdown(LEGEND)
        go.click(analyse, [up, view], [res_video, res_text, res_table], concurrency_limit=1)
    with gr.Tab("ℹ How it works"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(max_file_size="300mb", theme=gr.themes.Soft())
