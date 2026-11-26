"""
QyranEye — веб-демо: загрузить видео с дрона -> посмотреть, что нашла система.

Это только оболочка над настоящим конвейером: вызывает video_pipeline.process_video
с config.yaml проекта и показывает результат в браузере. fall_logic.py и config.yaml
не меняются; пороги, тревоги FALL и сигналы CHECK — те же, что у main.py.

Запуск (из папки проекта):
    .venv/bin/pip install flask
    .venv/bin/python web/app.py
Откроется http://127.0.0.1:8080
"""

import copy
import csv
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path

import cv2
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

WEB_DIR = Path(__file__).resolve().parent
REPO = WEB_DIR.parent
sys.path.insert(0, str(REPO))

from fall_logic import FALLEN, FALLING, LYING, UPRIGHT  # noqa: E402
from video_pipeline import VIDEO_EXTS, load_config, process_video, resolve  # noqa: E402

RUNS_DIR = WEB_DIR / "runs"            # загруженные видео и результаты (в git не хранятся)
PORT = 8080
MAX_TIMELINE_POINTS = 400

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 ** 3          # до 3 ГБ (4K с дрона)

jobs = {}
jobs_lock = threading.Lock()
pipeline_lock = threading.Lock()       # одна обработка за раз: модель и трекер не делим между потоками
_models = {}


def base_config():
    cfg = load_config(REPO / "config.yaml")
    cfg.show_preview = False           # в веб-демо окна OpenCV нет
    if str(cfg.device) == "mps":
        try:
            import torch
            if not torch.backends.mps.is_available():
                cfg.device = "cpu"     # не Mac с Apple Silicon (например, сервер) — считаем на процессоре
        except Exception:  # noqa: BLE001
            cfg.device = "cpu"
    return cfg


def get_models(cfg):
    """YOLO грузим один раз на всё время работы сервера (как process_source в main.py)."""
    if "model" not in _models:
        from ultralytics import YOLO
        _models["model"] = YOLO(str(resolve(cfg, cfg.model)))
        _models["pose"] = (YOLO(str(resolve(cfg, cfg.pose_model)))
                           if cfg.pose_mode in ("classifier", "hybrid") else None)
    return _models["model"], _models["pose"]


def set_job(job_id, **kw):
    with jobs_lock:
        jobs[job_id].update(kw)


def read_csv(path):
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_browser_playable(video):
    """OpenCV на Mac пишет H.264 (играет в браузере). Если записался MPEG-4 (mp4v) и есть ffmpeg —
    перекодируем, иначе браузер видео не покажет (его всё равно можно скачать)."""
    cap = cv2.VideoCapture(str(video))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).lower()
    if codec in ("avc1", "h264"):
        return True
    if not shutil.which("ffmpeg"):
        return False
    tmp = video.with_name(video.stem + "_h264.mp4")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(tmp)], capture_output=True)
    if r.returncode == 0 and tmp.is_file():
        tmp.replace(video)
        return True
    return False


def snapshot(video, t, box, out_path):
    """Кадр из размеченного видео в момент события (рамка уже нарисована конвейером)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(float(t), 0) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if box:
        # box — в пикселях исходного видео; размеченное видео может быть уменьшено
        h, w = frame.shape[:2]
        sx, sy = w / box["src_w"], h / box["src_h"]
        x1, y1, x2, y2 = box["x1"] * sx, box["y1"] * sy, box["x2"] * sx, box["y2"] * sy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1, h / 4)            # вокруг человека, но не меньше четверти кадра
        X1, Y1 = int(max(cx - half, 0)), int(max(cy - half * 0.75, 0))
        X2, Y2 = int(min(cx + half, w)), int(min(cy + half * 0.75, h))
        if X2 - X1 > 20 and Y2 - Y1 > 20:
            frame = frame[Y1:Y2, X1:X2]
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_path.name


def run_job(job_id, video_path, view):
    out_dir = RUNS_DIR / job_id
    try:
        set_job(job_id, state="queued")
        with pipeline_lock:
            cfg = base_config()
            cfg.camera_view = view
            set_job(job_id, state="loading", device=str(cfg.device))
            model, pose_model = get_models(cfg)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError("Не удалось открыть видео. Попробуйте .mp4 или .mov.")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            duration = total / fps if total else 0
            set_job(job_id, state="running", duration=round(duration, 2))

            timeline, info = [], {}
            started = time.time()

            def on_frame(t, frame, people, warp, monitor, visible):
                states = [tr.state for tr in visible]
                timeline.append([round(t, 2), len(visible),
                                 states.count(UPRIGHT), states.count(LYING),
                                 states.count(FALLING), states.count(FALLEN)])
                info["view"], info["view_sure"] = monitor.view, monitor.view_sure
                info["view_reason"] = monitor.view_reason
                info["cancelled"] = monitor.cancelled
                elapsed = time.time() - started
                set_job(job_id, t=round(t, 2), elapsed=round(elapsed, 1))

            events = process_video(video_path, cfg, model, out_dir, pose_model, on_frame=on_frame)
            if events is None:
                raise ValueError("Не удалось открыть видео. Попробуйте .mp4 или .mov.")

        stem = video_path.stem
        out_video = out_dir / f"{stem}_fall.mp4"
        playable = make_browser_playable(out_video)
        complete = (out_dir / f"{stem}_events.csv").is_file()
        alarms = read_csv(out_dir / f"{stem}_events.csv")
        checks = read_csv(out_dir / f"{stem}_checks.csv")

        snaps = out_dir / "snapshots"
        snaps.mkdir(exist_ok=True)
        for i, e in enumerate(alarms):
            box = dict(x1=float(e["x1"]), y1=float(e["y1"]), x2=float(e["x2"]), y2=float(e["y2"]),
                       src_w=src_w, src_h=src_h)
            e["snapshot"] = snapshot(out_video, float(e["alarm_s"]), box, snaps / f"alarm_{i:02d}.jpg")
        for i, e in enumerate(checks):
            box = dict(x1=float(e["x1"]), y1=float(e["y1"]), x2=float(e["x2"]), y2=float(e["y2"]),
                       src_w=src_w, src_h=src_h)
            e["snapshot"] = snapshot(out_video, float(e["check_s"]), box, snaps / f"check_{i:02d}.jpg")

        step = max(1, len(timeline) // MAX_TIMELINE_POINTS)
        people = [p[1] for p in timeline]
        result = dict(
            video_name=video_path.name, width=src_w, height=src_h, fps=round(fps, 2),
            duration=round(duration, 2), complete=complete, playable=playable,
            processing_s=round(time.time() - started, 1), device=str(cfg.device),
            view=info.get("view"), view_sure=info.get("view_sure"), view_reason=info.get("view_reason", ""),
            view_requested=view, cancelled=info.get("cancelled", 0),
            people_max=max(people) if people else 0,
            people_avg=round(sum(people) / len(people), 1) if people else 0,
            alarms=alarms, checks=checks, timeline=timeline[::step],
            files=dict(video=out_video.name, events=f"{stem}_events.csv", checks=f"{stem}_checks.csv"),
        )
        set_job(job_id, state="done", result=result)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        set_job(job_id, state="error", error=str(e) or type(e).__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def api_analyze():
    f = request.files.get("video")
    view = request.form.get("view", "oblique")
    if view not in ("oblique", "nadir", "auto"):
        return jsonify(error="Неизвестный ракурс."), 400
    if not f or not f.filename:
        return jsonify(error="Файл не выбран."), 400
    name = secure_filename(f.filename) or "video.mp4"
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        return jsonify(error="Поддерживаются видео: " + ", ".join(sorted(VIDEO_EXTS))), 400

    job_id = uuid.uuid4().hex[:10]
    job_dir = RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / name
    f.save(video_path)

    with jobs_lock:
        jobs[job_id] = dict(state="queued", t=0, duration=0, elapsed=0, result=None, error=None)
    threading.Thread(target=run_job, args=(job_id, video_path, view), daemon=True).start()
    return jsonify(job_id=job_id)


@app.get("/api/status/<job_id>")
def api_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job)


@app.get("/runs/<job_id>/<path:filename>")
def run_file(job_id, filename):
    if job_id not in jobs:
        abort(404)
    return send_from_directory(RUNS_DIR / job_id, filename,
                               as_attachment=request.args.get("download") == "1")


if __name__ == "__main__":
    RUNS_DIR.mkdir(exist_ok=True)
    print(f"QyranEye веб-демо: http://127.0.0.1:{PORT}  (остановить: Ctrl+C)")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
