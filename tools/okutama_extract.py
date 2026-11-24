"""
Okutama-Action -> короткие ролики, где люди ЛЕЖАТ (для тестов QyranEye).

Разметка Okutama: каждая строка
    track_id xmin ymin xmax ymax frame lost occluded generated "Person" "Action" [...]
Берём файлы SingleActionTrackingLabels (у человека постоянный ID).

Для каждого видео:
  1. находим отрезки, где у человека действие "Lying" (не меньше --min-lying с);
  2. помечаем, был ли перед этим тот же человек в другой позе (переход
     стоял -> лёг = самое похожее на падение, что есть в датасете);
  3. склеиваем пересекающиеся отрезки и вырезаем их (+ --pre с до и --post с после)
     в mp4 1080p без звука.
Итог: папка с роликами + index.csv (что, где, какой человек).

Запуск:
    python tools/okutama_extract.py <папка_с_видео> <папка_с_разметкой> <папка_вывода>
"""
import argparse
import csv
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def parse_labels(path):
    """-> {track_id: [(frame, action, box, lost)]}"""
    tracks = defaultdict(list)
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = shlex.split(line)
            if len(parts) < 10:
                continue
            tid, x1, y1, x2, y2, frame, lost = (int(float(v)) for v in parts[:7])
            actions = [a for a in parts[10:] if a]
            action = "Lying" if "Lying" in actions else (actions[0] if actions else "")
            tracks[tid].append((frame, action, (x1, y1, x2, y2), lost))
    for v in tracks.values():
        v.sort()
    return tracks


def lying_segments(tracks, fps, min_lying):
    """Отрезки (start_frame, end_frame, tid, had_transition)."""
    segs = []
    for tid, rows in tracks.items():
        rows = [r for r in rows if r[3] == 0]          # только видимые кадры
        i = 0
        while i < len(rows):
            if rows[i][1].lower() != "lying":
                i += 1
                continue
            j = i
            while j + 1 < len(rows) and rows[j + 1][1].lower() == "lying" \
                    and rows[j + 1][0] - rows[j][0] <= fps:   # разрыв не больше 1 с
                j += 1
            start, end = rows[i][0], rows[j][0]
            if (end - start) / fps >= min_lying:
                before = [r for r in rows[:i] if start - r[0] <= 5 * fps]
                transition = any(r[1].lower() != "lying" for r in before)
                segs.append((start, end, tid, transition))
            i = j + 1
    return segs


def merge(segs, pre, post, nframes):
    """Склеить пересекающиеся окна [start-pre, end+post]."""
    wins = sorted((max(0, s - pre), min(nframes - 1, e + post) if nframes else e + post, tid, tr)
                  for s, e, tid, tr in segs)
    out = []
    for s, e, tid, tr in wins:
        if out and s <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], e)
            out[-1]["tids"].add(tid)
            out[-1]["transition"] |= tr
        else:
            out.append(dict(start=s, end=e, tids={tid}, transition=tr))
    return out


def probe(video):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return fps, n


def cut(video, out, start_s, dur_s, height):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start_s:.3f}", "-i", str(video),
           "-t", f"{dur_s:.3f}", "-an", "-vf", f"scale=-2:'min({height},ih)'",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", str(out)]
    subprocess.run(cmd, check=True)


def find_label(labels_dir, stem):
    cands = [p for p in labels_dir.rglob(f"{stem}.txt")]      # точное имя: 1.2.1 != 1.2.10
    pref = [p for p in cands if "SingleActionTracking" in str(p)]
    return (pref or cands or [None])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--pre", type=float, default=3.0, help="секунд до того, как человек лёг")
    ap.add_argument("--post", type=float, default=6.0, help="секунд после (чтобы сработал таймер 5 с)")
    ap.add_argument("--min-lying", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()

    vdir, ldir, odir = Path(a.videos), Path(a.labels), Path(a.out)
    odir.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in vdir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"Нет видео в {vdir}")

    rows = []
    for v in videos:
        lab = find_label(ldir, v.stem)
        if lab is None:
            print(f"[skip] {v.name}: нет разметки")
            continue
        fps, n = probe(v)
        segs = lying_segments(parse_labels(lab), fps, a.min_lying)
        wins = merge(segs, round(a.pre * fps), round(a.post * fps), n)
        print(f"{v.name}: лежащих отрезков {len(segs)}, роликов {len(wins)}")
        for k, w in enumerate(wins, 1):
            tag = "transition" if w["transition"] else "lying"
            name = f"{v.stem}_{k:02d}_{tag}.mp4"
            s, d = w["start"] / fps, (w["end"] - w["start"]) / fps
            if d < 1.0:                                   # пустой или слишком короткий отрезок
                continue
            cut(v, odir / name, s, d, a.height)
            rows.append(dict(clip=name, source=v.name, start_s=round(s, 2), dur_s=round(d, 2),
                             people=" ".join(map(str, sorted(w["tids"]))),
                             stood_then_lay=w["transition"]))

    with open(odir / "index.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["clip", "source", "start_s", "dur_s", "people", "stood_then_lay"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"Готово: {len(rows)} роликов -> {odir}")


if __name__ == "__main__":
    main()
