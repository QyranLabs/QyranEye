"""Сколько людей находит детектор в зависимости от размера человека в кадре (то есть от высоты дрона).

Берёт каждый N-й кадр роликов Okutama с разметкой, прогоняет детектор в нескольких вариантах
(разрешение, нарезка на куски) и считает долю найденных размеченных людей по размерам.
Трекер и логика падения не участвуют: это проверка только детектора, самого первого звена.

    .venv/bin/python eval/detect_by_size.py                 # 11 роликов, каждый 45-й кадр, ~15 мин
    .venv/bin/python eval/detect_by_size.py --step 90       # вдвое быстрее
    .venv/bin/python eval/detect_by_size.py --variants base,tiles4k_3x3

Размер человека — длинная сторона размеченной рамки в пикселях кадра 1080p (то, что видит система
при max_height 1080). Человек «найден», если рамка детектора перекрывает размеченную с IoU >= 0.2.
Разметка части роликов сдвинута по времени — сдвиг подбирается для каждого ролика (как в evaluate.py).
«Лишние» рамки — рамки детектора, не совпавшие ни с кем из размеченных: это и ошибки, и неразмеченные
люди (зрители, прохожие), поэтому это верхняя граница ложных срабатываний.
"""
import argparse
import copy
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT), str(PROJECT / "tools")]
import okutama_extract as ox                      # noqa: E402
import video_pipeline as vp                       # noqa: E402

LBL = PROJECT / "drone videos/TestSetVideos/Labels/SingleActionTrackingLabels/3840x2160"
VIDEOS = [(PROJECT / "drone videos/Sample/1.1.1.mov", PROJECT / "drone videos/Sample/1.1.1.txt")] + [
    (v, LBL / f"{v.stem}.txt") for v in sorted((PROJECT / "drone videos/TestSetVideos").glob("Drone*/*/*.mp4"))]
BINS = [0, 30, 45, 60, 80, 120, 10_000]           # длинная сторона человека, px в кадре 1080p
SHIFTS = range(-40, 41, 2)
IOU_MIN = 0.2

# Варианты детектора: (высота кадра, нарезка (rows, cols) или None, imgsz)
VARIANTS = {
    "base":        (1080, None, 1920),     # как сейчас в config.yaml
    "tiles1080":   (1080, (2, 2), 1280),   # как config_tiles.yaml
    "imgsz2560":   (1080, None, 2560),     # тот же кадр, YOLO смотрит крупнее
    "tiles4k_2x2": (2160, (2, 2), 1920),   # исходный 4K, 4 куска
    "tiles4k_3x3": (2160, (3, 3), 1280),   # исходный 4K, 9 кусков
}


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter)


def match(gt, det):
    """Жадное сопоставление по IoU -> (найден ли каждый размеченный, сколько рамок лишние)."""
    m = iou_matrix(gt, det)
    found, used = [False] * len(gt), set()
    for i, j in sorted(((i, j) for i in range(len(gt)) for j in range(len(det)) if m[i, j] >= IOU_MIN),
                       key=lambda ij: -m[ij]):
        if not found[i] and j not in used:
            found[i] = True
            used.add(j)
    return found, len(det) - len(used)


def labels_by_frame(path):
    out = defaultdict(list)
    for tid, rows in ox.parse_labels(path).items():
        for frame, action, box, lost in rows:
            if not lost:
                out[frame].append((tuple(v / 2 for v in box), action))      # 4K -> 1080p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step", type=int, default=45, help="брать каждый N-й кадр (по умолчанию 45 = 1.5 с)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--conf", type=float, default=None, help="порог уверенности (по умолчанию conf из config.yaml)")
    a = ap.parse_args()
    names = a.variants.split(",")
    base = vp.load_config(PROJECT / "config.yaml")
    conf = a.conf if a.conf is not None else base.conf
    from ultralytics import YOLO
    model = YOLO(str(vp.resolve(base, base.model)))
    cfgs = {}
    for n in names:
        h, tiles, imgsz = VARIANTS[n]
        c = copy.deepcopy(base)
        c.imgsz = imgsz
        c.tiles.enabled = tiles is not None
        if tiles:
            c.tiles.rows, c.tiles.cols, c.tiles.imgsz = *tiles, imgsz
        cfgs[n] = (h, c)

    # stats[вариант][(bin, lying)] = [найдено, всего]
    stats = {n: defaultdict(lambda: [0, 0]) for n in names}
    extra, secs, n_frames = defaultdict(int), defaultdict(float), 0
    for video, lpath in VIDEOS:
        if not video.exists() or not lpath.exists():
            print(f"нет {video.name} или разметки — пропускаю")
            continue
        lab = labels_by_frame(lpath)
        ars = sorted((b[2] - b[0]) / max(1.0, b[3] - b[1]) for v in lab.values() for b, _ in v)
        view = "nadir" if ars and ars[len(ars) // 2] > 0.8 else "oblique"   # сверху человек ≈ квадратный
        cap = cv2.VideoCapture(str(video))
        dets = {n: {} for n in names}                    # кадр -> рамки (в 1080p)
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % a.step == 0:
                _, frame = cap.retrieve()
                small = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
                for n in names:
                    h, c = cfgs[n]
                    img, k = (small, 1.0) if h == 1080 else (frame, 0.5)
                    t0 = time.time()
                    d = vp.detect(model, img, c, conf)
                    secs[n] += time.time() - t0
                    # min_person_px задан для кадра 1080p — на 4K отсекаем по пересчитанному размеру
                    xy = d.xyxy * k
                    size = np.maximum(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]) if len(xy) else xy[:, 0]
                    dets[n][idx] = xy[size >= c.min_person_px]
                n_frames += 1
            idx += 1
        cap.release()
        # сдвиг разметки подбираем по варианту base (или первому), один на ролик
        ref = names[0]
        best = max(SHIFTS, key=lambda s: sum(sum(match([b for b, _ in lab.get(f - s, [])], d)[0])
                                             for f, d in dets[ref].items()))
        line = []
        for n in names:
            got = tot = 0
            for f, d in dets[n].items():
                gt = lab.get(f - best, [])
                found, ex = match([b for b, _ in gt], d)
                extra[n] += ex
                for (b, act), ok in zip(gt, found):
                    size = max(b[2] - b[0], b[3] - b[1])
                    bi = np.searchsorted(BINS, size, side="right") - 1
                    s = stats[n][(bi, act == "Lying", view)]
                    s[0] += ok; s[1] += 1
                    got += ok; tot += 1
            line.append(f"{n} {100 * got / max(tot, 1):.0f}%")
        print(f"{video.name} ({view}): сдвиг {best}, кадров {len(dets[ref])}: " + ", ".join(line), flush=True)

    frames = n_frames
    print(f"\nВсего кадров: {frames}. Доля найденных размеченных людей (порог уверенности {conf}):\n")
    head = "| Размер человека (1080p) | людей | " + " | ".join(names) + " |"
    print(head)
    print("|" + "---|" * (len(names) + 2))
    nb, views = len(BINS) - 1, ("oblique", "nadir")
    def row(title, keys):
        tot = sum(stats[names[0]][k][1] for k in keys)
        cells = [f"{100 * sum(stats[n][k][0] for k in keys) / tot:.0f} %" if tot else "—" for n in names]
        print(f"| {title} | {tot} | " + " | ".join(cells) + " |")
    for bi in range(nb):
        lo, hi = BINS[bi], BINS[bi + 1]
        title = f"< {hi} px" if lo == 0 else (f"≥ {lo} px" if hi >= 10_000 else f"{lo}–{hi} px")
        row(title, [(bi, l, v) for l in (False, True) for v in views])
    row("**все**", [(bi, l, v) for bi in range(nb) for l in (False, True) for v in views])
    for v, name in (("oblique", "под углом"), ("nadir", "сверху")):
        row(f"{name}: все", [(bi, l, v) for bi in range(nb) for l in (False, True)])
        row(f"{name}: < 45 px", [(bi, l, v) for bi in range(2) for l in (False, True)])
        row(f"{name}: лежащие", [(bi, True, v) for bi in range(nb)])
    row("лежащие, < 60 px", [(bi, True, v) for bi in range(3) for v in views])
    row("лежащие, все", [(bi, True, v) for bi in range(nb) for v in views])
    print("| лишние рамки на кадр | | " + " | ".join(f"{extra[n] / frames:.2f}" for n in names) + " |")
    print("| секунд на кадр (детектор) | | " + " | ".join(f"{secs[n] / frames:.2f}" for n in names) + " |")


if __name__ == "__main__":
    main()
