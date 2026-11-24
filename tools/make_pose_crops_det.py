"""
Картинки для нейросети позы v2 — по рамкам ДЕТЕКТОРА, как в работе системы.

Отличия от make_pose_crops.py (v1, рамки из разметки):
  1. На обучающих кадрах Okutama запускается тот же детектор (yolo26s), что в системе.
     Каждая его рамка сопоставляется с разметкой:
       перекрытие с размеченным человеком >= 0.3  -> поза этого человека (lying/sitting/upright);
       ни с кем не перекрывается (< 0.1)          -> not_person (сумка, стул, тень...);
       что-то среднее                              -> пропускаем (непонятно).
  2. Картинка вырезается той же функцией pose_crop, что и в системе (video_pipeline.py).
  3. Половина обучающих картинок «портится как в видео»: уменьшение, размытие, сильное JPEG-сжатие.
     Проверочные картинки не портятся.

Деление по видео — как в v1 (те же 7 проверочных видео), 1.1.1 исключено (оно в тестовых роликах).

Запуск:
    python tools/make_pose_crops_det.py okutama_train okutama_train/pose_crops_v2
"""
import argparse
import random
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO
from video_pipeline import detect, load_config, pose_crop

EXCLUDE = {"1.1.1"}
VAL_VIDEOS = {"1.1.4", "1.2.8", "2.1.4", "2.2.2", "1.1.3", "1.2.7", "2.2.9"}
STEP_LYING_VIDEO, STEP_OTHER_VIDEO = 4, 10   # в видео с лежащими кадры берём чаще
CAP = {"train": {"sitting": 6000, "upright": 12000, "not_person": 6000},
       "val": {"sitting": 2000, "upright": 4000, "not_person": 2000}}
SCALE = 1 / 3                                # разметка в 4K, кадры 1280x720


def pose_class(actions):
    if "Lying" in actions:
        return "lying"
    if "Sitting" in actions:
        return "sitting"
    if actions & {"Standing", "Walking", "Running"}:
        return "upright"
    return None


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def degrade(img, rnd):
    """Испортить картинку «как в видео с дрона»: меньше деталей, размытие, артефакты сжатия."""
    h, w = img.shape[:2]
    s = rnd.uniform(0.35, 0.9)
    img = cv2.resize(cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA),
                     (w, h), interpolation=cv2.INTER_LINEAR)
    if rnd.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), rnd.uniform(0.3, 1.2))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rnd.randint(25, 70)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_root")
    ap.add_argument("out")
    a = ap.parse_args()
    root, out = Path(a.train_root), Path(a.out)
    proj = Path(__file__).resolve().parent.parent
    cfg = load_config(proj / "config.yaml")
    model = YOLO(str(proj / cfg.model))

    frame_dirs = {p.name: p for p in root.glob("Drone*/*/Extracted-Frames-1280x720/*") if p.is_dir()}
    gt = defaultdict(lambda: defaultdict(list))          # видео -> кадр -> [(класс|None, рамка 720p)]
    lying_videos = set()
    for lab in sorted((root / "Labels/MultiActionLabels/3840x2160").glob("*.txt")):
        v = lab.stem
        if v in EXCLUDE or v not in frame_dirs:
            continue
        for line in open(lab, encoding="utf-8", errors="ignore"):
            p = shlex.split(line)
            if len(p) < 10 or int(p[6]) == 1:            # lost
                continue
            cls = pose_class(set(p[10:]))
            if cls == "lying":
                lying_videos.add(v)
            occluded = int(p[7]) == 1
            gt[v][int(p[5])].append((None if occluded else cls, tuple(int(x) * SCALE for x in p[1:5])))

    items = defaultdict(list)                            # (split, класс) -> [(видео, кадр, рамка детектора)]
    low = min(0.1, cfg.conf)
    for v in sorted(gt):
        split = "val" if v in VAL_VIDEOS else "train"
        step = STEP_LYING_VIDEO if v in lying_videos else STEP_OTHER_VIDEO
        frames = sorted(f for f in gt[v] if f % step == 0)
        for f in frames:
            img = cv2.imread(str(frame_dirs[v] / f"{f}.jpg"))
            if img is None:
                continue
            det = detect(model, img, cfg, low)
            for box, conf in zip(det.xyxy.tolist(), det.conf.tolist()):
                if conf < cfg.conf:                      # в системе показываются только уверенные треки
                    continue
                best, best_iou = None, 0.0
                for cls, g in gt[v][f]:
                    o = iou(box, g)
                    if o > best_iou:
                        best, best_iou = cls, o
                if best_iou >= 0.3 and best is not None:
                    items[split, best].append((v, f, tuple(box)))
                elif best_iou < 0.1:
                    items[split, "not_person"].append((v, f, tuple(box)))
        print(v, split, {k[1]: len(x) for k, x in items.items() if k[0] == split}, flush=True)

    rnd = random.Random(0)
    for (split, cls), lst in items.items():
        cap = CAP[split].get(cls)
        if cap and len(lst) > cap:
            items[split, cls] = rnd.sample(lst, cap)

    by_frame = defaultdict(list)
    for (split, cls), lst in items.items():
        for v, f, box in lst:
            by_frame[v, f].append((split, cls, box))
    counts = Counter()
    for (v, f), objs in sorted(by_frame.items()):
        img = cv2.imread(str(frame_dirs[v] / f"{f}.jpg"))
        for k, (split, cls, box) in enumerate(objs):
            crop = pose_crop(img, box, cfg.pose_imgsz)
            if split == "train" and rnd.random() < 0.5:
                crop = degrade(crop, rnd)
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{v}_{f}_{k}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[split, cls] += 1
    for split in ("train", "val"):
        print(split, {c: counts[split, c] for c in ("lying", "sitting", "upright", "not_person")})


if __name__ == "__main__":
    main()
