"""
Нарезка картинок людей из ОБУЧАЮЩЕЙ части Okutama-Action для нейросети позы.

Каждая картинка — один человек (квадрат вокруг его рамки с запасом), в папке своего класса:
    lying   — в разметке есть действие Lying
    sitting — есть Sitting (и нет Lying)
    upright — есть Standing / Walking / Running (и нет Lying, Sitting)
Люди без этих действий (только Reading, Calling и т.п.) пропускаются: позу не знаем.

Деление на обучение и проверку — ПО ВИДЕО, а не по кадрам: соседние кадры почти
одинаковые, и при делении по кадрам нейросеть «подсмотрела» бы ответы.
Видео 1.1.1 не используется: оно есть в наших тестовых роликах (okutama_lying/).

Запуск:
    python tools/make_pose_crops.py okutama_train okutama_train/pose_crops
Получится формат, который понимает Ultralytics:
    pose_crops/train/{lying,sitting,upright}/*.jpg
    pose_crops/val/{lying,sitting,upright}/*.jpg
"""
import argparse
import random
import shlex
from collections import Counter, defaultdict
from pathlib import Path

import cv2

EXCLUDE = {"1.1.1"}                                   # в тестовых роликах
VAL_VIDEOS = {"1.1.4", "1.2.8", "2.1.4", "2.2.2",     # с лежащими (~20% всех лежащих)
              "1.1.3", "1.2.7", "2.2.9"}              # без лежащих
FRAME_STEP = {"lying": 2, "sitting": 5, "upright": 10}  # лежащих мало — берём чаще
CAP = {"train": {"sitting": 6000, "upright": 12000},    # чтобы стоящих не было в 30 раз больше
       "val": {"sitting": 2000, "upright": 4000}}
SCALE = 1 / 3            # разметка в координатах 4K, кадры 1280x720
CONTEXT = 1.3            # квадрат = длинная сторона рамки * 1.3 (немного фона вокруг)
OUT_SIZE = 96            # сохраняем 96x96
MIN_SIDE = 8             # рамки меньше (в пикселях 720p) — пропускаем


def pose_class(actions):
    if "Lying" in actions:
        return "lying"
    if "Sitting" in actions:
        return "sitting"
    if actions & {"Standing", "Walking", "Running"}:
        return "upright"
    return None


def crop_square(img, x1, y1, x2, y2):
    """Квадрат вокруг рамки с запасом; за краем кадра — повтор крайних пикселей."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * CONTEXT / 2
    a, b = int(round(cx - half)), int(round(cy - half))
    side = int(round(2 * half))
    pad = side + 2
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = big[b + pad:b + pad + side, a + pad:a + pad + side]
    return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA if side > OUT_SIZE else cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_root", help="папка okutama_train (Drone1/, Drone2/, Labels/)")
    ap.add_argument("out", help="куда сложить картинки")
    a = ap.parse_args()
    root, out = Path(a.train_root), Path(a.out)

    frame_dirs = {p.name: p for p in root.glob("Drone*/*/Extracted-Frames-1280x720/*") if p.is_dir()}
    labels = sorted((root / "Labels/MultiActionLabels/3840x2160").glob("*.txt"))

    items = defaultdict(list)          # (split, класс) -> [(видео, кадр, рамка)]
    for lab in labels:
        video = lab.stem
        if video in EXCLUDE or video not in frame_dirs:
            continue
        split = "val" if video in VAL_VIDEOS else "train"
        for line in open(lab, encoding="utf-8", errors="ignore"):
            p = shlex.split(line)
            if len(p) < 10 or int(p[6]) == 1 or int(p[7]) == 1:   # lost или occluded
                continue
            cls = pose_class(set(p[10:]))
            frame = int(p[5])
            if cls is None or frame % FRAME_STEP[cls]:
                continue
            x1, y1, x2, y2 = (int(v) * SCALE for v in p[1:5])
            if max(x2 - x1, y2 - y1) < MIN_SIDE:
                continue
            items[split, cls].append((video, frame, (x1, y1, x2, y2)))

    rnd = random.Random(0)                                       # одинаковый результат при повторе
    for (split, cls), lst in items.items():
        cap = CAP[split].get(cls)
        if cap and len(lst) > cap:
            items[split, cls] = rnd.sample(lst, cap)

    by_frame = defaultdict(list)                                 # читаем каждый кадр один раз
    for (split, cls), lst in items.items():
        for video, frame, box in lst:
            by_frame[video, frame].append((split, cls, box))

    counts = Counter()
    for (video, frame), objs in sorted(by_frame.items()):
        img = cv2.imread(str(frame_dirs[video] / f"{frame}.jpg"))
        if img is None:
            continue
        for k, (split, cls, box) in enumerate(objs):
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{video}_{frame}_{k}.jpg"), crop_square(img, *box), [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[split, cls] += 1

    for split in ("train", "val"):
        print(split, {cls: counts[split, cls] for cls in ("lying", "sitting", "upright")})


if __name__ == "__main__":
    main()
