"""
Набор для дообучения детектора людей на съёмке с дрона (в первую очередь — сверху и с высоты).

Берёт обучающую часть Okutama-Action (кадры 1280x720 + разметка в координатах 4K) и делает набор
в формате YOLO: images/{train,val}/*.jpg (ссылки на исходные кадры, место не занимают) и
labels/{train,val}/*.txt (один класс 0 = человек, центр и размер рамки в долях кадра) + data.yaml.

Деление по видео — те же 7 проверочных видео, что у нейросети позы (tools/make_pose_crops.py);
1.1.1 исключено (оно среди тестовых роликов). Тестовые ролики Okutama (eval/sets) сюда не попадают
вообще: они в другой части набора (TestSetVideos), поэтому eval/detect_by_size.py после обучения —
честное сравнение на роликах, которых детектор не видел.

Соседние кадры почти одинаковы, поэтому берём каждый --step-й (по умолчанию 20 = 1.5 раза в секунду).
Размеченные как lost (человек вне кадра) пропускаем; закрытые частично (occluded) оставляем — это тоже люди.

Запуск:
    .venv/bin/python tools/make_det_dataset.py okutama_train okutama_train/det_dataset
Потом обучение: tools/train_det.py okutama_train/det_dataset/data.yaml
"""
import argparse
import os
import shlex
from collections import defaultdict
from pathlib import Path

EXCLUDE = {"1.1.1"}
VAL_VIDEOS = {"1.1.4", "1.2.8", "2.1.4", "2.2.2", "1.1.3", "1.2.7", "2.2.9"}
W, H = 1280, 720                 # размер кадров обучающей части
SCALE = W / 3840                 # разметка в 4K


def read_boxes(label_file):
    """-> {кадр: [(x1, y1, x2, y2) в пикселях 1280x720]}"""
    out = defaultdict(list)
    for line in open(label_file, encoding="utf-8", errors="ignore"):
        p = shlex.split(line)
        if len(p) < 10 or int(p[6]) == 1:          # lost: человека нет в кадре
            continue
        x1, y1, x2, y2 = (float(v) * SCALE for v in p[1:5])
        x1, x2 = max(0.0, min(x1, x2)), min(W, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out[int(p[5])].append((x1, y1, x2, y2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("train_root", help="папка okutama_train (Drone1/, Drone2/, Labels/)")
    ap.add_argument("out", help="куда сложить набор")
    ap.add_argument("--step", type=int, default=20, help="брать каждый N-й кадр")
    a = ap.parse_args()
    root, out = Path(a.train_root).resolve(), Path(a.out).resolve()

    frame_dirs = {p.name: p for p in root.glob("Drone*/*/Extracted-Frames-1280x720/*") if p.is_dir()}
    labels = sorted((root / "Labels/MultiActionLabels/3840x2160").glob("*.txt"))
    if not frame_dirs or not labels:
        raise SystemExit(f"В {root} нет Drone*/*/Extracted-Frames-1280x720/<видео>/ или Labels/MultiActionLabels/3840x2160/")
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    total = defaultdict(lambda: [0, 0, 0])            # split -> [кадров, людей, пустых кадров]
    for lab in labels:
        v = lab.stem
        if v in EXCLUDE or v not in frame_dirs:
            continue
        split = "val" if v in VAL_VIDEOS else "train"
        boxes = read_boxes(lab)
        frames = sorted(int(f.stem) for f in frame_dirs[v].glob("*.jpg") if f.stem.isdigit())
        n = 0
        for f in frames:
            if f % a.step:
                continue
            name = f"{v}_{f:05d}"
            img = out / "images" / split / f"{name}.jpg"
            if img.is_symlink() or img.exists():
                img.unlink()
            os.symlink(frame_dirs[v] / f"{f}.jpg", img)
            rows = [f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                    for x1, y1, x2, y2 in boxes.get(f, [])]
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            t = total[split]
            t[0] += 1; t[1] += len(rows); t[2] += not rows
            n += 1
        print(f"{v} -> {split}: {n} кадров", flush=True)

    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    for split, (nf, npers, empty) in sorted(total.items()):
        print(f"{split}: {nf} кадров, {npers} рамок людей, {empty} кадров без людей")
    print(f"-> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
