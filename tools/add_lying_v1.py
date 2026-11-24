"""Добавить в набор v2 картинки «лежит» из v1 (рамки разметки): детектор на обучающих кадрах
720p почти не находит лежащих, и по его рамкам их набралось мало. Половина добавленных
картинок «портится как в видео» (та же функция degrade, что в make_pose_crops_det.py).
    python tools/add_lying_v1.py okutama_train/pose_crops okutama_train/pose_crops_v2"""
import random, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pose_crops_det import degrade

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
rnd = random.Random(1)
n = 0
for f in sorted((src / "train" / "lying").glob("*.jpg")):
    img = cv2.imread(str(f))
    if rnd.random() < 0.5:
        img = degrade(img, rnd)
    cv2.imwrite(str(dst / "train" / "lying" / f"v1_{f.name}"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    n += 1
print("добавлено «лежит»:", n)
