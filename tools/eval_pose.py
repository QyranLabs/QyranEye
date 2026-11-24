"""Проверка нейросети позы на проверочных видео: точность по каждому классу и по размеру человека.
    python tools/eval_pose.py <веса.pt> okutama_train/pose_crops/val [порог_lying]"""
import sys
from collections import Counter, defaultdict
from pathlib import Path
from ultralytics import YOLO

weights, val = sys.argv[1], Path(sys.argv[2])
model = YOLO(weights)
names = model.names
conf = Counter(); probs_lying = defaultdict(list)
files = [(c, f) for c in ("lying", "sitting", "upright") for f in sorted((val / c).glob("*.jpg"))]
for i in range(0, len(files), 256):
    chunk = files[i:i + 256]
    for (true, f), r in zip(chunk, model.predict([str(f) for _, f in chunk], imgsz=96, device="mps", verbose=False)):
        p = {names[k]: float(v) for k, v in enumerate(r.probs.data.tolist())}
        pred = max(p, key=p.get)
        conf[true, pred] += 1
        probs_lying[true].append(p["lying"])
classes = ("lying", "sitting", "upright")
print("строки — правда, столбцы — ответ сети (доля):")
for t in classes:
    n = sum(conf[t, p] for p in classes)
    print(f"  {t:8s} n={n:5d}  " + "  ".join(f"{p}={100*conf[t, p]/n:5.1f}%" for p in classes))
acc = sum(conf[c, c] for c in classes) / sum(conf.values())
bal = sum(conf[c, c] / sum(conf[c, p] for p in classes) for c in classes) / 3
print(f"общая точность {100*acc:.1f}%, средняя по классам {100*bal:.1f}%")
for thr in (0.3, 0.5, 0.7):
    tp = sum(x >= thr for x in probs_lying["lying"]) / len(probs_lying["lying"])
    fp = sum(x >= thr for c in ("sitting", "upright") for x in probs_lying[c]) / sum(len(probs_lying[c]) for c in ("sitting", "upright"))
    print(f"  порог «лежит» {thr}: лежащих узнано {100*tp:.0f}%, не лежащих ошибочно «лежит» {100*fp:.1f}%")
