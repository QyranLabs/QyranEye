"""Проверка нейросети позы на проверочных картинках: что сеть отвечает для каждого класса.
    .venv/bin/python tools/eval_pose.py <веса.pt> <папка val с подпапками классов>
Печатает матрицу «правда -> ответ сети» и главное для системы: насколько хорошо сеть
отличает «лежит» от «не лежит» и (если у сети есть такой класс) «не человека»."""
import sys
from collections import Counter, defaultdict
from pathlib import Path
from ultralytics import YOLO

weights, val = sys.argv[1], Path(sys.argv[2])
model = YOLO(weights)
names = model.names
true_classes = sorted(d.name for d in val.iterdir() if d.is_dir())
conf = Counter(); probs = defaultdict(list)
files = [(c, f) for c in true_classes for f in sorted((val / c).glob("*.jpg"))]
for i in range(0, len(files), 256):
    chunk = files[i:i + 256]
    for (true, f), r in zip(chunk, model.predict([str(f) for _, f in chunk], imgsz=96, device="mps", verbose=False)):
        p = {names[k]: float(v) for k, v in enumerate(r.probs.data.tolist())}
        conf[true, max(p, key=p.get)] += 1
        probs[true].append(p)
pred_classes = list(names.values())
print("строки — правда, столбцы — ответ сети (доля):")
for t in true_classes:
    n = sum(conf[t, p] for p in pred_classes)
    print(f"  {t:10s} n={n:5d}  " + "  ".join(f"{p}={100 * conf[t, p] / n:5.1f}%" for p in pred_classes))
people_not_lying = [p for c in ("sitting", "upright") if c in probs for p in probs[c]]
for thr in (0.3, 0.5, 0.7):
    tp = sum(p["lying"] >= thr for p in probs["lying"]) / len(probs["lying"])
    fp = sum(p["lying"] >= thr for p in people_not_lying) / len(people_not_lying)
    line = f"  порог «лежит» {thr}: лежащих узнано {100 * tp:.0f}%, стоящих/сидящих ошибочно «лежит» {100 * fp:.1f}%"
    if "not_person" in probs:
        fpn = sum(p["lying"] >= thr for p in probs["not_person"]) / len(probs["not_person"])
        line += f", «не людей» ошибочно «лежит» {100 * fpn:.1f}%"
    print(line)
if "not_person" in names.values() and "not_person" in probs:
    rec = sum(p["not_person"] >= 0.5 for p in probs["not_person"]) / len(probs["not_person"])
    lost = sum(p["not_person"] >= 0.5 for c in ("lying", "sitting", "upright") for p in probs.get(c, [])) / \
        sum(len(probs.get(c, [])) for c in ("lying", "sitting", "upright"))
    print(f"  «не человек» >= 0.5: ложных срабатываний детектора отсеяно {100 * rec:.0f}%, "
          f"настоящих людей ошибочно отсеяно {100 * lost:.1f}%")
