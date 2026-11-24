"""
Выбор режима позы и порогов на ПРОВЕРОЧНЫХ видео (обучающая часть Okutama, eval/sets/val.csv) —
тестовые ролики при этом не используются.

Нужна запись входов логики (eval/record_inputs.py с EXTRA_POSE_MODEL=pose_cls_v2.pt):
    .venv/bin/python eval/select_mode.py eval_results/inputs/val [набор=val]

Правило выбора (записано ДО того, как посмотрели результаты):
  1. ложных тревог FALL на проверочных видео — 0 (падений там нет);
  2. среди таких — больше всего людей с правильным статусом (от всех размеченных);
  3. при равенстве (разница < 0.5 пункта) — более простой вариант: правила, затем hybrid, затем сеть.
"""
import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "eval"))
sys.path.insert(0, str(PROJECT))
import evaluate as ev                 # noqa: E402
import replay as rp                   # noqa: E402
from video_pipeline import load_config  # noqa: E402

indir = Path(sys.argv[1])
set_name = sys.argv[2] if len(sys.argv) > 2 else "val"
out_root = PROJECT / "eval_results" / "select" / set_name
items = list(csv.DictReader(open(PROJECT / "eval" / "sets" / f"{set_name}.csv")))

VARIANTS = [("geometry", "main", {})]
for net, tag in (("main", "v1"), ("extra", "v2")):
    VARIANTS.append((f"classifier {tag}", net, {"pose_mode": "classifier"}))
    for veto in (0.05, 0.1, 0.2, 0.3):
        for down in (0.7, 0.9):
            VARIANTS.append((f"hybrid {tag} отмена<={veto} добавить>={down}", net,
                             {"pose_mode": "hybrid", "pose_upright_max": veto, "pose_net_down_min": down}))
SIMPLE = {"geometry": 0, "hybrid": 1, "classifier": 2}

rows = []
for name, net, over in VARIANTS:
    cfg = load_config(PROJECT / "config.yaml")
    for k, v in over.items():
        setattr(cfg, k, v)
    rp.NET = net
    out = out_root / name.replace(" ", "_").replace("<=", "le").replace(">=", "ge")
    rp.run(cfg, indir, out)
    tot, per_video, alarms = ev.compare(items, out)
    L, O = "lying", "other"
    lab = tot[L, "n"] + tot[O, "n"]
    lying_ok = tot[L, "LYING"] + tot[L, "FALL?"] + tot[L, "FALL"]
    correct = lying_ok + tot[O, "UP"]
    found = tot[L, "found"] + tot[O, "found"]
    bad = tot[O, "LYING"] + tot[O, "FALL?"] + tot[O, "FALL"]
    rows.append(dict(name=name, alarms=len(alarms), correct_all=100 * correct / lab,
                     correct_found=100 * correct / max(found, 1),
                     lying=100 * lying_ok / max(tot[L, "n"], 1), false_lying=100 * bad / max(tot[O, "found"], 1),
                     simple=SIMPLE[name.split()[0]]))
    print(f"{name:42s} тревог {len(alarms)}  правильно у всех {rows[-1]['correct_all']:5.1f}%  "
          f"у найденных {rows[-1]['correct_found']:5.1f}%  лежащие {rows[-1]['lying']:5.1f}%  "
          f"ложные «лежит» {rows[-1]['false_lying']:5.1f}%", flush=True)

ok = [r for r in rows if r["alarms"] == 0] or rows
best_score = max(r["correct_all"] for r in ok)
close = [r for r in ok if best_score - r["correct_all"] < 0.5]
best = min(close, key=lambda r: (r["simple"], -r["correct_all"]))
print(f"\nВЫБРАН: {best['name']}  (правило: 0 тревог -> максимум правильного статуса -> проще)")
