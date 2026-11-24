"""
Сравнение режимов позы на записанном наборе (eval/record_inputs.py) — без повторного запуска нейросетей.
    .venv/bin/python eval/compare_modes.py eval_results/inputs/okutama_lying okutama_lying
Сравниваются: правила (geometry), только сеть v1 (classifier), совместный режим с сетью v2 (hybrid,
пороги выбраны на проверочных видео: отмена «лежит» при P <= 0.1, добавление при P >= 0.7).
"""
import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "eval"))
sys.path.insert(0, str(PROJECT))
import evaluate as ev                   # noqa: E402
import replay as rp                     # noqa: E402
from video_pipeline import load_config  # noqa: E402

MODES = [
    ("правила (geometry)", "main", {"pose_mode": "geometry"}),
    ("сеть v1 (classifier)", "main", {"pose_mode": "classifier", "pose_upright_max": 0.3}),   # порог, с которым оценивалась v1
    ("правила + сеть v2 (hybrid)", "extra", {"pose_mode": "hybrid", "pose_upright_max": 0.1, "pose_net_down_min": 0.7}),
]


def main():
    indir, set_name = Path(sys.argv[1]), sys.argv[2]
    items = list(csv.DictReader(open(PROJECT / "eval" / "sets" / f"{set_name}.csv")))
    print(f"набор {set_name}:")
    for name, net, over in MODES:
        cfg = load_config(PROJECT / "config.yaml")
        for k, v in over.items():
            setattr(cfg, k, v)
        rp.NET = net
        out = PROJECT / "eval_results" / "modes" / set_name / over["pose_mode"]
        rp.run(cfg, indir, out)
        tot, _, alarms = ev.compare(items, out)
        L, O = "lying", "other"
        lab = tot[L, "n"] + tot[O, "n"]
        found = tot[L, "found"] + tot[O, "found"]
        lying_ok = tot[L, "LYING"] + tot[L, "FALL?"] + tot[L, "FALL"]
        correct = lying_ok + tot[O, "UP"]
        bad = tot[O, "LYING"] + tot[O, "FALL?"] + tot[O, "FALL"]
        lying = f"{100 * lying_ok / tot[L, 'n']:5.1f}%" if tot[L, "n"] else "   — "
        print(f"  {name:28s} тревог {len(alarms)}   правильно у найденных {100 * correct / found:5.1f}%   "
              f"у всех {100 * correct / lab:5.1f}%   лежащие {lying}   ложные «лежит» {100 * bad / tot[O, 'found']:5.1f}%")
        for a in alarms:
            print(f"      тревога: {a[0]} трек #{a[1]} в {a[2]} — {a[3]}")


if __name__ == "__main__":
    main()
