"""Повтор логики падения по записанным входам (eval/record_inputs.py) с любыми настройками.
Детектор и нейросеть не запускаются — только fall_logic.py, поэтому это занимает секунды.
Пишет журналы (logs/<ролик>.jsonl) и тревоги (<ролик>_events.csv) в формате eval/evaluate.py,
так что таблицу потом можно посчитать: eval/evaluate.py <набор> --no-run --out <папка>.

    .venv/bin/python eval/replay.py config.yaml eval_results/inputs_val eval_results/replay/val pose_mode=hybrid
    .venv/bin/python eval/evaluate.py val --no-run --out eval_results/replay
Любую настройку из config.yaml можно поменять так: ключ=значение.
"""
import copy, csv, json, sys
from pathlib import Path

proj = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, proj)
from video_pipeline import load_config
from fall_logic import FallMonitor


def parse(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


NET = "main"   # main — ответы сети из config.yaml при записи; extra — второй сети (EXTRA_POSE_MODEL)


def run(cfg, indir, outdir, fps=29.97, stride=2):
    outdir.mkdir(parents=True, exist_ok=True)
    total_alarms = 0
    for f in sorted(indir.glob("*.inputs.jsonl")):
        stem = f.name[:-len(".inputs.jsonl")]
        lines = [json.loads(l) for l in open(f)]
        meta = lines[0]["meta"]
        c = copy.copy(cfg)
        c.camera_view = meta["view"] or c.camera_view_default    # тот же ракурс, что решил предпросмотр
        mon = FallMonitor(c, fps / stride, tuple(meta["frame_size"]))
        (outdir / "logs").mkdir(parents=True, exist_ok=True)
        with open(outdir / "logs" / f"{stem}.jsonl", "w") as out:
            for rec in lines[1:]:
                k = 3 if (NET == "extra" and len(rec["people"]) and len(rec["people"][0]) > 3) else 2
                people = [(p[0], tuple(p[1]), None, p[k]) if p[k] is not None else (p[0], tuple(p[1]), None)
                          for p in rec["people"]]
                vis = mon.update(rec["t"], people, rec["warp"])
                out.write(json.dumps({"t": rec["t"], "p": [[tr.tid, list(tr.box), tr.state, tr.raw_zone, tr.zone, tr.cue]
                                                           for tr in vis]}) + "\n")
        with open(outdir / f"{stem}_events.csv", "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["video", "track_id", "earlier_ids", "fall_start_s", "fall_start_time", "alarm_s", "alarm_time",
                        "alarm_frame", "transition_s", "x1", "y1", "x2", "y2", "frame_h"])
            frame_h = min(meta["frame_size"])      # рамки — в пикселях кадра обработки этой высоты
            for e in mon.events:
                x1, y1, x2, y2 = (round(v) for v in e["box"])
                w.writerow([stem, e["track_id"], e["earlier_ids"], e["fall_start_s"], "", e["alarm_s"], "",
                            round(e["alarm_s"] * fps), e["transition_s"], x1, y1, x2, y2, frame_h])
        total_alarms += len(mon.events)
    return total_alarms


if __name__ == "__main__":
    cfg_path, indir, outdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    cfg = load_config(cfg_path)
    for kv in sys.argv[4:]:
        k, v = kv.split("=", 1)
        if k == "net":
            NET = v               # net=extra — взять ответы второй сети
        else:
            setattr(cfg, k, parse(v))
    print("тревог:", run(cfg, indir, outdir))
