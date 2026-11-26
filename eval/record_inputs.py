"""Запись ВХОДОВ логики падения для подбора порогов (вместе с eval/replay.py).

Запускает систему (как `python main.py <источник> --config X`) и на каждом анализируемом кадре
записывает, что получает логика: время, люди (id, рамка, ответ нейросети позы, точки тела), сдвиг камеры,
размер кадра, ракурс. Детектор и нейросеть работают один раз; потом eval/replay.py прогоняет
только логику с любыми настройками за секунды.

    .venv/bin/python eval/record_inputs.py config.yaml eval_results/inputs_val okutama_train/val_videos/ [auto|oblique|nadir]
(последнее — ракурс, как ключ --view у main.py; по умолчанию auto)
(чтобы записались ответы нейросети, в конфиге должен быть pose_mode: hybrid или classifier)
"""
import json, sys, time
from pathlib import Path
proj = str(Path(__file__).resolve().parent.parent)
cfg, logdir, source = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
view = sys.argv[4] if len(sys.argv) > 4 else "auto"
sys.path.insert(0, proj)
logdir.mkdir(parents=True, exist_ok=True)
import video_pipeline as vp, fall_logic as fl, main as m

cur = {"f": None, "meta": None, "extra": None}
orig_pv, orig_up, orig_pp = vp.process_video, fl.FallMonitor.update, vp.pose_probs

# Необязательно: вторая нейросеть позы (EXTRA_POSE_MODEL=путь.pt) — её ответы записываются рядом,
# чтобы на повторе сравнить две сети без второго прохода по видео.
import os
extra_model = None
if os.environ.get("EXTRA_POSE_MODEL"):
    from ultralytics import YOLO
    extra_model = YOLO(os.environ["EXTRA_POSE_MODEL"])


def pp(pose_model, frame, boxes, cfg_):
    res = orig_pp(pose_model, frame, boxes, cfg_)
    cur["extra"] = orig_pp(extra_model, frame, boxes, cfg_) if extra_model is not None else None
    return res


def pv(path, cfg_, model, out_dir, *rest):
    cur["f"] = open(logdir / (Path(path).stem + ".inputs.jsonl"), "w")
    t0 = time.time()
    try:
        return orig_pv(path, cfg_, model, out_dir, *rest)
    finally:
        cur["f"].close()
        (logdir / (Path(path).stem + ".time")).write_text(str(time.time() - t0))


def up(self, t, people, warp=None):
    if self.frame_size is not None:                      # только основной проход (не предварительный)
        if cur["f"].tell() == 0:
            cur["f"].write(json.dumps({"meta": {"view": self.view, "view_reason": self.view_reason,
                                                "frame_size": list(self.frame_size),
                                                "baseline_len": self.baseline_len}}) + "\n")
        cur["f"].write(json.dumps({"t": t, "warp": None if warp is None else [list(map(float, r)) for r in warp],
                                   "people": [[p[0], [round(float(v), 1) for v in p[1]], p[3] if len(p) > 3 else None,
                                               cur["extra"][i] if cur["extra"] else None,
                                               None if p[2] is None else [[round(float(v), 2) for v in k] for k in p[2]]]
                                              for i, p in enumerate(people)]}) + "\n")
    return orig_up(self, t, people, warp)


vp.process_video, fl.FallMonitor.update, vp.pose_probs = pv, up, pp
sys.argv = ["main.py", source, "--config", cfg, "--view", view]
m.main()
