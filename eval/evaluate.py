"""
Проверка системы на видео Okutama с разметкой: запускает обработку (как main.py) и
сравнивает результат с разметкой — кого нашли, какой статус дали, были ли ложные тревоги.

Запуск (из папки проекта):
    .venv/bin/python eval/evaluate.py okutama_lying          # 4 ролика с лежащими людьми
    .venv/bin/python eval/evaluate.py heldout                # 3 ролика без лежащих (ложные срабатывания)
    .venv/bin/python eval/evaluate.py untouched              # 4 ролика без лежащих
    .venv/bin/python eval/evaluate.py val                    # 7 проверочных видео обучающей части
    .venv/bin/python eval/evaluate.py okutama_lying --config my.yaml   # другие настройки
    .venv/bin/python eval/evaluate.py okutama_lying --no-run           # только пересчитать таблицу
Наборы описаны в eval/sets/<имя>.csv (видео, файл разметки, с какой секунды вырезан ролик).
Результаты: eval_results/<набор>/ — видео, CSV тревог и журналы состояний (logs/).

Как считается:
  * разметка Okutama в координатах 4K, система работает в 1080p -> делим на 2;
  * человек «найден», если рамка системы перекрывает размеченную хотя бы на 20% (IoU >= 0.2);
  * «правильный статус»: у лежащего — LYING / FALL? / FALL, у остальных — UP;
  * разметка некоторых роликов сдвинута по времени на ~20 кадров — сдвиг подбирается автоматически;
  * в Okutama НЕТ падений, поэтому все тревоги FALL здесь — ложные (или спокойное укладывание).
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))
import okutama_extract as ox                                    # noqa: E402  (чтение разметки Okutama)

FPS = 29.97
IOU_MIN = 0.2
STATES = ("UP", "?", "LYING", "FALL?", "FALL")


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


# ------------------------------------------------------------------ запуск системы с журналом
def run_set(items, cfg_path, out_dir):
    """Обрабатывает видео набора и пишет журнал состояний каждого человека на каждом кадре."""
    import fall_logic as fl
    import video_pipeline as vp
    from ultralytics import YOLO

    cfg = vp.load_config(cfg_path)
    model = YOLO(str(vp.resolve(cfg, cfg.model)))
    pose_model = YOLO(str(vp.resolve(cfg, cfg.pose_model))) if cfg.pose_mode in ("classifier", "hybrid") else None
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    orig_update = fl.FallMonitor.update
    current = {"f": None}

    def update(self, t, people, warp=None):
        vis = orig_update(self, t, people, warp)
        if self.frame_size is not None and current["f"] is not None:      # основной проход
            current["f"].write(json.dumps({"t": t, "p": [[tr.tid, [round(v, 1) for v in tr.box], tr.state]
                                                         for tr in vis]}) + "\n")
        return vis

    fl.FallMonitor.update = update
    try:
        for i, it in enumerate(items, 1):
            video = PROJECT / it["video"]
            print(f"[{i}/{len(items)}] {video.name}", flush=True)
            with open(logs / f"{video.stem}.jsonl", "w") as f:
                current["f"] = f
                vp.process_video(video, cfg, model, out_dir, pose_model)
            current["f"] = None
    finally:
        fl.FallMonitor.update = orig_update


# ------------------------------------------------------------------ сравнение с разметкой
def load_labels(path, scale=0.5):
    by_frame = defaultdict(list)
    for tid, rows in ox.parse_labels(path).items():
        for fr, act, (x1, y1, x2, y2), lost in rows:
            if lost == 0:
                by_frame[fr].append((tid, (x1 * scale, y1 * scale, x2 * scale, y2 * scale), act))
    return by_frame


def best_shift(log, by_frame, off):
    """Разметка части роликов сдвинута по времени: ищем сдвиг с наибольшим числом совпадений."""
    best = (-1, 0)
    for s in range(-60, 61, 2):
        n = sum(1 for rec in log[::5] for _, lb, _ in by_frame.get(round(rec["t"] * FPS) + off + s, [])
                if any(iou(lb, p[1]) >= 0.3 for p in rec["p"]))
        best = max(best, (n, -abs(s), s))
    return best[2]


def compare(items, out_dir):
    tot = Counter()
    alarms = []
    per_video = []
    for it in items:
        stem = Path(it["video"]).stem
        log = [json.loads(l) for l in open(out_dir / "logs" / f"{stem}.jsonl")]
        by_frame = load_labels(PROJECT / it["labels"])
        off = round(float(it["start_s"]) * FPS)
        sh = best_shift(log, by_frame, off)
        v = Counter()
        for rec in log:
            for tid, lb, act in by_frame.get(round(rec["t"] * FPS) + off + sh, []):
                key = "lying" if act == "Lying" else "other"
                m = max(rec["p"], key=lambda p: iou(lb, p[1]), default=None)
                found = m is not None and iou(lb, m[1]) >= IOU_MIN
                for c in (tot, v):
                    c[key, "n"] += 1
                    if found:
                        c[key, "found"] += 1
                        c[key, m[2]] += 1
        per_video.append((stem, v, sh))
        ev = out_dir / f"{stem}_events.csv"
        cap = cv2.VideoCapture(str(PROJECT / it["video"]))
        k = 1080 / min(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        for e in (csv.DictReader(open(ev)) if ev.exists() else []):
            # координаты в CSV — в пикселях исходного видео (или кадра высотой frame_h у replay.py);
            # переводим в 1080p, как разметку
            kk = 1080 / float(e["frame_h"]) if e.get("frame_h") else k
            box = tuple(float(e[c]) * kk for c in ("x1", "y1", "x2", "y2"))
            labs = by_frame.get(int(e["alarm_frame"]) + off + sh, [])
            m = max(labs, key=lambda l: iou(l[1], box), default=None)
            what = f"размечен как {m[2]}" if m and iou(m[1], box) >= IOU_MIN else "не человек из разметки"
            alarms.append((stem, e["track_id"], e["alarm_time"], what))
    return tot, per_video, alarms


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "—"


def report(tot, per_video, alarms):
    L, O = "lying", "other"
    lying_ok = tot[L, "LYING"] + tot[L, "FALL?"] + tot[L, "FALL"]
    other_bad = tot[O, "LYING"] + tot[O, "FALL?"] + tot[O, "FALL"]
    found = tot[L, "found"] + tot[O, "found"]
    correct = lying_ok + tot[O, "UP"]
    print("\n================ ИТОГ ================")
    print(f"Размеченных людей (человек × кадр): {tot[L, 'n'] + tot[O, 'n']} (лежащих {tot[L, 'n']})")
    print(f"Найдено системой:                    {pct(found, tot[L, 'n'] + tot[O, 'n'])}")
    print(f"Правильный статус у найденных:       {pct(correct, found)}")
    print(f"Правильный статус у всех размеченных: {pct(correct, tot[L, 'n'] + tot[O, 'n'])}")
    if tot[L, "n"]:
        print(f"Лежащие: найдено {pct(tot[L, 'found'], tot[L, 'n'])}, "
              f"с правильным статусом (LYING/FALL) {pct(lying_ok, tot[L, 'n'])} от всех лежащих")
    print(f"Не лежащие: ошибочно «лежит»/«упал» {pct(other_bad, tot[O, 'found'])} от найденных")
    print(f"Тревог FALL: {len(alarms)}" + ("  (в Okutama падений нет — все они ложные)" if alarms else ""))
    for a in alarms:
        print(f"   {a[0]}: трек #{a[1]} в {a[2]} — {a[3]}")
    print("\nПо роликам (найдено / правильный статус у найденных / ложные «лежит»):")
    for stem, v, sh in per_video:
        f = v[L, "found"] + v[O, "found"]
        ok = v[L, "LYING"] + v[L, "FALL?"] + v[L, "FALL"] + v[O, "UP"]
        bad = v[O, "LYING"] + v[O, "FALL?"] + v[O, "FALL"]
        print(f"   {stem:24s} {pct(f, v[L, 'n'] + v[O, 'n']):>5s} / {pct(ok, f):>5s} / {pct(bad, v[O, 'found']):>5s}"
              f"   (сдвиг разметки {sh} кадров)")


def main():
    ap = argparse.ArgumentParser(description="Проверка QyranEye на видео Okutama с разметкой")
    ap.add_argument("set", help="okutama_lying | heldout | untouched | val (файл eval/sets/<set>.csv)")
    ap.add_argument("--config", default=str(PROJECT / "config.yaml"))
    ap.add_argument("--out", default=str(PROJECT / "eval_results"))
    ap.add_argument("--no-run", action="store_true", help="не запускать систему, только пересчитать таблицу")
    a = ap.parse_args()
    items = list(csv.DictReader(open(PROJECT / "eval" / "sets" / f"{a.set}.csv")))
    missing = [it["video"] for it in items if not (PROJECT / it["video"]).exists()]
    if missing:
        sys.exit(f"Нет видео: {missing}")
    out_dir = Path(a.out) / a.set
    t0 = time.time()
    if not a.no_run:
        run_set(items, a.config, out_dir)
    report(*compare(items, out_dir))
    print(f"\nГотово за {time.time() - t0:.0f} с. Видео и CSV: {out_dir}")


if __name__ == "__main__":
    main()
