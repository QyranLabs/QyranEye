"""Все главные числа проекта одной командой (их и только их пишем в README).

Прогоняет ТЕКУЩУЮ логику (fall_logic.py + config.yaml) по записанным входам
(eval/record_inputs.py: рамки, номера людей, сдвиг камеры, ракурс) и точкам тела.
Детектор и нейросети не запускаются, поэтому весь отчёт — 10–40 секунд.

    .venv/bin/python eval/report.py                    # таблица в терминал
    .venv/bin/python eval/report.py --md docs/report.md   # и в файл
    .venv/bin/python eval/report.py --set lying_alarm_s=5.0   # «что если» (config.yaml не меняется)
    .venv/bin/python eval/report.py --set pose_mode=hybrid --net extra   # режим с нейросетью v2
    .venv/bin/python eval/report.py --set pose_mode=classifier --set pose_upright_max=0.3   # сеть v1

Что нужно на диске (в git не хранится, см. README «Данные»):
  eval_results/inputs/<набор>/*.inputs.jsonl  — записанные входы;
  eval_results/keypoints/<набор>__<ролик>.json — точки тела на вырезках (если их нет во входах);
  разметка и видео Okutama (пути в eval/sets/*.csv; из видео берётся только размер кадра).
Ответы нейросетей позы — те, что записаны во входах: в записях Okutama «main» — сеть v1, «extra» — сеть v2.
Ракурс каждого ролика — тот, что выбрал при записи авто-ракурс (для MOD20 — заданный вручную); для ясных
роликов он совпадает с тем, что видно глазами (проверено 25.11.2026). --set camera_view=... задаёт всем один ракурс.
Если каких-то записей нет, отчёт пишет это и НЕ говорит «все проверки пройдены».
"""
import argparse
import copy
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

PROJECT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT), str(PROJECT / "eval")]
import numpy as np                      # noqa: E402
import evaluate as ev                   # noqa: E402
import fall_logic as fl                 # noqa: E402
from video_pipeline import load_config  # noqa: E402

INPUTS, KEYPOINTS = PROJECT / "eval_results/inputs", PROJECT / "eval_results/keypoints"
OUT = PROJECT / "eval_results/report"
LABELLED = {"4 ролика с лежащими": ["okutama_lying"],
            "11 тестовых роликов": ["okutama_lying", "heldout", "untouched"],
            "7 проверочных видео": ["val"]}
OTHER = ["drone", "pexels", "nomad", "nomad_full", "mod20"]
# Известные события в роликах без разметки Okutama (проверено глазами по кадрам).
# (набор, ролик, с, по, ожидание): "alarm" — тревога должна прийти в этом окне; "none" — тревоги быть не должно.
EXPECT = [
    ("drone", "mixkit_man_falls_snow_23814", 16.5, 22.0, "alarm", "рухнул в снег и лежит"),
    ("drone", "mixkit_man_falls_snow_23814", 0.0, 9.0, "none", "споткнулся, на четвереньках, встал сам"),
    ("nomad_full", "Actor042_a10_from0.0s", 0.0, 1e9, "none", "лёг сам (NOMAD), падений нет"),
    ("nomad_full", "Actor099_a10_from0.0s", 0.0, 1e9, "none", "лёг сам (NOMAD), падений нет"),
    ("nomad", "Actor035_a10_from27.027s", 0.0, 1e9, "none", "присел, потом лёг (NOMAD)"),
    ("mod20", "*", 0.0, 1e9, "none", "55 роликов MOD20 сверху: танцы, скейт, бои — без падений с потерей движения"),
]


def load_kp(set_name, stem):
    f = KEYPOINTS / f"{set_name}__{stem}.json"
    out = defaultdict(dict)
    if f.exists():
        for t, tid, k in json.load(open(f)):
            out[round(t, 3)][tid] = np.array(k)
    return out


NET = "main"     # чьи ответы сети брать из записи: main (p[2]) или extra (p[3])
FORCE_VIEW = None


def run_clip(cfg, set_name, f, out_dir):
    """Логика по одному записанному ролику -> журнал и CSV тревог (формат eval/evaluate.py)."""
    stem = f.name[:-len(".inputs.jsonl")]
    lines = [json.loads(l) for l in open(f)]
    meta, recs = lines[0]["meta"], lines[1:]
    c = copy.copy(cfg)
    c.camera_view = FORCE_VIEW or meta["view"] or c.camera_view_default
    dt = median(np.diff([r["t"] for r in recs[:60]])) if len(recs) > 2 else 1 / 15
    mon = fl.FallMonitor(c, 1 / dt, tuple(meta["frame_size"]))
    kp = load_kp(set_name, stem)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    states = Counter()
    with open(out_dir / "logs" / f"{stem}.jsonl", "w") as log:
        for r in recs:
            ks = kp.get(round(round(r["t"], 4), 3), {})
            people = []
            for p in r["people"]:
                kpt = p[4] if len(p) > 4 and p[4] is not None else ks.get(p[0])
                net = p[3 if NET == "extra" else 2] if len(p) > (3 if NET == "extra" else 2) else None
                people.append((p[0], tuple(p[1]), kpt) + ((net,) if net is not None else ()))
            vis = mon.update(r["t"], people, r["warp"])
            states.update(tr.state for tr in vis)
            log.write(json.dumps({"t": r["t"], "p": [[tr.tid, list(tr.box), tr.state, tr.raw_zone, tr.zone, tr.cue]
                                                     for tr in vis]}) + "\n")
    fh = min(meta["frame_size"])
    with open(out_dir / f"{stem}_events.csv", "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["track_id", "alarm_s", "alarm_time", "alarm_frame", "fall_speed", "x1", "y1", "x2", "y2", "frame_h"])
        for e in mon.events:
            w.writerow([e["track_id"], e["alarm_s"], f'{e["alarm_s"]:.2f} с', round(e["alarm_s"] * ev.FPS), e.get("fall_speed", ""),
                        *[round(v) for v in e["box"]], fh])
    checks = getattr(mon, "checks", [])
    return stem, mon.events, checks, states, recs[-1]["t"] if recs else 0


def pct(a, b):
    return f"{100 * a / b:.1f} %" if b else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--md", help="сохранить отчёт в файл (Markdown)")
    ap.add_argument("--set", action="append", default=[], help="ключ=значение поверх config.yaml (для «что если»)")
    ap.add_argument("--net", choices=["main", "extra"], default="main",
                    help="ответы какой сети брать из записи (Okutama: main = v1, extra = v2)")
    a = ap.parse_args()
    global NET, FORCE_VIEW
    NET = a.net
    cfg = load_config(PROJECT / "config.yaml")
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(cfg, k, type(getattr(cfg, k))(v) if hasattr(cfg, k) and not isinstance(getattr(cfg, k), bool)
                else {"true": True, "false": False}.get(v.lower(), v))
        if k == "camera_view":
            FORCE_VIEW = v
    commit = subprocess.run(["git", "-C", str(PROJECT), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    extra = a.set + ([f"сеть {a.net}"] if a.net != "main" else [])
    L = [f"# QyranEye — отчёт `eval/report.py` (версия {commit}{'; ' + ', '.join(extra) if extra else ''})", ""]
    ok_all = True

    runs = {}
    for s in sorted({x for v in LABELLED.values() for x in v} | set(OTHER)):
        d = INPUTS / s
        files = sorted(d.glob("*.inputs.jsonl")) if d.exists() else []
        runs[s] = [run_clip(cfg, s, f, OUT / s) for f in files]

    # ---------- 1. Okutama с разметкой
    L += ["## 1. Okutama-Action с разметкой (падений нет: проверка «лежит» и ложных тревог)", "",
          "| Набор | Найдено людей | Верный статус у найденных | Верный статус у всех | Лежащие: LYING/FALL?/FALL | "
          "Ложные «лежит» | Тревоги |", "|---|---|---|---|---|---|---|"]
    alarm_rows = []
    for name, sets in LABELLED.items():
        T, n_al, missing = Counter(), 0, []
        for s in sets:
            items = list(csv.DictReader(open(PROJECT / "eval/sets" / f"{s}.csv")))
            have = {r[0] for r in runs.get(s, [])}
            lack = [Path(i["video"]).stem for i in items if Path(i["video"]).stem not in have
                    or not (PROJECT / i["video"]).exists() or not (PROJECT / i["labels"]).exists()]
            if lack:                      # неполный набор дал бы числа, которые выглядят полными
                missing.append(f"{s}: {', '.join(lack)}")
                continue
            t, _, al = ev.compare(items, OUT / s)
            T.update(t)
            n_al += len(al)
            alarm_rows += [(name, *x) for x in al]
        if missing:
            L.append(f"| {name} | **нет данных** ({'; '.join(missing)}) | | | | | |")
            ok_all = False
            continue
        lo, ot = "lying", "other"
        lok = T[lo, "LYING"] + T[lo, "FALL?"] + T[lo, "FALL"]
        found, n = T[lo, "found"] + T[ot, "found"], T[lo, "n"] + T[ot, "n"]
        bad = T[ot, "LYING"] + T[ot, "FALL?"] + T[ot, "FALL"]
        L.append(f"| {name} | {pct(found, n)} | {pct(lok + T[ot, 'UP'], found)} | {pct(lok + T[ot, 'UP'], n)} | "
                 f"{pct(lok, T[lo, 'n'])} | {pct(bad, T[ot, 'found'])} | {n_al} |")
    L.append("")
    seen = set()
    for name, stem, tid, tm, what in alarm_rows:
        if (stem, tid, tm) not in seen:
            seen.add((stem, tid, tm))
            L.append(f"- тревога: {stem}, человек #{tid}, {tm} — {what} (в Okutama падений нет: тревога ложная)")
    ok_all &= not seen         # в Okutama нет падений, любая тревога — ложная
    L.append("")

    # ---------- 2. Известные события
    L += ["## 2. Известные события (видео без разметки Okutama)", "",
          "| Набор / ролик | Что происходит | Окно, с | Ожидание | Тревоги в окне | Итог |", "|---|---|---|---|---|---|"]
    for s, stem, t0, t1, want, what in EXPECT:
        clips = [r for r in runs.get(s, []) if stem in ("*", r[0])]
        if not clips:
            L.append(f"| {s}/{stem} | {what} | | {want} | нет записи | **НЕТ** |")
            ok_all = False
            continue
        hits = [(r[0], e["alarm_s"]) for r in clips for e in r[1] if t0 <= e["alarm_s"] <= t1]
        good = bool(hits) if want == "alarm" else not hits
        ok_all &= good
        win = "весь ролик" if t1 > 1e8 else f"{t0}–{t1}"
        L.append(f"| {s}/{stem if stem != '*' else f'{len(clips)} роликов'} | {what} | {win} | "
                 f"{'тревога' if want == 'alarm' else 'без тревоги'} | "
                 f"{', '.join(f'{c} {x:.2f} с' for c, x in hits) or '0'} | {'ДА' if good else '**НЕТ**'} |")
    L.append("")
    other = [(s, r[0], e["alarm_s"]) for s in OTHER for r in runs.get(s, []) for e in r[1]]
    known = {(s, stem) for s, stem, *_ in EXPECT}
    extra = [x for x in other if (x[0], x[1]) not in known and (x[0], "*") not in known]
    for s, stem, tm in extra:
        L.append(f"- тревога без разметки: {s}/{stem} {tm:.2f} с (проверить глазами)")
    if extra:
        L.append("")

    # ---------- 3. Сигнал «лежит без движения» (если включён)
    ch = [(s, r[0], c) for s in runs for r in runs[s] for c in r[2]]
    if getattr(cfg, "still_check_s", 0):
        L += [f"## 3. Сигнал «лежит без движения {cfg.still_check_s:g} с» (для оператора, не тревога)", ""]
        lab = Counter()
        for s, stem, e in ch:
            t, tid, box = e["check_s"], e["track_id"], e["box"]
            what = label_at(s, stem, t, box)
            lab[what] += 1
            L.append(f"- {s}/{stem} {t:.1f} с, человек #{tid}: {what}")
        L += ["", "Итого: " + (", ".join(f"{k} {v}" for k, v in lab.most_common()) or "сигналов нет"), ""]

    # ---------- 4. Время в состояниях для MOD20
    if runs.get("mod20"):
        st, secs = Counter(), 0.0
        for r in runs["mod20"]:
            st.update(r[3])
            secs += r[4]
        tot = sum(st.values())
        L += ["## 4. MOD20, 55 роликов сверху (без падений), доля времени по людям", "",
              f"{len(runs['mod20'])} роликов, {secs / 60:.1f} мин: " +
              ", ".join(f"{k} {pct(v, tot)}" for k, v in st.most_common()), ""]

    # ---------- 5. Синтетические тесты логики
    r = subprocess.run([sys.executable, str(PROJECT / "tests/test_fall_logic.py")], capture_output=True, text=True)
    n_ok, n_fail = r.stdout.count("\nOK "), r.stdout.count("\nFAIL")
    itog = next((l for l in r.stdout.splitlines() if l.startswith("ИТОГ")), "ИТОГ: ?")
    L += ["## 5. Синтетические тесты (tests/test_fall_logic.py)", "", f"{n_ok + n_fail} сценариев, {itog}", ""]
    ok_all &= r.returncode == 0
    L.append("Все проверки пройдены." if ok_all else "**Есть непройденные проверки (ложная тревога в Okutama, раздел 2 или тесты).**")
    text = "\n".join(L)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n", encoding="utf-8")
    return 0 if ok_all else 1


_labels = {}


def label_at(s, stem, t, box):
    """Что в разметке Okutama у человека в этой рамке (для сигналов). Для роликов без разметки — «нет разметки»."""
    p = PROJECT / "eval/sets" / f"{s}.csv"
    if not p.exists():
        return "нет разметки"
    if (s, stem) not in _labels:
        it = next((i for i in csv.DictReader(open(p)) if Path(i["video"]).stem == stem), None)
        if it is None:
            return "нет разметки"
        log = [json.loads(l) for l in open(OUT / s / "logs" / f"{stem}.jsonl")]
        by = ev.load_labels(PROJECT / it["labels"])
        off = round(float(it["start_s"]) * ev.FPS)
        _labels[s, stem] = (by, off + ev.best_shift(log, by, off))
    by, off = _labels[s, stem]
    labs = by.get(round(t * ev.FPS) + off, [])
    m = max(labs, key=lambda l: ev.iou(l[1], box), default=None)
    return m[2] if m and ev.iou(m[1], box) >= ev.IOU_MIN else "не человек из разметки"


if __name__ == "__main__":
    sys.exit(main())
