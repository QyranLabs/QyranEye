"""Сколько падений поймала система: сравнение тревог (CSV из main.py) с разметкой съёмки.

    .venv/bin/python main.py fimi/videos/oblique/ --view oblique   # сначала обработать видео -> results/
    .venv/bin/python main.py fimi/videos/nadir/ --view nadir
    .venv/bin/python eval/fall_recall.py                  # ролики для настройки (set=tune)
    .venv/bin/python eval/fall_recall.py --set test       # итоговая проверка — смотреть ОДИН раз, в конце

Разметка fimi/labels.csv — одна строка на одно действие:
    video     имя файла (FIMI0001.MP4)
    set       tune (можно смотреть и настраивать) или test (только итог)
    person    A, B, C... (без имён)
    view      oblique (под углом) или nadir (сверху)
    height_m  высота дрона
    kind      fall — упал и лежит >= 7 с: тревога ДОЛЖНА быть. Всё остальное — тревоги быть НЕ должно:
              short_fall (упал и сразу встал), slow_lie (лёг медленно), sit, squat, bag, pushups, walk...
    start_s   начало действия (у падения — момент, когда начал падать), секунды видео
    end_s     конец действия (у падения — момент, когда начал вставать)
    note      направление и прочее (side, along, knees, ...)

Падение поймано, если тревога пришла между start_s и end_s + 1 с. Одна тревога засчитывается одному падению.
Все остальные тревоги — ложные (повторная тревога на том же падении считается отдельно).
"""
import argparse, csv, math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

P = Path(__file__).resolve().parent.parent
TAIL_S, MIN_LIE_S = 1.0, 7.0      # запас после start_s..end_s; короче MIN_LIE_S тревога (4 с) может не успеть
COLS = {"video", "set", "person", "view", "height_m", "kind", "start_s", "end_s", "note"}


def wilson(k, n, z=1.96):
    """95% доверительный интервал доли k/n (способ Уилсона)."""
    if not n:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def match(actions, alarms):
    """actions: список словарей с kind/start/end; alarms: времена тревог одного видео.
    Возвращает ({номер падения: время тревоги}, [повторные], [(время, на каком действии)] ложные)."""
    caught, repeats, false = {}, [], []
    falls = [(i, a) for i, a in enumerate(actions) if a["kind"] == "fall"]
    for t in sorted(alarms):
        inside = [(i, a) for i, a in falls if a["start"] <= t <= a["end"] + TAIL_S]
        free = [i for i, _ in inside if i not in caught]
        if free:
            caught[free[0]] = t
        elif inside:
            repeats.append(t)
        else:
            on = [a["kind"] for a in actions if a["kind"] != "fall" and a["start"] <= t <= a["end"] + TAIL_S]
            false.append((t, on[0] if on else "вне разметки"))
    return caught, repeats, false


def main():
    ap = argparse.ArgumentParser(description="Доля пойманных падений и ложные тревоги")
    ap.add_argument("--labels", default=str(P / "fimi/labels.csv"))
    ap.add_argument("--events", default=str(P / "results"), help="папка с *_events.csv")
    ap.add_argument("--videos", default=str(P / "fimi/videos"), help="нужна только для длительности видео")
    ap.add_argument("--set", default="tune", choices=["tune", "test"])
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.labels, encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"Разметка пустая: {a.labels} — заполните по docs/shooting_plan.md")
    if not COLS <= set(rows[0]):
        raise SystemExit(f"В {a.labels} нужны столбцы: {', '.join(sorted(COLS))}")
    by_video, sets = defaultdict(list), defaultdict(set)
    for n, r in enumerate(rows, 2):
        try:
            r["start"], r["end"] = float(r["start_s"]), float(r["end_s"])
        except ValueError:
            raise SystemExit(f"Строка {n}: start_s и end_s должны быть числами (секунды)")
        if r["end"] <= r["start"]:
            raise SystemExit(f"Строка {n}: end_s должно быть больше start_s")
        if r["set"] not in ("tune", "test"):
            raise SystemExit(f"Строка {n}: set должен быть tune или test")
        sets[r["video"]].add(r["set"])
        if r["set"] == a.set:
            by_video[r["video"]].append(r)
    mixed = [v for v, s in sets.items() if len(s) > 1]
    if mixed:
        raise SystemExit(f"Один ролик не может быть и tune, и test: {mixed}")
    if not by_video:
        raise SystemExit(f"В разметке нет строк set={a.set}")
    missing = [v for v in by_video if not (Path(a.events) / f"{Path(v).stem}_events.csv").exists()]
    if missing:      # нет файла — это не «ноль тревог», а необработанное видео
        raise SystemExit(f"Нет результатов (сначала main.py): {missing}")

    groups, hard = defaultdict(Counter), Counter()
    n_caught = n_falls = n_rep = 0
    delays, missed, false_all, minutes, no_len = [], [], [], 0.0, []
    for v, acts in sorted(by_video.items()):
        acts.sort(key=lambda r: r["start"])
        ev = csv.DictReader(open(Path(a.events) / f"{Path(v).stem}_events.csv"))
        caught, rep, false = match(acts, [float(e["alarm_s"]) for e in ev])
        n_rep += len(rep)
        false_all += [(v, t, k) for t, k in false]
        for i, r in enumerate(acts):
            if r["kind"] != "fall":
                hard[r["kind"]] += 1
                continue
            if r["end"] - r["start"] < MIN_LIE_S:
                print(f"! {v} {r['start_s']} с: лежал меньше {MIN_LIE_S:.0f} с — для fall мало, проверьте разметку")
            n_falls += 1
            hit = i in caught
            n_caught += hit
            for g in (f"ракурс {r['view']}", f"высота {r['height_m']} м", f"человек {r['person']}", f"тип {r['note'] or '—'}"):
                groups[g]["n"] += 1
                groups[g]["ok"] += hit
            if hit:
                delays.append(caught[i] - r["start"])
            else:
                missed.append(f"{v} {r['start_s']} с ({r['view']}, {r['height_m']} м, {r['note']})")
        try:
            import cv2
            cap = cv2.VideoCapture(str(next(Path(a.videos).rglob(v), "")))   # видео может лежать в подпапке
            d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1)
            cap.release()
        except ImportError:
            d = 0
        minutes += d / 60
        if d <= 0:
            no_len.append(v)

    print(f"\n=== набор {a.set}: видео {len(by_video)}, падений {n_falls} ===")
    if n_falls:
        lo, hi = wilson(n_caught, n_falls)
        print(f"Поймано падений: {n_caught} из {n_falls} = {100 * n_caught / n_falls:.0f} %"
              f"  (95 % интервал {100 * lo:.0f}–{100 * hi:.0f} %)")
    if delays:
        print(f"Тревога через {median(delays):.1f} с после начала падения (медиана; максимум {max(delays):.1f} с)")
    per = f" = {len(false_all) / minutes:.2f} в минуту ({minutes:.1f} мин видео)" if minutes and not no_len else ""
    print(f"Ложных тревог: {len(false_all)}{per};  повторных на том же падении: {n_rep}")
    fk = Counter(k for _, _, k in false_all)
    if hard or fk:
        print("Обычные действия (ложных тревог / всего):",
              ", ".join(f"{k} {fk.get(k, 0)}/{n}" for k, n in sorted(hard.items())),
              f"; вне разметки {fk.get('вне разметки', 0)}")
    for g, c in sorted(groups.items()):
        print(f"  {g:22s} {c['ok']}/{c['n']}")
    for m in missed:
        print("  ПРОПУЩЕНО:", m)
    for v, t, k in false_all:
        print(f"  ЛОЖНАЯ: {v} {t:.1f} с ({k})")


if __name__ == "__main__":
    main()
