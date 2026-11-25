"""
Синтетические тесты логики падения (fall_logic.py) — без видео и без нейросетей.

Каждый тест — придуманный сценарий: рамка человека (и, если нужно, угол туловища или
ответ нейросети позы) меняется во времени так, как при падении, укладывании, ходьбе и т.д.
Тест проверяет итоговое состояние и число тревог FALL.

Запуск (из папки проекта):
    .venv/bin/python tests/test_fall_logic.py
Печатает OK / FAIL по каждому сценарию и в конце ИТОГ. Код выхода 1, если что-то не прошло.

Известное ограничение: N4 (широкий шаг при съёмке сверху в режиме правил) не проходит —
это граница простого правила по форме рамки, описана в README.
"""
import copy, math, sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import numpy as np
from video_pipeline import load_config
from fall_logic import FallMonitor

BASE = load_config(PROJECT / "config.yaml")
KNOWN_LIMITS = {"N4 широкий шаг каждые 0.4 с (40x75)"}

FPS = 15.0
fails = []


def cfg_with(**kw):
    c = copy.copy(BASE)
    c.pose_mode = "geometry"          # по умолчанию тесты проверяют правила; режим сети — явно
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def kpts_for(angle_deg, cx=500, cy=500, torso=60):
    k = np.zeros((17, 3)); a = math.radians(angle_deg)
    sh = (cx + torso * math.sin(a), cy - torso * math.cos(a))
    for i in (5, 6): k[i] = (*sh, 0.9)
    for i in (11, 12): k[i] = (*(cx, cy), 0.9)
    return k


def box(w, h, cx=500, cy=500):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def run(name, timeline, expect_final, expect_alarms, view="oblique", secs=20):
    """timeline: t -> (tid, box, kpts|None) или None"""
    m = FallMonitor(cfg_with(camera_view=view), FPS); states = []
    for i in range(int(secs * FPS)):
        t = i / FPS
        item = timeline(t)
        vis = m.update(t, [item] if item else [])
        if vis and (not states or states[-1][1] != vis[0].state):
            states.append((round(t, 2), vis[0].state))
    final = states[-1][1] if states else None
    ok = final == expect_final and len(m.events) == expect_alarms
    if not ok:
        fails.append(name)
    print(f"{'OK ' if ok else 'FAIL'} {name}  [view={m.view}]\n      states={states}  alarms={len(m.events)}"
          + (f" (ожидали {expect_final}, тревог {expect_alarms})" if not ok else ""))
    return m


def ramp(t, t0, dur, a, b):
    if t < t0: return a
    if t > t0 + dur: return b
    return a + (b - a) * (t - t0) / dur


# ---------------- ПОД УГЛОМ (старые сценарии, точки тела) ----------------
STAND, LIE = box(60, 160), box(200, 70)
ob = lambda ang: STAND if ang < 45 else LIE
print("=== под углом ===")
run("A падение за 0.5 с, лежит", lambda t: (1, ob(ramp(t, 3, .5, 5, 85)), kpts_for(ramp(t, 3, .5, 5, 85))), "FALL", 1)
run("B медленно лёг за 3 с", lambda t: (1, ob(ramp(t, 3, 3, 5, 85)), kpts_for(ramp(t, 3, 3, 5, 85))), "LYING", 0)
run("C упал и встал через 2.5 с", lambda t: (1, ob(a := (ramp(t, 3, .5, 5, 85) if t < 6 else 5)), kpts_for(a)), "UP", 0)
run("E без точек тела, только рамка", lambda t: (1, STAND if t < 3.3 else LIE, None), "FALL", 1)
run("F смена ID во время падения",
    lambda t: None if 3.3 < t < 3.6 else (1 if t < 3.3 else 7, ob(ramp(t, 3, .5, 5, 85)), kpts_for(ramp(t, 3, .5, 5, 85))), "FALL", 1)
run("G уже лежит при появлении", lambda t: (1, LIE, kpts_for(85)), "LYING", 0)
run("H один шумный кадр", lambda t: (1, STAND, kpts_for(85 if abs(t - 4) < .05 else 5)), "UP", 0)
run("I лежит головой к камере (170°)", lambda t: (1, ob(5 if t < 3 else 170), kpts_for(5 if t < 3 else 170)), "FALL", 1)

# ---------------- СВЕРХУ (только рамки) ----------------
# стоит: «пятно» 40x52 (вытянутость 1.3); лежит: 32x95 (вытянутость 3.0, длина x1.8)
def nad(t, t0, dur, end_w, end_h):
    return box(ramp(t, t0, dur, 40, end_w), ramp(t, t0, dur, 52, end_h))
print("=== сверху ===")
run("N1 падение за 0.5 с, лежит вдоль кадра", lambda t: (1, nad(t, 3, .5, 32, 95), None), "FALL", 1, "nadir")
run("N2 падение за 0.5 с, лежит поперёк кадра", lambda t: (1, nad(t, 3, .5, 95, 32), None), "FALL", 1, "nadir")
run("N3 сел (50x60)", lambda t: (1, nad(t, 3, .5, 50, 60), None), "UP", 0, "nadir")
run("N4 широкий шаг каждые 0.4 с (40x75)",
    lambda t: (1, box(40, 75) if int(t / .4) % 2 else box(40, 52), None), "UP", 0, "nadir")
run("N5 уже лежит при появлении", lambda t: (1, box(32, 95), None), "LYING", 0, "nadir")
run("N6 медленно лёг за 3 с", lambda t: (1, nad(t, 3, 3, 32, 95), None), "LYING", 0, "nadir")
run("N7 упал, лежит 3 с, встал", lambda t: (1, nad(t, 3, .5, 32, 95) if t < 6.5 else box(40, 52), None), "UP", 0, "nadir")
run("N8 угол туловища сверху НЕ используется",
    lambda t: (1, nad(t, 3, .5, 32, 95), kpts_for(5)), "FALL", 1, "nadir")

run("N9 смена ID при падении сверху (длина x1.7)",
    lambda t: None if 3.3 < t < 3.6 else (1 if t < 3.3 else 7, nad(t, 3, .5, 32, 90), None), "FALL", 1, "nadir")

# ---------------- ИСПРАВЛЕНИЯ ----------------
print("=== исправления ===")
# R1: человек пропал на 1 кадр, рядом появилась «сумка» вдвое меньше -> история не передаётся
m = FallMonitor(cfg_with(camera_view="oblique"), FPS)
P, BAG = (485, 470, 515, 532), (505, 500, 536, 534)          # человек 30x62, сумка 31x34
for i in range(int(20 * FPS)):
    t = i / FPS
    people = [] if i == 75 else [(1, P, None)]
    if i >= 75: people.append((2, BAG, None))
    m.update(t, people)
bag = m.tracks[2]
ok = len(m.events) == 0 and not bag.earlier_ids
print(f"{'OK ' if ok else 'FAIL'} R1 сумка рядом с пропавшим человеком: тревог {len(m.events)}, "
      f"история сумке передана: {bool(bag.earlier_ids)}, состояние сумки {bag.state}")
if not ok: fails.append("R1")
# R2: форма меняется медленно (0.6 -> 1.66 за 7 с, как №5 в Okutama 1.2.10) -> эталон не должен «уползти»
run("R2 медленный дрейф формы -> LYING, не UP",
    lambda t: (1, box(ramp(t, 3, 7, 36, 63), ramp(t, 3, 7, 60, 38)), None), "LYING", 0, "oblique")

# R3: очень узкий стоя (28x82), присел и сидит (53x82): рамка выросла в 1.9 раза, но она всё ещё выше, чем шире
# (NOMAD Actor035: раньше ложная тревога). R4: тот же человек по-настоящему упал -> тревога остаётся
NARROW, SIT, FELL = box(28, 82), box(53, 82), box(85, 60)
run("R3 узкий человек присел и сидит", lambda t: (1, NARROW if t < 3 else SIT, None), "UP", 0, "oblique")
run("R4 узкий человек упал", lambda t: (1, NARROW if t < 3 else FELL, None), "FALL", 1, "oblique")

# R5 (NOMAD Actor099): человека впервые увидели лежащим, он встал и 10 с идёт, рамка «дышит» (w/h 0.68-1.12),
# потом падает. Раньше LYING держался всю ходьбу, и падения система не видела.
def r5(t):
    if t < 4: b = box(150, 100, cx=400)
    elif t < 14: b = box(100 * (.9 + .22 * math.sin(2 * math.pi * t)), 100, cx=400 + 90 * (t - 4))
    else: b = box(165, 75, cx=1300)
    return (1, b, None)
run("R5 лежал, встал, идёт -> UP", r5, "UP", 0, "oblique", secs=13.5)
run("R5b ...и упал -> тревога", r5, "FALL", 1, "oblique", secs=22)
# R6: лежит на месте, рамка дрожит (w/h 0.95-1.45, поза часто переходная) -> LYING не снимается
run("R6 лежит, рамка дрожит", lambda t: (1, box(160, 100) if t < 3 else
    box(120 + 25 * math.sin(5 * t), 100, cx=500 + 10 * math.sin(9 * t)), None), "LYING", 0, "oblique")

# ---------------- СКОРОСТЬ ПАДЕНИЯ ПО ТОЧКАМ ТЕЛА ----------------
print("=== скорость падения ===")
def body(deg, cx=500, foot=580, height=160):
    """Человек, который поворачивается вокруг ступней: 0 — стоит, 90 — лежит. Рамка и точки тела."""
    a = math.radians(deg); at = lambda q: (cx + q * height * math.sin(a), foot - q * height * math.cos(a))
    k = np.zeros((17, 3))
    for i, q in ((5, .8), (6, .8), (11, .5), (12, .5), (15, .03), (16, .03)): k[i] = (*at(q), 0.9)
    hx, hy = at(1.0)
    return (min(cx, hx) - 28, min(foot, hy) - 5, max(cx, hx) + 28, foot + 5), k
def fall_body(secs):
    return lambda t: (1, *body(ramp(t, 3, secs, 0, 88)))
run("S1 упал за 0.8 с -> тревога", fall_body(.8), "FALL", 1, "oblique")
run("S2 лёг сам за 2.5 с -> LYING", fall_body(2.5), "LYING", 0, "oblique")
run("S3 упал в другую сторону", lambda t: (1, *body(-ramp(t, 3, .8, 0, 88))), "FALL", 1, "oblique")
def s4(t):
    """стоит на полпути (наклонился на 25°) 3 с, потом рухнул за 0.6 с"""
    return (1, *body(ramp(t, 3, 1, 0, 25) if t < 7 else ramp(t, 7, .6, 25, 88)))
run("S4 наклонился, постоял, рухнул -> тревога", s4, "FALL", 1, "oblique")
# S5: ложился 3.6 с, но середину поворота прошёл за 0.6 с (как NOMAD Actor042). Раньше — ложная тревога.
def s5(t):
    return (1, *body(ramp(t, 3, 1.5, 0, 25) if t < 4.5 else ramp(t, 4.5, .6, 25, 65) if t < 5.1 else ramp(t, 5.1, 1.5, 65, 88)))
run("S5 лёг сам рывком в середине -> LYING", s5, "LYING", 0, "oblique")

# ---------------- ПРОВЕРКА ПЕРЕД ТРЕВОГОЙ, КРАЙ КАДРА, ДВИЖЕНИЕ ДРОНА ----------------
print("=== проверка перед тревогой ===")
def loop(name, frame_fn, expect_alarms, expect_state=None, view="oblique", secs=20, frame_size=None, pose_mode="geometry"):
    """frame_fn(i, t) -> (people, warp)"""
    m = FallMonitor(cfg_with(camera_view=view, pose_mode=pose_mode), FPS, frame_size)
    for i in range(int(secs * FPS)):
        t = i / FPS
        people, warp = frame_fn(i, t)
        m.update(t, people, warp)
    st = m.tracks[1].state if 1 in m.tracks else None
    ok = len(m.events) == expect_alarms and (expect_state is None or st == expect_state)
    print(f"{'OK ' if ok else 'FAIL'} {name}: тревог {len(m.events)}, отменено {m.cancelled}, итог {st}")
    if not ok: fails.append(name)
    return m

# T1: упал, через 2 с встал и пошёл (зона прыгает UP/MID, как #355 в 2.2.10)
def t1(i, t):
    if t < 3: b = STAND
    elif t < 5: b = LIE
    else: b = box(60, 160 if int(t / .3) % 2 else 110, cx=500 + 60 * (t - 5))     # идёт, форма «прыгает»
    return [(1, b, None)], None
loop("T1 упал, встал и пошёл до recover_s", t1, 0)
# T2: «сумка»: резко стала широкой и перемещается на ~1 длину тела в секунду
loop("T2 широкое и движется (сумку несут)",
     lambda i, t: ([(1, STAND if t < 3 else box(200, 70, cx=500 + 200 * (t - 3)), None)], None), 0, "LYING")
# T3: человек лежит неподвижно, но дрон плывёт на 15 px за кадр (~1.1 длины тела в секунду)
def pan(i, t, with_warp):
    shift = 15 * i
    b = STAND if t < 3 else LIE
    b = (b[0] + shift, b[1], b[2] + shift, b[3])
    warp = [[1, 0, 15], [0, 1, 0]] if (with_warp and i > 0) else None
    return [(1, b, None)], warp
loop("T3 дрон плывёт, есть поправка -> тревога", lambda i, t: pan(i, t, True), 1, "FALL")
loop("T3b дрон плывёт, БЕЗ поправки -> ошибочная отмена", lambda i, t: pan(i, t, False), 0)
# T4: рамка касается края кадра -> поза не определяется (нет ложного падения)
loop("T4 рамка у края кадра", lambda i, t: ([(1, STAND if t < 3 else (0, 470, 200, 540), None)], None), 0,
     frame_size=(1920, 1080))

# ---------------- НЕЙРОСЕТЬ ПОЗЫ ----------------
print("=== нейросеть позы ===")
def net(i, t, p_fn, b_fn=lambda t: STAND):
    return [(1, b_fn(t), None, p_fn(t))], None
loop("P1 сеть: стоит 3 с, за 0.4 с «лежит» -> тревога", lambda i, t: net(i, t, lambda t: 0.05 if t < 3 else (0.4 if t < 3.4 else 0.9)), 1, "FALL", pose_mode="classifier")
loop("P2 сеть: рамка вытянута как у лежащего, но сеть говорит «стоит» -> нет тревоги",
     lambda i, t: net(i, t, lambda t: 0.05, lambda t: STAND if t < 3 else LIE), 0, "UP", pose_mode="classifier")
loop("P3 сеть: человек с сумкой сверху (вытянутая рамка), сеть «стоит» -> UP",
     lambda i, t: net(i, t, lambda t: 0.1, lambda t: box(40, 90)), 0, "UP", view="nadir", pose_mode="classifier")
loop("P4 сеть: уже лежит при появлении -> LYING, без тревоги", lambda i, t: net(i, t, lambda t: 0.95), 0, "LYING", pose_mode="classifier")
m = FallMonitor(cfg_with(camera_view="oblique", pose_mode="geometry"), FPS)
for i in range(int(12 * FPS)):
    t = i / FPS; m.update(t, [(1, STAND, None, 0.99)])
ok = m.tracks[1].state == "UP"
print(f"{'OK ' if ok else 'FAIL'} P5 pose_mode=geometry: ответ сети игнорируется -> {m.tracks[1].state}")
if not ok: fails.append("P5")

# ---------------- СОВМЕСТНЫЙ РЕЖИМ (hybrid) ----------------
print("=== hybrid ===")
def hyb(b_fn, p_fn, np_fn=lambda t: 0.0):
    return lambda i, t: ([(1, b_fn(t), None, {"lying": p_fn(t), "not_person": np_fn(t)})], None)
H = dict(pose_mode="hybrid")
loop("H1 настоящее падение: форма быстро, сеть «лежит» -> тревога",
     hyb(lambda t: ob(ramp(t, 3, .3, 5, 85)), lambda t: 0.05 if t < 3.3 else 0.9), 1, "FALL", **H)
loop("H2 сверху человек с сумкой: форма «лежит», сеть «нет» -> не лежит",
     hyb(lambda t: box(40, 90), lambda t: 0.05), 0, "UP", view="nadir", **H)
loop("H3 «лежит» только по сети (форма стоя) -> LYING, без тревоги",
     hyb(lambda t: STAND, lambda t: 0.05 if t < 3 else 0.95), 0, "LYING", **H)
loop("H4 медленно лёг за 3 с (угол туловища), сеть переключилась резко -> LYING, без тревоги",
     lambda i, t: ([(1, ob(ramp(t, 3, 3, 5, 85)), kpts_for(ramp(t, 3, 3, 5, 85)),
                     {"lying": 0.05 if t < 4 else 0.95, "not_person": 0.0})], None), 0, "LYING", **H)
loop("H5 «не человек» (сумка) быстро «легла» -> без тревоги",
     hyb(lambda t: STAND if t < 3 else LIE, lambda t: 0.9, lambda t: 0.9), 0, None, **H)
loop("H6 форма упала быстро, но сеть не согласна (0.4) -> отмена, без тревоги",
     hyb(lambda t: STAND if t < 3 else LIE, lambda t: 0.05 if t < 3 else 0.4), 0, None, **H)

# ---------------- AUTO: определение ракурса ----------------
print("=== auto ===")
def scene(name, falling, others, expect_view, objects=None, secs=20):
    """Сцена из нескольких людей: falling(t) -> рамка падающего (id 1), others — рамки
    стоящих (id 2, 3, ...), objects — рамки предметов только в первых 2 кадрах."""
    m = FallMonitor(cfg_with(camera_view="auto"), FPS)
    for i in range(int(secs * FPS)):
        t = i / FPS
        people = [(1, falling(t), None)] + [(k + 2, b, None) for k, b in enumerate(others)]
        if objects and i < 2:
            people += [(100 + k, b, None) for k, b in enumerate(objects)]
        m.update(t, people)
    ok = m.view == expect_view and len(m.events) == 1 and m.tracks[1].state == "FALL"
    print(f"{'OK ' if ok else 'FAIL'} {name}: view={m.view} ({m.view_reason}), тревог {len(m.events)}, "
          f"падающий {m.tracks[1].state}")
    if not ok: fails.append(name)

TALL = [box(30, 80, 200, 200), box(28, 75, 800, 300), box(32, 85, 400, 700)]     # под углом: высокие
BLOB = [box(40, 52, 200, 200), box(38, 50, 800, 300), box(42, 55, 400, 700)]      # сверху: «пятна»
WIDE = [box(90, 40, 100 + 60 * k, 900) for k in range(18)]                       # скамейки/сумки
scene("V1 auto, сцена под углом, падение на 7 с", lambda t: ob(ramp(t, 7, .5, 5, 85)), TALL, "oblique")
scene("V2 auto, сцена сверху, падение на 7 с", lambda t: nad(t, 7, .5, 32, 95), BLOB, "nadir")
scene("V5 auto, как 1.2.3: в первых кадрах много широких предметов", lambda t: ob(ramp(t, 7, .5, 5, 85)),
      TALL, "oblique", objects=WIDE)
m = FallMonitor(cfg_with(camera_view="auto"), FPS)
for i in range(int(20 * FPS)):
    m.update(i / FPS, [(1, STAND, None)] if i == 20 else [])
print(f"{'OK ' if m.view == 'oblique' else 'FAIL'} V3 мало людей -> camera_view_default: {m.view} ({m.view_reason})")
if m.view != BASE.camera_view_default: fails.append("V3")
try:
    FallMonitor(cfg_with(camera_view="top"), FPS); fails.append("V4")
except ValueError as e:
    print("OK  V4 неверное значение camera_view ->", e)

unexpected = [f for f in fails if f not in KNOWN_LIMITS]
print("\nИТОГ:", "все тесты прошли" if not fails else f"не прошли: {fails}")
if fails and not unexpected:
    print("(только известные ограничения — см. README)")
sys.exit(1 if unexpected else 0)
