"""
Логика «упал ли человек» — без нейросетей и без OpenCV, только геометрия и время.

На вход каждый анализируемый кадр получает список людей от трекера:
    (track_id, рамка (x1, y1, x2, y2), точки тела или None)
и время кадра в секундах видео.

Для каждого человека (track_id) хранится своя история (класс Track).
Решение принимается в три шага:

  0. РАКУРС КАМЕРЫ (один раз на видео, метод _decide_view): под углом (oblique)
     или строго сверху (nadir). Если в конфиге auto — через 5 с после появления
     первого человека, каждый человек голосует своей медианой ширина/высота:
     сверху стоящие люди — «пятна» (~1), под углом — узкие высокие рамки (~0.5).
     Пока ракурс не ясен, выводов не делаем.
  1. ПОЗА В ЭТОМ КАДРЕ -> одна из зон UP / MID / DOWN (функция classify):
     РЕЖИМ (pose_mode):
       geometry   — только правила по форме рамки (ниже);
       classifier — только нейросеть позы (вероятность «лежит» по картинке человека);
       hybrid     — правила + сеть: сеть отменяет ложное «лежит» правил (человек с сумкой)
                    и добавляет «лежит», если уверена, но FALL? запускает только быстрая
                    смена ФОРМЫ рамки; «не человек» (сумка, стул) -> выводов нет;
                    тревога — только если сеть тоже говорила «лежит».
     ПРАВИЛА ПО ФОРМЕ РАМКИ:
     ПОД УГЛОМ:
       - если человек крупный и видны плечи и бёдра: угол туловища к вертикали кадра;
       - иначе: насколько рамка стала «шире», чем была у этого же человека стоя;
       - пока эталона нет: абсолютное отношение ширина/высота.
     СВЕРХУ:
       - вытянутость рамки (длинная сторона / короткая, направление не важно)
         и рост длинной стороны по сравнению с этим же человеком стоя;
       - пока эталона нет: только вытянутость.
  2. СГЛАЖИВАНИЕ: зона по большинству голосов за vote_window_s (чтобы один
     шумный кадр не считался падением).
  3. КОНЕЧНЫЙ АВТОМАТ (метод _step):
       UP --(перешёл в DOWN быстро)--------------------> FALL?  (жёлтый)
           «быстро»: есть точки тела — плечи сдвинулись относительно щиколоток со скоростью
           >= fall_speed_min ростов в секунду; точек нет — переход короче fall_transition_s
       UP --(перешёл в DOWN медленнее)----------------> LYING  (просто лёг, без тревоги)
       FALL? --(через lying_alarm_s: лежал >= alarm_down_share кадров, лежит сейчас
                и не перемещается быстрее still_speed_max)--> FALL (красный, событие в CSV)
       FALL? --(через lying_alarm_s: лежал мало или перемещается)--> отмена (UP или LYING)
       FALL?/FALL/LYING --(стоит recover_s подряд)----> UP
       LYING --(recover_s подряд стоит или идёт: поза переходная и скорость > still_speed_max;
              в режиме classifier — только стоит)--> UP
"""

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median

# Номера точек тела в формате COCO (их выдаёт YOLO pose)
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 5, 6, 11, 12, 15, 16

# Зоны позы в одном кадре
UP, MID, DOWN = "UP", "MID", "DOWN"

# Состояния человека
UNKNOWN = "?"        # ещё мало данных
UPRIGHT = "UP"       # стоит / идёт
FALLING = "FALL?"    # быстро перешёл в горизонталь, ждём lying_alarm_s
FALLEN = "FALL"      # тревога
LYING = "LYING"      # лежит, но падения не было (лёг медленно или уже лежал)

# Ракурс камеры
OBLIQUE = "oblique"  # под углом (45–60°)
NADIR = "nadir"      # строго вниз (~90°)


def torso_angle(kpts, kp_conf):
    """Угол туловища к «верху» кадра в градусах (0 = стоит прямо, 90 = лежит поперёк,
    180 = вниз головой в кадре). None, если плечо и бедро одной стороны не видны.

    Вектор идёт от середины бёдер к середине плеч. Знак важен: у стоящего человека
    плечи в кадре ВЫШЕ бёдер при любом наклоне камеры дрона. Если плечи ниже бёдер —
    человек лежит головой к камере, и это тоже даст большой угол (почти 180°).
    """
    sides = []
    for sh, hip in ((L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP)):
        if kpts[sh][2] >= kp_conf and kpts[hip][2] >= kp_conf:
            sides.append((kpts[sh], kpts[hip]))
    if not sides:
        return None, 0.0
    sx = sum(s[0][0] for s in sides) / len(sides)
    sy = sum(s[0][1] for s in sides) / len(sides)
    hx = sum(s[1][0] for s in sides) / len(sides)
    hy = sum(s[1][1] for s in sides) / len(sides)
    dx, dy = sx - hx, sy - hy            # ось y в картинке направлена ВНИЗ
    length = math.hypot(dx, dy)
    if length < 1:
        return None, 0.0
    return math.degrees(math.atan2(abs(dx), -dy)), length


@dataclass
class Track:
    tid: int
    baseline_len: int
    state: str = UNKNOWN
    zone: str | None = None              # сглаженная зона
    raw_zone: str | None = None          # зона в этом кадре (до сглаживания)
    cue: str = ""                        # каким признаком решили (для подписи на видео)
    box: tuple = (0, 0, 0, 0)
    kpts: object = None
    last_seen: float = 0.0
    last_up_t: float | None = None       # когда последний раз стоял
    down_since: float | None = None      # когда перешёл в горизонталь
    up_since: float | None = None        # когда снова встал (для снятия тревоги)
    event: dict | None = None            # текущее событие падения (строка CSV)
    earlier_ids: list = field(default_factory=list)
    votes: deque = field(default_factory=deque)
    aspects: deque = None                # эталон: ширина/высота, пока стоял (под углом)
    longs: deque = None                  # эталон: длинная сторона рамки, пока стоял (сверху)
    path: deque = field(default_factory=deque)   # (t, x, y, длина) за still_window_s, в координатах текущего кадра
    speed: float | None = None           # скорость в длинах тела / с (с поправкой на движение дрона)
    wait_votes: dict = field(default_factory=dict)   # зоны за время ожидания тревоги (FALL?)
    wait_speeds: list = field(default_factory=list)  # скорости за время ожидания тревоги
    wait_net: list = field(default_factory=list)     # hybrid: «лежит» ли по сети за время ожидания
    geo_votes: deque = field(default_factory=deque)  # hybrid: голоса правил по форме (отдельно)
    geo_zone: str | None = None                      # hybrid: сглаженная зона только по форме
    geo_raw: str | None = None
    p_lying: float | None = None                     # ответ сети в этом кадре
    reach: deque = field(default_factory=deque)      # (t, dx, dy): плечи относительно щиколоток, по точкам тела
    net_down_since: float | None = None              # hybrid: с какого момента «лежит» только по сети

    def __post_init__(self):
        self.aspects = deque(maxlen=self.baseline_len)
        self.longs = deque(maxlen=self.baseline_len)


class FallMonitor:
    def __init__(self, cfg, analysis_fps, frame_size=None):
        self.cfg = cfg
        self.frame_size = frame_size     # (ширина, высота) кадра — для правила «рамка у края»
        self.tracks: dict[int, Track] = {}
        self.events: list[dict] = []
        self.cancelled = 0               # сколько FALL? отменено проверкой перед тревогой
        # сколько замеров «стоя» хранить для эталона формы рамки
        self.baseline_len = max(3, round(cfg.baseline_window_s * analysis_fps))
        self.baseline_min = max(3, self.baseline_len // 4)

        # скорость падения по точкам тела (в старых конфигах этих строк может не быть — тогда правило выключено)
        self.fall_speed_min = getattr(cfg, "fall_speed_min", 0)
        self.fall_win_s = getattr(cfg, "fall_speed_window_s", 0.3)
        self.fall_back_s = getattr(cfg, "fall_speed_back_s", 1.5)

        if cfg.pose_mode not in ("classifier", "geometry", "hybrid"):
            raise ValueError(f"pose_mode: ожидается geometry, classifier или hybrid, а не {cfg.pose_mode!r}")
        if cfg.camera_view not in ("auto", OBLIQUE, NADIR):
            raise ValueError(f"camera_view: ожидается auto, oblique или nadir, а не {cfg.camera_view!r}")
        self.view = None if cfg.camera_view == "auto" else cfg.camera_view
        self.view_reason = "" if self.view is None else "задан вручную"
        self.view_sure = self.view is not None            # False — ракурс взят по умолчанию, людей было мало
        self._view_samples: dict[int, list[float]] = {}   # tid -> замеры ширина/высота
        self._view_first_t: float | None = None

    # ------------------------------------------------------------------
    def update(self, t, people, warp=None):
        """people: список (tid, box, kpts|None). warp: сдвиг камеры 2x3 с прошлого
        анализируемого кадра (или None). Возвращает список Track, видимых в кадре."""
        if self.view is None:
            self._decide_view(t, people)
        if warp is not None:                     # дрон сдвинулся -> пересчитать прошлые положения людей
            for tr in self.tracks.values():
                tr.path = deque((pt, warp[0][0]*x + warp[0][1]*y + warp[0][2],
                                 warp[1][0]*x + warp[1][1]*y + warp[1][2], L) for pt, x, y, L in tr.path)
        current = {p[0] for p in people}
        visible = []
        for tid, box, kpts, *extra in people:
            net = extra[0] if extra else None        # ответ нейросети позы: {класс: вероятность} или число
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self._inherit(tid, box, t, current) or Track(tid, self.baseline_len)
                self.tracks[tid] = tr
            tr.box, tr.kpts, tr.last_seen = box, kpts, t
            self._update_speed(tr, t, box)
            self._update_reach(tr, t, box, kpts)

            tr.raw_zone = self.classify(tr, box, kpts, net)
            tr.zone = self._smooth(tr, t)
            if self.cfg.pose_mode == "hybrid":
                tr.geo_zone = self._vote(tr.geo_votes, tr.geo_raw, t, tr.geo_zone)
            self._step(tr, t)

            # эталон формы пополняем только когда человек уверенно стоит И рамка сама по себе
            # похожа на стоящего — иначе при медленном изменении формы эталон «уползает»
            if tr.zone == UP and tr.state == UPRIGHT and self._upright_shape(box):
                tr.aspects.append(self._aspect(box))
                tr.longs.append(max(box[2] - box[0], box[3] - box[1]))
            visible.append(tr)

        # забываем тех, кого давно нет
        for tid in [k for k, v in self.tracks.items() if t - v.last_seen > self.cfg.forget_s]:
            del self.tracks[tid]
        return visible

    # ------------------------------------------------------------------
    @staticmethod
    def _aspect(box):
        x1, y1, x2, y2 = box
        return max(1.0, x2 - x1) / max(1.0, y2 - y1)

    def _update_speed(self, tr, t, box):
        """Скорость = смещение центра рамки за still_window_s (в координатах текущего
        кадра, то есть с поправкой на движение дрона), делённое на длину тела."""
        w = self.cfg.still_window_s
        tr.path.append((t, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, max(box[2] - box[0], box[3] - box[1])))
        while tr.path and t - tr.path[0][0] > w:
            tr.path.popleft()
        t0, x0, y0, _ = tr.path[0]
        if t - t0 >= 0.8 * w:
            length = median(p[3] for p in tr.path)
            tr.speed = math.hypot(tr.path[-1][1] - x0, tr.path[-1][2] - y0) / max(1.0, length) / (t - t0)
        else:
            tr.speed = None

    def _update_reach(self, tr, t, box, kpts):
        """Запоминаем вектор «щиколотки -> плечи» (если точки видны). Он не зависит от того,
        идёт ли человек и летит ли дрон, и резко меняется, когда человек падает в любую сторону."""
        c = self.cfg
        if kpts is not None and max(box[2] - box[0], box[3] - box[1]) >= c.min_keypoint_px:
            sh = [kpts[i] for i in (L_SHOULDER, R_SHOULDER) if kpts[i][2] >= c.keypoint_conf]
            an = [kpts[i] for i in (L_ANKLE, R_ANKLE) if kpts[i][2] >= c.keypoint_conf]
            if sh and an:
                tr.reach.append((t, sum(p[0] for p in sh) / len(sh) - sum(p[0] for p in an) / len(an),
                                 sum(p[1] for p in sh) / len(sh) - sum(p[1] for p in an) / len(an)))
        while tr.reach and t - tr.reach[0][0] > self.fall_back_s + self.fall_win_s:
            tr.reach.popleft()

    def _fall_speed(self, tr):
        """Наибольшая скорость плеч относительно щиколоток за последние fall_speed_back_s,
        в ростах за секунду (рост — рамка этого же человека стоя). None — точек или роста нет."""
        if len(tr.longs) < 3:
            return None
        height, best, pts = median(tr.longs), None, list(tr.reach)
        for i, (t0, x0, y0) in enumerate(pts):
            for t1, x1, y1 in pts[i + 1:]:
                if t1 - t0 >= self.fall_win_s - 0.02:         # первая точка через fall_speed_window_s
                    if t1 - t0 <= self.fall_win_s * 1.6:      # (если точки не пропадали надолго)
                        v = math.hypot(x1 - x0, y1 - y0) / (t1 - t0) / height
                        best = v if best is None else max(best, v)
                    break
        return best

    def _went_down_fast(self, tr, t):
        """Упал (быстро) или лёг сам (медленно)? Считается в момент перехода в DOWN."""
        tr_fast = t - tr.last_up_t <= self.cfg.fall_transition_s
        v = self._fall_speed(tr) if self.fall_speed_min and self.view == OBLIQUE else None
        return (tr_fast if v is None else v >= self.fall_speed_min), v

    def _at_edge(self, box):
        if self.frame_size is None:
            return False
        m, (fw, fh) = self.cfg.edge_margin_px, self.frame_size
        return box[0] <= m or box[1] <= m or box[2] >= fw - m or box[3] >= fh - m

    def _upright_shape(self, box):
        """Похожа ли рамка на стоящего человека сама по себе (без сравнения с эталоном).
        Под углом: узкая и высокая; сверху: компактное «пятно»."""
        w, h = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
        if self.view == NADIR:
            return max(w, h) / min(w, h) < self.cfg.nadir_elong_upright_max
        return w / h <= self.cfg.aspect_abs_upright_max

    def _decide_view(self, t, people):
        """auto: ракурс по форме рамок людей. Каждый человек (track_id) голосует
        один раз — медианой своего ширина/высота. Решаем через view_auto_s секунд
        после появления первого человека, если набралось view_min_tracks человек;
        ждём до view_max_wait_s и берём camera_view_default: по одному-двум людям форме
        верить нельзя (лежащий под углом похож на стоящего, снятого сверху)."""
        c = self.cfg
        for tid, box, *_ in people:
            if self._view_first_t is None:
                self._view_first_t = t
            self._view_samples.setdefault(tid, []).append(self._aspect(box))
        if self._view_first_t is None:
            return
        waited = t - self._view_first_t
        votes = [median(v) for v in self._view_samples.values() if len(v) >= 3]
        if waited < c.view_auto_s or (len(votes) < c.view_min_tracks and waited < c.view_max_wait_s):
            return
        if len(votes) >= c.view_min_tracks:
            m = median(votes)
            self.view_sure = True
            self.view = NADIR if m >= c.view_nadir_min_wh else OBLIQUE
            self.view_reason = f"auto: медиана ширина/высота {m:.2f}, голосов {len(votes)}"
        else:
            self.view = c.camera_view_default
            self.view_reason = f"auto: людей всего {len(votes)}, взят camera_view_default"

    def classify(self, tr, box, kpts, net=None):
        """Зона позы UP / MID / DOWN в одном кадре (или None — не понять)."""
        c = self.cfg
        p_lying, p_np = (net, 0.0) if isinstance(net, (int, float)) else \
            ((net.get("lying"), net.get("not_person", 0.0)) if net else (None, 0.0))
        tr.p_lying, tr.geo_raw = p_lying, None
        if self.view is None:
            tr.cue = "view?"                    # ракурс ещё не определён
            return None
        if self._at_edge(box):
            tr.cue = "edge"                     # рамка обрезана краем кадра — форма искажена
            return None
        if c.pose_mode == "classifier" and p_lying is not None:
            return self._classify_net(tr, p_lying)
        if c.pose_mode == "hybrid" and p_lying is not None and p_np >= c.pose_not_person_min:
            tr.cue = f"not person {p_np:.2f}"   # сеть: это не человек (сумка, стул, тень)
            return None
        g = self._classify_nadir(tr, box) if self.view == NADIR else self._classify_oblique(tr, box, kpts)
        if c.pose_mode != "hybrid" or p_lying is None:
            return g
        tr.geo_raw = g
        if g == DOWN and p_lying <= c.pose_upright_max:
            tr.cue += f" net{p_lying:.2f}:no"   # форма «лежит», но сеть уверена, что стоит/сидит
            return UP
        if g != DOWN and p_lying >= c.pose_net_down_min:
            tr.cue += f" net{p_lying:.2f}"      # сеть уверена «лежит», форма ещё нет
            return DOWN
        tr.cue += f" net{p_lying:.2f}"
        return g

    def _classify_net(self, tr, p_lying):
        """Нейросеть позы смотрит на самого человека (картинку вокруг рамки) и даёт
        вероятность «лежит». Она одинаково работает под углом и сверху, и её не обманывает
        вытянутая рамка человека с сумкой или тележкой (в отличие от правил по форме)."""
        c = self.cfg
        tr.cue = f"p{p_lying:.2f}"
        if p_lying >= c.pose_lying_min:
            return DOWN
        if p_lying <= c.pose_upright_max:
            return UP
        return MID

    def _classify_nadir(self, tr, box):
        """Съёмка сверху: стоящий — компактное «пятно», лежащий — вытянутая полоса
        в любом направлении, длиннее, чем был он же стоя."""
        c = self.cfg
        w, h = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
        elong, long_side = max(w, h) / min(w, h), max(w, h)

        if len(tr.longs) >= self.baseline_min:
            grow = long_side / median(tr.longs)
            tr.cue = f"e{elong:.1f} L{grow:.1f}"
            if elong >= c.nadir_elong_lying_min and grow >= c.nadir_size_factor:
                return DOWN
            if elong < c.nadir_elong_upright_max and grow < c.nadir_size_factor:
                return UP
            return MID

        tr.cue = f"e{elong:.1f}"                # эталона нет — только вытянутость
        if elong < c.nadir_elong_upright_max:
            return UP
        if elong >= c.nadir_elong_lying_min:
            return DOWN
        return MID

    def _classify_oblique(self, tr, box, kpts):
        """Съёмка под углом: стоящий — узкая высокая рамка, лежащий — широкая."""
        c = self.cfg
        x1, y1, x2, y2 = box
        size = max(x2 - x1, y2 - y1)

        # Признак 1: угол туловища по точкам тела
        if kpts is not None and size >= c.min_keypoint_px:
            ang, length = torso_angle(kpts, c.keypoint_conf)
            if ang is not None and length >= c.torso_min_frac * size:
                tr.cue = f"{ang:.0f}deg"
                if ang <= c.torso_upright_max_deg:
                    return UP
                if ang >= c.torso_lying_min_deg:
                    return DOWN
                return MID

        aspect = self._aspect(box)
        # Признак 2: сравниваем человека с ним же самим, пока он стоял
        if len(tr.aspects) >= self.baseline_min:
            ratio = aspect / median(tr.aspects)
            tr.cue = f"x{ratio:.1f}"
            if ratio < c.aspect_up_factor:
                return UP
            # «Лежит» — только если рамка и выросла, и сама по себе шире, чем у стоящего:
            # иначе присевший человек, который стоя был очень узким, выглядит как упавший.
            if ratio >= c.aspect_down_factor and aspect > c.aspect_abs_upright_max:
                return DOWN
            return MID

        # Признак 3: эталона ещё нет — абсолютная форма рамки
        tr.cue = f"w/h {aspect:.1f}"
        if aspect <= c.aspect_abs_upright_max:
            return UP
        if aspect >= c.aspect_abs_lying_min:
            return DOWN
        return MID

    def _smooth(self, tr, t):
        """Большинство голосов за последние vote_window_s секунд."""
        return self._vote(tr.votes, tr.raw_zone, t, tr.zone)

    def _vote(self, votes, zone, t, prev):
        votes.append((t, zone))
        while votes and t - votes[0][0] > self.cfg.vote_window_s:
            votes.popleft()
        counts = {}
        for _, z in votes:
            if z is not None:
                counts[z] = counts.get(z, 0) + 1
        if not counts:
            return prev
        best = max(counts.values())
        # при равенстве голосов побеждает самый свежий
        for _, z in reversed(votes):
            if counts.get(z) == best:
                return z

    def _step(self, tr, t):
        """Конечный автомат: меняет tr.state по сглаженной зоне."""
        c = self.cfg
        z = tr.zone

        # «Лежит» (LYING), но поза переходная и человек перемещается — значит, он встал и идёт: при ходьбе
        # рамка то шире, то уже, и «стоит» recover_s подряд может не набраться. Тревогу (FALL) это не снимает.
        # В режиме classifier не применяем: там MID значит «сеть сомневается», и терялись лежащие.
        walking = (c.pose_mode != "classifier" and tr.state == LYING and z == MID
                   and tr.speed is not None and tr.speed > c.still_speed_max)

        if z == UP or walking:
            if z == UP:
                tr.last_up_t = t
            if tr.state in (FALLING, FALLEN, LYING):
                if tr.up_since is None:
                    tr.up_since = t
                if t - tr.up_since >= c.recover_s:      # встал и стоит -> снимаем тревогу
                    if tr.event is not None:
                        tr.event["recovered_s"] = round(t, 2)
                        tr.event = None
                    tr.state, tr.down_since, tr.up_since = UPRIGHT, None, None
            else:
                tr.state = UPRIGHT

        elif z == DOWN:
            tr.up_since = None
            if tr.state in (UNKNOWN, UPRIGHT) and c.pose_mode == "hybrid" and tr.geo_zone != DOWN \
                    and tr.p_lying is not None:
                # «лежит» только по сети: падение так не засчитываем. Ждём, догонит ли форма;
                # если сеть держит «лежит» pose_net_lying_s, а форма нет -> просто LYING.
                if tr.net_down_since is None:
                    tr.net_down_since = t
                if t - tr.net_down_since >= c.pose_net_lying_s:
                    tr.state, tr.down_since, tr.net_down_since = LYING, t, None
            elif tr.state in (UNKNOWN, UPRIGHT):
                tr.net_down_since = None
                tr.down_since = t
                if tr.last_up_t is None:
                    tr.state = LYING                    # стоящим его не видели
                else:
                    fast, v = self._went_down_fast(tr, t)
                    if fast:
                        tr.state = FALLING              # быстрый переход = падение
                        tr.event = {"track_id": tr.tid,
                                    "fall_start_s": round(tr.last_up_t, 2),
                                    "transition_s": round(t - tr.last_up_t, 2),
                                    "fall_speed": "" if v is None else round(v, 2)}
                        tr.wait_votes, tr.wait_speeds, tr.wait_net = {}, [], []
                    else:
                        tr.state = LYING                # ложился медленно

        elif z == MID:
            tr.up_since = None                          # переходная поза: ничего не решаем
        if z != DOWN:
            tr.net_down_since = None

        # FALL? — копим, как человек вёл себя во время ожидания, и через lying_alarm_s решаем
        if tr.state == FALLING:
            if z is not None:
                tr.wait_votes[z] = tr.wait_votes.get(z, 0) + 1
            if tr.speed is not None:
                tr.wait_speeds.append(tr.speed)
            if tr.p_lying is not None:
                tr.wait_net.append(tr.p_lying >= c.pose_lying_min)
            if t - tr.down_since >= c.lying_alarm_s:
                self._check_alarm(tr, t, z)

    def _check_alarm(self, tr, t, z):
        """Тревога, только если человек действительно лежал всё это время и не перемещался.
        Иначе FALL? отменяется: человек встал и пошёл, или это предмет, который несут."""
        c = self.cfg
        n = sum(tr.wait_votes.values())
        share = tr.wait_votes.get(DOWN, 0) / n if n else 0.0
        speed = median(tr.wait_speeds) if len(tr.wait_speeds) >= 3 else None
        moving = speed is not None and speed > c.still_speed_max
        # hybrid: сеть тоже должна была «видеть» лежащего человека большую часть ожидания
        net_no = c.pose_mode == "hybrid" and len(tr.wait_net) >= 3 and \
            sum(tr.wait_net) / len(tr.wait_net) < c.alarm_net_share
        if share < c.alarm_down_share or moving or net_no:
            self.cancelled += 1
            tr.event = None
            tr.state = LYING if z == DOWN else UPRIGHT
            if tr.state == UPRIGHT:
                tr.down_since = None
        elif z == DOWN:
            tr.state = FALLEN                           # лежит и не двигается -> тревога
            tr.event.update(alarm_s=round(t, 2), box=tr.box,
                            earlier_ids=" ".join(map(str, tr.earlier_ids)), recovered_s="",
                            down_share=round(share, 2), speed="" if speed is None else round(speed, 2))
            self.events.append(tr.event)
        # иначе (лежал, не двигался, но именно сейчас не DOWN) — ждём следующего кадра

    def _inherit(self, tid, box, t, current):
        """Новый ID появился рядом с человеком, которого только что потеряли?
        При падении рамка резко меняет форму, и трекер может выдать новый номер.
        Тогда переносим историю, чтобы не потерять уже начавшееся падение."""
        c = self.cfg
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best, best_d = None, None
        for old in self.tracks.values():
            if old.tid in current or t - old.last_seen > c.inherit_s:
                continue
            ox1, oy1, ox2, oy2 = old.box
            old_long = max(ox2 - ox1, oy2 - oy1)
            # новая рамка должна быть похожа по размеру: человек при падении не уменьшается вдвое
            size_ratio = max(box[2] - box[0], box[3] - box[1]) / max(1.0, old_long)
            if not c.inherit_size_min <= size_ratio <= c.inherit_size_max:
                continue
            d = math.hypot(cx - (ox1 + ox2) / 2, cy - (oy1 + oy2) / 2)
            if d <= c.inherit_dist * old_long and (best_d is None or d < best_d):
                best, best_d = old, d
        if best is None:
            return None
        del self.tracks[best.tid]
        best.earlier_ids.append(best.tid)
        best.tid = tid
        if best.event is not None:
            best.event["track_id"] = tid
            if "earlier_ids" in best.event:
                best.event["earlier_ids"] = " ".join(map(str, best.earlier_ids))
        return best
