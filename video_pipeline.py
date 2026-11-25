"""
Обработка записанных видео: файл или папка -> results/<имя>_fall.mp4 + results/<имя>_events.csv

Схема для каждого кадра:
  1. читаем кадр; если он больше max_height (например 4K) — уменьшаем;
  2. YOLO pose находит людей (рамка + 17 точек тела) — на всём кадре в imgsz
     или по кускам (tiles), на видеокарте Apple (device=mps);
  3. ByteTrack даёт каждому человеку постоянный номер (ID);
  3а. под углом, у людей от min_keypoint_px: точки тела на увеличенной вырезке (crop_keypoints);
  4. FallMonitor (fall_logic.py) решает: стоит / упал / лежит / тревога;
  5. рисуем рамки и пишем кадр в выходное видео, тревоги — в CSV.

Время берётся из видео: номер_кадра / fps. Поэтому «5 секунд» — это 5 секунд
видео, сколько бы ни длилась обработка.
"""

import copy
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.gmc import GMC
from ultralytics.trackers.utils.stracks import multi_gmc

from fall_logic import (FALLEN, FALLING, LYING, UNKNOWN, UPRIGHT, FallMonitor,
                        L_HIP, L_SHOULDER, OBLIQUE, R_HIP, R_SHOULDER)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Цвета рамок (OpenCV использует порядок B, G, R)
COLORS = {UNKNOWN: (170, 170, 170), UPRIGHT: (0, 200, 0), LYING: (255, 160, 0),
          FALLING: (0, 220, 255), FALLEN: (0, 0, 255)}


# ---------------------------------------------------------------- config
def read_config(path):
    """Настройки из yaml как словарь. `base: другой.yaml` = взять всё оттуда, а в этом файле
    задать только отличия (вложенные разделы, например tiles, дополняются, а не заменяются)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("base", None)
    if base is None:
        return data
    merged = read_config(Path(path).parent / base)
    for k, v in data.items():
        merged[k] = {**merged[k], **v} if isinstance(v, dict) and isinstance(merged.get(k), dict) else v
    return merged


def load_config(path):
    path = Path(path).resolve()
    data = read_config(path)
    data["tiles"] = SimpleNamespace(**data["tiles"])
    cfg = SimpleNamespace(**data)
    cfg.base_dir = path.parent          # относительные пути считаем от папки конфига
    return cfg


def resolve(cfg, p):
    p = Path(p)
    return p if p.is_absolute() else cfg.base_dir / p


# ---------------------------------------------------------------- detection
class Detections:
    """Детекции одного кадра в numpy. Трекеру ByteTrack нужны поля conf, cls, xywh
    и выборка по маске det[mask] — этот класс ровно это и даёт."""

    def __init__(self, xyxy, conf, kpts):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.kpts = kpts                                  # (N, 17, 3) или None
        self.cls = np.zeros(len(self.conf), dtype=np.float32)

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Detections(self.xyxy[mask], self.conf[mask],
                          None if self.kpts is None else self.kpts[mask])


def _from_result(r, dx=0, dy=0):
    """Результат YOLO -> numpy, со сдвигом (dx, dy) для кусков кадра."""
    xyxy = r.boxes.xyxy.cpu().numpy() + [dx, dy, dx, dy]
    conf = r.boxes.conf.cpu().numpy()
    kpts = None
    if r.keypoints is not None:
        kpts = r.keypoints.data.cpu().numpy().copy()      # (N, 17, 3): x, y, уверенность
        kpts[..., 0] += dx
        kpts[..., 1] += dy
    return xyxy, conf, kpts


def tile_rects(w, h, rows, cols, overlap):
    """Прямоугольники кусков кадра с перекрытием."""
    tw = int(w / (cols - (cols - 1) * overlap))
    th = int(h / (rows - (rows - 1) * overlap))
    rects = []
    for r in range(rows):
        for c in range(cols):
            x = min(int(c * tw * (1 - overlap)), w - tw)
            y = min(int(r * th * (1 - overlap)), h - th)
            rects.append((x, y, x + tw, y + th))
    return rects


def merge_duplicates(xyxy, conf, ios_thresh):
    """Один человек на стыке кусков найден дважды -> оставляем самую уверенную рамку.
    Мера — пересечение / площадь меньшей рамки (на стыке одна рамка часто
    «обрезана» и лежит внутри другой)."""
    order = np.argsort(-conf)
    keep = []
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for i in order:
        dup = False
        for j in keep:
            iw = min(xyxy[i, 2], xyxy[j, 2]) - max(xyxy[i, 0], xyxy[j, 0])
            ih = min(xyxy[i, 3], xyxy[j, 3]) - max(xyxy[i, 1], xyxy[j, 1])
            if iw > 0 and ih > 0 and iw * ih / max(1e-6, min(area[i], area[j])) > ios_thresh:
                dup = True
                break
        if not dup:
            keep.append(i)
    return np.array(keep, dtype=int)


def detect(model, frame, cfg, low_conf):
    """Найти людей на кадре (весь кадр или по кускам)."""
    args = dict(conf=low_conf, device=cfg.device, classes=[0], verbose=False)
    if not cfg.tiles.enabled:
        xyxy, conf, kpts = _from_result(model.predict(frame, imgsz=cfg.imgsz, **args)[0])
    else:
        h, w = frame.shape[:2]
        rects = tile_rects(w, h, cfg.tiles.rows, cfg.tiles.cols, cfg.tiles.overlap)
        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in rects]
        results = model.predict(crops, imgsz=cfg.tiles.imgsz, **args)
        parts = [_from_result(r, x1, y1) for r, (x1, y1, _, _) in zip(results, rects)]
        xyxy = np.concatenate([p[0] for p in parts])
        conf = np.concatenate([p[1] for p in parts])
        kpts = None if parts[0][2] is None else np.concatenate([p[2] for p in parts])
        keep = merge_duplicates(xyxy, conf, cfg.tiles.merge_ios)
        xyxy, conf = xyxy[keep], conf[keep]
        kpts = None if kpts is None else kpts[keep]

    det = Detections(xyxy, conf, kpts)
    # слишком мелкие рамки — шум
    size = np.maximum(det.xyxy[:, 2] - det.xyxy[:, 0], det.xyxy[:, 3] - det.xyxy[:, 1])
    return det[size >= cfg.min_person_px]


class CameraAwareBotSort(BOTSORT):
    """BoT-SORT: перед сравнением рамок переносит прежние рамки на сдвиг камеры
    (оптический поток по фону). Сдвиг запоминается в last_warp, чтобы логика
    падения использовала тот же самый, а не считала его второй раз."""
    last_warp = None

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high):
        self.last_warp = None
        if self.gmc.method is not None and img is not None:
            try:
                warp = self.gmc.apply(img, results_high.xyxy)
            except Exception:
                warp = np.eye(2, 3)
            multi_gmc(strack_pool, warp)
            multi_gmc(unconfirmed, warp)
            self.last_warp = warp


def make_tracker(cfg, analysis_fps):
    """Трекер связывает рамки соседних кадров по положению и перекрытию.
    botsort — с поправкой на движение дрона (по умолчанию), bytetrack — без неё."""
    args = SimpleNamespace(
        tracker_type=cfg.tracker,
        track_high_thresh=cfg.conf,          # уверенные детекции — первый проход
        track_low_thresh=0.1,                # слабые — второй проход (человек частично скрыт/падает)
        new_track_thresh=cfg.conf,
        track_buffer=max(1, round(cfg.lost_track_s * analysis_fps)),  # в анализируемых кадрах
        match_thresh=0.8,
        fuse_score=True,
    )
    if cfg.tracker == "botsort":
        args.gmc_method = None if str(cfg.camera_motion).lower() == "none" else cfg.camera_motion
        args.proximity_thresh, args.appearance_thresh, args.with_reid, args.model = 0.5, 0.8, False, "auto"
        return CameraAwareBotSort(args)
    if cfg.tracker != "bytetrack":
        raise ValueError(f"tracker: ожидается botsort или bytetrack, а не {cfg.tracker!r}")
    return BYTETracker(args)


# ---------------------------------------------------------------- drawing
def fmt_time(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def draw_text(img, text, org, color, scale, thick):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(frame, overlay, t, n_alarms, cfg, view):
    k = frame.shape[0] / 1080                          # масштаб надписей под размер кадра
    th = max(1, round(2 * k))
    for o in overlay:
        x1, y1, x2, y2 = map(int, o["box"])
        color = COLORS[o["state"]]
        fallen = o["state"] == FALLEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, th * (3 if fallen else 1))
        if cfg.draw_keypoints and o["kpts"] is not None:
            kp = o["kpts"]
            for a, b in ((L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP)):
                if kp[a][2] >= cfg.keypoint_conf and kp[b][2] >= cfg.keypoint_conf:
                    cv2.line(frame, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])),
                             (255, 0, 255), th, cv2.LINE_AA)
        label = f"#{o['tid']} {o['state']} {o['cue']}"
        draw_text(frame, label, (x1, max(int(20 * k), y1 - int(8 * k))), color, 0.6 * k, th)
        if fallen:
            draw_text(frame, "FALL", (x1, y2 + int(40 * k)), (0, 0, 255), 1.4 * k, th + 1)

    header = f"t={fmt_time(t)}  people={len(overlay)}  alarms={n_alarms}  view={view or '?'}"
    draw_text(frame, header, (int(15 * k), int(40 * k)), (255, 255, 255), 0.9 * k, th)
    if any(o["state"] == FALLEN for o in overlay):
        draw_text(frame, "FALL DETECTED", (int(15 * k), int(95 * k)), (0, 0, 255), 1.6 * k, th + 1)


def open_writer(path, fps, size):
    for codec in ("avc1", "mp4v"):                     # H.264, если есть; иначе MPEG-4
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if wr.isOpened():
            return wr
    raise RuntimeError(f"Не удалось создать видеофайл {path}")


# ---------------------------------------------------------------- one video
POSE_CONTEXT = 1.3      # как при обучении (tools/make_pose_crops.py): квадрат = длинная сторона * 1.3


def pose_crop(img, box, out_size):
    """Квадрат вокруг рамки человека с запасом; за краем кадра — повтор крайних пикселей.
    Ровно так же нарезались картинки для обучения нейросети позы."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * POSE_CONTEXT / 2
    a, b, side = int(round(cx - half)), int(round(cy - half)), max(2, int(round(2 * half)))
    pad = side + 2
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = big[b + pad:b + pad + side, a + pad:a + pad + side]
    return cv2.resize(crop, (out_size, out_size),
                      interpolation=cv2.INTER_AREA if side > out_size else cv2.INTER_LINEAR)


def pose_probs(pose_model, frame, boxes, cfg):
    """Ответ нейросети позы для каждой рамки: {класс: вероятность} (одним пакетом на видеокарте).
    Классы: lying, sitting, upright и (у v2) not_person."""
    if not boxes:
        return []
    crops = [pose_crop(frame, b, cfg.pose_imgsz) for b in boxes]
    res = pose_model.predict(crops, imgsz=cfg.pose_imgsz, device=cfg.device, verbose=False)
    return [{pose_model.names[i]: round(float(v), 4) for i, v in enumerate(r.probs.data.tolist())} for r in res]


KP_CONTEXT = 1.4        # вырезка для точек тела: квадрат = длинная сторона рамки * 1.4
_kp_models = {}         # нейросеть точек грузим один раз на все видео


def crop_kp_model(cfg):
    """Нейросеть точек тела для вырезок или None, если они выключены. Проверено только с geometry."""
    if not getattr(cfg, "crop_keypoints", False) or cfg.pose_mode != "geometry":
        return None
    path = str(resolve(cfg, cfg.crop_keypoints_model))
    if path not in _kp_models:
        _kp_models[path] = YOLO(path)
    return _kp_models[path]


def crop_keypoints(kp_model, frame, boxes, cfg):
    """Точки тела (17, 3) в координатах кадра для каждой рамки, или None (мелкий / не нашлись).
    На целом кадре человек для сети слишком мелкий, поэтому вырезаем его и увеличиваем."""
    out, size = [None] * len(boxes), cfg.crop_keypoints_imgsz
    crops, where = [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        long_side = max(x2 - x1, y2 - y1)
        if long_side < cfg.min_keypoint_px:
            continue
        half = long_side * KP_CONTEXT / 2
        a, b, side = int((x1 + x2) / 2 - half), int((y1 + y2) / 2 - half), max(4, int(2 * half))
        crop = np.full((side, side, 3), 114, np.uint8)         # за краем кадра — серый
        xa, ya, xb, yb = max(0, a), max(0, b), min(frame.shape[1], a + side), min(frame.shape[0], b + side)
        if xb <= xa or yb <= ya:
            continue
        crop[ya - b:yb - b, xa - a:xb - a] = frame[ya:yb, xa:xb]
        crops.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC if side < size else cv2.INTER_AREA))
        where.append((i, a, b, side / size))
    if not crops:
        return out
    for (i, a, b, k), r in zip(where, kp_model.predict(crops, imgsz=size, conf=0.1, device=cfg.device, verbose=False)):
        if len(r.boxes):
            kp = r.keypoints.data[int(r.boxes.conf.argmax())].cpu().numpy().copy()   # самый уверенный человек
            kp[:, 0], kp[:, 1] = kp[:, 0] * k + a, kp[:, 1] * k + b
            out[i] = kp
    return out


def find_people(model, tracker, frame, cfg, low_conf, pose_model=None, kp_model=None):
    """Детекция + трекинг одного кадра -> список (track_id, рамка, точки тела|None[, ответ сети позы])."""
    det = detect(model, frame, cfg, low_conf)
    people = []
    for row in tracker.update(det, frame):           # строки: x1 y1 x2 y2 id score cls idx
        tid, di = int(row[4]), int(row[7])
        box = tuple(float(v) for v in det.xyxy[di])  # рамка из детектора (не сглаженная)
        people.append((tid, box, None if det.kpts is None else det.kpts[di]))
    if kp_model is not None:
        kps = crop_keypoints(kp_model, frame, [p[1] for p in people], cfg)
        people = [(tid, box, kp) for (tid, box, _), kp in zip(people, kps)]
    if pose_model is not None:
        probs = pose_probs(pose_model, frame, [p[1] for p in people], cfg)
        people = [(*p, pr) for p, pr in zip(people, probs)]
    return people


def prescan_view(path, cfg, model, scale, size, stride, fps, low_conf):
    """Видео записанное: сначала просматриваем начало только чтобы определить ракурс,
    потом обработка пойдёт с первого кадра уже с известным ракурсом."""
    cap = cv2.VideoCapture(str(path))
    tracker = make_tracker(cfg, fps / stride)
    mon = FallMonitor(cfg, fps / stride)
    idx = 0
    while mon.view is None:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            if scale < 1:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            mon.update(idx / fps, find_people(model, tracker, frame, cfg, low_conf))
        idx += 1
    cap.release()
    if mon.view is None:
        return cfg.camera_view_default, "auto: видео кончилось раньше, взят camera_view_default"
    return mon.view, f"{mon.view_reason}; предварительный просмотр {idx / fps:.1f} с"


def process_video(path, cfg, model, out_dir, pose_model=None):
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ОШИБКА: не удалось открыть {path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Пункт 4: больше 1080p -> уменьшаем (по короткой стороне, чтобы работало и для вертикального видео)
    scale = min(1.0, cfg.max_height / min(w0, h0))
    size = (round(w0 * scale) // 2 * 2, round(h0 * scale) // 2 * 2)   # чётные размеры для кодека
    print(f"  {w0}x{h0} @ {fps:.1f} fps, {total} кадров"
          + (f" -> обработка в {size[0]}x{size[1]}" if scale < 1 else ""))

    stride = max(1, int(cfg.frame_stride))
    analysis_fps = fps / stride
    low_conf = min(0.1, cfg.conf)
    tracker = make_tracker(cfg, analysis_fps)
    if cfg.camera_view == "auto" and cfg.view_prescan:
        view, reason = prescan_view(path, cfg, model, scale, size, stride, fps, low_conf)
        vcfg = copy.copy(cfg)
        vcfg.camera_view = view
        monitor = FallMonitor(vcfg, analysis_fps, size)
        monitor.view_reason = reason
        print(f"  Ракурс: {view} ({reason})")
    else:
        monitor = FallMonitor(cfg, analysis_fps, size)
    # поправка на движение дрона: сдвиг кадра по фону между анализируемыми кадрами
    # (BoT-SORT считает сдвиг сам и отдаёт его в tracker.last_warp; для ByteTrack считаем отдельно)
    gmc = None
    if not isinstance(tracker, CameraAwareBotSort) and str(cfg.camera_motion).lower() != "none":
        gmc = GMC(cfg.camera_motion, downscale=2)

    out_video = out_dir / f"{path.stem}_fall.mp4"
    out_csv = out_dir / f"{path.stem}_events.csv"
    writer = open_writer(out_video, fps, size)

    kp_model = crop_kp_model(cfg)
    overlay, idx = [], 0
    t_start = t_print = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            t = idx / fps                                    # время ВИДЕО, а не часов

            if idx % stride == 0:
                people = find_people(model, tracker, frame, cfg, low_conf, pose_model,
                                     kp_model if monitor.view == OBLIQUE else None)   # сверху точки не работают
                if isinstance(tracker, CameraAwareBotSort):
                    warp = tracker.last_warp             # тот же сдвиг камеры, что использовал трекер
                else:
                    warp = None if gmc is None else gmc.apply(frame)
                had_view = monitor.view is not None
                overlay = [dict(tid=tr.tid, state=tr.state, cue=tr.cue, box=tr.box, kpts=tr.kpts)
                           for tr in monitor.update(t, people, warp)]
                if not had_view and monitor.view is not None:
                    print(f"  Ракурс: {monitor.view} ({monitor.view_reason}), t={t:.1f} с")

            draw(frame, overlay, t, len(monitor.events), cfg, monitor.view)
            writer.write(frame)

            if cfg.show_preview:
                cv2.imshow("QyranEye", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("  Остановлено пользователем.")
                    break

            idx += 1
            if time.time() - t_print > 3:
                t_print = time.time()
                pct = f"{100 * idx / total:.0f}%" if total else ""
                print(f"    кадр {idx}/{total} {pct}  скорость {idx / (t_print - t_start):.1f} кадр/с")
    finally:
        cap.release()
        writer.release()
        if cfg.show_preview:
            cv2.destroyAllWindows()
        write_csv(out_csv, path.name, monitor.events, fps, scale, monitor.view)

    took = time.time() - t_start
    print(f"  Готово за {took:.0f} с ({idx / max(took, 1e-6):.1f} кадр/с). Тревог: {len(monitor.events)}, "
          f"отменено проверкой перед тревогой: {monitor.cancelled}")
    print(f"  -> {out_video}\n  -> {out_csv}")
    return monitor.events


def write_csv(path, video_name, events, fps, scale, view):
    """Одна строка на тревогу. Координаты рамки — в пикселях ИСХОДНОГО видео."""
    cols = ["video", "camera_view", "track_id", "earlier_ids", "fall_start_s", "fall_start_time",
            "alarm_s", "alarm_time", "alarm_frame", "transition_s", "fall_speed", "down_share", "speed", "recovered_s",
            "x1", "y1", "x2", "y2"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for e in events:
            x1, y1, x2, y2 = (round(v / scale) for v in e["box"])
            wr.writerow({
                "video": video_name, "camera_view": view, "track_id": e["track_id"],
                "earlier_ids": e["earlier_ids"],
                "fall_start_s": e["fall_start_s"], "fall_start_time": fmt_time(e["fall_start_s"]),
                "alarm_s": e["alarm_s"], "alarm_time": fmt_time(e["alarm_s"]),
                "alarm_frame": round(e["alarm_s"] * fps), "transition_s": e["transition_s"],
                "fall_speed": e.get("fall_speed", ""),
                "down_share": e["down_share"], "speed": e["speed"],
                "recovered_s": e["recovered_s"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })


# ---------------------------------------------------------------- file or folder
def process_source(source, config_path):
    cfg = load_config(config_path)
    src = Path(source)
    if src.is_dir():
        videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not videos:
            raise SystemExit(f"В папке {src} нет видео ({', '.join(sorted(VIDEO_EXTS))})")
    elif src.is_file():
        videos = [src]
    else:
        raise SystemExit(f"Не найдено: {src}")

    out_dir = resolve(cfg, cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(resolve(cfg, cfg.model)))      # модель грузим один раз на все видео
    pose_model = YOLO(str(resolve(cfg, cfg.pose_model))) if cfg.pose_mode in ("classifier", "hybrid") else None

    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v.name}")
        process_video(v, cfg, model, out_dir, pose_model)
