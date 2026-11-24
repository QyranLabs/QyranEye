"""
QyranEye — детекция падений.

Запуск:
    python main.py                 веб-камера 1 (как раньше)
    python main.py 0               веб-камера с номером 0 (как раньше)
    python main.py video.mp4       один видеофайл  -> results/
    python main.py videos/         все видео в папке по очереди -> results/
    python main.py video.mp4 --config my.yaml   другой файл настроек

Видеофайлы обрабатываются конвейером для дрона (video_pipeline.py + fall_logic.py,
настройки в config.yaml). Веб-камера — прежним кодом ниже (MediaPipe).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
 
import cv2
import mediapipe as mp
from ultralytics import YOLO
 
# ---------------- Settings ----------------
YOLO_MODEL = "yolo26n.pt"
POSE_MODEL = "pose_landmarker_heavy.task"
DEFAULT_CAMERA = 1
FRAME_SCALE = 0.5        # process at half resolution for speed
MIRROR = False
MAX_POSES = 5
 
VISIBLE = 0.5            # landmark visibility threshold
LYING_RATIO = 1.15       # body box wider than tall -> lying
SITTING_GAP = 0.15       # hip-to-knee height / body height below this -> sitting
STILL_LEVEL = 0.015      # smoothed movement (fraction of body size) counted as still
STILL_SECONDS = 5.0      # lying and still this long -> alert
FALL_SPEED = 1.5         # hip drop in body-heights per second -> possible fall
ALERT_SECONDS = 5.0      # how long a fall alert stays on screen
 
# Body-only skeleton (no face landmarks)
BODY_LINES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
 
 
# ---------------- Helpers ----------------
def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.8):
    for c, t in (((0, 0, 0), 6), (color, 2)):  # dark outline, then colour
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, c, t, cv2.LINE_AA)
 
 
def visible(pose, *ids):
    return all(pose[i].visibility > VISIBLE for i in ids)
 
 
def pose_box(pose, w, h):
    pts = [(p.x * w, p.y * h) for p in pose if p.visibility > VISIBLE]
    if len(pts) < 10:
        return None
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)
 
 
def classify_posture(pose, box, frame_h):
    """STANDING / SITTING / LYING from simple body geometry."""
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
 
    if bw / bh > LYING_RATIO:
        return "LYING"
 
    if visible(pose, 23, 24, 25, 26):
        # Hip-to-knee height as a fraction of the person's own body height.
        gap = ((pose[25].y - pose[23].y) + (pose[26].y - pose[24].y)) / 2
        if gap * frame_h / bh < SITTING_GAP:
            return "SITTING"
 
    return "STANDING"
 
 
def movement(prev, pose, w, h, size):
    """Average landmark movement since the last frame, as a fraction of body size."""
    d = [math.hypot((a.x - b.x) * w, (a.y - b.y) * h)
         for a, b in zip(pose, prev)
         if a.visibility > VISIBLE and b.visibility > VISIBLE]
    return sum(d) / len(d) / size if d else None
 
 
def match_poses_to_tracks(people, centers):
    """Pair each pose with the YOLO person whose box contains its center."""
    pairs = []
    for pi, c in enumerate(centers):
        if c is None:
            continue
        for tid, (x1, y1, x2, y2) in people:
            if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
                dist = math.hypot(c[0] - (x1 + x2) / 2, c[1] - (y1 + y2) / 2)
                pairs.append((dist, pi, tid))
    pairs.sort()
 
    track_to_pose, used = {}, set()
    for _, pi, tid in pairs:
        if pi not in used and tid not in track_to_pose:
            track_to_pose[tid] = pi
            used.add(pi)
    return track_to_pose
 
 
def new_state():
    return {"prev": None, "move": 1.0, "still_since": None,
            "hip": None, "hip_t": None, "fall_until": 0.0}
 
 
def reset_tracking(s):
    s.update(prev=None, move=1.0, still_since=None, hip=None, hip_t=None)
 
 
# ---------------- Webcam (live) ----------------
def run_webcam(camera_index):
    if not os.path.isfile(POSE_MODEL):
        sys.exit(f"ERROR: {POSE_MODEL} not found next to main.py")

    yolo = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open camera {camera_index}. "
                 "Try another index: python main.py 0")
 
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=MAX_POSES,
        )
    )
 
    states = defaultdict(new_state)
    mirror = MIRROR
    last_ms = -1
    failed_reads = 0
 
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads > 30:
                    print("ERROR: Camera stopped sending frames.")
                    break
                continue
            failed_reads = 0
 
            if mirror:
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            h, w = frame.shape[:2]
            now = time.monotonic()
 
            # --- Detect and track people (class 0 = person) ---
            boxes = yolo.track(frame, persist=True, classes=[0], verbose=False)[0].boxes
            people = []
            if boxes is not None and boxes.id is not None:
                people = list(zip(boxes.id.int().cpu().tolist(),
                                  boxes.xyxy.cpu().numpy()))
 
            # --- Estimate poses ---
            ms = max(int(now * 1000), last_ms + 1)
            last_ms = ms
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            poses = landmarker.detect_for_video(mp_image, ms).pose_landmarks
 
            boxes_p = [pose_box(p, w, h) for p in poses]
            centers = [None if b is None else ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                       for b in boxes_p]
            track_to_pose = match_poses_to_tracks(people, centers)
 
            for pose in poses:
                for a, b in BODY_LINES:
                    if visible(pose, a, b):
                        cv2.line(frame,
                                 (int(pose[a].x * w), int(pose[a].y * h)),
                                 (int(pose[b].x * w), int(pose[b].y * h)),
                                 (255, 0, 0), 3, cv2.LINE_AA)
 
            # --- Analyse each person ---
            for tid, (x1, y1, x2, y2) in people:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_text(frame, f"Person #{tid}", (x1, max(35, y1 - 15)), scale=0.9)
 
                s = states[tid]
                pi = track_to_pose.get(tid)
                if pi is None:
                    reset_tracking(s)
                    continue
 
                pose, box = poses[pi], boxes_p[pi]
                bh = max(1, box[3] - box[1])
                size = max(box[2] - box[0], bh)
 
                posture = classify_posture(pose, box, h)
 
                # Smoothed body movement (1.0 until we have data = "moving")
                if s["prev"] is not None:
                    m = movement(s["prev"], pose, w, h, size)
                    if m is not None:
                        s["move"] = 0.8 * s["move"] + 0.2 * m
                s["prev"] = pose
 
                # Rapid hip drop -> possible fall
                if visible(pose, 23, 24):
                    hip = (pose[23].y + pose[24].y) / 2 * h
                    if s["hip"] is not None and now > s["hip_t"]:
                        drop = (hip - s["hip"]) / (now - s["hip_t"]) / bh
                        if drop > FALL_SPEED:
                            s["fall_until"] = now + ALERT_SECONDS
                    s["hip"], s["hip_t"] = hip, now
                else:
                    s["hip"] = s["hip_t"] = None
 
                # Lying and continuously still
                if posture == "LYING" and s["move"] < STILL_LEVEL:
                    if s["still_since"] is None:
                        s["still_since"] = now
                else:
                    s["still_since"] = None
 
                alert = None
                if now < s["fall_until"]:
                    alert = "POSSIBLE FALL - CHECK"
                if s["still_since"] is not None and now - s["still_since"] >= STILL_SECONDS:
                    alert = "LYING STILL - CHECK"
 
                y = min(h - 40, y2 + 45)
                draw_text(frame, f"Posture: {posture}", (x1, y), (255, 150, 0))
                if alert:
                    draw_text(frame, alert, (max(10, x1), y + 35), (0, 0, 255), 1.0)
 
            cv2.imshow("Emergency Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("m"), ord("M")):
                mirror = not mirror
            elif key in (ord("q"), 27):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


# ---------------- Entry point ----------------
def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="QyranEye: fall detection")
    parser.add_argument("source", nargs="?",
                        help="видеофайл, папка с видео или номер камеры (по умолчанию камера 1)")
    parser.add_argument("--config", default=str(here / "config.yaml"),
                        help="файл настроек для обработки видео")
    args = parser.parse_args()

    # Нет аргумента или число -> веб-камера, как раньше
    if args.source is None or args.source.isdigit():
        run_webcam(int(args.source) if args.source else DEFAULT_CAMERA)
        return

    # Иначе путь к файлу или папке -> конвейер для видео с дрона
    from video_pipeline import process_source
    process_source(args.source, args.config)

 
if __name__ == "__main__":
    main()