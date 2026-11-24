"""
Обучение нейросети позы (lying / sitting / upright) на картинках из make_pose_crops.py.

Берём готовую маленькую модель-классификатор YOLO26n-cls (обучена на ImageNet)
и дообучаем её на картинках людей с дронов. Обучение на видеокарте Mac (mps).

Аугментации (искажения картинок при обучении, чтобы сеть не заучивала конкретные кадры):
  - отражение слева-направо — можно: человек остаётся в той же позе;
  - отражение сверху-вниз и повороты на 90° — НЕЛЬЗЯ: при съёмке под углом
    повёрнутый стоящий человек выглядит лежащим, и сеть научилась бы ошибке;
  - случайная обрезка/масштаб, цвет, яркость — можно.

Запуск:
    python tools/train_pose.py okutama_train/pose_crops
Результат: okutama_train/pose_runs/<name>/weights/best.pt (лучшая по проверке эпоха)
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="папка с train/ и val/")
    ap.add_argument("--name", default="pose_v1")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=96)
    a = ap.parse_args()

    model = YOLO("yolo26n-cls.pt")
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=128, device="mps",
        project="okutama_train/pose_runs", name=a.name, exist_ok=True,
        patience=a.patience,         # остановиться, если столько эпох подряд нет улучшения на проверке
        dropout=a.dropout,           # случайное «выключение» нейронов — против заучивания
        fliplr=0.5, flipud=0.0,      # только отражение слева-направо
        scale=0.4,                   # случайный масштаб (люди бывают разного размера)
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        workers=2, seed=0, deterministic=True, plots=True, verbose=False,
    )


if __name__ == "__main__":
    main()
