"""
Дообучение детектора людей (yolo26s.pt) на съёмке с дрона — набор из tools/make_det_dataset.py.

Зачем: eval/detect_by_size.py показал, что под углом детектор находит 89 % людей, а сверху — 62 %,
и ни нарезка кадра, ни 4K этого не меняют. Обычный YOLO учился на людях, снятых сбоку.

Аугментации (в отличие от нейросети позы): отражение сверху-вниз МОЖНО — детектору нужно найти человека,
а не понять его позу; человек сверху вверх ногами остаётся человеком. Случайный масштаб шире обычного:
кадры обучения 1280x720, а система смотрит 1080p — люди в обучении в 1.5 раза мельче, чем в работе.

Запуск (на Mac с видеокартой Apple; по пробе ~0.5 с на кадр, то есть ~3000 кадров — ~25 мин на эпоху):
    .venv/bin/python tools/train_det.py okutama_train/det_dataset/data.yaml
    .venv/bin/python tools/train_det.py okutama_train/det_dataset/data.yaml --epochs 2 --fraction 0.05   # проба
Результат: runs/detect/<name>/weights/best.pt. Проверка на тестовых роликах (их детектор не видел):
    .venv/bin/python eval/detect_by_size.py --variants base --model runs/detect/<name>/weights/best.pt

ВНИМАНИЕ: рамки разметки Okutama свободнее, чем рамки обычного YOLO (больше человека). Дообученный детектор
научится таким же — а правила позы (форма рамки, w/h) подбирались под рамки обычного YOLO. Поэтому после
обучения обязательно прогнать всю систему (eval/evaluate.py, tests/) и только потом менять model в config.yaml.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="data.yaml из tools/make_det_dataset.py")
    ap.add_argument("--model", default=str(PROJECT / "yolo26s.pt"), help="с какой модели начинаем")
    ap.add_argument("--name", default="det_okutama")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=1280, help="кадры обучения 1280x720 — без сжатия")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="доля обучающих кадров (для пробы)")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    YOLO(a.model).train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        project=str(PROJECT / "runs/detect"), name=a.name, exist_ok=True,
        patience=a.patience,       # остановиться, если столько эпох подряд нет улучшения на проверке
        fraction=a.fraction,
        single_cls=True,           # один класс: человек
        fliplr=0.5, flipud=0.5,    # сверху человек выглядит одинаково в любую сторону
        degrees=0.0,               # поворот рамку не повернёт, она станет шире человека — не нужно
        scale=0.6,                 # люди бывают разного размера (и в обучении мельче, чем в работе)
        mosaic=1.0, close_mosaic=5,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        optimizer="AdamW", lr0=0.0005, warmup_epochs=1,   # дообучение: шаг небольшой, чтобы не забыть, что умела
                                                           # модель (при optimizer=auto lr0 игнорируется)
        workers=2, seed=0, deterministic=True, plots=True, verbose=False,
    )


if __name__ == "__main__":
    main()
