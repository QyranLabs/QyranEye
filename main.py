"""
QyranEye — детекция падений людей на видео с дрона.

Запуск (ракурс камеры указывается всегда: от него зависит, как система понимает позу):
    python main.py video.mp4 --view oblique     камера под углом (45–60°) -> results/
    python main.py videos/ --view nadir         камера строго вниз; все видео в папке по очереди
    python main.py video.mp4 --view auto        угадать по людям в начале видео (ненадёжно при 1–2 людях)
    python main.py video.mp4 --view oblique --config my.yaml   другой файл настроек

Обработка — video_pipeline.py + fall_logic.py, все пороги — в config.yaml.
"""
import argparse
from pathlib import Path

from video_pipeline import process_source


def main():
    parser = argparse.ArgumentParser(description="QyranEye: детекция падений на видео с дрона")
    parser.add_argument("source", help="видеофайл или папка с видео")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"),
                        help="файл настроек (по умолчанию config.yaml рядом с main.py)")
    parser.add_argument("--view", choices=["oblique", "nadir", "auto"], required=True,
                        help="ракурс камеры: oblique — под углом, nadir — строго вниз, auto — угадать "
                             "(ошибка в ракурсе даёт ложные тревоги и пропуск падений, поэтому лучше указать)")
    args = parser.parse_args()
    process_source(args.source, args.config, args.view)


if __name__ == "__main__":
    main()
