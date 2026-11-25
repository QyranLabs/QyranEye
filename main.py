"""
QyranEye — детекция падений людей на видео с дрона.

Запуск:
    python main.py video.mp4                    один видеофайл -> results/
    python main.py videos/                      все видео в папке по очереди -> results/
    python main.py video.mp4 --config my.yaml   другой файл настроек
    python main.py video.mp4 --view oblique     камера под углом (nadir — строго вниз). Надёжнее, чем авто

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
    parser.add_argument("--view", choices=["oblique", "nadir"],
                        help="ракурс камеры: oblique — под углом, nadir — строго вниз (если не указан — camera_view из настроек)")
    args = parser.parse_args()
    process_source(args.source, args.config, args.view)


if __name__ == "__main__":
    main()
