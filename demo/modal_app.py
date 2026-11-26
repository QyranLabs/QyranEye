"""QyranEye online demo on Modal (cloud CPU, free monthly credits). The same Gradio app as demo/app.py.

    .venv/bin/modal serve demo/modal_app.py     # temporary test URL (stops with Ctrl+C)
    .venv/bin/modal deploy demo/modal_app.py    # permanent URL https://<workspace>--qyraneye-ui.modal.run

The container runs exactly the tested pipeline (video_pipeline.py + fall_logic.py + config.yaml) on CPU.
One container at a time (Gradio keeps its queue in memory); it sleeps after 10 min without visitors
and wakes up on the next visit (~30-60 s).
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("ultralytics==8.4.172", "opencv-python-headless", "numpy", "PyYAML", "lap",
                 "gradio==6.32.0", "fastapi")
    .add_local_file(DEMO / "app.py", "/app/app.py")
    .add_local_file(ROOT / "fall_logic.py", "/app/fall_logic.py")
    .add_local_file(ROOT / "video_pipeline.py", "/app/video_pipeline.py")
    .add_local_file(ROOT / "config.yaml", "/app/config.yaml")
    .add_local_file(ROOT / "yolo26s.pt", "/app/yolo26s.pt")
    .add_local_file(ROOT / "yolo26n-pose.pt", "/app/yolo26n-pose.pt")
    .add_local_dir(DEMO / "examples", "/app/examples")
)

app = modal.App("qyraneye", image=image)


@app.function(cpu=2.0, memory=4096, timeout=1800, max_containers=1, scaledown_window=600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import sys
    sys.path.insert(0, "/app")
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    import app as qyraneye            # /app/app.py: builds `demo`, loads the models once

    return mount_gradio_app(FastAPI(), qyraneye.demo.queue(max_size=8), path="/",
                            max_file_size="300mb", theme=gr.themes.Soft())
