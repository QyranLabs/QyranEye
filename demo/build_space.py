"""Builds the Hugging Face Space folder for the online demo and (optionally) uploads it.

    .venv/bin/python demo/build_space.py                         # build into eval_results/space/ and check it
    .venv/bin/python demo/build_space.py --upload USER/qyraneye  # build and upload (after `hf auth login`)

The Space is a flat copy: app.py + the project files the pipeline needs + the two YOLO models + examples.
Nothing else from the repository goes there (no datasets, no training data, no other code).
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
OUT = ROOT / "eval_results/space"
FILES = {"app.py": DEMO / "app.py", "fall_logic.py": ROOT / "fall_logic.py",
         "video_pipeline.py": ROOT / "video_pipeline.py", "config.yaml": ROOT / "config.yaml",
         "yolo26s.pt": ROOT / "yolo26s.pt", "yolo26n-pose.pt": ROOT / "yolo26n-pose.pt"}

README = """---
title: QyranEye
emoji: 🚁
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: {gradio}
python_version: "3.12"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Fall detection in drone video for search and rescue
---

# QyranEye — fall detection from drone video (prototype)

Upload a short drone video (up to 30 s) and see who fell and lies still. Student project; see the
"How it works" tab for the method, honest limits and credits. Runs on a free CPU, so a video takes 1–5 minutes.
"""
REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
ultralytics==8.4.172
opencv-python-headless
numpy
PyYAML
lap
"""


def build():
    import gradio
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name, src in FILES.items():
        if not src.exists():
            sys.exit(f"Missing {src}")
        shutil.copy(src, OUT / name)
    ex = DEMO / "examples"
    if not any(ex.glob("*.json")):
        sys.exit("No examples: run demo/make_examples.py first")
    shutil.copytree(ex, OUT / "examples")
    (OUT / "README.md").write_text(README.format(gradio=gradio.__version__), encoding="utf-8")
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (OUT / "packages.txt").write_text("ffmpeg\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"Built {OUT}: {sum(1 for f in OUT.rglob('*') if f.is_file())} files, {size / 1e6:.0f} MB")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT)}  {f.stat().st_size / 1e6:.1f} MB")


def upload(repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()["name"]                       # fails clearly if not logged in
    print("Logged in as", who)
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="QyranEye demo", delete_patterns=["*"])
    print(f"Uploaded: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", metavar="USER/NAME", help="Space to create or update")
    a = ap.parse_args()
    build()
    if a.upload:
        upload(a.upload)
