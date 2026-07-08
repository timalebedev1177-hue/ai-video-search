import whisper
import json
import sys
import os
from pathlib import Path
import tqdm
import cv2
from datetime import datetime, timedelta

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v"
)

os.makedirs("transcriptions", exist_ok=True)

print("Загрузка Whisper...")
model = whisper.load_model("base")


def transcribe_video(video_path):
    print(f"Processing: {Path(video_path).name}")


    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    seconds = int(frames / fps)

    processing_time = seconds // 4

    print(f"The approximate processing time is {processing_time // 60} min")

    finish_time = datetime.now() + timedelta(seconds=processing_time)

    print(f"Estimated finish: {finish_time.strftime('%H:%M:%S')}")

    result = model.transcribe(
        video_path,
        verbose=False,
        language="ru",
    )

    segments = []

    for seg in result["segments"]:
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip()
        })

    name = Path(video_path).stem

    out_path = os.path.join(
        "transcriptions",
        f"{name}.json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            segments,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved: {out_path}")



class NoTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def __iter__(self):
        if self.iterable is None:
            return iter([])
        return iter(self.iterable)

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def refresh(self):
        pass

tqdm.tqdm = NoTqdm

if len(sys.argv) < 2:
    print("Using:")
    print("python src/_transcribe.py videos")
    print("or")
    print("python src/_transcribe.py videos/video.mp4")
    sys.exit()

target = sys.argv[1]

if os.path.isdir(target):

    files = []

    for file in os.listdir(target):
        if file.lower().endswith(VIDEO_EXTENSIONS):
            files.append(os.path.join(target, file))

    if not files:
        print("Videos not found.")
        sys.exit()

    print(f"Video found: {len(files)}")

    total = len(files)

    for i, video in enumerate(files, 1):
        print(f"VIDEO_PROGRESSВидео {i}/{total}")
        transcribe_video(video)
        print(f"Ready {i}/{total}")

else:

    if not os.path.exists(target):
        print("File not found:", target)
        sys.exit()

    transcribe_video(target)

print("\nReady.")