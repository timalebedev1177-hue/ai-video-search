# make_segments.py
# Создает перекрывающиеся окна сегментов для поиска.
# Чуть аккуратнее чистит текст и сохраняет только полезные окна.

import json
import os
from pathlib import Path
import pickle
import re

WINDOW = 110
STEP = 35
MIN_WORDS = 5

os.makedirs("indexes", exist_ok=True)


def clean_text(s):
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(ээ|эм|ну|как бы|в общем|типа)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


for file in os.listdir("transcriptions"):
    if not file.endswith(".json"):
        continue

    path = os.path.join("transcriptions", file)
    with open(path, encoding="utf-8") as f:
        segments = json.load(f)

    words = []
    times = []

    for seg in segments:
        text = clean_text(seg["text"])
        if not text:
            continue

        ws = text.split()
        if not ws:
            continue

        for w in ws:
            words.append(w)
            times.append((float(seg["start"]), float(seg["end"])))

    windows = []
    for i in range(0, len(words), STEP):
        chunk = words[i:i + WINDOW]
        if len(chunk) < MIN_WORDS:
            continue

        text = " ".join(chunk)
        start = times[i][0]
        end = times[min(i + WINDOW - 1, len(times) - 1)][1]

        if end - start < 0.5:
            continue

        windows.append({
            "text": text,
            "start": start,
            "end": end
        })

    name = Path(file).stem
    out = f"indexes/{name}.segments"

    with open(out, "wb") as f:
        pickle.dump(windows, f)

    print("Создано сегментов:", name, len(windows))