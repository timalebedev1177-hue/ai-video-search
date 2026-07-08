# src/cut_clip.py
# Лёгкая вырезка клипа без перезапуска полного поиска.
# Загружает только LLM (для уточнения границ) и транскрипцию.

import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import subprocess
import argparse
from pathlib import Path

try:
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# подавить tqdm progress bars
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# -----------------------------
# ПАРАМЕТРЫ
# -----------------------------

LLM_MODEL = "models/llm"
CUT_PAD_SECONDS = 1
CUT_MODE = "reencode"

# -----------------------------
# ПАРСИНГ АРГУМЕНТОВ
# -----------------------------

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("query", nargs="*", default=[])
parser.add_argument("--video", required=True)
parser.add_argument("--start", required=True, type=float)
parser.add_argument("--end", required=True, type=float)
parser.add_argument("--no-llm", action="store_true")
parser.add_argument("--llm-refine", action="store_true", default=False)
parser.add_argument("--no-llm-refine", action="store_true")
parser.add_argument("--text", default="", help="Текст фрагмента для LLM уточнения")
args, _unknown = parser.parse_known_args()

query = " ".join(args.query).strip()
USE_LLM = not args.no_llm
USE_LLM_REFINE = args.llm_refine and not args.no_llm_refine

# -----------------------------
# LLM
# -----------------------------

llm_pipeline = None

if USE_LLM and USE_LLM_REFINE and TRANSFORMERS_AVAILABLE and os.path.exists(LLM_MODEL):
    try:
        print("Загрузка LLM для уточнения границ...")

        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            local_files_only=True,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            local_files_only=True,
            trust_remote_code=True
        )

        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

        print("LLM загружена")

    except Exception as e:
        print("Ошибка загрузки LLM:", e)
        llm_pipeline = None
elif not USE_LLM_REFINE:
    print("LLM уточнение границ отключено")
else:
    print("LLM не используется")


# -----------------------------
# ВСПОМОГАТЕЛЬНЫЕ
# -----------------------------

def extract_json_value(text):
    if not text:
        return None

    raw = text.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*?\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


def llm_generate(prompt, max_new_tokens=120):
    if not llm_pipeline:
        return None

    try:
        messages = [{"role": "user", "content": prompt}]

        formatted = llm_pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        out = llm_pipeline(
            formatted,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
            eos_token_id=llm_pipeline.tokenizer.eos_token_id,
            pad_token_id=llm_pipeline.tokenizer.eos_token_id
        )[0]["generated_text"]

        return out

    except Exception as e:
        print("LLM вызов не удался:", e)
        return None


# -----------------------------
# УТОЧНЕНИЕ ГРАНИЦ
# -----------------------------

def load_transcription(video_name):
    stem = Path(video_name).stem
    path = os.path.join("transcriptions", f"{stem}.json")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def refine_clip_bounds_with_transcript(video_name, start, end, pad=CUT_PAD_SECONDS):
    segments = load_transcription(video_name)
    if not segments:
        return max(0.0, start - pad), max(start, end + pad)

    start = float(start)
    end = float(end)

    covering = [s for s in segments if not (float(s["end"]) < start or float(s["start"]) > end)]
    if covering:
        new_start = float(min(s["start"] for s in covering))
        new_end = float(max(s["end"] for s in covering))
    else:
        left = None
        right = None
        for s in segments:
            if float(s["end"]) <= start:
                left = s
            if float(s["start"]) >= end:
                right = s
                break
        new_start = float(left["start"]) if left else start
        new_end = float(right["end"]) if right else end

    new_start = max(0.0, new_start - pad)
    new_end = max(new_start + 0.2, new_end + pad)

    return new_start, new_end


def refine_clip_with_llm(query, result_text, start, end):
    if not llm_pipeline:
        return start, end

    prompt = f"""
Запрос:
"{query}"

Текст видео:
"{result_text}"

Найди минимальный непрерывный фрагмент текста, который точно отвечает запросу.

Правила:
- убери лишний контекст
- оставь только суть
- не добавляй ничего нового

Верни JSON:
{{
  "start_ratio": 0.0,
  "end_ratio": 1.0
}}

Где:
start_ratio — начало (0 = начало текста)
end_ratio — конец (1 = конец текста)
"""

    out = llm_generate(prompt, max_new_tokens=120)
    data = extract_json_value(out)

    if not isinstance(data, dict):
        return start, end

    try:
        sr = float(data.get("start_ratio", 0.0))
        er = float(data.get("end_ratio", 1.0))

        sr = max(0.0, min(1.0, sr))
        er = max(sr, min(1.0, er))

        duration = end - start

        new_start = start + duration * sr
        new_end = start + duration * er

        return new_start, new_end

    except Exception:
        return start, end


# -----------------------------
# ВЫРЕЗКА
# -----------------------------

def cut_clip(query, video_name, start, end, text=""):
    start = float(start)
    end = float(end)
    video_path = os.path.join("videos", video_name)

    if not os.path.exists(video_path):
        print("Видео не найдено:", video_path)
        return

    # LLM уточнение (полный текст — без обрезки)
    if query and text and USE_LLM_REFINE:
        print("LLM уточнение границ...")
        start, end = refine_clip_with_llm(query, text, start, end)

    # подгонка под транскрипцию
    start, end = refine_clip_bounds_with_transcript(video_name, start, end)

    os.makedirs("clips", exist_ok=True)
    safe_start = f"{start:.2f}".replace(".", "_")
    safe_end = f"{end:.2f}".replace(".", "_")
    out = f"clips/result_{safe_start}_{safe_end}.mp4"

    if CUT_MODE == "copy":
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            out
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out
        ]

    print("Вырезка клипа...")
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Фрагмент сохранён:", out)
    print("CLIP_PATH:", out)


# -----------------------------
# СТАРТ
# -----------------------------

if not query:
    print("Пустой запрос.")
    sys.exit(0)

cut_clip(query, args.video, args.start, args.end, args.text)
