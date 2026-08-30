import os
import sys
import json
import subprocess
from pathlib import Path


# ==========================================
# UTF-8 ДЛЯ WINDOWS / EXE
# ==========================================

def setup_utf8():
    """
    Принудительно переводит stdout/stderr в UTF-8,
    чтобы символы вроде ✂ 🎥 ❌ не вызывали
    UnicodeEncodeError на Windows.
    """

    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(
                encoding="utf-8",
                errors="replace"
            )

        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(
                encoding="utf-8",
                errors="replace"
            )

    except Exception:
        pass


setup_utf8()


# ==========================================
# ПУТИ
# ==========================================

if getattr(sys, "frozen", False):
    # Если этот файл когда-нибудь будет запускаться
    # непосредственно из EXE.
    ROOT = Path(sys.executable).resolve().parent
else:
    # cut_clip.py находится:
    #
    # ROOT/src/cut_clip.py
    #
    ROOT = Path(__file__).resolve().parent.parent


VIDEOS_DIR = ROOT / "videos"
TRANSCRIPTIONS_DIR = ROOT / "transcriptions"
CLIPS_DIR = ROOT / "clips"
FFMPEG_EXE = ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"


# ==========================================
# НАСТРОЙКИ
# ==========================================

CUT_PAD_SECONDS = 2.0

MIN_CLIP_DURATION = 0.2

DEFAULT_CUT_MODE = "copy"


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def get_ffmpeg():
    """
    Возвращает путь к FFmpeg.

    Приоритет:

    1. ROOT/ffmpeg/bin/ffmpeg.exe
    2. ffmpeg из PATH
    """

    if FFMPEG_EXE.exists():
        return str(FFMPEG_EXE)

    return "ffmpeg"


def safe_print(*args, **kwargs):
    """
    Безопасный print для Windows.

    Даже если консоль использует cp1251,
    UTF-8 символы не должны ломать программу.
    """

    kwargs.setdefault("flush", True)

    try:
        print(*args, **kwargs)

    except UnicodeEncodeError:
        try:
            text = " ".join(
                str(arg)
                for arg in args
            )

            text = text.encode(
                "utf-8",
                errors="replace"
            ).decode(
                "utf-8",
                errors="replace"
            )

            print(
                text,
                **kwargs
            )

        except Exception:
            pass


# ==========================================
# ТРАНСКРИПЦИЯ
# ==========================================

def load_transcription(video_name):
    """
    Загружает Whisper-транскрипцию видео.

    Ожидается:

        transcriptions/<имя_видео>.json
    """

    stem = Path(video_name).stem

    path = (
        TRANSCRIPTIONS_DIR
        / f"{stem}.json"
    )

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if not isinstance(data, list):
            return None

        return data

    except Exception as e:

        safe_print(
            f"⚠ Ошибка загрузки транскрипции: {e}"
        )

        return None


# ==========================================
# ВЫРАВНИВАНИЕ ПО ТРАНСКРИПЦИИ
# ==========================================

def refine_clip_bounds_with_transcript(
    video_name,
    start,
    end,
    pad=CUT_PAD_SECONDS
):

    start = float(start)
    end = float(end)

    if end <= start:

        return (
            start,
            start + MIN_CLIP_DURATION
        )

    segments = load_transcription(
        video_name
    )

    # --------------------------------------
    # НЕТ ТРАНСКРИПЦИИ
    # --------------------------------------

    if not segments:

        new_start = max(
            0.0,
            start - pad
        )

        new_end = max(
            new_start + MIN_CLIP_DURATION,
            end + pad
        )

        return (
            new_start,
            new_end
        )

    # --------------------------------------
    # ПЕРЕСЕКАЮЩИЕСЯ СЕГМЕНТЫ
    # --------------------------------------

    overlapping = []

    for segment in segments:

        try:

            s_start = float(
                segment["start"]
            )

            s_end = float(
                segment["end"]
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            continue

        if (
            s_end >= start
            and
            s_start <= end
        ):

            overlapping.append(
                segment
            )

    # --------------------------------------
    # НАШЛИ ПЕРЕСЕЧЕНИЕ
    # --------------------------------------

    if overlapping:

        new_start = min(
            float(s["start"])
            for s in overlapping
        )

        new_end = max(
            float(s["end"])
            for s in overlapping
        )

    # --------------------------------------
    # НЕ НАШЛИ
    # --------------------------------------

    else:

        nearest = None

        nearest_distance = float(
            "inf"
        )

        for segment in segments:

            try:

                s_start = float(
                    segment["start"]
                )

                s_end = float(
                    segment["end"]
                )

            except (
                KeyError,
                TypeError,
                ValueError
            ):

                continue

            if s_end < start:

                distance = (
                    start - s_end
                )

            elif s_start > end:

                distance = (
                    s_start - end
                )

            else:

                distance = 0.0

            if distance < nearest_distance:

                nearest_distance = distance
                nearest = segment

        if nearest is not None:

            new_start = float(
                nearest["start"]
            )

            new_end = float(
                nearest["end"]
            )

        else:

            new_start = start
            new_end = end

    # --------------------------------------
    # PAD
    # --------------------------------------

    new_start = max(
        0.0,
        new_start - pad
    )

    new_end = max(
        new_start + MIN_CLIP_DURATION,
        new_end + pad
    )

    return (
        new_start,
        new_end
    )


# ==========================================
# AI УТОЧНЕНИЕ
# ==========================================

def refine_clip_with_llm(
    query,
    result,
    llm_pipeline=None,
    llm_generate=None,
    extract_json_value=None
):

    original_start = float(
        result["start"]
    )

    original_end = float(
        result["end"]
    )

    if (
        not llm_pipeline
        or not llm_generate
        or not extract_json_value
    ):

        return (
            original_start,
            original_end
        )

    text = str(
        result.get(
            "text",
            ""
        )
    ).strip()

    if not text:

        return (
            original_start,
            original_end
        )

    prompt = f"""
Запрос:
"{query}"

Текст найденного фрагмента:
"{text}"

Определи минимальный непрерывный фрагмент текста,
который непосредственно отвечает на запрос.

Правила:

- не добавляй информацию;
- не меняй смысл;
- убери лишний контекст;
- оставь ответ максимально коротким;
- если сомневаешься, НЕ сокращай фрагмент;
- start_ratio и end_ratio должны быть от 0 до 1;
- end_ratio должен быть больше start_ratio.

Верни ТОЛЬКО JSON:

{{
  "start_ratio": 0.0,
  "end_ratio": 1.0
}}
"""

    try:

        output = llm_generate(
            prompt,
            max_new_tokens=120
        )

        data = extract_json_value(
            output
        )

    except Exception as e:

        safe_print(
            f"⚠ AI refine ошибка: {e}"
        )

        return (
            original_start,
            original_end
        )

    if not isinstance(
        data,
        dict
    ):

        return (
            original_start,
            original_end
        )

    try:

        start_ratio = float(
            data.get(
                "start_ratio",
                0.0
            )
        )

        end_ratio = float(
            data.get(
                "end_ratio",
                1.0
            )
        )

        start_ratio = max(
            0.0,
            min(
                1.0,
                start_ratio
            )
        )

        end_ratio = max(
            start_ratio,
            min(
                1.0,
                end_ratio
            )
        )

        duration = (
            original_end
            - original_start
        )

        new_start = (
            original_start
            + duration * start_ratio
        )

        new_end = (
            original_start
            + duration * end_ratio
        )

        if new_end <= new_start:

            return (
                original_start,
                original_end
            )

        return (
            new_start,
            new_end
        )

    except (
        TypeError,
        ValueError,
        KeyError
    ):

        return (
            original_start,
            original_end
        )


# ==========================================
# БЕЗОПАСНОЕ ИМЯ
# ==========================================

def make_safe_time(value):

    value = float(value)

    return (
        f"{value:.2f}"
        .replace(
            ".",
            "_"
        )
        .replace(
            "-",
            "m"
        )
    )


# ==========================================
# ВЫРЕЗКА КЛИПА
# ==========================================

def cut_clip(
    sel,
    query=None,
    use_llm_refine=False,
    llm_pipeline=None,
    llm_generate=None,
    extract_json_value=None,
    cut_mode=DEFAULT_CUT_MODE
):

    if not sel:

        safe_print(
            "❌ Не передан результат."
        )

        return None

    # ======================================
    # ИСХОДНЫЕ ДАННЫЕ
    # ======================================

    try:

        start = float(
            sel["start"]
        )

        end = float(
            sel["end"]
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ):

        safe_print(
            "❌ Некорректные границы клипа."
        )

        return None

    video_name = str(
        sel.get(
            "video",
            ""
        )
    ).strip()

    if not video_name:

        safe_print(
            "❌ Не указано имя видео."
        )

        return None

    # ======================================
    # ЗАЩИТА ОТ ПУТЕЙ ВНУТРИ VIDEO
    # ======================================

    video_name = Path(
        video_name
    ).name

    # ======================================
    # ВИДЕО
    # ======================================

    video_path = (
        VIDEOS_DIR
        / video_name
    )

    if not video_path.exists():

        safe_print(
            f"❌ Видео не найдено: {video_path}"
        )

        safe_print(
            f"📁 Папка videos: {VIDEOS_DIR}"
        )

        return None

    # ======================================
    # AI REFINE
    # ======================================

    if (
        use_llm_refine
        and query
    ):

        refined_start, refined_end = (
            refine_clip_with_llm(
                query=query,
                result={
                    **sel,
                    "start": start,
                    "end": end
                },
                llm_pipeline=llm_pipeline,
                llm_generate=llm_generate,
                extract_json_value=extract_json_value
            )
        )

        start = refined_start
        end = refined_end

    # ======================================
    # WHISPER ALIGNMENT
    # ======================================

    start, end = (
        refine_clip_bounds_with_transcript(
            video_name,
            start,
            end
        )
    )

    # ======================================
    # ПРОВЕРКА
    # ======================================

    if end <= start:

        safe_print(
            "❌ Некорректные границы клипа."
        )

        return None

    duration = end - start

    if duration < MIN_CLIP_DURATION:

        safe_print(
            "❌ Клип слишком короткий."
        )

        return None

    # ======================================
    # CLIPS
    # ======================================

    CLIPS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_start = make_safe_time(
        start
    )

    safe_end = make_safe_time(
        end
    )

    output_path = (
        CLIPS_DIR
        / (
            f"{Path(video_name).stem}"
            f"_{safe_start}"
            f"_{safe_end}"
            f".mp4"
        )
    )

    # ======================================
    # FFmpeg
    # ======================================

    ffmpeg = get_ffmpeg()

    if cut_mode == "reencode":

        cmd = [

            ffmpeg,

            "-y",

            "-ss",
            str(start),

            "-i",
            str(video_path),

            "-t",
            str(duration),

            "-c:v",
            "libx264",

            "-preset",
            "veryfast",

            "-crf",
            "18",

            "-c:a",
            "aac",

            "-movflags",
            "+faststart",

            str(output_path)
        ]

    else:

        cmd = [

            ffmpeg,

            "-y",

            "-ss",
            str(start),

            "-i",
            str(video_path),

            "-t",
            str(duration),

            "-c",
            "copy",

            str(output_path)
        ]

    safe_print(
        f"✂ Вырезка: "
        f"{start:.2f} - {end:.2f}"
    )

    safe_print(
        f"🎥 Видео: {video_name}"
    )

    safe_print(
        f"📤 Клип: {output_path}"
    )

    # ======================================
    # ЗАПУСК FFMPEG
    # ======================================

    try:

        env = os.environ.copy()

        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(

            cmd,

            check=False,

            stdout=subprocess.DEVNULL,

            stderr=subprocess.PIPE,

            text=True,

            encoding="utf-8",

            errors="replace",

            env=env
        )

    except FileNotFoundError:

        safe_print(
            "❌ FFmpeg не найден."
        )

        safe_print(
            f"Проверен путь: {FFMPEG_EXE}"
        )

        safe_print(
            "Также выполнена попытка запуска FFmpeg из PATH."
        )

        return None

    except Exception as e:

        safe_print(
            f"❌ Ошибка запуска FFmpeg: {e}"
        )

        return None

    # ======================================
    # ОШИБКА FFmpeg
    # ======================================

    if result.returncode != 0:

        safe_print(
            "❌ FFmpeg ошибка:"
        )

        if result.stderr:

            safe_print(
                result.stderr
            )

        return None

    # ======================================
    # ПРОВЕРКА ФАЙЛА
    # ======================================

    if not output_path.exists():

        safe_print(
            "❌ FFmpeg завершился, "
            "но файл клипа не создан."
        )

        return None

    try:

        if output_path.stat().st_size <= 0:

            safe_print(
                "❌ Создан пустой файл клипа."
            )

            return None

    except OSError:

        return None

    # ======================================
    # УСПЕХ
    # ======================================

    safe_print(
        f"Фрагмент сохранён: "
        f"{output_path}"
    )

    # GUI ищет именно эту строку.
    safe_print(
        f"CLIP_PATH: {output_path}"
    )

    return str(output_path)


# ==========================================
# COMMAND LINE
# ==========================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="VideoSearch AI — вырезка клипа"
    )

    parser.add_argument(
        "query",
        nargs="*",
        default=[]
    )

    parser.add_argument(
        "--video",
        required=True
    )

    parser.add_argument(
        "--start",
        required=True,
        type=float
    )

    parser.add_argument(
        "--end",
        required=True,
        type=float
    )

    parser.add_argument(
        "--text",
        default=""
    )

    parser.add_argument(
        "--llm-refine",
        action="store_true"
    )

    parser.add_argument(
        "--no-llm-refine",
        action="store_true"
    )

    parser.add_argument(
        "--cut-mode",
        choices=[
            "copy",
            "reencode"
        ],
        default=DEFAULT_CUT_MODE
    )

    args = parser.parse_args()

    # ======================================
    # QUERY
    # ======================================

    query = " ".join(
        args.query
    ).strip()

    # ======================================
    # RESULT
    # ======================================

    result = {

        "video": args.video,

        "start": args.start,

        "end": args.end,

        "text": args.text
    }

    # ======================================
    # LLM
    # ======================================

    use_llm_refine = (
        args.llm_refine
        and
        not args.no_llm_refine
    )

    # ======================================
    # LLM В ОТДЕЛЬНОМ ПРОЦЕССЕ
    # ======================================

    if use_llm_refine:

        safe_print(
            "⚠ AI refine запрошен, "
            "но LLM не загружена в отдельном "
            "процессе cut_clip.py."
        )

        safe_print(
            "ℹ Используется выравнивание "
            "по транскрипции."
        )

    # ======================================
    # CUT
    # ======================================

    output = cut_clip(

        result,

        query=query,

        use_llm_refine=False,

        llm_pipeline=None,

        llm_generate=None,

        extract_json_value=None,

        cut_mode=args.cut_mode
    )

    if not output:

        safe_print(
            "❌ Клип не создан."
        )

        return 1

    return 0


# ==========================================
# START
# ==========================================

if __name__ == "__main__":

    sys.exit(
        main()
    )
