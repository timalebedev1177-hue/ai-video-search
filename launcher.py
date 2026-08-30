import os
import sys
import subprocess
import urllib.request
import shutil
import zipfile
from pathlib import Path


# ============================================================
# КОРЕНЬ ПРОЕКТА
# ============================================================

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent


# ============================================================
# ПУТИ
# ============================================================

PYTHON_DIR = ROOT / "python"

# ВАЖНО:
# Правильный Python находится непосредственно здесь:
#
# ROOT/
#   python/
#       python.exe
#
PYTHON_EXE = PYTHON_DIR / "python.exe"

REQUIREMENTS = ROOT / "requirements.txt"
APP_FILE = ROOT / "src" / "gui.py"

TEMP = ROOT / "temp"


# ============================================================
# PYTHON
# ============================================================

PYTHON_VERSION = "3.10.11"

PYTHON_URL = (
    "https://www.python.org/ftp/python/3.10.11/"
    "python-3.10.11-embed-amd64.zip"
)


# ============================================================
# ЛОГ
# ============================================================

def log(text=""):
    print(text, flush=True)


# ============================================================
# ЗАГРУЗКА ФАЙЛА
# ============================================================

def download(url, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    log("")
    log("Downloading:")
    log(url)

    urllib.request.urlretrieve(
        url,
        path
    )

    if not path.exists():
        raise Exception(
            f"Файл не был скачан:\n{path}"
        )

    if path.stat().st_size <= 0:
        raise Exception(
            f"Скачан пустой файл:\n{path}"
        )


# ============================================================
# PYTHON
# ============================================================

def get_python():

    # Никаких Scripts/python.exe.
    # Используем только:
    #
    # ROOT/python/python.exe

    return str(PYTHON_EXE)


# ============================================================
# ПРОВЕРКА PYTHON
# ============================================================

def check_python():

    python = get_python()

    log("")
    log("==========================================")
    log("ПРОВЕРКА PYTHON")
    log("==========================================")

    log(
        f"Python: {python}"
    )

    if not PYTHON_EXE.exists():

        raise Exception(
            "Встроенный Python не найден:\n"
            f"{PYTHON_EXE}"
        )

    # --------------------------------------------------------
    # VERSION
    # --------------------------------------------------------

    result = subprocess.run(
        [
            python,
            "--version"
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    version = (
        result.stdout.strip()
        or result.stderr.strip()
    )

    log(
        f"Version: {version}"
    )

    if result.returncode != 0:

        raise Exception(
            "Не удалось запустить встроенный Python."
        )

    # --------------------------------------------------------
    # ПРОВЕРЯЕМ ФАКТИЧЕСКИЙ EXECUTABLE
    # --------------------------------------------------------

    result = subprocess.run(
        [
            python,
            "-c",
            "import sys; print(sys.executable)"
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    actual = result.stdout.strip()

    expected = str(
        PYTHON_EXE.resolve()
    )

    if result.returncode != 0:

        raise Exception(
            "Не удалось определить sys.executable."
        )

    log(
        f"Executable: {actual}"
    )

    if (
        Path(actual).resolve().as_posix().lower()
        !=
        Path(expected).resolve().as_posix().lower()
    ):

        raise Exception(
            "ОШИБКА: используется неправильный Python!\n\n"
            f"Ожидался:\n{expected}\n\n"
            f"Получен:\n{actual}"
        )

    log(
        "✓ Используется правильный Python."
    )


# ============================================================
# УСТАНОВКА PYTHON
# ============================================================

def install_python():

    # --------------------------------------------------------
    # PYTHON УЖЕ ЕСТЬ
    # --------------------------------------------------------

    if PYTHON_EXE.exists():

        log("")
        log(
            "✓ Встроенный Python уже существует."
        )

        log(
            str(PYTHON_EXE)
        )

        return

    # --------------------------------------------------------
    # НЕТ PYTHON
    # --------------------------------------------------------

    log("")
    log("==========================================")
    log("УСТАНОВКА PYTHON")
    log("==========================================")

    TEMP.mkdir(
        parents=True,
        exist_ok=True
    )

    python_zip = (
        TEMP /
        "python-3.10.11-embed-amd64.zip"
    )

    # --------------------------------------------------------
    # СКАЧИВАНИЕ
    # --------------------------------------------------------

    download(
        PYTHON_URL,
        python_zip
    )

    # --------------------------------------------------------
    # РАСПАКОВКА
    # --------------------------------------------------------

    log("")
    log(
        "Распаковка Python..."
    )

    PYTHON_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with zipfile.ZipFile(
        python_zip,
        "r"
    ) as archive:

        archive.extractall(
            PYTHON_DIR
        )

    python_zip.unlink(
        missing_ok=True
    )

    # --------------------------------------------------------
    # ПРОВЕРКА
    # --------------------------------------------------------

    if not PYTHON_EXE.exists():

        raise Exception(
            "Python был распакован, "
            "но python.exe не найден:\n"
            f"{PYTHON_EXE}"
        )

    log(
        "✓ Python установлен."
    )


# ============================================================
# PYTHONPATH / SITE-PACKAGES
# ============================================================

def configure_embedded_python():

    pth_file = (
        PYTHON_DIR /
        "python310._pth"
    )

    if not pth_file.exists():

        log(
            "⚠ python310._pth не найден."
        )

        return

    log("")
    log(
        "Настройка embedded Python..."
    )

    try:

        content = pth_file.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        content = pth_file.read_text(
            encoding="utf-8-sig"
        )

    lines = content.splitlines()

    # --------------------------------------------------------
    # ДОБАВЛЯЕМ SITE-PACKAGES
    # --------------------------------------------------------

    site_packages = "Lib\\site-packages"

    if not any(
        line.strip() == site_packages
        for line in lines
    ):

        # Добавляем перед import site
        insert_index = len(lines)

        for i, line in enumerate(lines):

            if line.strip() == "import site":

                insert_index = i
                break

        lines.insert(
            insert_index,
            site_packages
        )

    # --------------------------------------------------------
    # IMPORT SITE
    # --------------------------------------------------------

    if not any(
        line.strip() == "import site"
        for line in lines
    ):

        lines.append(
            "import site"
        )

    pth_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    log(
        "✓ Embedded Python настроен."
    )





# ============================================================
# ОЧИСТКА LLM generation_config.json
# ============================================================

def clean_llm_generation_config():

    config_file = (
        ROOT
        / "models"
        / "llm"
        / "generation_config.json"
    )

    log("")
    log("==========================================")
    log("ОЧИСТКА LLM CONFIG")
    log("==========================================")

    if not config_file.exists():

        log(
            "⚠ generation_config.json не найден:"
        )

        log(
            str(config_file)
        )

        return

    try:

        # Полностью очищаем файл
        config_file.write_text(
            "{}",
            encoding="utf-8"
        )

        log(
            "✓ generation_config.json очищен."
        )

        log(
            str(config_file)
        )

    except Exception as e:

        raise Exception(
            "Не удалось очистить generation_config.json:\n"
            f"{config_file}\n\n"
            f"{e}"
        )

# ============================================================
# ПРОВЕРКА PIP
# ============================================================

def check_pip():

    python = get_python()

    result = subprocess.run(
        [
            python,
            "-m",
            "pip",
            "--version"
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode == 0:

        log(
            "✓ pip уже установлен."
        )

        log(
            result.stdout.strip()
        )

        return True

    return False


# ============================================================
# УСТАНОВКА PIP
# ============================================================

def install_pip():

    if check_pip():
        return

    log("")
    log("==========================================")
    log("УСТАНОВКА PIP")
    log("==========================================")

    TEMP.mkdir(
        parents=True,
        exist_ok=True
    )

    get_pip = (
        TEMP /
        "get-pip.py"
    )

    download(
        "https://bootstrap.pypa.io/get-pip.py",
        get_pip
    )

    python = get_python()

    subprocess.check_call(
        [
            python,
            str(get_pip),
            "--no-warn-script-location"
        ],
        cwd=str(ROOT)
    )

    get_pip.unlink(
        missing_ok=True
    )

    if not check_pip():

        raise Exception(
            "pip не удалось установить."
        )

    log(
        "✓ pip установлен."
    )


# ============================================================
# УСТАНОВКА REQUIREMENTS
# ============================================================

def install_requirements():

    python = get_python()

    if not REQUIREMENTS.exists():

        raise Exception(
            "requirements.txt не найден:\n"
            f"{REQUIREMENTS}"
        )

    log("")
    log("==========================================")
    log("УСТАНОВКА ЗАВИСИМОСТЕЙ")
    log("==========================================")

    log(
        f"Requirements: {REQUIREMENTS}"
    )

    log(
        f"Python: {python}"
    )

    # --------------------------------------------------------
    # ОБНОВЛЯЕМ PIP
    # --------------------------------------------------------

    subprocess.check_call(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip"
        ],
        cwd=str(ROOT)
    )

    # --------------------------------------------------------
    # SETUPTOOLS
    # --------------------------------------------------------

    subprocess.check_call(
        [
            python,
            "-m",
            "pip",
            "install",
            "setuptools==69.5.1",
            "wheel"
        ],
        cwd=str(ROOT)
    )

    # --------------------------------------------------------
    # REQUIREMENTS
    # --------------------------------------------------------

    subprocess.check_call(
        [
            python,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS)
        ],
        cwd=str(ROOT)
    )

    log("")
    log(
        "✓ Все зависимости установлены."
    )


# ============================================================
# ПРОВЕРКА КРИТИЧЕСКИХ БИБЛИОТЕК
# ============================================================

def verify_dependencies():

    python = get_python()

    log("")
    log("==========================================")
    log("ПРОВЕРКА БИБЛИОТЕК")
    log("==========================================")

    # Эти импорты соответствуют твоему search_ultimate.py
    test_code = """
import faiss
import numpy
import sentence_transformers
import rank_bm25
import transformers
"""

    result = subprocess.run(
        [
            python,
            "-c",
            test_code
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode == 0:

        log(
            "✓ FAISS"
        )

        log(
            "✓ NumPy"
        )

        log(
            "✓ SentenceTransformers"
        )

        log(
            "✓ rank_bm25"
        )

        log(
            "✓ Transformers"
        )

        return True

    log(
        "⚠ Не все необходимые библиотеки установлены."
    )

    if result.stderr:

        log(
            result.stderr
        )

    return False



# ============================================================
# ПРОВЕРКА И УСТАНОВКА AI-МОДЕЛЕЙ
# ============================================================

def install_models():

    models_dir = ROOT / "models"

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log("")
    log("==========================================")
    log("ПРОВЕРКА AI МОДЕЛЕЙ")
    log("==========================================")

    # --------------------------------------------------------
    # МОДЕЛИ, КОТОРЫЕ ИСПОЛЬЗУЕТ search_ultimate.py
    # --------------------------------------------------------

    embedding_model = (
        models_dir /
        "paraphrase-multilingual-mpnet-base-v2"
    )

    reranker_model = (
        models_dir /
        "ms-marco-MiniLM-L-6-v2"
    )

    llm_model = (
        models_dir /
        "llm"
    )

    # --------------------------------------------------------
    # ПРОВЕРКА EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_ok = (
        embedding_model.exists()
        and any(
            embedding_model.iterdir()
        )
    )

    # --------------------------------------------------------
    # ПРОВЕРКА RERANKER
    # --------------------------------------------------------

    reranker_ok = (
        reranker_model.exists()
        and any(
            reranker_model.iterdir()
        )
    )

    # --------------------------------------------------------
    # ПРОВЕРКА LLM
    # --------------------------------------------------------

    llm_ok = (
        llm_model.exists()
        and any(
            llm_model.iterdir()
        )
    )

    # --------------------------------------------------------
    # ВЫВОД
    # --------------------------------------------------------

    if embedding_ok:

        log(
            "✓ Embedding model найдена."
        )

    else:

        log(
            "❌ Embedding model отсутствует."
        )

    if reranker_ok:

        log(
            "✓ Reranker model найдена."
        )

    else:

        log(
            "❌ Reranker model отсутствует."
        )

    if llm_ok:

        log(
            "✓ LLM найдена."
        )

    else:

        log(
            "❌ LLM отсутствует."
        )

    # --------------------------------------------------------
    # ВСЁ ЕСТЬ
    # --------------------------------------------------------

    if (
        embedding_ok
        and reranker_ok
        and llm_ok
    ):

        log("")
        log(
            "✓ Все AI модели уже установлены."
        )

        return

    # --------------------------------------------------------
    # ИЩЕМ DOWNLOAD_MODELS.PY
    # --------------------------------------------------------

    possible_paths = [

        ROOT /
        "download_models.py",

        ROOT /
        "src" /
        "download_models.py",

        models_dir /
        "download_models.py"

    ]

    downloader = None

    for path in possible_paths:

        if path.exists():

            downloader = path
            break

    # --------------------------------------------------------
    # ЗАГРУЗЧИК НЕ НАЙДЕН
    # --------------------------------------------------------

    if downloader is None:

        raise Exception(
            "AI модели отсутствуют, "
            "но download_models.py не найден.\n\n"
            "Ожидался файл:\n"
            f"{ROOT / 'download_models.py'}"
        )

    # --------------------------------------------------------
    # ЗАПУСК DOWNLOAD_MODELS.PY
    # --------------------------------------------------------

    python = get_python()

    log("")
    log("==========================================")
    log("СКАЧИВАНИЕ AI МОДЕЛЕЙ")
    log("==========================================")

    log(
        f"Python: {python}"
    )

    log(
        f"Downloader: {downloader}"
    )

    log("")

    try:

        subprocess.check_call(
            [
                python,
                str(downloader)
            ],
            cwd=str(ROOT)
        )

    except subprocess.CalledProcessError as e:

        raise Exception(
            "Не удалось скачать AI модели.\n"
            f"download_models.py завершился с кодом: "
            f"{e.returncode}"
        )

    # --------------------------------------------------------
    # ПРОВЕРЯЕМ ПОСЛЕ СКАЧИВАНИЯ
    # --------------------------------------------------------

    log("")
    log(
        "Проверка скачанных моделей..."
    )

    embedding_ok = (
        embedding_model.exists()
        and any(
            embedding_model.iterdir()
        )
    )

    reranker_ok = (
        reranker_model.exists()
        and any(
            reranker_model.iterdir()
        )
    )

    llm_ok = (
        llm_model.exists()
        and any(
            llm_model.iterdir()
        )
    )

    # --------------------------------------------------------
    # РЕЗУЛЬТАТ
    # --------------------------------------------------------

    if not embedding_ok:

        raise Exception(
            "Embedding model не была установлена:\n"
            f"{embedding_model}"
        )

    if not reranker_ok:

        raise Exception(
            "Reranker model не была установлена:\n"
            f"{reranker_model}"
        )

    if not llm_ok:

        raise Exception(
            "LLM не была установлена:\n"
            f"{llm_model}"
        )

    log("")
    log(
        "✓ Embedding model установлена."
    )

    log(
        "✓ Reranker model установлена."
    )

    log(
        "✓ LLM установлена."
    )

    log("")
    log(
        "✓ Все AI модели готовы."
    )


    # --------------------------------------------------------
    # Проверяем downloader
    # --------------------------------------------------------

    possible_paths = [

        ROOT /
        "download_models.py",

        ROOT /
        "src" /
        "download_models.py",

        ROOT /
        "models" /
        "download_models.py"

    ]

    downloader = None

    for path in possible_paths:

        if path.exists():

            downloader = path
            break

    # --------------------------------------------------------
    # Если models.ready существует, считаем модели готовыми
    # --------------------------------------------------------

    if flag.exists():

        log("")
        log(
            "✓ AI models already installed."
        )

        return

    # --------------------------------------------------------
    # downloader не найден
    # --------------------------------------------------------

    if downloader is None:

        log("")
        log(
            "⚠ download_models.py не найден."
        )

        log(
            "Пропускаем автоматическую загрузку моделей."
        )

        return

    # --------------------------------------------------------
    # ЗАПУСК
    # --------------------------------------------------------

    python = get_python()

    log("")
    log("==========================================")
    log("ЗАГРУЗКА AI МОДЕЛЕЙ")
    log("==========================================")

    log(
        f"Python: {python}"
    )

    log(
        f"Downloader: {downloader}"
    )

    subprocess.check_call(
        [
            python,
            str(downloader)
        ],
        cwd=str(ROOT)
    )

    flag.write_text(
        "ok",
        encoding="utf-8"
    )

    log(
        "✓ AI models installed."
    )


# ============================================================
# ЗАПУСК GUI
# ============================================================

def start_app():

    python = get_python()

    if not PYTHON_EXE.exists():

        raise Exception(
            f"Python не найден:\n{PYTHON_EXE}"
        )

    if not APP_FILE.exists():

        raise Exception(
            f"gui.py не найден:\n{APP_FILE}"
        )

    log("")
    log("==========================================")
    log("ЗАПУСК VIDEOSearch")
    log("==========================================")

    log(
        f"Python: {python}"
    )

    log(
        f"GUI: {APP_FILE}"
    )

    log(
        f"ROOT: {ROOT}"
    )

    subprocess.Popen(
        [
            python,
            str(APP_FILE)
        ],
        cwd=str(ROOT)
    )


# ============================================================
# ОЧИСТКА
# ============================================================

def cleanup():

    if TEMP.exists():

        shutil.rmtree(
            TEMP,
            ignore_errors=True
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        log(
            "=========================================="
        )

        log(
            "VideoSearch Launcher"
        )

        log(
            "=========================================="
        )

        log(
            f"ROOT: {ROOT}"
        )

        # ----------------------------------------------------
        # 1. PYTHON
        # ----------------------------------------------------

        install_python()

        # ----------------------------------------------------
        # 2. НАСТРАИВАЕМ EMBEDDED PYTHON
        # ----------------------------------------------------

        configure_embedded_python()

        # ----------------------------------------------------
        # 3. ПРОВЕРЯЕМ PYTHON
        # ----------------------------------------------------

        check_python()

        # ----------------------------------------------------
        # 4. PIP
        # ----------------------------------------------------

        install_pip()

        # ----------------------------------------------------
        # 5. БИБЛИОТЕКИ
        # ----------------------------------------------------

        dependencies_ok = verify_dependencies()

        if not dependencies_ok:

            install_requirements()

            # После установки проверяем ещё раз
            if not verify_dependencies():

                raise Exception(
                    "После установки requirements.txt "
                    "необходимые библиотеки всё ещё "
                    "недоступны."
                )

        else:

            log(
                "✓ Все основные зависимости уже установлены."
            )

        # ----------------------------------------------------
        # 6. МОДЕЛИ
        # ----------------------------------------------------

        install_models()


        # ----------------------------------------------------
        # 6.1 ОЧИСТКА LLM CONFIG
        # ----------------------------------------------------

        clean_llm_generation_config()


        # ----------------------------------------------------
        # 7. CLEANUP
        # ----------------------------------------------------

        cleanup()

        # ----------------------------------------------------
        # 8. GUI
        # ----------------------------------------------------

        start_app()

    except Exception as e:

        log("")
        log("==========================================")
        log("ERROR")
        log("==========================================")

        log(
            str(e)
        )

        log("")

        input(
            "Press Enter..."
        )
