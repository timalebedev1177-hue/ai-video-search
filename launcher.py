import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path


# ----------------------------
# НАСТРОЙКИ
# ----------------------------

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).parent
else:
    ROOT = Path(__file__).parent


PYTHON_DIR = ROOT / "python"

REQUIREMENTS = ROOT / "requirements.txt"

APP_FILE = ROOT / "src" / "gui.py"


PYTHON_URL = (
    "https://www.python.org/ftp/python/"
    "3.10.11/python-3.10.11-embed-amd64.zip"
)


TEMP = ROOT / "temp"


# ----------------------------
# ЛОГ
# ----------------------------

def log(text):
    print(text)


# ----------------------------
# СКАЧИВАНИЕ
# ----------------------------

def download(url, path):

    path.parent.mkdir(
        exist_ok=True,
        parents=True
    )

    log(
        f"Downloading {url}"
    )

    urllib.request.urlretrieve(
        url,
        path
    )


# ----------------------------
# PYTHON PATH
# ----------------------------

def get_python():

    return str(
        PYTHON_DIR / "python.exe"
    )


# ----------------------------
# УСТАНОВКА PYTHON
# ----------------------------

def install_python():

    python = get_python()

    if os.path.exists(python):
        return


    log(
        "Installing Python..."
    )


    TEMP.mkdir(
        exist_ok=True
    )


    archive = TEMP / "python.zip"


    download(
        PYTHON_URL,
        archive
    )


    PYTHON_DIR.mkdir(
        exist_ok=True
    )


    with zipfile.ZipFile(
        archive
    ) as z:

        z.extractall(
            PYTHON_DIR
        )


    archive.unlink()


    # включаем site-packages

    pth = PYTHON_DIR / "python310._pth"


    if pth.exists():

        text = pth.read_text(
            encoding="utf-8"
        )


        text = text.replace(
            "#import site",
            "import site"
        )


        pth.write_text(
            text,
            encoding="utf-8"
        )


    python = get_python()


    if not os.path.exists(python):

        raise Exception(
            f"Python не найден: {python}"
        )


    # установка pip

    log(
        "Installing pip..."
    )


    get_pip = TEMP / "get-pip.py"


    download(
        "https://bootstrap.pypa.io/get-pip.py",
        get_pip
    )


    subprocess.check_call(
        [
            python,
            str(get_pip)
        ]
    )


    get_pip.unlink()



# ----------------------------
# ПРОВЕРКА PYTHON
# ----------------------------

def check_python():

    python = get_python()


    result = subprocess.run(
        [
            python,
            "--version"
        ],
        capture_output=True,
        text=True
    )


    print(
        "Python version:",
        result.stdout.strip()
    )



# ----------------------------
# ПРОВЕРКА БИБЛИОТЕК
# ----------------------------

def check_packages():

    python = get_python()


    packages = [
        "torch",
        "whisper",
        "PySide6",
        "transformers",
        "faiss-cpu",
        "huggingface_hub"
    ]


    for package in packages:


        result = subprocess.run(
            [
                python,
                "-m",
                "pip",
                "show",
                package
            ],
            capture_output=True,
            text=True
        )


        if result.returncode == 0:

            print(
                f"✅ {package}"
            )

        else:

            print(
                f"❌ {package} НЕ НАЙДЕН"
            )



# ----------------------------
# УСТАНОВКА БИБЛИОТЕК
# ----------------------------

def install_requirements():

    flag = ROOT / "python" / "installed.flag"


    if flag.exists():
        return


    python = get_python()


    log(
        "Installing libraries..."
    )


    # обновляем pip
    subprocess.check_call(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip"
        ]
    )


    # ВАЖНО:
    # Whisper требует pkg_resources
    subprocess.check_call(
        [
            python,
            "-m",
            "pip",
            "install",
            "setuptools==69.5.1",
            "wheel"
        ]
    )


    # ставим зависимости проекта
    subprocess.check_call(
        [
            python,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS)
        ]
    )


    flag.write_text(
        "ok",
        encoding="utf-8"
    )



# ----------------------------
# МОДЕЛИ ИИ
# ----------------------------

def install_models():

    models_dir = ROOT / "models"


    models_dir.mkdir(
        exist_ok=True
    )


    flag = models_dir / "models.ready"


    if flag.exists():

        return


    downloader = (
        models_dir /
        "download_models.py"
    )


    if not downloader.exists():

        raise Exception(
            "download_models.py не найден"
        )


    python = get_python()


    log(
        "Downloading AI models..."
    )


    subprocess.check_call(
        [
            python,
            str(downloader)
        ]
    )


    flag.write_text(
        "ok",
        encoding="utf-8"
    )



# ----------------------------
# ЗАПУСК
# ----------------------------

def start_app():

    python = get_python()


    subprocess.Popen(
        [
            python,
            str(APP_FILE)
        ],
        cwd=str(ROOT)
    )



# ----------------------------
# ОЧИСТКА
# ----------------------------

def cleanup():

    if TEMP.exists():

        shutil.rmtree(
            TEMP,
            ignore_errors=True
        )



# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":


    try:

        log(
            "VideoSearch Installer"
        )


        install_python()


        check_python()


        install_requirements()


        check_packages()


        install_models()


        cleanup()


        if not APP_FILE.exists():

            raise Exception(
                "gui.py not found"
            )


        start_app()



    except Exception as e:


        log(
            "ERROR:"
        )


        log(
            str(e)
        )


        input(
            "Press Enter..."
        )