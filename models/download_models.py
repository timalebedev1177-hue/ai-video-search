import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download


if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).parent
else:
    ROOT = Path(__file__).parent


# папка для моделей
MODELS_DIR = ROOT / "models"

MODELS_DIR.mkdir(
    exist_ok=True,
    parents=True
)


MODELS = [

    {
        "name": "Qwen2.5-3B-Instruct",
        "repo": "Qwen/Qwen2.5-3B-Instruct"
    },

    {
        "name": "llm",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct"
    },

    {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "repo": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    },

    {
        "name": "cross-encoder-ms-marco-MiniLM-L6-v2",
        "repo": "cross-encoder/ms-marco-MiniLM-L6-v2"
    },

]


def download_model(name, repo):

    target = MODELS_DIR / name


    if target.exists():

        print(
            f"✅ {name} уже есть"
        )

        return


    print(
        f"\n⬇ Скачивание {name}"
    )


    snapshot_download(

        repo_id=repo,

        local_dir=str(target),

        local_dir_use_symlinks=False

    )


    print(
        f"✅ {name} загружена"
    )



def main():

    print(
        "🤖 Загрузка AI моделей HuggingFace"
    )


    for model in MODELS:

        download_model(
            model["name"],
            model["repo"]
        )


    # флаг завершения

    flag = MODELS_DIR / "models.ready"

    flag.write_text(
        "ok",
        encoding="utf-8"
    )


    print(
        "\n🎉 Все модели установлены"
    )



if __name__ == "__main__":

    main()