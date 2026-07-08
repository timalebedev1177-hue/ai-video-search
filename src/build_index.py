# build_index.py
# Сборка индекса FAISS с уменьшением памяти:
# - один вектор на сегмент по умолчанию
# - при длинном тексте добавляется head-вектор
# - можно включить PQ-квантование для более компактного индекса

import os
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


EMBED_MODEL = "models/paraphrase-multilingual-mpnet-base-v2"


# Оптимизация памяти
USE_COMPRESSED_INDEX = True
ADD_HEAD_VECTOR = True
ADD_TAIL_VECTOR = False   # выключено, чтобы не дублировать без необходимости
MIN_WORDS_FOR_EXTRA_PARTS = 24

# Параметры IVFPQ
NLIST = 64
M = 8
BITS = 8
NPROBE = 10

os.makedirs("indexes", exist_ok=True)

model = SentenceTransformer(EMBED_MODEL)

print("MODEL CHECK:", model.encode(["test"]).shape)
print("MODEL DIM:", model.get_sentence_embedding_dimension())

all_embeddings = []
all_meta = []


def make_parts(text: str):
    words = text.split()
    parts = [("full", text)]

    if len(words) >= MIN_WORDS_FOR_EXTRA_PARTS:
        if ADD_HEAD_VECTOR:
            parts.append(("head", " ".join(words[:30])))
        if ADD_TAIL_VECTOR:
            parts.append(("tail", " ".join(words[-30:])))

    return parts


for file in os.listdir("indexes"):
    if not file.endswith(".segments"):
        continue

    name = file.replace(".segments", "")

    with open(os.path.join("indexes", file), "rb") as f:
        segments = pickle.load(f)

    texts = []
    metas = []

    for i, s in enumerate(segments):
        text = s["text"]
        parts = make_parts(text)

        for part_type, part_text in parts:
            texts.append(part_text)
            metas.append({
                "video": name + ".mp4",
                "start": s["start"],
                "end": s["end"],
                "orig_idx": i,
                "part": part_type,
                "text": part_text
            })

    if not texts:
        continue

    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    print("VEC SHAPE:", emb.shape)
    print("VEC SAMPLE:", emb[0][:10])

    faiss.normalize_L2(emb)

    all_embeddings.append(emb)
    all_meta.extend(metas)

    print(f"Processed {name}, vectors added: {len(texts)}")

if not all_embeddings:
    raise RuntimeError("Нет данных для индексации.")

all_embeddings = np.vstack(all_embeddings).astype("float32")
d = all_embeddings.shape[1]
print("FINAL INDEX DIM:", d)
print("TOTAL VECTORS:", len(all_meta))
print("Всего векторов:", len(all_meta))
print("Размерность:", d)

# -----------------------------
# IVF ИНДЕКС (IVFFlat)
# -----------------------------

USE_IVF = True

if USE_IVF:
    print("Используется IVF (IVFFlat) индекс...")

    # параметры IVF
    NLIST = 100   # число кластеров (для 2k ~ 50–100, для 100k ~ 200–1000)
    NPROBE = 10   # сколько кластеров смотреть при поиске

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_INNER_PRODUCT)

    # обучение
    train_size = min(len(all_embeddings), max(2000, NLIST * 50))
    train_x = all_embeddings[:train_size]

    print(f"Обучение IVF на {len(train_x)} векторах...")
    index.train(train_x)

    # добавление
    index.add(all_embeddings)

    # параметр поиска
    index.nprobe = NPROBE

else:
    print("Используется IndexFlatIP...")
    index = faiss.IndexFlatIP(d)
    index.add(all_embeddings)

faiss.write_index(index, "indexes/video.index")

with open("indexes/video.meta", "wb") as f:
    pickle.dump(all_meta, f)

print("Индекс создан, количество векторов:", len(all_meta))