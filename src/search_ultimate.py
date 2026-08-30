# src/search_ultimate.py
# Поиск видео: query expansion + FAISS + BM25 + rerank + merge + AI result selection + smart cut

import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import pickle
import subprocess
import argparse
import time
from pathlib import Path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from cut_clip import cut_clip
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


# -----------------------------
# ПАРАМЕТРЫ ПОИСКА
# -----------------------------

TOP_K = 200
BM25_K = 200

FINAL_RESULTS = 10

# Объединение соседних сегментов
MERGE_DISTANCE = 5
SEMANTIC_MERGE_THRESHOLD = 0.82
MAX_MERGE_DURATION = 30

# Максимальная длина результата
MAX_RESULT_DURATION = 120

# Минимальные пороги
MIN_EMBED_SCORE = 0.20
MIN_RERANK_SCORE = 0.08

# CrossEncoder
MAX_RERANK_LENGTH = 800

# Сколько кандидатов реально отправлять в CrossEncoder
RERANK_K = 180

QSIM_K = 50
# Финальный рейтинг
W_EMBED = 0.15
W_BM25 = 0.10
W_RERANK = 0.50
W_QUERY = 0.10
W_HITS = 0.05

# Предварительный рейтинг
PRE_EMBED_WEIGHT = 0.55
PRE_BM25_WEIGHT = 0.30
PRE_HITS_WEIGHT = 0.15

# -----------------------------
# LLM
# -----------------------------

EXPAND_VARIANTS_K = 3
MIN_VARIANT_SIMILARITY = 0.72
MAX_VARIANT_WORDS = 10
MAX_EXPAND_VARIANTS = 2

EMBEDDINGS_FILE = "indexes/video.embeddings.npy"

# Штраф за слишком длинный результат
LENGTH_PENALTY_WEIGHT = 0.003
LENGTH_PENALTY_TARGET = 60.0



# -----------------------------
# ПУТИ ПРОЕКТА
# -----------------------------

# Корень проекта VideoSearch
BASE_DIR = Path(__file__).resolve().parent.parent

# Модели
EMBED_MODEL_PATH = BASE_DIR / "models" / "paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL_PATH = BASE_DIR / "models" / "ms-marco-MiniLM-L-6-v2"
LLM_MODEL = BASE_DIR / "models" / "llm"

# Индексы
INDEX_FILE = BASE_DIR / "indexes" / "video.index"
META_FILE = BASE_DIR / "indexes" / "video.meta"
EMBEDDINGS_FILE = BASE_DIR / "indexes" / "video.embeddings.npy"

# Кэш
CACHE_FILE = BASE_DIR / "indexes" / "query_cache.json"

# Видео
VIDEOS_DIR = BASE_DIR / "videos"

# Транскрипции
TRANSCRIPTIONS_DIR = BASE_DIR / "transcriptions"


# -----------------------------
# ПАРСИНГ АРГУМЕНТОВ
# -----------------------------

parser = argparse.ArgumentParser(add_help=False)

# ВАЖНО:
# query должен забирать ВСЁ оставшееся
# иначе --cut-index ломает парсинг
parser.add_argument(
    "query",
    nargs="*",
    default=[]
)
parser.add_argument("--no-cut", action="store_true")
parser.add_argument("--cut-index", type=int, default=None)
parser.add_argument("--auto-cut", action="store_true", help="Автоматически вырезать лучший фрагмент")
parser.add_argument("--no-llm", action="store_true", help="Отключить LLM")
parser.add_argument(
    "--llm-select",
    action="store_true",
    help="Включить AI выбор лучшего результата"
)
parser.add_argument(
    "--llm-expand",
    action="store_true",
    help="Включить LLM query expansion"
)
parser.add_argument(
    "--gui",
    action="store_true",
    help="Запуск из GUI"
)
parser.add_argument(
    "--llm-refine",
    action="store_true",
    help="Включить LLM уточнение границ клипа"
)
args, _unknown = parser.parse_known_args()
USE_LLM_QUERY_EXPANSION = args.llm_expand

NO_CUT = args.no_cut
CUT_INDEX = args.cut_index
AUTO_CUT = args.auto_cut
GUI_MODE = args.gui
USE_LLM = not args.no_llm

USE_LLM_RESULT_SELECTION = args.llm_select
USE_LLM_REFINE = args.llm_refine

# -----------------------------
# МОДЕЛИ
# -----------------------------

print("Загрузка embedding модели...")

embed_model = SentenceTransformer(
    str(EMBED_MODEL_PATH)
)



reranker = CrossEncoder(
    str(RERANKER_MODEL_PATH)
)

llm_pipeline = None

if USE_LLM and TRANSFORMERS_AVAILABLE and LLM_MODEL.exists():

    try:
        print("Попытка загрузить локальную LLM...")

        tokenizer = AutoTokenizer.from_pretrained(
            str(LLM_MODEL),
            local_files_only=True,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            str(LLM_MODEL),
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

else:
    if USE_LLM and not LLM_MODEL.exists():
        print(f"LLM не найдена: {LLM_MODEL}")

    print("LLM не используется")

# -----------------------------
# FAISS / META
# -----------------------------

print("Загрузка индекса...")

if not INDEX_FILE.exists():
    print(f"Ошибка: FAISS индекс не найден:")
    print(INDEX_FILE)
    sys.exit(1)

if not META_FILE.exists():
    print(f"Ошибка: metadata не найдены:")
    print(META_FILE)
    sys.exit(1)

index = faiss.read_index(
    str(INDEX_FILE)
)

# -----------------------------
# Настройка FAISS
# -----------------------------

if hasattr(index, "nprobe"):
    # IVF
    # Баланс между скоростью и recall.
    index.nprobe = min(
        8,
        getattr(index, "nlist", 8)
    )

elif hasattr(index, "hnsw"):
    # HNSW
    index.hnsw.efSearch = 64

print(
    f"FAISS: {index.ntotal} векторов, "
    f"dimension={index.d}"
)

# -----------------------------
# Metadata
# -----------------------------

with open(META_FILE, "rb") as f:
    meta = pickle.load(f)

print(f"Metadata: {len(meta)} сегментов")


# -----------------------------
# ГОТОВЫЕ EMBEDDING СЕГМЕНТОВ
# -----------------------------

segment_embeddings = None
meta_index = {}

if EMBEDDINGS_FILE.exists():

    try:
        print("Загрузка готовых embeddings...")

        segment_embeddings = np.load(
            EMBEDDINGS_FILE,
            mmap_mode="r"
        )

        if (
            segment_embeddings.ndim == 2
            and segment_embeddings.shape[0] == len(meta)
            and segment_embeddings.shape[1] == index.d
        ):

            # Индекс:
            # (video, start, end, part) -> vector index

            meta_index = {
                (
                    m["video"],
                    m["start"],
                    m["end"],
                    m.get("part", "full")
                ): idx
                for idx, m in enumerate(meta)
            }

            print(
                f"Готовые embeddings загружены: "
                f"{segment_embeddings.shape}"
            )

        else:

            print(
                "Предупреждение: embeddings "
                "не совпадают с metadata/FAISS."
            )

            segment_embeddings = None

    except Exception as e:

        print(
            "Не удалось загрузить embeddings:",
            e
        )

        segment_embeddings = None

else:

    print(
        f"Готовые embeddings не найдены: "
        f"{EMBEDDINGS_FILE}"
    )


# -----------------------------
# КЭШ EMBEDDING
# -----------------------------

QUERY_CACHE_DIRTY = False

if CACHE_FILE.exists():

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            query_cache = json.load(f)

        if not isinstance(query_cache, dict):
            query_cache = {}

    except Exception as e:

        print(
            "Не удалось загрузить query cache:",
            e
        )

        query_cache = {}

else:

    query_cache = {}

QUERY_CACHE_DIRTY = False
# -----------------------------
# НОРМАЛИЗАЦИЯ QUERY
# -----------------------------

RU_FIXES = {

}


def normalize_query_text(q: str):
    q = q.lower().strip()

    # убрать мусор
    q = re.sub(r"[^\w\sа-яА-ЯёЁ]", " ", q)

    # схлопнуть пробелы
    q = re.sub(r"\s+", " ", q)

    words = q.split()

    fixed = []
    for w in words:
        fixed.append(RU_FIXES.get(w, w))

    q = " ".join(fixed)

    return q

def get_query_embedding(q: str) -> np.ndarray:

    global QUERY_CACHE_DIRTY

    q = normalize_query_text(q)

    if not q:
        return np.zeros(
            index.d,
            dtype="float32"
        )

    cached = query_cache.get(q)

    if cached is not None:

        vec = np.asarray(
            cached,
            dtype="float32"
        )

        # Защита от поврежденного cache
        if vec.shape[0] == index.d:
            return vec

    vec = embed_model.encode(
        q,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    query_cache[q] = vec.tolist()

    QUERY_CACHE_DIRTY = True

    return vec

# -----------------------------
# СОХРАНЕНИЕ QUERY CACHE
# -----------------------------

def save_query_cache():
    global QUERY_CACHE_DIRTY

    if not QUERY_CACHE_DIRTY:
        return

    try:
        os.makedirs(
            os.path.dirname(CACHE_FILE),
            exist_ok=True
        )

        temp_file = Path(
            str(CACHE_FILE) + ".tmp"
        )

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                query_cache,
                f,
                ensure_ascii=False
            )

        os.replace(
            temp_file,
            CACHE_FILE
        )

        QUERY_CACHE_DIRTY = False

    except Exception as e:
        print(
            "Не удалось сохранить query cache:",
            e
        )

# -----------------------------
# BM25
# -----------------------------

print("Подготовка BM25...")

corpus = []
corpus_meta = []
seen = set()

for m in meta:

    key = (
        m["video"],
        m["start"],
        m["end"],
        m.get("part", "full")
    )

    if key in seen:
        continue

    seen.add(key)

    corpus.append(
        re.findall(
            r"\w+",
            m["text"].lower()
        )
    )

    corpus_meta.append({
        "video": m["video"],
        "start": m["start"],
        "end": m["end"],
        "text": m["text"],
        "orig_idx": m.get("orig_idx"),
        "part": m.get("part", "full")
    })

bm25 = BM25Okapi(corpus) if corpus else None

# -----------------------------
# ВСПОМОГАТЕЛЬНЫЕ
# -----------------------------

def normalize(scores):
    arr = np.array(scores, dtype=np.float32)
    if len(arr) == 0:
        return arr
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def normalize_query_variants(variants):
    clean = []
    seen = set()

    for v in variants:
        if not isinstance(v, str):
            continue

        v = re.sub(r"\s+", " ", v.strip())

        if len(v) < 3:
            continue

        key = v.lower()
        if key in seen:
            continue

        seen.add(key)
        clean.append(v)

    return clean


def get_meta_vector(item):
    if segment_embeddings is None:
        return None
    key = (item["video"], item["start"], item["end"], item.get("part", "full"))
    idx = meta_index.get(key)
    return segment_embeddings[idx] if idx is not None else None


def remove_duplicates(results):
    seen = set()
    unique = []

    for r in results:
        key = (r["video"], int(r["start"]), int(r["end"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    return unique


def semantic_merge(segments):
    if not segments:
        return []

    segments = sorted(
        segments,
        key=lambda x: (x["video"], float(x["start"]))
    )

    merged = []

    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]

        # Другое видео
        if seg["video"] != last["video"]:
            merged.append(seg)
            continue

        dist = float(seg["start"]) - float(last["end"])

        # Слишком далеко
        if dist > MERGE_DISTANCE:
            merged.append(seg)
            continue

        combined_duration = (
            max(float(last["end"]), float(seg["end"]))
            - float(last["start"])
        )

        # Слишком длинный объединенный результат
        if combined_duration > MAX_MERGE_DURATION:
            merged.append(seg)
            continue

        a = last.get("vector")
        b = seg.get("vector")

        # Если оба embedding доступны — проверяем смысл.
        if a is not None and b is not None:
            sim = float(np.dot(a, b))

            if sim < SEMANTIC_MERGE_THRESHOLD:
                merged.append(seg)
                continue

        # Если embeddings отсутствуют,
        # не объединяем автоматически.
        elif a is None or b is None:
            merged.append(seg)
            continue

        # -----------------------------
        # ОБЪЕДИНЕНИЕ
        # -----------------------------

        last["end"] = max(
            float(last["end"]),
            float(seg["end"])
        )

        last["text"] = (
            last.get("text", "").strip()
            + " "
            + seg.get("text", "").strip()
        ).strip()

        last["embed_score"] = max(
            float(last.get("embed_score", 0.0)),
            float(seg.get("embed_score", 0.0))
        )

        last["bm25_score"] = max(
            float(last.get("bm25_score", 0.0)),
            float(seg.get("bm25_score", 0.0))
        )

        last["hits"] = (
            last.get("hits", 0)
            + seg.get("hits", 0)
        )

        # Источники
        last.setdefault("source_segments", [])
        last["source_segments"].extend(
            seg.get("source_segments", [])
        )

        # Запросы
        last.setdefault("matched_queries", set())
        last["matched_queries"].update(
            seg.get("matched_queries", set())
        )

        # Объединенный vector
        if a is not None and b is not None:
            merged_vector = (
                np.asarray(a, dtype="float32")
                + np.asarray(b, dtype="float32")
            )

            norm = np.linalg.norm(merged_vector)

            if norm > 1e-8:
                merged_vector /= norm

            last["vector"] = merged_vector.astype("float32")

    return merged


def llm_generate(prompt, max_new_tokens=120):
    if not llm_pipeline:
        return None

    try:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

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

        return out.strip()

    except Exception as e:
        print("LLM вызов не удался:", e)
        return None


def extract_json_value(text):
    if not text:
        return None

    raw = text.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"\[[\s\S]*?\]", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    m = re.search(r"\{[\s\S]*?\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    raw = raw.strip().strip('"').strip("'")
    return raw if raw else None


def shorten_variant_text(text, max_words=MAX_VARIANT_WORDS):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip() + "..."


def query_similarity_filter(
    base_query,
    variants,
    threshold=MIN_VARIANT_SIMILARITY
):
    if not variants:
        return []

    base_query = normalize_query_text(
        base_query
    )

    clean_variants = []

    for v in variants:

        if not isinstance(v, str):
            continue

        normalized = normalize_query_text(v)

        if normalized:
            clean_variants.append(
                (v, normalized)
            )

    if not clean_variants:
        return []

    texts = [
        base_query
    ] + [
        normalized
        for _, normalized in clean_variants
    ]

    # Один batch encode
    vecs = embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    base_vec = vecs[0]
    candidate_vecs = vecs[1:]

    similarities = np.dot(
        candidate_vecs,
        base_vec
    )

    scored = sorted(
        (
            (float(score), original)
            for (original, _), score
            in zip(
                clean_variants,
                similarities
            )
        ),
        reverse=True
    )

    good = [
        original
        for score, original in scored
        if score >= threshold
    ]

    return good or [base_query]

# -----------------------------
# QUERY EXPANSION
# -----------------------------

def generate_variants_with_llm(query, n_variants=3):
    if not llm_pipeline:
        return None

    prompt = f"""
Ты оптимизируешь поисковый запрос для поиска фрагмента видео.

Исходный запрос:
"{query}"

Создай {n_variants} коротких вариантов того же запроса.

Правила:
- смысл НЕ менять;
- не добавлять новые факты;
- не расширять тему;
- не придумывать информацию;
- можно исправлять ошибки;
- можно использовать близкие синонимы;
- можно менять порядок слов;
- каждый вариант должен быть пригоден для поиска по транскрипции.

Верни только варианты, по одному на строке.
Без нумерации.
Без пояснений.
"""

    out = llm_generate(
        prompt,
        max_new_tokens=80
    )

    if not out:
        return None

    lines = []

    for line in out.splitlines():
        line = re.sub(
            r"^\s*[\d\-\*•]+[\).\:\-\s]*",
            "",
            line
        )

        line = line.strip(
            " \t\n\r\"'[]{}"
        )

        if line:
            lines.append(line)

    clean = []
    seen = set()

    original_words = len(
        normalize_query_text(query).split()
    )

    for v in lines:
        v = re.sub(r"\s+", " ", v)

        normalized = normalize_query_text(v)

        if not normalized:
            continue

        words = normalized.split()

        if len(words) > MAX_VARIANT_WORDS:
            continue

        if abs(len(words) - original_words) > 3:
            continue

        key = normalized.lower()

        if key in seen:
            continue

        seen.add(key)
        clean.append(v)

    return clean[:n_variants] or None


def generate_variants_semantic(query, top_k=5):
    q_vec = get_query_embedding(query).reshape(1, -1)

    scores, ids = index.search(q_vec, top_k)

    variants = [query]

    for idx in ids[0]:
        if idx < 0:
            continue

        t = meta[idx]["text"].strip()
        sent = t.split(".")
        candidate = sent[0].strip() if sent else t
        if len(candidate.split()) > MAX_VARIANT_WORDS:
            candidate = " ".join(candidate.split()[:MAX_VARIANT_WORDS])

        if candidate and candidate not in variants:
            variants.append(candidate)

    return variants[:top_k]


def expand_query(query):
    original = normalize_query_text(query)

    if not original:
        return [query]

    variants = generate_variants_with_llm(
        query,
        EXPAND_VARIANTS_K
    )

    if variants:
        variants = normalize_query_variants(
            [query] + variants
        )

        variants = query_similarity_filter(
            query,
            variants,
            MIN_VARIANT_SIMILARITY
        )

        # Оригинальный запрос всегда первый
        normalized_original = normalize_query_text(query)

        final = [query]

        for v in variants:
            if normalize_query_text(v) == normalized_original:
                continue

            if len(final) >= MAX_EXPAND_VARIANTS + 1:
                break

            final.append(v)

        return final

    # -----------------------------
    # FALLBACK
    # -----------------------------

    variants = generate_variants_semantic(
        query,
        top_k=EXPAND_VARIANTS_K
    )

    variants = normalize_query_variants(variants)

    variants = query_similarity_filter(
        query,
        variants,
        MIN_VARIANT_SIMILARITY
    )

    if not variants:
        return [query]

    result = [query]

    for v in variants:
        if normalize_query_text(v) == original:
            continue

        if len(result) >= MAX_EXPAND_VARIANTS + 1:
            break

        result.append(v)

    return result

# -----------------------------
# СОБИРАТЕЛИ КАНДИДАТОВ
# -----------------------------

def add_faiss_candidates(
    search_query,
    candidates,
    min_score=MIN_EMBED_SCORE
):
    queries = (
        list(search_query)
        if isinstance(search_query, (list, tuple))
        else [search_query]
    )

    queries = [
        q for q in queries
        if isinstance(q, str) and q.strip()
    ]

    if not queries:
        return

    # Batch embedding
    q_vecs = np.vstack([
        get_query_embedding(q)
        for q in queries
    ]).astype("float32")

    if q_vecs.shape[1] != index.d:
        return

    scores, ids = index.search(
        q_vecs,
        TOP_K
    )

    for q_idx, q in enumerate(queries):

        for rank, idx in enumerate(ids[q_idx]):

            if idx < 0:
                continue

            score = float(
                scores[q_idx][rank]
            )

            if score < min_score:
                continue

            m = meta[idx]

            key = (
                m["video"],
                m["start"],
                m["end"]
            )

            vector = get_meta_vector(m)

            if key not in candidates:

                candidates[key] = {
                    "video": m["video"],
                    "start": float(m["start"]),
                    "end": float(m["end"]),
                    "text": m["text"],
                    "embed_score": score,
                    "bm25_score": 0.0,
                    "hits": 1,
                    "matched_queries": {q},
                    "source_segments": [m],
                    "vector": vector
                }

            else:

                candidate = candidates[key]

                candidate["embed_score"] = max(
                    candidate["embed_score"],
                    score
                )

                if q not in candidate["matched_queries"]:
                    candidate["matched_queries"].add(q)
                    candidate["hits"] += 1

                candidate["source_segments"].append(m)

                if (
                    candidate.get("vector") is None
                    and vector is not None
                ):
                    candidate["vector"] = vector

def add_bm25_candidates(
    search_query,
    candidates,
    limit=BM25_K
):
    if bm25 is None:
        return

    search_query = normalize_query_text(
        search_query
    )

    tokens = re.findall(
        r"\w+",
        search_query
    )

    if not tokens:
        return

    scores = bm25.get_scores(tokens)

    if len(scores) == 0:
        return

    limit = min(
        limit,
        len(scores)
    )

    # Быстрее полного argsort
    if len(scores) > limit:
        top_ids = np.argpartition(
            scores,
            -limit
        )[-limit:]

        top_ids = top_ids[
            np.argsort(
                scores[top_ids]
            )[::-1]
        ]
    else:
        top_ids = np.argsort(
            scores
        )[::-1]

    for idx in top_ids:

        score = float(scores[idx])

        if score <= 0:
            continue

        m = corpus_meta[idx]

        key = (
            m["video"],
            m["start"],
            m["end"]
        )

        if key not in candidates:

            candidates[key] = {
                "video": m["video"],
                "start": float(m["start"]),
                "end": float(m["end"]),
                "text": m["text"],
                "embed_score": 0.0,
                "bm25_score": score,
                "hits": 1,
                "matched_queries": {search_query},
                "source_segments": [m]
            }

        else:

            candidate = candidates[key]

            candidate["bm25_score"] = max(
                candidate["bm25_score"],
                score
            )

            if (
                search_query
                not in candidate["matched_queries"]
            ):
                candidate["matched_queries"].add(
                    search_query
                )
                candidate["hits"] += 1

# -----------------------------
# AI ВЫБОР ЛУЧШЕГО РЕЗУЛЬТАТА
# -----------------------------

def choose_best_result_with_llm(
    query,
    results,
    k=6
):
    if (
        not llm_pipeline
        or not results
        or not USE_LLM_RESULT_SELECTION
    ):
        return 0, results[0] if results else None

    top = results[:k]

    # ---------------------------------
    # Передаём LLM ТОЛЬКО тексты
    # ---------------------------------

    payload = []

    for i, r in enumerate(top, start=1):

        payload.append({
            "id": i,
            "text": r["text"][:1000]
        })

    prompt = f"""
Ты выбираешь лучший фрагмент видео для поискового запроса.

Запрос пользователя:
"{query}"

Перед тобой несколько уже найденных фрагментов видео.

Твоя задача — выбрать ОДИН фрагмент, который лучше всего отвечает
на запрос пользователя по смыслу.

ВАЖНО:

- Ты НЕ должен учитывать никакие поисковые оценки.
- Тебе НЕ передаются scores, similarity, rerank или другие оценки.
- Твой выбор должен основываться ТОЛЬКО на запросе пользователя
  и содержании текста кандидатов.
- Не оценивай качество поиска.
- Не пытайся улучшить или изменить запрос.
- Не придумывай информацию, которой нет в тексте.
- Выбирай тот фрагмент, который наиболее непосредственно отвечает
  на вопрос или запрос пользователя.

Если пользователь задаёт вопрос:

"нужно ли X?"
"почему X?"
"как X?"
"когда X?"
"кто X?"

ищи именно фрагмент, содержащий ответ на этот вопрос.

Например:

Если запрос:
"нужно ли прививаться от коронавируса?"

и один кандидат говорит:
"Коронавирус существует. Пожалуйста, прививайтесь, прививаться нужно."

а другой говорит:
"Коронавирус оказал огромное влияние на общество..."

выбирай первый, потому что он непосредственно отвечает на вопрос.

Если кандидат только упоминает тему запроса, но не отвечает на него,
не выбирай его.

Кандидаты:

{json.dumps(payload, ensure_ascii=False, indent=2)}

Верни ТОЛЬКО JSON в формате:

{{"id": 1}}

где id — номер наиболее подходящего кандидата.
"""

    out = llm_generate(
        prompt,
        max_new_tokens=20
    )

    if not out:
        return 0, top[0]

    data = extract_json_value(out)

    if isinstance(data, dict):

        try:
            idx = int(data["id"]) - 1

            if 0 <= idx < len(top):
                return idx, top[idx]

        except Exception:
            pass

    return 0, top[0]

# -----------------------------
# ВВОД
# -----------------------------

if args.query:
    query = " ".join(args.query).strip()
else:
    query = input("Введите запрос: ").strip()

if not query:
    print("Пустой запрос.")
    sys.exit(0)


# -----------------------------
# QUERY PIPELINE
# -----------------------------

profile = {}
start_total = time.perf_counter()

# ==========================================
# QUERY EXPANSION
# ==========================================

if USE_LLM_QUERY_EXPANSION:
    t0 = time.perf_counter()

    expanded_queries = expand_query(query)

    profile["query_expansion"] = time.perf_counter() - t0
else:
    expanded_queries = [query]
    profile["query_expansion"] = 0.0

# ==========================================
# ФОРМИРУЕМ ЕДИНЫЙ СПИСОК ЗАПРОСОВ
# ==========================================

# Всегда сохраняем оригинальный запрос первым.
# Затем добавляем только уникальные варианты.

search_queries = normalize_query_variants(
    [query] + expanded_queries
)

search_queries = [
    q for q in search_queries
    if len(q.strip()) > 2
]

if not search_queries:
    search_queries = [query]

print("\nВарианты запроса:")
for q in expanded_queries:
    print("-", q)

print("\nИспользуемые запросы:")
for q in search_queries:
    print("-", q)

# ==========================================
# СБОР КАНДИДАТОВ
# ==========================================

candidates = {}

# ==========================================
# FAISS
# ==========================================

t0 = time.perf_counter()

add_faiss_candidates(
    search_queries,
    candidates,
    MIN_EMBED_SCORE
)

profile["faiss_search"] = time.perf_counter() - t0

# ==========================================
# BM25
# ==========================================

t0 = time.perf_counter()

for q in search_queries:
    add_bm25_candidates(
        q,
        candidates,
        BM25_K
    )

profile["bm25_search"] = time.perf_counter() - t0

profile["faiss_bm25_collect"] = (
    profile["faiss_search"]
    + profile["bm25_search"]
)

# ==========================================
# ПОДГОТОВКА РЕЗУЛЬТАТОВ
# ==========================================

results = list(candidates.values())

# Удаляем точные дубликаты
results = remove_duplicates(results)

# ==========================================
# SEMANTIC MERGE
# ==========================================

t0 = time.perf_counter()

results = semantic_merge(results)

profile["semantic_merge"] = (
    time.perf_counter() - t0
)

# ==========================================
# FALLBACK
# ==========================================

if not results and candidates:
    results = list(candidates.values())

if not results:
    print("\nНичего не найдено.")
    sys.exit(0)


# ==========================================
# PRE-RANK + QSIM
# ==========================================

qsim_base_query = query


# ------------------------------------------
# Нормализация исходных сигналов
# ------------------------------------------

embed_scores = normalize([
    r.get("embed_score", 0.0)
    for r in results
])

bm25_scores = normalize([
    r.get("bm25_score", 0.0)
    for r in results
])

hit_scores = normalize([
    r.get("hits", 0)
    for r in results
])


for i, r in enumerate(results):

    r["embed_score_norm"] = float(
        embed_scores[i]
    )

    r["bm25_score_norm"] = float(
        bm25_scores[i]
    )

    r["hits_score_norm"] = float(
        hit_scores[i]
    )



# ==========================================
# PRE-RANK
# ==========================================

results.sort(
    key=lambda x: x.get("embed_score", 0.0),
    reverse=True
)


# ==========================================
# Кандидаты для QSIM
# ==========================================

if len(results) > QSIM_K:

    qsim_candidates = results[:QSIM_K]

else:

    qsim_candidates = results


print(
    f"QSIM: "
    f"{len(qsim_candidates)} кандидатов",
    flush=True
)


# ==========================================
# QUERY SIMILARITY
# ==========================================

t0 = time.perf_counter()


# ------------------------------------------
# Embedding запроса
# ------------------------------------------

query_vec = get_query_embedding(
    qsim_base_query
).astype("float32")


# ------------------------------------------
# Нормализуем embedding запроса
# ------------------------------------------

query_norm = np.linalg.norm(
    query_vec
)

if query_norm > 0:

    query_vec = (
        query_vec
        /
        query_norm
    )


# ------------------------------------------
# Считаем cosine similarity
# ------------------------------------------

query_sim_scores = []


for r in qsim_candidates:

    vector = r.get("vector")


    # --------------------------------------
    # Если vector отсутствует —
    # создаём embedding текста
    # --------------------------------------

    if vector is None:

        text = r.get(
            "text",
            ""
        ).strip()


        if text:

            vector = embed_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype("float32")

            r["vector"] = vector


    # --------------------------------------
    # Similarity
    # --------------------------------------

    if vector is not None:

        vector = np.asarray(
            vector,
            dtype=np.float32
        )


        vector_norm = np.linalg.norm(
            vector
        )


        if (
            vector_norm > 0
            and query_norm > 0
        ):

            vector_normalized = (
                vector
                /
                vector_norm
            )


            sim = float(
                np.dot(
                    vector_normalized,
                    query_vec
                )
            )

        else:

            sim = 0.0


        query_sim_scores.append(
            sim
        )

    else:

        query_sim_scores.append(
            0.0
        )


query_sim_scores = np.asarray(
    query_sim_scores,
    dtype=np.float32
)


# ==========================================
# Нормализация QSIM
# ==========================================
#
# Cosine similarity:
#
# -1 = полностью противоположно
#  0 = нейтрально
# +1 = максимально похоже
#
# Переводим в диапазон 0..1:
#
# -1 -> 0
#  0 -> 0.5
# +1 -> 1
#
# ==========================================

query_sim_scores_n = np.clip(
    query_sim_scores,
    0.0,
    1.0
).astype(np.float32)


profile["query_similarity"] = (
    time.perf_counter()
    - t0
)


# ==========================================
# ДИАГНОСТИКА QSIM
# ==========================================

#print(
#    "RAW QSIM SCORES:",
#    query_sim_scores[:20],
#    flush=True
#)


#if len(query_sim_scores) > 0:

#    print(
#        f"QSIM MIN: "
#        f"{query_sim_scores.min():.6f}",
#        flush=True
#    )

#    print(
#        f"QSIM MAX: "
#        f"{query_sim_scores.max():.6f}",
#        flush=True
#    )

#    print(
#        f"QSIM MEAN: "
#        f"{query_sim_scores.mean():.6f}",
#        flush=True
#    )


# ==========================================
# FINAL SCORE = QSIM × ШТРАФ ЗА ДЛИНУ
# ==========================================

for i, r in enumerate(qsim_candidates):

    # --------------------------------------
    # QSIM
    # --------------------------------------

    r["query_sim_score"] = float(
        query_sim_scores_n[i]
    )


    # --------------------------------------
    # Длительность
    # --------------------------------------

    duration = max(
        0.0,
        float(r["end"])
        -
        float(r["start"])
    )


    # --------------------------------------
    # Мягкий штраф длинных результатов
    # --------------------------------------

    if duration > LENGTH_PENALTY_TARGET:

        penalty = 1.0 / (
            1.0
            +
            LENGTH_PENALTY_WEIGHT
            *
            (
                duration
                -
                LENGTH_PENALTY_TARGET
            )
        )

    else:

        penalty = 1.0


    # --------------------------------------
    # Сохраняем штраф для диагностики
    # --------------------------------------

    r["length_penalty"] = float(
        penalty
    )


    # --------------------------------------
    # ФИНАЛЬНЫЙ SCORE
    # --------------------------------------

    r["final_score"] = float(
        r["query_sim_score"]
        *
        penalty
    )

# ==========================================
# СОРТИРОВКА
# ==========================================

qsim_candidates.sort(
    key=lambda x: x["final_score"],
    reverse=True
)


# ==========================================
# ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ
# ==========================================

results = qsim_candidates[
    :FINAL_RESULTS
]


# ==========================================
# СОХРАНЕНИЕ QUERY CACHE
# ==========================================

save_query_cache()


# ==========================================
# PROFILE
# ==========================================

profile["total"] = (
    time.perf_counter()
    - start_total
)


# ==========================================
# ФОРМАТИРОВАНИЕ ВРЕМЕНИ
# ==========================================

def format_time(seconds):

    seconds = int(seconds)

    h = seconds // 3600

    m = (
        seconds % 3600
    ) // 60

    s = seconds % 60


    if h > 0:

        return (
            f"{h:02d}:"
            f"{m:02d}:"
            f"{s:02d}"
        )


    return (
        f"{m:02d}:"
        f"{s:02d}"
    )


# ==========================================
# СОЗДАНИЕ ССЫЛКИ
# ==========================================

def make_link(
    video_name,
    start
):

    base = video_name.replace(
        ".mp4",
        ""
    )

    return (
        f"{video_name}"
        f"#t={int(start)}"
    )


# ==========================================
# ВЫВОД РЕЗУЛЬТАТОВ
# ==========================================

print(
    "\nРезультаты:\n",
    flush=True
)


for i, r in enumerate(results):

    start = r["start"]

    end = r["end"]


    start_f = format_time(
        start
    )

    end_f = format_time(
        end
    )


    print(
        f"{i + 1}) "
        f"{r['video']}",
        flush=True
    )


    print(
        f"[TIME] "
        f"{start_f} - {end_f}",
        flush=True
    )


    print(
        f"final="
        f"{r['final_score']:.3f}"
        f" | qsim="
        f"{r['query_sim_score']:.3f}"
        f" | length_penalty="
        f"{r['length_penalty']:.3f}",
        flush=True
    )


    print(
        f"📝 "
        f"{r['text'][:2000]}",
        flush=True
    )


    print(
        flush=True
    )


# ==========================================
# AI ВЫБОР ЛУЧШЕГО РЕЗУЛЬТАТА
# ==========================================

ai_choice_idx = 0
ai_choice = results[0] if results else None

if USE_LLM_RESULT_SELECTION and results:

    try:

        idx, chosen = choose_best_result_with_llm(
            query,
            results,
            k=min(15, len(results))
        )

        if chosen is not None:

            ai_choice_idx = idx
            ai_choice = chosen

            print(
                "\nAI выбрал лучший результат:",
                flush=True
            )

            print(
                f"{ai_choice_idx + 1}) "
                f"{ai_choice['video']} "
                f"[{ai_choice['start']:.1f}-"
                f"{ai_choice['end']:.1f}] "
                f"score={ai_choice['final_score']:.3f} "
                f"| qsim={ai_choice['query_sim_score']:.3f}",
                flush=True
            )

            # ВАЖНО:
            # GUI должен получать этот индекс
            print(
                f"AI_CHOICE_INDEX: {ai_choice_idx + 1}",
                flush=True
            )

    except Exception as e:

        print(
            f"AI selection error: {e}",
            flush=True
        )

        # Если AI сломался — первым считается обычный результат
        ai_choice_idx = 0
        ai_choice = results[0] if results else None

        print(
            "AI_CHOICE_INDEX: 1",
            flush=True
        )

else:

    # AI выключен — первый результат является обычным первым
    if results:

        print(
            "AI_CHOICE_INDEX: 1",
            flush=True
        )


# ==========================================
# ПРОФИЛЬ ПОИСКА
# ==========================================

print(
    "\nПрофиль поиска:"
)


for key in [

    "query_expansion",

    "faiss_search",

    "bm25_search",

    "faiss_bm25_collect",

    "semantic_merge",

    "query_similarity",

    "total",

]:

    if key in profile:

        print(
            f" - {key}: "
            f"{profile[key]:.2f} сек"
        )

# -----------------------------
# ВЫРЕЗКА
# -----------------------------

# Если указали конкретный индекс через аргумент
if CUT_INDEX is not None:

    index = CUT_INDEX - 1

    if index < 0 or index >= len(results):

        print(
            "Неверный индекс клипа.",
            flush=True
        )

        sys.exit(0)

    selected_result = results[index]

    print(
        f"\n🎯 Выбран результат #{CUT_INDEX}",
        flush=True
    )

    cut_clip(
        selected_result,
        query=query,
        use_llm_refine=USE_LLM_REFINE,
        llm_pipeline=llm_pipeline,
        llm_generate=llm_generate,
        extract_json_value=extract_json_value,
        cut_mode="copy"
    )

    sys.exit(0)


# -----------------------------
# GUI MODE
# -----------------------------

if GUI_MODE:
    sys.exit(0)

if AUTO_CUT:

    if ai_choice is not None:

        print(
            "\n🤖 AI выбрал лучший результат:",
            flush=True
        )

        print(
            f"AI_CHOICE_INDEX: {ai_choice_idx + 1}",
            flush=True
        )

    else:

        print(
            "❌ AI не выбрал результат.",
            flush=True
        )

    # НИЧЕГО НЕ ВЫРЕЗАЕМ ЗДЕСЬ.
    # GUI получит результаты и сам вызовет cut_result().
    sys.exit(0)


# -----------------------------
# NO CUT
# -----------------------------

if NO_CUT:

    sys.exit(0)


