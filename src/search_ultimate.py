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
from pathlib import Path

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
# ПАРАМЕТРЫ
# -----------------------------

TOP_K = 150
BM25_K = 150
FINAL_RESULTS = 10

MERGE_DISTANCE = 7
SEMANTIC_MERGE_THRESHOLD = 0.65

MIN_EMBED_SCORE = 0.35
MIN_RERANK_SCORE = 0.18

MAX_RERANK_LENGTH = 1000

W_EMBED = 0.25
W_BM25 = 0.10
W_RERANK = 0.45
W_QUERY = 0.10
W_HITS = 0.10

LLM_MODEL = "models/llm"

# LLM используется для:
# 1) генерации вариантов запроса
# 2) выбора лучшего результата из уже найденных
#USE_LLM_RESULT_SELECTION = True
EXPAND_VARIANTS_K = 6

MIN_VARIANT_SIMILARITY = 0.53

# Обрезка
CUT_PAD_SECONDS = 1
CUT_MODE = "reencode"  # "reencode" точнее, чем "-c copy"


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
    "--llm-refine",
    action="store_true",
    help="Включить LLM уточнение границ клипа"
)
args, _unknown = parser.parse_known_args()
USE_LLM_QUERY_EXPANSION = args.llm_expand

NO_CUT = args.no_cut
CUT_INDEX = args.cut_index
AUTO_CUT = args.auto_cut
USE_LLM = not args.no_llm

USE_LLM_RESULT_SELECTION = args.llm_select
USE_LLM_REFINE = args.llm_refine

# -----------------------------
# МОДЕЛИ
# -----------------------------
print("поиск длится примерно 5 минут")
print("Загрузка embedding модели...")
embed_model = SentenceTransformer("models/paraphrase-multilingual-mpnet-base-v2")

print("Загрузка reranker...")
reranker = CrossEncoder("models/ms-marco-MiniLM-L-6-v2")

llm_pipeline = None

if USE_LLM and TRANSFORMERS_AVAILABLE and os.path.exists(LLM_MODEL):
    try:
        print("Попытка загрузить локальную LLM...")

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
else:
    print("LLM не используется")


# -----------------------------
# FAISS / META
# -----------------------------

print("Загрузка индекса...")

index = faiss.read_index("indexes/video.index")

if hasattr(index, "nprobe"):
    index.nprobe = 10

elif hasattr(index, "hnsw"):
    index.hnsw.efSearch = 100

with open("indexes/video.meta", "rb") as f:
    meta = pickle.load(f)

# -----------------------------
# КЭШ EMBEDDING
# -----------------------------

CACHE_FILE = "indexes/query_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf8") as f:
        query_cache = json.load(f)
else:
    query_cache = {}

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
    q = normalize_query_text(q)

    q = re.sub(r"\s+", " ", q.strip())

    print("NORMALIZED QUERY:", q)

    if q in query_cache:
        return np.array(query_cache[q]).astype("float32")

    vec = embed_model.encode(
        [q],
        convert_to_numpy=True
    )[0].astype("float32")

    # нормализация под cosine/IP
    faiss.normalize_L2(vec.reshape(1, -1))

    query_cache[q] = vec.tolist()

    with open(CACHE_FILE, "w", encoding="utf8") as f:
        json.dump(query_cache, f, ensure_ascii=False)

    return vec


# -----------------------------
# BM25
# -----------------------------

print("Подготовка BM25...")

corpus = []
corpus_meta = []
seen = set()

for m in meta:
    key = (m["video"], m["start"], m["end"], m.get("part", "full"))
    if key in seen:
        continue
    seen.add(key)

    corpus.append(re.findall(r"\w+", m["text"].lower()))
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
        return np.ones_like(arr)
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

    print("VARIANTS AFTER CLEAN:", clean)
    return clean


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
    segments.sort(key=lambda x: (x["video"], x["start"]))
    merged = []

    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]

        if seg["video"] == last["video"]:
            dist = seg["start"] - last["end"]

            if dist <= MERGE_DISTANCE:
                texts = [last["text"], seg["text"]]
                emb = embed_model.encode(texts).astype("float32")
                faiss.normalize_L2(emb)
                sim = float(np.dot(emb[0], emb[1]))

                if sim > SEMANTIC_MERGE_THRESHOLD:
                    last["end"] = max(last["end"], seg["end"])
                    last["text"] += " " + seg["text"]
                    last["embed_score"] = max(
                        last.get("embed_score", 0.0),
                        seg.get("embed_score", 0.0)
                    )
                    last["bm25_score"] = max(
                        last.get("bm25_score", 0.0),
                        seg.get("bm25_score", 0.0)
                    )
                    last["hits"] = last.get("hits", 0) + seg.get("hits", 0)

                    if "matched_queries" in last and "matched_queries" in seg:
                        last["matched_queries"].update(seg["matched_queries"])

                    if "source_segments" in last and "source_segments" in seg:
                        last["source_segments"].extend(seg["source_segments"])

                    continue

        merged.append(seg)

    return merged


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


def query_similarity_filter(base_query, variants, threshold=MIN_VARIANT_SIMILARITY):
    if not variants:
        return []

    base_query = normalize_query_text(base_query)

    base_vec = embed_model.encode([base_query])[0]
    base_norm = np.linalg.norm(base_vec) + 1e-12

    good = []
    for v in variants:
        if not isinstance(v, str):
            continue
        v = normalize_query_text(v)

        vec = embed_model.encode([v])[0]
        sim = float(np.dot(base_vec, vec) / (base_norm * (np.linalg.norm(vec) + 1e-12)))
        print(f"SIM: {sim:.3f} | {v}")

        if sim >= threshold:
            good.append(v)

    if good:
        return good
    print("AFTER SIM FILTER:", good)
    return [base_query]


# -----------------------------
# QUERY EXPANSION
# -----------------------------

def generate_variants_with_llm(query, n_variants=5):
    if not llm_pipeline:
        return None

    prompt = f"""
Ты переписываешь поисковый запрос.

Исходный запрос:
"{query}"

Сгенерируй {n_variants} вариантов ЭТОГО ЖЕ запроса.

Строгие правила:
- смысл должен остаться тем же
- нельзя расширять тему
- нельзя добавлять новые идеи
- можно только:
  - исправлять ошибки
  - переставлять слова
  - использовать очень близкие синонимы

Верни ТОЛЬКО список строк (каждый вариант с новой строки, БЕЗ нумерации).
"""

    out = llm_generate(prompt, max_new_tokens=120)
    print("RAW LLM OUTPUT:", out)

    if not out:
        return None

    text = out.strip()

    # --- 1. пробуем JSON ---
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            variants = parsed
        else:
            variants = None
    except:
        variants = None

    # --- 2. fallback: regex извлечение строк ---
    if not variants:
        lines = re.findall(r'"([^"]+)"', text)

        if not lines:
            # fallback: по строкам
            lines = text.split("\n")

        variants = []
        for line in lines:
            line = re.sub(r"^\d+[\).\s-]*", "", line)  # убрать "1. "
            line = line.strip(" -•\t\n\r\"")
            if line:
                variants.append(line)

    print("PARSED VARIANTS:", variants)

    # --- очистка ---
    clean = []
    seen = set()

    for v in variants:
        if not isinstance(v, str):
            continue

        v = re.sub(r"\s+", " ", v.strip())

        if len(v) < 3:
            continue

        if abs(len(v.split()) - len(query.split())) > 4:
            continue

        key = v.lower()
        if key in seen:
            continue

        seen.add(key)
        clean.append(v)

    print("FINAL CLEAN VARIANTS:", clean)

    return clean[:n_variants] if clean else None


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

        if len(candidate.split()) > 12:
            candidate = " ".join(candidate.split()[:12]) + "..."

        if candidate and candidate not in variants:
            variants.append(candidate)

    return variants[:top_k]


def expand_query(query):
    variants = generate_variants_with_llm(query, EXPAND_VARIANTS_K)

    def clean_list(lst):
        clean = []
        seen = set()

        banned_phrases = [
            "исходный запрос",
            "ты переписываешь",
            "сгенерируй",
            "варианты",
            "строгие правила",
            "верни только",
            "запрос:",
        ]

        for v in lst:
            if not isinstance(v, str):
                continue

            v = v.strip()

            # убрать мусор
            if len(v) < 3:
                continue

            if not any(c.isalnum() for c in v):
                continue

            low = v.lower()

            # фильтр prompt leakage
            if any(bad in low for bad in banned_phrases):
                continue


            # мусор с двоеточиями
            if low.endswith(":"):
                continue

            key = low

            if key in seen:
                continue

            seen.add(key)
            clean.append(v)

        return clean

    # ---------------- LLM ----------------
    if variants:
        if query not in variants:
            variants.insert(0, query)

        print("BEFORE CLEAN:", variants)

        variants = normalize_query_variants(variants)
        variants = clean_list(variants)

        print("AFTER CLEAN:", variants)

        filtered = query_similarity_filter(query, variants, MIN_VARIANT_SIMILARITY)
        filtered = clean_list(filtered)

        if not filtered:
            print("❌ SIM FILTER УДАЛИЛ ВСЁ → fallback на исходный запрос")
            return [query]

        print("FINAL VARIANTS FROM LLM:", filtered)
        return filtered

    # ---------------- FALLBACK ----------------
    print("⚠️ LLM не сработала → semantic fallback")

    variants = generate_variants_semantic(query, EXPAND_VARIANTS_K)
    variants = normalize_query_variants(variants)
    variants = clean_list(variants)

    variants = query_similarity_filter(query, variants, MIN_VARIANT_SIMILARITY)
    variants = clean_list(variants)

    if not variants:
        return [query]

    return variants


# -----------------------------
# СОБИРАТЕЛИ КАНДИДАТОВ
# -----------------------------

def add_faiss_candidates(search_query, candidates, min_score=MIN_EMBED_SCORE):
    q_vec = get_query_embedding(search_query).reshape(1, -1)
    assert q_vec.shape[1] == index.d
    print("QUERY VEC SHAPE:", q_vec.shape)

    scores, ids = index.search(q_vec, TOP_K)

    for i, idx in enumerate(ids[0]):
        if idx < 0:
            continue

        score = float(scores[0][i])
        if score < min_score:
            continue

        m = meta[idx]
        key = (m["video"], m["start"], m["end"])

        if key not in candidates:
            candidates[key] = {
                "video": m["video"],
                "start": m["start"],
                "end": m["end"],
                "text": m["text"],
                "embed_score": score,
                "bm25_score": 0.0,
                "hits": 0,
                "matched_queries": set(),
                "source_segments": [m]
            }
        else:
            candidates[key]["source_segments"].append(m)

        if search_query not in candidates[key]["matched_queries"]:
            candidates[key]["matched_queries"].add(search_query)
            candidates[key]["hits"] += 1

        candidates[key]["embed_score"] = max(
            candidates[key]["embed_score"], score
        )


def add_bm25_candidates(search_query, candidates, limit=BM25_K):
    if bm25 is None:
        return

    search_query = normalize_query_text(search_query)

    tokens = re.findall(r"\w+", search_query.lower())
    if not tokens:
        return

    scores = bm25.get_scores(tokens)
    if len(scores) == 0:
        return

    top_ids = np.argsort(scores)[::-1][:limit]

    for idx in top_ids:
        score = float(scores[idx])
        if score <= 0:
            continue

        m = corpus_meta[idx]
        key = (m["video"], m["start"], m["end"])

        if key not in candidates:
            candidates[key] = {
                "video": m["video"],
                "start": m["start"],
                "end": m["end"],
                "text": m["text"],
                "embed_score": 0.0,
                "bm25_score": score,
                "hits": 1,
                "matched_queries": {search_query},
                "source_segments": [m]
            }
        else:
            if search_query not in candidates[key]["matched_queries"]:
                candidates[key]["matched_queries"].add(search_query)
                candidates[key]["hits"] += 1

            candidates[key]["bm25_score"] = max(candidates[key]["bm25_score"], score)


# -----------------------------
# AI ВЫБОР ЛУЧШЕГО РЕЗУЛЬТАТА
# -----------------------------

def choose_best_result_with_llm(query, results, k=8):
    if not llm_pipeline or not results or not USE_LLM_RESULT_SELECTION:
        return 0, None

    top = results[:k]

    payload = []
    for i, r in enumerate(top, start=1):
        payload.append({
            "id": i,
            "video": r["video"],
            "start": round(float(r["start"]), 2),
            "end": round(float(r["end"]), 2),
            "score": round(float(r["final_score"]), 4),
            "rerank": round(float(r["rerank_score"]), 4),
            "qsim": round(float(r["query_sim_score"]), 4),
            "text": r["text"][:500]
        })

    prompt = f"""
Ты выбираешь лучший видеофрагмент для запроса.

Запрос:
"{query}"

ВАЖНО:
- короткие и точные фрагменты лучше длинных
- избегай лишнего контекста
- если есть 2 одинаково хороших варианта — выбирай более КОРОТКИЙ
- старайся выбирать короткие фрагменты
- обращай внимание в 1 очередь на поле text


Варианты:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Задача:
- выбрать ОДИН лучший вариант
- не менять тему
- не выдумывать новый вариант
- вернуть только JSON вида:
{{"id": 1}}

Если лучший вариант очевиден, выбери его по смыслу, а не только по score.
"""

    out = llm_generate(prompt, max_new_tokens=80)
    if not out:
        return 0, None

    data = extract_json_value(out)
    if isinstance(data, dict) and "id" in data:
        try:
            idx = int(data["id"]) - 1
            if 0 <= idx < len(top):
                return idx, top[idx]
        except Exception:
            pass

    if isinstance(data, int):
        idx = data - 1
        if 0 <= idx < len(top):
            return idx, top[idx]

    return 0, top[0] if top else None


# -----------------------------
# УТОЧНЕНИЕ ГРАНИЦ КЛИПА
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
        # если нет прямого пересечения, берем ближайшие сегменты вокруг окна
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


def refine_clip_with_llm(query, result):
    if not llm_pipeline:
        return result["start"], result["end"]

    text = result["text"]

    prompt = f"""
Запрос:
"{query}"

Текст видео:
"{text}"

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
        return result["start"], result["end"]

    try:
        sr = float(data.get("start_ratio", 0.0))
        er = float(data.get("end_ratio", 1.0))

        sr = max(0.0, min(1.0, sr))
        er = max(sr, min(1.0, er))

        duration = result["end"] - result["start"]

        new_start = result["start"] + duration * sr
        new_end = result["start"] + duration * er

        return new_start, new_end

    except Exception:
        return result["start"], result["end"]

def cut_clip(sel):
    start = float(sel["start"])
    end = float(sel["end"])
    video_path = os.path.join("videos", sel["video"])

    if not os.path.exists(video_path):
        print("Видео не найдено:", video_path)
        return

    # сначала AI режет смысл
    if USE_LLM_REFINE:
        start, end = refine_clip_with_llm(query, sel)

    # потом подгоняем под реальные тайминги
    start, end = refine_clip_bounds_with_transcript(sel["video"], start, end)

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
        # более точная обрезка, чем copy
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

    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Фрагмент сохранён:", out, flush=True)
    print("CLIP_PATH:", out, flush=True)


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

if USE_LLM_QUERY_EXPANSION:
    expanded_queries = expand_query(query)
else:
    expanded_queries = [query]

# ВАЖНО:
# никакого авто-выбора лучшего запроса нет.
# используем ВСЕ варианты запроса для поиска.
search_queries = [
    q for q in normalize_query_variants([query] + expanded_queries)
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


# -----------------------------
# ПОИСК
# -----------------------------

candidates = {}

for q in search_queries:
    add_faiss_candidates(q, candidates, MIN_EMBED_SCORE)
    add_bm25_candidates(q, candidates, BM25_K)

results = list(candidates.values())
results = remove_duplicates(results)
results = semantic_merge(results)

if not results and candidates:
    results = list(candidates.values())

if not results:
    print("\nНичего не найдено.")
    sys.exit(0)


# -----------------------------
# RERANK + ДОП. СИГНАЛЫ
# -----------------------------

rerank_base_query = query

pairs = []
texts_for_query_sim = []

for r in results:
    text = r["text"][:MAX_RERANK_LENGTH]
    pairs.append((rerank_base_query, text))
    texts_for_query_sim.append(text)

print("\nRerank...")

try:
    rerank_scores = reranker.predict(pairs)
except Exception as e:
    print("Reranker error:", e)
    rerank_scores = np.zeros(len(results), dtype=np.float32)

embed_scores = normalize([r.get("embed_score", 0.0) for r in results])
bm25_scores = normalize([r.get("bm25_score", 0.0) for r in results])
rerank_scores_n = normalize(rerank_scores)
hit_scores = normalize([r.get("hits", 0) for r in results])

query_vec = embed_model.encode([rerank_base_query]).astype("float32")
cand_vecs = embed_model.encode(texts_for_query_sim).astype("float32")
faiss.normalize_L2(query_vec)
faiss.normalize_L2(cand_vecs)
query_sim_scores = np.dot(cand_vecs, query_vec[0])
query_sim_scores_n = normalize(query_sim_scores)

for i, r in enumerate(results):
    r["rerank_score"] = float(rerank_scores_n[i])
    r["query_sim_score"] = float(query_sim_scores_n[i])

    score = (
        W_EMBED * float(embed_scores[i]) +
        W_BM25 * float(bm25_scores[i]) +
        W_RERANK * float(rerank_scores_n[i]) +
        W_QUERY * float(query_sim_scores_n[i]) +
        W_HITS * float(hit_scores[i])
    )

    r["final_score"] = float(score)


# for i, r in enumerate(results):
#     r["rerank_score"] = float(rerank_scores_n[i])
#     r["query_sim_score"] = float(query_sim_scores_n[i])

#     score = (
#         W_EMBED * float(embed_scores[i]) +
#         W_BM25 * float(bm25_scores[i]) +
#         W_RERANK * float(rerank_scores_n[i]) +
#         W_QUERY * float(query_sim_scores_n[i]) +
#         W_HITS * float(hit_scores[i])
#     )

#     r["final_score"] = float(score)

#     # ---------------- LENGTH PENALTY ----------------
#     duration = r["end"] - r["start"]

#     LENGTH_PENALTY_WEIGHT = 0.10
#     TARGET_DURATION = 500.0

#     length_penalty = np.exp(
#         -LENGTH_PENALTY_WEIGHT * max(0, duration - TARGET_DURATION)
#     )

#     r["final_score"] *= float(length_penalty)

#     # ---------------- HARD PENALTY ----------------
#     if duration > 2000:
#         r["final_score"] *= 0.5
        
# мягкий фильтр по rerank, но если все отсеяло — оставляем исходный список
filtered = [r for r in results if r["rerank_score"] >= MIN_RERANK_SCORE]
if filtered:
    results = filtered

results.sort(key=lambda x: x["final_score"], reverse=True)
results = results[:FINAL_RESULTS]


# -----------------------------
# ВЫВОД
# -----------------------------

def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def make_link(video_name, start):
    # если это локальный файл — просто путь
    # если YouTube — можно вставить нормальную ссылку
    base = video_name.replace(".mp4", "")
    return f"{video_name}#t={int(start)}"


print("\nРезультаты:\n", flush=True)

for i, r in enumerate(results):
    start = r["start"]
    end = r["end"]

    start_f = format_time(start)
    end_f = format_time(end)


    print(f"{i+1}) {r['video']}", flush=True)
    print(f"[TIME] {start_f} - {end_f}", flush=True)
    print(f"score={r['final_score']:.3f} | rerank={r['rerank_score']:.3f} | qsim={r['query_sim_score']:.3f}", flush=True)
    print(f"📝 {r['text'][:2000]}", flush=True)
    print(flush=True)

# -----------------------------
# AI ВЫБОР ЛУЧШЕГО РЕЗУЛЬТАТА
# -----------------------------

ai_choice_idx = 0
ai_choice = results[0] if results else None

if USE_LLM_RESULT_SELECTION and results:
    try:
        idx, chosen = choose_best_result_with_llm(query, results, k=min(8, len(results)))
        if chosen is not None:
            ai_choice_idx = idx
            ai_choice = chosen
            print("\nAI выбрал лучший результат:")
            print(
                f"{ai_choice_idx + 1}) {ai_choice['video']} "
                f"[{ai_choice['start']:.1f}-{ai_choice['end']:.1f}] "
                f"score={ai_choice['final_score']:.3f}"
            )
    except Exception as e:
        print("AI selection error:", e)


# -----------------------------
# ВЫРЕЗКА
# -----------------------------

def cut_clip_by_index(idx):
    if idx < 0 or idx >= len(results):
        print("Неверный индекс клипа.")
        return
    cut_clip(results[idx])


if CUT_INDEX is not None:
    cut_clip_by_index(CUT_INDEX - 1)
    sys.exit(0)

if AUTO_CUT:
    if ai_choice is not None:
        cut_clip(ai_choice)
    sys.exit(0)

if NO_CUT:
    sys.exit(0)

try:
    choice = input("Номер фрагмента: ").strip()
except EOFError:
    sys.exit(0)

if not choice.isdigit():
    sys.exit(0)

sel_idx = int(choice) - 1
if sel_idx < 0 or sel_idx >= len(results):
    print("Неверный номер.")
    sys.exit(0)

cut_clip(results[sel_idx])