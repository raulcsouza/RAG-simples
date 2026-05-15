# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import multiprocessing as mp
import re
import shutil
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chromadb
import requests
import torch
import torch.nn.functional as F
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer

from datetime import datetime

PRINT_LOCK = threading.Lock()
MAX_WORKERS = 6

# --------------------------------------
# Carrega variáveis do .env
# --------------------------------------
load_dotenv()

# =========================
# Configuração
# =========================
NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"

_tokenizer = None
_model = None

BATCH = int(os.getenv("BATCH", 64))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "base_pdfs")

CHUNK_SIZES = (128,256,512,1024,2048,4096)
CHUNK_OVERLAPS = (196)


MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
TOP_K = int(os.getenv("TOP_K", 5))

# garante expansão do ~ no Linux
DATA_DIR = Path(os.getenv("DATA_DIR") + "/diario_oficial_menor").expanduser()


NUMERO_EXPERIMENTO = int(os.getenv("NUMERO_EXPERIMENTO"))
LOGS_DIR = Path(f"{PROJECT_ROOT}/logs_experimento_{NUMERO_EXPERIMENTO}").expanduser()
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    tasks = [(chunk, overlap) for chunk in CHUNK_SIZES for overlap in CHUNK_OVERLAPS]
    extracted_pages = extract_pdf_pages(DATA_DIR)

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(run_chunk_experiment, chunk_size, chunk_overlap, extracted_pages)
            for chunk_size, chunk_overlap in tasks
        ]

        for future in as_completed(futures):
            future.result()


def make_logger(log_path: Path):
    def log(*args, sep: str = " ", end: str = "\n") -> None:
        message = sep.join(str(arg) for arg in args) + end
        with PRINT_LOCK:
            print(*args, sep=sep, end=end, flush=True)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(message)
                log_file.flush()

    return log


def get_nomic_components():
    global _tokenizer, _model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(NOMIC_MODEL, trust_remote_code=True)

    if _model is None:
        _model = AutoModel.from_pretrained(
            NOMIC_MODEL,
            trust_remote_code=True,
            safe_serialization=True,
        )
        _model.eval()

    return _tokenizer, _model


def extract_pdf_pages(data_dir: Path) -> list[dict]:
    pdf_paths = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    print(f"📄 Extraindo texto uma vez de {len(pdf_paths)} PDFs em {data_dir}", flush=True)

    extracted_pages = []

    for pdf_idx, pdf_path in enumerate(pdf_paths, start=1):
        print(f"📖 Extraindo PDF {pdf_idx}/{len(pdf_paths)}: {pdf_path.name}", flush=True)
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:
            print(f"⚠️ Falha ao abrir {pdf_path.name}: {e}", flush=True)
            continue

        page_count = 0
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            extracted_pages.append(
                {
                    "source_pdf": pdf_path.name,
                    "page": page_idx,
                    "text": page_text,
                }
            )
            page_count += 1

        print(
            f"✅ Extração concluída {pdf_idx}/{len(pdf_paths)}: {pdf_path.name} | páginas={page_count}",
            flush=True,
        )

    print(f"✅ Total de páginas extraídas: {len(extracted_pages)}", flush=True)
    return extracted_pages


def run_chunk_experiment(chunk_size: int, chunk_overlap: int, extracted_pages: list[dict]) -> None:
    log_path = LOGS_DIR / (
        f"log_{chunk_size}_overlap_{chunk_overlap}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log = make_logger(log_path)

    chromadb_path = f"chromadb_chunk_size_{chunk_size}_overlap_{chunk_overlap}"
    index_path = Path(f"{PROJECT_ROOT}/{chromadb_path}").expanduser()
    audit_path = Path(
        f"{PROJECT_ROOT}/chunks_audit_{chunk_size}_overlap_{chunk_overlap}.json"
    ).expanduser()

    log(f"[LOG] Saída também será gravada em: {log_path.resolve()}")

    print_env_status(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        index_path=index_path,
        audit_path=audit_path,
        log=log,
    )
    docs = build_docs(
        extracted_pages=extracted_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_len=MIN_CHUNK_LEN,
        log=log,
    )
    client, collection = create_collection(
        index_path=index_path,
        collection_name=COLLECTION_NAME,
        log=log,
    )
    ingest_docs(collection, docs, batch_size=BATCH, log=log)

    query = "Qual é o valor total do contrato de pneus?"
    hits = chroma_retrieve(collection, query, top_k=TOP_K)

    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        log(
            f"\n=== Hit {i} | dist={h['distance']:.4f} | {m['source_pdf']} | pág {m['page']} | chunk {m['chunk']} ==="
        )
        log(h["text"][:900], "...")

    save_audit(docs, audit_path=audit_path, log=log)
    inspect_first_item(collection, log=log)

# =========================
# Utilidades
# =========================
def print_env_status(
    chunk_size: int,
    chunk_overlap: int,
    index_path: Path,
    audit_path: Path,
    log,
) -> None:
    log("Diretório atual:", os.getcwd())
    dotenv_path = find_dotenv()
    log("Arquivo .env encontrado:", dotenv_path)

    vars_env = [
        "DATA_DIR",
        "PROJECT_ROOT",
        "COLLECTION_NAME",
        "NUMERO_EXPERIMENTO",
        "MIN_CHUNK_LEN",
        "OLLAMA_BASE_URL",
        "EMBED_MODEL",
        "TOP_K",
        "BATCH",
    ]

    log("\n===== TESTE DAS VARIÁVEIS DE AMBIENTE =====\n", end="")
    for v in vars_env:
        value = os.getenv(v)
        if value is None:
            log(f"{v:20} -> ❌ NÃO CARREGADA (None)")
        else:
            log(f"{v:20} -> ✅ {value}")

    log(f"{'CHUNK_SIZE'.ljust(20)} -> ✅ {chunk_size}")
    log(f"{'CHUNK_OVERLAP'.ljust(20)} -> ✅ {chunk_overlap}")
    log(f"{'INDEX_PATH'.ljust(20)} -> ✅ {index_path}")
    log(f"{'AUDIT_PATH'.ljust(20)} -> ✅ {audit_path}")
    log(f"{'DEVICE'.ljust(20)} -> ✅ {DEVICE}")


# =========================
# Embeddings via Ollama
# =========================
def ollama_embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Gera embedding via Ollama /api/embeddings."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["embedding"]


# =========================
# Pooling
# =========================
def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


# =========================
# Hugging Face Embedding
# =========================
def hf_nomic_embed(
    text: str,
    prefix: str = "search_document:",
    device: str = DEVICE,
) -> list[float]:
    """
    Gera embedding usando Nomic local do Hugging Face.

    Parâmetros:
      - text: texto de entrada
      - prefix:
          * 'search_document:' para indexação
          * 'search_query:' para consulta
      - device: 'cpu', 'cuda' ou 'auto'
    """
    if not text:
        return []

    tokenizer, model = get_nomic_components()
    model = model.to(device)
    model.eval()

    text = f"{prefix} {text}"

    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings[0].detach().cpu().tolist()


# =========================
# Limpeza e chunking
# =========================
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s



def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: int,
    min_chunk_len: int,
):
    """Chunking por tokens usando tokenizer."""
    text = clean_text(text)
    if not text:
        return []

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) == 0:
        return []

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk_tokens = tokens[start:end]
        chunk_str = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()

        if len(chunk_str) >= min_chunk_len:
            chunks.append(chunk_str)

        if end == len(tokens):
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            break

        start = next_start

    return chunks

def build_docs(
    extracted_pages: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_len: int,
    log,
) -> list[dict]:
    tokenizer, _ = get_nomic_components()
    log(f"📄 Páginas extraídas recebidas: {len(extracted_pages)}")

    docs = []
    current_pdf = None

    for page_idx_global, page_data in enumerate(extracted_pages, start=1):
        source_pdf = page_data["source_pdf"]
        page_number = page_data["page"]
        page_text = page_data["text"]

        if source_pdf != current_pdf:
            current_pdf = source_pdf
            log(f"📖 Processando texto extraído: {source_pdf}")

        page_chunks = chunk_text(
            text=page_text,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            min_chunk_len=min_chunk_len,
        )

        if not page_chunks:
            continue

        page_embeddings = [
            hf_nomic_embed(chunk, prefix="search_document:")
            for chunk in page_chunks
        ]

        for chunk_idx, (chunk, emb) in enumerate(zip(page_chunks, page_embeddings), start=1):
            docs.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "embedding": emb,
                    "metadata": {
                        "source_pdf": source_pdf,
                        "page": page_number,
                        "chunk": chunk_idx,
                        "chunk_tokens": len(tokenizer.encode(chunk, add_special_tokens=False)),
                    },
                }
            )

        if page_idx_global % 50 == 0:
            log(f"📄 Páginas processadas: {page_idx_global}/{len(extracted_pages)}")

    log(f"✅ Total de chunks gerados: {len(docs)}")
    return docs



def create_collection(index_path: Path, collection_name: str, log):
    index_dir = Path(index_path)
    index_dir.mkdir(parents=True, exist_ok=True)

    if index_dir.exists():
        shutil.rmtree(index_dir)

    settings = Settings(_env_file=None, anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=str(index_dir), settings=settings)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    log("📦 Coleção:", collection_name)
    log("📁 Persistência em:", index_dir)
    log("🔢 Itens atuais na coleção:", collection.count())

    return client, collection



def ingest_docs(collection, docs: list[dict], batch_size: int, log) -> None:
    if not docs:
        raise RuntimeError(
            "Não há chunks (docs) para inserir. Verifique DATA_DIR, o chunking e a geração de embeddings."
        )

    total_batches = (len(docs) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(docs), batch_size), start=1):
        batch = docs[i : i + batch_size]

        ids = [d["id"] for d in batch]
        texts = [d["text"] for d in batch]
        metas = [d["metadata"] for d in batch]
        embeds = [d["embedding"] for d in batch]

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embeds,
        )

        log(f"📥 Lote inserido {batch_idx}/{total_batches} | tamanho={len(batch)}")

    log(f"✅ Ingestão concluída. Total na coleção: {collection.count()}")



def chroma_retrieve(collection, query: str, top_k: int):
    q_emb = hf_nomic_embed(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"distance": float(dist), "metadata": meta, "text": doc})
    return hits



def save_audit(docs: list[dict], audit_path: Path, log) -> None:
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"id": d["id"], **d["metadata"], "text_preview": d["text"]} for d in docs],
            f,
            ensure_ascii=False,
            indent=2,
        )
    log("📝 Auditoria salva em:", audit_path)



def inspect_first_item(collection, log) -> None:
    data = collection.get(limit=1, include=["documents", "metadatas"])

    log("ID:", data["ids"][0])
    log("\nTexto do chunk:\n", end="")
    log(data["documents"][0][:500])

    log("\nMetadata:")
    log(data["metadatas"][0])

    first_id = data["ids"][0]
    emb = collection.get(ids=[first_id], include=["embeddings"])["embeddings"][0]

    log("\nDimensão do embedding:", len(emb))
    log("Primeiros valores:", emb[:10])


if __name__ == "__main__":
    main()
