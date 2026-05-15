# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import hashlib
import re
import shutil
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
import torch
import torch.nn.functional as F
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer

PRINT_LOCK = threading.Lock()

# --------------------------------------
# Carrega variáveis do .env
# --------------------------------------
DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(DOTENV_PATH)

# =========================
# Configuração
# =========================
NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
NOMIC_REPO_CACHE_DIR = HF_CACHE_DIR / "models--nomic-ai--nomic-embed-text-v1.5"
NOMIC_REF_PATH = NOMIC_REPO_CACHE_DIR / "refs" / "main"

_tokenizer = None
_model = None

BATCH = int(os.getenv("BATCH", 64))
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "base_pdfs")
CHUNK_SIZES = (128, 256, 512, 1024, 2048, 4096)
CHUNK_OVERLAP_PERCENTAGES = (0, 30, 60)
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN"))
TOP_K = int(os.getenv("TOP_K", 5))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
MODEL_SOURCE = (
    str(NOMIC_REPO_CACHE_DIR / "snapshots" / NOMIC_REF_PATH.read_text(encoding="utf-8").strip())
    if NOMIC_REF_PATH.exists()
    else NOMIC_MODEL
)
MODEL_LOAD_MODE = "local_snapshot" if MODEL_SOURCE != NOMIC_MODEL else "cache_only"

# garante expansão do ~ no Linux
DATA_DIR = Path(os.getenv("DATA_DIR") + "/diario_oficial_menor").expanduser()
NUMERO_EXPERIMENTO = int(os.getenv("NUMERO_EXPERIMENTO"))
LOGS_DIR = Path(f"{PROJECT_ROOT}/logs_experimento_{NUMERO_EXPERIMENTO}").expanduser()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_LOG_PATH = LOGS_DIR / f"master_experimento_{NUMERO_EXPERIMENTO}_{RUN_TIMESTAMP}.log"


def main() -> None:
    redirect_std_streams(MASTER_LOG_PATH)
    tasks = [
        (chunk_size, chunk_overlap_percentage)
        for chunk_size in CHUNK_SIZES
        for chunk_overlap_percentage in CHUNK_OVERLAP_PERCENTAGES
    ]
    extracted_pages = extract_pdf_pages(DATA_DIR)
    print(
        f"🚦 Execução sequencial iniciada para {len(tasks)} combinações.",
        flush=True,
    )

    completed = 0
    failed = 0

    for task_idx, (chunk_size, chunk_overlap_percentage) in enumerate(tasks, start=1):
        print(
            f"▶️ Iniciando combinação {task_idx}/{len(tasks)} | "
            f"chunk_size={chunk_size} | overlap_pct={chunk_overlap_percentage}",
            flush=True,
        )
        try:
            run_chunk_experiment(chunk_size, chunk_overlap_percentage, extracted_pages)
            completed += 1
            print(
                f"✅ Combinação concluída {task_idx}/{len(tasks)} | "
                f"chunk_size={chunk_size} | overlap_pct={chunk_overlap_percentage}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(
                f"❌ Falha na combinação {task_idx}/{len(tasks)} | "
                f"chunk_size={chunk_size} | overlap_pct={chunk_overlap_percentage} | erro={exc}",
                flush=True,
            )

    print(
        f"🏁 Execução sequencial finalizada | concluídas={completed} | falhas={failed}",
        flush=True,
    )


def redirect_std_streams(target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(target_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream


def make_logger(log_path: Path):
    def log(*args, sep: str = " ", end: str = "\n") -> None:
        message = sep.join(str(arg) for arg in args) + end
        with PRINT_LOCK:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(message)
                log_file.flush()

    return log


def get_nomic_components():
    global _tokenizer, _model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SOURCE,
            trust_remote_code=True,
            local_files_only=True,
        )

    if _model is None:
        _model = AutoModel.from_pretrained(
            MODEL_SOURCE,
            trust_remote_code=True,
            safe_serialization=True,
            local_files_only=True,
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


def calculate_chunk_overlap(chunk_size: int, chunk_overlap_percentage: int) -> int:
    if not 0 <= chunk_overlap_percentage < 100:
        raise ValueError(
            f"chunk_overlap_percentage deve estar entre 0 e 99. Recebido: {chunk_overlap_percentage}"
        )

    chunk_overlap = round(chunk_size * (chunk_overlap_percentage / 100))
    return min(chunk_overlap, chunk_size - 1)


def run_chunk_experiment(
    chunk_size: int,
    chunk_overlap_percentage: int,
    extracted_pages: list[dict],
) -> None:
    chunk_overlap = calculate_chunk_overlap(chunk_size, chunk_overlap_percentage)
    log_path = LOGS_DIR / (
        f"log_{chunk_size}_overlap_pct_{chunk_overlap_percentage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log = make_logger(log_path)

    chromadb_path = f"chromadb_chunk_size_{chunk_size}_overlap_pct_{chunk_overlap_percentage}"
    index_path = Path(f"{PROJECT_ROOT}/{chromadb_path}").expanduser()
    audit_path = Path(
        f"{PROJECT_ROOT}/chunks_audit_{chunk_size}_overlap_pct_{chunk_overlap_percentage}.json"
    ).expanduser()
    progress_dir = Path(f"{PROJECT_ROOT}/progress_experimento_{NUMERO_EXPERIMENTO}").expanduser()
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / (
        f"progress_chunk_{chunk_size}_overlap_pct_{chunk_overlap_percentage}.json"
    )

    log(f"[LOG] Saída também será gravada em: {log_path.resolve()}")

    if audit_path.exists() and index_path.exists():
        log("⏭️ Execução já concluída anteriormente. Pulando esta combinação.")
        write_progress(
            progress_path=progress_path,
            payload={
                "status": "completed",
                "chunk_size": chunk_size,
                "chunk_overlap_percentage": chunk_overlap_percentage,
                "index_path": str(index_path),
                "audit_path": str(audit_path),
                "resumed": False,
            },
        )
        return

    progress = load_progress(progress_path)
    resume_candidate = bool(progress and progress.get("status") in {"in_progress", "failed"})
    has_partial_state = index_path.exists() and any(index_path.iterdir()) if index_path.exists() else False
    resumed = resume_candidate and has_partial_state

    try:
        print_env_status(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_overlap_percentage=chunk_overlap_percentage,
            index_path=index_path,
            audit_path=audit_path,
            log=log,
        )
        if resume_candidate and not has_partial_state:
            log("ℹ️ Progresso anterior encontrado sem índice parcial disponível. Reiniciando do zero.")

        write_progress(
            progress_path=progress_path,
            payload={
                "status": "in_progress",
                "chunk_size": chunk_size,
                "chunk_overlap_percentage": chunk_overlap_percentage,
                "index_path": str(index_path),
                "audit_path": str(audit_path),
                "resumed": resumed,
            },
        )
        if resumed:
            log("🔁 Retomando execução a partir do progresso salvo.")

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
            progress=progress,
            log=log,
        )
        ingest_docs(
            collection,
            docs,
            batch_size=BATCH,
            progress_path=progress_path,
            chunk_size=chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            index_path=index_path,
            audit_path=audit_path,
            log=log,
        )

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
        write_progress(
            progress_path=progress_path,
            payload={
                "status": "completed",
                "chunk_size": chunk_size,
                "chunk_overlap_percentage": chunk_overlap_percentage,
                "index_path": str(index_path),
                "audit_path": str(audit_path),
                "completed_docs": len(docs),
                "collection_count": collection.count(),
                "resumed": resumed,
            },
        )
        log("✅ Combinação finalizada com sucesso.")
    except Exception as exc:
        log(f"❌ Falha na combinação: {type(exc).__name__}: {exc}")
        mark_failed_progress(
            progress_path=progress_path,
            chunk_size=chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            index_path=index_path,
            audit_path=audit_path,
            resumed=resumed,
            exc=exc,
        )
        raise

    
def mark_failed_progress(
    progress_path: Path,
    chunk_size: int,
    chunk_overlap_percentage: int,
    index_path: Path,
    audit_path: Path,
    resumed: bool,
    exc: Exception,
) -> None:
    payload = {
        "status": "failed",
        "chunk_size": chunk_size,
        "chunk_overlap_percentage": chunk_overlap_percentage,
        "index_path": str(index_path),
        "audit_path": str(audit_path),
        "resumed": resumed,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }

    if index_path.exists():
        payload["index_exists"] = True

    if audit_path.exists():
        payload["audit_exists"] = True

    write_progress(progress_path=progress_path, payload=payload)

# =========================
# Utilidades
# =========================
def print_env_status(
    chunk_size: int,
    chunk_overlap: int,
    chunk_overlap_percentage: int,
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
    log(f"{'CHUNK_OVERLAP_%'.ljust(20)} -> ✅ {chunk_overlap_percentage}")
    log(f"{'CHUNK_OVERLAP'.ljust(20)} -> ✅ {chunk_overlap}")
    log(f"{'INDEX_PATH'.ljust(20)} -> ✅ {index_path}")
    log(f"{'AUDIT_PATH'.ljust(20)} -> ✅ {audit_path}")
    log(f"{'DEVICE'.ljust(20)} -> ✅ {DEVICE}")
    log(f"{'MODEL_LOAD_MODE'.ljust(20)} -> ✅ {MODEL_LOAD_MODE}")


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
    if not text:
        return []

    tokenizer, model = get_nomic_components()
    model.eval()

    text = f"{prefix} {text}"

    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    if device == "cuda":
        model = model.to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = F.normalize(embeddings, p=2, dim=1)

    result = embeddings[0].detach().cpu().tolist()

    if device == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()

    return result


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
                    "id": build_doc_id(
                        source_pdf=source_pdf,
                        page_number=page_number,
                        chunk_idx=chunk_idx,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_text=chunk,
                    ),
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

def create_collection(index_path: Path, collection_name: str, progress: dict, log):
    index_dir = Path(index_path)
    index_dir.mkdir(parents=True, exist_ok=True)

    progress_status = (progress or {}).get("status")
    if progress_status != "in_progress" and index_dir.exists():
        shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

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

def ingest_docs(
    collection,
    docs: list[dict],
    batch_size: int,
    progress_path: Path,
    chunk_size: int,
    chunk_overlap_percentage: int,
    index_path: Path,
    audit_path: Path,
    log,
) -> None:
    if not docs:
        raise RuntimeError(
            "Não há chunks (docs) para inserir. Verifique DATA_DIR, o chunking e a geração de embeddings."
        )

    total_batches = (len(docs) + batch_size - 1) // batch_size
    already_inserted = min(collection.count(), len(docs))

    if already_inserted:
        skipped_batches = (already_inserted + batch_size - 1) // batch_size
        log(
            f"🔁 Retomando ingestão com {already_inserted} documentos já persistidos "
            f"({skipped_batches}/{total_batches} lotes)."
        )

    for batch_idx, i in enumerate(range(0, len(docs), batch_size), start=1):
        if i < already_inserted:
            continue

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
        write_progress(
            progress_path=progress_path,
            payload={
                "status": "in_progress",
                "chunk_size": chunk_size,
                "chunk_overlap_percentage": chunk_overlap_percentage,
                "index_path": str(index_path),
                "audit_path": str(audit_path),
                "completed_docs": min(i + len(batch), len(docs)),
                "collection_count": collection.count(),
                "completed_batches": batch_idx,
                "total_batches": total_batches,
            },
        )

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


def build_doc_id(
    source_pdf: str,
    page_number: int,
    chunk_idx: int,
    chunk_size: int,
    chunk_overlap: int,
    chunk_text: str,
) -> str:
    text_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()
    stable_key = (
        f"{source_pdf}|{page_number}|{chunk_idx}|{chunk_size}|{chunk_overlap}|{text_hash}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))


def load_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {}

    with open(progress_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_progress(progress_path: Path, payload: dict) -> None:
    payload = {
        **payload,
        "updated_at": datetime.now().isoformat(),
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
