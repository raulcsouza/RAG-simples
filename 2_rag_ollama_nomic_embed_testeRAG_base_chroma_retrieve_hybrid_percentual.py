# -*- coding: utf-8 -*-

# ==============================================================================
# Este programa pode ser usado em produção por possibilitar execução remota e
# permite que vários experimentos sejam feito apenas alterando parametros no 
# arquivo env.
# ==============================================================================

# =========================================================
# CONFIGURAÇÕES GERAIS
# =========================================================

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

# ==============================================================================
# Importações das bibliotecas
# ==============================================================================
import os
import time
import json
import random
import math
import numpy as np
import re
import requests
import chromadb
import torch
import torch.nn.functional as F

from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from chromadb.config import Settings
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModel

from huggingface_hub import login
import transformers

# --------------------------------------
# Carrega variáveis do .env
# --------------------------------------
load_dotenv(find_dotenv())

login(os.getenv("TOKEN_HF"))

BATCH = int(os.getenv("BATCH", 64))
TIMEOUT = int(os.getenv("TIMEOUT", 120))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "base_pdfs")

# Root do projeto
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser()

# garante expansão do ~ no Linux
DATA_DIR = Path(os.getenv("DATA_DIR")).expanduser()

HF_MODEL_ID = os.getenv("HF_MODEL_ID","Qwen/Qwen2.5-3B-Instruct")
# HF_MODEL_ID = os.getenv("HF_MODEL_ID", "microsoft/phi-3-mini-4k-instruct")
# HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TORCH_DTYPE = torch.bfloat16

# pipeline HF lazy-load
_hf_pipeline = None

# --------------------------------------
# Transformers / Embeddings locais
# --------------------------------------
NOMIC_MODEL = os.getenv("NOMIC_MODEL", "nomic-ai/nomic-embed-text-v1.5")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = AutoTokenizer.from_pretrained(
    NOMIC_MODEL,
    trust_remote_code=True
)

if torch.cuda.is_available():
    DEVICE = "cuda"
    HF_TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DEVICE = "cpu"
    HF_TORCH_DTYPE = torch.float32

_model = AutoModel.from_pretrained(
    NOMIC_MODEL,
    trust_remote_code=True,
    safe_serialization=True,
    dtype=HF_TORCH_DTYPE
).to(DEVICE)

_model.eval()

# --------------------------------------
# Embedding / Ollama
# --------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"

# --------------------------------------
# Retrieval
# --------------------------------------
TOP_KS = (1,3,5,10)
RETRIEVAL_MODES = ("bm25","embedding","hybrid")

# --------------------------------------
# Experimento
# --------------------------------------
NUMERO_EXPERIMENTO = int(os.getenv("NUMERO_EXPERIMENTO"))
BASE_EXPERIMENTOS = ("b_extrato","a_objeto")

# --------------------------------------
# Chunking
# --------------------------------------
CHUNK_SIZES = (128,256,512,1024,2048,4096)
CHUNK_OVERLAP_PERCENTAGES = (0,30,60)

MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", 200))

LOGS_ROOT_DIR = Path(f"{PROJECT_ROOT}/logs_experimento_{NUMERO_EXPERIMENTO}").expanduser()
LOGS_DIR = LOGS_ROOT_DIR
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_word_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")


def calculate_chunk_overlap(chunk_size: int, chunk_overlap_percentage: int) -> int:
    if not 0 <= chunk_overlap_percentage < 100:
        raise ValueError(
            f"chunk_overlap_percentage deve estar entre 0 e 99. Recebido: {chunk_overlap_percentage}"
        )

    chunk_overlap = round(chunk_size * (chunk_overlap_percentage / 100))
    return min(chunk_overlap, chunk_size - 1)

def ollama_embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Gera embedding via Ollama /api/embeddings."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["embedding"]


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def _resolve_device(device: str = "auto") -> str:
    device = (device or "auto").lower().strip()

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device not in {"cpu", "cuda"}:
        raise ValueError("device deve ser 'cpu', 'cuda' ou 'auto'.")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitado, mas não está disponível neste ambiente.")

    return device


def hf_nomic_embed(
    text: str,
    prefix: str = "search_document:",
    device: str = "auto"
) -> list[float]:
    if not text:
        return []

    resolved_device = _resolve_device(device)
    model = _model.to(resolved_device)
    model.eval()

    text = f"{prefix} {text}"
    encoded_input = _tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    encoded_input = {k: v.to(resolved_device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings[0].detach().cpu().tolist()


def tokenize(text: str):
    return _word_re.findall((text or "").lower())


def chroma_retrieve_hybrid(
    query: str,
    docs,
    metas,
    emb_matrix,
    top_k: int,
    bm25_k: int = 30,
    emb_k: int = 30,
    rrf_k: int = 60,
    embed_fn=hf_nomic_embed,
    mode: str = "hybrid"
):
    mode = mode.lower().strip()
    valid_modes = {"bm25", "embedding", "hybrid"}
    if mode not in valid_modes:
        raise ValueError(f"mode deve ser um de {valid_modes}, recebido: {mode}")

    if mode in {"embedding", "hybrid"} and embed_fn is None:
        raise ValueError("Para mode='embedding' ou mode='hybrid', você deve fornecer embed_fn.")

    if not docs or not metas:
        raise RuntimeError("docs e metas precisam estar carregados em memória.")

    if len(docs) != len(metas):
        raise RuntimeError("docs e metas não estão alinhados.")

    if mode in {"embedding", "hybrid"}:
        if emb_matrix is None:
            raise RuntimeError("emb_matrix precisa estar carregado em memória.")
        if len(docs) != len(emb_matrix):
            raise RuntimeError("docs, metas e emb_matrix não estão alinhados.")

    q_tokens = tokenize(query)
    q_set = set(q_tokens)
    if not q_tokens:
        return []

    def passes_filter(doc: str) -> bool:
        return len(q_set.intersection(tokenize(doc))) > 0

    def key_of(text, meta):
        meta = meta or {}
        return (
            text,
            meta.get("source_pdf"),
            meta.get("page"),
            meta.get("chunk"),
            meta.get("id"),
        )

    def make_item(doc, meta, bm25_score=None, dist=None, sim=None, rrf_score=None):
        item = {
            "text": doc,
            "metadata": meta,
            "bm25_score": bm25_score,
            "distance": dist,
            "similarity": sim,
        }
        if rrf_score is not None:
            item["rrf_score"] = rrf_score
        return item

    bm25_rank = []
    if mode in {"bm25", "hybrid"}:
        tokenized_docs = [tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_rank = sorted(
            [
                (i, float(score))
                for i, score in enumerate(bm25_scores)
                if score > 0 and passes_filter(docs[i])
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:bm25_k]

    emb_rank = []
    if mode in {"embedding", "hybrid"}:
        q_emb = np.array(embed_fn(query), dtype=np.float32)
        if q_emb.ndim != 1:
            raise RuntimeError("O embedding da consulta não é um vetor 1D.")
        if emb_matrix.shape[1] != len(q_emb):
            raise RuntimeError(
                f"Dimensão incompatível: emb_matrix tem {emb_matrix.shape[1]} colunas e q_emb tem {len(q_emb)}."
            )

        q_norm = np.linalg.norm(q_emb)
        doc_norms = np.linalg.norm(emb_matrix, axis=1)
        sims = np.full(len(docs), -1.0, dtype=np.float32)

        if q_norm > 0:
            valid = doc_norms > 0
            sims[valid] = (emb_matrix[valid] @ q_emb) / (doc_norms[valid] * q_norm)

        top_idx = np.argsort(sims)[::-1][:emb_k]
        for idx in top_idx:
            doc = docs[idx]
            meta = metas[idx]
            sim = float(sims[idx])
            if passes_filter(doc):
                emb_rank.append((key_of(doc, meta), 1.0 - sim, sim, doc, meta))

        emb_rank.sort(key=lambda x: x[1])

    if mode == "bm25":
        return [
            make_item(doc=docs[idx], meta=metas[idx], bm25_score=bm25_score)
            for idx, bm25_score in bm25_rank[:top_k]
        ]

    if mode == "embedding":
        return [
            make_item(doc=doc, meta=meta, dist=float(dist), sim=float(sim))
            for _, dist, sim, doc, meta in emb_rank[:top_k]
        ]

    scores_rrf = {}
    payload = {}

    for rank_pos, (idx, bm25_score) in enumerate(bm25_rank, start=1):
        doc = docs[idx]
        meta = metas[idx]
        key = key_of(doc, meta)
        scores_rrf[key] = scores_rrf.get(key, 0.0) + 1.0 / (rrf_k + rank_pos)
        payload[key] = make_item(doc=doc, meta=meta, bm25_score=bm25_score)

    for rank_pos, (key, dist, sim, doc, meta) in enumerate(emb_rank, start=1):
        scores_rrf[key] = scores_rrf.get(key, 0.0) + 1.0 / (rrf_k + rank_pos)
        if key not in payload:
            payload[key] = make_item(doc=doc, meta=meta, dist=float(dist), sim=float(sim))
        else:
            payload[key]["distance"] = float(dist)
            payload[key]["similarity"] = float(sim)

    final = sorted(scores_rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    hits = []
    for key, score in final:
        item = payload[key]
        item["rrf_score"] = float(score)
        hits.append(item)
    return hits


def ollama_generate(
    prompt: str,
    model: str = "llama3.1",
    timeout_s: int = 600,
    retries: int = 2,
    num_ctx: int = 8192,
    num_predict: int = 256
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
            r.raise_for_status()
            return (r.json().get("response") or "").strip()
        except Exception as e:
            last_err = e
            print(f"[Tentativa {attempt+1}] Erro Ollama: {e}")
            time.sleep(2 * (attempt + 1))

    return f"ERRO_OLLAMA: {last_err}"


def get_hf_pipeline(model_id: str = HF_MODEL_ID):
    global _hf_pipeline

    current_model_name = None
    if _hf_pipeline is not None:
        current_model_name = getattr(getattr(_hf_pipeline, "model", None), "name_or_path", None)

    if _hf_pipeline is None or current_model_name != model_id:
        _hf_pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": HF_TORCH_DTYPE},
            device_map="auto",
        )

    return _hf_pipeline


def hf_generate_chat(
    user_prompt: str,
    system_prompt: str = "Você é um assistente útil e objetivo.",
    model_id: str = HF_MODEL_ID,
    max_new_tokens: int = 96,
    timeout_s: int = 600,
    retries: int = 2,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    pipe = get_hf_pipeline(model_id=model_id)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None
    for attempt in range(retries + 1):
        try:
            prompt = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            outputs = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                return_full_text=False,
                pad_token_id=pipe.tokenizer.eos_token_id,
                eos_token_id=pipe.tokenizer.eos_token_id,
            )
            return (outputs[0].get("generated_text") or "").strip()
        except Exception as e:
            last_err = e
            print(f"[Tentativa {attempt+1}] Erro HF: {e}")
            time.sleep(2 * (attempt + 1))

    return f"ERRO_HF: {last_err}"


def generate_text(
    prompt: str,
    backend: str = "ollama_generate",
    ollama_model: str = "llama3.1",
    num_ctx: int = 8192,
    num_predict: int = 256,
    hf_model_id: str = HF_MODEL_ID,
    system_prompt: str = "Você é um assistente útil e objetivo.",
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout_s: int = 600,
    retries: int = 2,
) -> str:
    backend = (backend or "").strip().lower()

    if backend == "ollama_generate":
        return ollama_generate(
            prompt=prompt,
            model=ollama_model,
            timeout_s=timeout_s,
            retries=retries,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )

    if backend == "hf_generate_chat":
        return hf_generate_chat(
            user_prompt=prompt,
            system_prompt=system_prompt,
            model_id=hf_model_id,
            max_new_tokens=max_new_tokens,
            timeout_s=timeout_s,
            retries=retries,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    raise ValueError("backend inválido. Use 'ollama_generate' ou 'hf_generate_chat'.")


def build_prompt(
    question: str,
    retrieved,
    max_chars: int | None = 12000,
    forbid_trailing_symbol: bool = False,
):
    partes = []
    total = 0

    for item in retrieved:
        texto = item.get("text", "")
        if not texto:
            continue

        if max_chars is not None and total + len(texto) > max_chars:
            restante = max_chars - total
            if restante > 0:
                partes.append(texto[:restante])
            break

        partes.append(texto)
        total += len(texto)

    contexto = "\n\n".join(partes)
    trailing_instruction = ""
    if forbid_trailing_symbol:
        trailing_instruction = "\n- Não coloque ponto final ou qualquer outro simbolo no valor."

    prompt = f"""Você é um assistente de RAG para responder perguntas usando SOMENTE os trechos fornecidos.
Se a resposta não estiver claramente suportada pelos trechos, diga que não encontrou no contexto.

CONTEXTO (trechos recuperados):
{contexto}

PERGUNTA:
{question}

INSTRUÇÕES DE SAÍDA:
- Responda de forma objetiva.
- Se possível, retorne apenas o valor pedido (exemplo: "R$ 00.000,00").{trailing_instruction}

RESPOSTA:
"""
    return prompt


def answer(
    question: str,
    docs,
    metas,
    emb_matrix,
    top_k: int,
    backend: str = "ollama_generate",
    retrieval_mode: str = "hybrid",
    embed_fn=None,
    ollama_model: str = "llama3.1",
    num_ctx: int = 8192,
    num_predict: int = 256,
    hf_model_id: str = HF_MODEL_ID,
    system_prompt: str = (
        "Você é um assistente de RAG. "
        "Responda apenas com base no contexto fornecido. "
        "Se a resposta não estiver no contexto, diga que não encontrou no contexto."
    ),
    max_new_tokens: int = 96,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout_s: int = 600,
    retries: int = 2,
    prompt_path: str = "prompt.txt",
    max_prompt_chars: int | None = 12000,
    forbid_trailing_symbol: bool = False,
):
    valid_backends = {"ollama_generate", "hf_generate_chat"}
    if backend not in valid_backends:
        raise ValueError(f"backend inválido: {backend}. Use um de {valid_backends}")

    retrieved = chroma_retrieve_hybrid(
        question,
        docs=docs,
        metas=metas,
        emb_matrix=emb_matrix,
        top_k=top_k,
        mode=retrieval_mode,
        embed_fn=embed_fn,
    )

    if not retrieved:
        return {
            "question": question,
            "answer": "Não encontrei contexto recuperado para responder.",
            "retrieved": [],
            "prompt": "",
            "backend": backend,
            "retrieval_mode": retrieval_mode,
        }

    prompt = build_prompt(
        question,
        retrieved,
        max_chars=max_prompt_chars,
        forbid_trailing_symbol=forbid_trailing_symbol,
    )

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    generate_kwargs = {
        "prompt": prompt,
        "backend": backend,
        "timeout_s": timeout_s,
        "retries": retries,
    }

    if backend == "ollama_generate":
        generate_kwargs.update({
            "ollama_model": ollama_model,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        })
    else:
        generate_kwargs.update({
            "hf_model_id": hf_model_id,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        })

    resp = generate_text(**generate_kwargs)

    return {
        "question": question,
        "answer": (resp or "").strip(),
        "retrieved": retrieved,
        "prompt": prompt,
        "backend": backend,
        "retrieval_mode": retrieval_mode,
    }


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    jsonl_ok = True
    jsonl_records = []
    for ln in lines:
        try:
            jsonl_records.append(json.loads(ln))
        except Exception:
            jsonl_ok = False
            break
    if jsonl_ok and jsonl_records:
        return jsonl_records

    decoder = json.JSONDecoder()
    idx = 0
    records = []
    n = len(raw)
    while idx < n:
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(raw, idx)
        if isinstance(obj, dict):
            records.append(obj)
        else:
            raise ValueError("Encontrei um JSON que não é dict (objeto). Ajuste o parser conforme necessário.")
        idx = end
    return records


def count_words(s: str) -> int:
    return len(re.findall(r"\S+", (s or "").strip()))


def formatar_moeda_br(valor):
    if isinstance(valor, dict):
        valor = valor.get("answer") or valor.get("resposta") or str(valor)

    valor_str = str(valor).strip()
    valor_str = re.sub(r"[^\d,\.]", "", valor_str)

    if not valor_str:
        return "R$ 0,00"

    if "," in valor_str:
        partes = valor_str.rsplit(",", 1)
        inteiro = partes[0].replace(".", "").strip()
        decimal = partes[1].strip() if len(partes) > 1 else "00"

        if inteiro == "":
            inteiro = "0"
        if decimal == "":
            decimal = "00"

        decimal = (decimal + "00")[:2]
    else:
        if "." in valor_str:
            try:
                num = float(valor_str)
            except Exception:
                num = 0.0
        else:
            try:
                num = int(valor_str)
            except Exception:
                num = 0

        inteiro = f"{int(num):,}".replace(",", ".")
        decimal = f"{num:.2f}".split(".")[1]
        return f"R$ {inteiro},{decimal}"

    try:
        inteiro = f"{int(inteiro):,}".replace(",", ".")
    except Exception:
        inteiro = "0"

    return f"R$ {inteiro},{decimal}"


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return precision, recall, f1

for CHUNK_SIZE in CHUNK_SIZES:
    for CHUNK_OVERLAP_PERCENTAGE in CHUNK_OVERLAP_PERCENTAGES:
        for TOP_K in TOP_KS:
            for RETRIEVAL_MODE in RETRIEVAL_MODES:
                for BASE_EXPERIMENTO in BASE_EXPERIMENTOS:
                    CHUNK_OVERLAP = calculate_chunk_overlap(
                        CHUNK_SIZE,
                        CHUNK_OVERLAP_PERCENTAGE,
                    )

                    LOG_PATH = LOGS_DIR / (
                        f"log_{CHUNK_SIZE}_overlap_pct_{CHUNK_OVERLAP_PERCENTAGE}_k_{TOP_K}_mode_{RETRIEVAL_MODE}"
                        f"_exp_{BASE_EXPERIMENTO}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                    )
                    _LOG_FILE = open(LOG_PATH, "a", encoding="utf-8")

                    sys.stdout = Tee(sys.__stdout__, _LOG_FILE)
                    sys.stderr = Tee(sys.__stderr__, _LOG_FILE)

                    print(f"[LOG] Saída também será gravada em: {LOG_PATH.resolve()}")

                    # --------------------------------------
                    # Paths derivados
                    # --------------------------------------
                    CHROMADB_PATH = (
                        f"chromadb_chunk_size_{CHUNK_SIZE}_overlap_pct_{CHUNK_OVERLAP_PERCENTAGE}"
                    )
                    INDEX_PATH = (PROJECT_ROOT / CHROMADB_PATH).expanduser()

                    AUDIT_PATH = (
                        PROJECT_ROOT / f"chunks_audit_{CHUNK_SIZE}_overlap_pct_{CHUNK_OVERLAP_PERCENTAGE}.json"
                    ).expanduser()

                    max_pergunta_env = os.getenv("MAX_PERGUNTA", "").strip()
                    MAX_PERGUNTA = int(max_pergunta_env) if max_pergunta_env else None

                    JSON_PATH = (DATA_DIR / f"base_{BASE_EXPERIMENTO}_menor.jsonl").expanduser()
                    OUT_JSONL = (
                        PROJECT_ROOT
                        / f"experimento_{NUMERO_EXPERIMENTO}"
                        / (
                            f"rag_prompt_tests_{CHUNK_SIZE}_overlap_pct_{CHUNK_OVERLAP_PERCENTAGE}_"
                            f"{BASE_EXPERIMENTO}_{RETRIEVAL_MODE}.jsonl"
                        )
                    ).expanduser()

                    # --------------------------------------
                    # Cria diretórios necessários
                    # --------------------------------------
                    INDEX_PATH.mkdir(parents=True, exist_ok=True)
                    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
                    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

                    # --------------------------------------
                    # Exibe configuração carregada
                    # --------------------------------------
                    print("===== CONFIGURAÇÃO =====")
                    print("PROJECT_ROOT       :", PROJECT_ROOT)
                    print("DATA_DIR           :", DATA_DIR)
                    print("COLLECTION_NAME    :", COLLECTION_NAME)
                    print("CHUNK_SIZE         :", CHUNK_SIZE)
                    print("CHUNK_OVERLAP_%    :", CHUNK_OVERLAP_PERCENTAGE)
                    print("CHUNK_OVERLAP      :", CHUNK_OVERLAP)
                    print("MIN_CHUNK_LEN      :", MIN_CHUNK_LEN)
                    print("INDEX_PATH         :", INDEX_PATH)
                    print("AUDIT_PATH         :", AUDIT_PATH)
                    print("OLLAMA_BASE_URL    :", OLLAMA_BASE_URL)
                    print("EMBED_MODEL        :", EMBED_MODEL)
                    print("NOMIC_MODEL        :", NOMIC_MODEL)
                    print("DEVICE             :", DEVICE)
                    print("TOP_K              :", TOP_K)
                    print("RETRIEVAL_MODE     :", RETRIEVAL_MODE)
                    print("NUMERO_EXPERIMENTO :", NUMERO_EXPERIMENTO)
                    print("BASE_EXPERIMENTO   :", BASE_EXPERIMENTO)
                    print("MAX_PERGUNTA       :", MAX_PERGUNTA)
                    print("JSON_PATH          :", JSON_PATH )
                    print("HF_MODEL_ID        :", HF_MODEL_ID )
                    print("OUT_JSONL          :", OUT_JSONL)

                    print("cuda disponível:", torch.cuda.is_available())
                    print("bf16 suportado:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
                    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem gpu")

                    # ==============================================================================
                    # Conecta no ChromaDB
                    # ==============================================================================
                    INDEX_DIR = Path(INDEX_PATH)
                    INDEX_DIR.mkdir(parents=True, exist_ok=True)

                    settings = Settings(
                        _env_file=None,
                        anonymized_telemetry=False
                    )

                    client = chromadb.PersistentClient(path=str(INDEX_DIR), settings=settings)

                    collection = client.get_or_create_collection(
                        name=COLLECTION_NAME,
                        metadata={"hnsw:space": "cosine"},
                    )

                    print("📦 Coleção:", COLLECTION_NAME)
                    print("📁 Persistência em:", INDEX_DIR)
                    print("🔢 Itens atuais na coleção:", collection.count())

                    data = collection.get(
                        limit=1,
                        include=["documents", "metadatas"]
                    )

                    print("ID:", data["ids"][0])
                    print("\nTexto do chunk:\n")
                    print(data["documents"][0][:500])

                    print("\nMetadata:")
                    print(data["metadatas"][0])

                    first_id = data["ids"][0]
                    emb = collection.get(
                        ids=[first_id],
                        include=["embeddings"]
                    )["embeddings"][0]

                    print("\nDimensão do embedding:", len(emb))
                    print("Primeiros valores:", emb[:10])

                    PAGE = 1000
                    total = collection.count()

                    docs = []
                    metas = []
                    embs = []

                    for offset in tqdm(range(0, total, PAGE), desc="Carregando coleção"):
                        data = collection.get(
                            limit=PAGE,
                            offset=offset,
                            include=["documents", "metadatas", "embeddings"]
                        )

                        if data["documents"] is not None:
                            docs.extend(data["documents"])

                        if data["metadatas"] is not None:
                            metas.extend(data["metadatas"])

                        if data["embeddings"] is not None:
                            embs.extend(data["embeddings"])

                    emb_matrix = np.array(embs, dtype=np.float32)

                    print("Chunks:", len(docs))
                    print("Metadatas:", len(metas))
                    print("Embeddings shape:", emb_matrix.shape)

                    query = "Qual é o valor unitário com NSA DISTRIBUIDORA DE MEDICAMENTOS EIRELI, CNPJ n° 34.729.047/0001-02, item 14 ?."

                    hits = chroma_retrieve_hybrid(
                        query,
                        docs=docs,
                        metas=metas,
                        emb_matrix=emb_matrix,
                        top_k=TOP_K,
                        mode=RETRIEVAL_MODE,
                    )

                    for i, h in enumerate(hits, start=1):
                        m = h["metadata"]
                        print(
                            f"\n=== Hit {i} | rrf={h.get('rrf_score')} | "
                            f"bm25={h['bm25_score']} | dist={h['distance']} | "
                            f"{m.get('source_pdf')} | pág {m.get('page')} | chunk {m.get('chunk')} ==="
                        )
                        print(h["text"])

                    question = "Qual é o valor total do contrato nº 70/2019 entre DF/PMDF e a empresa Parts Lub Distribuidora e Serviços Eireli para a aquisição de pneus?"

                    result = answer(
                        question=question,
                        docs=docs,
                        metas=metas,
                        emb_matrix=emb_matrix,
                        top_k=TOP_K,
                        backend="hf_generate_chat",
                        retrieval_mode=RETRIEVAL_MODE,
                        embed_fn=hf_nomic_embed,
                        system_prompt="Responda apenas com base no contexto recuperado. Se não estiver no contexto, diga isso claramente."
                    )

                    print(result["answer"])

                    records = load_records(JSON_PATH)
                    print("Registros carregados:", len(records))
                    print("Campos do 1º registro:", list(records[0].keys()))

                    random.seed(42)
                    sample_size = math.ceil(len(records) * 0.30)
                    records_sample = random.sample(records, sample_size)

                    print("Total original:", len(records))
                    print("Total selecionado (30%):", len(records_sample))

                    iter_records = records_sample if MAX_PERGUNTA is None else records_sample[:MAX_PERGUNTA]

                    with open(OUT_JSONL, "w", encoding="utf-8") as f:
                        for rec in tqdm(iter_records, desc="Processando", unit="registro"):
                            pergunta = rec.get("pergunta", "")
                            esperada = rec["resposta"]

                            resultado = answer(
                                question=pergunta,
                                docs=docs,
                                metas=metas,
                                emb_matrix=emb_matrix,
                                top_k=TOP_K,
                                retrieval_mode=RETRIEVAL_MODE,
                                embed_fn=hf_nomic_embed,
                                forbid_trailing_symbol=True,
                                max_prompt_chars=None,
                            )
                            resposta_texto = str(resultado.get("answer", ""))
                            gerada = formatar_moeda_br(resposta_texto)

                            if esperada and re.search(re.escape(esperada), gerada):
                                extra = count_words(gerada) - count_words(esperada)
                                acerto_mais_palavras = max(0, extra)
                            else:
                                acerto_mais_palavras = False

                            acerto_valor = (
                                (esperada is not None and esperada == gerada)
                            )

                            row = {
                                "id_versao_pergunta": rec.get("id_versao_pergunta"),
                                "pergunta": pergunta,
                                "resposta_esperada": esperada,
                                "resposta_gerada": gerada,
                                "acerto": bool(acerto_valor),
                                "acerto_mais_palavras": acerto_mais_palavras,
                                "pdf": rec.get("pdf"),
                                "extrato": rec.get("extrato"),
                                "contextos_recuperados": resultado["retrieved"],
                                "top_k": TOP_K,
                                "retrieval_mode": RETRIEVAL_MODE,
                                "timestamp": time.time()
                            }

                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    print("Arquivo gerado:", OUT_JSONL)

                    total = 0
                    acertos_exatos = 0
                    contidos = 0
                    extras_lista = []

                    tp_exato = fp_exato = fn_exato = tn_exato = 0
                    tp_cont = fp_cont = fn_cont = tn_cont = 0

                    with open(OUT_JSONL, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue

                            total += 1
                            row = json.loads(line)
                            y_pred_exato = (row.get("acerto") is True)

                            if y_pred_exato:
                                tp_exato += 1
                                acertos_exatos += 1
                            else:
                                fn_exato += 1

                            val = row.get("acerto_mais_palavras")
                            y_pred_cont = (val is not False)

                            if y_pred_cont:
                                tp_cont += 1
                                contidos += 1
                                if isinstance(val, (int, float)):
                                    extras_lista.append(val)
                            else:
                                fn_cont += 1

                    perc_exato = (acertos_exatos / total * 100) if total else 0
                    perc_contido = (contidos / total * 100) if total else 0

                    media_extras = sum(extras_lista) / len(extras_lista) if extras_lista else 0
                    max_extras = max(extras_lista) if extras_lista else 0

                    prec_ex, rec_ex, f1_ex = prf(tp_exato, fp_exato, fn_exato)
                    prec_c, rec_c, f1_c = prf(tp_cont, fp_cont, fn_cont)

                    print(f"Total de testes: {total}\n")
                    print("=== ACERTO EXATO ===")
                    print(f"Acertos exatos: {acertos_exatos}")
                    print(f"Percentual acerto exato: {perc_exato:.2f}%")
                    print(f"Precision (exato): {prec_ex:.4f}")
                    print(f"Recall (exato):    {rec_ex:.4f}")
                    print(f"F1-score (exato):  {f1_ex:.4f}\n")

                    print("=== CONTÉM RESPOSTA (com ou sem palavras extras) ===")
                    print(f"Respostas que contêm o esperado: {contidos}")
                    print(f"Percentual que contém a resposta: {perc_contido:.2f}%")
                    print(f"Média de palavras extras: {media_extras:.2f}")
                    print(f"Máximo de palavras extras: {max_extras}")
                    print(f"Precision (contém): {prec_c:.4f}")
                    print(f"Recall (contém):    {rec_c:.4f}")
                    print(f"F1-score (contém):  {f1_c:.4f}")

                    del collection
                    del client
