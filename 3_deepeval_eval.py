# -*- coding: utf-8 -*-
"""
Etapa 3 — Avaliação dos top 20 experimentos por ranking.

Lê os JSONLs gerados pelo script 2, seleciona os top 20 experimentos
(experimentos 3 e 4 apenas) em três rankings: total, base_a e base_b.
Avalia com modelo juiz HuggingFace — diferente do Qwen2.5-3B usado na geração.

Métricas customizadas com pontuação numérica (0-10): sem dependência de JSON
estruturado — compatível com modelos de 3B que não seguem schemas complexos.

Instalar dependências:
    pip install deepeval transformers torch

Variáveis de ambiente adicionais (opcional, via .env):
    JUDGE_BACKEND         backend do juiz: "hf" (padrão) ou "ollama"
    JUDGE_MODEL_ID        modelo HuggingFace como juiz   (padrão: microsoft/phi-3-mini-4k-instruct)
    JUDGE_OLLAMA_MODEL    modelo Ollama como juiz fallback (padrão: llama3.2:latest)
    MAX_RECORDS_PER_FILE  registros avaliados por JSONL   (padrão: 20)
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

import requests
import torch
import transformers
from dotenv import load_dotenv

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase
except ImportError:
    sys.exit("deepeval não instalado. Execute: pip install deepeval")

# ------------------------------------------------------------------
# Configuração
# ------------------------------------------------------------------
DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(DOTENV_PATH)

RESULTS_DIR = Path(os.getenv("PROJECT_ROOT", "~/Experimentos/Resultados")).expanduser()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

JUDGE_BACKEND = os.getenv("JUDGE_BACKEND", "hf").lower().strip()
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID", "microsoft/phi-3-mini-4k-instruct")
JUDGE_OLLAMA_MODEL = os.getenv("JUDGE_OLLAMA_MODEL", "llama3.2:latest")

EXPERIMENTOS_ALVO = {"experimento_3", "experimento_4"}
MIN_RECORDS = 20
TOP_N = 10
MAX_RECORDS_PER_FILE = int(os.getenv("MAX_RECORDS_PER_FILE", "20"))

NUMERO_EXPERIMENTO = os.getenv("NUMERO_EXPERIMENTO", "eval")
OUT_DIR = RESULTS_DIR / "Avaliacao"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

_HF_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else torch.float32
)


# ------------------------------------------------------------------
# Logging Tee (terminal + arquivo simultâneo)
# ------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def setup_tee_logging() -> Path:
    log_path = OUT_DIR / f"log_deepeval_experimento_{NUMERO_EXPERIMENTO}_{RUN_TS}.log"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_path


# ------------------------------------------------------------------
# Modelos juiz
# ------------------------------------------------------------------
class HuggingFaceJudge(DeepEvalBaseLLM):
    """Juiz local via HuggingFace Transformers (phi-3-mini ou similar)."""

    def __init__(self, model_id: str = JUDGE_MODEL_ID):
        self._model_id = model_id
        self._pipeline = None

    def load_model(self):
        if self._pipeline is None:
            token = os.getenv("TOKEN_HF")
            self._pipeline = transformers.pipeline(
                "text-generation",
                model=self._model_id,
                model_kwargs={"torch_dtype": _HF_DTYPE},
                device_map="auto",
                token=token if token else None,
            )
        return self._pipeline

    def generate(self, prompt: str) -> str:
        pipe = self.load_model()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluation assistant. "
                    "Respond with ONLY a single integer from 0 to 10. "
                    "No explanation, no text — just the number."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        formatted = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Trunca para caber na janela do modelo, reservando 64 tokens para o número.
        max_tokens = getattr(pipe.tokenizer, "model_max_length", 4096)
        input_ids = pipe.tokenizer(formatted, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > max_tokens - 64:
            formatted = pipe.tokenizer.decode(
                input_ids[0, -(max_tokens - 64):], skip_special_tokens=False
            )

        outputs = pipe(
            formatted,
            max_new_tokens=64,
            do_sample=False,
            return_full_text=False,
            truncation=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
        )
        return (outputs[0].get("generated_text") or "").strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model_id

    def unload(self) -> None:
        del self._pipeline
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OllamaJudge(DeepEvalBaseLLM):
    """Juiz local via Ollama (ativado com JUDGE_BACKEND=ollama)."""

    def __init__(self, model: str = JUDGE_OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self._model = model
        self._base_url = base_url

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        r = requests.post(
            f"{self._base_url}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["response"].strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model

    def unload(self) -> None:
        pass


def build_judge() -> HuggingFaceJudge | OllamaJudge:
    if JUDGE_BACKEND == "ollama":
        print(f"Juiz backend  : Ollama ({JUDGE_OLLAMA_MODEL})")
        return OllamaJudge()
    print(f"Juiz backend  : HuggingFace ({JUDGE_MODEL_ID})")
    return HuggingFaceJudge()


# ------------------------------------------------------------------
# Métricas numéricas customizadas
#
# Usam pontuação 0-10 via pergunta direta ao juiz — sem JSON schema,
# compatível com modelos de 3B que falham em schemas complexos do DeepEval.
# ------------------------------------------------------------------
class _NumericMetric:
    def __init__(self, judge: HuggingFaceJudge | OllamaJudge, threshold: float = 0.5):
        self.judge = judge
        self.threshold = threshold
        self.score = 0.0

    def _extract_score(self, response: str) -> float:
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
        if not match:
            return 0.0
        raw = float(match.group(1))
        normalized = raw / 10.0 if raw > 1.0 else raw
        return round(min(1.0, max(0.0, normalized)), 4)

    def _ask(self, prompt: str) -> float:
        try:
            response = self.judge.generate(prompt)
            return self._extract_score(response)
        except Exception as e:
            raise RuntimeError(str(e)) from e


class ContextualRelevancyScore(_NumericMetric):
    """Os contextos recuperados são relevantes para a pergunta? (0-10)"""

    def measure(self, test_case: LLMTestCase) -> float:
        contexts = "\n---\n".join(test_case.retrieval_context or [])
        prompt = (
            f"Question: {test_case.input}\n\n"
            f"Retrieved contexts:\n{contexts}\n\n"
            "On a scale from 0 to 10, how relevant are these contexts to the question?\n"
            "Respond with ONLY a number from 0 to 10."
        )
        self.score = self._ask(prompt)
        return self.score


class FaithfulnessScore(_NumericMetric):
    """A resposta gerada é fiel aos contextos recuperados? (0-10)"""

    def measure(self, test_case: LLMTestCase) -> float:
        contexts = "\n---\n".join(test_case.retrieval_context or [])
        prompt = (
            f"Retrieved contexts:\n{contexts}\n\n"
            f"Answer: {test_case.actual_output}\n\n"
            "On a scale from 0 to 10, how faithful is the answer to the retrieved contexts "
            "(does it avoid hallucination and stay grounded in the context)?\n"
            "Respond with ONLY a number from 0 to 10."
        )
        self.score = self._ask(prompt)
        return self.score


class AnswerRelevancyScore(_NumericMetric):
    """A resposta gerada é relevante para a pergunta? (0-10)"""

    def measure(self, test_case: LLMTestCase) -> float:
        prompt = (
            f"Question: {test_case.input}\n\n"
            f"Answer: {test_case.actual_output}\n\n"
            "On a scale from 0 to 10, how relevant is the answer to the question?\n"
            "Respond with ONLY a number from 0 to 10."
        )
        self.score = self._ask(prompt)
        return self.score


METRIC_KEYS = ("faithfulness", "answer_relevancy", "contextual_relevancy")


def build_metrics(judge: HuggingFaceJudge | OllamaJudge) -> list:
    return [
        FaithfulnessScore(judge),
        AnswerRelevancyScore(judge),
        ContextualRelevancyScore(judge),
    ]


# ------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------
def _detect_base(filename: str) -> str:
    if "_a_objeto_" in filename:
        return "a_objeto"
    if "_b_extrato_" in filename:
        return "b_extrato"
    return "desconhecida"


def _aggregate(path: Path) -> dict | None:
    total = acertos = contidos = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                if row.get("acerto") is True:
                    acertos += 1
                if row.get("acerto_mais_palavras") is not False:
                    contidos += 1
    except Exception as e:
        print(f"[AVISO] Erro ao ler {path.name}: {e}")
        return None

    if total < MIN_RECORDS:
        return None

    return {
        "path": path,
        "file": path.name,
        "experimento": path.parent.name,
        "base": _detect_base(path.name),
        "total": total,
        "acerto_exato_pct": round(acertos / total * 100, 2),
        "contido_pct": round(contidos / total * 100, 2),
    }


def build_rankings(results_dir: Path) -> dict[str, list[dict]]:
    files = sorted(results_dir.rglob("rag_prompt_tests_*.jsonl"))
    files = [p for p in files if p.parent.name in EXPERIMENTOS_ALVO]

    summaries = [s for p in files if (s := _aggregate(p)) is not None]

    key = lambda x: (x["acerto_exato_pct"], x["contido_pct"])

    def melhores(lst: list[dict]) -> list[dict]:
        return sorted(lst, key=key, reverse=True)[:TOP_N]

    def piores(lst: list[dict]) -> list[dict]:
        return sorted(lst, key=key, reverse=False)[:TOP_N]

    total    = summaries
    base_a   = [s for s in summaries if s["base"] == "a_objeto"]
    base_b   = [s for s in summaries if s["base"] == "b_extrato"]

    return {
        "total_melhores":     melhores(total),
        "total_piores":       piores(total),
        "a_objeto_melhores":  melhores(base_a),
        "a_objeto_piores":    piores(base_a),
        "b_extrato_melhores": melhores(base_b),
        "b_extrato_piores":   piores(base_b),
    }


# ------------------------------------------------------------------
# Avaliação
# ------------------------------------------------------------------
def _load_test_cases(path: Path, max_records: int) -> list[LLMTestCase]:
    cases: list[LLMTestCase] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or len(cases) >= max_records:
                break
            row = json.loads(line)
            retrieval_context = [
                h["text"]
                for h in (row.get("contextos_recuperados") or [])
                if h.get("text")
            ]
            if not retrieval_context:
                continue
            cases.append(
                LLMTestCase(
                    input=row.get("pergunta", ""),
                    actual_output=row.get("resposta_gerada", ""),
                    expected_output=row.get("resposta_esperada", ""),
                    retrieval_context=retrieval_context,
                )
            )
    return cases


def evaluate_file(entry: dict, metrics: list, max_records: int) -> dict:
    path: Path = entry["path"]
    print(f"\n  Avaliando: {path.name} ({max_records} registros max)")

    test_cases = _load_test_cases(path, max_records)
    if not test_cases:
        print("  [AVISO] Nenhum test case carregado.")
        result = {k: v for k, v in entry.items() if k != "path"}
        result.update({"path": str(path), "scores": None, "error": "sem test cases"})
        return result

    acc: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    errors = 0

    for i, tc in enumerate(test_cases, 1):
        for metric, key in zip(metrics, METRIC_KEYS):
            try:
                metric.measure(tc)
                acc[key].append(metric.score)
            except Exception as e:
                print(f"    [ERRO] registro {i} | {key}: {e}")
                errors += 1

    scores = {
        k: round(sum(v) / len(v), 4) if v else None
        for k, v in acc.items()
    }
    print(f"  Scores: {scores} | erros={errors}")

    result = {k: v for k, v in entry.items() if k != "path"}
    result.update({
        "path": str(path),
        "scores": scores,
        "records_evaluated": len(test_cases),
        "errors": errors,
    })
    return result


def evaluate_ranking(
    ranking: list[dict],
    label: str,
    judge: HuggingFaceJudge | OllamaJudge,
    max_records: int,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Avaliando ranking: {label} ({len(ranking)} experimentos)")
    print(f"{'='*60}")

    metrics = build_metrics(judge)
    results = []
    for i, entry in enumerate(ranking, 1):
        print(f"\n[{i}/{len(ranking)}]", end="")
        results.append(evaluate_file(entry, metrics, max_records))

    return results


# ------------------------------------------------------------------
# Saída: JSON
# ------------------------------------------------------------------
def save_ranking_report(results: list[dict], label: str, judge_name: str) -> None:
    json_path = OUT_DIR / f"ranking_{label}_{RUN_TS}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": label,
                "generated_at": RUN_TS,
                "judge_backend": JUDGE_BACKEND,
                "judge_model": judge_name,
                "top_n": TOP_N,
                "max_records_per_file": MAX_RECORDS_PER_FILE,
                "experimentos_avaliados": sorted(EXPERIMENTOS_ALVO),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nJSON salvo: {json_path}")


def print_summary(results: list[dict], label: str) -> None:
    print(f"\n--- Resumo: {label} ---")
    print(f"{'#':>2}  {'Experimento':<15} {'Base':<10} {'Exato%':>7} {'Faithful':>9} {'AnsRel':>7} {'CtxRel':>7}")
    print("-" * 72)
    for i, r in enumerate(results, 1):
        sc = r.get("scores") or {}
        faith = f"{sc['faithfulness']:.3f}"        if sc.get("faithfulness")        is not None else "   N/A"
        ans   = f"{sc['answer_relevancy']:.3f}"    if sc.get("answer_relevancy")    is not None else "   N/A"
        ctx   = f"{sc['contextual_relevancy']:.3f}" if sc.get("contextual_relevancy") is not None else "   N/A"
        print(
            f"{i:>2}. {r['experimento']:<15} {r['base']:<10} "
            f"{r['acerto_exato_pct']:>6.2f}% {faith:>9} {ans:>7} {ctx:>7}"
        )


# ------------------------------------------------------------------
# Ranking divergentes (pós-avaliação)
# ------------------------------------------------------------------
def build_divergentes(all_results: list[dict], top_n: int = TOP_N) -> list[dict]:
    """
    Identifica experimentos onde o DeepEval avalia bem mas o acerto exato é baixo.
    divergencia = deepeval_medio - (acerto_exato_pct / 100)
    Valor positivo alto → DeepEval vê qualidade que o string matching não captura.
    """
    seen = set()
    unique = []
    for r in all_results:
        if r["file"] not in seen:
            seen.add(r["file"])
            unique.append(r)

    divergentes = []
    for r in unique:
        sc = r.get("scores") or {}
        vals = [v for v in sc.values() if v is not None]
        if not vals:
            continue
        deepeval_medio = sum(vals) / len(vals)
        divergencia = round(deepeval_medio - (r["acerto_exato_pct"] / 100), 4)
        divergentes.append({**r, "deepeval_medio": round(deepeval_medio, 4), "divergencia": divergencia})

    return sorted(divergentes, key=lambda x: x["divergencia"], reverse=True)[:top_n]


def print_summary_divergentes(results: list[dict]) -> None:
    print("\n--- Resumo: divergentes (DeepEval bom, acerto exato baixo) ---")
    print(f"{'#':>2}  {'Experimento':<15} {'Base':<10} {'Exato%':>7} {'DV_medio':>9} {'Diverg':>8}")
    print("-" * 62)
    for i, r in enumerate(results, 1):
        print(
            f"{i:>2}. {r['experimento']:<15} {r['base']:<10} "
            f"{r['acerto_exato_pct']:>6.2f}% "
            f"{r['deepeval_medio']:>9.3f} "
            f"{r['divergencia']:>+8.3f}"
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    log_path = setup_tee_logging()
    print(f"Log de execução: {log_path}")
    print(f"Results dir   : {RESULTS_DIR}")
    print(f"Output dir    : {OUT_DIR}")
    print(f"Max records   : {MAX_RECORDS_PER_FILE} por arquivo")
    print(f"Experimentos  : {sorted(EXPERIMENTOS_ALVO)}")

    rankings = build_rankings(RESULTS_DIR)
    print(f"\nRankings gerados (top/bottom {TOP_N} cada):")
    for label, lst in rankings.items():
        print(f"  {label}: {len(lst)} experimentos")

    judge = build_judge()
    all_results: list[dict] = []

    try:
        for label, ranking in rankings.items():
            results = evaluate_ranking(ranking, label, judge, MAX_RECORDS_PER_FILE)
            save_ranking_report(results, label, judge.get_model_name())
            print_summary(results, label)
            all_results.extend(results)
    finally:
        judge.unload()

    # Ranking divergentes — calculado após todos os scores estarem disponíveis
    divergentes = build_divergentes(all_results)
    save_ranking_report(divergentes, "divergentes", judge.get_model_name())
    print_summary_divergentes(divergentes)

    print(f"\nAvaliação concluída. Todos os arquivos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
