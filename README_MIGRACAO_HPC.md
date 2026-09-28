# Migração para o HPC (ROCm / Slurm)

Pacote transferido via rsync para `/home1/raulcsouza/RAG/` no servidor
`216.114.73.39`. Conteúdo:

- `RAG-simples/2_rag_ollama_nomic_embed_testeRAG_base_chroma_retrieve_hybrid_percentual.py`
- `RAG-simples/.env` (renomeado a partir de `.env.hpc`; `PROJECT_ROOT` e `DATA_DIR`
  já apontam para `/home1/raulcsouza/RAG/...`)
- `RAG-simples/requirements.txt`
- `Experimentos/Resultados/chromadb_chunk_size_{128,256,512,1024,2048,4096}_overlap_pct_{0,30,60}`
  (18 coleções — as únicas lidas por este script; as 15 pastas legadas
  `overlap_0/128/150` não foram incluídas por não serem usadas por ele)
- `Experimentos/DiarioOficial-Contratos-BR-GT/base_a_objeto_menor.jsonl` e
  `base_b_extrato_menor.jsonl` (únicos arquivos que o script lê de `DATA_DIR`;
  os `_maior.jsonl` e os PDFs originais não são necessários para esta execução)

## 1. Ambiente Python (sem reinstalar torch)

O ambiente de origem usa `torch==2.10.0`, que **coincide exatamente** com um dos
módulos disponíveis no cluster. Não é necessário portar nada para HIP — o
script é Python puro usando a API `torch.cuda.*`, que funciona sem alteração
em builds ROCm do PyTorch (o hipify/hipcc do guia do cluster é só para código
CUDA C/C++ `.cu`, que não existe aqui).

```bash
module load rocm/7.2.0
module load pytorch/2.10.0

python -m venv ~/venvs/rag-simples --system-site-packages
source ~/venvs/rag-simples/bin/activate
pip install -r /home1/raulcsouza/RAG/RAG-simples/requirements.txt
```

`--system-site-packages` é importante: garante que o `torch` (ROCm) do módulo
seja enxergado pelo venv, sem o pip tentar baixar um `torch` do PyPI (que seria
build CUDA e não funcionaria nas MI210).

## 2. Pré-cache dos modelos Hugging Face (fazer no login node, com internet)

Nós de GPU do Slurm normalmente não têm saída para internet. Baixe os modelos
uma vez no login node antes de submeter o job:

```bash
module load rocm/7.2.0 pytorch/2.10.0
source ~/venvs/rag-simples/bin/activate

python - <<'EOF'
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel
import os
login(os.environ["HF_TOKEN"])  # defina antes: export HF_TOKEN=<seu token>
AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
AutoModel.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
EOF
```

Isso popula `~/.cache/huggingface`. No job do Slurm, defina `HF_HUB_OFFLINE=1`
para não depender de rede no nó de GPU.

## 3. Rodar via Slurm (nunca no login node)

Teste rápido (fila `devel`, limite 30 min):

```bash
srun --partition=devel --gpus=1 --pty bash -c '
  module load rocm/7.2.0 pytorch/2.10.0
  source ~/venvs/rag-simples/bin/activate
  export HF_HUB_OFFLINE=1
  cd /home1/raulcsouza/RAG/RAG-simples
  python 2_rag_ollama_nomic_embed_testeRAG_base_chroma_retrieve_hybrid_percentual.py
'
```

Execução completa (fila `mi3001x`, MI300X):

```bash
sbatch --partition=mi3001x --gpus=1 --job-name=rag-hybrid --wrap="
  module load rocm/7.2.0 pytorch/2.10.0 &&
  source ~/venvs/rag-simples/bin/activate &&
  export HF_HUB_OFFLINE=1 &&
  cd /home1/raulcsouza/RAG/RAG-simples &&
  python 2_rag_ollama_nomic_embed_testeRAG_base_chroma_retrieve_hybrid_percentual.py
"
```

O script só usa 1 GPU (`device_map="auto"` resolve para `cuda:0`, que no build
ROCm mapeia para a primeira GPU visível do nó) — `--gpus=1` é suficiente. A
MI300X (`gfx942`) é uma arquitetura diferente da MI210 (`gfx90a`) usada na
origem, mas isso não exige nenhuma mudança no script: o binário do módulo
`pytorch/2.10.0` já traz suporte a `gfx942`, e o código só fala com a GPU
através da API genérica `torch.cuda.*`.

## 4. Observações

- `chromadb` é puro CPU/disco — não depende de ROCm/CUDA. Basta apontar
  `PersistentClient` para as pastas copiadas; não há paths absolutos
  "gravados" dentro do índice.
- `OLLAMA_BASE_URL`/`EMBED_MODEL` no `.env` não são usados por este script
  (o embedding de consulta usa `hf_nomic_embed`, local via `transformers`, e a
  geração usa `hf_generate_chat`). Não é necessário instalar/rodar Ollama no
  HPC para este script específico.
- O `.env` contém um token real do Hugging Face (`TOKEN_HF`). Trate o arquivo
  como segredo (permissões `chmod 600 .env`) e considere rotacionar esse
  token, já que ele foi exposto no terminal durante a preparação deste pacote.
