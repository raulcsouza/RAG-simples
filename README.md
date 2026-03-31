## RAG com Ollama ou Hugging Face + ChromaDB para PDFs

Este notebook implementa um pipeline de Retrieval-Augmented Generation (RAG) para documentos PDF utilizando:

- Ollama para geração de embeddings e texto
- Hugging Face para geração de embeddings e texto
- Modelo de Embedding `nomic-embed-text`
- Modelo para Geração de texto pode ser qualquer um no Ollama ou Hugging Face
- ChromaDB como banco vetorial
- Chunking de texto
- Indexação persistente

O objetivo é converter documentos PDF em chunks vetorizados, armazená-los no ChromaDB e permitir busca semântica eficiente.

## Requisitos

Python 3.10+

Instalar dependências:

pip install chromadb pypdf tqdm requests python-dotenv

---

## Instalar e iniciar Ollama

Instale o Ollama:

https://ollama.ai

Baixe o modelo de embeddings:

ollama pull nomic-embed-text

Inicie o servidor:

ollama serve

Servidor padrão:

http://localhost:11434

---

## Configuração via .env

O projeto utiliza variáveis de ambiente para facilitar a configuração.

Crie um arquivo chamado:

.env

na raiz do projeto.

---

## Exemplo de .env

## Diretório contendo os PDFs
DATA_PDFs_DIR=/path/to/your/pdf/folder

## Diretório raiz do projeto
PROJECT_ROOT=/path/to/project

## Nome da coleção vetorial
COLLECTION_NAME=pdf_collection

## Configuração de chunking
CHUNK_SIZE=2048
CHUNK_OVERLAP=150
MIN_CHUNK_LEN=200

## Embeddings
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text

## Geração com Hugging Face
HF_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ou
HF_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

## Recuperação
TOP_K=5

## Processamento
BATCH=64

# Numero experimento
NUMERO_EXPERIMENTO = 1

---

## Descrição das Variáveis

DATA_PDFs_DIR  
Diretório contendo os PDFs que serão processados.

PROJECT_ROOT  
Diretório raiz onde serão armazenados índice vetorial e auditoria de chunks.

COLLECTION_NAME  
Nome da coleção utilizada no ChromaDB.

CHUNK_SIZE  
Tamanho máximo do chunk em caracteres.

CHUNK_OVERLAP  
Sobreposição entre chunks consecutivos.

MIN_CHUNK_LEN  
Chunks menores que esse valor são ignorados.

OLLAMA_BASE_URL  
URL do servidor Ollama.

EMBED_MODEL  
Modelo utilizado para gerar embeddings.

TOP_K  
Quantidade de documentos retornados na busca vetorial.

BATCH  
Quantidade de chunks enviados para geração de embeddings por lote.

---

## Execução

Após configurar o .env, execute o notebook.

O pipeline irá:

1. Ler os PDFs
2. Extrair texto
3. Criar chunks
4. Gerar embeddings via Ollama
5. Armazenar no ChromaDB

---

## Arquivos Gerados

Durante a execução serão criados:

chromadb_chunk_size_2048_overlap_150/

contendo o índice vetorial persistente.

chunks_audit_2048.json

arquivo contendo auditoria dos chunks gerados.

---

## Boas práticas

- Ajustar CHUNK_SIZE dependendo do tipo de documento
- Evitar chunks muito pequenos
- Monitorar consumo de memória ao aumentar BATCH
- Manter .env fora de repositórios públicos

