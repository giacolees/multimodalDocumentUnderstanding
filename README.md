# MultiModal Document Understanding

Vision LLM benchmark for **unanswerable question detection** from document images. The system corrupts a DocVQA-style dataset to produce questions that cannot be answered from the document, then benchmarks Vision LLMs on their ability to identify those questions as unanswerable.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository layout](#repository-layout)
3. [Quick start — CLI mode](#quick-start--cli-mode)
4. [Quick start — microservices mode](#quick-start--microservices-mode)
5. [Pipeline stages](#pipeline-stages)
6. [Microservices architecture](#microservices-architecture)
7. [Observability](#observability)
8. [Configuration reference](#configuration-reference)
9. [Development](#development)

---

## Overview

The project runs in three sequential phases:

```
Phase 1 — Corrupt          Phase 2 — Benchmark         Phase 3 — Mitigate
────────────────────        ───────────────────         ──────────────────
DocVQA / DUDE /      →     Run Vision LLMs on    →     Test in-context
MP-DocVQA dataset           corrupted questions          learning strategies
                            (mistral, gemini,            (few-shot, CoT,
Corruption types:           llama.cpp, vLLM)             knowledge injection,
• NLP entity swap                                        RAG with hybrid
• Element corruption        Metrics: accuracy,           vector+BM25 search)
• Layout corruption         precision, recall, F1
LLM judge verifies
each corruption
```

---

## Repository layout

```
.
├── src/                        # Core research code (CLI entry points)
│   ├── dataset/
│   │   ├── loaders/            # DocVQA, DUDE, MP-DocVQA loaders → QASample
│   │   ├── corruption/         # NLPEntityCorruptor, ElementCorruptor, LayoutCorruptor
│   │   ├── quality_check/      # LLMJudge (verifies corrupted questions)
│   │   └── pipeline.py         # LCEL chain: load → corrupt → judge → save
│   ├── benchmark/
│   │   ├── models/             # MistralModel, GoogleModel, OpenRouterModel, LlamaCppModel
│   │   ├── evaluation/         # compute_metrics() → BenchmarkMetrics
│   │   └── run_benchmark.py    # CLI benchmark runner
│   └── mitigation/
│       ├── strategies/         # few_shot, chain_of_thought, knowledge_injection, finetuning
│       └── run_mitigation.py   # CLI mitigation runner
│
├── services/                   # FastAPI microservices (production API layer)
│   ├── shared/
│   │   └── observability.py    # setup_tracing(), setup_metrics(), get_logger()
│   ├── api-gateway/            # :8000  client entry point
│   ├── model-gateway/          # :8001  unified inference pool (llama + vllm + cloud APIs)
│   ├── document-svc/           # :8002  chunking, embedding, hybrid RAG search
│   ├── evaluation-svc/         # :8003  LLM judge + metrics
│   └── job-runner/             # :8004  async job orchestration
│
├── configs/
│   ├── dataset_config.yaml     # corruption distribution, max_samples, judge settings
│   ├── benchmark_config.yaml   # model backends, evaluation metrics
│   └── mitigation_config.yaml  # strategies, fine-tuning config
│
├── data/
│   ├── raw/                    # downloaded datasets (gitignored)
│   ├── corrupted/              # generated corrupted JSON (gitignored)
│   └── results/                # benchmark and mitigation results (gitignored)
│
├── observability/
│   └── prometheus.yml          # Prometheus scrape config
│
├── docker-compose.yml          # production stack
└── docker-compose.test.yml     # test stack (stub GPU workers)
```

---

## Quick start — CLI mode

### 1. Install

```bash
uv sync                          # core deps
uv sync --extra dev              # also installs jupyter + spacy
uv run python -m spacy download en_core_web_sm
```

Python 3.11 is pinned (`.python-version`). Always prefix commands with `uv run`.

### 2. Set API keys

```bash
cp .env.example .env             # edit with your keys
export $(grep -v '^#' .env | xargs)
```

Required keys (depending on which models you run):

| Key | Used by |
|---|---|
| `GEMINI_API_KEY` | LLM judge + Google model backend |
| `MISTRAL_API_KEY` | Mistral/Pixtral model backend |
| `GEMINI_API_KEY` | Gemini model backend |
| `OPENROUTER_API_KEY` | OpenRouter model backend |
| `HF_TOKEN` | vLLM gated HuggingFace models |

### 3. Download data

```bash
uv run download-data             # downloads DocVQA val split to data/raw/docvqa/
```

### 4. Run the three phases

```bash
# Phase 1 — generate corrupted dataset
uv run corrupt-dataset \
  --dataset docvqa \
  --data_dir data/raw/docvqa \
  --output_dir data/corrupted \
  --config configs/dataset_config.yaml
  # add --no_judge to skip LLM verification (faster)

# Phase 2 — benchmark Vision LLMs
uv run run-benchmark \
  --corrupted_dataset data/corrupted/docvqa_corrupted.json \
  --config configs/benchmark_config.yaml \
  --output_dir results/benchmark

# Phase 3 — test mitigation strategies
uv run run-mitigation \
  --corrupted_dataset data/corrupted/docvqa_corrupted.json \
  --baseline_results results/benchmark/benchmark_results.json \
  --config configs/mitigation_config.yaml \
  --output_dir results/mitigation
```

---

## Quick start — microservices mode

The microservices layer wraps the pipeline behind a REST API with async job tracking, parallel GPU inference, and observability.

### Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU workers)
- Two NVIDIA A6000 GPUs (or adjust `ENABLED_MODELS`)
- GGUF model files in `./models/` for llama.cpp, or HF model ID for vLLM

### 1. Install services dependencies

```bash
uv sync --extra services
```

### 2. Start the stack

```bash
# Full production stack (requires GPU workers)
docker compose up -d

# Test stack (stub GPU workers, no real GPU needed)
docker compose -f docker-compose.test.yml up -d
```

Services come up in dependency order. GPU workers take 30–90 s to load their model — wait for model-gateway to report healthy before sending inference requests.

### 3. Submit a job

```bash
# Generate corrupted dataset
curl -s -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"type": "corrupt", "config": {"dataset": "docvqa", "data_dir": "data/raw/docvqa"}}' \
  | jq .

# Poll until done
JOB_ID="<id from above>"
watch -n 2 "curl -s http://localhost:8000/jobs/$JOB_ID | jq '{status, progress, total}'"

# Run benchmark across all enabled models (parallel)
curl -s -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"type": "benchmark", "config": {"corrupted_dataset": "data/corrupted/docvqa_corrupted.json"}}' \
  | jq .

# Index documents for RAG, then run mitigation
curl -s -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"type": "index", "config": {"dataset": "docvqa", "data_dir": "data/raw/docvqa"}}' | jq .

curl -s -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"type": "mitigation", "config": {"corrupted_dataset": "data/corrupted/docvqa_corrupted.json", "strategies": ["few_shot", "chain_of_thought", "rag"]}}' | jq .
```

### 4. Job types

| Type | Description | Required config keys |
|---|---|---|
| `corrupt` | Run corruption pipeline (Phase 1) | `dataset`, `data_dir` |
| `benchmark` | Benchmark all enabled models (Phase 2) | `corrupted_dataset` |
| `mitigation` | Run mitigation strategies (Phase 3) | `corrupted_dataset`, `strategies` |
| `index` | Index documents into Redis for RAG | `dataset`, `data_dir` |

---

## Pipeline stages

### Phase 1 — Dataset corruption

Each QA sample from DocVQA goes through one of three corruptors chosen at random:

| Corruptor | Technique | Example |
|---|---|---|
| `NLPEntityCorruptor` | spaCy NER + Wikipedia category API — finds a peer entity of the same type and swaps it | `"2019"` → `"1987"` |
| `ElementCorruptor` | Replaces a document element reference (table, figure, section) with one that doesn't exist | `"Table 3"` → `"Table 7"` |
| `LayoutCorruptor` | Replaces a spatial reference with a non-existent region | `"top-right chart"` → `"bottom-left chart"` |

After corruption, an **LLM judge** verifies the question is genuinely unanswerable from the document. Questions that slip through (still answerable) are either revised by the judge or dropped.

Output schema (`data/corrupted/<dataset>_corrupted.json`):

```json
{
  "sample_id": "docvqa-val-001",
  "document_path": "data/raw/docvqa/val/documents/xxx.png",
  "original_question": "What year is shown in the header?",
  "corrupted_question": "What year is shown in the header in 1492?",
  "original_answer": "2019",
  "corruption_type": "nlp_entity",
  "corruption_detail": "year:2019→1492",
  "page_index": 0,
  "metadata": {},
  "judge_verified": true,
  "judge_reason": "The document shows 2019; 1492 is absent."
}
```

### Phase 2 — Benchmarking

Each Vision LLM receives a document image (base64 PNG) and the corrupted question. The model should respond `UNANSWERABLE` or provide an answer. Binary classification metrics are computed per model and per corruption type.

**Model backends:**

| Backend | Key | Notes |
|---|---|---|
| `mistral` | `MISTRAL_API_KEY` | Pixtral-12B via Mistral API |
| `google` | `GEMINI_API_KEY` | Gemini 2.0 Flash |
| `openrouter` | `OPENROUTER_API_KEY` | Any multimodal model via OpenRouter |
| `llama_cpp` | — | Local GGUF model via llama-server or in-process |
| `llama` / `vllm` | — | Microservices pool (see below) |

### Phase 3 — Mitigation strategies

| Strategy | Technique |
|---|---|
| `few_shot` | Injects 2 labeled examples (answerable/unanswerable) before the question |
| `chain_of_thought` | Prompts the model to reason step-by-step before deciding |
| `knowledge_injection` | Prepends document metadata (tables, figures, layout regions) |
| `rag` | Retrieves top-k document chunks via hybrid vector+BM25 search; injects as context |
| `finetuning` | Full supervised fine-tuning with Unsloth (requires GPU + separate install) |

---

## Microservices architecture

```
Client
  │
  ▼ POST /jobs
┌─────────────────┐
│   api-gateway   │  :8000  validates job type, proxies to job-runner
└────────┬────────┘
         │ POST /jobs/dispatch
         ▼
┌─────────────────┐         Redis Stack :6379
│   job-runner    │  :8004  ──────────────────────────────────────────
└────────┬────────┘         job:{id} hash   → status, progress, result
         │                  doc:{id}:chunk  → text + vector embedding
         ├─[corrupt]──────► evaluation-svc :8003  (LLM judge)
         │
         ├─[benchmark]────► model-gateway :8001
         │                    ├── llama-pool → llama-worker-0 :8081 (GPU 0)
         │                    │               llama-worker-1 :8082 (GPU 1)
         │                    ├── vllm-pool  → vllm-worker-0  :8083 (GPU 0)
         │                    │               vllm-worker-1  :8084 (GPU 1)
         │                    ├── Mistral API  (async HTTP)
         │                    ├── Google API   (async HTTP)
         │                    └── OpenRouter   (async HTTP)
         │
         └─[mitigation]───► model-gateway :8001  (inference)
                    │
                    ├──[rag]──► document-svc :8002  (hybrid search)
                    │
                    └──[rag eval]► evaluation-svc :8003  (RAG scoring)
```

### Service responsibilities

| Service | Port | Responsibility |
|---|---|---|
| `api-gateway` | 8000 | Client entry point — validates + proxies /jobs/* to job-runner |
| `model-gateway` | 8001 | Unified inference API — GPU workers + cloud APIs, round-robin pool |
| `document-svc` | 8002 | Chunking, embedding, Redis hybrid search (vector + BM25) for RAG |
| `evaluation-svc` | 8003 | LLM-as-a-judge, RAG scorer, metrics |
| `job-runner` | 8004 | Async job dispatch (corrupt/benchmark/mitigation/index job types) |
| `llama-worker-0` | 8081 | llama.cpp server, CUDA_VISIBLE_DEVICES=0 (GGUF models) |
| `llama-worker-1` | 8082 | llama.cpp server, CUDA_VISIBLE_DEVICES=1 (GGUF models) |
| `vllm-worker-0` | 8083 | vLLM server, CUDA_VISIBLE_DEVICES=0 (HF safetensor models) |
| `vllm-worker-1` | 8084 | vLLM server, CUDA_VISIBLE_DEVICES=1 (HF safetensor models) |
| `redis-stack` | 6379 | Job state hashes + vector index for RAG |
| `jaeger` | 16686/4317 | Distributed trace backend (UI / OTLP gRPC) |
| `prometheus` | 9090 | Metrics scraper (scrapes all 5 app services every 15s) |

### Model pool configuration

The model-gateway manages two local pools configured via environment variables:

```bash
# llama.cpp pool — comma-separated worker URLs
LLAMA_URLS=http://llama-worker-0:8080,http://llama-worker-1:8080

# vLLM pool — comma-separated worker URLs
VLLM_URLS=http://vllm-worker-0:8000,http://vllm-worker-1:8000

# Which model IDs to include in fan-out when model_id is omitted
ENABLED_MODELS=llama,vllm,mistral

# vLLM model (HuggingFace model ID)
VLLM_MODEL=Qwen/Qwen2-VL-7B-Instruct
VLLM_MODEL_1=Qwen/Qwen2-VL-7B-Instruct  # override for GPU 1 if different
HF_TOKEN=hf_...                          # required for gated models
```

POST `/infer` with `model_id: null` fans out to all `ENABLED_MODELS` concurrently, aggregating results from all GPU workers and cloud APIs simultaneously.

### Redis data model

```
job:{job_id}              Hash — job lifecycle state (24h TTL)
  status:     pending | running | done | failed | cancelled
  type:       corrupt | benchmark | mitigation | index
  progress:   <int>
  total:      <int>
  result_path: data/results/...
  error:      <string>
  created_at: ISO8601

doc:{doc_id}:chunk:{n}    Hash + HNSW vector field (RediSearch index)
  text:        <chunk text>
  embedding:   <384-dim float32 vector>
  page_index:  <int>
  doc_path:    data/raw/...
```

### RAG hybrid search

Document chunks are indexed with embeddings from `sentence-transformers/all-MiniLM-L6-v2`. At retrieval time, hybrid search combines:

- **Vector KNN** (`alpha` weight) — cosine similarity via Redis HNSW index
- **BM25 full-text** (`1 - alpha` weight) — Redis FT.SEARCH
- **RRF fusion** — Reciprocal Rank Fusion merges the two ranked lists

Default `alpha=0.5` gives equal weight. Set `alpha=1.0` for pure semantic, `alpha=0.0` for pure keyword.

---

## Observability

The full stack ships with distributed tracing and metrics out of the box.

### Jaeger — distributed traces

```
http://localhost:16686
```

Every HTTP request across all 5 services generates a trace. The `job_id` is attached as a span attribute on job-runner background tasks, making the full lifecycle of a job visible as one trace from dispatch through completion.

### Prometheus — metrics

```
http://localhost:9090
```

Key metrics exposed by the services:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `inference_latency_seconds` | Histogram | `model_id` | Round-trip inference time per model |
| `inference_errors_total` | Counter | `model_id`, `error_type` | Inference failures |
| `pool_healthy_workers` | Gauge | `pool` (`llama`/`vllm`) | Live healthy worker count |
| `jobs_total` | Counter | `job_type`, `status` | Job completion counts |
| `http_requests_total` | Counter | `method`, `path`, `status` | Per-endpoint request counts (auto) |
| `http_request_duration_seconds` | Histogram | `method`, `path` | Per-endpoint latency (auto) |

### Structured logs

All services emit JSON-structured logs to stdout:

```json
{"time": "2026-06-06T10:00:00", "level": "INFO", "logger": "job-runner",
 "message": "Job dispatched", "job_id": "abc-123", "job_type": "benchmark"}
```

Query with:
```bash
docker compose logs job-runner --follow
docker compose logs model-gateway 2>&1 | jq 'select(.level == "ERROR")'
```

---

## Configuration reference

### `configs/dataset_config.yaml`

```yaml
dataset: docvqa               # docvqa | dude | mp_docvqa
corruption:
  seed: 42
  max_samples: 500            # -1 = all
  distribution:
    nlp_entity: 0.40
    element: 0.35
    layout: 0.25
quality_check:
  use_judge: true
  judge_model: gemini-2.0-flash
  confidence_threshold: 0.60  # drop verdicts below this
```

### `configs/benchmark_config.yaml`

```yaml
models:
  - backend: mistral
    model_id: pixtral-12b-2409
  - backend: google
    model_id: gemini-2.0-flash
  - backend: llama_cpp
    mode: server
    base_url: http://localhost:8080/v1
evaluation:
  metrics: [accuracy, precision, recall, f1]
  breakdown_by_corruption_type: true
```

### `configs/mitigation_config.yaml`

Configure which strategies to run and the model used for prompting.

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini / LLM judge |
| `MISTRAL_API_KEY` | — | Mistral API |
| `GEMINI_API_KEY` | — | Google AI API |
| `OPENROUTER_API_KEY` | — | OpenRouter API |
| `HF_TOKEN` | — | HuggingFace (gated vLLM models) |
| `LLAMA_URLS` | `http://llama-worker-0:8080,...` | Comma-separated llama.cpp worker URLs |
| `VLLM_URLS` | `http://vllm-worker-0:8000,...` | Comma-separated vLLM worker URLs |
| `VLLM_MODEL` | `Qwen/Qwen2-VL-7B-Instruct` | HF model ID for vLLM workers |
| `ENABLED_MODELS` | `llama,vllm` | Models included in fan-out |
| `GPU_LAYERS` | `99` | llama.cpp GPU offload layers |
| `VLLM_MAX_LEN` | `4096` | vLLM max context length |
| `LOG_LEVEL` | `INFO` | Log verbosity for all services |

---

## Development

### Run tests

```bash
# Per-service unit tests (required — flat module names conflict when run together)
PYTHONPATH=. uv run pytest services/evaluation-svc/tests/ -v
PYTHONPATH=. uv run pytest services/model-gateway/tests/ -v
PYTHONPATH=. uv run pytest services/document-svc/tests/ -v
PYTHONPATH=. uv run pytest services/job-runner/tests/ -v
PYTHONPATH=. uv run pytest services/api-gateway/tests/ -v

# Shared observability module
PYTHONPATH=. uv run pytest services/shared/tests/ -v

# Integration smoke tests (requires live docker-compose stack)
docker compose -f docker-compose.test.yml up -d
uv run pytest tests/integration/ -v --timeout=120
docker compose -f docker-compose.test.yml down
```

### Validate compose before build

```bash
docker compose config --quiet        # production
docker compose -f docker-compose.test.yml config --quiet
```

### Adding a new corruption type

1. Subclass `BaseCorruptor` in `src/dataset/corruption/`
2. Implement `corrupt(question) → CorruptedSample | None` (return `None` if nothing to match)
3. Add the class to `CORRUPTORS` in `src/dataset/pipeline.py`

### Adding a new model backend (CLI)

1. Subclass `BaseVisionModel` in `src/benchmark/models/`
2. Implement `predict_unanswerable(document_path, question, prompt_template) → PredictionResult`
3. Register the backend string in `load_model()` in `src/benchmark/run_benchmark.py`

### Adding a new model backend (microservices)

Add a client function `_infer_<name>` to `services/model-gateway/clients.py` and register it in `async_infer`'s dispatch block in the same file.

### Project specs and plans

Implementation specs and plans are tracked under `docs/superpowers/`:
- `docs/superpowers/specs/` — design documents
- `docs/superpowers/plans/` — implementation plans
