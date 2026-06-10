# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Master AI assignment: build a benchmark to test Vision LLMs on *unanswerable question detection* from document images. Three sequential phases: (1) corrupt a DocVQA-style dataset to produce unanswerable questions, (2) benchmark Vision LLMs on that dataset, (3) test in-context learning mitigations.

Deliverable is a zip with Python scripts + max-5-page report.

## Environment

```bash
uv sync                    # install core deps + editable package
uv sync --extra dev        # also installs jupyter + spacy
uv run python -m spacy download en_core_web_sm   # required after uv sync --extra dev
# API keys are stored in .env (gitignored) — source before running:
export $(grep -v '^#' .env | xargs)
```

Python 3.11 is pinned (`.python-version`). Always prefix commands with `uv run` or activate `.venv`.

## Running the three pipeline stages

```bash
# Part 1 – generate corrupted (unanswerable) dataset
uv run python -m src.dataset.pipeline \
  --dataset docvqa \
  --data_dir data/raw/docvqa \
  --output_dir data/corrupted \
  --config configs/dataset_config.yaml
  [--no_judge]   # skip LLM-as-a-judge quality filter

# Part 2 – benchmark Vision LLMs
uv run python -m src.benchmark.run_benchmark \
  --corrupted_dataset data/corrupted/docvqa_corrupted.json \
  --config configs/benchmark_config.yaml \
  --output_dir results/benchmark

# Part 3 – mitigation strategies
uv run python -m src.mitigation.run_mitigation \
  --corrupted_dataset data/corrupted/docvqa_corrupted.json \
  --baseline_results results/benchmark/benchmark_results.json \
  --config configs/mitigation_config.yaml \
  --output_dir results/mitigation
```

The four entry points: `uv run download-data`, `uv run corrupt-dataset`, `uv run run-benchmark`, `uv run run-mitigation`.

## Architecture

### Data flow

```text
data/raw/{docvqa,dude,mp_docvqa}/
        │
        ▼  src/dataset/pipeline.py
        │  ├── loaders/  (BaseLoader → QASample stream)
        │  ├── corruption/  (BaseCorruptor → CorruptedSample)
        │  └── quality_check/llm_judge.py  (Claude Vision rejects still-answerable questions)
        ▼
data/corrupted/{dataset}_corrupted.json   ← list[dict] with original + corrupted question
        │
        ▼  src/benchmark/run_benchmark.py
        │  ├── models/  (BaseVisionModel → PredictionResult)
        │  └── evaluation/metrics.py  (accuracy/precision/recall/F1)
        ▼
results/benchmark/benchmark_results.json
        │
        ▼  src/mitigation/run_mitigation.py
        │  └── strategies/  (few_shot | chain_of_thought | knowledge_injection)
        ▼
results/mitigation/mitigation_results.json
```

### Key abstractions

- **`QASample`** (`src/dataset/loaders/base_loader.py`) — canonical record passed between loader and pipeline; fields: `sample_id`, `document_path`, `question`, `answer`, `page_index`, `metadata`.
- **`BaseCorruptor`** (`src/dataset/corruption/base_corruptor.py`) — `corrupt(question) → CorruptedSample | None`; returns `None` when the corruptor has nothing to match (pipeline tries next one). Three concrete implementations: `NLPEntityCorruptor`, `ElementCorruptor`, `LayoutCorruptor`. `NLPEntityCorruptor` uses spaCy NER + Wikipedia category API to find peer-entity replacements (no key needed; ~1–2 s latency per unique entity, cached per run). Pass `web_lookup=False` for offline/fast mode (static fallback pools).
- **`BaseVisionModel`** (`src/benchmark/models/base_model.py`) — `predict_unanswerable(document_path, question, prompt_template) → PredictionResult`. Active backends: `MistralModel` (`MISTRAL_API_KEY`), `GoogleModel` (`GOOGLE_API_KEY`), `OpenRouterModel` (`OPENROUTER_API_KEY`), `LlamaCppModel` (server or direct GGUF). All pass images as base64 PNG.
- **Mitigation strategies** (`src/mitigation/strategies/`) are plain functions that take a question string and return a fully-formed prompt string. The runner in `run_mitigation.py` maps strategy name → function via `_STRATEGIES` dict.

### Dataset JSON schema

Each record in `data/corrupted/*.json`:

```json
{
  "sample_id": "...",
  "document_path": "data/raw/docvqa/val/documents/xxx.png",
  "original_question": "...",
  "corrupted_question": "...",
  "original_answer": "...",
  "corruption_type": "nlp_entity | element | layout",
  "corruption_detail": "year:2019→1987",
  "page_index": 0,
  "metadata": {},
  "judge_verified": true,
  "judge_reason": "..."
}
```

### Configuration

All three stages are driven by YAML files in `configs/`. Edit `configs/dataset_config.yaml` to change corruption distribution, `max_samples`, and `window_size` (for multi-page sliding window). Edit `configs/benchmark_config.yaml` to switch between mistral/google/openrouter/llama_cpp backends.

### IDE / type-checker notes

**Pylance false positives:** optional deps (`spacy`) and declared deps (`requests`) may show "cannot be resolved" in the IDE — suppress with `# type: ignore[import-untyped]`. Not a runtime issue.

### Adding a new corruptor

1. Subclass `BaseCorruptor` in `src/dataset/corruption/`.
2. Implement `corrupt()` returning `CorruptedSample | None` and set `corruption_type`.
3. Add the class to the `CORRUPTORS` list in `src/dataset/pipeline.py`.

### Adding a new model backend

1. Subclass `BaseVisionModel` in `src/benchmark/models/`.
2. Implement `predict_unanswerable()` and `name()`.
3. Register the backend string in `load_model()` inside `src/benchmark/run_benchmark.py`.

## Microservices layer (services/)

Five FastAPI services wrap the `src/` pipeline. `src/` code is NOT rewritten — services import and wrap it.

| Service | Port | Responsibility |
| --- | --- | --- |
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

### Running the stack

```bash
docker compose up -d                               # production (needs models/ dir with GGUF files)
docker compose -f docker-compose.test.yml up -d   # testing (stub GPU workers, no GGUF needed)
docker compose config --quiet                      # validate compose YAML syntax (fast, no build)
```

### Testing services

Each service has `services/<name>/tests/`. Do NOT run all services together — they share flat module names (`main.py`, `state.py`, etc.) that conflict in one pytest process.

```bash
# Run per service (required):
PYTHONPATH=. uv run pytest services/evaluation-svc/tests/ -v
PYTHONPATH=. uv run pytest services/model-gateway/tests/ -v
PYTHONPATH=. uv run pytest services/document-svc/tests/ -v
PYTHONPATH=. uv run pytest services/job-runner/tests/ -v
PYTHONPATH=. uv run pytest services/api-gateway/tests/ -v

# Integration smoke tests (requires live docker-compose stack):
uv run pytest tests/integration/ -v --timeout=120
```

`uv run pytest` (no args) only discovers `tests/` — see `[tool.pytest.ini_options]` in `pyproject.toml`.

### Services dependencies

```bash
uv sync --extra services   # installs fastapi, uvicorn, httpx, redis, redisvl, sentence-transformers, fakeredis, opentelemetry-*, prometheus-*
```

### Observability

- Jaeger UI: `http://localhost:16686` — distributed traces across all 5 services
- Prometheus: `http://localhost:9090` — inference latency, pool health, job counters
- Structured JSON logs to stdout — query with `docker compose logs <svc>`
- Shared module: `services/shared/observability.py` — `setup_tracing(name)`, `setup_metrics(app)`, `get_logger(name)`
- Key metrics: `inference_latency_seconds` (histogram, label: model_id), `pool_healthy_workers` (gauge, label: pool), `jobs_total` (counter, labels: job_type/status)

### model-gateway pool config

- `model_id: "llama"` → round-robins across `LLAMA_URLS` (comma-separated llama.cpp worker URLs)
- `model_id: "vllm"` → round-robins across `VLLM_URLS` (comma-separated vLLM worker URLs)
- `model_id: null` → fan-out to all models listed in `ENABLED_MODELS`
- vLLM uses HuggingFace safetensor format — set `VLLM_MODEL` to HF model ID; `HF_TOKEN` needed for gated models

### Known gotchas

- **redisvl + redis 8.x**: `redis 8.0` dropped `redis.commands.search.indexDefinition`; `redisvl` imports it at module load. Fix: lazy-import redisvl inside function bodies (already done in `document-svc`).
- **Docker build context**: all Dockerfiles assume repo root as build context. Use `docker compose build`, not `docker build .` from within a service directory.
- **Redis Stack**: requires `redis/redis-stack:latest` (not plain `redis`) for RediSearch + vector support.
- **GPU workers**: model-gateway `depends_on` llama and vllm workers with `condition: service_healthy`. llama.cpp takes 30–90 s to load a model — don't expect immediate inference on cold start.
- **Nested Docker Compose var interpolation broken**: `${VAR1:-${VAR2:-default}}` is treated as a literal string. Use flat defaults or resolve in `.env`.
- **`runtime: nvidia` + `deploy.resources` are redundant**: for GPU containers, use only `deploy.resources.reservations.devices`.
- **pytest `sys.path` for shared module**: service test files need TWO inserts — `sys.path.insert(0, "..")` for the service dir AND `sys.path.insert(0, "../..")` for `services/` (required for `from shared.observability import ...`).
- **prometheus_client 0.20+ strips `_total` from Counter `.name`**: `Counter("foo_total", ...)` has `.name = "foo"` in `REGISTRY.collect()` — the `_total` suffix only appears in sample names.
- **prometheus_client `Duplicated timeseries` on module reload**: wrap module-level metric definitions in `try/except ValueError` when tests use `importlib.reload()`.
- **OTel `DEADLINE_EXCEEDED` in tests**: harmless — `BatchSpanProcessor` silently drops spans when Jaeger is unreachable; `setup_tracing` is already wrapped in try/except.
