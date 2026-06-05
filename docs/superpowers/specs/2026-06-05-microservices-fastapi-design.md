# Microservices FastAPI Architecture
**Date:** 2026-06-05
**Status:** Approved

## Context

The current codebase is a three-stage CLI pipeline (corrupt → benchmark → mitigate) for a Master AI assignment benchmarking Vision LLMs on unanswerable question detection. The goal is to wrap it in a real-world, capability-based microservices architecture using FastAPI, with two NVIDIA A6000 GPUs for local inference and hybrid (local + cloud API) model support.

## Goals

- Real async REST API with job tracking (no blocking HTTP)
- Maximum throughput: parallel inference across two A6000s + cloud APIs simultaneously
- Hybrid search RAG mitigation (vector + BM25)
- Reusable platform services — not designed around assignment steps
- Portfolio-quality architecture that reflects real-world LLM platform design

## Non-goals

- Kubernetes / cloud deployment (Docker Compose on local hardware)
- Authentication / multi-tenancy
- Production database (Redis is sufficient at this scale)

---

## Service Map

| Container | Port | Responsibility |
|---|---|---|
| `api-gateway` | 8000 | Client-facing entry point — request validation, routing, rate limiting |
| `model-gateway` | 8001 | Unified LLM inference API — routes to GPU workers or cloud APIs, load balances, retries |
| `document-svc` | 8002 | Document storage, text extraction, chunking, Redis hybrid search (RAG) |
| `evaluation-svc` | 8003 | LLM-as-a-judge, metrics computation, result scoring |
| `job-runner` | 8004 | Async experiment orchestration — corruption/benchmark/mitigation as job types |
| `gpu0-worker` | 8081 | llama.cpp server, `CUDA_VISIBLE_DEVICES=0` |
| `gpu1-worker` | 8082 | llama.cpp server, `CUDA_VISIBLE_DEVICES=1` |
| `redis-stack` | 6379 | Job state (hashes) + vector index + BM25 (RediSearch) |
| `shared-volume` | — | Host bind-mount at `./data` — all services read/write files here |

### Why this split

The three pipeline stages (corrupt, benchmark, mitigate) are **job types** inside `job-runner`, not services. The real services are platform capabilities that exist independently of the research task:

- `model-gateway` — any future evaluation task reuses it without change
- `document-svc` — any RAG or document QA use case calls here
- `evaluation-svc` — any team evaluating LLM outputs calls here

---

## Data Flow

```
Client
  │
  ▼ POST /jobs  {type: "benchmark", config: {...}}
api-gateway :8000
  │
  ▼ POST /jobs
job-runner :8004
  │ generates job_id, writes {status: pending} to Redis
  │ returns job_id immediately
  │
  ├─[corrupt job]──────────────────────────────────────────────────────┐
  │  reads raw data from shared-volume                                 │
  │  runs corruption (NLPEntity / Element / Layout corruptors)         │
  │  POST /evaluate/answerability → evaluation-svc (judge verification)│
  │  writes corrupted.json to shared-volume                            │
  │  updates Redis job state                                           └──►
  │
  ├─[benchmark job]────────────────────────────────────────────────────┐
  │  reads corrupted.json from shared-volume                           │
  │  for each sample, fans out to ALL configured models concurrently:  │
  │    POST /infer → model-gateway :8001                               │
  │      ├── gpu0-worker :8081  (llama.cpp, CUDA 0)                   │
  │      ├── gpu1-worker :8082  (llama.cpp, CUDA 1)                   │
  │      ├── Mistral API  (async HTTP)                                 │
  │      ├── Google API   (async HTTP)                                 │
  │      └── OpenRouter   (async HTTP)                                 │
  │  aggregates results, writes benchmark_results.json                 │
  │  updates Redis progress counter per sample                         └──►
  │
  └─[mitigation job]───────────────────────────────────────────────────┐
     reads corrupted.json from shared-volume                           │
     for each strategy (few_shot, cot, knowledge_injection, rag):      │
       builds prompt (strategy-specific)                               │
       [rag only] POST /search → document-svc (hybrid retrieval)       │
       POST /infer → model-gateway                                      │
       POST /evaluate/rag → evaluation-svc (judge scores output)        │
     writes mitigation_results.json                                    │
     updates Redis                                                      └──►

Client polls GET /jobs/{job_id} until status == "done" | "failed"
```

---

## API Contracts

### api-gateway :8000

```
POST   /jobs                     Submit a new job
GET    /jobs/{job_id}            Poll status + progress + result path
DELETE /jobs/{job_id}            Cancel a running job
GET    /jobs                     List all jobs (paginated)
```

Job request body:
```json
{
  "type": "corrupt | benchmark | mitigation | index",
  "dataset": "docvqa | dude | mp_docvqa",
  "config": { ... }
}
```

Job response:
```json
{
  "job_id": "abc-123",
  "status": "pending | running | done | failed",
  "progress": { "current": 42, "total": 200 },
  "result_path": "data/results/benchmark/abc-123.json",
  "error": null,
  "created_at": "2026-06-05T10:00:00Z"
}
```

### model-gateway :8001

```
POST   /infer                    Run inference on a document + question
GET    /models                   List available models and their status
GET    /models/{id}/health       Check if a specific backend is reachable
```

Infer request:
```json
{
  "model_id": "gpu0 | gpu1 | mistral | google | openrouter",
  "document_path": "data/raw/docvqa/val/documents/xxx.png",
  "prompt": "...",
  "max_tokens": 512
}
```

Infer response:
```json
{
  "model_id": "gpu0",
  "raw_response": "UNANSWERABLE",
  "predicted_unanswerable": true,
  "latency_ms": 1240
}
```

The gateway fans out a single `/infer` call to all models when `model_id` is omitted, returning a list of results — used by `job-runner` for benchmark jobs.

### document-svc :8002

```
POST   /documents/index          Chunk + embed a dataset, load into Redis
POST   /search                   Hybrid search (vector + BM25), returns top-k chunks
GET    /documents/{doc_id}       Retrieve document metadata
DELETE /documents/index          Clear the vector index
```

Search request:
```json
{
  "query": "What is the invoice total?",
  "top_k": 5,
  "alpha": 0.5
}
```

`alpha` blends vector score (1.0) vs BM25 score (0.0). Default 0.5 = equal weight.

Search response:
```json
{
  "chunks": [
    {
      "doc_id": "xxx",
      "page_index": 0,
      "text": "Invoice total: $1,234.00",
      "score": 0.91
    }
  ]
}
```

### evaluation-svc :8003

```
POST   /evaluate/answerability   Judge whether a question is unanswerable from a document
POST   /evaluate/rag             Score a model's RAG-augmented answer
POST   /evaluate/metrics         Compute F1/precision/recall from predictions + labels
```

Answerability request:
```json
{
  "question": "What was the revenue in 1987?",
  "document_path": "data/raw/docvqa/val/documents/xxx.png",
  "confidence_threshold": 0.5
}
```

Answerability response:
```json
{
  "verdict": "unanswerable",
  "confidence": 0.92,
  "reason": "The document contains 2019 revenue data only.",
  "suggested_question": null
}
```

RAG evaluation request:
```json
{
  "question": "What was the revenue in 1987?",
  "retrieved_context": ["Invoice total: $1,234.00 (2019)"],
  "model_answer": "UNANSWERABLE",
  "ground_truth": "unanswerable"
}
```

### job-runner :8004

Internal service — not exposed through the gateway directly. Receives job dispatch from `api-gateway`, runs experiments as background asyncio tasks.

```
POST   /jobs/dispatch            Receive job from gateway, start background task
GET    /jobs/{job_id}/logs       Stream logs for a running job (SSE)
```

---

## Redis Schema

All keys namespaced to avoid collisions:

```
job:{job_id}                    Hash — job state
  status:     pending|running|done|failed
  type:       corrupt|benchmark|mitigation|index
  progress:   42
  total:      200
  result_path: data/results/...
  error:      ""
  created_at: ISO8601
  ttl:        86400 (24h auto-expiry)

doc:{doc_id}:chunk:{n}          Hash + HNSW vector field (RediSearch index)
  text:        "Invoice total: $1,234..."
  embedding:   [0.12, -0.34, ...]   (384-dim, sentence-transformers)
  page_index:  0
  doc_path:    data/raw/docvqa/...
```

Redis image: `redis/redis-stack:latest` (includes RediSearch, RedisJSON).

---

## Benchmark Parallelism

`job-runner` reads the corrupted dataset and dispatches inference concurrently using `asyncio.gather`. The `model-gateway` load-balances across GPU workers using a round-robin pool with health checks. API-based models are called concurrently via `httpx.AsyncClient`.

```python
# inside job-runner, benchmark job
async def run_benchmark(dataset: list[dict], model_ids: list[str]):
    async with httpx.AsyncClient() as client:
        tasks = [
            infer_all_models(client, item, model_ids)
            for item in dataset
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
```

With two A6000s + three API models, each sample is processed by 5 models concurrently. For 1000 samples, this is 5000 inference calls running in parallel (GPU-bound by llama.cpp capacity, network-bound for APIs).

---

## RAG Mitigation Flow

1. **Index** (once): `POST /jobs {type: "index", dataset: "docvqa"}` → `document-svc` chunks all documents, embeds with `sentence-transformers/all-MiniLM-L6-v2`, loads into Redis HNSW index.
2. **Retrieve**: for each corrupted question, `job-runner` calls `POST /search` on `document-svc` with `alpha=0.5` (hybrid).
3. **Augment**: retrieved chunks are injected into the prompt via the existing `knowledge_injection.py` strategy.
4. **Infer**: augmented prompt → `model-gateway`.
5. **Evaluate**: model answer + retrieved context → `POST /evaluate/rag` on `evaluation-svc`.

---

## Project Structure (new layout)

```
services/
├── api-gateway/
│   ├── main.py
│   └── Dockerfile
├── model-gateway/
│   ├── main.py
│   ├── pool.py          (round-robin worker pool)
│   └── Dockerfile
├── document-svc/
│   ├── main.py
│   ├── indexer.py       (chunking + embedding)
│   ├── search.py        (hybrid search, RRF fusion)
│   └── Dockerfile
├── evaluation-svc/
│   ├── main.py
│   ├── judge.py         (wraps existing llm_judge.py)
│   ├── metrics.py       (wraps existing metrics.py)
│   └── Dockerfile
├── job-runner/
│   ├── main.py
│   ├── jobs/
│   │   ├── corrupt.py   (wraps existing pipeline.py)
│   │   ├── benchmark.py (wraps existing run_benchmark.py)
│   │   └── mitigation.py(wraps existing run_mitigation.py)
│   └── Dockerfile
docker-compose.yml
src/                     (existing package — imported by services)
configs/
data/
```

Existing `src/` code is **not rewritten** — each service imports and wraps the existing modules. The services are thin HTTP adapters around the existing business logic.

---

## Error Handling

- **model-gateway**: if a GPU worker is unreachable, retries 3x with exponential backoff, then marks the model as unhealthy and skips it for the remainder of the job. Job continues with remaining models.
- **job-runner**: exceptions in background tasks are caught, written to `job.error` in Redis, and status set to `failed`. Partial results written to disk before failure.
- **evaluation-svc**: judge failures (API timeout, malformed response) return `{verdict: null, confidence: 0}` — the pipeline treats these as unverified rather than crashing.
- All services return RFC 7807 problem JSON on errors: `{type, title, status, detail}`.

---

## Testing Strategy

- **Unit**: each service's core logic tested in isolation (existing test patterns apply).
- **Integration**: `docker-compose.test.yml` spins up all services with a small fixture dataset (10 samples). A single test script hits `POST /jobs`, polls until done, asserts results file exists and metrics are non-zero.
- **GPU workers**: tested independently with a small GGUF model; CI can skip GPU tests when no CUDA device is available.

---

## Dependencies to Add

```toml
# per service (via uv)
fastapi>=0.111
uvicorn[standard]>=0.29
httpx>=0.27          # async HTTP client
redis[hiredis]>=5.0  # Redis client
redisvl>=0.2         # Redis vector library (RediSearch)
sentence-transformers>=3.0  # embeddings for document-svc
```
