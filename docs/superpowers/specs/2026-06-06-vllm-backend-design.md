# vLLM Backend + Symmetric Local Pool Design

**Date:** 2026-06-06
**Status:** Approved

## Context

The model-gateway currently serves local inference via two hard-coded llama.cpp workers (`gpu0`, `gpu1`), each mapping to a specific URL. vLLM is being added as a second local inference pool. Rather than adding a third asymmetric pattern, this spec fixes the inconsistency: both llama.cpp and vLLM become named pools (`llama`, `vllm`) with the same round-robin behaviour, configured identically via comma-separated URL env vars.

## Goals

- Rename `gpu0`/`gpu1` → single `llama` pool (round-robin across all llama.cpp workers)
- Add `vllm` pool (round-robin across all vLLM workers)
- Both pools configured via `LLAMA_URLS` / `VLLM_URLS` env vars — no hard-coded worker count
- Zero changes to callers: `POST /infer` with `model_id: "llama"` or `model_id: "vllm"` just works
- Fan-out (`model_id: null`) includes both pools when enabled

## Non-goals

- Tensor-parallel vLLM (each worker runs on one GPU independently)
- Model hot-swapping at runtime
- Replacing llama.cpp with vLLM

---

## Architecture

```
model-gateway :8001
├── _llama_pool → [llama-worker-0 :8081, llama-worker-1 :8082]  (llama.cpp, GGUF)
└── _vllm_pool  → [vllm-worker-0  :8083, vllm-worker-1  :8084]  (vLLM, HF safetensors)

POST /infer  { model_id: "llama" } → _llama_pool.next() → llama.cpp /v1/chat/completions
POST /infer  { model_id: "vllm"  } → _vllm_pool.next()  → vLLM      /v1/chat/completions
POST /infer  { model_id: null    } → fan-out to all _ENABLED models
```

Both backends expose the same OpenAI-compatible `/v1/chat/completions` endpoint. The client function `_infer_local` is shared by both — pool determines which worker receives the request.

---

## Files Changed

| File | Change |
|---|---|
| `docker-compose.yml` | Rename `gpu0-worker`/`gpu1-worker` → `llama-worker-0`/`llama-worker-1`; add `vllm-worker-0`/`vllm-worker-1`; update model-gateway env |
| `docker-compose.test.yml` | Rename stub GPU workers; add stub `vllm-worker-0` |
| `services/model-gateway/clients.py` | Rename `_infer_llama_cpp` → `_infer_local`; drop `model_id` param (pool handles routing); vLLM uses the same function |
| `services/model-gateway/main.py` | Replace `_GPU_URLS` dict + `gpu0`/`gpu1` IDs with `_llama_pool` from `LLAMA_URLS`; add `_vllm_pool` from `VLLM_URLS`; update `_ALL_MODELS` |
| `services/model-gateway/tests/test_model_gateway.py` | Replace `gpu0`/`gpu1` references with `llama`; add vLLM routing tests |

`pool.py` — no changes. `WorkerPool` is already generic.

---

## docker-compose.yml

### llama.cpp workers (renamed, no other change)

```yaml
llama-worker-0:
  image: ghcr.io/ggerganov/llama.cpp:server-cuda
  environment:
    CUDA_VISIBLE_DEVICES: "0"
  ports:
    - "8081:8080"
  volumes:
    - ./models:/models:ro
  command: >
    --model /models/${LLAMA0_MODEL_FILE:-model.gguf}
    --host 0.0.0.0 --port 8080
    --n-gpu-layers ${GPU_LAYERS:-99}
    --ctx-size 4096
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
    interval: 10s
    timeout: 5s
    retries: 12
    start_period: 60s
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["0"]
            capabilities: [gpu]

llama-worker-1:
  image: ghcr.io/ggerganov/llama.cpp:server-cuda
  environment:
    CUDA_VISIBLE_DEVICES: "1"
  ports:
    - "8082:8080"
  volumes:
    - ./models:/models:ro
  command: >
    --model /models/${LLAMA1_MODEL_FILE:-model.gguf}
    --host 0.0.0.0 --port 8080
    --n-gpu-layers ${GPU_LAYERS:-99}
    --ctx-size 4096
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
    interval: 10s
    timeout: 5s
    retries: 12
    start_period: 60s
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["1"]
            capabilities: [gpu]
```

### vLLM workers (new)

```yaml
vllm-worker-0:
  image: vllm/vllm-openai:latest
  runtime: nvidia
  environment:
    CUDA_VISIBLE_DEVICES: "0"
    HF_HOME: /models/hf_cache
    HF_TOKEN: ${HF_TOKEN:-}
  ports:
    - "8083:8000"
  volumes:
    - ./models:/models
  command: >
    --model ${VLLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}
    --host 0.0.0.0 --port 8000
    --dtype auto
    --max-model-len ${VLLM_MAX_LEN:-4096}
    --limit-mm-per-prompt image=1
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 15s
    timeout: 5s
    retries: 20
    start_period: 120s
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["0"]
            capabilities: [gpu]

vllm-worker-1:
  image: vllm/vllm-openai:latest
  runtime: nvidia
  environment:
    CUDA_VISIBLE_DEVICES: "1"
    HF_HOME: /models/hf_cache
    HF_TOKEN: ${HF_TOKEN:-}
  ports:
    - "8084:8000"
  volumes:
    - ./models:/models
  command: >
    --model ${VLLM_MODEL_1:-${VLLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}}
    --host 0.0.0.0 --port 8000
    --dtype auto
    --max-model-len ${VLLM_MAX_LEN:-4096}
    --limit-mm-per-prompt image=1
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 15s
    timeout: 5s
    retries: 20
    start_period: 120s
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["1"]
            capabilities: [gpu]
```

`VLLM_MODEL` sets the model for both vLLM workers. Override `VLLM_MODEL_1` to run a different model on GPU 1. `HF_HOME` maps into `./models/hf_cache` so weights survive container restarts.

### model-gateway env (updated)

```yaml
model-gateway:
  environment:
    LLAMA_URLS: http://llama-worker-0:8080,http://llama-worker-1:8080
    VLLM_URLS: http://vllm-worker-0:8000,http://vllm-worker-1:8000
    MISTRAL_API_KEY: ${MISTRAL_API_KEY:-}
    GOOGLE_API_KEY: ${GOOGLE_API_KEY:-}
    OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}
    ENABLED_MODELS: ${ENABLED_MODELS:-llama,vllm}
  depends_on:
    llama-worker-0:
      condition: service_healthy
    llama-worker-1:
      condition: service_healthy
```

---

## clients.py

Remove `_infer_llama_cpp`. Both local backends share `_infer_local`:

```python
async def _infer_local(document_path: str, prompt: str, max_tokens: int, pool) -> dict:
    """Shared client for llama.cpp and vLLM — both expose OpenAI-compatible /v1/chat/completions."""
    import httpx
    url = pool.next()
    if url is None:
        raise RuntimeError("No healthy workers available in pool")
    b64 = await _image_to_b64(document_path)
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
    return {"raw_response": resp.json()["choices"][0]["message"]["content"]}
```

`async_infer` dispatch updated:

```python
async def async_infer(model_id, document_path, prompt, max_tokens=256, llama_pool=None, vllm_pool=None):
    t0 = time.monotonic()
    if model_id == "llama":
        result = await _infer_local(document_path, prompt, max_tokens, llama_pool)
    elif model_id == "vllm":
        result = await _infer_local(document_path, prompt, max_tokens, vllm_pool)
    elif model_id == "mistral":
        result = await _infer_mistral(document_path, prompt, max_tokens)
    elif model_id == "google":
        result = await _infer_google(document_path, prompt, max_tokens)
    elif model_id == "openrouter":
        result = await _infer_openrouter(document_path, prompt, max_tokens)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = result["raw_response"]
    return {
        "model_id": model_id,
        "raw_response": raw,
        "predicted_unanswerable": _parse_unanswerable(raw),
        "latency_ms": latency_ms,
    }
```

---

## main.py

```python
_LLAMA_URLS_RAW = os.getenv("LLAMA_URLS", "")
_llama_pool = WorkerPool([u.strip() for u in _LLAMA_URLS_RAW.split(",") if u.strip()])

_VLLM_URLS_RAW = os.getenv("VLLM_URLS", "")
_vllm_pool = WorkerPool([u.strip() for u in _VLLM_URLS_RAW.split(",") if u.strip()])

_ENABLED = set(os.getenv("ENABLED_MODELS", "llama,vllm").split(","))
_ALL_MODELS = ["llama", "vllm", "mistral", "google", "openrouter"]
```

`POST /infer` dispatch passes both pools:

```python
tasks = [
    clients.async_infer(mid, req.document_path, req.prompt, req.max_tokens,
                        llama_pool=_llama_pool, vllm_pool=_vllm_pool)
    for mid in targets
]
```

`GET /models` — both local backends report their pool status:

```python
pool_status = (
    _llama_pool.status() if m == "llama"
    else _vllm_pool.status() if m == "vllm"
    else None
)
```

`GET /models/{model_id}/health`:

- `"llama"` → checks each URL in `_llama_pool`, marks healthy/unhealthy
- `"vllm"` → checks each URL in `_vllm_pool`, marks healthy/unhealthy

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `LLAMA_URLS` or `VLLM_URLS` not set | Pool has 0 URLs. `pool.next()` returns `None`. `_infer_local` raises `RuntimeError`. Fan-out returns `{"model_id": "...", "error": "..."}`. |
| All workers in a pool unhealthy | Same as above. |
| Single worker down | `mark_unhealthy` removes it; remaining workers serve requests. |

---

## Tests

Updated and new tests in `services/model-gateway/tests/test_model_gateway.py`:

```python
# Existing test updated: gpu0 → llama
def test_infer_llama_routes_to_local_client():
    fake = {"model_id": "llama", "raw_response": "UNANSWERABLE",
            "predicted_unanswerable": True, "latency_ms": 100}
    with patch("clients.async_infer", return_value=fake):
        resp = client.post("/infer", json={
            "model_id": "llama",
            "document_path": "data/raw/doc.png",
            "prompt": "...",
        })
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "llama"

# New: vLLM routing
def test_infer_vllm_routes_to_local_client():
    fake = {"model_id": "vllm", "raw_response": "UNANSWERABLE",
            "predicted_unanswerable": True, "latency_ms": 80}
    with patch("clients.async_infer", return_value=fake):
        resp = client.post("/infer", json={
            "model_id": "vllm",
            "document_path": "data/raw/doc.png",
            "prompt": "...",
        })
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "vllm"

# New: empty pool error surface
def test_infer_vllm_no_workers_returns_error_in_fanout():
    with patch("clients.async_infer",
               side_effect=RuntimeError("No healthy workers available in pool")):
        resp = client.post("/infer", json={
            "model_id": "vllm",
            "document_path": "data/raw/doc.png",
            "prompt": "...",
        })
    assert resp.status_code == 200
    assert "error" in resp.json()

# Pool round-robin (unchanged — WorkerPool is generic)
def test_pool_round_robin():
    from pool import WorkerPool
    pool = WorkerPool(["http://a:8080", "http://b:8080"])
    assert pool.next() == "http://a:8080"
    assert pool.next() == "http://b:8080"
    assert pool.next() == "http://a:8080"
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `LLAMA_URLS` | `` | Comma-separated llama.cpp worker URLs |
| `LLAMA0_MODEL_FILE` | `model.gguf` | GGUF filename for llama-worker-0 |
| `LLAMA1_MODEL_FILE` | `model.gguf` | GGUF filename for llama-worker-1 |
| `GPU_LAYERS` | `99` | llama.cpp GPU offload layers |
| `VLLM_URLS` | `` | Comma-separated vLLM worker URLs |
| `VLLM_MODEL` | `Qwen/Qwen2-VL-7B-Instruct` | HF model ID for vLLM workers |
| `VLLM_MODEL_1` | falls back to `VLLM_MODEL` | Override model for vllm-worker-1 |
| `VLLM_MAX_LEN` | `4096` | Max context length (tokens) |
| `HF_TOKEN` | `` | HuggingFace token for gated models |
| `ENABLED_MODELS` | `llama,vllm` | Models included in fan-out |
