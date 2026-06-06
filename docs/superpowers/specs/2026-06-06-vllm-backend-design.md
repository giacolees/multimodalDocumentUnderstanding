# vLLM Inference Backend Design
**Date:** 2026-06-06
**Status:** Approved

## Context

The model-gateway currently serves local inference via two llama.cpp workers (`gpu0`, `gpu1`) backed by GGUF models. vLLM is being added as a second local inference pool (`vllm`) for HuggingFace safetensor models, enabling higher-throughput continuous batching alongside the existing GGUF path.

## Goals

- Add `model_id: "vllm"` as a new logical pool in model-gateway
- Round-robin across one or two vLLM worker containers, exactly like the existing GPU pool
- Zero changes to callers — `POST /infer` with `model_id: "vllm"` just works
- llama.cpp workers (`gpu0`, `gpu1`) unchanged

## Non-goals

- Tensor-parallel vLLM (each worker runs on one GPU independently)
- Model hot-swapping at runtime
- Replacing llama.cpp

---

## Architecture

```
model-gateway :8001
├── _gpu_pool   → [gpu0-worker :8081, gpu1-worker :8082]  (llama.cpp, GGUF)
└── _vllm_pool  → [vllm-worker-0 :8083, vllm-worker-1 :8084]  (vLLM, HF safetensors)

POST /infer  { model_id: "gpu0"  } → _gpu_pool.next()   → llama.cpp /v1/chat/completions
POST /infer  { model_id: "gpu1"  } → _gpu_pool.next()   → llama.cpp /v1/chat/completions
POST /infer  { model_id: "vllm"  } → _vllm_pool.next()  → vLLM      /v1/chat/completions
POST /infer  { model_id: null    } → fan-out to all _ENABLED models
```

Both llama.cpp and vLLM expose the same OpenAI-compatible `/v1/chat/completions` endpoint, so the client code is structurally identical.

---

## Files Changed

| File | Change |
|---|---|
| `docker-compose.yml` | Add `vllm-worker-0` and `vllm-worker-1` services |
| `docker-compose.test.yml` | Add stub `vllm-worker-0` service (Python echo server) |
| `services/model-gateway/clients.py` | Add `_infer_vllm()` function |
| `services/model-gateway/main.py` | Add `_vllm_pool`, register `"vllm"` in `_ALL_MODELS`, route dispatch |
| `services/model-gateway/tests/test_model_gateway.py` | Add test for `model_id: "vllm"` routing |

`pool.py` — no changes. `WorkerPool` is already generic.

---

## docker-compose.yml — vLLM workers

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

`VLLM_MODEL` sets the model for both workers. Override `VLLM_MODEL_1` to run a different model on GPU 1. `HF_HOME` maps into `./models/hf_cache` so weights survive container restarts. `HF_TOKEN` is required for gated models.

model-gateway env updated:
```yaml
VLLM_URLS: http://vllm-worker-0:8000,http://vllm-worker-1:8000
```

---

## clients.py — `_infer_vllm`

```python
async def _infer_vllm(document_path: str, prompt: str, max_tokens: int, pool) -> dict:
    import httpx
    url = pool.next()
    if url is None:
        raise RuntimeError("No healthy vLLM workers available")
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

Structurally identical to `_infer_llama_cpp`. Key differences:
- No model name in payload (vLLM is started with a fixed model)
- Uses `_vllm_pool` not `_gpu_pool`
- Timeout 180 s (vLLM may queue requests under load)

---

## main.py — pool + dispatch

```python
# Existing GPU pool (unchanged)
_GPU_URLS = {
    "gpu0": os.getenv("GPU0_URL", "http://gpu0-worker:8080"),
    "gpu1": os.getenv("GPU1_URL", "http://gpu1-worker:8080"),
}
_gpu_pool = WorkerPool(list(_GPU_URLS.values()))

# New vLLM pool — empty list = pool disabled (no error unless "vllm" is requested)
_VLLM_URLS_RAW = os.getenv("VLLM_URLS", "")
_vllm_pool = WorkerPool(
    [u.strip() for u in _VLLM_URLS_RAW.split(",") if u.strip()]
)

_ENABLED = set(os.getenv("ENABLED_MODELS", "gpu0,gpu1").split(","))
_ALL_MODELS = ["gpu0", "gpu1", "vllm", "mistral", "google", "openrouter"]
```

Dispatch in `async_infer`:
```python
if model_id in ("gpu0", "gpu1"):
    result = await clients.async_infer(model_id, document_path, prompt, max_tokens, _gpu_pool)
elif model_id == "vllm":
    result = await clients.async_infer(model_id, document_path, prompt, max_tokens, _vllm_pool)
elif model_id in ("mistral", "google", "openrouter"):
    result = await clients.async_infer(model_id, document_path, prompt, max_tokens, None)
```

`GET /models` response for `vllm`:
```json
{"model_id": "vllm", "enabled": true, "pool_status": [
    {"url": "http://vllm-worker-0:8000", "healthy": true},
    {"url": "http://vllm-worker-1:8000", "healthy": true}
]}
```

`GET /models/vllm/health` — same health-check pattern as GPU workers (HTTP GET `/health`).

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `VLLM_URLS` not set | `_vllm_pool` has 0 URLs. `pool.next()` returns `None`. `_infer_vllm` raises `RuntimeError`. Fan-out returns `{"model_id": "vllm", "error": "No healthy vLLM workers available"}`. |
| All vLLM workers unhealthy | Same as above. |
| Single worker down | `mark_unhealthy` removes it from rotation; remaining worker(s) serve requests. |
| vLLM model not loaded yet | Request times out (180 s). Worker marked unhealthy after retries. |

---

## Testing

**Unit test added** to `services/model-gateway/tests/test_model_gateway.py`:

```python
def test_infer_vllm_routes_to_vllm_client():
    fake_result = {
        "model_id": "vllm",
        "raw_response": "UNANSWERABLE",
        "predicted_unanswerable": True,
        "latency_ms": 80,
    }
    with patch("clients.async_infer", return_value=fake_result):
        resp = client.post("/infer", json={
            "model_id": "vllm",
            "document_path": "data/raw/doc.png",
            "prompt": "Is this answerable?",
        })
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "vllm"


def test_infer_vllm_no_workers_returns_error():
    """Empty VLLM_URLS → pool has no URLs → error in fan-out result, not HTTP 500."""
    with patch("clients.async_infer", side_effect=RuntimeError("No healthy vLLM workers available")):
        resp = client.post("/infer", json={
            "model_id": "vllm",
            "document_path": "data/raw/doc.png",
            "prompt": "...",
        })
    assert resp.status_code == 200
    assert "error" in resp.json()
```

**docker-compose.test.yml** gets a stub `vllm-worker-0`:
```yaml
vllm-worker-0:
  image: python:3.11-slim
  command: >
    python -c "
    import json, http.server, socketserver
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers()
            self.wfile.write(b'ok')
        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'choices':[{'message':{'content':'UNANSWERABLE'}}]}).encode())
        def log_message(self, *a): pass
    with socketserver.TCPServer(('',8000),H) as s: s.serve_forever()
    "
  ports:
    - "8085:8000"
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `VLLM_MODEL` | `Qwen/Qwen2-VL-7B-Instruct` | HF model ID for both vLLM workers |
| `VLLM_MODEL_1` | falls back to `VLLM_MODEL` | Override model for GPU 1 worker |
| `VLLM_MAX_LEN` | `4096` | Max context length (tokens) |
| `VLLM_URLS` | `` (empty) | Comma-separated vLLM worker URLs for model-gateway |
| `HF_TOKEN` | `` (empty) | HuggingFace token for gated models |
| `ENABLED_MODELS` | `gpu0,gpu1` | Add `vllm` here to include in fan-out |
