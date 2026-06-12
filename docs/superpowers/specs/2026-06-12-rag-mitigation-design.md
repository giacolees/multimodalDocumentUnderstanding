# RAG Mitigation Strategy + Mitigation Module Refactor

Date: 2026-06-12
Status: Approved design — pending implementation plan

## Goal

Add a Retrieval-Augmented Generation (RAG) mitigation strategy for unanswerable-question
detection, and refactor `src/mitigation/` so each strategy is a self-contained file behind a
single interface with a thin runner.

The RAG idea for *this* task: retrieve the document's own content most relevant to the
question and inject it as grounding. If the document demonstrably lacks what the question
asks about, the model is steered to answer `UNANSWERABLE`. Document text is produced by the
vision model itself (self-transcription), so no new OCR engine dependency is introduced.

## Design decisions (locked)

- **RAG target:** document content (vision-model self-transcription), not external knowledge
  or dynamic few-shot examples.
- **Retrieval:** hybrid — BM25 sparse + SentenceTransformer dense, fused with Reciprocal
  Rank Fusion (RRF, k=60). BM25 directly flags corrupted entity/element names absent from
  the document; dense handles semantic similarity.
- **Signal mode:** inject retrieved context into the prompt and let the model decide
  (no hard similarity-threshold gate in v1).
- **Integration:** refactor the mitigation module — strategies become isolated files behind
  a `MitigationStrategy` interface; `run_mitigation.py` becomes a thin runner; the eval +
  MLflow logging block moves into a shared `evaluation.py`.
- **Transcription API:** add a generic `generate()` to `VllmModel` (and a default on
  `BaseVisionModel` that raises `NotImplementedError`).

## 1. Refactored module architecture

```text
src/mitigation/
├── run_mitigation.py          THIN runner: parse args, load config/dataset/baseline,
│                              build model once, dispatch strategies, persist results
├── evaluation.py              NEW: evaluate_strategy(strategy, dataset, model, baseline, …)
│                              — per-item loop + metrics + all MLflow logging
│                              (lifted verbatim from today's inline block)
├── registry.py                NEW: STRATEGIES = {"few_shot": FewShotStrategy, …}
└── strategies/
    ├── base.py                NEW: MitigationStrategy ABC
    ├── few_shot.py            FewShotStrategy (wraps existing build_few_shot_prompt)
    ├── chain_of_thought.py    ChainOfThoughtStrategy
    ├── knowledge_injection.py KnowledgeInjectionStrategy
    ├── rag.py                 NEW: RagStrategy + RagRetriever
    └── finetuning.py          unchanged (different shape; runner keeps its branch)
```

### Strategy interface (`strategies/base.py`)

```python
class MitigationStrategy(ABC):
    name: str

    def prepare(self, dataset: list[dict], model: BaseVisionModel) -> None:
        """Optional one-time setup. No-op for prompt strategies; RAG may use it."""

    @abstractmethod
    def build_prompt(self, item: dict, model: BaseVisionModel) -> str:
        ...
```

- Prompt strategies (`few_shot`, `chain_of_thought`, `knowledge_injection`) ignore `model`
  and just format their template around `item["corrupted_question"]`. Existing pure builder
  functions (`build_few_shot_prompt`, `build_cot_prompt`,
  `build_knowledge_injection_prompt`) are kept and wrapped by the classes.
- `RagStrategy` uses `model` inside `build_prompt` for self-transcription + retrieval.

### Shared evaluation (`evaluation.py`)

`evaluate_strategy(strategy, dataset, model, baseline_metrics, model_id,
corrupted_dataset_path) -> dict` contains the loop currently inlined in
`run_mitigation.run_mitigation`:

- for each item: `prompt = strategy.build_prompt(item, model)` →
  `model.predict_unanswerable(document_path, corrupted_question, prompt)`
- collect preds/labels/records/inference_times
- `compute_metrics`, `compute_per_type_metrics`
- MLflow run: tags, params, dataset input, prompt artifact, all metrics + deltas vs baseline,
  per-type metrics, confusion-matrix figure, per-type F1 bar chart.

This is a **relocation, not a behavior change** — same metrics, same artifacts, same MLflow
experiment (`"mitigation"`). It returns `{"records": …, "metrics": metrics.__dict__}`.

### Runner (`run_mitigation.py`)

Slimmed to:

1. parse args, load YAML config.
2. load corrupted dataset + baseline metrics.
3. instantiate the vision model once via `_load_model` (kept here or moved to registry).
4. for each requested strategy name in the prompt-based set: look up class in
   `registry.STRATEGIES`, instantiate (passing its config sub-dict), call
   `strategy.prepare(dataset, model)`, then
   `results[name] = evaluation.evaluate_strategy(...)`.
5. keep the `if "finetuning" in requested` branch (finetuning trains; it is not a per-item
   prompt strategy).
6. persist `results/mitigation/mitigation_results.json`.

`finetuning` remains separate by design — it has a fundamentally different shape and is not
forced under the `MitigationStrategy` interface.

## 2. RAG strategy internals (`strategies/rag.py`)

Two passes; transcription cached to disk so re-runs are cheap.

```python
class RagRetriever:
    def __init__(self, embed_model, top_k, chunk_max_chars,
                 transcribe_max_tokens, cache_dir): ...

    def transcribe(self, item, model) -> str:
        # cache key = sanitized(document_path) + f"_p{page_index}"
        # read {cache_dir}/<key>.txt if present;
        # else model.generate(document_path, _TRANSCRIBE_PROMPT,
        #                      page_indices=[page_index],
        #                      max_tokens=transcribe_max_tokens) → write cache → return

    def chunks(self, text) -> list[str]:
        # split on blank lines / newlines, greedily packed into chunks of <= chunk_max_chars

    def retrieve(self, item, question, model) -> list[str]:
        # transcribe → chunk
        # SPARSE: BM25Index(chunks).get_scores(question.split()) → rank list
        # DENSE:  SentenceTransformer.encode(chunks + [question]) → cosine sim → rank list
        # FUSION: RRF score(chunk) = 1/(k+rank_sparse) + 1/(k+rank_dense), k=60
        # return top_k chunks by RRF score


class RagStrategy(MitigationStrategy):
    name = "rag"

    def __init__(self, config: dict):
        self.retriever = RagRetriever(
            embed_model=config.get("embed_model",
                                   "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=config.get("top_k", 4),
            chunk_max_chars=config.get("chunk_max_chars", 400),
            transcribe_max_tokens=config.get("transcribe_max_tokens", 1024),
            cache_dir=config.get("cache_dir", "data/ocr_cache"),
        )
        self._embedder = None  # lazy-loaded SentenceTransformer, shared across items

    def build_prompt(self, item, model) -> str:
        chunks = self.retriever.retrieve(item, item["corrupted_question"], model)
        context = "\n".join(f"- {c}" for c in chunks)
        return _RAG_TEMPLATE.format(context=context, question="{question}")
```

- **Hybrid retrieval:** two rankers over the same chunk list, fused with RRF:
  - *BM25 (sparse):* `BM25Okapi(tokenized_chunks)` from `rank-bm25`. Directly catches
    corrupted entity/element names (`"Table 5"`, `"1964"`) that are literally absent from
    the document — the core unanswerability signal for `nlp_entity` and `element` types.
  - *Dense:* one `SentenceTransformer` instance (`all-MiniLM-L6-v2`) loaded lazily and
    reused across all items. Handles semantic similarity.
  - *RRF:* `score(c) = 1/(60+rank_bm25(c)) + 1/(60+rank_dense(c))` — parameter-free,
    robust, no tuning needed.
- Per-document chunk embeddings and BM25 indices are computed in memory each run (fast);
  only transcription **text** is persisted to `cache_dir`.
- **`_TRANSCRIBE_PROMPT`:** instructs the model to transcribe all visible text from the page
  faithfully, preserving tables/labels, no commentary.
- **`_RAG_TEMPLATE`:** grounding frame — e.g. *"Here are the passages retrieved from this
  document most relevant to the question:\n{context}\n\nYou are also shown the full document
  image. If the answer is not supported by the document, respond UNANSWERABLE; otherwise
  provide the answer.\n\nQuestion: {question}"*. The literal `{question}` placeholder is left
  intact so `VllmModel.predict_unanswerable` fills it via its existing
  `prompt_template.format(question=question)`.

### Scope / YAGNI

- v1 transcribes a **single page** (`page_index`, defaulting to 0), matching today's
  mitigation loop which calls `predict_unanswerable` without `page_indices`.
- Multi-page windows for `mp_docvqa` (via `metadata.window_pages`) are a noted future
  extension, not in v1.
- No similarity-threshold short-circuit in v1 (signal mode = inject-and-decide).

## 3. `generate()` on the model interface

```python
# base_model.py — default keeps non-vLLM backends valid
def generate(self, document_path, prompt, page_indices=None, max_tokens=1024) -> str:
    raise NotImplementedError(f"{self.name()} does not support generate()")

# vllm_model.py — real implementation
#   reuses _load_image_b64 and the same /v1/chat/completions POST shape as
#   predict_unanswerable, but returns the raw completion text (no _parse_unanswerable),
#   temperature 0.0, max_tokens from the argument.
```

`generate()` is added to `BaseVisionModel` as a concrete default (raises) rather than an
abstractmethod, so existing backends remain instantiable without change. Only `VllmModel`
(the primary backend) implements it in this work.

## 4. Configuration + dependency

`configs/mitigation_config.yaml`:

```yaml
strategies:
  - few_shot
  - chain_of_thought
  - knowledge_injection
  - rag

rag:
  embed_model: sentence-transformers/all-MiniLM-L6-v2
  top_k: 4
  chunk_max_chars: 400
  transcribe_max_tokens: 1024
  cache_dir: data/ocr_cache
```

`pyproject.toml`: add a new optional extra so RAG's embedding dependency is opt-in
(`sentence-transformers>=3.0` is already declared under the `services` extra):

```toml
[project.optional-dependencies]
mitigation = [
    "sentence-transformers>=3.0",
    "rank-bm25>=0.2",
]
```

Install with `uv sync --extra mitigation`.

## 5. Testing

Extend `tests/`:

1. `RagRetriever.chunks()` packs text into chunks each `<= chunk_max_chars` and loses no
   content.
2. `RagRetriever.retrieve()` ranks an obviously-relevant chunk above an irrelevant one for
   both the BM25 path and the dense path, and that the RRF fusion selects the right top-k
   (real MiniLM + BM25Okapi, or a stubbed embedder injected for the dense path).
3. Transcription cache hit: second call for the same document does **not** invoke
   `model.generate` again (mock model, assert call count == 1).
4. `evaluate_strategy` runs end-to-end on a 2-item dataset with a stub model and a trivial
   strategy, returns metrics, and writes the expected MLflow params/metrics — guarding that
   the refactor preserved logging behavior.

Tests use the stub/mock pattern already in `tests/test_mlflow_tracking.py` and avoid network
or GPU.

## Out of scope

- Multi-page transcription for `mp_docvqa` windows.
- Similarity-threshold gating / model short-circuit.
- Persisting chunk embeddings (only transcription text is cached).
- Changing `finetuning` to the `MitigationStrategy` interface.
- New backends implementing `generate()` beyond `VllmModel`.
