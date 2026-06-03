# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Master AI assignment: build a benchmark to test Vision LLMs on *unanswerable question detection* from document images. Three sequential phases: (1) corrupt a DocVQA-style dataset to produce unanswerable questions, (2) benchmark Vision LLMs on that dataset, (3) test in-context learning mitigations.

Deadline: **20 June 2026** — deliverable is a zip with Python scripts + max-5-page report.

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
