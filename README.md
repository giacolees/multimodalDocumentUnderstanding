# Multimodal Document Understanding — Unanswerable Question Detection

Benchmark for testing Vision LLMs on *unanswerable question detection* over document images. Builds a DocVQA-style dataset with synthetically corrupted (unanswerable) questions, benchmarks Vision LLMs on the resulting mixed set, and evaluates in-context-learning mitigation strategies for reducing false-positive "answerable" predictions.

## Pipeline overview

```
data/raw/{docvqa,dude,mp_docvqa}/
        │  src/dataset/pipeline.py        (corruption + LLM-judge filtering)
        ▼
data/corrupted/{dataset}_corrupted.json
        │  src/dataset/prepare_benchmark.py
        ▼
data/final/{dataset}_final.json          (mixed answerable + unanswerable)
        │  src/benchmark/run_benchmark.py
        ▼
results/benchmark/{model}_benchmark_result.json
        │  src/mitigation/run_mitigation.py
        ▼
results/mitigation/mitigation_results.json
```

## Setup

```bash
uv sync                                          # core deps + editable package
uv sync --extra dev                              # + jupyter, spacy
uv run python -m spacy download en_core_web_sm   # required after --extra dev

# API keys live in .env (gitignored)
export $(grep -v '^#' .env | xargs)
```

Python 3.11 is pinned (`.python-version`). Always run via `uv run` or the project's `.venv`.

## Running the pipeline

```bash
# 1. Generate corrupted (unanswerable) questions
uv run python -m src.dataset.pipeline \
  --dataset docvqa \
  --data_dir data/raw/docvqa \
  --output_dir data/corrupted \
  --config configs/dataset_config.yaml
  # [--no_judge] to skip the LLM-as-a-judge quality filter

# 1b. Build the mixed answerable/unanswerable benchmark set
uv run prepare-benchmark \
  --corrupted_dataset data/corrupted/docvqa_corrupted.json \
  --output_dir data/final

# 2. Benchmark Vision LLMs (dataset path + output dir set in config)
uv run python -m src.benchmark.run_benchmark \
  --config configs/benchmark_config.yaml

# 3. Evaluate mitigation strategies (few-shot, RAG)
uv run python -m src.mitigation.run_mitigation \
  --corrupted_dataset data/corrupted/docvqa_corrupted.json \
  --baseline_results results/benchmark/benchmark_results.json \
  --config configs/mitigation_config.yaml \
  --output_dir results/mitigation
```

Equivalent console entry points: `download-data`, `corrupt-dataset`, `corrupt-all-datasets`, `prepare-benchmark`, `run-benchmark`, `run-mitigation`.

## Architecture

- **`QASample`** (`src/dataset/loaders/base_loader.py`) — canonical record between loaders and the corruption pipeline.
- **`BaseCorruptor`** (`src/dataset/corruption/`) — `NLPEntityCorruptor` (spaCy NER + Wikipedia peer-entity swap), `ElementCorruptor`, `LayoutCorruptor`.
- **`BaseVisionModel`** (`src/benchmark/models/`) — backends: `VllmModel` (local vLLM server, primary), `MistralModel`, `GoogleModel`, `OpenRouterModel`, `LlamaCppModel`, plus a `SiglipClassifier` baseline.
- **Mitigation strategies** (`src/mitigation/strategies/`) — `few_shot` and `rag`, registered in `src/mitigation/registry.py`.

See `CLAUDE.md` for full architecture notes, JSON schemas, GPU worker infra (`docker-compose.yml`), and gotchas.

## Configuration

YAML configs live in `configs/`:
- `dataset_config.yaml` — corruption distribution, `max_samples`, multi-page `window_size`.
- `benchmark_config.yaml` — model backend (vllm/mistral/google/openrouter/llama_cpp), `corrupted_dataset`, `output_dir`.
- `mitigation_config.yaml` — mitigation strategy params and baseline results path.
- `siglip_classifier_config.yaml` — SigLIP baseline classifier training/eval.

## Tracking

All stages log to a local MLflow store:

```bash
uv run mlflow ui   # http://localhost:5000
```

Experiments: `corruption`, `benchmark`, `mitigation`.

## Tests

```bash
uv run pytest
```

## Report

`REPORT.md` / `report.tex` contain the write-up; figures are generated via `scripts/make_report_figures.py` into `figs/`.
