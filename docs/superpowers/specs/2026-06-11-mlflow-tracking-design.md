# MLflow Experiment Tracking Design

**Date:** 2026-06-11  
**Approach:** Option A — inline tracking directly in the three runner scripts

## Overview

Add MLflow experiment tracking to all three pipeline stages. Each stage writes to a shared local `mlruns/` directory at the repo root. No new modules, no abstraction layer — `mlflow` calls go directly into `pipeline.py`, `run_benchmark.py`, and `run_mitigation.py`.

## Experiment Structure

| Stage | Experiment name | Run name pattern |
|---|---|---|
| Part 1 – corruption | `dataset-corruption` | `{dataset}_{timestamp}` |
| Part 2 – benchmark | `benchmark` | `{model_id}_{dataset}_{timestamp}` |
| Part 3 – mitigation | `mitigation` | `{strategy}_{model_id}_{timestamp}` |

All experiments use the default MLflow tracking URI (`mlruns/` in the working directory). No server setup required.

## What Gets Logged

### Part 1 — `src/dataset/pipeline.py`

One MLflow run per `pipeline()` call.

**Params:**
- `dataset` — dataset name (e.g. `docvqa`)
- `max_samples` — from config
- `window_size` — from config
- `corruption_types` — comma-separated list of active corruptors
- `no_judge` — boolean flag

**Metrics:**
- `total_corrupted`
- `judge_accepted`
- `judge_rejected`
- `nlp_entity_count`, `element_count`, `layout_count` — per-type breakdown

**Artifact:** output JSON file (`data/corrupted/{dataset}_corrupted.json`)

### Part 2 — `src/benchmark/run_benchmark.py`

One MLflow run per model (inner loop over `config["models"]`).

**Params:**
- `model_id`
- `backend`
- `dataset_path`
- `num_samples`

**Metrics:**
- `accuracy`, `precision`, `recall`, `f1`
- `tp`, `fp`, `tn`, `fn`
- `specificity` — TN / (TN + FP), how well the model avoids false alarms
- `balanced_accuracy` — mean of recall and specificity, robust when class distribution is skewed
- `mcc` — Matthews Correlation Coefficient, single score that accounts for all four confusion matrix cells
- Per-corruption-type breakdown: `f1_nlp_entity`, `f1_element`, `f1_layout`

**Artifacts:**
- Per-model result JSON (`results/benchmark_*/...`)
- Confusion matrix PNG (`confusion_matrix_{model_id}.png`) logged via `mlflow.log_figure()`

### Part 3 — `src/mitigation/run_mitigation.py`

One MLflow run per strategy × model combination.

**Params:**
- `strategy`
- `model_id`
- `dataset_path`
- `num_samples`

**Metrics:**
- `accuracy`, `precision`, `recall`, `f1`
- `tp`, `fp`, `tn`, `fn`
- `specificity`, `balanced_accuracy`, `mcc`
- Per-corruption-type breakdown: `f1_nlp_entity`, `f1_element`, `f1_layout`
- `delta_f1`, `delta_mcc` — improvement over baseline for the same model

**Artifact:**
- Per-strategy result JSON
- Confusion matrix PNG

**Artifact:** per-strategy result JSON

## Dependency Change

`mlflow` added to `[project.dependencies]` in `pyproject.toml`.

`mlruns/` added to `.gitignore`.

## Non-Goals

- No MLflow server, remote tracking URI, or model registry
- No new wrapper modules or helper functions
- No changes to the microservices layer
