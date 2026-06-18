# SigLIP-based unanswerable-question classifier — design

## Motivation

The current benchmark backends (`VllmModel`, `MistralModel`, `GoogleModel`, etc.) are full Vision-Language Models: every prediction requires autoregressive text generation, which is slow. As an additional, much faster baseline for the report, we want a pure vision-transformer-based classifier: a frozen SigLIP image encoder + a frozen sentence-transformer text encoder feeding a small trained MLP head, doing a single forward pass per prediction instead of token-by-token generation.

This is explicitly a comparison point, not a replacement: it trades some accuracy on subtle entity-level corruptions (which require language reasoning) for large inference-speed gains.

## Architecture

- **Image encoder**: `google/siglip-so400m-patch14-384`, frozen. Outputs a 1152-dim pooled image embedding.
- **Text encoder**: `sentence-transformers/all-MiniLM-L6-v2`, frozen. Embeds the (corrupted or original) question into a 384-dim vector.
- **Fusion**: concatenate image + text embeddings → 1536-dim vector.
- **Head**: small trainable MLP (2 hidden layers, e.g. 1536 → 512 → 128 → 1), BCEWithLogits loss, sigmoid at inference for the `confidence` score. This is the only trained component.
- **Multi-page handling** (mp_docvqa): each sample's `metadata.window_pages` lists every page image in its context window. The image embedding for such a sample is the **mean** of per-page SigLIP embeddings across all window pages. Single-page docvqa samples just use the one page from `document_path`.

## Data

- **Source**: `data/final/docvqa_final.json` (2758 samples) + `data/final/mp_docvqa_final.json` (8334 samples), combined into one pool (11092 samples).
- **Split**: 80/10/10 train/val/test, stratified by `is_unanswerable` so each split preserves the answerable/unanswerable ratio. Split is randomized with a fixed seed and the resulting sample-id lists are persisted to disk (e.g. `data/cache/siglip_classifier_split.json`) so re-runs are reproducible and comparable.
- **Embedding cache**: embeddings are frozen-encoder outputs, so they're precomputed once and cached to `data/cache/siglip_embeddings.pt`, keyed by `sample_id` (image embedding, text embedding, label). A content hash (dataset file mtimes/sizes + encoder checkpoint ids) gates cache reuse — if it doesn't match, embeddings are recomputed. This makes head-training iterations fast and fully offline after the first run.

## New files

| File | Purpose |
|---|---|
| `src/benchmark/models/siglip_classifier.py` | `SiglipClassifierModel(BaseVisionModel)` — loads frozen encoders + trained head checkpoint; implements `predict_unanswerable()` and `name()`. Used at benchmark-inference time, not training time. |
| `src/benchmark/train_classifier.py` | CLI entry point: builds the combined dataset, performs the stratified split, builds/loads the embedding cache, trains the MLP head with early stopping on validation F1, evaluates on the held-out test split, logs metrics + confusion matrix to MLflow (experiment `"siglip_classifier"`), and saves the best head weights to `models/siglip_classifier_head.pt`. |
| `configs/siglip_classifier_config.yaml` | Encoder checkpoint ids, hidden-layer sizes, learning rate, epochs, batch size, split seed, cache/output paths. |

## Integration with existing benchmark pipeline

- `run_benchmark.py`'s `load_model()` gains a new backend string, `"siglip_classifier"`, which instantiates `SiglipClassifierModel` from a config entry pointing at the trained head checkpoint.
- Evaluation reuses the existing `compute_metrics`, `compute_per_type_metrics`, and `plot_confusion_matrix` functions from `src/benchmark/evaluation/metrics.py` — no duplicate metrics code.
- Results land in `results/benchmark_{dataset}/siglip_classifier_benchmark_result.json`, the same shape as VLM backend results, so it's directly comparable in report tables.

## Dependencies

Add a new optional-dependency group to `pyproject.toml`:

```toml
vit-classifier = [
    "torch>=2.2",
    "transformers>=4.40",
    "scikit-learn>=1.4",
]
```

`sentence-transformers` (already in the `mitigation` extra) is reused for the MiniLM text encoder rather than adding a second text-embedding dependency.

## Error handling / edge cases

- If a `document_path` image fails to load (corrupt file, missing path), the sample is skipped during embedding precomputation and logged, mirroring how `PredictionResult.skipped` is handled elsewhere in the benchmark pipeline.
- If the embedding cache exists but its content hash doesn't match the current dataset/encoder config, it is fully regenerated (no partial/stale mixing).

## Out of scope

- Fine-tuning the SigLIP or MiniLM encoders themselves (both stay frozen).
- Hyperparameter search beyond what's exposed in the config file.
- Any change to existing VLM backends or mitigation strategies.
