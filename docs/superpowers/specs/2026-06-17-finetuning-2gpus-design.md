# Design: `finetuning_2gpus` mitigation strategy

Date: 2026-06-17

## Purpose

The existing `finetuning` mitigation strategy (`src/mitigation/strategies/finetuning.py`) uses
Unsloth's `FastVisionModel` to QLoRA-tune `Qwen2.5-VL-3B-Instruct` on a single GPU. Unsloth's
fast kernels and 4-bit path don't support sharding a model across multiple GPUs, so any model
too large to fit on one card is out of reach with that strategy.

This adds a second, independent fine-tuning path — `finetuning_2gpus` — built on plain HF
`transformers` + `peft` + `accelerate`/DeepSpeed ZeRO-3, so a larger model
(`Qwen/Qwen2.5-VL-32B-Instruct`) can be LoRA-tuned by sharding the frozen base weights and
LoRA adapters across 2 GPUs (~40–48GB VRAM each, e.g. A100-40GB/A6000).

## Non-goals

- Not a faster drop-in replacement for the existing `finetuning` strategy — different model
  size class, different infra requirements, different launch mechanism.
- Not wired into `run_mitigation.py`'s single-process strategy loop (see "Launch mechanism").
- No quantization (4-bit/8-bit) of the base model — DeepSpeed ZeRO-3 parameter sharding and
  bitsandbytes quantization are not well-supported together upstream. Memory is managed via
  ZeRO-3 sharding + CPU optimizer offload instead.

## Architecture

### Launch mechanism

DeepSpeed multi-GPU training requires the process to be launched with `accelerate launch`
(or `deepspeed`), which spawns one process per GPU. `run_mitigation.py`'s `run_mitigation()`
runs as a single process and loops over strategies in-process — that model is incompatible
with DeepSpeed's multi-process requirement.

`finetuning_2gpus.py` is therefore a **standalone script**, not part of the `STRATEGIES`
registry or the `strategies:` list consumed by `run_mitigation.py`. It is run directly:

```bash
accelerate launch --config_file configs/accelerate_2gpu.yaml \
  -m src.mitigation.strategies.finetuning_2gpus \
  --corrupted_dataset data/final/mp_docvqa_final.json \
  --config configs/mitigation_config.yaml
```

### New files

1. **`configs/deepspeed_zero3.json`**
   - ZeRO Stage 3 (`"stage": 3`), `bf16: {"enabled": true}`.
   - `offload_optimizer: {"device": "cpu"}` — optimizer states are small since only LoRA
     params are trainable; offloading them frees GPU memory for the frozen base weights.
   - Base model params stay on GPU (no param offload) — 64GB of bf16 weights split across
     2×40–48GB cards should fit once optimizer state and activations are off-loaded /
     checkpointed.
   - `stage3_gather_16bit_weights_on_model_save: true` so `trainer.save_model()` produces a
     normal consolidated (non-sharded) checkpoint on rank 0.
   - `gradient_clipping`, `train_micro_batch_size_per_gpu: "auto"`,
     `gradient_accumulation_steps: "auto"` (driven by `TrainingArguments`).

2. **`configs/accelerate_2gpu.yaml`**
   - `num_processes: 2`, `distributed_type: DEEPSPEED`,
     `deepspeed_config_file: configs/deepspeed_zero3.json`, `mixed_precision: bf16`.

3. **`src/mitigation/strategies/finetuning_2gpus.py`**

   - `Finetuning2GpuConfig` dataclass:
     ```python
     model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct"
     output_dir: str = "results/finetuned_judge_2gpu"
     lora_r: int = 16
     lora_alpha: int = 16
     lora_target_modules: list[str] = field(default_factory=lambda: [
         "q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj",
     ])
     train_split: float = 0.8
     max_steps: int = 60
     per_device_train_batch_size: int = 1
     gradient_accumulation_steps: int = 8
     learning_rate: float = 2e-4
     seed: int = 42
     deepspeed_config: str = "configs/deepspeed_zero3.json"
     ```

   - **Data prep**: imports `prepare_dataset` and `_build_messages` from
     `src/mitigation/strategies/finetuning.py` unchanged — those helpers have no Unsloth
     dependency (just PIL + stdlib `random`), so this avoids duplicating the
     train/test-split-without-leakage logic.

   - **Model + LoRA setup**:
     - `Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)`
       + `AutoProcessor.from_pretrained(model_name)`.
     - `peft.get_peft_model(model, LoraConfig(r=..., target_modules=config.lora_target_modules, ...))`.

   - **Data collation**: a small `collate_fn` (passed as `data_collator` to `Trainer`) that,
     for each conversation dict from `prepare_dataset`, calls
     `processor.apply_chat_template(...)` then `processor(images=..., text=..., return_tensors="pt")`,
     builds `labels` by masking the prompt tokens (-100) and keeping only the assistant's
     `ANSWERABLE`/`UNANSWERABLE` token(s), and pads/stacks the batch.

   - **Training**: standard HF `Trainer` with
     `TrainingArguments(deepspeed=config.deepspeed_config, bf16=True, gradient_checkpointing=True,
     max_steps=..., per_device_train_batch_size=..., gradient_accumulation_steps=...,
     learning_rate=..., save_strategy="no", report_to="none")`. `accelerate launch` +
     `Trainer`'s built-in DeepSpeed integration handle process orchestration — no manual
     `accelerator.prepare()` calls needed.
     - `trainer.train()`, then `trainer.save_model(output_dir/"lora_adapter")` +
       `processor.save_pretrained(...)` (rank-0 only, guarded by
       `trainer.is_world_process_zero()`).

   - **Evaluation**: same metric logic as `finetuning.py::_evaluate` (TP/TN/FP/FN →
     accuracy/precision/recall/F1), but:
     - Every rank must call `model.generate(..., synced_gpus=True)` because ZeRO-3 shards
       params and all ranks must participate in the gather collectives during generation —
       even though only rank 0 uses the output.
     - Only `trainer.is_world_process_zero()` prints/writes the metrics JSON.

   - **CLI** (`if __name__ == "__main__":`): argparse with `--corrupted_dataset`, `--config`
     (YAML path, reads the `finetuning_2gpus:` block), `--output_dir` override. Loads the
     dataset JSON the same way `run_mitigation.py` does (no subset/baseline logic — this is a
     standalone training run, not part of the tracked mitigation comparison pipeline).

4. **`configs/mitigation_config.yaml`** — add, near the existing `finetuning:` block:
   ```yaml
   # finetuning_2gpus is NOT run through run_mitigation.py — launch separately with:
   #   accelerate launch --config_file configs/accelerate_2gpu.yaml \
   #     -m src.mitigation.strategies.finetuning_2gpus --config configs/mitigation_config.yaml
   finetuning_2gpus:
     model_name: Qwen/Qwen2.5-VL-32B-Instruct
     output_dir: results/finetuned_judge_2gpu
     lora_r: 16
     lora_alpha: 16
     train_split: 0.8
     max_steps: 60
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 8
     learning_rate: 0.0002
     seed: 42
     deepspeed_config: configs/deepspeed_zero3.json
   ```
   Not added to the `strategies:` list (that list is only consumed by `run_mitigation.py`'s
   in-process loop).

5. **`pyproject.toml`** — new optional extra:
   ```toml
   # GPU machine (2x GPU) required; launch via `accelerate launch`, see finetuning_2gpus.py
   finetune2gpu = [
       "transformers>=4.46",
       "peft>=0.13",
       "accelerate>=1.0",
       "deepspeed>=0.15",
       "torch>=2.4",
   ]
   ```

## Error handling

- Same pattern as `finetuning.py`: wrap the `transformers`/`peft`/`deepspeed` imports in a
  `try/except ImportError` that raises a clear message pointing at
  `uv sync --extra finetune2gpu`.
- No other special error handling — this is a standalone training script run by hand on a
  GPU machine, not part of the automated benchmark/mitigation pipeline, so failures should
  surface directly (stack trace) rather than being caught and degraded.

## Testing

- No unit tests — same as `finetuning.py`, this requires real GPUs and downloads a 32B model,
  so it isn't testable in CI. Manual verification: run with `max_steps: 2` on the 2-GPU
  machine and confirm a LoRA adapter is saved and eval metrics print.

## Out of scope / future work

- Wiring `finetuning_2gpus` results into the `mitigation_results.json` / MLflow comparison
  pipeline that `run_mitigation.py` produces — would need a follow-up design once it's clear
  whether this becomes a regularly-run comparison or stays a one-off experiment.
