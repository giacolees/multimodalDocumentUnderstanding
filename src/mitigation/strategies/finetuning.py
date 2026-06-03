"""Fine-tuning mitigation strategy using Unsloth + Qwen2.5-VL-3B-Instruct.

Each record in the corrupted dataset contributes two training examples:
  - UNANSWERABLE  ← corrupted_question  (judge_verified=True)
  - ANSWERABLE    ← original_question   (same document, known answerable)

The model is QLoRA-adapted via Unsloth for 4-bit efficiency, evaluated on a
held-out split, and saved to disk so it can replace the LLM judge in future
pipeline runs.

Install requirements (GPU machine):
    pip install "unsloth[cu124-torch260] @ https://github.com/unslothai/unsloth/releases/..."
    # or follow https://github.com/unslothai/unsloth for the current wheel
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FinetuningConfig:
    model_name: str = "unsloth/Qwen2.5-VL-3B-Instruct"
    output_dir: str = "results/finetuned_judge"
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    train_split: float = 0.8
    max_steps: int = 60
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    seed: int = 42


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_INSTRUCTION = (
    "You are given a document image and a question. "
    "Reply with exactly one word: ANSWERABLE or UNANSWERABLE."
)


def _user_text(question: str) -> str:
    return f"{_INSTRUCTION}\n\nQuestion: {question}"


def _build_messages(document_path: str, question: str, label: str) -> dict:
    image = Image.open(document_path).convert("RGB")
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _user_text(question)},
                ],
            },
            {"role": "assistant", "content": label},
        ]
    }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(
    records: list[dict],
    train_split: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split records into train/test and build conversation dicts.

    Each record yields two examples: one UNANSWERABLE (corrupted question)
    and one ANSWERABLE (original question from the same document).
    Both examples of a pair are kept in the same split to avoid leakage.
    """
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_split)
    train_records, test_records = shuffled[:split_idx], shuffled[split_idx:]

    def expand(recs: list[dict]) -> list[dict]:
        out = []
        for r in recs:
            out.append(_build_messages(r["document_path"], r["corrupted_question"], "UNANSWERABLE"))
            out.append(_build_messages(r["document_path"], r["original_question"], "ANSWERABLE"))
        return out

    return expand(train_records), expand(test_records)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune(records: list[dict], config: FinetuningConfig) -> dict:
    """Run the full Unsloth QLoRA training loop and return evaluation metrics."""
    try:
        from unsloth import FastVisionModel, is_bf16_supported  # type: ignore[import-untyped]
        from unsloth.trainer import UnslothVisionDataCollator  # type: ignore[import-untyped]
        from trl import SFTTrainer, SFTConfig  # type: ignore[import-untyped]
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "Fine-tuning requires unsloth and trl. "
            "Install with: uv sync --extra finetune\n"
            f"Original error: {e}"
        )

    train_convs, test_convs = prepare_dataset(records, config.train_split, config.seed)

    # --- Model + LoRA setup ---
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config.model_name,
        load_in_4bit=config.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
    )

    # --- Dataset ---
    train_dataset = Dataset.from_list(train_convs)

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir=config.output_dir,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=config.max_seq_length,
            seed=config.seed,
        ),
    )
    trainer.train()

    # --- Save LoRA adapter ---
    out_path = Path(config.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path / "lora_adapter"))
    tokenizer.save_pretrained(str(out_path / "lora_adapter"))
    print(f"LoRA adapter saved → {out_path / 'lora_adapter'}")

    # --- Evaluation ---
    metrics = _evaluate(model, tokenizer, test_convs)
    print(f"[finetuning] test metrics: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, tokenizer, test_convs: list[dict]) -> dict:
    """Run inference on test conversations and return accuracy/precision/recall/F1."""
    try:
        from unsloth import FastVisionModel  # type: ignore[import-untyped]
    except ImportError:
        return {}

    FastVisionModel.for_inference(model)

    tp = tn = fp = fn = 0

    for conv in test_convs:
        user_turn = conv["messages"][0]["content"]
        image = next(p["image"] for p in user_turn if p["type"] == "image")
        text = next(p["text"] for p in user_turn if p["type"] == "text")
        ground_truth_unanswerable = conv["messages"][1]["content"] == "UNANSWERABLE"

        messages = [{"role": "user", "content": user_turn}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt,
            images=[image],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=8, use_cache=True)
        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip().upper()

        predicted_unanswerable = "UNANSWERABLE" in response

        if predicted_unanswerable and ground_truth_unanswerable:
            tp += 1
        elif not predicted_unanswerable and not ground_truth_unanswerable:
            tn += 1
        elif predicted_unanswerable and not ground_truth_unanswerable:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": total,
    }
