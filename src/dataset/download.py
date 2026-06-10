"""Download DocVQA-family datasets from HuggingFace and store them in the layout
expected by the loaders (DocVQALoader, DUDELoader, MPDocVQALoader)."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


def _parse_str_list(val: object) -> list:
    """HF MP-DocVQA stores list fields as stringified Python lists; parse them back."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else [val]
        except (ValueError, SyntaxError):
            return [val] if val else []
    return []


# ---------------------------------------------------------------------------
# DocVQA
# HF id: lmms-lab/DocVQA  config: DocVQA  split: validation
# Expected output layout:
#   data_dir/val/val_v1.0.json
#   data_dir/val/documents/<image>
# ---------------------------------------------------------------------------

def _download_docvqa(data_dir: Path, split: str, max_samples: int) -> None:
    import datasets

    hf_split = "validation" if split == "val" else split
    print(f"[docvqa] Loading from HuggingFace (split={hf_split}) …")
    ds = datasets.load_dataset("lmms-lab/DocVQA", "DocVQA", split=hf_split, trust_remote_code=True)

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    out_dir = data_dir / split
    img_dir = out_dir / "documents"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Group by docId so we can merge multiple QA pairs per document
    doc_map: dict[str, dict] = {}
    for row in ds:
        doc_id = str(row["docId"])
        img_filename = row.get("image_filename") or f"{doc_id}.png"

        if doc_id not in doc_map:
            doc_map[doc_id] = {"image": img_filename, "docId": doc_id, "qa_pairs": []}
            # Save image once
            img_path = img_dir / img_filename
            if not img_path.exists():
                pil_image = row["image"]
                pil_image.save(img_path, format="PNG")

        doc_map[doc_id]["qa_pairs"].append(
            {
                "questionId": row["questionId"],
                "question": row["question"],
                "answers": row.get("answers") or [],
            }
        )

    annotation = {"dataset_name": "DocVQA", "split": split, "data": list(doc_map.values())}
    ann_path = out_dir / f"{split}_v1.0.json"
    ann_path.write_text(json.dumps(annotation, indent=2))
    print(f"[docvqa] Saved {len(doc_map)} documents → {out_dir}")


# ---------------------------------------------------------------------------
# MP-DocVQA
# HF id: lmms-lab/MP-DocVQA  split: val (or validation)
# Expected output layout:
#   data_dir/val_v1.0.json
#   data_dir/images/<page_id>.png   (one file per page)
# ---------------------------------------------------------------------------

def _download_mp_docvqa(data_dir: Path, split: str, max_samples: int) -> None:
    import datasets

    hf_split = split  # MP-DocVQA uses "val" / "test" natively
    print(f"[mp_docvqa] Loading from HuggingFace (split={hf_split}) …")
    ds = datasets.load_dataset("lmms-lab/MP-DocVQA", split=hf_split, trust_remote_code=True)

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir = data_dir / "images"
    img_dir.mkdir(exist_ok=True)

    records = []
    for row in ds:
        page_ids: list[str] = _parse_str_list(row.get("page_ids"))
        answers: list[str] = _parse_str_list(row.get("answers"))
        answer_page_idx = int(row.get("answer_page_idx") or 0)

        # Images are stored as individual columns image_1 … image_N (1-indexed)
        for i, page_id in enumerate(page_ids, start=1):
            pil_img = row.get(f"image_{i}")
            if pil_img is None:
                continue
            img_path = img_dir / f"{page_id}.png"
            if not img_path.exists():
                pil_img.save(img_path, format="PNG")

        records.append(
            {
                "questionId": str(row["questionId"]),
                "question": row["question"],
                "answers": answers,
                "page_ids": page_ids,
                "answer_page_idx": answer_page_idx,
                "docId": str(row.get("doc_id") or row.get("docId") or ""),
            }
        )

    annotation = {"dataset_name": "MP-DocVQA", "split": split, "data": records}
    ann_path = data_dir / f"{split}_v1.0.json"
    ann_path.write_text(json.dumps(annotation, indent=2))
    print(f"[mp_docvqa] Saved {len(records)} questions → {data_dir}")


# ---------------------------------------------------------------------------
# DUDE
# HF id: lmms-lab/DUDE  split: val (or validation)
# Expected output layout:
#   data_dir/dude_val.json
#   data_dir/pdfs/<docId>            (PDF bytes or skipped if not available)
# ---------------------------------------------------------------------------

def _download_dude(data_dir: Path, split: str, max_samples: int) -> None:
    import datasets

    # The original lmms-lab/DUDE is gated; fall back to the public 100-sample subset.
    HF_IDS = ["lmms-lab/DUDE", "jordyvl/DUDE_subset_100val"]
    hf_split = "validation" if split == "val" else split

    ds = None
    for hf_id in HF_IDS:
        try:
            # The public subset uses the split name "train" (it only has one split)
            actual_split = hf_split if hf_id == "lmms-lab/DUDE" else "train"
            print(f"[dude] Trying {hf_id} (split={actual_split}) …")
            ds = datasets.load_dataset(hf_id, split=actual_split, trust_remote_code=True)
            print(f"[dude] Loaded {len(ds)} rows from {hf_id}")
            break
        except Exception as exc:
            print(f"[dude] {hf_id} unavailable: {exc}")

    if ds is None:
        raise RuntimeError("Could not load DUDE from any known HuggingFace source.")

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine whether we have PDF bytes or PIL images.
    sample = ds[0]
    has_pdf = bool(sample.get("pdf") or sample.get("document"))
    has_images = bool(sample.get("images"))

    if has_pdf:
        doc_dir = data_dir / "pdfs"
    else:
        doc_dir = data_dir / "documents"
    doc_dir.mkdir(exist_ok=True)

    doc_map: dict[str, dict] = {}
    for row in ds:
        qid = str(row["questionId"])
        # Fall back to deriving docId from questionId when the field is absent.
        doc_id = str(row["docId"]) if row.get("docId") else qid.split("_")[0]

        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "docId": doc_id,
                "num_pages": row.get("num_pages", 1),
                "qa_pairs": [],
            }
            if has_pdf:
                pdf_bytes = row.get("pdf") or row.get("document")
                if pdf_bytes is not None:
                    pdf_path = doc_dir / doc_id
                    if not pdf_path.exists():
                        if isinstance(pdf_bytes, bytes):
                            pdf_path.write_bytes(pdf_bytes)
                        elif hasattr(pdf_bytes, "read"):
                            pdf_path.write_bytes(pdf_bytes.read())
            elif has_images:
                images = row.get("images") or []
                for i, pil_img in enumerate(images):
                    if pil_img is not None:
                        img_path = doc_dir / f"{doc_id}_page{i}.png"
                        if not img_path.exists():
                            pil_img.save(img_path, format="PNG")
                # Update num_pages from actual images
                doc_map[doc_id]["num_pages"] = max(len(images), 1)

        answers_raw = row.get("answers")
        if answers_raw is None:
            answer_val = row.get("answer", "")
            answers_list = [answer_val] if answer_val else []
        elif isinstance(answers_raw, list):
            answers_list = answers_raw
        else:
            answers_list = [str(answers_raw)]

        doc_map[doc_id]["qa_pairs"].append(
            {
                "questionId": qid,
                "question": row["question"],
                "answers": answers_list,
                "page": row.get("answer_page_idx", 0),
            }
        )

    annotation = list(doc_map.values())
    ann_path = data_dir / f"dude_{split}.json"
    ann_path.write_text(json.dumps(annotation, indent=2))
    print(f"[dude] Saved {len(doc_map)} documents → {data_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "docvqa": (_download_docvqa, "data/raw/docvqa"),
    "mp_docvqa": (_download_mp_docvqa, "data/raw/mp_docvqa"),
    "dude": (_download_dude, "data/raw/dude"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DocVQA-family datasets from HuggingFace."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DOWNLOADERS),
        nargs="+",
        default=list(DOWNLOADERS),
        help="Which dataset(s) to download (default: all).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory where raw data is stored (default: data/raw).",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split to download (default: val).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Limit number of samples per dataset (-1 = all).",
    )
    args = parser.parse_args()

    for name in args.dataset:
        fn, default_subdir = DOWNLOADERS[name]
        target = args.data_dir / name if args.data_dir != Path("data/raw") else Path(default_subdir)
        try:
            fn(target, args.split, args.max_samples)
        except Exception as exc:
            print(f"[{name}] ERROR: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
