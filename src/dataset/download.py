"""Download DocVQA-family datasets from HuggingFace and store them in the layout
expected by the loaders (DocVQALoader, DUDELoader, MPDocVQALoader)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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

    hf_split = "validation" if split == "val" else split
    print(f"[mp_docvqa] Loading from HuggingFace (split={hf_split}) …")
    ds = datasets.load_dataset("lmms-lab/MP-DocVQA", split=hf_split, trust_remote_code=True)

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir = data_dir / "images"
    img_dir.mkdir(exist_ok=True)

    records = []
    for row in ds:
        page_ids: list[str] = row.get("page_ids") or []

        # Save individual page images when provided as PIL objects
        images = row.get("images") or []
        for page_id, pil_img in zip(page_ids, images):
            img_path = img_dir / f"{page_id}.png"
            if not img_path.exists() and pil_img is not None:
                pil_img.save(img_path, format="PNG")

        records.append(
            {
                "questionId": str(row["questionId"]),
                "question": row["question"],
                "answers": row.get("answers") or [],
                "page_ids": page_ids,
                "answer_page_idx": row.get("answer_page_idx", 0),
                "docId": str(row.get("docId", "")),
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

    hf_split = "validation" if split == "val" else split
    print(f"[dude] Loading from HuggingFace (split={hf_split}) …")
    ds = datasets.load_dataset("lmms-lab/DUDE", split=hf_split, trust_remote_code=True)

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    data_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = data_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    doc_map: dict[str, dict] = {}
    for row in ds:
        doc_id = str(row["docId"])

        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "docId": doc_id,
                "num_pages": row.get("num_pages", 1),
                "qa_pairs": [],
            }
            # Save PDF bytes if the dataset provides them
            pdf_bytes = row.get("pdf") or row.get("document")
            if pdf_bytes is not None:
                pdf_path = pdf_dir / doc_id
                if not pdf_path.exists():
                    if isinstance(pdf_bytes, bytes):
                        pdf_path.write_bytes(pdf_bytes)
                    elif hasattr(pdf_bytes, "read"):
                        pdf_path.write_bytes(pdf_bytes.read())

        doc_map[doc_id]["qa_pairs"].append(
            {
                "questionId": str(row["questionId"]),
                "question": row["question"],
                "answers": row.get("answers") or [],
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
