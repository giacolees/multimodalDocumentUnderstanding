"""Corrupt job: wraps src.dataset.pipeline.run_pipeline()."""

import json
import os
from pathlib import Path

import state


async def run_corrupt_job(redis, job_id: str, config: dict) -> None:
    state.update_job(redis, job_id, status="running")
    try:
        dataset = config["dataset"]
        data_dir = config["data_dir"]
        output_dir = config.get("output_dir", "data/corrupted")
        use_judge = config.get("use_judge", True)

        import sys
        sys.path.insert(0, "/app")
        from src.dataset.pipeline import run_pipeline
        import yaml

        cfg_path = config.get("pipeline_config", "configs/dataset_config.yaml")
        with open(cfg_path) as f:
            pipeline_cfg = yaml.safe_load(f)

        results = run_pipeline(
            dataset=dataset,
            data_dir=data_dir,
            output_dir=output_dir,
            config=pipeline_cfg,
            use_judge=use_judge,
            seed=config.get("seed", 42),
        )
        result_path = str(Path(output_dir) / f"{dataset}_corrupted.json")
        state.update_job(
            redis, job_id,
            status="done",
            total=len(results),
            progress=len(results),
            result_path=result_path,
        )
    except Exception as e:
        state.update_job(redis, job_id, status="failed", error=str(e))
        raise
