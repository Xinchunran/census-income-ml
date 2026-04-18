from __future__ import annotations

import argparse
import json
from pathlib import Path

from segment.pipeline_v1 import run_segmentation_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmentation pipeline and export artifacts.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="segment/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    diagnostics = run_segmentation_v1(
        project_root=args.project_root,
        output_root="segment",
        output_subdir=Path(args.output_dir).name,
        random_state=args.seed,
    )
    print(json.dumps({"status": "ok", "n_clusters": diagnostics.get("n_clusters")}, indent=2))


if __name__ == "__main__":
    main()
