from __future__ import annotations

import argparse
import json

from segment.pipeline_v1 import run_segmentation_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw-feature unsupervised segmentation (V1).")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="segment")
    parser.add_argument("--output-subdir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    diagnostics = run_segmentation_v1(
        project_root=args.project_root,
        output_root=args.output_root,
        output_subdir=args.output_subdir,
        random_state=args.seed,
    )
    print(json.dumps({"status": "ok", "scheme_name": diagnostics.get("scheme_name"), "n_clusters": diagnostics.get("n_clusters")}, indent=2))


if __name__ == "__main__":
    main()
