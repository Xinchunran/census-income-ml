from __future__ import annotations

import argparse
import json

from segment.pipeline_v2 import run_segmentation_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run score-augmented classifier-informed segmentation (V2).")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="segment")
    parser.add_argument("--output-subdir", default="v2_score_augmented_classifier_informed")
    parser.add_argument("--score-path", required=True)
    parser.add_argument("--score-column", default="income_score")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    diagnostics = run_segmentation_v2(
        project_root=args.project_root,
        output_root=args.output_root,
        output_subdir=args.output_subdir,
        score_path=args.score_path,
        random_state=args.seed,
        score_column=args.score_column,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "scheme_name": diagnostics.get("scheme_name"),
                "score_column_used": diagnostics.get("score_column_used"),
                "n_clusters": diagnostics.get("n_clusters"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
