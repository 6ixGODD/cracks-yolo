"""CLI entry: model efficiency analysis (params, MACs, latency, VRAM).

Example:
    python -m scripts.analyze_model --model yolov5s_sactr --output-dir output/model_analysis
    python -m scripts.analyze_model --all --output-dir output/model_analysis_all
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cracks_yolo.analysis.model import analyze_model
from cracks_yolo.analysis.model import save_model_analysis
from cracks_yolo.zoo import ZOO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze model efficiency (params, MACs, latency)."
    )
    parser.add_argument("--model", help="ZOO key (required unless --all)")
    parser.add_argument("--all", action="store_true", help="Analyze every model in ZOO")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-runs", type=int, default=50)
    args = parser.parse_args()

    if not args.all and not args.model:
        parser.error("either --model or --all is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(ZOO.keys()) if args.all else [args.model]
    summaries: list[dict[str, object]] = []
    for key in keys:
        cls = ZOO[key]
        print(f"analyzing {key} ...")
        model = cls(num_classes=1, input_size=args.input_size)
        report = analyze_model(
            model,
            input_size=args.input_size,
            device=args.device,
            n_runs=args.n_runs,
        )
        save_model_analysis(report, args.output_dir / key)
        summaries.append({"model": key, **report.to_dict()})

    if summaries:
        csv_path = args.output_dir / "model_analysis_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
        print(f"summary: {csv_path}")


if __name__ == "__main__":
    main()
