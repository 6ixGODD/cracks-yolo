"""Typer CLI for cracks_yolo: train / test / compose."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from typer import Option
from typer import Typer

app = Typer(
    name="cracks-yolo",
    help="Tongue surface crack detection model zoo.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command()
def train(
    model: Annotated[
        str,
        Option("--model", "-m", help="ZOO key (e.g. yolov5s_sac)"),
    ],
    dataset: Annotated[
        str,
        Option("--dataset", "-d", help="Dataset root path"),
    ],
    output_dir: Annotated[
        Path,
        Option("--output-dir", "-o", help="Output directory"),
    ],
    epochs: Annotated[
        int,
        Option("--epochs", "-e", help="Number of epochs"),
    ] = 300,
    batch_size: Annotated[
        int,
        Option("--batch-size", "-b", help="Batch size"),
    ] = 64,
    lr: Annotated[float, Option("--lr", help="Learning rate")] = 1e-3,
    pretrained: Annotated[
        bool,
        Option("--pretrained/--no-pretrained", help="Load COCO pretrained weights"),
    ] = True,
    device: Annotated[
        str,
        Option("--device", help="Device (cuda/cpu)"),
    ] = "cuda",
    seed: Annotated[
        int,
        Option("--seed", help="Random seed"),
    ] = 42,
    num_workers: Annotated[
        int,
        Option("--num-workers", "-w", help="DataLoader workers"),
    ] = 8,
    optimizer: Annotated[
        str,
        Option("--optimizer", help="Optimizer (adamw/sgd)"),
    ] = "adamw",
    cosine_lr: Annotated[
        bool, Option("--cosine-lr/--no-cosine-lr", help="Cosine LR scheduler")
    ] = True,
    use_ema: Annotated[
        bool,
        Option("--ema/--no-ema", help="Exponential moving average"),
    ] = True,
    early_stopping_patience: Annotated[
        int,
        Option("--patience", help="Early stopping patience"),
    ] = 100,
    clip_grad_norm: Annotated[
        float,
        Option("--clip-grad-norm", help="Gradient clipping max norm"),
    ] = 10.0,
) -> None:
    """Train a model on the tongue crack dataset."""
    from cracks_yolo.pipeline.train import run_train

    run_train(
        model_name=model,
        dataset=dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        pretrained=pretrained,
        device=device,
        seed=seed,
        num_workers=num_workers,
        optimizer=optimizer,
        cosine_lr=cosine_lr,
        use_ema=use_ema,
        early_stopping_patience=early_stopping_patience,
        clip_grad_norm=clip_grad_norm,
    )


@app.command()
def test(
    model: Annotated[
        str,
        Option("--model", "-m", help="ZOO key (ignored when --torchscript, pass any value)"),
    ],
    weights: Annotated[
        Path,
        Option("--weights", help="Path to best.pt"),
    ],
    dataset: Annotated[
        str,
        Option("--dataset", "-d", help="Dataset root path"),
    ],
    output_dir: Annotated[
        Path,
        Option("--output-dir", "-o", help="Output directory"),
    ],
    batch_size: Annotated[
        int,
        Option("--batch-size", "-b", help="Batch size"),
    ] = 32,
    device: Annotated[
        str,
        Option("--device", help="Device"),
    ] = "cuda",
    seed: Annotated[
        int,
        Option("--seed", help="Random seed"),
    ] = 42,
    torchscript: Annotated[
        bool,
        Option("--torchscript", help="Weights is a TorchScript file (static YOLOv5 inference)"),
    ] = False,
) -> None:
    """Test a trained model on test + val splits."""
    from cracks_yolo.pipeline.test import run_test

    if torchscript:
        from cracks_yolo.zoo.static_yolo import StaticYOLOv5

        model_obj = StaticYOLOv5()
        model_obj.load(weights)
        # TS model stays on CPU (mixed device constants from old export)
        import json
        from pathlib import Path as _Path

        import torch

        from cracks_yolo.dataset.torchadapter import DetectionDataset
        from cracks_yolo.dataset.torchadapter import detection_collate
        from cracks_yolo.dataset.transforms import build_transforms
        from cracks_yolo.dataset.yolo import YOLOSource
        from cracks_yolo.pipeline.test import _compute_metrics
        from cracks_yolo.pipeline.test import _save_metrics_csv

        out = _Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        isize = model_obj.input_size
        src = YOLOSource(dataset)
        results: dict = {"model": "static_yolov5"}

        for split in ("test", "valid"):
            records = src.load_split(split)
            if not records:
                continue
            ds = DetectionDataset(
                records, transform=build_transforms(isize, train=False, augment=False)
            )
            from torch.utils.data import DataLoader

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=detection_collate,
            )

            all_preds = []
            with torch.no_grad():
                for imgs, targets in loader:
                    res_list = model_obj.inference(imgs)  # TS runs on CPU
                    for b, res in enumerate(res_list):
                        raw_id = (
                            targets[b].get("image_id", torch.tensor(b)).item()
                            if isinstance(targets, list)
                            else b
                        )
                        img_id = int(raw_id) + 1
                        for j in range(len(res.boxes)):
                            bx = res.boxes[j].tolist()
                            all_preds.append({
                                "image_id": img_id,
                                "category_id": int(res.labels[j]) + 1,
                                "bbox": [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]],
                                "score": float(res.scores[j]),
                            })

            metrics = _compute_metrics(records, all_preds, isize, dataset)
            results[f"{split}_metrics"] = metrics
            results[f"{split}_predictions"] = all_preds
            pred_file = out / f"best_predictions_{split}.json"
            pred_file.write_text(json.dumps(all_preds))
            print(
                f"  {split}: mAP@50={metrics.get('map50', 0):.4f}  mAP@50-95={metrics.get('map5095', 0):.4f}"
            )

        _save_metrics_csv(results, out / "metrics.csv")
        return

    run_test(
        model_name=model,
        weights=weights,
        dataset=dataset,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )


@app.command()
def run(
    config: Annotated[
        Path,
        Option("--config", "-c", help="Single experiment YAML config file"),
    ],
    output_dir: Annotated[
        Path | None,
        Option("--output-dir", "-o", help="Override output directory"),
    ] = None,
    device: Annotated[
        str | None,
        Option("--device", help="Override device (cuda/cpu)"),
    ] = None,
    test_only: Annotated[
        bool,
        Option("--test-only", help="Skip training, run test only. Requires --weights."),
    ] = False,
    weights: Annotated[
        Path | None,
        Option("--weights", "-w", help="Path to checkpoint .pt for --test-only"),
    ] = None,
) -> None:
    """Run a single experiment from a YAML config file (train or test)."""
    import yaml

    from cracks_yolo.pipeline.test import run_test
    from cracks_yolo.pipeline.train import run_train

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid experiment config: {config}")

    exp_type = cfg.pop("type", "train")
    cfg.pop("name", None)  # metadata, not a pipeline arg

    # Map YAML field names to pipeline kwargs
    if "model" in cfg:
        cfg["model_name"] = cfg.pop("model")

    if output_dir is not None:
        cfg["output_dir"] = output_dir
    if device is not None:
        cfg["device"] = device

    if test_only:
        if weights is None:
            raise ValueError("--weights is required when --test-only is set")
        test_cfg = {
            k: v
            for k, v in cfg.items()
            if k in ("model_name", "dataset", "batch_size", "device", "seed")
        }
        test_cfg["weights"] = weights
        test_cfg["output_dir"] = cfg["output_dir"] / "test"
        run_test(**test_cfg)
    elif exp_type == "train":
        train_report = run_train(**cfg)
        # Auto-run test on the best checkpoint
        weights_path = train_report.checkpoint_path or (cfg["output_dir"] / "weights" / "best.pt")
        if weights_path.exists():
            print(f"\n{'=' * 60}\nRunning test on best checkpoint: {weights_path}\n{'=' * 60}")
            test_cfg = {
                k: v
                for k, v in cfg.items()
                if k in ("model_name", "dataset", "output_dir", "batch_size", "device", "seed")
            }
            test_cfg["weights"] = weights_path
            test_cfg["output_dir"] = cfg["output_dir"] / "test"
            run_test(**test_cfg)
    elif exp_type == "test":
        run_test(**cfg)
    else:
        raise ValueError(f"unknown experiment type: {exp_type}")


@app.command()
def compose(
    config: Annotated[
        Path,
        Option("--config", "-c", help="Compose YAML with $include"),
    ],
    output_dir: Annotated[
        Path,
        Option("--output-dir", "-o", help="Output directory"),
    ],
    max_parallel: Annotated[
        int,
        Option("--max-parallel", "-p", help="Max parallel experiments"),
    ] = 1,
) -> None:
    """Run batch experiments from a YAML compose config."""
    from cracks_yolo.pipeline.compose import run_compose

    run_compose(config=config, output_dir=output_dir, max_parallel=max_parallel)


@app.command()
def aggregate(
    root: Annotated[
        Path,
        Option("--root", "-r", help="Root directory containing model subdirs (e.g. output/)"),
    ],
    output: Annotated[
        Path,
        Option("--output", "-o", help="Output CSV path"),
    ] = Path("test_metrics.csv"),
    excel: Annotated[
        bool,
        Option("--excel/--no-excel", help="Also write Excel (.xlsx)"),
    ] = True,
) -> None:
    """Aggregate test/metrics.csv across model subdirectories into one table."""
    from cracks_yolo.analysis.aggregate import aggregate as _aggregate

    out_csv = output
    out_xlsx = output.with_suffix(".xlsx") if excel else None
    result = _aggregate(root.resolve(), out_csv, out_xlsx)
    print(f"Aggregated to {result}")
    if out_xlsx:
        print(f"Excel: {out_xlsx}")


@app.command()
def visualize(
    root: Annotated[
        Path,
        Option("--root", "-r", help="Root directory containing model subdirs (e.g. output/)"),
    ],
    dataset: Annotated[
        str,
        Option("--dataset", "-d", help="Dataset root path"),
    ],
    output_dir: Annotated[
        Path,
        Option("--output-dir", "-o", help="Output directory for plots"),
    ] = Path("plots"),
    split: Annotated[
        str,
        Option("--split", "-s", help="Dataset split: test or valid"),
    ] = "test",
    models: Annotated[
        str | None,
        Option("--models", "-m", help="Comma-separated model names (default: all in root)"),
    ] = None,
) -> None:
    """Visualize: PR/ROC curves, confusion matrix, metric bars from predictions."""
    from cracks_yolo.viz.plotting import _compute_confusion
    from cracks_yolo.viz.plotting import _compute_pr_roc
    from cracks_yolo.viz.plotting import _load_ground_truths
    from cracks_yolo.viz.plotting import _load_predictions
    from cracks_yolo.viz.plotting import plot_confusion_matrix
    from cracks_yolo.viz.plotting import plot_metric_bars
    from cracks_yolo.viz.plotting import plot_pr_curve
    from cracks_yolo.viz.plotting import plot_roc_curve

    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model list
    if models:
        model_names = [m.strip() for m in models.split(",")]
    else:
        model_names = sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir() and (d / "test" / f"best_predictions_{split}.json").exists()
        )

    if not model_names:
        raise ValueError(f"no models found with predictions for split={split} in {root}")

    pred_file = f"best_predictions_{split}.json"

    # Load GT once
    gts, img_sizes = _load_ground_truths(dataset, split)
    model_isize = 640  # default; adjustable if needed

    # Compute per-model data
    pr_data: dict[str, dict] = {}
    roc_data: dict[str, dict] = {}
    metrics_data: dict[str, dict[str, float]] = {}

    for name in model_names:
        pred_path = root / name / "test" / pred_file
        if not pred_path.exists():
            print(f"  skip {name}: {pred_path} not found")
            continue
        preds = _load_predictions(pred_path)
        curve_data = _compute_pr_roc(preds, gts, img_sizes, model_isize)
        pr_data[name] = curve_data
        roc_data[name] = curve_data
        metrics_data[name] = {
            "auc_pr": curve_data["auc_pr"],
            "auc_roc": curve_data["auc_roc"],
            "f1_max": curve_data["f1_max"],
        }

        # Per-model confusion matrix (image-level)
        cm = _compute_confusion(preds, gts, total_imgs=len(img_sizes))
        plot_confusion_matrix(
            cm,
            output_dir / f"confusion_{name}_{split}.png",
            title=f"{name} — {split}",
        )

        print(
            f"  {name}: AUC-PR={curve_data['auc_pr']:.4f}  "
            f"AUC-ROC={curve_data['auc_roc']:.4f}  F1-max={curve_data['f1_max']:.4f}"
        )

    # Multi-model comparison plots
    plot_pr_curve(
        pr_data, output_dir / f"pr_curve_{split}.png", title=f"Precision-Recall — {split}"
    )
    plot_roc_curve(roc_data, output_dir / f"roc_curve_{split}.png", title=f"ROC — {split}")

    # Metric bar charts
    for metric in ("auc_pr", "auc_roc", "f1_max"):
        plot_metric_bars(
            metrics_data,
            metric,
            output_dir / f"bar_{metric}_{split}.png",
            title=f"{metric.upper()} — {split}",
        )

    print(f"\nPlots saved to {output_dir}/")


@app.command(name="analyze-model")
def analyze_model_cmd(
    model: Annotated[
        str,
        Option("--model", "-m", help="ZOO key (e.g. yolov5s_sactr)"),
    ],
    output_dir: Annotated[
        Path,
        Option("--output-dir", "-o", help="Output directory"),
    ],
    device: Annotated[
        str,
        Option("--device", help="Device (cuda/cpu)"),
    ] = "cuda",
    input_size: Annotated[
        int,
        Option("--input-size", help="Input resolution"),
    ] = 640,
) -> None:
    """Analyze a single model: param count, GFLOPs, latency percentiles, peak VRAM.

    Saves model_analysis.json with MACs, FLOPs, latency p50/p95/mean, peak
    memory, and parameter counts.
    """
    from cracks_yolo.analysis.model import analyze_model
    from cracks_yolo.analysis.model import save_model_analysis
    from cracks_yolo.zoo import ZOO

    cls = ZOO[model]
    m = cls(num_classes=1)
    report = analyze_model(m, input_size=input_size, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_analysis(report, output_dir)
    g = 1e9
    fps = 1000.0 / report.latency_mean_ms if report.latency_mean_ms > 0 else 0
    print(
        f"{model} @ {input_size}:\n"
        f"  params: {report.n_parameters:,}  ({report.n_trainable_parameters:,} trainable)\n"
        f"  GFLOPs: {report.flops / g:.2f}  MACs: {report.macs / g:.2f}\n"
        f"  latency: mean={report.latency_mean_ms:.2f}ms  p50={report.latency_p50_ms:.2f}ms  "
        f"p95={report.latency_p95_ms:.2f}ms\n"
        f"  FPS: {fps:.1f}\n"
        f"  peak VRAM: {report.peak_vram_bytes / 1e9:.2f} GB"
    )
    print(f"  saved → {output_dir / 'model_analysis.json'}")


@app.command(name="analyze-dataset")
def analyze_dataset_cmd(
    dataset: Annotated[
        str,
        Option("--dataset", "-d", help="Dataset root path"),
    ],
    output_dir: Annotated[
        Path,
        Option("--output-dir", "-o", help="Output directory"),
    ],
    splits: Annotated[
        str,
        Option("--splits", "-s", help="Comma-separated splits (default: train,valid,test)"),
    ] = "train,valid,test",
) -> None:
    """Analyze a dataset: diversity metrics + distribution plots.

    Outputs dataset_analysis.json + .csv (per-split + combined) and one PNG per
    analysis per split: bbox-size histogram (log bins), area-bucket bar,
    aspect-ratio histogram, objects-per-image histogram, bbox-center heatmap.
    """
    import csv as _csv

    from cracks_yolo.analysis.dataset import analyze_dataset
    from cracks_yolo.analysis.dataset import save_dataset_analysis
    from cracks_yolo.dataset.yolo import YOLOSource
    from cracks_yolo.viz import dataset as dsviz

    src = YOLOSource(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_list = [s.strip() for s in splits.split(",") if s.strip()]

    # For the combined summary CSV/Excel
    summary_rows: list[dict] = []

    all_records: list = []
    for split in split_list:
        records = src.load_split(split)
        if not records:
            print(f"  {split}: (empty) skip")
            continue
        all_records.extend(records)
        report = analyze_dataset(records)
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        save_dataset_analysis(report, split_dir)

        # Per-split plots
        dsviz.plot_bbox_size_distribution(records, split_dir / "bbox_size.png")
        dsviz.plot_bbox_area_buckets(records, split_dir / "area_buckets.png")
        dsviz.plot_aspect_ratio(records, split_dir / "aspect_ratio.png")
        dsviz.plot_objects_per_image(records, split_dir / "objects_per_image.png")
        dsviz.plot_bbox_position_heatmap(records, split_dir / "bbox_heatmap.png")

        d = report.to_dict()
        d["split"] = split
        summary_rows.append(d)
        print(
            f"  {split}: {report.n_images} images, {report.n_annotations} anns, "
            f"entropy={report.class_shannon_entropy:.3f}, "
            f"coverage={report.spatial_coverage:.3f}"
        )

    # Combined report + plots
    if all_records:
        report = analyze_dataset(all_records)
        save_dataset_analysis(report, output_dir)
        d = report.to_dict()
        d["split"] = "combined"
        summary_rows.append(d)
        dsviz.plot_bbox_size_distribution(all_records, output_dir / "bbox_size.png")
        dsviz.plot_bbox_area_buckets(all_records, output_dir / "area_buckets.png")
        dsviz.plot_aspect_ratio(all_records, output_dir / "aspect_ratio.png")
        dsviz.plot_objects_per_image(all_records, output_dir / "objects_per_image.png")
        dsviz.plot_bbox_position_heatmap(all_records, output_dir / "bbox_heatmap.png")
        print(
            f"\nCombined: {report.n_images} images, {report.n_annotations} anns, "
            f"imbalance={report.imbalance_ratio:.2f}, entropy={report.class_shannon_entropy:.3f}"
        )

    # Summary CSV: one row per split (+ combined)
    if summary_rows:
        summary_csv = output_dir / "dataset_summary.csv"

        fields = [
            "split",
            "n_images",
            "n_annotations",
            "n_classes",
            "imbalance_ratio",
            "class_shannon_entropy",
            "bbox_area_small",
            "bbox_area_medium",
            "bbox_area_large",
            "bbox_aspect_ratio_buckets",
            "spatial_coverage",
            "image_width_mean",
            "image_width_std",
            "image_height_mean",
            "image_height_std",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        # Excel
        try:
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.title = "dataset_summary"
            ws.append(fields)
            for row in summary_rows:
                ws.append([row.get(k, "") for k in fields])
            wb.save(output_dir / "dataset_summary.xlsx")
        except ImportError:
            pass
        print(f"\nSummary: {summary_csv}")

    print(f"\nAnalysis saved to {output_dir}/")
