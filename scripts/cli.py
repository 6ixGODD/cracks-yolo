"""Typer CLI for cracks_yolo: train / test / compose."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from typer import Option

app = typer.Typer(
    name="cracks-yolo",
    help="Tongue surface crack detection model zoo.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command()
def train(
    model: Annotated[str, Option("--model", "-m", help="ZOO key (e.g. yolov5s_sac)")],
    dataset: Annotated[str, Option("--dataset", "-d", help="Dataset root path")],
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")],
    epochs: Annotated[int, Option("--epochs", "-e", help="Number of epochs")] = 300,
    batch_size: Annotated[int, Option("--batch-size", "-b", help="Batch size")] = 64,
    lr: Annotated[float, Option("--lr", help="Learning rate")] = 1e-3,
    pretrained: Annotated[
        bool, Option("--pretrained/--no-pretrained", help="Load COCO pretrained weights")
    ] = True,
    device: Annotated[str, Option("--device", help="Device (cuda/cpu)")] = "cuda",
    seed: Annotated[int, Option("--seed", help="Random seed")] = 42,
    num_workers: Annotated[int, Option("--num-workers", "-w", help="DataLoader workers")] = 8,
    optimizer: Annotated[str, Option("--optimizer", help="Optimizer (adamw/sgd)")] = "adamw",
    cosine_lr: Annotated[
        bool, Option("--cosine-lr/--no-cosine-lr", help="Cosine LR scheduler")
    ] = True,
    use_ema: Annotated[bool, Option("--ema/--no-ema", help="Exponential moving average")] = True,
    early_stopping_patience: Annotated[
        int, Option("--patience", help="Early stopping patience")
    ] = 100,
    clip_grad_norm: Annotated[
        float, Option("--clip-grad-norm", help="Gradient clipping max norm")
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
    model: Annotated[str, Option("--model", "-m", help="ZOO key")],
    weights: Annotated[Path, Option("--weights", help="Path to best.pt")],
    dataset: Annotated[str, Option("--dataset", "-d", help="Dataset root path")],
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")],
    batch_size: Annotated[int, Option("--batch-size", "-b", help="Batch size")] = 32,
    device: Annotated[str, Option("--device", help="Device")] = "cuda",
    seed: Annotated[int, Option("--seed", help="Random seed")] = 42,
) -> None:
    """Test a trained model on test + val splits."""
    from cracks_yolo.pipeline.test import run_test

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
    config: Annotated[Path, Option("--config", "-c", help="Single experiment YAML config file")],
    output_dir: Annotated[
        Path | None, Option("--output-dir", "-o", help="Override output directory")
    ] = None,
    device: Annotated[str | None, Option("--device", help="Override device (cuda/cpu)")] = None,
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

    if exp_type == "train":
        run_train(**cfg)
    elif exp_type == "test":
        run_test(**cfg)
    else:
        raise ValueError(f"unknown experiment type: {exp_type}")


@app.command()
def compose(
    config: Annotated[Path, Option("--config", "-c", help="Compose YAML with $include")],
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")],
    max_parallel: Annotated[
        int, Option("--max-parallel", "-p", help="Max parallel experiments")
    ] = 1,
) -> None:
    """Run batch experiments from a YAML compose config."""
    from cracks_yolo.pipeline.compose import run_compose

    run_compose(config=config, output_dir=output_dir, max_parallel=max_parallel)


if __name__ == "__main__":
    app()
