"""Train pipeline: thin wrapper around BaseModel.train_model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cracks_yolo.zoo import ZOO
from cracks_yolo.zoo.base import BaseModel
from cracks_yolo.zoo.base import TrainConfig
from cracks_yolo.zoo.base import TrainReport


def run_train(
    model_name: str,
    dataset: str,
    output_dir: Path,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    pretrained: bool = True,
    device: str = "cuda",
    seed: int = 42,
    num_workers: int = 8,
    **kwargs: Any,
) -> TrainReport:
    """Build a model, train it, save artifacts.

    Args:
        model_name: ZOO key (e.g. "yolov5s_sac").
        dataset: dataset root path.
        output_dir: where to save checkpoints + metrics.
        epochs, batch_size, lr, etc.: training hyperparams.
        pretrained: load COCO pretrained weights before training.
    """
    if model_name not in ZOO:
        raise ValueError(f"unknown model: {model_name}. Available: {sorted(ZOO.keys())}")

    cls = ZOO[model_name]

    # Build model — ultralytics models take model_name, torchvision don't
    import inspect

    sig = inspect.signature(cls.__init__)
    if "model_name" in sig.parameters:
        model: BaseModel = cls(model_name=model_name, num_classes=1, logger=None)
    else:
        model = cls(num_classes=1, logger=None)

    if pretrained and hasattr(model, "from_pretrained") and hasattr(cls, "from_pretrained"):
        # Reload via from_pretrained
        if "model_name" in sig.parameters:
            model = cls.from_pretrained(model_name=model_name, num_classes=1)
        else:
            model = cls.from_pretrained(num_classes=1)

    config = TrainConfig(
        output_dir=output_dir,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        pretrained=pretrained,
        device=device,
        seed=seed,
        num_workers=num_workers,
        **kwargs,
    )

    report = model.train_model(config)

    # Post-train: save + analyze
    model.save(output_dir / "best.pt", torchscript=False, onnx=False)
    try:
        analysis = model.analyze(device=device)
        import json

        (output_dir / "analysis.json").write_text(
            json.dumps(
                {
                    "model_name": analysis.model_name,
                    "n_parameters": analysis.n_parameters,
                    "n_trainable_parameters": analysis.n_trainable_parameters,
                    "gflops": analysis.gflops,
                    "fps_mean": analysis.fps_mean,
                    "latency_mean_ms": analysis.latency_mean_ms,
                    "peak_vram_bytes": analysis.peak_vram_bytes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    print(f"Train complete: best_epoch={report.best_epoch} best_map50={report.best_map50:.4f}")
    print(f"  output: {output_dir}")
    return report
