"""Generate per-model and composed scheduler YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
MODEL_DIR = EXPERIMENTS / "models"
DATASET = "data/CrackDetection_Augmentation.v1.yolov5pytorch"


def _dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _official_v5(name: str) -> list[dict[str, Any]]:
    size = name.removeprefix("yolov5")[0]
    batch = {"n": 128, "s": 64, "m": 32, "l": 16, "x": 8}[size]
    epochs = 400 if size in {"n", "s"} else 300 if size == "m" else 250
    output = f"output/{name}"
    return [
        {
            "name": f"{name}_train",
            "type": "v5_official_train",
            "model": name,
            "dataset": DATASET,
            "output_dir": output,
            "weights": f"weights/yolov5{size}.pt",
            "epochs": epochs,
            "batch_size": batch,
            "input_size": 640,
            "device": "0",
            "num_workers": 8,
            "early_stopping_patience": 30,
        },
        {
            "name": f"{name}_test",
            "type": "v5_official_test",
            "model": name,
            "dataset": DATASET,
            "weights": f"{output}/weights/best.pt",
            "output_dir": f"{output}/test",
            "batch_size": batch,
            "input_size": 640,
            "device": "0",
        },
    ]


def main() -> None:
    existing: list[dict[str, Any]] = []
    if MODEL_DIR.exists():
        for path in MODEL_DIR.glob("*.yaml"):
            existing.extend(
                (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("experiments", [])
            )
    else:
        source = yaml.safe_load(
            (EXPERIMENTS / "all_models_direct.yaml").read_text(encoding="utf-8")
        )
        existing = source.get("experiments", [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for experiment in existing:
        model = str(experiment["model"])
        if not model.startswith("yolov5"):
            grouped.setdefault(model, []).append(experiment)

    grouped["detr_r50"] = [
        {
            "name": "detr_r50_train",
            "type": "train",
            "model": "detr_r50",
            "dataset": DATASET,
            "output_dir": "output/detr_r50",
            "epochs": 150,
            "batch_size": 4,
            "lr": 0.0001,
            "input_size": 640,
            "device": "cuda",
            "num_workers": 8,
            "pretrained": True,
            "clip_grad_norm": 0.1,
            "early_stopping_patience": 20,
        },
        {
            "name": "detr_r50_test",
            "type": "test",
            "model": "detr_r50",
            "weights": "output/detr_r50/best.pt",
            "dataset": DATASET,
            "output_dir": "output/detr_r50/test",
            "batch_size": 4,
            "input_size": 640,
            "device": "cuda",
            "num_workers": 8,
        },
    ]

    for model, batch in (
        ("yolov3_official", 32),
        ("yolov3_tiny_official", 64),
        ("yolov3_spp_official", 24),
    ):
        grouped[model] = [
            {
                "name": f"{model}_train",
                "type": "train",
                "model": model,
                "dataset": DATASET,
                "output_dir": f"output/{model}",
                "epochs": 200,
                "batch_size": batch,
                "lr": 0.0001,
                "input_size": 640,
                "device": "cuda",
                "num_workers": 8,
                "pretrained": True,
                "clip_grad_norm": 10.0,
                "early_stopping_patience": 30,
            },
            {
                "name": f"{model}_test",
                "type": "test",
                "model": model,
                "weights": f"output/{model}/best.pt",
                "dataset": DATASET,
                "output_dir": f"output/{model}/test",
                "batch_size": batch,
                "input_size": 640,
                "device": "cuda",
                "num_workers": 8,
            },
        ]

    for model, batch, epochs in (
        ("yolov9t_official", 128, 400),
        ("yolov9s_official", 64, 400),
        ("yolov9m_official", 32, 300),
        ("yolov9e_official", 8, 250),
        ("yolov10n_official", 128, 400),
        ("yolov10m_official", 32, 300),
        ("yolov10b_official", 16, 300),
        ("yolov10l_official", 12, 250),
        ("yolov10x_official", 8, 250),
    ):
        grouped[model] = [
            {
                "name": f"{model}_train",
                "type": "train",
                "model": model,
                "dataset": DATASET,
                "output_dir": f"output/{model}",
                "epochs": epochs,
                "batch_size": batch,
                "lr": 0.001,
                "input_size": 640,
                "device": "cuda",
                "num_workers": 8,
                "pretrained": True,
                "clip_grad_norm": 10.0,
                "early_stopping_patience": 30,
            },
            {
                "name": f"{model}_test",
                "type": "test",
                "model": model,
                "weights": f"output/{model}/best.pt",
                "dataset": DATASET,
                "output_dir": f"output/{model}/test",
                "batch_size": batch,
                "input_size": 640,
                "device": "cuda",
                "num_workers": 8,
            },
        ]

    variants = ("baseline", "sac", "tr", "sactr")
    for size in "nsmlx":
        for variant in variants:
            name = f"yolov5{size}_{variant}"
            grouped[name] = _official_v5(name)

    model_names = sorted(grouped)
    for name in model_names:
        _dump(
            MODEL_DIR / f"{name}.yaml",
            {"scheduler": {"max_parallel": 1}, "experiments": grouped[name]},
        )

    relative = [f"models/{name}.yaml" for name in model_names]
    _dump(
        EXPERIMENTS / "all_models_direct.yaml",
        {"scheduler": {"max_parallel": 1}, "$include": relative},
    )
    _dump(
        EXPERIMENTS / "all_models_compose.yaml",
        {"scheduler": {"max_parallel": 1}, "$include": relative},
    )
    for count in (4, 6):
        for index in range(count):
            includes = relative[index::count]
            _dump(
                EXPERIMENTS / f"compose_{count}" / f"group_{index + 1}.yaml",
                {"scheduler": {"max_parallel": 1}, "$include": [f"../{item}" for item in includes]},
            )


if __name__ == "__main__":
    main()
