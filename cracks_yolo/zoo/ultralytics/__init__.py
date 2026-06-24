"""UltralyticsAdapter and explicit model classes for each YOLO variant.

Every model gets its own class that inherits from ``UltralyticsAdapter``.
Each class hardcodes its own: cfg yaml, pretrained asset, SAC/TR indices,
decode format, and whether to use RTDETR.

Training: creates ``DetectionTrainer`` directly, sets ``trainer.model = self._inner``
so ultralytics' ``setup_model()`` sees an ``nn.Module`` and returns early —
preserving our SAC/TR injection.

Inference: eval forward → decode to ``InferenceResult``.
Pretrained: load COCO .pt → intersect state_dict.
SAC/TR: ``apply_sac_tr()`` operates on ``_inner.model`` (the ``nn.Sequential``
returned by ``parse_model``), replacing C3/C2f blocks by index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cracks_yolo.zoo.base import BaseModel
from cracks_yolo.zoo.base import InferenceResult
from cracks_yolo.zoo.base import ModelState
from cracks_yolo.zoo.base import TrainConfig
from cracks_yolo.zoo.base import TrainReport
from cracks_yolo.zoo.ultralytics.sac_injection import apply_sac_tr


def _print_model_summary(
    model: nn.Module,
    sac_indices: tuple[int, ...],
    tr_indices: tuple[int, ...],
) -> None:
    """Print model backbone layers, marking SAC/TR injections."""
    seq = model.model  # nn.Sequential
    print(f"\n{'':>3} {'from':>4} {'n':>4}  {'params':>12}  {'module':<45}  {'arguments':<30}")
    print("-" * 110)
    for i, m in enumerate(seq):
        n_params = sum(p.numel() for p in m.parameters())
        mod_name = type(m).__module__ + "." + type(m).__name__
        # Get routing attrs
        f_val = getattr(m, "f", -1)
        n_val = getattr(m, "n", 1) if hasattr(m, "n") else 1
        tag = ""
        if i in sac_indices:
            tag = "  ← SAC"
        elif i in tr_indices:
            tag = "  ← C3TR"
        print(
            f"{i:>3} {f_val!s:>4} {n_val!s:>4}  {n_params:>12,}  {mod_name:<45}  {'...':<30}{tag}"
        )
    print()


def _sync_ultralytics_output(ultra_dir: Path, target_dir: Path) -> None:
    """Copy ultralytics training artifacts into our output directory.

    ultralytics saves to ``project/name/``; we want everything under
    ``output_dir/`` so the pipeline can find results.csv, weights, plots, etc.
    """
    import shutil

    if ultra_dir.resolve() == target_dir.resolve():
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    for src in ultra_dir.iterdir():
        dst = target_dir / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


class UltralyticsAdapter(BaseModel):
    """Wraps an ultralytics ``DetectionModel`` for any YOLO family.

    Subclasses override constructor arguments to pin a specific architecture.
    """

    def __init__(
        self,
        *,
        cfg: str,
        asset: str = "",
        sac_indices: tuple[int, ...] = (),
        tr_indices: tuple[int, ...] = (),
        decode_format: str = "anchor_free",
        use_rtdetr: bool = False,
        num_classes: int = 1,
        input_size: int = 640,
        logger: Any = None,
    ) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, logger=logger)
        self._cfg = cfg
        self._asset = asset
        self._sac_indices = sac_indices
        self._tr_indices = tr_indices
        self._decode_format = decode_format
        self._use_rtdetr = use_rtdetr

        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import DEFAULT_CFG

        if use_rtdetr:
            from ultralytics.models.rtdetr.model import RTDETRDetectionModel

            self._inner: nn.Module = RTDETRDetectionModel(cfg, ch=3, nc=num_classes, verbose=False)
        else:
            self._inner: nn.Module = DetectionModel(cfg, ch=3, nc=num_classes, verbose=False)
        self._inner.args = DEFAULT_CFG
        if sac_indices or tr_indices:
            apply_sac_tr(self._inner, sac_indices=sac_indices, tr_indices=tr_indices)

        # Print model summary so user can verify SAC/TR injection visually.
        # setup_model() is skipped (we inject trainer.model = self._inner),
        # so ultralytics never prints its own summary.
        _print_model_summary(self._inner, sac_indices, tr_indices)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stride(self) -> torch.Tensor:
        return self._inner.stride

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Any:
        return self._inner(x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(self, config: TrainConfig) -> TrainReport:
        """Run training via ultralytics' ``DetectionTrainer`` directly.

        Does NOT go through ``YOLO().train()`` because that internally
        creates a fresh ``DetectionTrainer`` whose ``setup_model()`` calls
        ``get_model()`` — which builds a vanilla ``DetectionModel`` from
        yaml, undoing our SAC/TR injection.

        Instead we create the trainer ourselves, then set
        ``trainer.model = self._inner`` (our SAC-injected nn.Module).
        ``setup_model()`` checks ``isinstance(self.model, nn.Module)`` and
        returns early — training proceeds with the injected architecture.
        """
        if self._use_rtdetr:
            from ultralytics.models.rtdetr.train import RTDETRTrainer as Trainer
        else:
            from ultralytics.models.yolo.detect import DetectionTrainer as Trainer

        if config.pretrained and self._asset:
            self._load_pretrained_weights()

        data_yaml = config.data_yaml or str(config.output_dir / "data.yaml")
        if not config.data_yaml:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            Path(data_yaml).write_text(
                f"path: {config.dataset}\ntrain: train/images\nval: valid/images\n"
                f"test: test/images\n\nnc: {self.num_classes}\nnames: {{0: 'cracks'}}\n",
                encoding="utf-8",
            )

        # Only keys that exist in ultralytics DEFAULT_CFG_DICT.
        overrides: dict[str, Any] = {
            "model": self._cfg,
            "data": data_yaml,
            "epochs": config.epochs,
            "batch": config.batch_size,
            "imgsz": self.input_size,
            "device": config.device,
            "workers": config.num_workers,
            "project": str(config.output_dir.resolve().parent),
            "name": config.output_dir.name,
            "exist_ok": True,
            "optimizer": config.optimizer,
            "lr0": config.lr,
            "lrf": config.cosine_lrf,
            "cos_lr": config.cosine_lr,
            "patience": config.early_stopping_patience or 100,
            "single_cls": self.num_classes == 1,
            "pretrained": False,
            "verbose": True,
            "seed": config.seed,
            "amp": config.amp,
            "momentum": config.momentum,
        }
        if config.warmup_epochs:
            overrides["warmup_epochs"] = config.warmup_epochs

        trainer = Trainer(overrides=overrides)
        # Replace yaml-path-string model with our SAC-injected nn.Module.
        # setup_model() will see nn.Module and skip the get_model() rebuild.
        trainer.model = self._inner
        trainer.train()

        # After training: move ultralytics output into our output_dir if
        # ultralytics saved elsewhere (e.g. when config.output_dir is relative)
        _sync_ultralytics_output(trainer.save_dir, config.output_dir)

        self._inner = trainer.model
        self._set_state(ModelState.TRAINED)
        return self._build_train_report(config)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inference(self, images: torch.Tensor) -> list[InferenceResult]:
        self._assert_state(ModelState.TRAINED, "inference")
        from ultralytics.utils.nms import non_max_suppression

        self._inner.eval()
        results: list[InferenceResult] = []
        with torch.no_grad():
            raw = self._inner(images)
            # non_max_suppression decodes raw grid-offset output → pixel xyxy
            preds = non_max_suppression(
                raw,
                conf_thres=0.001,
                iou_thres=0.7,
                nc=self.num_classes,
                max_det=300,
            )
            for det in preds:
                if det is None or len(det) == 0:
                    results.append(
                        InferenceResult(
                            boxes=torch.zeros((0, 4)),
                            scores=torch.zeros(0),
                            labels=torch.zeros(0, dtype=torch.long),
                        )
                    )
                else:
                    # det: (N, 6) = (x1, y1, x2, y2, conf, cls), clipped to image
                    boxes = det[:, :4].clamp(0, self.input_size)
                    results.append(
                        InferenceResult(
                            boxes=boxes.cpu(),
                            scores=det[:, 4].cpu(),
                            labels=det[:, 5].long().cpu(),
                        )
                    )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path, torchscript: bool = False, onnx: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self._inner.state_dict()}, path)
        if torchscript or onnx:
            from ultralytics import RTDETR
            from ultralytics import YOLO

            cls = RTDETR if self._use_rtdetr else YOLO
            exporter = cls(self._cfg, verbose=False)
            exporter.model = self._inner
            if torchscript:
                exporter.export(format="torchscript")
            if onnx:
                exporter.export(format="onnx")

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # ultralytics format: ckpt["model"] is a DetectionModel instance
        if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            sd = ckpt["model"].state_dict()
        else:
            sd = ckpt.get("model_state_dict", ckpt)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
        self._inner.load_state_dict(sd, strict=False)
        self._set_state(ModelState.TRAINED)

    # ------------------------------------------------------------------
    # Pretrained
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        num_classes: int = 1,
        **kwargs: Any,
    ) -> UltralyticsAdapter:
        model = cls(num_classes=num_classes, **kwargs)
        model._load_pretrained_weights()
        return model

    def _load_pretrained_weights(self) -> None:
        from ultralytics import RTDETR
        from ultralytics import YOLO

        cls = RTDETR if self._use_rtdetr else YOLO
        try:
            src = cls(f"{self._asset}.pt", verbose=False).model.state_dict()
        except Exception as e:
            if self._logger:
                self._logger.warning(f"pretrained load failed for {self._asset}: {e}")
            return

        msd = self._inner.state_dict()
        matched = 0
        for k in list(src.keys()):
            if k in msd and src[k].shape == msd[k].shape:
                msd[k] = src[k]
                matched += 1
        self._inner.load_state_dict(msd, strict=False)
        if self._logger:
            self._logger.info(f"pretrained {self._asset}: matched {matched}/{len(msd)} keys")
        self._set_state(ModelState.PRETRAINED)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _build_train_report(self, config: TrainConfig) -> TrainReport:
        import csv

        results_csv = config.output_dir / "results.csv"
        best_map50 = 0.0
        best_epoch = 0
        final_loss = 0.0
        final_map50 = 0.0
        final_map5095 = 0.0
        total_epochs = 0

        if results_csv.exists():
            rows = list(csv.DictReader(results_csv.open()))
            if rows:
                last = {k.strip(): v for k, v in rows[-1].items()}
                final_loss = float(last.get("train/box_loss", 0) or 0)
                final_map50 = float(
                    last.get("metrics/mAP50(B)", last.get("metrics/mAP_0.5", 0)) or 0
                )
                final_map5095 = float(
                    last.get("metrics/mAP50-95(B)", last.get("metrics/mAP_0.5:0.95", 0)) or 0
                )
                total_epochs = len(rows)
                for row in rows:
                    d = {k.strip(): v for k, v in row.items()}
                    m = float(d.get("metrics/mAP50(B)", d.get("metrics/mAP_0.5", 0)) or 0)
                    if m > best_map50:
                        best_map50 = m
                        best_epoch = int(d.get("epoch", 0))

        return TrainReport(
            output_dir=config.output_dir,
            best_epoch=best_epoch,
            best_map50=best_map50,
            final_train_loss=final_loss,
            final_val_map50=final_map50,
            final_val_map5095=final_map5095,
            total_epochs=total_epochs,
            elapsed_sec=0.0,
            checkpoint_path=config.output_dir / "weights" / "best.pt",
        )


# ======================================================================
# Explicit model classes — each hardcodes its architecture
# ======================================================================


class YOLOv3(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov3.yaml",
            asset="yolov3u",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv5n(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov5n.yaml",
            asset="yolov5nu",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv5s(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov5s.yaml",
            asset="yolov5su",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv5sSAC(UltralyticsAdapter):
    _CN = "YOLOv5s + SAC at backbone indices (2, 4, 6)"

    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov5s.yaml",
            asset="yolov5su",
            sac_indices=(2, 4, 6),
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv5sTR(UltralyticsAdapter):
    _CN = "YOLOv5s + C3TR at backbone index (8,)"

    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov5s.yaml",
            asset="yolov5su",
            tr_indices=(8,),
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv5sSACTR(UltralyticsAdapter):
    _CN = "YOLOv5s + SAC (2,4,6) + C3TR (8,)"

    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov5s.yaml",
            asset="yolov5su",
            sac_indices=(2, 4, 6),
            tr_indices=(8,),
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv5m(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov5m.yaml",
            asset="yolov5mu",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv5l(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov5l.yaml",
            asset="yolov5lu",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv5x(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov5x.yaml",
            asset="yolov5xu",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv6n(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov6.yaml",
            asset="yolov6n",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv6nSAC(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov6.yaml",
            asset="yolov6n",
            sac_indices=(2, 4, 6, 8),
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


# --- v8 ---------------------------------------------------------------


class YOLOv8n(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov8n.yaml",
            asset="yolov8n",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv8nSAC(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov8n.yaml",
            asset="yolov8n",
            sac_indices=(2, 4, 6, 8),
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv8s(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov8s.yaml",
            asset="yolov8s",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv8sSAC(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov8s.yaml",
            asset="yolov8s",
            sac_indices=(2, 4, 6, 8),
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv8m(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov8m.yaml",
            asset="yolov8m",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv8l(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov8l.yaml",
            asset="yolov8l",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv8x(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov8x.yaml",
            asset="yolov8x",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


# --- v9 ---------------------------------------------------------------


class YOLOv9t(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov9t.yaml",
            asset="yolov9t",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv9s(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov9s.yaml",
            asset="yolov9s",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv9m(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov9m.yaml",
            asset="yolov9m",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv9c(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov9c.yaml",
            asset="yolov9c",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv9cSAC(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov9c.yaml",
            asset="yolov9c",
            sac_indices=(2, 4, 6, 8),
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv9e(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov9e.yaml",
            asset="yolov9e",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


# --- v10 --------------------------------------------------------------


class YOLOv10n(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov10n.yaml",
            asset="yolov10n",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv10s(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov10s.yaml",
            asset="yolov10s",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv10sSAC(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov10s.yaml",
            asset="yolov10s",
            sac_indices=(2, 4, 6, 8),
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv10m(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov10m.yaml",
            asset="yolov10m",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv10b(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov10b.yaml",
            asset="yolov10b",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv10l(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov10l.yaml",
            asset="yolov10l",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLOv10x(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolov10x.yaml",
            asset="yolov10x",
            decode_format="anchor_based",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


# --- RT-DETR ----------------------------------------------------------


class RTDETRr50(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="rtdetr-resnet50.yaml",
            asset="rtdetr-resnet50",
            decode_format="anchor_free",
            use_rtdetr=True,
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class RTDETRr50SAC(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="rtdetr-resnet50.yaml",
            asset="rtdetr-resnet50",
            sac_indices=(2, 4, 6, 8),
            decode_format="anchor_free",
            use_rtdetr=True,
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


# --- YOLO11 / YOLO12 / YOLO26 (baseline only) -------------------------


class YOLO11n(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolo11n.yaml",
            asset="yolo11n",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLO11s(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolo11s.yaml",
            asset="yolo11s",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLO12n(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolo12n.yaml",
            asset="yolo12n",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLO12s(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolo12s.yaml",
            asset="yolo12s",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLO26n(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolo26n.yaml",
            asset="yolo26n",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


class YOLO26s(UltralyticsAdapter):
    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(
            cfg="yolo26s.yaml",
            asset="yolo26s",
            decode_format="anchor_free",
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
        )


# ======================================================================
# ZOO registry — maps short names to explicit classes
# ======================================================================

ZOO: dict[str, type[UltralyticsAdapter]] = {
    # v3
    "yolov3": YOLOv3,
    # v5
    "yolov5n": YOLOv5n,
    "yolov5s": YOLOv5s,
    "yolov5s_sac": YOLOv5sSAC,
    "yolov5s_tr": YOLOv5sTR,
    "yolov5s_sactr": YOLOv5sSACTR,
    "yolov5m": YOLOv5m,
    "yolov5l": YOLOv5l,
    "yolov5x": YOLOv5x,
    # v6
    "yolov6n": YOLOv6n,
    "yolov6n_sac": YOLOv6nSAC,
    # v8
    "yolov8n": YOLOv8n,
    "yolov8n_sac": YOLOv8nSAC,
    "yolov8s": YOLOv8s,
    "yolov8s_sac": YOLOv8sSAC,
    "yolov8m": YOLOv8m,
    "yolov8l": YOLOv8l,
    "yolov8x": YOLOv8x,
    # v9
    "yolov9t": YOLOv9t,
    "yolov9s": YOLOv9s,
    "yolov9m": YOLOv9m,
    "yolov9c": YOLOv9c,
    "yolov9c_sac": YOLOv9cSAC,
    "yolov9e": YOLOv9e,
    # v10
    "yolov10n": YOLOv10n,
    "yolov10s": YOLOv10s,
    "yolov10s_sac": YOLOv10sSAC,
    "yolov10m": YOLOv10m,
    "yolov10b": YOLOv10b,
    "yolov10l": YOLOv10l,
    "yolov10x": YOLOv10x,
    # RT-DETR
    "rtdetr_r50": RTDETRr50,
    "rtdetr_r50_sac": RTDETRr50SAC,
    # YOLO11 / 12 / 26
    "yolo11n": YOLO11n,
    "yolo11s": YOLO11s,
    "yolo12n": YOLO12n,
    "yolo12s": YOLO12s,
    "yolo26n": YOLO26n,
    "yolo26s": YOLO26s,
}
