"""UltralyticsAdapter — wraps ultralytics YOLO/RTDETR for all YOLO families.

Training: delegates to YOLO().train() / RTDETR().train().
Inference: eval forward → decode to InferenceResult.
Pretrained: YOLO(asset.pt) → intersect state_dict.
SAC/TR: apply_sac_tr() at construction time.
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
from cracks_yolo.zoo.ultralytics.configs import MODEL_LOOKUP
from cracks_yolo.zoo.ultralytics.configs import ULTRALYTICS_MODELS as _MODELS
from cracks_yolo.zoo.ultralytics.sac_injection import apply_sac_tr


class UltralyticsAdapter(BaseModel):
    """Wraps an ultralytics DetectionModel for any YOLO family."""

    def __init__(
        self,
        model_name: str = "",
        num_classes: int = 1,
        input_size: int = 640,
        logger: Any = None,
        **_extra_kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, logger=logger)
        self.model_name = model_name
        if model_name not in MODEL_LOOKUP:
            raise ValueError(f"unknown model: {model_name}")

        _, cfg, asset, sac_idx, tr_idx, decode_fmt, use_rtdetr = MODEL_LOOKUP[model_name]
        self._cfg = cfg
        self._asset = asset
        self._sac_indices = sac_idx
        self._tr_indices = tr_idx
        self._decode_format = decode_fmt
        self._use_rtdetr = use_rtdetr

        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import DEFAULT_CFG

        self._inner: nn.Module = DetectionModel(cfg, ch=3, nc=num_classes, verbose=False)
        self._inner.args = DEFAULT_CFG
        if sac_idx or tr_idx:
            apply_sac_tr(self._inner, sac_indices=sac_idx, tr_indices=tr_idx)

    @property
    def stride(self) -> torch.Tensor:
        return self._inner.stride

    def forward(self, x: torch.Tensor) -> Any:
        return self._inner(x)

    def train_model(self, config: TrainConfig) -> TrainReport:
        from ultralytics import RTDETR
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionTrainer

        cls = RTDETR if self._use_rtdetr else YOLO
        trainer = cls(self._cfg, verbose=False)  # suppress: throwaway instance, real training below
        trainer.model = self._inner
        trainer.model.args = trainer.model.args or {}

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

        # Build a custom trainer class that preserves SAC/TR injection.
        # Ultralytics' YOLO.train() internally creates a fresh DetectionTrainer whose
        # BaseTrainer.__init__ sets self.model to the yaml path string (line 184).
        # Then _setup_train() → setup_model() sees a string (not nn.Module) and calls
        # get_model() which creates a vanilla DetectionModel — undoing our SAC/TR.
        #
        # Fix: override __init__ to replace self.model with our SAC-injected nn.Module
        # so setup_model()'s isinstance(model, nn.Module) check passes and it returns
        # early. The get_model() override is a safety net.
        sac_model = self._inner
        sac_indices = self._sac_indices
        tr_indices = self._tr_indices
        _apply_sac_tr = apply_sac_tr

        class SACDetectionTrainer(DetectionTrainer):
            def __init__(self, overrides=None, _callbacks=None):
                super().__init__(overrides, _callbacks)
                self.model = sac_model  # Replace yaml path string with SAC-injected nn.Module

            def get_model(self, cfg=None, weights=None, verbose=True):
                model = super().get_model(cfg, weights, verbose)
                if sac_indices or tr_indices:
                    _apply_sac_tr(model, sac_indices=sac_indices, tr_indices=tr_indices)
                return model

        trainer.train(
            data=data_yaml,
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=self.input_size,
            device=config.device,
            workers=config.num_workers,
            project=str(config.output_dir.parent),
            name=config.output_dir.name,
            exist_ok=True,
            optimizer=config.optimizer,
            lr0=config.lr,
            lrf=config.cosine_lrf,
            cos_lr=config.cosine_lr,
            patience=config.early_stopping_patience or 100,
            single_cls=self.num_classes == 1,
            pretrained=False,
            verbose=True,
            trainer=SACDetectionTrainer,
            **config.extra_kwargs,
        )

        self._inner = trainer.model
        self._set_state(ModelState.TRAINED)
        return self._build_train_report(config)

    def inference(self, images: torch.Tensor) -> list[InferenceResult]:
        self._assert_state(ModelState.TRAINED, "inference")
        self._inner.eval()
        results: list[InferenceResult] = []
        with torch.no_grad():
            outs = self._inner(images)
            if isinstance(outs, (list, tuple)) and len(outs) > 0:
                pred = outs[0]
            elif isinstance(outs, torch.Tensor):
                pred = outs
            elif isinstance(outs, dict):
                pred = outs.get("one2one", outs.get("one2many", None))
                if pred is None:
                    return []
            else:
                return []

            if not isinstance(pred, torch.Tensor):
                return []

            bsz = pred.shape[0]
            for b in range(bsz):
                if self._decode_format == "anchor_free":
                    d = (
                        pred[b].permute(1, 0)
                        if pred.ndim == 3 and pred.shape[1] < pred.shape[2]
                        else pred[b]
                    )
                    boxes = d[:, :4]
                    cls_data = d[:, 4 : 4 + self.num_classes]
                    scores = cls_data.max(dim=1).values
                    labels = cls_data.argmax(dim=1)
                else:
                    d = pred[b]
                    boxes = d[:, :4]
                    scores = d[:, 4]
                    labels = (
                        d[:, 5].long()
                        if d.shape[-1] > 5
                        else torch.zeros(d.shape[0], dtype=torch.long)
                    )

                mask = scores > 0.001
                results.append(
                    InferenceResult(
                        boxes=boxes[mask].cpu(),
                        scores=scores[mask].cpu(),
                        labels=labels[mask].cpu(),
                    )
                )
        return results

    def save(self, path: Path, torchscript: bool = False, onnx: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self._inner.state_dict()}, path)
        if torchscript or onnx:
            from ultralytics import RTDETR
            from ultralytics import YOLO

            cls = RTDETR if self._use_rtdetr else YOLO
            trainer = cls(self._cfg)
            trainer.model = self._inner
            if torchscript:
                trainer.export(format="torchscript")
            if onnx:
                trainer.export(format="onnx")

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        self._inner.load_state_dict(sd, strict=False)
        self._set_state(ModelState.TRAINED)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "",
        num_classes: int = 1,
        **kwargs: Any,
    ) -> UltralyticsAdapter:
        model = cls(model_name=model_name, num_classes=num_classes, **kwargs)
        model._load_pretrained_weights()
        return model

    def _load_pretrained_weights(self) -> None:
        from ultralytics import RTDETR
        from ultralytics import YOLO

        cls = RTDETR if self._use_rtdetr else YOLO
        try:
            src = cls(f"{self._asset}.pt").model.state_dict()
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


# Factory: build a class for each model name
def _make_class(name: str) -> type[UltralyticsAdapter]:
    return type(name.replace("-", "_").title().replace("_", ""), (UltralyticsAdapter,), {})


ZOO: dict[str, type[UltralyticsAdapter]] = {name: _make_class(name) for name, *_ in _MODELS}
