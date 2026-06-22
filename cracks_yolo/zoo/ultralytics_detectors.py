"""Ultralytics-backed DetectorModel adapters for all YOLO families (v3/v5/v8/
v9/v10). SAC/TR variants are produced by runtime module replacement
(:func:`cracks_yolo.zoo.ultralytics_sac.apply_sac_tr`) — no per-architecture
reimplementation.

All models register under clean names (no _official suffix):
yolov5s, yolov5s_sactr, yolov8s, yolov8s_sac, yolov9c, yolov10s, yolov3_tiny, ...

Pretrained COCO weights load via ``YOLO(asset).model`` intersect + strict=False;
SAC/TR layers (SAConv2d switches, Transformer blocks) have no COCO counterpart
and stay randomly initialized.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.ultralytics_sac import apply_sac_tr

# Ultralytics asset name for each cfg (so we can fetch COCO pretrained).
_CFG_ASSET = {
    "yolov3": "yolov3u", "yolov3-tiny": "yolov3-tinyu", "yolov3-spp": "yolov3-sppu",
    "yolov5n": "yolov5nu", "yolov5s": "yolov5su", "yolov5m": "yolov5mu",
    "yolov5l": "yolov5lu", "yolov5x": "yolov5xu",
    "yolov8n": "yolov8n", "yolov8s": "yolov8s", "yolov8m": "yolov8m",
    "yolov8l": "yolov8l", "yolov8x": "yolov8x",
    "yolov9t": "yolov9t", "yolov9s": "yolov9s", "yolov9m": "yolov9m",
    "yolov9c": "yolov9c", "yolov9e": "yolov9e",
    "yolov10n": "yolov10n", "yolov10s": "yolov10s", "yolov10m": "yolov10m",
    "yolov10b": "yolov10b", "yolov10l": "yolov10l", "yolov10x": "yolov10x",
}

# SAC/TR insertion indices per family (into model.model Sequential).
# v5: backbone C3 at 2,4,6 (P2/P3/P4); TR at 8 (P5). v8: C2f at 2,4,6,8.
_SAC_TR = {
    "v5": {"sac": (2, 4, 6), "tr": (8,)},
    "v8": {"sac": (2, 4, 6, 8), "tr": ()},
    "v9": {"sac": (2, 4, 6, 8), "tr": ()},
    "v10": {"sac": (2, 4, 6, 8), "tr": ()},
    "v3": {"sac": (), "tr": ()},  # v3 uses BottleneckCSP, not C3/C2f — no SAC/TR.
}


def _family_for_cfg(cfg: str) -> str:
    for fam in ("yolov5", "yolov8", "yolov9", "yolov10", "yolov3"):
        if cfg.startswith(fam):
            return fam.replace("yolov", "v")
    return ""


class _UltralyticsDetector(nn.Module):
    cfg = ""
    sac_indices: tuple[int, ...] = ()
    tr_indices: tuple[int, ...] = ()
    pretrained_spec: PretrainedSpec | None = None
    loss_parts_schema = ("box", "cls", "dfl")
    decode_format = "anchor_free"

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__()
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import DEFAULT_CFG

        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self._inner = DetectionModel(self.cfg, ch=3, nc=num_classes, verbose=False)
        self._inner.args = DEFAULT_CFG
        if self.sac_indices or self.tr_indices:
            apply_sac_tr(self._inner, sac_indices=self.sac_indices, tr_indices=self.tr_indices)

    @property
    def stride(self) -> torch.Tensor:
        return self._inner.stride

    def forward(self, x: torch.Tensor) -> Any:
        return self._inner(x)

    def compute_loss(self, preds, targets, imgs=None):
        batch = {"batch_idx": targets[:, 0].long(), "cls": targets[:, 1:2], "bboxes": targets[:, 2:6]}
        loss, parts = self._inner.loss(batch, preds)
        return loss.sum(), parts

    def decode(self, preds):
        decoded = preds if isinstance(preds, torch.Tensor) else None
        if isinstance(preds, (tuple, list)) and preds and isinstance(preds[0], torch.Tensor):
            decoded = preds[0]
        if decoded is not None:
            if decoded.ndim == 3 and decoded.shape[-1] == 6:
                boxes = decoded[..., :4]
                out = decoded.new_zeros((*decoded.shape[:2], 5 + self.num_classes))
                out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
                out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
                out[..., 2] = boxes[..., 2] - boxes[..., 0]
                out[..., 3] = boxes[..., 3] - boxes[..., 1]
                out[..., 4] = 1
                labels = decoded[..., 5].long().clamp(0, self.num_classes - 1)
                out.scatter_(2, (labels + 5).unsqueeze(-1), decoded[..., 4:5])
                return out
            return decoded
        if isinstance(preds, dict):
            for key in ("one2one", "one2many"):
                v = preds.get(key)
                if isinstance(v, torch.Tensor):
                    return v
        raise TypeError(f"Unsupported Ultralytics prediction type: {type(preds)}")

    def build_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    @classmethod
    def from_pretrained(cls, num_classes, weights_dir=None, strict=False):
        m = cls(num_classes=num_classes)
        cfg_key = cls.cfg.replace(".yaml", "")
        asset = _CFG_ASSET.get(cfg_key, cfg_key)
        # Load COCO weights via ultralytics' YOLO() (handles v5/v8/v9/v10 formats).
        try:
            from ultralytics import YOLO
            src_sd = YOLO(f"{asset}.pt").model.state_dict()
            msd = m.state_dict()
            matched = 0
            for k in list(src_sd.keys()):
                # src keys are "model.<i>...."; our wrapper prefixes "_inner.".
                tgt = f"_inner.{k}" if not k.startswith("_inner.") else k
                if tgt in msd and src_sd[k].shape == msd[tgt].shape:
                    msd[tgt] = src_sd[k]
                    matched += 1
            m.load_state_dict(msd, strict=False)
            print(f"pretrained {asset}: matched {matched}/{len(msd)} keys")
        except Exception as e:
            print(f"WARNING: pretrained load failed for {asset}: {e}")
        return m


def _make_class(name, cfg, sac=None, tr=None, decode_format="anchor_free"):
    fam = _family_for_cfg(cfg)
    spec = _SAC_TR.get(fam, {"sac": (), "tr": ()})
    # sac/tr=None → use family default; sac/tr=() → explicitly none (baseline).
    sac_idx = spec.get("sac", ()) if sac is None else tuple(sac)
    tr_idx = spec.get("tr", ()) if tr is None else tuple(tr)
    return type(name, (_UltralyticsDetector,), {
        "cfg": cfg,
        "sac_indices": sac_idx,
        "tr_indices": tr_idx,
        "decode_format": decode_format,
        "pretrained_spec": PretrainedSpec(
            key=_CFG_ASSET.get(cfg, cfg), url="", state_dict_key_map={},
        ),
    })


ULTRALYTICS_ZOO: dict[str, type[_UltralyticsDetector]] = {}

# v3 family (BottleneckCSP — no SAC/TR insertion defined).
ULTRALYTICS_ZOO["yolov3"] = _make_class("YOLOv3", "yolov3.yaml", sac=(), tr=())
ULTRALYTICS_ZOO["yolov3_tiny"] = _make_class("YOLOv3Tiny", "yolov3-tiny.yaml", sac=(), tr=())
ULTRALYTICS_ZOO["yolov3_spp"] = _make_class("YOLOv3SPP", "yolov3-spp.yaml", sac=(), tr=())

# v5 family: n/s/m/l/x baseline; s gets sac/tr/sactr variants.
for _sz in ("n", "s", "m", "l", "x"):
    ULTRALYTICS_ZOO[f"yolov5{_sz}"] = _make_class(f"YOLOv5{_sz.upper()}", f"yolov5{_sz}.yaml", sac=(), tr=())
ULTRALYTICS_ZOO["yolov5s_sac"] = _make_class("YOLOv5sSAC", "yolov5s.yaml", sac=(2, 4, 6), tr=())
ULTRALYTICS_ZOO["yolov5s_tr"] = _make_class("YOLOv5sTR", "yolov5s.yaml", sac=(), tr=(8,))
ULTRALYTICS_ZOO["yolov5s_sactr"] = _make_class("YOLOv5sSACTR", "yolov5s.yaml", sac=(2, 4, 6), tr=(8,))

# v8 family: n/s/m/l/x baseline; n/s get sac.
for _sz in ("n", "s", "m", "l", "x"):
    ULTRALYTICS_ZOO[f"yolov8{_sz}"] = _make_class(f"YOLOv8{_sz.upper()}", f"yolov8{_sz}.yaml", sac=(), tr=())
ULTRALYTICS_ZOO["yolov8n_sac"] = _make_class("YOLOv8nSAC", "yolov8n.yaml", sac=(2, 4, 6, 8), tr=())
ULTRALYTICS_ZOO["yolov8s_sac"] = _make_class("YOLOv8sSAC", "yolov8s.yaml", sac=(2, 4, 6, 8), tr=())

# v9 family
for _sz in ("t", "s", "m", "c", "e"):
    ULTRALYTICS_ZOO[f"yolov9{_sz}"] = _make_class(f"YOLOv9{_sz.upper()}", f"yolov9{_sz}.yaml", sac=(), tr=())
ULTRALYTICS_ZOO["yolov9c_sac"] = _make_class("YOLOv9cSAC", "yolov9c.yaml", sac=(2, 4, 6, 8), tr=())

# v10 family (anchor_based decode)
for _sz in ("n", "s", "m", "b", "l", "x"):
    ULTRALYTICS_ZOO[f"yolov10{_sz}"] = _make_class(
        f"YOLOv10{_sz.upper()}", f"yolov10{_sz}.yaml", sac=(), tr=(), decode_format="anchor_based")
ULTRALYTICS_ZOO["yolov10s_sac"] = _make_class(
    "YOLOv10sSAC", "yolov10s.yaml", sac=(2, 4, 6, 8), tr=(), decode_format="anchor_based")


__all__ = ["ULTRALYTICS_ZOO", "_UltralyticsDetector"]
