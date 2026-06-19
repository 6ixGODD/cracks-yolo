"""YOLOv10 E2E loss: dual one2many + one2one ``v8DetectionLoss`` instances.

The one2many head supervises training (topk=10). The one2one head enables
NMS-free inference (topk=7, topk2=1). An exponential decay schedule shifts
weight from one2many → one2one over training.

Ported from ``deps/ultralytics/ultralytics/utils/loss.py`` (``E2ELoss``).
"""

from __future__ import annotations

import torch

from cracks_yolo.losses.yolov8 import v8DetectionLoss


class E2ELoss:
    """YOLOv10 end-to-end loss.

    Wraps two :class:`v8DetectionLoss` instances with topk settings for the
    one2many (training supervision) and one2one (inference) heads.

    Args:
        nc, reg_max, stride, hyp, device: Forwarded to each v8DetectionLoss.
        initial_o2m: Initial one2many weight (default 0.8).
        final_o2m: Final one2many weight after decay (default 0.1).
    """

    def __init__(
        self,
        nc: int,
        reg_max: int,
        stride: torch.Tensor,
        hyp: dict[str, float],
        device: torch.device,
        initial_o2m: float = 0.8,
        final_o2m: float = 0.1,
    ) -> None:
        self.one2many = v8DetectionLoss(
            nc=nc, reg_max=reg_max, stride=stride, hyp=hyp, device=device, tal_topk=10
        )
        self.one2one = v8DetectionLoss(
            nc=nc,
            reg_max=reg_max,
            stride=stride,
            hyp=hyp,
            device=device,
            tal_topk=7,
            tal_topk2=1,
        )
        self.updates = 0
        self.total = 1.0
        self.o2m = initial_o2m
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        self.final_o2m = final_o2m

    def __call__(
        self,
        preds: dict[str, dict[str, torch.Tensor | list[torch.Tensor]]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one2many = preds["one2many"]
        one2one = preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return (
            loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o,
            loss_one2one[1],
        )

    def update(self) -> None:
        """Advance the o2m/o2o decay schedule by one step."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0.0)

    def decay(self, x: int) -> float:
        """Exponential decay from ``o2m_copy`` toward ``final_o2m``.

        Mirrors ultralytics' E2ELoss.decay schedule.
        """
        # Empirical ultralytics schedule: o2m(t) = final + (init - final) * exp(-x/300).
        return self.final_o2m + (self.o2m_copy - self.final_o2m) * float(
            torch.exp(torch.tensor(-x / 300.0))
        )
