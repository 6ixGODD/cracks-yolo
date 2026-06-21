"""YOLOv5 ``ComputeLoss`` — CIoU box + BCE obj + BCE cls.

Decoupled from the upstream ``model.hyp`` / ``model.model[-1]`` coupling: the
config and detect-head metadata are passed explicitly so the loss can be
constructed standalone.

Ported from ``deps/yolov5/utils/loss.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cracks_yolo.losses._common import FocalLoss
from cracks_yolo.losses._common import bbox_iou
from cracks_yolo.losses._common import smooth_BCE


class ComputeLoss:
    """YOLOv5 training loss.

    Loss = ``hyp["box"] * (1 - CIoU) + hyp["obj"] * BCEobj + hyp["cls"] * BCEcls``.

    Anchor matching: per-level anchor_t=4.0 wh-ratio filter + 5-tuple grid
    offsets (j, k, l, m at bias 0.5). Objectness targets are IoU-aware.

    Args:
        nc: Number of classes.
        anchors: ``(nl, na, 2)`` tensor of anchor sizes (in stride units).
        stride: ``(nl,)`` per-level stride tensor.
        hyp: Hyperparameter dict with keys ``box``, ``obj``, ``cls``,
            ``cls_pw``, ``obj_pw``, ``fl_gamma``, ``anchor_t``,
            ``label_smoothing`` (optional, default 0).
        device: Device to place criterion tensors on.
        autobalance: If True, enable per-level obj balance auto-tuning.
    """

    sort_obj_iou: bool = False

    def __init__(
        self,
        nc: int,
        anchors: torch.Tensor,
        stride: torch.Tensor,
        hyp: dict[str, float],
        device: torch.device,
        autobalance: bool = False,
    ) -> None:
        self.device = device
        self.hyp = hyp
        self.autobalance = autobalance

        bcecls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp["cls_pw"]], device=device))
        bceobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp["obj_pw"]], device=device))

        self.cp, self.cn = smooth_BCE(eps=hyp.get("label_smoothing", 0.0))

        g = hyp.get("fl_gamma", 0.0)
        if g > 0:
            bcecls = FocalLoss(bcecls, g)  # type: ignore[assignment]
            bceobj = FocalLoss(bceobj, g)  # type: ignore[assignment]

        self.nl = anchors.shape[0]
        self.na = anchors.shape[1]
        self.nc = nc
        self.anchors = anchors
        self.stride = stride
        self.balance: list[float] = {3: [4.0, 1.0, 0.4]}.get(  # P3-P7 fallback
            self.nl, [4.0, 1.0, 0.25, 0.06, 0.02]
        )
        self.ssi = list(stride.tolist()).index(16.0) if autobalance else 0
        self.BCEcls: nn.Module = bcecls
        self.BCEobj: nn.Module = bceobj
        self.gr = 1.0

    def __call__(
        self,
        p: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total loss.

        Args:
            p: Per-level predictions, each ``(B, na, ny, nx, nc+5)``.
            targets: ``(N, 6)`` — ``(image_idx, class, x, y, w, h)`` in
                normalized xywh.

        Returns:
            ``(total_loss * bs, cat((lbox, lobj, lcls)).detach())``.
        """
        # Sync internal tensors to the prediction device (model may have been
        # moved to CUDA after this loss was constructed on CPU).
        pred_dev = p[0].device
        if self.device != pred_dev:
            self.device = pred_dev
            self.anchors = self.anchors.to(pred_dev)
            self.stride = self.stride.to(pred_dev)
            self.BCEcls = self.BCEcls.to(pred_dev)
            self.BCEobj = self.BCEobj.to(pred_dev)
        targets = targets.to(pred_dev)
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

            n = b.shape[0]
            if n := b.shape[0]:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # type: ignore[no-untyped-call]

                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox = lbox + (1.0 - iou).mean()

                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou

                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp
                    lcls = lcls + self.BCEcls(pcls, t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj = lobj + obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox = lbox * self.hyp["box"]
        lobj = lobj * self.hyp["obj"]
        lcls = lcls * self.hyp["cls"]
        bs = tobj.shape[0]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lcls, lobj)).detach()

    def build_targets(
        self,
        p: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        list[torch.Tensor],
    ]:
        """Build per-level (cls, box, indices, anchors) targets from inputs."""
        na, nt = self.na, targets.shape[0]
        tcls: list[torch.Tensor] = []
        tbox: list[torch.Tensor] = []
        indices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        anch: list[torch.Tensor] = []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g = 0.5
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                ],
                device=self.device,
            ).float()
            * g
        )

        for i in range(self.nl):
            anchors_i, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchors_i[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                left, right = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, left, right))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = torch.zeros(1, device=self.device)

            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors_i[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
