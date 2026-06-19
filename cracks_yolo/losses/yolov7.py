"""YOLOv7 ``ComputeLossOTA`` — SimOTA-style assignment + CIoU/BCE losses.

Decoupled from upstream's ``model.hyp`` / ``model.gr`` / ``model.model[-1]``
coupling. Builds positive samples via 3-anchor heuristic + dynamic top-k
matching using a cost = ``cls_loss + 3 * iou_loss``.

Ported from ``deps/yolov7/utils/loss.py`` (``ComputeLossOTA``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from cracks_yolo.losses._common import FocalLoss
from cracks_yolo.losses._common import bbox_iou_v7
from cracks_yolo.losses._common import smooth_BCE
from cracks_yolo.losses._common import xywh2xyxy


class ComputeLossOTA:
    """YOLOv7 OTA training loss.

    Loss = ``hyp["box"] * (1 - CIoU) + hyp["obj"] * BCEobj + hyp["cls"] * BCEcls``.

    Args:
        nc: Number of classes.
        anchors: ``(nl, na, 2)`` anchor sizes (stride units).
        stride: ``(nl,)`` per-level stride tensor.
        hyp: Hyperparameter dict (see :class:`cracks_yolo.losses.yolov5.ComputeLoss`).
        device: Device to place criterion tensors on.
        autobalance: If True, enable per-level obj balance auto-tuning.
        gr: Objectness-IoU gain ratio (default 1.0).
    """

    def __init__(
        self,
        nc: int,
        anchors: torch.Tensor,
        stride: torch.Tensor,
        hyp: dict[str, float],
        device: torch.device,
        autobalance: bool = False,
        gr: float = 1.0,
    ) -> None:
        self.device = device
        self.hyp = hyp
        self.autobalance = autobalance
        self.gr = gr

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
        self.balance: list[float] = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(stride.tolist()).index(16.0) if autobalance else 0
        self.BCEcls: nn.Module = bcecls
        self.BCEobj: nn.Module = bceobj

    def __call__(
        self,
        p: list[torch.Tensor],
        targets: torch.Tensor,
        imgs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total loss.

        Args:
            p: Per-level predictions, each ``(B, na, ny, nx, nc+5)``.
            targets: ``(N, 6)`` — ``(image_idx, class, x, y, w, h)`` (normalized).
            imgs: ``(B, C, H, W)`` image batch (used to determine image size
                for OTA assignment).

        Returns:
            ``(total_loss * bs, cat((lbox, lobj, lcls, loss)).detach())``.
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
        device = pred_dev
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        bs_t, as_t, gjs_t, gis_t, targets_t, anchors_t = self.build_targets(p, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p]

        for i, pi in enumerate(p):
            b, a, gj, gi = bs_t[i], as_t[i], gjs_t[i], gis_t[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]

                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors_t[i]
                pbox = torch.cat((pxy, pwh), 1)
                selected_tbox = targets_t[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] = selected_tbox[:, :2] - grid
                iou = bbox_iou_v7(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                lbox = lbox + (1.0 - iou).mean()

                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(
                    tobj.dtype
                )

                selected_tcls = targets_t[i][:, 1].long()
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), selected_tcls] = self.cp
                    lcls = lcls + self.BCEcls(ps[:, 5:], t)

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

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(
        self,
        p: list[torch.Tensor],
        targets: torch.Tensor,
        imgs: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Run 3-positive heuristic + SimOTA dynamic-k matching.

        Returns six lists (one entry per level) of:
        ``(batch_idx, anchor_idx, gridy, gridx, targets_subset, anchors_subset)``.
        """
        indices, anch = self.find_3_positive(p, targets)
        device = targets.device
        matching_bs: list[list[torch.Tensor]] = [[] for _ in p]
        matching_as: list[list[torch.Tensor]] = [[] for _ in p]
        matching_gjs: list[list[torch.Tensor]] = [[] for _ in p]
        matching_gis: list[list[torch.Tensor]] = [[] for _ in p]
        matching_targets: list[list[torch.Tensor]] = [[] for _ in p]
        matching_anchs: list[list[torch.Tensor]] = [[] for _ in p]

        nl = len(p)

        for batch_idx in range(p[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys: list[torch.Tensor] = []
            p_cls: list[torch.Tensor] = []
            p_obj: list[torch.Tensor] = []
            from_which_layer: list[torch.Tensor] = []
            all_b: list[torch.Tensor] = []
            all_a: list[torch.Tensor] = []
            all_gj: list[torch.Tensor] = []
            all_gi: list[torch.Tensor] = []
            all_anch: list[torch.Tensor] = []

            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                idx = b == batch_idx
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2 - 0.5 + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys_cat = torch.cat(pxyxys, dim=0)
            if pxyxys_cat.shape[0] == 0:
                continue
            p_obj_cat = torch.cat(p_obj, dim=0)
            p_cls_cat = torch.cat(p_cls, dim=0)
            from_which_layer_cat = torch.cat(from_which_layer, dim=0)
            all_b_cat = torch.cat(all_b, dim=0)
            all_a_cat = torch.cat(all_a, dim=0)
            all_gj_cat = torch.cat(all_gj, dim=0)
            all_gi_cat = torch.cat(all_gi, dim=0)
            all_anch_cat = torch.cat(all_anch, dim=0)

            pair_wise_iou = torchvision_box_iou(txyxy, pxyxys_cat)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            num_gt = this_target.shape[0]
            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys_cat.shape[0], 1)
            )
            cls_preds_ = (
                p_cls_cat.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj_cat.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)),
                gt_cls_per_image,
                reduction="none",
            ).sum(-1)

            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss

            matching_matrix = torch.zeros_like(cost, device=device)
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=int(dynamic_ks[gt_idx].item()), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer_cat = from_which_layer_cat[fg_mask_inboxes]
            all_b_cat = all_b_cat[fg_mask_inboxes]
            all_a_cat = all_a_cat[fg_mask_inboxes]
            all_gj_cat = all_gj_cat[fg_mask_inboxes]
            all_gi_cat = all_gi_cat[fg_mask_inboxes]
            all_anch_cat = all_anch_cat[fg_mask_inboxes]
            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer_cat == i
                matching_bs[i].append(all_b_cat[layer_idx])
                matching_as[i].append(all_a_cat[layer_idx])
                matching_gjs[i].append(all_gj_cat[layer_idx])
                matching_gis[i].append(all_gi_cat[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch_cat[layer_idx])

        out_bs: list[torch.Tensor] = []
        out_as: list[torch.Tensor] = []
        out_gjs: list[torch.Tensor] = []
        out_gis: list[torch.Tensor] = []
        out_targets: list[torch.Tensor] = []
        out_anchs: list[torch.Tensor] = []
        for i in range(nl):
            if matching_targets[i]:
                out_bs.append(torch.cat(matching_bs[i], dim=0))
                out_as.append(torch.cat(matching_as[i], dim=0))
                out_gjs.append(torch.cat(matching_gjs[i], dim=0))
                out_gis.append(torch.cat(matching_gis[i], dim=0))
                out_targets.append(torch.cat(matching_targets[i], dim=0))
                out_anchs.append(torch.cat(matching_anchs[i], dim=0))
            else:
                out_bs.append(torch.tensor([], device=device, dtype=torch.int64))
                out_as.append(torch.tensor([], device=device, dtype=torch.int64))
                out_gjs.append(torch.tensor([], device=device, dtype=torch.int64))
                out_gis.append(torch.tensor([], device=device, dtype=torch.int64))
                out_targets.append(torch.tensor([], device=device, dtype=torch.int64))
                out_anchs.append(torch.tensor([], device=device, dtype=torch.int64))

        return (
            out_bs,
            out_as,
            out_gjs,
            out_gis,
            out_targets,
            out_anchs,
        )

    def find_3_positive(
        self,
        p: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        list[torch.Tensor],
    ]:
        """3-anchor heuristic: anchor_t wh-ratio filter + 5-tuple grid offsets."""
        na, nt = self.na, targets.shape[0]
        indices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        anch: list[torch.Tensor] = []
        gain = torch.ones(7, device=targets.device).long()
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

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
                device=targets.device,
            ).float()
            * g
        )

        for i in range(self.nl):
            anchors_i = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors_i[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < self.hyp["anchor_t"]
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                left, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, left, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = torch.zeros(1, device=targets.device)

            b, _c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            anch.append(anchors_i[a])

        return indices, anch


def torchvision_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` xyxy boxes."""
    import torchvision

    out: torch.Tensor = torchvision.ops.box_iou(box1, box2)
    return out
