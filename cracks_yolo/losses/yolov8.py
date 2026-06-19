"""YOLOv8 detection loss: TaskAlignedAssigner + BboxLoss + v8DetectionLoss.

Decoupled from upstream's ``model.args`` / ``model.model[-1]`` coupling. The
model's detect head, hyperparameter dict, stride tensor, and ``reg_max`` are
passed explicitly.

Ported from:
- ``deps/ultralytics/ultralytics/utils/tal.py`` (TaskAlignedAssigner, make_anchors, dist2bbox, bbox2dist).
- ``deps/ultralytics/ultralytics/utils/loss.py`` (DFLoss, BboxLoss, v8DetectionLoss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from cracks_yolo.losses._common import DFLoss
from cracks_yolo.losses._common import bbox_iou
from cracks_yolo.ops.detect_heads import bbox2dist
from cracks_yolo.ops.detect_heads import dist2bbox
from cracks_yolo.ops.detect_heads import make_anchors


class TaskAlignedAssigner(nn.Module):
    """Task-aligned ground-truth to anchor assignment (v8).

    Metric = ``cls_score^alpha * iou^beta``; pick top-k anchors per GT, then
    resolve overlaps by keeping the highest-IoU assignment.

    Args:
        topk: Number of top candidates per GT.
        num_classes: Number of classes.
        alpha: Classification exponent.
        beta: Localization (IoU) exponent.
        stride: Per-level stride list (used only for tiny-box fallback).
        eps: Numerical stability.
        topk2: Secondary topk for additional filtering.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list[float] | None = None,
        eps: float = 1e-9,
        topk2: int | None = None,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8.0, 16.0, 32.0]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps
        self.bs = 0
        self.n_max_boxes = 0

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute task-aligned assignment.

        Returns ``(target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx)``.
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )
        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        )
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        na = pd_bboxes.shape[-2]
        mask_gt_bool = mask_gt.bool()
        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device
        )
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt_bool] = pd_scores[ind[0], :, ind[1]][mask_gt_bool]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt_bool]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt_bool]
        overlaps[mask_gt_bool] = self.iou_calculation(gt_boxes, pd_boxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(
        self,
        metrics: torch.Tensor,
        topk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs = topk_idxs.masked_fill(~topk_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        count_tensor = count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_targets(
        self,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        target_labels = target_labels.clamp_(0)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )
        target_scores = target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(
        self,
        xy_centers: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """Boolean mask ``(b, n_boxes, n_anchors)`` of anchors inside GTs."""
        from cracks_yolo.losses._common import xywh2xyxy
        from cracks_yolo.losses._common import xyxy2xywh

        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(
                self.stride_val,
                dtype=gt_bboxes_xywh.dtype,
                device=gt_bboxes_xywh.device,
            ),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
            bs, n_boxes, n_anchors, -1
        )
        return bbox_deltas.amin(3).gt_(eps)

    def select_highest_overlaps(
        self,
        mask_pos: torch.Tensor,
        overlaps: torch.Tensor,
        n_max_boxes: int,
        align_metric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(
                mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device
            )
            is_max_overlaps = is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx = topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos = mask_pos * topk_idx
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos


class BboxLoss(nn.Module):
    """Box loss for v8/v10: weighted CIoU + DFL."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.dfl_loss: DFLoss | None = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss is not None:
            assert self.dfl_loss.reg_max is not None
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] = target_ltrb[..., 0::2] / imgsz[1]
            target_ltrb[..., 1::2] = target_ltrb[..., 1::2] / imgsz[0]
            pred_dist_norm = pred_dist * stride
            pred_dist_norm[..., 0::2] = pred_dist_norm[..., 0::2] / imgsz[1]
            pred_dist_norm[..., 1::2] = pred_dist_norm[..., 1::2] / imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_norm[fg_mask], target_ltrb[fg_mask], reduction="none").mean(
                    -1, keepdim=True
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class v8DetectionLoss:  # noqa: N801
    """YOLOv8 detection loss.

    Loss = ``hyp["box"] * box_loss + hyp["cls"] * cls_loss + hyp["dfl"] * dfl_loss``.

    Args:
        nc: Number of classes.
        reg_max: DFL bins (must match the detect head).
        stride: ``(nl,)`` per-level stride tensor.
        hyp: Hyperparameter dict with keys ``box``, ``cls``, ``dfl``.
        device: Device for criterion tensors.
        tal_topk: TaskAlignedAssigner topk.
        tal_topk2: Optional secondary topk.
    """

    def __init__(
        self,
        nc: int,
        reg_max: int,
        stride: torch.Tensor,
        hyp: dict[str, float],
        device: torch.device,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
    ) -> None:
        self.device = device
        self.hyp = hyp
        self.stride = stride
        self.nc = nc
        self.no = nc + reg_max * 4
        self.reg_max = reg_max

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.use_dfl = reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=nc,
            alpha=0.5,
            beta=6.0,
            stride=stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(reg_max).to(device)
        self.proj = torch.arange(reg_max, dtype=torch.float, device=device)

    def preprocess(
        self,
        targets: torch.Tensor,
        batch_size: int,
        scale_tensor: torch.Tensor,
    ) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.unique(return_counts=True)  # type: ignore[no-untyped-call]
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, int(counts.max()), ne - 1, device=self.device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets = offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _sync_device(self, preds: dict[str, torch.Tensor | list[torch.Tensor]]) -> None:
        """Sync internal tensors to the prediction device.

        The model may have been moved to CUDA after this loss was constructed
        on CPU. Internal tensors (stride, BCE) are not nn.Parameters and don't
        move with ``.to()`` — we sync them here on first call.
        """
        feats = preds.get("feats")
        if feats is not None and isinstance(feats, list) and len(feats) > 0:
            pred_dev = feats[0].device
        else:
            boxes = preds.get("boxes")
            assert isinstance(boxes, torch.Tensor)
            pred_dev = boxes.device
        if self.device != pred_dev:
            self.device = pred_dev
            self.stride = self.stride.to(pred_dev)
            self.bce = self.bce.to(pred_dev)
            self.proj = self.proj.to(pred_dev)

    def __call__(
        self,
        preds: dict[str, torch.Tensor | list[torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._sync_device(preds)
        return self.loss(preds, batch)

    def loss(
        self,
        preds: dict[str, torch.Tensor | list[torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._sync_device(preds)
        batch_size = preds["boxes"].shape[0]  # type: ignore[union-attr]
        loss, loss_detach = self._compute(preds, batch)[1:]
        return loss * batch_size, loss_detach

    def _compute(
        self,
        preds: dict[str, torch.Tensor | list[torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        loss = torch.zeros(3, device=self.device)
        boxes = preds["boxes"]
        scores = preds["scores"]
        feats = preds["feats"]
        assert isinstance(boxes, torch.Tensor)
        assert isinstance(scores, torch.Tensor)
        assert isinstance(feats, list)
        pred_distri = boxes.permute(0, 2, 1).contiguous()
        pred_scores = scores.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # type: ignore[no-untyped-call]
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = torch.clamp(target_scores.sum(), min=1.0)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] = loss[0] * self.hyp["box"]
        loss[1] = loss[1] * self.hyp["cls"]
        loss[2] = loss[2] * self.hyp["dfl"]
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


# Late import to avoid circulars.
from cracks_yolo.losses._common import xywh2xyxy  # noqa: E402
