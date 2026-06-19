"""Detection heads and bbox helpers.

Includes:
- Anchor-based heads (YOLOv5/v7): :class:`DetectAnchorBased`, :class:`IDetect`,
  :class:`IAuxDetect`.
- Anchor-free heads (YOLOv8/v10): :class:`DetectAnchorFree`, :class:`v10Detect`.
- :class:`DFL` (Distribution Focal Loss integral module).
- Helpers: :func:`make_anchors`, :func:`dist2bbox`, :func:`bbox2dist`.

Ported from:
- ``deps/yolov5/models/yolo.py`` (Detect)
- ``deps/yolov7/models/yolo.py`` (Detect, IDetect, IAuxDetect)
- ``deps/ultralytics/ultralytics/nn/modules/head.py`` (Detect, v10Detect)
- ``deps/ultralytics/ultralytics/nn/modules/block.py`` (DFL)
- ``deps/ultralytics/ultralytics/utils/tal.py`` (make_anchors, dist2bbox,
  bbox2dist).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from cracks_yolo.ops.conv import Conv
from cracks_yolo.ops.conv import DWConv
from cracks_yolo.ops.implicit import ImplicitA
from cracks_yolo.ops.implicit import ImplicitM

# ============================================================================
# Anchor-free helpers
# ============================================================================


def make_anchors(
    feats: list[torch.Tensor] | torch.Tensor,
    strides: torch.Tensor,
    grid_cell_offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate anchor points and stride tensors from feature maps.

    Args:
        feats: List of feature maps ``(B, C, H, W)`` per detection level.
        strides: Per-level stride tensor.
        grid_cell_offset: Offset added to grid coordinates (default 0.5).

    Returns:
        Tuple of (anchor_points ``(N_total, 2)``, stride_tensor ``(N_total, 1)``).
    """
    anchor_points: list[torch.Tensor] = []
    stride_tensor: list[torch.Tensor] = []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):
        stride = strides[i]
        if isinstance(feats, list):
            h, w = feats[i].shape[2:]
        else:
            h, w = int(feats[i][0]), int(feats[i][1])
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))  # type: ignore[call-overload]
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    xywh: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """Transform distance (lt, rb) to box (xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(
    anchor_points: torch.Tensor,
    bbox: torch.Tensor,
    reg_max: int | None = None,
) -> torch.Tensor:
    """Transform bbox (xyxy) to distance (lt, rb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)
    return dist


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss.

    A fixed (non-learnable) 1x1 conv whose weights are ``arange(c1)``. Applies
    softmax over the ``c1`` distribution dimension and computes a weighted sum
    to recover a continuous regression offset.

    Args:
        c1: Number of distribution bins (``reg_max``).

    Reference: Li et al., "Generalized Focal Loss" (https://arxiv.org/abs/2006.04388).
    """

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# ============================================================================
# Anchor-based heads (YOLOv5 / YOLOv7)
# ============================================================================


class DetectAnchorBased(nn.Module):
    """YOLOv5/v7 anchor-based detection head.

    Per-level 1x1 Conv produces ``na * (nc + 5)`` channels, reshaped to
    ``(B, na, ny, nx, nc+5)``. Inference applies sigmoid + grid + anchor
    decoding; training returns the raw reshaped tensors for loss computation.

    Args:
        nc: Number of classes.
        anchors: Tuple of anchor sizes per level (flattened pairs). Length must
            be ``nl * na * 2``.
        ch: Tuple of input channel counts per level.
        inplace: If True, use in-place slice assignment during decode.

    Attributes:
        stride: Per-level stride tensor. Set by the parent model after
            instantiation via a dummy forward pass.
    """

    stride: torch.Tensor | None
    anchors: torch.Tensor
    m: nn.ModuleList

    def __init__(
        self,
        nc: int = 80,
        anchors: tuple[tuple[int, ...], ...] = (),
        ch: tuple[int, ...] = (),
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer(
            "anchors", torch.tensor(anchors, dtype=torch.float32).view(self.nl, -1, 2)
        )
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(
        self, x: list[torch.Tensor]
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        z: list[torch.Tensor] = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                assert self.stride is not None
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)  # type: ignore[no-untyped-call]
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(
        self, nx: int = 20, ny: int = 20, i: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = (1, self.na, ny, nx, 2)
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)  # type: ignore[index]
        return grid, anchor_grid


class IDetect(nn.Module):
    """YOLOv7 IDetect: anchor-based Detect with ImplicitA (additive) and
    ImplicitM (multiplicative) biases around the output conv.

    Args:
        nc: Number of classes.
        anchors: Anchor sizes per level.
        ch: Input channels per level.
    """

    stride: torch.Tensor | None
    anchors: torch.Tensor
    anchor_grid: torch.Tensor
    m: nn.ModuleList
    ia: nn.ModuleList
    im: nn.ModuleList

    def __init__(
        self,
        nc: int = 80,
        anchors: tuple[tuple[int, ...], ...] = (),
        ch: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1) for _ in range(self.nl)]
        a = torch.tensor(anchors, dtype=torch.float32).view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(
        self, x: list[torch.Tensor]
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        z: list[torch.Tensor] = []
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                assert self.stride is not None
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        yv, xv = torch.meshgrid(
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2))

    def fuse(self) -> None:
        """Fuse ImplicitA into the conv bias and ImplicitM into the conv weight.

        After fusion, ``self.ia`` and ``self.im`` are replaced with
        :class:`nn.Identity` so the forward no longer double-applies them.
        """
        for i in range(len(self.m)):
            conv_i = self.m[i]
            ia_i = self.ia[i]
            im_i = self.im[i]
            assert isinstance(conv_i, nn.Conv2d)
            assert isinstance(ia_i, ImplicitA)
            assert isinstance(im_i, ImplicitM)
            c1, c2, _, _ = conv_i.weight.shape
            c1_, c2_, _, _ = ia_i.implicit.shape
            assert conv_i.bias is not None
            conv_i.bias.data += torch.matmul(
                conv_i.weight.reshape(c1, c2),
                ia_i.implicit.reshape(c2_, c1_),
            ).squeeze(1)
        for i in range(len(self.m)):
            conv_i = self.m[i]
            im_i = self.im[i]
            assert isinstance(conv_i, nn.Conv2d)
            assert isinstance(im_i, ImplicitM)
            c1, c2, _, _ = im_i.implicit.shape
            assert conv_i.bias is not None
            conv_i.bias.data *= im_i.implicit.reshape(c2)
            conv_i.weight.data *= im_i.implicit.transpose(0, 1)
        # Replace Implicit modules with Identity so forward stops re-applying
        # the absorbed transforms.
        self.ia = nn.ModuleList(nn.Identity() for _ in self.m)
        self.im = nn.ModuleList(nn.Identity() for _ in self.m)


class IAuxDetect(nn.Module):
    """YOLOv7 IAuxDetect: IDetect with an auxiliary head (extra ``m2`` convs).

    Used in v7's lead/head distillation. The auxiliary head consumes the
    second half of ``ch``; the main head consumes the first half.

    Args:
        nc: Number of classes.
        anchors: Anchor sizes per level.
        ch: Input channels per level. Length must be ``2 * nl`` (main + aux).
    """

    stride: torch.Tensor | None
    anchors: torch.Tensor
    anchor_grid: torch.Tensor
    m: nn.ModuleList
    m2: nn.ModuleList
    ia: nn.ModuleList
    im: nn.ModuleList

    def __init__(
        self,
        nc: int = 80,
        anchors: tuple[tuple[int, ...], ...] = (),
        ch: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1) for _ in range(self.nl)]
        a = torch.tensor(anchors, dtype=torch.float32).view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[: self.nl])
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl :])
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[: self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[: self.nl])

    def forward(
        self, x: list[torch.Tensor]
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        z: list[torch.Tensor] = []
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x[i + self.nl] = self.m2[i](x[i + self.nl])
            x[i + self.nl] = (
                x[i + self.nl]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            if not self.training:
                assert self.stride is not None
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x[: self.nl])

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        yv, xv = torch.meshgrid(
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2))


# ============================================================================
# Anchor-free heads (YOLOv8 / YOLOv10)
# ============================================================================


class DetectAnchorFree(nn.Module):
    """YOLOv8 anchor-free detection head.

    Per-level separate box branch (``cv2``: 3-conv stack producing
    ``4 * reg_max`` channels) and cls branch (``cv3``: light DWConv-based
    stack producing ``nc`` channels). No objectness branch.

    Training forward returns a dict ``{boxes, scores, feats}`` for loss
    computation. Inference forward decodes via :func:`dist2bbox` and returns
    a tensor of shape ``(B, N_total, 4 + nc)`` (xyxy + class scores).

    Args:
        nc: Number of classes.
        reg_max: DFL bins. Default 16.
        ch: Tuple of input channel counts per level.

    Attributes:
        stride: Per-level stride tensor. Set by the parent model after build.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        ch: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        # Inference-time caches (set on first inference forward).
        self.shape: tuple[int, ...] = ()
        self.anchors: torch.Tensor = torch.empty(0)
        self.strides: torch.Tensor = torch.empty(0)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]] | torch.Tensor:
        bs = x[0].shape[0]
        boxes = torch.cat(
            [self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1
        )
        scores = torch.cat(
            [self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        if self.training:
            return {"boxes": boxes, "scores": scores, "feats": x}
        return self._inference({"boxes": boxes, "scores": scores, "feats": x})

    def _inference(self, x: dict[str, torch.Tensor | list[torch.Tensor]]) -> torch.Tensor:
        dbox = self._get_decode_boxes(x)
        scores = x["scores"]
        assert isinstance(scores, torch.Tensor)
        return torch.cat((dbox, scores.sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor | list[torch.Tensor]]) -> torch.Tensor:
        feats = x["feats"]
        assert isinstance(feats, list)
        shape = feats[0].shape
        if self.shape != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1) for a in make_anchors(feats, self.stride, 0.5)
            )
            self.shape = shape
        boxes = x["boxes"]
        assert isinstance(boxes, torch.Tensor)
        return self.decode_bboxes(self.dfl(boxes), self.anchors.unsqueeze(0)) * self.strides

    def decode_bboxes(
        self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True
    ) -> torch.Tensor:
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)

    def bias_init(self) -> None:
        """Initialize the box/cls head biases. Requires ``stride`` to be set."""
        assert self.stride is not None
        for i, (a, b) in enumerate(zip(self.cv2, self.cv3, strict=True)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class v10Detect(DetectAnchorFree):  # noqa: N801
    """YOLOv10 detection head with dual assignment (one2many + one2one).

    The one2many head supervises training (each GT assigned to many anchors).
    The one2one head enables NMS-free inference (each GT assigned to exactly
    one anchor). Call :meth:`fuse` before deployment to drop the one2many
    head.

    Args:
        nc: Number of classes.
        ch: Tuple of input channel counts per level.
    """

    end2end: bool = True

    def __init__(self, nc: int = 80, ch: tuple[int, ...] = ()) -> None:
        super().__init__(nc=nc, ch=ch)
        c3 = max(ch[0], min(nc, 100))
        # Override cv3 with v10's light cls head (pointwise + DWConv).
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        # Dual one2one heads (deepcopy of the one2many heads).
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(  # type: ignore[override]
        self, x: list[torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor | list[torch.Tensor]]] | torch.Tensor:
        bs = x[0].shape[0]
        one2many = self._forward_head(x, self.cv2, self.cv3, bs)
        x_detach = [xi.detach() for xi in x]
        one2one = self._forward_head(x_detach, self.one2one_cv2, self.one2one_cv3, bs)
        if self.training:
            return {"one2many": one2many, "one2one": one2one}
        return self._inference(one2one)

    def _forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.ModuleList,
        cls_head: nn.ModuleList,
        bs: int,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": x}

    def bias_init(self) -> None:
        """Initialize both one2many and one2one head biases."""
        assert self.stride is not None
        for i, (a, b) in enumerate(zip(self.cv2, self.cv3, strict=True)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        for i, (a, b) in enumerate(zip(self.one2one_cv2, self.one2one_cv3, strict=True)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def fuse(self) -> None:
        """Drop the one2many head for inference optimization."""
        self.cv2 = None  # type: ignore[assignment]
        self.cv3 = None  # type: ignore[assignment]
