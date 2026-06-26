"""Model efficiency analysis — params, MACs, inference latency, peak VRAM."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn as nn


@dataclass
class ModelAnalysisReport:
    """Structured result of analyze_model."""

    model_name: str = ""
    n_parameters: int = 0
    n_trainable_parameters: int = 0
    macs: float = 0.0  # multiply-accumulate operations
    flops: float = 0.0  # 2 * macs (convention)
    gflops: float = 0.0  # flops / 1e9
    fps_mean: float = 0.0  # 1000 / latency_mean_ms
    fps_p50: float = 0.0
    fps_p95: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0
    peak_vram_bytes: int = 0  # 0 if not on CUDA
    input_size: int = 0
    device: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _count_macs(model: nn.Module, input_size: int) -> float:
    """Count MACs (multiply-accumulate ops) via FlopCounterMode.

    Uses PyTorch's built-in ``FlopCounterMode`` which hooks at the ATen-op
    level (``aten::conv2d``, ``aten::linear``, etc.) — this correctly counts
    SAConv2d's internal convolutions which bypass ``nn.Conv2d.forward()``
    via ``_conv_forward()`` and are invisible to thop's per-module hooks.

    Returns MACs; GFLOPs = 2 × MACs / 1e9.
    """
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    x = torch.zeros((1, 3, input_size, input_size), device=dev)
    macs = 0.0
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fcm:
            _ = model(x)
        # FlopCounterMode returns total FLOPs; MACs = FLOPs / 2.
        macs = float(fcm.get_total_flops()) / 2.0
    except Exception:
        try:
            from thop import profile

            macs, _ = profile(model, inputs=(x,), verbose=False)
            macs = float(macs)
        except Exception:
            macs = 0.0
    finally:
        if was_training:
            model.train()
    return macs


def analyze_model(
    model: nn.Module,
    input_size: int = 640,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ModelAnalysisReport:
    """Compute params, MACs, latency p50/p95, peak VRAM."""
    report = ModelAnalysisReport()
    report.model_name = type(model).__name__
    report.input_size = input_size
    report.device = device

    report.n_parameters = sum(p.numel() for p in model.parameters())
    report.n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report.macs = _count_macs(model, input_size)
    report.flops = 2.0 * report.macs
    report.gflops = report.flops / 1e9

    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.zeros((1, 3, input_size, input_size), device=dev)

    # Warmup — enough runs to saturate GPU clocks.
    warmup = 50 if dev.type == "cuda" else 5
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # Latency — use CUDA events for precise GPU timing.
    latencies_ms: list[float] = []
    runs = 100 if dev.type == "cuda" else 30
    with torch.no_grad():
        for _ in range(runs):
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()
                latencies_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(x)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    if latencies_ms:
        latencies_sorted = sorted(latencies_ms)
        report.latency_p50_ms = _percentile(latencies_sorted, 50)
        report.latency_p95_ms = _percentile(latencies_sorted, 95)
        report.latency_mean_ms = statistics.fmean(latencies_ms)
        report.fps_mean = 1000.0 / report.latency_mean_ms if report.latency_mean_ms > 0 else 0
        report.fps_p50 = 1000.0 / report.latency_p50_ms if report.latency_p50_ms > 0 else 0
        report.fps_p95 = 1000.0 / report.latency_p95_ms if report.latency_p95_ms > 0 else 0

    if dev.type == "cuda":
        report.peak_vram_bytes = int(torch.cuda.max_memory_allocated(dev))

    return report


def _percentile(sorted_xs: list[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_xs) - 1)
    if f == c:
        return sorted_xs[f]
    return sorted_xs[f] + (sorted_xs[c] - sorted_xs[f]) * (k - f)


def save_model_analysis(report: ModelAnalysisReport, out_dir: Path) -> None:
    """Write ``model_analysis.json`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_analysis.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )


__all__ = ["ModelAnalysisReport", "analyze_model", "save_model_analysis"]
