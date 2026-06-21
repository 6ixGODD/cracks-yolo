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
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0
    peak_vram_bytes: int = 0  # 0 if not on CUDA
    input_size: int = 0
    device: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _count_macs(model: nn.Module, input_size: int) -> float:
    """Best-effort MAC count via thop (primary) or fvcore (fallback).

    thop registers per-layer forward hooks so it counts conv/linear MACs
    without tracing the Detect head's non-tensor control flow — robust for
    YOLO + torchvision. fvcore's FlopCountAnalysis traces the forward and
    aborts on custom Detect heads (returns 0); kept only as a fallback.

    Returns MACs (multiply-accumulate ops). ``gflops = 2 * macs`` upstream.
    """
    was_training = model.training
    model.eval()
    # Place the dummy input on the same device as the model parameters
    # (analyze_model may call us before/after .to(device); thop runs a real
    # forward so input and model must agree).
    dev = next(model.parameters()).device
    x = torch.zeros((1, 3, input_size, input_size), device=dev)
    macs = 0.0
    try:
        from thop import profile

        # thop returns (macs, params); wrap to silence its verbose prints.
        macs, _ = profile(model, inputs=(x,), verbose=False)
        macs = float(macs)
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis

            flops = (
                FlopCountAnalysis(model, x)
                .unsupported_ops_warnings(False)
                .uncalled_modules_warnings(False)
            )
            macs = float(flops.total()) / 2.0  # flops = 2 * macs convention
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
    n_warmup: int = 5,
    n_runs: int = 50,
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

    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.zeros((1, 3, input_size, input_size), device=dev)

    # Warmup.
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # Latency.
    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
    if latencies_ms:
        latencies_sorted = sorted(latencies_ms)
        report.latency_p50_ms = _percentile(latencies_sorted, 50)
        report.latency_p95_ms = _percentile(latencies_sorted, 95)
        report.latency_mean_ms = statistics.fmean(latencies_ms)

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
