"""CLI entry: YAML-driven experiment scheduler.

Each experiment runs in a subprocess so a crash is contained. Stdout/stderr
captured to ``<output-dir>/scheduler/<exp_name>.log``. On any exception:
append to ``errors.jsonl`` with config + traceback + timestamp. On success:
append to ``results.jsonl``.

Retry mode: ``--retry-failed <errors.jsonl>`` auto-generates a YAML from
failed experiments and reruns them.

Example YAML::

    scheduler:
      max_parallel: 1
      seed: 42

    experiments:
      - name: yolov5s_baseline
        type: train
        model: yolov5s
        dataset: data/Crack
        epochs: 100
        batch_size: 32
        output_dir: output/yolov5s_baseline
        seed: 42
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

from loguru import logger
import yaml


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _exp_to_cmd(exp: dict[str, Any]) -> list[str]:
    """Translate one experiment dict to a ``python -m scripts.<type>`` command."""
    exp_type = exp.get("type", "train")
    if exp_type == "train":
        cmd = [
            sys.executable,
            "-m",
            "scripts.train",
            "--model",
            str(exp["model"]),
            "--dataset",
            str(exp["dataset"]),
            "--output-dir",
            str(exp["output_dir"]),
            "--epochs",
            str(exp.get("epochs", 100)),
            "--batch-size",
            str(exp.get("batch_size", 16)),
            "--lr",
            str(exp.get("lr", 1e-3)),
            "--input-size",
            str(exp.get("input_size", 640)),
            "--device",
            str(exp.get("device", "cuda")),
            "--seed",
            str(exp.get("seed", 42)),
            "--num-workers",
            str(exp.get("num_workers", 0)),
        ]
        if exp.get("cross_val"):
            cmd.extend(["--cross-val", "--n-folds", str(exp.get("n_folds", 5))])
            if "val_fraction" in exp:
                cmd.extend(["--val-fraction", str(exp["val_fraction"])])
        if exp.get("no_amp"):
            cmd.append("--no-amp")
        if exp.get("pretrained"):
            cmd.append("--pretrained")
        return cmd
    if exp_type == "test":
        return [
            sys.executable,
            "-m",
            "scripts.test",
            "--model",
            str(exp["model"]),
            "--weights",
            str(exp["weights"]),
            "--dataset",
            str(exp["dataset"]),
            "--split",
            str(exp.get("split", "test")),
            "--output-dir",
            str(exp["output_dir"]),
            "--batch-size",
            str(exp.get("batch_size", 16)),
            "--input-size",
            str(exp.get("input_size", 640)),
            "--device",
            str(exp.get("device", "cuda")),
            "--num-workers",
            str(exp.get("num_workers", 0)),
        ]
    raise ValueError(f"unknown experiment type: {exp_type!r}")


def _run_one(
    exp: dict[str, Any],
    scheduler_dir: Path,
    errors_path: Path,
    results_path: Path,
) -> int:
    name = exp.get("name", "unnamed")
    log_path = scheduler_dir / f"{name}.log"
    cmd = _exp_to_cmd(exp)
    logger.info(f"running experiment '{name}': {' '.join(cmd)}")
    # Per-experiment env overrides (e.g. CUDA_VISIBLE_DEVICES) merged over the
    # current process env so GPU pinning works without shell wrappers.
    env = os.environ.copy()
    env_overrides = exp.get("env")
    if isinstance(env_overrides, dict):
        for k, v in env_overrides.items():
            env[str(k)] = str(v)
    start = datetime.datetime.now()
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
                env=env,
            )
        elapsed = (datetime.datetime.now() - start).total_seconds()
        if proc.returncode != 0:
            record = {
                "exp_name": name,
                "type": exp.get("type", "train"),
                "config": exp,
                "exit_code": proc.returncode,
                "log_path": str(log_path),
                "traceback": None,
                "elapsed_sec": elapsed,
                "timestamp": _now_iso(),
            }
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.error(
                f"experiment '{name}' failed (exit_code={proc.returncode}), see {log_path}"
            )
            return proc.returncode
        record = {
            "exp_name": name,
            "type": exp.get("type", "train"),
            "status": "ok",
            "exit_code": 0,
            "output_dir": str(exp.get("output_dir", "")),
            "elapsed_sec": elapsed,
            "timestamp": _now_iso(),
        }
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"experiment '{name}' succeeded in {elapsed:.1f}s")
        return 0
    except Exception:
        elapsed = (datetime.datetime.now() - start).total_seconds()
        tb = traceback.format_exc()
        record = {
            "exp_name": name,
            "type": exp.get("type", "train"),
            "config": exp,
            "exit_code": -1,
            "log_path": str(log_path),
            "traceback": tb,
            "elapsed_sec": elapsed,
            "timestamp": _now_iso(),
        }
        with errors_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.error(f"experiment '{name}' raised an exception:\n{tb}")
        return -1


def run_from_yaml(config_path: Path, output_dir: Path) -> int:
    """Run all experiments from a YAML config."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiments = cfg.get("experiments", [])
    if not experiments:
        logger.warning("no experiments in YAML")
        return 0

    scheduler_cfg = cfg.get("scheduler", {}) or {}
    max_parallel = int(scheduler_cfg.get("max_parallel", 1))
    if max_parallel < 1:
        max_parallel = 1

    scheduler_dir = output_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    errors_path = scheduler_dir / "errors.jsonl"
    results_path = scheduler_dir / "results.jsonl"

    # Reset log files for a fresh run (caller can preserve old ones manually).
    if not errors_path.exists():
        errors_path.touch()
    if not results_path.exists():
        results_path.touch()

    exit_codes: list[int] = []
    if max_parallel == 1:
        for exp in experiments:
            exit_codes.append(_run_one(exp, scheduler_dir, errors_path, results_path))
    else:
        # Parallel execution via a ThreadPoolExecutor. Each _run_one spawns a
        # subprocess (subprocess.run) and waits on it, so the GIL is released
        # while we wait — true parallelism is achieved by the OS scheduling
        # the subprocesses. GPU pinning is via per-experiment env overrides
        # (CUDA_VISIBLE_DEVICES) in the YAML.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [
                pool.submit(_run_one, exp, scheduler_dir, errors_path, results_path)
                for exp in experiments
            ]
            for f in futures:
                exit_codes.append(f.result())

    n_ok = sum(1 for c in exit_codes if c == 0)
    n_fail = len(exit_codes) - n_ok
    logger.info(f"scheduler done: {n_ok} ok, {n_fail} failed")
    return 0 if n_fail == 0 else 1


def retry_from_errors(errors_path: Path, output_dir: Path) -> int:
    """Re-run experiments listed in an errors.jsonl file."""
    if not errors_path.exists():
        logger.error(f"errors file not found: {errors_path}")
        return 1
    experiments: list[dict[str, Any]] = []
    for line in errors_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        cfg = record.get("config")
        if cfg:
            experiments.append(cfg)
    if not experiments:
        logger.info("no failed experiments to retry")
        return 0
    logger.info(f"retrying {len(experiments)} failed experiment(s)")
    # Back up the old errors file before re-running.
    backup = errors_path.with_suffix(".bak.jsonl")
    errors_path.rename(backup)
    retry_yaml = output_dir / "scheduler" / "retry_from_errors.yaml"
    retry_yaml.parent.mkdir(parents=True, exist_ok=True)
    retry_yaml.write_text(
        yaml.safe_dump({"scheduler": {"max_parallel": 1}, "experiments": experiments}),
        encoding="utf-8",
    )
    return run_from_yaml(retry_yaml, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="YAML-driven experiment scheduler.")
    parser.add_argument("--config", type=Path, help="YAML config with experiments list")
    parser.add_argument(
        "--retry-failed",
        type=Path,
        default=None,
        help="errors.jsonl to retry (overrides --config)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.retry_failed is not None:
        code = retry_from_errors(args.retry_failed, args.output_dir)
    elif args.config is not None:
        code = run_from_yaml(args.config, args.output_dir)
    else:
        parser.error("either --config or --retry-failed is required")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
