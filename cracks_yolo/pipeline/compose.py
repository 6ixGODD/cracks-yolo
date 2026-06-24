"""YAML-driven experiment scheduler with $include support."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from loguru import logger
import yaml


def run_compose(config: Path, output_dir: Path, max_parallel: int = 1) -> int:
    """Load a compose YAML (with $include), run each experiment sequentially.

    Each included YAML is a single experiment config with fields:
        name, type (train/test), model, dataset, output_dir, epochs, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scheduler_dir = output_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    errors_path = scheduler_dir / "errors.jsonl"
    results_path = scheduler_dir / "results.jsonl"

    # Load + resolve $include
    cfg = _load_config(config)
    experiments = cfg.get("experiments", [])
    if not experiments:
        logger.warning("no experiments in config")
        return 0

    logger.info(f"compose: {len(experiments)} experiments, max_parallel={max_parallel}")

    exit_codes: list[int] = []
    for exp in experiments:
        name = exp.get("name", "unnamed")
        log_path = scheduler_dir / f"{name}.log"

        # Build command
        cmd = _build_cmd(exp)
        logger.info(f"running '{name}': {' '.join(cmd)}")

        # Per-experiment env overrides
        env = os.environ.copy()
        env_overrides = exp.get("env")
        if isinstance(env_overrides, dict):
            for k, v in env_overrides.items():
                env[str(k)] = str(v)

        try:
            with log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=env,
                )
            if proc.returncode != 0:
                _record_error(errors_path, name, exp, proc.returncode, log_path)
                logger.error(f"'{name}' failed (exit {proc.returncode})")
            else:
                _record_success(results_path, name, exp, log_path)
                logger.info(f"'{name}' succeeded")
            exit_codes.append(proc.returncode)
        except Exception as e:
            _record_error(errors_path, name, exp, -1, log_path, str(e))
            logger.error(f"'{name}' crashed: {e}")
            exit_codes.append(1)

    n_ok = sum(1 for c in exit_codes if c == 0)
    n_fail = len(exit_codes) - n_ok
    logger.info(f"compose done: {n_ok} ok, {n_fail} failed")
    return n_fail


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML, resolve $include recursively.

    Three cases:
    1. File has ``$include`` → recurse into each included file.
    2. File has ``experiments`` → explicit experiment list.
    3. File has neither → the file *itself* is a single experiment.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        return {"experiments": []}

    includes = cfg.pop("$include", [])
    if isinstance(includes, str):
        includes = [includes]

    experiments: list[dict[str, Any]] = []
    for inc in includes:
        child = _load_config(config_path.parent / inc)
        experiments.extend(child.get("experiments", []))

    # If cfg has explicit experiments list, use that.
    # Otherwise, if cfg has meaningful content (and no $include), treat itself as one experiment.
    if "experiments" in cfg:
        experiments.extend(cfg.pop("experiments", []))
    elif not includes and any(k not in ("scheduler",) for k in cfg):
        experiments.append(cfg)

    cfg["experiments"] = experiments
    return cfg


def _build_cmd(exp: dict[str, Any]) -> list[str]:
    """Build a ``cy train/test`` or ``cy run`` command."""
    exp_type = exp.get("type", "train")
    cmd = ["cy", exp_type]

    flag_map = {
        "model": "--model",
        "dataset": "--dataset",
        "output_dir": "--output-dir",
        "weights": "--weights",
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "lr": "--lr",
        "device": "--device",
        "seed": "--seed",
        "num_workers": "--num-workers",
        "optimizer": "--optimizer",
    }
    for key, flag in flag_map.items():
        if key in exp and exp[key] is not None:
            cmd.extend([flag, str(exp[key])])

    if exp.get("pretrained"):
        cmd.append("--pretrained")
    if exp.get("cosine_lr") is False:
        cmd.append("--no-cosine-lr")
    if exp.get("use_ema") is False:
        cmd.append("--no-ema")
    if "early_stopping_patience" in exp:
        cmd.extend(["--patience", str(exp["early_stopping_patience"])])
    if "clip_grad_norm" in exp:
        cmd.extend(["--clip-grad-norm", str(exp["clip_grad_norm"])])
    return cmd


def _record_error(
    path: Path,
    name: str,
    _exp: dict,
    exit_code: int,
    log_path: Path,
    traceback: str | None = None,
) -> None:
    import datetime

    record = {
        "exp_name": name,
        "exit_code": exit_code,
        "log_path": str(log_path),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if traceback:
        record["traceback"] = traceback
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_success(path: Path, name: str, exp: dict, log_path: Path) -> None:
    import datetime

    record = {
        "exp_name": name,
        "status": "ok",
        "log_path": str(log_path),
        "output_dir": str(exp.get("output_dir", "")),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
