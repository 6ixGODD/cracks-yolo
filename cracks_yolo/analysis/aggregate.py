"""Aggregate test metrics + model analysis across model output directories.

Scans ``<root>/<model>/test/metrics.csv``, extracts the ``_test`` row
for each model, merges with ``<root>/<model>/analysis.json``, and
writes a single table (CSV + Excel).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _find_test_metrics(root: Path) -> list[tuple[str, dict[str, str]]]:
    """Return ``[(model_name, metrics_dict), ...]`` for each model under *root*."""
    results: list[tuple[str, dict[str, str]]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        csv_path = d / "test" / "metrics.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_col = row.get("model", "")
                if model_col.endswith("_test"):
                    clean = {k.strip(): v.strip() for k, v in row.items()}
                    results.append((d.name, clean))
                    break
    return results


def _load_analysis(root: Path) -> dict[str, dict[str, str]]:
    """Return ``{model_name: analysis_dict}`` for models with analysis.json."""
    analysis: dict[str, dict[str, str]] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        path = d / "analysis.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        flat: dict[str, str] = {}
        for k, v in data.items():
            if k == "model_name":
                continue
            if isinstance(v, float):
                flat[k] = f"{v:.4f}"
            elif isinstance(v, int):
                flat[k] = str(v)
            else:
                flat[k] = str(v)
        analysis[d.name] = flat
    return analysis


def aggregate(root: Path, out_csv: Path, out_excel: Path | None = None) -> Path:
    """Aggregate test metrics + analysis and write CSV (+ optional Excel).

    Returns the path to the written CSV.
    """
    rows = _find_test_metrics(root)
    if not rows:
        raise FileNotFoundError(f"no test/metrics.csv found under {root}")

    analysis = _load_analysis(root)

    # Collect all known field names: model first, then metrics, then analysis
    fieldnames = ["model"]
    seen_fields: set[str] = set()
    for _, metrics in rows:
        for k in metrics:
            if k != "model" and k not in seen_fields:
                fieldnames.append(k)
                seen_fields.add(k)
    analysis_keys: list[str] = []
    for a in analysis.values():
        for k in a:
            if k not in seen_fields:
                analysis_keys.append(k)
                seen_fields.add(k)
    fieldnames.extend(analysis_keys)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for model_name, metrics in rows:
            row: dict[str, str] = {"model": model_name, **metrics}
            if model_name in analysis:
                row.update(analysis[model_name])
            writer.writerow(row)

    if out_excel:
        _write_excel(rows, analysis, fieldnames, out_excel)

    return out_csv


def _write_excel(
    rows: list[tuple[str, dict[str, str]]],
    analysis: dict[str, dict[str, str]],
    fieldnames: list[str],
    path: Path,
) -> None:
    """Write aggregated data to Excel (openpyxl)."""
    try:
        from openpyxl import Workbook
    except ImportError:
        import warnings

        warnings.warn("openpyxl not installed, skipping Excel output", stacklevel=2)
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "test_metrics"
    ws.append(fieldnames)
    for model_name, metrics in rows:
        row_vals = [model_name]
        for k in fieldnames[1:]:
            val = metrics.get(k, "")
            if not val and model_name in analysis:
                val = analysis.get(model_name, {}).get(k, "")
            row_vals.append(val)
        ws.append(row_vals)
    wb.save(path)
