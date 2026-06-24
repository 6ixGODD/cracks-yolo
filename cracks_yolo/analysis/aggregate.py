"""Aggregate test metrics across model output directories.

Scans ``<root>/<model>/test/metrics.csv``, extracts the ``_test`` row
for each model, and merges them into a single table (CSV + Excel).
"""

from __future__ import annotations

import csv
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


def aggregate(root: Path, out_csv: Path, out_excel: Path | None = None) -> Path:
    """Aggregate test metrics and write CSV (+ optional Excel).

    Returns the path to the written CSV.
    """
    rows = _find_test_metrics(root)
    if not rows:
        raise FileNotFoundError(f"no test/metrics.csv found under {root}")

    # Collect all known field names
    fieldnames = ["model"]
    seen_fields: set[str] = set()
    for _, metrics in rows:
        for k in metrics:
            if k != "model" and k not in seen_fields:
                fieldnames.append(k)
                seen_fields.add(k)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for model_name, metrics in rows:
            row = {"model": model_name, **metrics}
            writer.writerow(row)

    if out_excel:
        _write_excel(rows, fieldnames, out_excel)

    return out_csv


def _write_excel(
    rows: list[tuple[str, dict[str, str]]],
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
        ws.append([model_name] + [metrics.get(k, "") for k in fieldnames[1:]])
    wb.save(path)
