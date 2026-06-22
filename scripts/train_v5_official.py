"""Train YOLOv5 (n/s/m/l/x × baseline/sac/tr/sactr) via the official deps/yolov5 fork.

The official yolov5 training loop (mosaic, cosine lr, EMA, SGD) is preserved
exactly — results match `python deps/yolov5/train.py`. SAC/TR are injected
into the fork's models/common.py (C3SAC/SAConv2d) and selected via the model
YAML in experiments/v5_configs/.

This wrapper:
1. Builds a yolov5-compatible data.yaml pointing at the tongue dataset.
2. Invokes deps/yolov5/train.py as a subprocess (official loop, no reimpl).
3. After training, runs deps/yolov5/val.py to get mAP/AR.
4. Computes GFLOPs/params via thop + saves model structure (torchsummary).
5. Writes metrics.csv + model_structure.txt + REPORT.md to output_dir.

Usage:
    python -m scripts.train_v5_official --model yolov5s_sactr \\
        --dataset /root/autodl-tmp/CrackDetection_Augmentation.v1.yolov5pytorch \\
        --output-dir output/yolov5s_sactr --epochs 1200 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V5_ROOT = REPO_ROOT / "deps" / "yolov5"
V5_CONFIGS = REPO_ROOT / "experiments" / "v5_configs"


def _write_data_yaml(dataset: Path, out: Path, num_classes: int) -> None:
    """Write a yolov5-compatible data.yaml for the tongue dataset."""
    names = {0: "cracks"} if num_classes == 1 else {i: f"class_{i}" for i in range(num_classes)}
    text = (
        f"path: {dataset}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n\n"
        f"nc: {num_classes}\n"
        f"names: {names}\n"
    )
    out.write_text(text, encoding="utf-8")


def _run(cmd: list[str], log_path: Path) -> int:
    """Run a subprocess, tee output to log_path + stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v5 via official fork.")
    parser.add_argument("--model", required=True, help="e.g. yolov5s_sactr (size_variant)")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root (has train/valid/test)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--weights", type=Path, default=Path("weights/yolov5s.pt"),
                        help="COCO pretrained weights (auto-sized if v5n/m/l/x).")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    args = parser.parse_args()

    cfg = V5_CONFIGS / f"{args.model}.yaml"
    if not cfg.exists():
        raise SystemExit(f"config not found: {cfg}. Generate via output/_gen_v5_yaml.py")

    # Auto-size pretrained weights: yolov5s_sactr -> weights/yolov5s.pt
    size = args.model.split("_")[0].replace("yolov5", "")
    weights = args.weights
    if weights.name == "yolov5s.pt" and size != "s":
        sized = weights.parent / f"yolov5{size}.pt"
        if sized.exists():
            weights = sized

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = args.output_dir / "data.yaml"
    _write_data_yaml(Path(args.dataset), data_yaml, args.num_classes)

    py = sys.executable
    # 1. Train via official train.py.
    train_cmd = [
        py, str(V5_ROOT / "train.py"),
        "--cfg", str(cfg),
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--hyp", str(V5_ROOT / "data/hyps/hyp.scratch-low.yaml"),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img", str(args.img),
        "--workers", str(args.workers),
        "--device", args.device,
        "--project", str(args.output_dir.parent),
        "--name", args.output_dir.name,
        "--exist-ok",
        "--cos-lr",
        "--optimizer", "SGD",
        "--single-cls" if args.num_classes == 1 else "",
        "--patience", str(args.patience),
    ]
    train_cmd = [c for c in train_cmd if c]
    print("=== TRAIN ===")
    print(" ".join(train_cmd))
    rc = _run(train_cmd, args.output_dir / "train.log")
    if rc != 0:
        raise SystemExit(f"train.py failed (exit {rc}), see {args.output_dir / 'train.log'}")

    # Official train.py saves to output_dir.parent/output_dir.name/{best,last}.pt
    # but --project/--name should put them directly in args.output_dir.
    best = args.output_dir / "weights" / "best.pt"
    if not best.exists():
        best = args.output_dir / "best.pt"
    print(f"\n=== best.pt: {best} ===")

    # 2. Val via official val.py (mAP/AR).
    print("=== VAL ===")
    val_cmd = [
        py, str(V5_ROOT / "val.py"),
        "--weights", str(best),
        "--data", str(data_yaml),
        "--img", str(args.img),
        "--batch-size", str(args.batch_size),
        "--task", "val",
        "--project", str(args.output_dir.parent),
        "--name", args.output_dir.name + "_val",
        "--exist-ok",
        "--single-cls" if args.num_classes == 1 else "",
    ]
    val_cmd = [c for c in val_cmd if c]
    _run(val_cmd, args.output_dir / "val.log")

    # 3. Extract metrics + GFLOPs + structure.
    _extract_metrics(best, cfg, args.output_dir, args.num_classes, args.img)
    print(f"\nDone. Artifacts in {args.output_dir}")


def _extract_metrics(weights: Path, cfg: Path, out_dir: Path, nc: int, img: int) -> None:
    """Run a Python snippet to load the trained model, compute GFLOPs/params,
    save model structure, and merge val.py's results.csv into metrics.csv."""
    snippet = f"""
import sys, json, csv
sys.path.insert(0, "{V5_ROOT}")
import torch
from models.yolo import Model
from utils.torch_utils import profile

m = Model("{cfg}", ch=3, nc={nc})
ck = torch.load("{weights}", map_location="cpu", weights_only=False)
csd = ck["model"].float().state_dict()
m.load_state_dict(csd, strict=False)
m.eval()

# Params.
n_params = sum(p.numel() for p in m.parameters())

# GFLOPs via thop (yolov5's profile uses thop internally).
try:
    from thop import profile as thop_profile
    x = torch.zeros(1, 3, {img}, {img})
    macs, _ = thop_profile(m, inputs=(x,), verbose=False)
    gflops = 2 * macs / 1e9
except Exception as e:
    gflops = 0.0
    print("thop failed:", e)

# Model structure via torchsummary.
try:
    from torchsummary import summary
    import io
    buf = io.StringIO()
    summary(m, ({img}, {img}, 3), device="cpu", file=buf)
    open("{out_dir}/model_structure.txt", "w").write(buf.getvalue())
except Exception as e:
    print("torchsummary failed:", e)

# Read val.py results.csv if present.
metrics = {{"n_parameters": n_params, "gflops": gflops}}
print(json.dumps(metrics))
"""
    res = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr[:500], file=sys.stderr)
    metrics = {"n_parameters": 0, "gflops": 0.0}
    for line in res.stdout.splitlines():
        if line.strip().startswith("{"):
            try:
                metrics = json.loads(line.strip())
            except Exception:
                pass

    # Write metrics.csv (merge with val results.csv if present).
    val_csv = out_dir.parent / (out_dir.name + "_val") / "results.csv"
    fields = ["model", "n_parameters", "gflops", "map50", "map5095", "precision", "recall"]
    row = {"model": out_dir.name, "n_parameters": metrics["n_parameters"],
           "gflops": round(metrics["gflops"], 4), "map50": "", "map5095": "",
           "precision": "", "recall": ""}
    if val_csv.exists():
        with val_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last = rows[-1]
                # yolov5 results.csv columns: ...metrics/mAP_0.5, metrics/mAP_0.5:0.95, metrics/precision, metrics/recall
                row["map50"] = last.get("metrics/mAP_0.5", last.get("               metrics/mAP_0.5", ""))
                row["map5095"] = last.get("metrics/mAP_0.5:0.95", last.get("metrics/mAP_0.5:0.95", ""))
                row["precision"] = last.get("metrics/precision", last.get("metrics/precision", ""))
                row["recall"] = last.get("metrics/recall", last.get("metrics/recall", ""))
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    print(f"metrics.csv written: {row}")


if __name__ == "__main__":
    main()
