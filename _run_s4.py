"""S4: train v5n/v5m/v5l + mask_rcnn + detr_r50."""
import subprocess, sys, os, time
os.environ["PATH"] = "/root/miniconda3/bin:/usr/bin:/bin"
os.chdir("/root/cracks-yolo")
os.makedirs("output", exist_ok=True)

DS = "/root/autodl-fs/CrackDetection_Augmentation.v1.yolov5pytorch"
W = "/root/autodl-fs/weights"

# Write data.yaml for yolov5
with open("/tmp/v5_data.yaml", "w") as f:
    f.write(f"path: {DS}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: 1\nnames: {{0: 'cracks'}}\n")

def v5_train(model, cfg, weights, epochs=1200, bs=64, patience=200):
    """Train via deps/yolov5 official train.py."""
    out = f"output/{model}"
    print(f"\n{'='*60}\n=== V5 TRAIN {model} ===\n{'='*60}", flush=True)
    cmd = ["python", "deps/yolov5/train.py",
        "--cfg", f"experiments/v5_configs/{cfg}",
        "--weights", f"{W}/{weights}",
        "--data", "/tmp/v5_data.yaml",
        "--hyp", "deps/yolov5/data/hyps/hyp.scratch-low.yaml",
        "--epochs", str(epochs), "--batch-size", str(bs),
        "--img", "640", "--workers", "8", "--device", "0",
        "--cos-lr", "--optimizer", "SGD", "--single-cls",
        "--patience", str(patience),
        "--project", "output", "--name", model, "--exist-ok"]
    with open(f"output/{model}_main.log", "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"FAILED: {model}", flush=True)
    else:
        print(f"DONE: {model}", flush=True)
    # Val on test set with save-json
    print(f"=== V5 TEST {model} ===", flush=True)
    cmd2 = ["python", "deps/yolov5/val.py",
        "--weights", f"{out}/weights/best.pt",
        "--data", "/tmp/v5_data.yaml",
        "--task", "test", "--single-cls", "--img", "640",
        "--batch-size", "32", "--save-json", "--save-txt", "--save-conf",
        "--project", "output", "--name", f"{model}_test", "--exist-ok"]
    with open(f"output/{model}_test.log", "w") as log:
        subprocess.run(cmd2, stdout=log, stderr=subprocess.STDOUT)
    print(f"TEST DONE: {model}", flush=True)

def ck_train(model, epochs, bs, lr="1e-4"):
    """Train via cracks_yolo scripts.train."""
    print(f"\n{'='*60}\n=== CK TRAIN {model} ===\n{'='*60}", flush=True)
    cmd = ["python", "-m", "scripts.train",
        "--model", model, "--dataset", DS, "--output-root", "output",
        "--epochs", str(epochs), "--batch-size", str(bs), "--lr", lr,
        "--pretrained", "--device", "cuda", "--num-workers", "8",
        "--cosine-lr", "--use-ema", "--clip-grad-norm", "10.0"]
    if "rcnn" not in model and "fcos" not in model and "ssd" not in model and "detr" not in model:
        cmd += ["--no-amp"]
    with open(f"output/{model}_main.log", "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"{'DONE' if p.returncode==0 else 'FAILED'}: {model}", flush=True)

# === S4 TASKS ===
# 1. v5n (small, fast)
v5_train("yolov5n", "yolov5n_baseline.yaml", "yolov5nu.pt", bs=128)
# 2. v5m
v5_train("yolov5m", "yolov5m_baseline.yaml", "yolov5mu.pt", bs=32)
# 3. v5l
v5_train("yolov5l", "yolov5l_baseline.yaml", "yolov5lu.pt", bs=16)
# 4. mask_rcnn
ck_train("mask_rcnn_r50", 150, 8)
# 5. detr_r50
ck_train("detr_r50", 150, 8)

print("\n=== S4 ALL DONE ===", flush=True)
