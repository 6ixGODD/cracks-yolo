"""S5: SACTR redo from scratch + v8/v10/v7 missing models."""
import subprocess, sys, os, time
os.environ["PATH"] = "/root/miniconda3/bin:/usr/bin:/bin"
os.chdir("/root/cracks-yolo")
os.makedirs("output", exist_ok=True)

DS = "/root/autodl-fs/CrackDetection_Augmentation.v1.yolov5pytorch"
W = "/root/autodl-fs/weights"

with open("/tmp/v5_data.yaml", "w") as f:
    f.write(f"path: {DS}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: 1\nnames: {{0: 'cracks'}}\n")

def v5_train(model, cfg, weights, epochs=1200, bs=64, patience=200):
    out = f"output/{model}"
    print(f"\n{'='*60}\n=== V5 TRAIN {model} (FROM SCRATCH, 1200ep, patience=200) ===\n{'='*60}", flush=True)
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
    # Test with save-json
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

def ul_train(model, epochs=300, bs=64):
    """Train via ultralytics YOLO().train()."""
    print(f"\n{'='*60}\n=== UL TRAIN {model} ===\n{'='*60}", flush=True)
    cmd = ["python", "-m", "scripts.train_ultralytics",
        "--model", model, "--dataset", DS, "--output-root", "output",
        "--epochs", str(epochs), "--batch-size", str(bs),
        "--device", "0", "--workers", "8", "--patience", "100"]
    with open(f"output/{model}_main.log", "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"{'DONE' if p.returncode==0 else 'FAILED'}: {model}", flush=True)

def ck_train(model, epochs, bs, lr="1e-3"):
    """Train via cracks_yolo scripts.train."""
    print(f"\n{'='*60}\n=== CK TRAIN {model} ===\n{'='*60}", flush=True)
    cmd = ["python", "-m", "scripts.train",
        "--model", model, "--dataset", DS, "--output-root", "output",
        "--epochs", str(epochs), "--batch-size", str(bs), "--lr", lr,
        "--pretrained", "--no-amp", "--device", "cuda", "--num-workers", "8",
        "--cosine-lr", "--use-ema", "--clip-grad-norm", "10.0"]
    with open(f"output/{model}_main.log", "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"{'DONE' if p.returncode==0 else 'FAILED'}: {model}", flush=True)

# === S5 TASKS ===
# 1. SACTR redo from scratch (COCO weights, NOT best.pt), 1200 ep, patience 200
v5_train("yolov5s_sactr_v3", "yolov5s_sactr.yaml", "yolov5su.pt", epochs=1200, bs=64, patience=200)
# 2. v8n_sac
ul_train("yolov8n_sac", bs=128)
# 3. v8s
ul_train("yolov8s", bs=64)
# 4. v8s_sac
ul_train("yolov8s_sac", bs=64)
# 5. v8x
ul_train("yolov8x", bs=8)
# 6. v10s_sac
ul_train("yolov10s_sac", bs=64)
# 7. v7w_sac
ck_train("yolov7w_sac", 300, 64)

print("\n=== S5 ALL DONE ===", flush=True)
