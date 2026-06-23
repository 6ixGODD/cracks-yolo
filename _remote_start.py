"""Run on each server: python _remote_start.py <SERVER_ID>"""
import subprocess, sys, os
os.environ["PATH"] = "/root/miniconda3/bin:/usr/bin:/bin"
os.chdir("/root/cracks-yolo")
os.makedirs("/root/cracks-yolo/output", exist_ok=True)

DS = "/root/autodl-tmp/CrackDetection_Augmentation.v1.yolov5pytorch"
OUT = "/root/cracks-yolo/output"
sid = int(sys.argv[1])

TASKS = {
    1: [
        ("yolov8n", "train_ultralytics", "300", "128", "0"),
        ("yolov8n_sac", "train_ultralytics", "300", "128", "0"),
        ("yolov8s", "train_ultralytics", "300", "64", "0"),
        ("yolov8s_sac", "train_ultralytics", "300", "64", "0"),
    ],
    2: [
        ("yolov8m", "train_ultralytics", "300", "32", "0"),
        ("yolov8l", "train_ultralytics", "300", "16", "0"),
        ("yolov9c", "train_ultralytics", "300", "32", "0"),
        ("yolov9c_sac", "train_ultralytics", "300", "32", "0"),
    ],
    3: [
        ("yolov10s", "train_ultralytics", "300", "64", "0"),
        ("yolov10s_sac", "train_ultralytics", "300", "64", "0"),
        ("yolov7w", "train", "300", "64", ""),
        ("yolov7w_sac", "train", "300", "64", ""),
    ],
    4: [
        ("retinanet_r50", "train", "150", "16", ""),
        ("faster_rcnn_r50", "train", "150", "8", ""),
        ("fcos_r50", "train", "150", "16", ""),
        ("ssd300_vgg16", "train", "150", "32", ""),
        ("ssdlite320_mobilenetv3", "train", "150", "64", ""),
    ],
}

for model, runner, epochs, bs, dev in TASKS[sid]:
    log = f"{OUT}/{model}_main.log"
    if runner == "train_ultralytics":
        cmd = ["python", "-m", "scripts.train_ultralytics", "--model", model, "--dataset", DS,
               "--output-root", OUT, "--epochs", epochs, "--batch-size", bs, "--device", dev, "--workers", "8"]
    else:
        cmd = ["python", "-m", "scripts." + runner, "--model", model, "--dataset", DS,
               "--output-root", OUT, "--epochs", epochs, "--batch-size", bs,
               "--lr", "1e-3", "--pretrained", "--no-amp", "--cosine-lr", "--use-ema",
               "--clip-grad-norm", "10.0", "--device", "cuda", "--num-workers", "8"]
        if "rcnn" in model or "fcos" in model or "ssd" in model:
            cmd[cmd.index("--lr") + 1] = "1e-4"

    print(f"=== {model} ===")
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"FAILED: {model} (exit {p.returncode})")
        # continue to next model anyway
    else:
        print(f"DONE: {model}")
