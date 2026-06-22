#!/usr/bin/env bash
# v5 official train + test + metrics. Usage: MODEL=yolov5s_sactr bash scripts/run_v5.sh
set -e
export PATH=/root/miniconda3/bin:$PATH
MODEL="${MODEL:-yolov5s_baseline}"
DS="${DS:-/root/autodl-tmp/CrackDetection_Augmentation.v1.yolov5pytorch}"
EPOCHS="${EPOCHS:-1200}"
BS="${BS:-64}"
cd /root/cracks-yolo

# data.yaml for yolov5 format
cat > /tmp/v5_data.yaml << EOF
path: $DS
train: train/images
val: valid/images
test: test/images
nc: 1
names: {0: 'cracks'}
EOF

CFG="experiments/v5_configs/${MODEL}.yaml"
WT="/root/autodl-fs/weights/yolov5su.pt"  # symlinked to shared weights
OUT="output/${MODEL}"

echo "=== TRAIN ${MODEL} ==="
cd deps/yolov5
python train.py --cfg ../../${CFG} --weights ${WT} --data /tmp/v5_data.yaml \
  --hyp data/hyps/hyp.scratch-low.yaml --epochs ${EPOCHS} --batch-size ${BS} \
  --img 640 --workers 16 --device 0 --cos-lr --optimizer SGD --single-cls \
  --patience 100 --project ../../output --name ${MODEL} --exist-ok

echo "=== VAL ${MODEL} ==="
python val.py --weights ../../${OUT}/weights/best.pt --data /tmp/v5_data.yaml \
  --task test --single-cls --img 640 --batch-size ${BS} \
  --project ../../output --name ${MODEL}_val --exist-ok

echo "=== METRICS ${MODEL} ==="
cd /root/cracks-yolo
python -c "
import torch, csv, json, time, io, os
from pathlib import Path
# Load model + compute params/GFLOPs/FPS
ckpt = torch.load('${OUT}/weights/best.pt', map_location='cpu', weights_only=False)
m = ckpt['model'] if 'model' in ckpt else ckpt
m = m.float().cuda().eval()
n = sum(p.numel() for p in m.parameters())
# GFLOPs
try:
    from thop import profile
    macs,_ = profile(m, inputs=(torch.zeros(1,3,640,640).cuda(),), verbose=False)
    gflops = 2*macs/1e9
except: gflops = 0
# FPS
times = []
with torch.no_grad():
    for _ in range(5): m(torch.zeros(1,3,640,640).cuda())  # warmup
    torch.cuda.synchronize()
    for _ in range(20):
        torch.cuda.synchronize()
        t0=time.perf_counter(); m(torch.zeros(1,3,640,640).cuda()); torch.cuda.synchronize()
        times.append(time.perf_counter()-t0)
fps = 20/sum(times); lat_ms = 1000*sum(times)/20
# Read val results
map50=map5095=p=r=''
val_csv = Path('output/${MODEL}_val/results.csv')
if val_csv.exists():
    rows=list(csv.DictReader(val_csv.open()))
    if rows:
        L=rows[-1]
        for k in L:
            if 'mAP_0.5' in k and ':' not in k: map50=L[k]
            elif 'mAP_0.5:0.95' in k: map5095=L[k]
            elif 'precision' in k: p=L[k]
            elif 'recall' in k: r=L[k]
# Write unified metrics.csv
with open('${OUT}/metrics.csv','w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=['model','n_parameters','gflops','fps_mean','latency_mean_ms','map50','map5095','precision','recall'])
    w.writeheader()
    w.writerow({'model':'${MODEL}','n_parameters':n,'gflops':round(gflops,2),'fps_mean':round(fps,1),'latency_mean_ms':round(lat_ms,2),'map50':map50,'map5095':map5095,'precision':p,'recall':r})
print('METRICS DONE:', json.dumps({'model':'${MODEL}','params':n,'gflops':gflops,'fps':fps,'map50':map50}))
"
echo "=== ALL DONE ${MODEL} ==="
