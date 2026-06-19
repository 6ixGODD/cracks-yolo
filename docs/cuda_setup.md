# CUDA setup

## Installing cu118 torch

The project pins `torch>=2.2,<2.7` and `torchvision>=0.17,<0.22` (see
`pyproject.toml`). For CUDA 11.8 support, install from the PyTorch cu118
index:

```bash
uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is available:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.5.1+cu118 True <GPU name>
```

If `torch.cuda.is_available()` returns `False` after install, the wheel
didn't pick up the GPU build. Reinstall with the explicit `+cu118` suffix
and verify `nvidia-smi` shows the GPU.

## VRAM scaling

The train pipeline auto-scales batch size to available VRAM via the
`--batch-size` flag — there is no auto-detection. For a rented 128GB GPU
server (e.g. A100 80GB SXM4 or H100 80GB), suggested starting points:

| Model family | Batch size @ 640×640 | Batch size @ 320×320 |
| --- | --- | --- |
| YOLOv5s / v7w / v8n / v8s / v10s / v9c | 64 | 256 |
| YOLOv8m / v8l | 32 | 128 |
| YOLOv8x | 16 | 64 |
| RetinaNet-R50 (torchvision) | 16 | 64 |
| Faster-RCNN-R50 (torchvision) | 8 | 32 |

Tune empirically — start at the table value, double until OOM, back off one
step. The `torch.amp.GradScaler` AMP path (enabled with `--amp`) roughly
halves VRAM use for ~10% throughput cost.

## AMP

`scripts/train.py --amp` enables `torch.amp.autocast("cuda")` + `GradScaler`:
forward runs in mixed precision (FP16 where safe, FP32 for reductions),
backward unscales + clips + steps. Default off — enable for any GPU with
Tensor Cores (Volta+).

The loss-scaling logic in `cracks_yolo/pipeline/train.py`:

```python
with torch.amp.autocast("cuda", enabled=scaler is not None):
    preds = model(images)
    loss, parts = model.compute_loss(preds, yolo_targets, imgs=images)
optimizer.zero_grad()
if scaler is not None:
    scaler.scale(loss).backward()
    if cfg.clip_grad_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    if cfg.clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
    optimizer.step()
```

## num_workers

`--num-workers` controls DataLoader workers. Default `min(4, os.cpu_count())`.
For rented servers with many CPU cores, set to 8-16 to keep the GPU fed.
Set to 0 for debugging (single-threaded data loading surfaces dataset bugs
faster).

## Multi-GPU via scheduler

The pipeline itself is single-GPU (no DDP). For multi-GPU servers, run
multiple experiments concurrently via the scheduler with
`CUDA_VISIBLE_DEVICES` per experiment:

```yaml
# experiments.yaml
scheduler:
  max_parallel: 4   # number of GPUs
experiments:
  - name: yolov5s_baseline
    model: yolov5s
    env: {CUDA_VISIBLE_DEVICES: "0"}
    ...
  - name: yolov5s_sactr
    model: yolov5s_sactr
    env: {CUDA_VISIBLE_DEVICES: "1"}
    ...
```

This is the intended usage for the 7-day rented-GPU-server scenario:
launch a large batch of experiments, distribute across GPUs via
`CUDA_VISIBLE_DEVICES`, capture errors to `errors.jsonl`, retry failures.

## Common CUDA errors

- `RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu`:
  loss module's internal tensors (anchors, BCE pos-weights, stride, proj)
  weren't synced to the prediction device. Every loss's `__call__` must sync
  via the pattern in `cracks_yolo/losses/yolov5.py` (see "Loss device-sync
  convention" in `CLAUDE.md`).
- `CUDA out of memory`: reduce `--batch-size` or `--input-size`. Enable
  `--amp`. The torchvision Faster-RCNN is the most memory-hungry model in
  the ZOO.
- `nvidia-smi` shows GPU but `torch.cuda.is_available()` returns `False`:
  the wrong torch wheel is installed. Reinstall with the cu118 index URL.
