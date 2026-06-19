# CUDA 环境配置

[English](cuda_setup.md) | [中文](cuda_setup.zh-CN.md)

## 安装 cu118 版 PyTorch

本项目固定了 `torch>=2.2,<2.7` 和 `torchvision>=0.17,<0.22`（参见 `pyproject.toml`）。如需 CUDA 11.8 支持，请从 PyTorch cu118 索引安装：

```bash
uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

验证 CUDA 是否可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 预期输出：2.5.1+cu118 True <GPU 名称>
```

如果安装后 `torch.cuda.is_available()` 返回 `False`，说明安装的 wheel 没有包含 GPU 构建版本。请使用显式的 `+cu118` 后缀重新安装，并确认 `nvidia-smi` 能显示 GPU。

## 显存扩展

训练流水线通过 `--batch-size` 参数手动调整批次大小以适应可用显存——没有自动检测机制。对于约 80 GB 显存的 GPU（例如 A100 或 H100），建议的起始值：

| 模型系列 | 640×640 下的批次大小 | 320×320 下的批次大小 |
| --- | --- | --- |
| YOLOv5s / v7w / v8n / v8s / v10s / v9c | 64 | 256 |
| YOLOv8m / v8l | 32 | 128 |
| YOLOv8x | 16 | 64 |
| RetinaNet-R50（torchvision） | 16 | 64 |
| Faster-RCNN-R50（torchvision） | 8 | 32 |

请根据实际情况调整——从表中的值开始，加倍直到 OOM，然后回退一步。`torch.amp.GradScaler` AMP 路径（通过 `--amp` 启用）大约可将显存占用减半，吞吐量成本约为 10%。

## AMP

`scripts/train.py --amp` 启用 `torch.amp.autocast("cuda")` + `GradScaler`：前向传播使用混合精度（安全时用 FP16，归约操作用 FP32），反向传播执行反缩放 + 梯度裁剪 + 参数更新。默认开启——对于任何带有 Tensor Core 的 GPU（Volta+）建议启用。

**稳定性说明**：AMP 配合 `lr=0.01` 在长时间运行时可能导致训练发散。如果观察到 loss 尖峰或 NaN 梯度，请将学习率降至 `1e-3` 或使用 `--no-amp` 禁用混合精度。

`cracks_yolo/pipeline/train.py` 中的损失缩放逻辑：

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

`--num-workers` 控制 DataLoader 的工作进程数。默认值为 `min(4, os.cpu_count())`。对于 CPU 核心数较多的机器，可设置为 8-16 以保持 GPU 不空闲。调试时可设为 0（单线程数据加载能更快暴露数据集相关问题）。

## 通过调度器实现多 GPU

流水线本身是单 GPU 的（不支持 DDP）。对于多 GPU 服务器，可以通过调度器为每个实验设置 `CUDA_VISIBLE_DEVICES` 来并发运行多个实验：

```yaml
# experiments/my_sweep.yaml
scheduler:
  max_parallel: 4   # GPU 数量
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

启动大批实验，通过 `CUDA_VISIBLE_DEVICES` 分配到多个 GPU，将错误捕获到 `errors.jsonl`，然后重试失败项。

## 常见 CUDA 错误

- `RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu`：损失模块的内部张量（anchors、BCE 正样本权重、stride、proj）未同步到预测张量所在的设备。每个损失的 `__call__` 必须按照 `cracks_yolo/losses/yolov5.py` 中的模式进行同步（参见 `CLAUDE.md` 中的"损失设备同步约定"）。
- `CUDA out of memory`：减小 `--batch-size` 或 `--input-size`。启用 `--amp`。torchvision 的 Faster-RCNN 是 ZOO 中内存占用最高的模型。
- `nvidia-smi` 显示 GPU 但 `torch.cuda.is_available()` 返回 `False`：安装了错误的 torch wheel。请使用 cu118 索引 URL 重新安装。
