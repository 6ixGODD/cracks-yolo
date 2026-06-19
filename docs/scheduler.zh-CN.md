# 实验调度器

[English](scheduler.md) | [中文](scheduler.zh-CN.md)

`scripts/schedule_experiments.py` 运行 YAML 定义的批量实验，具备子进程隔离、错误捕获和重试模式。

## 配置目录

两个开箱即用的 YAML 配置文件位于 [`experiments/`](../experiments/)：

- **`experiments/all_models_direct.yaml`** — 26 个模型 × 2 个实验（训练 + 测试）= 52 个实验。在 `train` 划分上训练，训练期间在 `valid` 划分上验证，然后在保留的 `test` 划分上测试 `best.pt`。
- **`experiments/all_models_cv5.yaml`** — 26 个模型 × 1 个交叉验证实验 = 26 个实验。CV 模式将 train+valid+test 合并为一个池，运行 5 折 CV（保留折 = 测试，剩余 90/10 训练/验证划分）。

参见 [`experiments/README.md`](../experiments/README.md) 了解 batch-size 调优和多 GPU 说明。

## YAML 格式

```yaml
# experiments/my_sweep.yaml
scheduler:
  max_parallel: 1              # 默认为串行；>1 则启动子进程
  seed: 42

experiments:
  - name: yolov5s_baseline
    type: train                # train | test（交叉验证为 `type: train` + cross_val: true）
    model: yolov5s
    dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
    epochs: 100
    batch_size: 32
    lr: 0.001
    output_dir: output/yolov5s_baseline
    seed: 42
    pretrained: true           # 加载官方 COCO 权重（strict=False）

  - name: yolov5s_sactr_cv
    type: train
    model: yolov5s_sactr
    dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
    cross_val: true
    n_folds: 5
    val_fraction: 0.1          # 每折训练池的 10% 用作验证（反向传播）
    epochs: 100
    batch_size: 32
    output_dir: output/yolov5s_sactr_cv

  - name: yolov5s_sactr_test
    type: test
    model: yolov5s_sactr
    weights: output/yolov5s_sactr_cv/best.pt
    dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
    split: test
    output_dir: output/yolov5s_sactr_test
```

### 每个实验的字段

| 字段 | 类型 | 适用场景 | 描述 |
| --- | --- | --- | --- |
| `name` | str | 全部 | 唯一实验标识符（用于日志文件名）。 |
| `type` | `train` \| `test` | 全部 | `train` 运行 `scripts.train`；`test` 运行 `scripts.test`。CV 模式为 `type: train` 加 `cross_val: true`。 |
| `model` | str | 全部 | ZOO 键（例如 `yolov5s_sactr`）。 |
| `dataset` | path | 全部 | 包含 `data.yaml` 的 YOLO 数据集根目录。 |
| `output_dir` | path | 全部 | 产物输出目录。 |
| `epochs` | int | train | 训练轮数。 |
| `batch_size` | int | 全部 | 每 GPU 的 batch size。 |
| `lr` | float | train | 学习率。YOLO=0.01，torchvision=0.0001。 |
| `input_size` | int | 全部 | 模型输入尺寸（默认 640；SSD300 为 300，SSDlite320 为 320）。 |
| `device` | str | 全部 | `cuda` 或 `cpu`。 |
| `seed` | int | 全部 | 可复现性种子。 |
| `num_workers` | int | 全部 | DataLoader 工作进程数。 |
| `pretrained` | bool | train | 通过 `from_pretrained` 加载官方 COCO 权重（`strict=False`）。 |
| `cross_val` | bool | train | 运行 N 折 CV 而非单次训练/验证划分。 |
| `n_folds` | int | train（CV） | CV 折数。 |
| `val_fraction` | float | train（CV） | 每折训练池中划出作为验证的比例。默认 0.1。 |
| `no_amp` | bool | train | 禁用混合精度训练。 |
| `weights` | path | test | 测试所用的 `best.pt` 路径。 |
| `split` | str | test | 要评估的数据集划分（默认 `test`）。 |
| `env` | dict | 全部 | 每个实验的环境变量覆盖（例如 `{CUDA_VISIBLE_DEVICES: "0"}`）。 |

## 行为

- **子进程隔离**：每个实验在一个新的 `subprocess.Popen` 中运行，因此崩溃被隔离。标准输出/标准错误被捕获到 `<output-dir>/scheduler/<exp_name>.log`。
- **错误捕获**：发生任何异常（或非零退出码）时，将 `{exp_name, type, config, exit_code, log_path, traceback, timestamp}` 追加到 `<output-dir>/scheduler/errors.jsonl`。继续执行下一个实验——绝不中止整个批次。
- **成功捕获**：将 `{exp_name, status, exit_code, output_dir, elapsed_sec, timestamp}` 追加到 `<output-dir>/scheduler/results.jsonl`。
- **并行执行**：`max_parallel > 1` 使用 `ThreadPoolExecutor` 同时启动最多 N 个子进程。每个子进程是一个真正的操作系统进程（等待时 GIL 被释放），因此实现了真正的并行。GPU 分配通过 YAML 中每个实验的 `env: {CUDA_VISIBLE_DEVICES: "N"}` 实现。单 GPU 机器应保持 `max_parallel: 1`——多个实验共享一个 GPU 更可能导致 OOM 而非加速整体批次。

## 重试模式

失败的实验可以重试，而无需重新运行整个批次：

```bash
python -m scripts.schedule_experiments \
    --retry-failed output/all_models_direct/scheduler/errors.jsonl \
    --output-dir output/all_models_direct_retry
```

调度器将：
1. 读取 `errors.jsonl`。
2. 生成 `retry_from_errors.yaml`，仅包含失败的实验。
3. 将原始 `errors.jsonl` 备份为 `errors.bak.jsonl`。
4. 重新运行失败的实验。

迭代工作流：启动大批量实验，检查 `errors.jsonl`，修复错误的配置（或库问题），重试失败项，重复直到 `errors.jsonl` 为空。

## 命令行

```bash
# 运行新的批次。
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct

# 5 折 CV 扫描。
python -m scripts.schedule_experiments \
    --config experiments/all_models_cv5.yaml \
    --output-dir output/all_models_cv5

# 重试失败的实验。
python -m scripts.schedule_experiments \
    --retry-failed output/all_models_direct/scheduler/errors.jsonl \
    --output-dir output/all_models_direct_retry
```
