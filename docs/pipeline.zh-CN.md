# 流水线

[English](pipeline.md) | [中文](pipeline.zh-CN.md)

流水线层为模型方法之上的薄过程封装。流水线不持有模型特定逻辑；全部分支通过模型声明的属性（`decode_format`、`loss_parts_schema`）或对 `DetectorModel` Protocol 的结构合规性中介。

## 架构概览

```
CLI (Typer)
  ├─ train  → run_train()           → model.train_model(config)
  ├─ test   → run_test()            → model.load() → model.inference() → COCO 评估
  ├─ run    → run_train() + run_test()  （从 YAML 配置先训后测）
  └─ compose → run_compose()        → 逐实验子进程
```

流水线层位于 `cracks_yolo/pipeline/`。其模块如下：

| 模块 | 职责 |
| --- | --- |
| `train.py` | `run_train()` — 模型构建、预训练加载、分发至 `model.train_model()` |
| `test.py` | `run_test()` — 检查点加载、批量推理、COCO 指标计算 |
| `compose.py` | `run_compose()` — YAML 驱动的实验调度器，支持 `$include` 递归 |
| `_helpers.py` | 共享工具：NMS、目标格式转换、随机种子设定、anchor-free 检测 |

## 训练流程

### 入口点

`run_train(model_name, dataset, output_dir, epochs, batch_size, lr, pretrained, device, seed, num_workers, **kwargs) -> TrainReport`

该函数执行三步：

1. **模型查询。** `model_name` 对照 `ZOO` 字典（`dict[str, type[BaseModel]]`）解析。未知键触发 `ValueError`。

2. **构造。** 实例化模型类。若 `pretrained=True` 且该类暴露 `from_pretrained`，则先调用构造函数获取架构元数据，随后 `from_pretrained` 通过键匹配的 `load_state_dict(strict=False)` 以 COCO 预训练权重重载模型。ultralytics 适配器模型额外接受 `model_name` 参数以指定 YAML 配置文件。

3. **分发。** 由函数实参填充 `TrainConfig` 数据类并传入 `model.train_model(config)`。流水线除检测构造函数签名中是否存在 `from_pretrained` 与 `model_name` 外，不检查模型类。

### TrainConfig

`TrainConfig` 是一普通 `@dataclass`（非 pydantic 模型），定义于 `cracks_yolo/zoo/base.py`。字段：`output_dir`、`dataset`、`data_yaml`、`epochs`、`batch_size`、`lr`、`weight_decay`、`warmup_epochs`、`warmup_lr`、`momentum`、`amp`、`clip_grad_norm`、`early_stopping_patience`、`cosine_lr`、`cosine_lrf`、`use_ema`、`ema_decay`、`optimizer`、`seed`、`device`、`num_workers`、`pretrained`、`extra_kwargs`。

采用普通数据类以使模型子类可在不受 pydantic 校验约束的条件下扩展配置。

### 两种训练实现

`train_model` 方法由模型定义，存在两种规范形态：

**UltralyticsAdapter.train_model**（YOLOv3--v10、YOLO12）。直接构造 ultralytics 的 `DetectionTrainer`（或 `RTDETRTrainer`），绕过会重新构建原始 `DetectionModel` 从而丢失 SAC/TR 注入的 `YOLO().train()` 路径。训练器的 `model` 属性在调用 `trainer.train()` 前被替换为已注入 SAC 的 `nn.Module`。训练结束后，ultralytics 的输出产物同步至 `config.output_dir`。

**TorchvisionBase._run_train_loop**（RetinaNet、Faster-RCNN、Mask-RCNN、FCOS、SSD300、SSDlite320）。手写 epoch 循环：带线性预热的余弦学习率调度、可选的 `torch.amp.GradScaler` AMP、可选的梯度裁剪、基于 `pycocotools` 的逐 epoch 验证、早停以及 CSV 指标日志。各子类从其自身的 `train_model` 调用 `_run_train_loop`，并传入架构特定的 `score_thresh`。

### 训练后

`train_model` 返回后，`run_train` 将模型检查点保存至 `best.pt`，若安装了 `thop`，写含参数量、GFLOPs、FPS、延迟及峰值显存的 `analysis.json`。

## 测试流程

### 入口点

`run_test(model_name, weights, dataset, output_dir, batch_size, device, seed) -> dict[str, Any]`

该函数：

1. 从 ZOO 字典实例化模型类。
2. 调用 `model.load(weights)` 恢复训练参数。load 方法兼处理 ultralytics 检查点格式（`ckpt["model"]` 是 `DetectionModel` 实例）与通用格式（`model_state_dict` 键）。
3. 将模型移至指定设备。
4. 对 `("test", "valid")` 中的每个划分：
   - 从 YOLO 格式数据集目录载入记录。
   - 构造使用评估变换（仅 resize，无数据增强）的 `DetectionDataset`。
   - 按 batch 通过 `model.inference(images)` 迭代，返回 `list[InferenceResult]`。
   - 将预测转换为 COCO JSON 格式（边界框从 xyxy 转为 xywh）。
   - 通过 `pycocotools.cocoeval.COCOeval` 在 IoU 阈值 0.5 与 0.5:0.95 下计算指标。
   - 计算逐图像 PR/ROC 曲线、混淆矩阵、灵敏度、特异度、PPV、NPV。
5. 写入 `metrics.csv` 及逐划分预测 JSON 文件。
6. 可选运行 `model.analyze()` 获取效率指标。

### 推理解码

各模型类声明 `decode_format` 类属性：`"anchor_free"`（YOLOv8/v9/v10）或 `"anchor_based"`（YOLOv5/v7、torchvision 包装器）。流水线读取该属性，而非基于类名分支。

**UltralyticsAdapter.inference。** 调用 `self._inner(images)` 获取原始网格预测，随后应用 `ultralytics.utils.nms.non_max_suppression`，参数为 `conf_thres=0.001`、`iou_thres=0.7`、`max_det=300`。返回 `InferenceResult` 对象，边界框裁剪至 `[0, input_size]`。

**TorchvisionBase.inference。** 调用 `self._inner(images)` 获取原始 torchvision 输出（边界框回归、分类 logits），随后应用模型内置后处理（分数阈值过滤、NMS）产生位于原始图像坐标的 `InferenceResult` 对象。

### InferenceResult

含三个 `torch.Tensor` 字段的普通容器：
- `boxes`：`(N, 4)` xyxy 像素坐标。
- `scores`：`(N,)` 置信度，取值区间 `[0, 1]`。
- `labels`：`(N,)` 整数类索引（0 起始）。

## Run 命令（训练 + 自动测试）

`run(config, output_dir, device, test_only, weights)` 加载单实验 YAML 配置。若 `type: train`，调用 `run_train()` 后自动在最佳检查点上运行 `run_test()`。若 `type: test`，直接调用 `run_test()`。`--test-only` 标志跳过训练并要求提供 `--weights`。

## 实验编排

`run_compose(config, output_dir, max_parallel)` 加载带递归 `$include` 解析的编排 YAML。每个被包含文件为单实验配置或包含 `experiments` 列表的文件。实验通过 `subprocess.run` 顺序执行，stdout/stderr 捕获至逐实验日志文件。失败写入 `errors.jsonl`；成功写入 `results.jsonl`。支持逐实验 `env` 覆写（如 `CUDA_VISIBLE_DEVICES`），适配多 GPU 调度。

## 关键设计决策

**薄流水线，厚模型。** 流水线层刻意保持极薄。`run_train` 含 99 行；`run_test` 含 342 行，大部分为 COCO 指标计算代码。全部训练循环逻辑驻留于 `model.train_model()`。新增模型族只要满足隐式 Protocol，便无需改动流水线。

**无抽象钩子。** 不存在 `on_epoch_start` / `on_batch_end` 回调体系。各模型的 `train_model` 为自包含方法，自管其循环、优化器构造、学习率调度及验证。以一定量的代码重复换取完全的显式性。

**数据类，非 pydantic。** `TrainConfig` 为普通 `@dataclass`。pydantic 校验将引入额外依赖并限制模型特定扩展。校验发生于 CLI 边界（Typer 类型强制）及模型的 `train_model` 方法内部。

**解码格式自描述。** 模型声明 `decode_format: str`，流水线据此施加正确的张量重排而无需 `isinstance` 检查。同一原则适用于 `loss_parts_schema`，用于日志记录损失分量。

**状态机。** `BaseModel` 维护 `ModelState` 枚举（`UNINITIALIZED`、`PRETRAINED`、`TRAINED`）。`inference()` 断言 `TRAINED`；`from_pretrained()` 转移至 `PRETRAINED`；`train_model()` 转移至 `TRAINED`。此机制防止静默误用（如在未训练模型上运行推理）。

**两种训练器策略。** ultralytics 适配器复用上游 `DetectionTrainer` 以保持与 ultralytics 数据增强流水线、EMA 及指标日志的兼容。torchvision 适配器实现手写循环，缘于 torchvision 检测模型采用不同目标格式（`List[Dict[str, Tensor]]`）且不共享 ultralytics 训练基础设施。两种策略产出相同模式的 `TrainReport`。

## 产出物

| 产物 | 生产者 | 内容 |
| --- | --- | --- |
| `best.pt` | train | 最佳验证 mAP@50 处的模型状态字典检查点 |
| `metrics.csv` | train, test | 逐 epoch 损失（训练）；逐划分指标（测试） |
| `analysis.json` | train | 参数量、GFLOPs、FPS、延迟、显存 |
| `best_predictions_{split}.json` | test | 逐图像 COCO 格式检测结果 |
| `run.log.jsonl` | train (ultralytics) | 结构化 JSONL 训练日志 |
| `results.csv` | train (ultralytics) | ultralytics 格式训练指标 |
| `config.yaml` | train (ultralytics) | 冻结的训练配置 |
| `errors.jsonl` | compose | 逐实验失败记录 |
| `results.jsonl` | compose | 逐实验成功记录 |

## CLI 参考

```
cy train  --model yolov5s_sac --dataset data/ --output-dir runs/exp1
cy test   --model yolov5s_sac --weights runs/exp1/weights/best.pt --dataset data/ --output-dir runs/exp1/test
cy run    --config experiments/models/yolov5s.yaml --output-dir runs/exp1
cy compose --config experiments/all_models_direct.yaml --output-dir runs/batch1 --max-parallel 2
```

`run` 命令支持通过 `--test-only` 在不重新训练的情况下重新评估已有检查点。
