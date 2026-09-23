# X-Trainer 冻结 VLM 训练教程

本文说明如何在单张 NVIDIA GPU 上冻结 LingBot-VLA 2.0 的完整视觉语言模型（VLM），只训练 action expert 和动作侧模块。教程适用于 Ubuntu 22.04 / 24.04 LTS x86_64。

> 这不是全参数微调，也不是 LoRA。仓库已有 `train.train_expert_only` 开关，不需要修改模型源码。显存占用仍受 GPU 型号、三路图像、未来帧监督和数据长度影响；正式训练前必须先跑 10 step smoke test。

## 1. 冻结范围

启用 `--train.train_expert_only true` 后，当前代码会：

- 将整个 `qwenvl` 切换到 `eval` 模式；
- 将 `qwenvl` 的所有参数设为 `requires_grad=False`；
- 让优化器跳过被冻结的参数，不为它们创建梯度和优化器状态；
- 继续训练 action expert、state/action projection 和启用的对齐头等非 VLM 模块。

`freeze_vision_encoder: true` 只冻结视觉编码器，不会冻结 VLM 的语言模型部分，因此不能替代 `train_expert_only`。

训练保存的仍是可供现有推理链加载的完整 checkpoint，不需要给 `scripts/serve_policy.py` 增加 adapter 加载逻辑。

## 2. 前置条件

推荐先准备：

- Ubuntu 22.04 或 24.04 LTS x86_64；
- NVIDIA Driver `>= 570.26`；
- Compute Capability 8.0 或更高的 NVIDIA GPU；
- Miniconda 或 Anaconda；
- 已升级为 LeRobot v3.0 的 X-Trainer 数据集（当前 `lerobot==0.4.2` 环境不能直接读取 v2.1）；
- 与该数据集对应的 normalization statistics。

Ubuntu 22.04 可先安装基础系统工具：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential
```

默认会安装预编译的 FlashAttention wheel，不要求系统安装 CUDA toolkit。只有传入 `--force-build-flash-attn` 时才需要 CUDA 12.8 toolkit、`nvcc`、`gcc` 和 `g++`。

## 3. 创建环境

在仓库根目录执行：

```bash
bash tools/create_environment --strict-system-check --recreate
conda activate lingbotvla
```

环境脚本会正式接受 Ubuntu 22.04（glibc 2.35）和 Ubuntu 24.04（glibc 2.39）。它只管理 Conda/Python 依赖，不安装显卡驱动、系统 CUDA toolkit、模型或数据集。

已有同名环境时可原地核对并补齐依赖：

```bash
bash tools/create_environment --resume
```

完成后检查：

```bash
python -c "import torch, transformers, lerobot; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(transformers.__version__)"
nvidia-smi
```

## 4. 下载基础模型

Hugging Face：

```bash
conda activate lingbotvla
bash tools/download_base_models.sh --source hf
```

中国大陆网络环境也可使用 ModelScope。MoGe-2 默认仍从 Hugging Face 下载，因为目前配置没有指定 ModelScope 镜像：

```bash
pip install -U modelscope
bash tools/download_base_models.sh --source modelscope
```

下载结果默认位于：

```text
models/Qwen3-VL-4B-Instruct/
models/lingbot-vla-v2-6b/
models/MoRGBD/moge2-vitb-normal.pt
```

这些路径与 `configs/vla/xtrainer/xtrainer.yaml` 的默认模型路径一致。若使用 `MODELS_DIR=/data/models` 改到其他磁盘，必须同步修改 YAML 中所有 `./models/...` 路径。

## 5. 计算 norm stats 并准备配置

如果现有数据还是 v2.1，先在已激活的训练环境中运行官方转换脚本。假设数据位于 `/data/xtrainer_lerobot`：

```bash
python tools/convert_dataset_v21_to_v30.py --root /data --repo-id xtrainer_lerobot --push-to-hub false
```

`--root` 是父目录。成功后原路径为 v3.0，原始数据保留在同级 `xtrainer_lerobot_old`。重试的目录处理及相机方向校正顺序见[转换工具说明](../tools/convert_dataset_v21_to_v30.md)；数据验证步骤见 [README 第 6 节](../README.md)。完成后再计算下面的统计量。

先确定统计文件的保存位置，并修改 `configs/vla/xtrainer/xtrainer.yaml` 中的占位路径：

```yaml
data:
  train_path: /data/xtrainer_lerobot
  norm_stats_file: /data/norm_stats/xtrainer.json

train:
  output_dir: /data/checkpoints/xtrainer_expert_only
```

同时核对 `model.model_path`、`model.tokenizer_path`、`align_params.depth.*` 和 `align_params.video.*` 指向真实存在的文件。

`norm stats`（normalization statistics，归一化统计量）记录数据集中 state/action 各维度的统计值，训练和推理都应使用同一份。第一次训练本数据集，或数据集发生变化时，先运行下面的命令计算。这里的 `data_name` 必须是 `configs/robot_configs/` 下存在的机器人配置名；X-Trainer 使用 `xtrainer`。`train_path` 指向单个 LeRobot 数据集目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py \
  ./configs/vla/norm_compute/post_data.yaml \
  --data.data_name xtrainer \
  --data.train_path /data/xtrainer_lerobot \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path /data/norm_stats/xtrainer.json \
  --data.data_ratio_for_norm_compute 1
```

`data_ratio_for_norm_compute 1` 表示使用完整数据集计算。命令完成后确认输出中有 `Writing stats to: /data/norm_stats/xtrainer.json`，并检查 JSON 可读且非空：

```bash
python -m json.tool /data/norm_stats/xtrainer.json >/dev/null
ls -lh /data/norm_stats/xtrainer.json
```

如果文件不存在、大小为 0，或计算过程中报数据集/字段错误，先处理该问题，不要开始训练。训练命令也必须加上 `--data.norm_stats_file /data/norm_stats/xtrainer.json`，明确读取刚生成的文件。


## 6. 单卡 smoke training

先只跑 10 个 optimizer step：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --data.norm_stats_file /data/norm_stats/xtrainer.json \
  --train.train_expert_only true \
  --train.enable_mixed_precision false \
  --train.micro_batch_size 1 \
  --train.gradient_accumulation_steps 8 \
  --train.global_batch_size 8 \
  --train.output_dir /data/checkpoints/xtrainer_expert_only_smoke \
  --train.max_steps 10 \
  --train.save_steps 10 \
  --train.use_wandb false
```

这里的 `enable_mixed_precision: false` 看起来有些反直觉，但对当前仓库的单卡路径是必要的：world size 为 1 时不会启用 FSDP mixed-precision policy；保留默认值 `true` 会先把模型提升为 FP32。设为 `false` 后，模型会直接按 BF16 加载，避免冻结的 VLM 仍以 FP32 常驻显存。多卡 FSDP 全参教程不要照搬这个设置。

batch 参数必须满足：

```text
global_batch_size = micro_batch_size × GPU 数 × gradient_accumulation_steps
                  = 1 × 1 × 8
                  = 8
```

梯度累积主要改变有效 batch 和训练时间，不会降低单个 forward/backward 的峰值显存。

## 7. 检查 smoke 结果

smoke test 至少应满足：

1. `log.txt` 中解析后的参数包含 `"train_expert_only": true` 和 `"enable_mixed_precision": false`。
2. 数据集、三路视频和 norm stats 均能加载。
3. loss 为有限值，没有 NaN/Inf。
4. 完成 backward 和 optimizer step，没有 CUDA OOM。
5. 输出目录生成 `global_step_10`，并成功导出 `.safetensors`。

可快速检查：

```bash
grep -E 'train_expert_only|enable_mixed_precision|Loss |CUDA out of memory|safetensors' log.txt
find /data/checkpoints/xtrainer_expert_only_smoke -maxdepth 3 -type f \
  \( -name '*.safetensors' -o -name 'metadata.json' \) -print
```

训练时另开终端观察真实峰值：

```bash
watch -n 1 nvidia-smi
```

## 8. 正式训练

确认 smoke 通过后，使用新的输出目录并去掉 smoke 的步数覆盖：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --data.norm_stats_file /data/norm_stats/xtrainer.json \
  --train.train_expert_only true \
  --train.enable_mixed_precision false \
  --train.micro_batch_size 1 \
  --train.gradient_accumulation_steps 8 \
  --train.global_batch_size 8 \
  --train.output_dir /data/checkpoints/xtrainer_expert_only
```

默认配置以 `max_steps: 20000` 为唯一停止条件。冻结训练的 global batch 从全参配置的 64 变为 8 后，相同 `max_steps` 对应的样本量也变为原来的八分之一。应按数据集样本数 `N` 和目标轮次重新计算：

```text
max_steps = ceil(目标轮次 × N / global_batch_size)
```

例如数据集有 80,000 帧、目标 2 轮、global batch 为 8，则约需 `ceil(2 × 80000 / 8) = 20000` step。相应调整 `save_steps`，保证每轮至少保存一次。

如需 W&B，将命令中的 `--train.use_wandb false` 去掉或改为 `true`，并提前执行 `wandb login`；离线服务器可设置 `WANDB_MODE=offline`。

## 9. OOM 与常见错误

### 启动时报 global batch 不一致

原因是 GPU 数、micro batch、梯度累积和 global batch 没有同步。单卡示例应保持 `1 × 1 × 8 = 8`。改成两卡时，同样的累积步数应设置 `global_batch_size=16`。

### 单卡仍然 OOM

先确认日志中 `enable_mixed_precision` 确实为 `false`，并确认没有其他进程占用显存。保持 `micro_batch_size=1` 和 gradient checkpointing 开启。降低 gradient accumulation 不会降低一次迭代的峰值显存。

如果仍然 OOM，不要直接删模型组件或关闭监督项来“跑通”；这会改变训练目标。应先记录峰值和报错阶段，再考虑增加 GPU、启用 activation offload，或单独评估关闭 future video/depth 辅助监督的效果。

### 设置了 freeze_vision_encoder 但显存仍高

`freeze_vision_encoder` 只冻结 VLM 的视觉编码器。完整冻结必须使用 `train_expert_only: true`。

### 下载完成但训练找不到模型

`train.sh` 会启用 Hugging Face 和 Transformers 离线模式。逐项检查 YAML 中所有模型路径，不能依赖运行时联网补齐。

## 10. 当前交付边界

- 本教程复用仓库已有冻结逻辑和全量 checkpoint 保存链路。
- Ubuntu 22.04 兼容已覆盖环境预检、glibc 2.35 和模型下载脚本。
- 本教程没有启用 LoRA；仓库中的通用 LoRA 工具尚不是 X-Trainer 的可复现训练入口。
- 48GB 单卡是合理的 smoke test 目标，不是无条件显存承诺。最终是否可训练以目标机器的 smoke test 为准。
