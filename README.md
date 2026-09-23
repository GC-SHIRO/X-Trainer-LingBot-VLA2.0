# X-Trainer 部署 LingBot-VLA 2.0 手册

版本：V1.0日期：2026-08-17适用代码：`GC-SHIRO/X-Trainer-LingBot-VLA2.0` `main`，commit `c2507c08677c190945e3a39eede95f918c3269af`

> 本手册按仓库当前代码编写。代码中存在但没有 X-Trainer 专用配置或验证入口的能力，会明确标记为“未交付”或“需验证”。

---

## 1. 文档目标

本文档指导用户在 Dobot X-Trainer 双臂平台上完成 LingBot-VLA 2.0 的完整部署链路：

1. 检查 X-Trainer 硬件，完成遥操作和示教数据采集。
2. 将原始数据转换为 LeRobot v2.1，再用仓库内的官方脚本升级为当前训练环境所需的 v3.0。
3. 安装 LingBot-VLA 2.0 环境，准备基础模型并计算 normalization statistics。
4. 执行 X-Trainer 全参数微调，检查训练产物并进行离线开环评估。
5. 启动 WebSocket policy server，通过 mock client 和真机 client 验证闭环推理。

当前仓库已提供 X-Trainer 数据映射、norm stats、全参训练、离线评估、policy server、mock server、基础控制测试和真机客户端。仓库虽然包含通用 LoRA 工具，但没有 X-Trainer 专用 LoRA 配置与可复现命令，因此本版本不将 LoRA 列为已交付能力。

---

## 2. 总体架构

### 2.1 端到端流程

```text
X-Trainer 硬件配置
  -> 遥操作与原始 demonstration 采集
  -> LeRobot v2.1 数据集
  -> 官方脚本转换为 LeRobot v3.0 数据集
  -> X-Trainer 字段映射与 delta action 转换
  -> normalization statistics
  -> LingBot-VLA 2.0 全参数微调
  -> safetensors checkpoint
  -> WebSocket policy server
  -> X-Trainer real client
  -> 真实机器人闭环验证
```

### 2.2 项目分工


| 模块           | 项目或目录                            | 职责                                                                    |
| ---------------- | --------------------------------------- | ------------------------------------------------------------------------- |
| 控制与原始采集 | `dobot_xtrainer` / Pi0.5 配套采集项目 | follower、leader、夹爪与 RealSense 连接，遥操作和 raw episode 采集。    |
| 数据转换       | Pi0.5 的 X-Trainer 转换链路           | 将 raw episode 转换为 LeRobot v2.1；LingBot 仓库不包含 raw 转换脚本。   |
| 训练与模型     | 本仓库                                | 数据映射、norm stats、LingBot-VLA 全参训练、checkpoint 导出和离线评估。 |
| 远程推理       | `scripts/serve_policy.py`             | 加载 checkpoint，提供 WebSocket policy 服务和健康检查。                 |
| 真机执行       | `scripts/run_xtrainer_real.py`        | 采集三路图像和 14 维状态，消费 action chunk 并控制真机。                |

### 2.3 数据契约

原始 LeRobot 样本必须包含：


| 字段                             |        形状 | 含义                                       |
| ---------------------------------- | ------------: | -------------------------------------------- |
| `observation.state`              |     `(14,)` | 左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。 |
| `action`                         |     `(14,)` | 顺序与`observation.state` 相同。           |
| `observation.images.top`         | image/video | 顶部相机 RGB 图像。                        |
| `observation.images.left_wrist`  | image/video | 左腕相机 RGB 图像。                        |
| `observation.images.right_wrist` | image/video | 右腕相机 RGB 图像。                        |
| `task`                           |      string | 自然语言任务描述。                         |

[`configs/robot_configs/xtrainer.yaml`](configs/robot_configs/xtrainer.yaml) 将 12 个机械臂关节映射为 `arm.position`，将两个夹爪映射为 `effector.position`。机械臂 action 会减去当前 state，转换为 joint delta；夹爪 action 保持 absolute。三路原始相机字段在模型内部映射为 `camera_top`、`camera_wrist_left`、`camera_wrist_right`。

---

## 3. 前置条件

### 3.1 硬件


| 硬件                     | 数量 | 用途                                     |
| -------------------------- | -----: | ------------------------------------------ |
| Dobot follower 机械臂    |    2 | 左右从臂执行动作。                       |
| X-Trainer leader 主手    |    2 | 人类遥操作输入。                         |
| Feetech / X-Trainer 夹爪 |    2 | 左右夹爪控制。                           |
| Intel RealSense 相机     |    3 | 顶部、左腕、右腕图像。                   |
| GPU 服务器               |    1 | 训练和 policy server。                   |
| Linux 机器人控制机       |    1 | 连接机器人、夹爪、相机并运行真机客户端。 |

真机代码的默认地址为：

```text
左臂 follower: 192.168.5.1
右臂 follower: 192.168.5.2
左夹爪: /dev/ttyUSB1, ID 21
右夹爪: /dev/ttyUSB0, ID 22
```

默认值必须按现场接线核对，不能直接假设有效。

### 3.2 推荐软件环境

环境脚本以以下组合为基准：


| 组件              | 版本或要求                      |
| ------------------- | --------------------------------- |
| OS                | Ubuntu 22.04 / 24.04 LTS x86_64 |
| Python            | 3.12                            |
| NVIDIA Driver     | `>= 570.26`                     |
| PyTorch           | 2.8.0 + CUDA 12.8 wheels        |
| Transformers      | 4.57.3                          |
| Hugging Face Hub  | 0.34.3                          |
| FlashAttention    | 2.8.3                           |
| LeRobot Python 包 | 0.4.2                           |
| Weights & Biases  | 0.21.0                          |
| GPU               | Compute Capability 8.0 或更高   |

LeRobot Python 包版本与数据集格式是两个概念。当前环境固定的 `lerobot==0.4.2` 使用 v3.0 数据格式，不能直接读取 v2.1；已有 v2.1 数据请先按第 6 节转换。

推理建议至少 24GB 显存。全参数训练的实际显存取决于模型、图像配置和并行策略；仓库配置启用 FSDP full shard，建议从多张 80GB GPU 或同等级训练资源开始。任何显存估算都应通过本机 smoke training 复核。

---

## 4. 环境安装

### 4.1 获取代码

```bash
git clone https://github.com/GC-SHIRO/X-Trainer-LingBot-VLA2.0.git
cd X-Trainer-LingBot-VLA2.0
git rev-parse HEAD
```

正式复现时记录 commit，不要只记录分支名。

### 4.2 创建训练环境

环境脚本只管理 Conda/Python 依赖，不安装 NVIDIA 驱动、系统 CUDA toolkit、模型或数据集。

```bash
bash tools/create_environment --strict-system-check --recreate
conda activate lingbotvla
```

已有环境可使用：

```bash
bash tools/create_environment --resume
```

如已有匹配 Python、Torch 和 CUDA ABI 的 FlashAttention wheel：

```bash
FLASH_ATTN_WHEEL=/path/to/flash_attn.whl \
bash tools/create_environment --strict-system-check --recreate
```

最小验证：

```bash
python -c "import torch, transformers, lerobot; print(torch.__version__); print(torch.cuda.is_available()); print(transformers.__version__)"
nvidia-smi
```

### 4.3 下载基础模型

```bash
conda activate lingbotvla
bash tools/download_base_models.sh
```

脚本下载：


| 资产               | Hugging Face 默认来源         | ModelScope 默认来源             |
| -------------------- | ------------------------------- | --------------------------------- |
| Qwen3-VL           | `Qwen/Qwen3-VL-4B-Instruct`   | `Qwen/Qwen3-VL-4B-Instruct`     |
| LingBot-VLA 2.0 6B | `robbyant/lingbot-vla-v2-6b`  | `Robbyant/lingbot-vla-v2-6b`    |
| MoGe-2             | `Ruicheng/moge-2-vitb-normal` | 无官方发布，回退到 Hugging Face |

可选下载源：

```bash
# Hugging Face（默认）
bash tools/download_base_models.sh --source hf

# ModelScope
pip install -U modelscope
bash tools/download_base_models.sh --source modelscope

# 等价写法
bash tools/download_base_models_modelscope.sh

# 也可通过环境变量指定默认来源
MODEL_SOURCE=modelscope bash tools/download_base_models.sh
```

ModelScope 尚未发布 `Ruicheng/moge-2-vitb-normal`，因此 `--source modelscope` 仍会用 Hugging Face 客户端下载该权重（需同时安装 `huggingface_hub`）。如已有 ModelScope 上的等价镜像，可显式指定：

```bash
MOGE_MODELSCOPE_REPOSITORY=<owner/repository> \
bash tools/download_base_models.sh --source modelscope
```

可覆盖各仓库 ID：

```bash
LINGBOT_REPOSITORY=<owner/repository> bash tools/download_base_models.sh
QWEN_REPOSITORY=<owner/repository> \
MOGE_REPOSITORY=<owner/repository> \
bash tools/download_base_models.sh
```

下载脚本默认把权重写入**仓库根目录的 `models/`**，与 [`configs/vla/xtrainer/xtrainer.yaml`](configs/vla/xtrainer/xtrainer.yaml) 读取的 `./models/...` 一致，因此下载完成后无需再改 YAML。实际目录布局：

```text
<repo>/models/Qwen3-VL-4B-Instruct/
<repo>/models/lingbot-vla-v2-6b/
<repo>/models/MoRGBD/moge2-vitb-normal.pt
```

对应 YAML 中的 `model.model_path`、`model.tokenizer_path`、`align_params.depth.moge_path`、`align_params.depth.morgbd_path`、`align_params.video.ckpt_path` 和 `align_params.video.config_path`。脚本结束时打印的绝对路径可用于核对。

如需改到其他磁盘，可用 `MODELS_DIR` 覆盖，同时必须同步修改 YAML 中所有 `./models/...`：

```bash
MODELS_DIR=/data/models bash tools/download_base_models.sh
```

`train.sh` 会设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，路径不一致时不会自动联网兜底，所以这些路径必须逐项确认，不能只看 `model.model_path`。

---

## 5. X-Trainer 硬件、遥操作与采集

LingBot 仓库不包含 leader 遥操作和 raw episode 采集程序。此阶段复用 X-Trainer Pi0.5 数据链路，参考：

- [X-Trainer Pi0.5-JAX](https://github.com/Dobot-Edu/X-Trainer-Pi0.5-JAX)
- 配套 `dobot_xtrainer` 控制与采集项目

推荐顺序：

1. 扫描左右 leader 和左右夹爪串口。
2. 在标准初始姿态标定 leader offset。
3. 检查 top、left wrist、right wrist 三路 RealSense。
4. 启动左右 follower server。
5. 先完成低速遥操作 smoke test，再开始录制。
6. 每条 episode 从稳定初始场景开始，任务完成后立即结束。
7. 检查 observation 与三路图像的帧号和数量是否一致。

原始数据至少应提供 14 维 `joint_positions`、14 维 `control` 和三路同步 RGB 图像。训练 prompt 应固定，转换和真机推理使用相同或语义一致的描述。

硬件配置中的密码、Token、相机序列号和本地串口映射不应提交到公共仓库。

---

## 6. 数据转换：raw → LeRobot v2.1 → v3.0

### 6.1 raw 转 v2.1 与相机方向校正

LingBot 训练入口读取 LeRobot 数据集，不能直接读取 X-Trainer raw episode。建议复用 Pi0.5 已验证的转换脚本，并保证以下映射：


| raw 来源          | LeRobot 字段                     | 要求                        |
| ------------------- | ---------------------------------- | ----------------------------- |
| `joint_positions` | `observation.state`              | `float32`，严格 14 维。     |
| `control`         | `action`                         | `float32`，严格 14 维。     |
| `topImg`          | `observation.images.top`         | RGB，帧号对齐。             |
| `leftImg`         | `observation.images.left_wrist`  | RGB，帧号对齐。             |
| `rightImg`        | `observation.images.right_wrist` | RGB，帧号对齐。             |
| task 参数         | `task`                           | 每个 episode 保存语言任务。 |

相机方向校正工具仅支持 v2.1 视频布局，必须在升级 v3.0 前运行。需要校正时，可先创建副本，顶视和左手腕保持不变，右手腕上下加左右翻转：

```bash
python tools/transform_xtrainer_dataset_images.py \
  --input-root /data/xtrainer_dataset_original \
  --output-root /data/xtrainer_lerobot
```

可先增加 `--dry-run` 验证输入；输出目录已存在时必须显式指定 `--overwrite-output`。随后对输出副本执行第 6.2 节的升级。

### 6.2 使用官方脚本升级为 v3.0

仓库包含与 `lerobot==0.4.2` 对应的[官方转换脚本](tools/convert_dataset_v21_to_v30.py)，原样保留上游实现和许可证。先激活训练环境；假设 v2.1 数据位于 `/data/xtrainer_lerobot`：

```bash
conda activate lingbotvla
python tools/convert_dataset_v21_to_v30.py \
  --root /data \
  --repo-id xtrainer_lerobot \
  --push-to-hub false
```

该版本实际读取 `root/repo-id`，因此 `--root` 填父目录。转换成功后，原路径保存 v3.0 数据，原始数据保留在 `/data/xtrainer_lerobot_old`。`--push-to-hub false` 关闭默认的 Hub 上传。脚本会清理已有的 `_v30` 临时目录，并可能在重试时用 `_old` 恢复原目录；重试前检查这些目录。完整说明及上游来源见[转换工具说明](tools/convert_dataset_v21_to_v30.md)。

后续归一化统计、训练和评估统一使用转换后的数据集**绝对路径**。

### 6.3 验证 v3.0 数据

升级后至少检查：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("<repo_id>", root="/path/to/lerobot_dataset")
assert dataset.meta.info["codebase_version"] == "v3.0"
sample = dataset[0]
assert tuple(sample["observation.state"].shape) == (14,)
assert tuple(sample["action"].shape[-1:]) == (14,)
for key in (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
):
    assert key in sample
print("dataset contract ok", len(dataset))
```

如果本地 LeRobot API 的构造参数不同，以环境中固定的 `lerobot==0.4.2` 为准。某一路图像缺失或损坏时，应丢弃整帧或整条 episode，不能让三路图像与 state/action 错位。

---

## 7. X-Trainer 数据配置

关键文件：

```text
configs/robot_configs/xtrainer.yaml
configs/vla/xtrainer/xtrainer.yaml
configs/vla/norm_compute/post_data.yaml
```

训练前必须修改：


| 配置                         | 说明                            |
| ------------------------------ | --------------------------------- |
| `model.model_path`           | LingBot-VLA base 模型目录。     |
| `model.tokenizer_path`       | Qwen3-VL tokenizer/model 目录。 |
| `data.train_path`            | LeRobot 数据集路径或数据清单。  |
| `data.norm_stats_file`       | 本数据集对应的 norm stats。     |
| `train.output_dir`           | checkpoint 输出目录。           |
| `train.align_params.depth.*` | MoGe/MoRGBD 权重路径。          |
| `train.align_params.video.*` | Video-DINO 权重与配置路径。     |

`train.action_dim=55`、`max_action_dim=55` 和 `max_state_dim=55` 是模型内部的最大/填充维度，不代表 X-Trainer 原始动作变成 55 维。X-Trainer 外部数据契约仍是 14 维。

---

## 8. 计算 Normalization Statistics

### 8.1 单数据集

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py \
  ./configs/vla/norm_compute/post_data.yaml \
  --data.data_name xtrainer \
  --data.train_path /path/to/lerobot_dataset \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/xtrainer.json \
  --data.data_ratio_for_norm_compute 1
```

### 8.2 多数据集

创建清单，每行一个机器人配置名和数据集路径：

```text
xtrainer /path/to/lerobot_dataset_a
xtrainer /path/to/lerobot_dataset_b
```

然后运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py \
  ./configs/vla/norm_compute/post_data.yaml \
  --data.data_name multi \
  --data.train_path /path/to/datasets.txt \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/xtrainer.json \
  --data.data_ratio_for_norm_compute 1
```

生成后检查：

```bash
python -m json.tool assets/norm_stats/xtrainer.json >/dev/null
ls -lh assets/norm_stats/xtrainer.json
```

训练配置、离线评估和 policy server 必须使用同一份 norm stats。数据集变化后应重新计算，不能沿用其他任务的统计文件。

---

## 9. 训练：全参数微调与冻结 VLM

如果使用单张 GPU 冻结整个 VLM、只训练 action expert 及动作侧模块，请改用独立教程：[`docs/FROZEN_VLM_TRAINING.md`](docs/FROZEN_VLM_TRAINING.md)。不要只设置 `freeze_vision_encoder`，该参数只冻结视觉编码器，并不会冻结完整 VLM。

### 冻结 VLM 快速入口（单卡）

`--train.train_expert_only true` 会让整个视觉语言模型 `qwenvl` 保持评估模式并停止更新其参数，优化器跳过这些参数；动作专家和其余可训练的动作侧模块继续更新。这能减少梯度与优化器状态的显存占用，但模型权重仍需加载，实际显存需求以试跑为准。该模式使用完整 checkpoint 保存和推理流程。

先完成第 6–8 节的数据升级、路径配置和归一化统计，再运行 10 step 单卡试跑：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --data.train_path /data/xtrainer_lerobot \
  --data.norm_stats_file assets/norm_stats/xtrainer.json \
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

这里 `enable_mixed_precision false` 用于当前单卡代码路径直接按 BF16 加载模型，避免按 FP32 加载冻结的权重；不要直接套用到多卡 FSDP。确认 loss 有限、完成参数更新并导出 `.safetensors` 后，按[完整冻结训练教程](docs/FROZEN_VLM_TRAINING.md)设置正式训练步数和新输出目录。以下第 9.1 节起继续介绍全参训练。

### 9.1 默认训练策略

[`configs/vla/xtrainer/xtrainer.yaml`](configs/vla/xtrainer/xtrainer.yaml) 当前主要设置：


| 参数                   |                     默认值 |
| ------------------------ | ---------------------------: |
| 精度                   | BF16；`enable_fp32: false` |
| 数据并行               |           FSDP2 full shard |
| Gradient checkpointing |                       开启 |
| Optimizer              |                       Muon |
| Learning rate          |           `5e-5`，constant |
| Micro batch size       |                        `1` |
| Gradient accumulation  |                        `8` |
| Global batch size      |                       `64` |
| Max steps              |                    `20000` |
| Save interval          |                     `5000` |
| Hugging Face 权重导出  |             开启，异步保存 |
| `torch.compile`        |                   默认关闭 |
| W&B 日志               |  开启，与 TensorBoard 同源 |

上表数值按 **8 张 A100 40GB**（`data_parallel_size=8`，`ulysses_parallel_size=1`）填写。`global_batch_size` 不是自由参数，必须严格等于下式，否则 `TrainingArguments` 在启动时直接抛 `ValueError`：

```text
global_batch_size = micro_batch_size * data_parallel_size * gradient_accumulation_steps
                  = 1 * 8 * 8 = 64
```

`micro_batch_size` 保持 `1`：单样本含三路相机、未来帧对齐和 36 层 MoE，本身已占满一个 step 的显存，40GB 卡上没有翻倍空间。扩大有效 batch 只能靠 `gradient_accumulation_steps`，而训练循环是按 micro batch 逐个 `backward()` 的（`tasks/vla/train_lingbotvla.py`），所以累积只增加wall-clock 时间，不增加激活显存。改动 `gradient_accumulation_steps` 后必须同步 `global_batch_size`，否则启动即失败。

### 9.2 训练轮次与 batch size 的关系

配置固定使用 `max_steps` 作为唯一停止条件（`num_train_epochs: null`），因此**轮次是由 steps 反推出来的**。每 epoch 的优化步数为 `floor(N / global_batch_size)`，其中 `N` 为数据集样本数（`floor` 因为 `data.drop_last` 默认为 `true`）：

```text
有效轮次 = max_steps * global_batch_size / N

max_steps = ceil(目标轮次 * N / global_batch_size)
```

按 `global_batch_size = 64` 换算，便于把默认的 `max_steps: 20000` 调成目标轮次：


| 数据集样本数 N | 20000 步对应的轮次 | 想跑 2 轮时的 max_steps |
| ---------------- | -------------------: | ------------------------: |
| 50,000         |               25.6 |                    1563 |
| 200,000        |                6.4 |                    6250 |
| 640,000        |                2.0 |                   20000 |
| 1,280,000      |                1.0 |                   40000 |

X-Trainer 这类遥操作数据集规模通常在几千条 episode 量级，`max_steps: 20000` 很可能对应十几轮甚至更多，容易过拟合。**先量出 `N`（`len(dataset)` 或数据集 `meta/info.json` 的 `total_frames`），再按上表取 `max_steps`**，并保持 `save_steps` 能在一轮内至少落一次 checkpoint。

### 9.3 40GB 显存下的调整顺序

40GB 比手册推荐的 80GB 紧张，先按 smoke test 实测，再按以下顺序逐项调整，**每次只改一个变量**：

1. 先跑 9.5 的 smoke training 拿真实峰值显存，不要凭估算决定。
2. 确认 `enable_gradient_checkpointing: true` 生效（默认已开）。
3. OOM 时把 `enable_activation_offload` 改为 `true`：这是既有的显存兜底开关，代价是明显变慢。`micro_batch_size` 已经是 `1`，没有下降空间。
4. 仍 OOM 再考虑 `data.num_workers`（8 卡 × 4 worker，只影响数据加载，不影响显存）和检查是否误开了 `use_compile`。
5. 不要靠下调 `global_batch_size` 硬凑：它与 `gradient_accumulation_steps` 强绑定，改了要一起改；需要更小的有效 batch 时应同时降低二者。

本节所有数值都是配置层面的推导，**显存是否真的够必须由本机 smoke training 复核**。

### 9.4 配置检查

先逐项修改 YAML 中的占位路径：

```bash
grep -nE '/path/to|\./models' configs/vla/xtrainer/xtrainer.yaml
```

确保命令没有输出未处理的 `/path/to/...`。`./models` 若保留，则对应资产必须实际位于仓库根目录 `models/`。

### 9.5 Smoke training

首次仅运行少量 step，并写入独立目录。**用与正式训练相同的 8 张卡**，否则显存结论没有参考价值：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --train.output_dir /path/to/save_ckpt/xtrainer_smoke \
  --train.max_steps 10 \
  --train.save_steps 10
```

smoke test 应确认：数据能读取、三路视频能解码、norm stats 能加载、loss 有限、反向传播无 OOM、最终能导出 checkpoint（含 `.safetensors`）。

注意 `global_batch_size` 与卡数强绑定，**换卡做 smoke 时必须同步改 batch 参数**，否则启动即抛 `ValueError`。例如只想用 2 张卡试跑：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --train.output_dir /path/to/save_ckpt/xtrainer_smoke \
  --train.gradient_accumulation_steps 8 \
  --train.global_batch_size 16 \
  --train.max_steps 10 \
  --train.save_steps 10
```

2 卡的 FSDP 切分更粗，每卡静态显存（参数、梯度、优化器状态）约为 8 卡的 4 倍，而激活显存与卡数无关。所以 2 卡能跑通说明激活放得下，8 卡基本也没问题；但 2 卡 OOM 常常是静态显存造成的，不能据此判定 8 卡不行。显存结论只在目标卡数下才完全有效。

### 9.6 正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml
```

`train.sh` 使用当前所有可见 GPU 启动 `torchrun`，默认写日志到仓库根目录 `log.txt`。正式训练前应在 YAML 中设置唯一 `train.output_dir`，不要复用 smoke 目录。

多机训练可设置：

```bash
NNODES=2 NODE_RANK=<0-or-1> MASTER_ADDR=<rank0-ip> MASTER_PORT=62500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml
```

### 9.7 Checkpoint 检查

用于推理的 checkpoint 目录必须包含一个或多个 `.safetensors` 文件。推理代码还会按以下关系寻找训练配置：

```text
Path(model_path).parent.parent.parent / "lingbotvla_cli.yaml"
```

启动服务前检查：

```bash
find /path/to/checkpoint -maxdepth 1 -name '*.safetensors' -print
```

若训练配置不在代码要求的位置，应调整 checkpoint 目录或训练产物布局，不能只复制 `.safetensors` 文件。

### 9.8 W&B 日志

`configs/vla/xtrainer/xtrainer.yaml` 默认开启 `use_wandb: true`，由 rank 0 同时写两处：


| 后端        | 位置                                       | 内容                                                    |
| ------------- | -------------------------------------------- | --------------------------------------------------------- |
| TensorBoard | `<train.output_dir>/runs/`                 | 全部 scalar，以及 MoE 专家选择直方图/柱状图（Images）。 |
| W&B         | `train.wandb_project`（默认 `lingbotvla`） | 每个 step 的全部 scalar，tag 与 TensorBoard 一致。      |

首次使用先登录，或通过环境变量提供 API key：

```bash
wandb login
# 或者
export WANDB_API_KEY=<your-key>
```

常用覆盖方式：

```bash
# 关闭 W&B，回到纯 TensorBoard
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --train.use_wandb false

# 指定 project 与 run 名称
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --train.wandb_project xtrainer-vla \
  --train.wandb_name xtrainer-finetune
```

无外网的训练机可设置 `WANDB_MODE=offline` 先本地落盘，事后 `wandb sync` 补传。`wandb.init` 或 `wandb.log` 失败不会中断训练，只会退回 TensorBoard。MoE 专家选择直方图/柱状图仅写 TensorBoard，W&B 只有 scalar 曲线。

---

## 10. LoRA 状态

当前仓库包含 [`lingbotvla/utils/lora_utils.py`](lingbotvla/utils/lora_utils.py) 和 PEFT 依赖，通用工具默认参数包括 `rank=4`、`alpha=4`。但是当前 X-Trainer 训练 YAML、`tasks/vla/train_lingbotvla.py` 的公开训练流程和 README 没有给出一套已验证的 X-Trainer LoRA 接入配置。

因此当前结论是：

```text
通用 LoRA 工具存在
  != X-Trainer LingBot-VLA 2.0 LoRA 已交付
```

在补齐 target modules、冻结规则、adapter 保存/恢复、推理加载和真机验证前，不应发布虚构的 LoRA 命令。当前请使用第 9 节的全参数微调流程。

---

## 11. 离线开环评估

训练完成后，可在 LeRobot 验证 episode 上比较预测 action 与 ground truth，输出 MSE、MAE 和轨迹图：

```bash
python scripts/open_loop_eval.py \
  --model_path /path/to/checkpoint \
  --robo_name xtrainer \
  --norm_path assets/norm_stats/xtrainer.json \
  --data_path /path/to/lerobot_validation_dataset \
  --traj_ids 0 1 2 \
  --use_length 50 \
  --max_infer_time 10 \
  --use_bf16 \
  --save_plot_path ./open_loop_test
```

离线 MSE/MAE 只能用于排查数据和 checkpoint，不等价于真实任务成功率。

---

## 12. Policy Server

### 12.1 启动

```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
python scripts/serve_policy.py \
  --model-path /path/to/checkpoint \
  --robot xtrainer \
  --norm-path assets/norm_stats/xtrainer.json \
  --host 0.0.0.0 \
  --port 8000 \
  --use-length 50 \
  --log
```

`--log` 为可选参数；不传时不写日志。传入后，服务端将模型收到的原始 WebSocket 请求、模型直接返回的 action 和三路输入 PNG 写入服务端当前目录的 `./log/<时间戳>-server/`。

常用参数：


| 参数           |       默认值 | 说明                                                  |
| ---------------- | -------------: | ------------------------------------------------------- |
| `--model-path` |         必填 | 含`.safetensors` 的 checkpoint 目录。                 |
| `--robot`      |   `xtrainer` | `configs/robot_configs` 下的配置名。                  |
| `--norm-path`  | 配置中的路径 | 覆盖 norm stats。                                     |
| `--host`       |    `0.0.0.0` | 监听地址。                                            |
| `--port`       |       `8000` | HTTP/WebSocket 端口。                                 |
| `--use-length` |         `50` | 每次返回的 action 数量。                              |
| `--num-steps`  |         `10` | flow-matching denoising steps。                       |
| `--step-mode`  |         关闭 | 每次只返回一个 action；真机 chunk client 不使用。     |
| `--fp32`       |         关闭 | 默认 BF16，启用后使用 FP32。                          |
| `--compile`    |         关闭 | 启用`torch.compile`。先完成 eager smoke test。        |
| `--log`        |         关闭 | 将原始请求、模型输出和输入 PNG 写到当前目录的`log/`。 |

健康检查：

```bash
curl http://127.0.0.1:8000/healthz
```

当前协议没有 TLS 和服务端鉴权，只能部署在可信局域网或 VPN 内，禁止将 8000 端口直接暴露到公网。

---

## 13. 真机推理

### 13.1 控制端依赖

在机器人控制机：

```bash
python -m venv .venv-xtrainer-client
source .venv-xtrainer-client/bin/activate
pip install -r deploy/xtrainer_real/requirements.txt
```

### 13.2 基础硬件测试

清空工作空间，站在急停旁，确认每个关节正负 5 度均不会碰撞：

```bash
python tests/run_xtrainer_basic_control.py \
  --camera-top-serial <TOP_SERIAL> \
  --camera-left-wrist-serial <LEFT_WRIST_SERIAL> \
  --camera-right-wrist-serial <RIGHT_WRIST_SERIAL> \
  --yes
```

脚本依次测试 12 个关节的正负 5 度动作和两个夹爪。没有 `--yes` 时脚本拒绝移动硬件。

### 13.3 Mock 链路

先用保持当前位置的假策略验证 WebSocket、图像和控制循环：

```bash
# 终端 1
python scripts/serve_mock_policy.py --host 127.0.0.1 --port 8000 --horizon 50

# 终端 2
python scripts/run_xtrainer_real.py \
  --host 127.0.0.1 \
  --task "hardware integration test" \
  --camera-top-serial <TOP_SERIAL> \
  --camera-left-wrist-serial <LEFT_WRIST_SERIAL> \
  --camera-right-wrist-serial <RIGHT_WRIST_SERIAL> \
  --action-horizon 5 \
  --max-steps 10
```

### 13.4 真实模型

服务端必须保持 chunk 模式，即不要传 `--step-mode`，且 `--use-length >= --action-horizon`。

```bash
python scripts/run_xtrainer_real.py \
  --host 172.15.0.1 \
  --port 8000 \
  --task "put the blue cuboid into the blue box" \
  --camera-top-serial 409122273405 \
  --camera-left-wrist-serial 412622272997 \
  --camera-right-wrist-serial 412622271417 \
  --action-horizon 50 \
  --control-hz 30 \
  --max-steps 1000 \
  --chunk-blend-steps 6 \
  --image-jpeg-quality 85 \
  --log
```

真机端的 `--log` 同样可选。启用后，它在控制机当前目录创建 `./log/<时间戳>-real/`，记录实际发给服务端的原始 observation、服务端返回的原始 response、三路 PNG，以及最终下发给真机环境的每一步 action。两台机器分别写各自当前目录的 `log`；日志写入失败只会停用日志，不会中断推理或控制循环。

首次真实模型测试先使用 10Hz 和较小 `max-steps`；验证稳定后再逐步提高到默认 30Hz。

### 13.5 安全与平滑默认值


| 参数                         | 默认值 | 作用                                                           |
| ------------------------------ | -------: | ---------------------------------------------------------------- |
| `--max-joint-delta`          |  `inf` | 默认不改写 policy 的关节目标；显式设置有限值时才触发平滑处理。 |
| `--ramp-step`                | `0.01` | 平滑过渡步长。                                                 |
| `--ramp-max-steps`           |  `100` | 平滑过渡最大步数。                                             |
| `--gripper-update-threshold` |    `0` | 默认发送每次夹爪目标变化。                                     |
| `--servo-step-limit`         |  `inf` | 默认不限制 follower 的关节目标跳变。                           |
| `--chunk-blend-steps`        |    `6` | chunk 边界拼接偏移的衰减步数；`<=1` 关闭混合。                 |
| `--max-delta-per-step`       |    `0` | 最终逐步限幅；`0` 表示关闭。                                   |
| `--image-jpeg-quality`       |   `85` | 相机观测的 JPEG 传输质量；`0` 发送原始 ndarray。               |

客户端会拒绝错误形状、NaN 和 Inf，但无法判断数值有效的动作是否会在真实场景中碰撞。急停看护不能被软件检查替代。

### 13.6 Action chunk 执行与拼接

客户端不做异步预取。它完整执行当前 action chunk，执行完后读一次观测，把其中的**机械臂实测关节位置**作为 hold 下发，**夹爪保留上一条下发目标**，再同步请求下一段（同一份观测也用于这次推理请求，不额外多读相机）。夹爪被物体挡住时，不用实测开度覆盖抓紧目标，避免等待推理时降低夹持力。

下发实测位姿而不是重发上一个动作，是因为机械臂在段内被外力推动或下坠时，继续下发最后目标会一直"顶"着那个旧目标；下发实测位姿则停在它当前所在的位置，让新 chunk 从那里起规划。hold 同时成为下一段拼接偏移的起算点。

每段新 chunk 的第一批动作会做**拼接偏移衰减**：切换时计算一次

```text
blend_offset = 上一个已下发动作 - 新 chunk 的第一个动作
```

随后在前 `--chunk-blend-steps` 步内用 smoothstep 把该偏移衰减到 0：

```text
progress = (index + 1) / blend_steps
weight   = progress^2 * (3 - 2 * progress)
target[关节] += (1 - weight) * blend_offset[关节]
```

只有常量偏移被衰减，新轨迹自身的运动不受影响；首步保留约 92.6% 的偏移，第 `blend_steps` 步精确回到模型目标。**夹爪列（索引 6 和 13）永远不参与混合**，避免开合指令被过渡抹平。`--chunk-blend-steps 0` 或 `1` 关闭混合。

### 13.7 相机观测传输

客户端默认用 JPEG 编码三路相机观测再发送（`--image-jpeg-quality 85`），把每次请求约 2.76 MB 的原始 ndarray 载荷压到十分之一量级——实测 640×480 合成画面约 29x，即使纯噪声这种最坏情况仍有约 3.5x。服务端在送入模型前解码回原始分辨率和 RGB 顺序，因此模型输入不变。

编码通过握手元数据协商：只有服务端在 `image_encodings` 中声明 `jpeg_rgb` 时客户端才压缩，对旧服务端自动回退为原始 ndarray。`--image-jpeg-quality 0` 可显式关闭。

---

## 14. 常见问题

### 14.1 环境脚本通过，但训练找不到模型

下载脚本默认写入仓库根目录 `models/`，与 YAML 的 `./models/...` 一致；出现该问题通常是因为手工移动过模型目录、在仓库外的目录执行了脚本，或改过 YAML 路径。逐项确认 `model.model_path`、`model.tokenizer_path`、`align_params.depth.moge_path`、`align_params.depth.morgbd_path`、`align_params.video.ckpt_path` 和 `align_params.video.config_path` 是否指向实际存在的文件。`train.sh` 设置了 `HF_HUB_OFFLINE=1`，不会联网兜底。

### 14.2 LeRobot 数据加载失败

检查数据集格式是否为 v3.0、环境是否为固定的 `lerobot==0.4.2`、绝对路径是否指向数据集根目录，以及五个必要 observation/action 字段是否存在。若仍为 v2.1，先执行第 6.2 节的官方转换脚本。

### 14.3 Norm stats 报错或动作异常

重新检查 state/action 是否为 14 维、是否包含 NaN/Inf、三路视频是否对齐。训练、评估和推理必须使用同一数据集生成的 `xtrainer.json`。

### 14.4 训练 OOM

按顺序尝试：降低 `micro_batch_size`、保持 gradient checkpointing、增加 gradient accumulation、确认 FSDP full shard 生效、关闭 compile、启用 activation offload，或增加 GPU。每次只改一个变量并重新做 smoke test。

### 14.5 Policy server 找不到 `lingbotvla_cli.yaml`

服务端按 `Path(model_path).parent.parent.parent` 查找该文件。检查 checkpoint 目录层级，并保留训练输出的完整结构。

### 14.6 服务端健康检查失败

确认进程监听 `0.0.0.0:8000`、防火墙只对控制机放行 TCP 8000，并从控制机使用服务端实际 IP，不能使用控制机自己的 `127.0.0.1`。

### 14.7 真机动作方向或幅度异常

停止执行并检查：相机视角、prompt、norm stats、左右臂顺序、夹爪方向、checkpoint 和训练数据是否匹配。不要用扩大限幅的方式掩盖映射错误。

### 14.8 Action chunk 不连续

先看服务端延迟：每次推理的 `server_timing.infer_ms` 会打到客户端日志。`infer_ms` 偏大时，chunk 之间的停顿会明显变长，可在服务端降低 `--num-steps`（10 → 4~6）压缩推理时间。仍抖动时再调整 `--chunk-blend-steps`（增大到 8~10 让拼接更缓）和 `--max-delta-per-step`。`--chunk-blend-steps 0` 可对比确认抖动是否来自混合本身。

---

## 15. 关键文件索引


| 文件                                       | 作用                                                           |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `tools/create_environment`                 | 创建固定版本的训练环境。                                       |
| [`tools/convert_dataset_v21_to_v30.py`](tools/convert_dataset_v21_to_v30.py) | 官方 v2.1 → v3.0 转换脚本；[使用说明](tools/convert_dataset_v21_to_v30.md)。 |
| `tools/download_base_models.sh`            | 下载 Qwen3-VL、LingBot-VLA 和 MoGe-2（支持 HF / ModelScope）。 |
| `tools/download_base_models_modelscope.sh` | 上述脚本的 ModelScope 便捷入口。                               |
| `configs/robot_configs/xtrainer.yaml`      | X-Trainer 字段、delta action 和相机映射。                      |
| `configs/vla/xtrainer/xtrainer.yaml`       | X-Trainer 全参训练配置。                                       |
| `docs/FROZEN_VLM_TRAINING.md`              | 单卡冻结 VLM 的训练教程与验证方法。                            |
| `scripts/compute_norm_stats.py`            | 计算 normalization statistics。                                |
| `tasks/vla/train_lingbotvla.py`            | 训练入口。                                                     |
| `scripts/open_loop_eval.py`                | 离线开环评估。                                                 |
| `scripts/serve_policy.py`                  | WebSocket policy server。                                      |
| `scripts/serve_mock_policy.py`             | 不加载模型的保持姿态策略。                                     |
| `scripts/run_xtrainer_real.py`             | 真机推理客户端。                                               |
| `tests/run_xtrainer_basic_control.py`      | 逐关节和夹爪基础测试。                                         |
| `deploy/xtrainer_real/README.md`           | 真机客户端专项说明。                                           |
| `deploy/image_codec.py`                    | 相机观测的 JPEG 传输编解码。                                   |
| `lingbotvla/utils/lora_utils.py`           | 通用 LoRA 工具；不等于 X-Trainer LoRA 已交付。                 |

---

## 16. 最小验收清单

部署完成前逐项确认：

1. 记录了仓库 commit 和实际环境版本。
2. `tools/create_environment` 成功，PyTorch 能识别全部训练 GPU。
3. 基础模型、tokenizer、depth 和 video 路径与 YAML 一致。
4. follower、leader、夹爪和三路 RealSense 均通过独立检查。
5. 遥操作和 raw episode 采集正常。
6. raw 数据先转换为 LeRobot v2.1，再升级为当前训练环境需要的 v3.0。
7. 数据集样本含 14 维 state/action、三路图像和 task。
8. 数据集中无 NaN/Inf，三路图像与 state/action 对齐。
9. `assets/norm_stats/xtrainer.json` 由当前训练数据生成并可解析。
10. 所选模式（全参或冻结 VLM）的 smoke training 完成，loss 有限且 checkpoint 可保存。
11. 正式训练导出了 `.safetensors` 和配套 `lingbotvla_cli.yaml`。
12. 离线开环评估可输出 MSE、MAE 和轨迹图。
13. policy server 能加载 checkpoint，`/healthz` 正常。
14. 基础控制测试逐关节和夹爪通过。
15. mock server 与真机客户端链路通过。
16. 真实模型返回 `(H, 14)` action chunk，无 NaN/Inf。
17. 首次低频、短 episode 测试动作平滑，急停有效。
18. 在固定测试条件下记录任务次数、成功次数和失败分类。

在没有真实运行证据前，不将 LoRA、显存占用、推理延迟或任务成功率标记为已验证结果。

---

## 17. License 与安全说明

本仓库使用 Apache-2.0 License。`deploy/xtrainer_real/scservo_sdk` 来源代码没有在该目录单独声明上游许可证；单独再分发前需要核对原 SDK 条款。

真实机器人运行具有碰撞和设备损坏风险。首次测试必须清空工作空间、降低控制频率和最大步数，并由人员在急停旁全程看护。
