# X-Trainer 部署 LingBot-VLA 2.0 手册

版本：V1.0  
日期：2026-08-17  
适用代码：`GC-SHIRO/X-Trainer-LingBot-VLA2.0` `main`，commit `c2507c08677c190945e3a39eede95f918c3269af`

> 本手册按仓库当前代码编写。代码中存在但没有 X-Trainer 专用配置或验证入口的能力，会明确标记为“未交付”或“需验证”。

---

## 1. 文档目标

本文档指导用户在 Dobot X-Trainer 双臂平台上完成 LingBot-VLA 2.0 的完整部署链路：

1. 检查 X-Trainer 硬件，完成遥操作和示教数据采集。
2. 将原始数据转换为 LingBot-VLA 训练所需的 LeRobot v2.1 数据集。
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
  -> X-Trainer 字段映射与 delta action 转换
  -> normalization statistics
  -> LingBot-VLA 2.0 全参数微调
  -> safetensors checkpoint
  -> WebSocket policy server
  -> X-Trainer real client
  -> 真实机器人闭环验证
```

### 2.2 项目分工

| 模块 | 项目或目录 | 职责 |
|---|---|---|
| 控制与原始采集 | `dobot_xtrainer` / Pi0.5 配套采集项目 | follower、leader、夹爪与 RealSense 连接，遥操作和 raw episode 采集。 |
| 数据转换 | Pi0.5 的 X-Trainer 转换链路 | 将 raw episode 转换为 LeRobot v2.1；LingBot 仓库不包含 raw 转换脚本。 |
| 训练与模型 | 本仓库 | 数据映射、norm stats、LingBot-VLA 全参训练、checkpoint 导出和离线评估。 |
| 远程推理 | `scripts/serve_policy.py` | 加载 checkpoint，提供 WebSocket policy 服务和健康检查。 |
| 真机执行 | `scripts/run_xtrainer_real.py` | 采集三路图像和 14 维状态，消费 action chunk 并控制真机。 |

### 2.3 数据契约

原始 LeRobot 样本必须包含：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `observation.state` | `(14,)` | 左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。 |
| `action` | `(14,)` | 顺序与 `observation.state` 相同。 |
| `observation.images.top` | image/video | 顶部相机 RGB 图像。 |
| `observation.images.left_wrist` | image/video | 左腕相机 RGB 图像。 |
| `observation.images.right_wrist` | image/video | 右腕相机 RGB 图像。 |
| `task` | string | 自然语言任务描述。 |

[`configs/robot_configs/xtrainer.yaml`](configs/robot_configs/xtrainer.yaml) 将 12 个机械臂关节映射为 `arm.position`，将两个夹爪映射为 `effector.position`。机械臂 action 会减去当前 state，转换为 joint delta；夹爪 action 保持 absolute。三路原始相机字段在模型内部映射为 `camera_top`、`camera_wrist_left`、`camera_wrist_right`。

---

## 3. 前置条件

### 3.1 硬件

| 硬件 | 数量 | 用途 |
|---|---:|---|
| Dobot follower 机械臂 | 2 | 左右从臂执行动作。 |
| X-Trainer leader 主手 | 2 | 人类遥操作输入。 |
| Feetech / X-Trainer 夹爪 | 2 | 左右夹爪控制。 |
| Intel RealSense 相机 | 3 | 顶部、左腕、右腕图像。 |
| GPU 服务器 | 1 | 训练和 policy server。 |
| Linux 机器人控制机 | 1 | 连接机器人、夹爪、相机并运行真机客户端。 |

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

| 组件 | 版本或要求 |
|---|---|
| OS | Ubuntu 24.04 LTS x86_64 |
| Python | 3.12 |
| NVIDIA Driver | `>= 570.26` |
| PyTorch | 2.8.0 + CUDA 12.8 wheels |
| Transformers | 4.57.3 |
| Hugging Face Hub | 0.34.3 |
| FlashAttention | 2.8.3 |
| LeRobot Python 包 | 0.4.2 |
| GPU | Compute Capability 8.0 或更高 |

LeRobot Python 包版本 `0.4.2` 与 LeRobot 数据集格式 `v2.1` 是两个概念，不要混用版本号。

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

| 资产 | 默认来源 |
|---|---|
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct` |
| LingBot-VLA 2.0 6B | `robbyant/lingbot-vla-v2-6b` |
| MoGe-2 | `Ruicheng/moge-2-vitb-normal` |

可覆盖 LingBot 仓库：

```bash
LINGBOT_REPOSITORY=<owner/repository> bash tools/download_base_models.sh
```

注意：当前下载脚本写入 `tools/models/`，而 [`configs/vla/xtrainer/xtrainer.yaml`](configs/vla/xtrainer/xtrainer.yaml) 默认读取仓库根目录下的 `./models/`。训练前必须选择一种方式统一路径：

1. 将 YAML 中所有 `./models/...` 改成 `./tools/models/...`；或
2. 将完整模型目录放到仓库根目录的 `models/`。

还需同时核对 tokenizer、MoRGBD、depth 和 Video-DINO 路径，不能只修改 `model.model_path`。

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

## 6. 转换为 LeRobot v2.1

LingBot 训练入口读取 LeRobot 数据集，不能直接读取 X-Trainer raw episode。建议复用 Pi0.5 已验证的转换脚本，并保证以下映射：

| raw 来源 | LeRobot 字段 | 要求 |
|---|---|---|
| `joint_positions` | `observation.state` | `float32`，严格 14 维。 |
| `control` | `action` | `float32`，严格 14 维。 |
| `topImg` | `observation.images.top` | RGB，帧号对齐。 |
| `leftImg` | `observation.images.left_wrist` | RGB，帧号对齐。 |
| `rightImg` | `observation.images.right_wrist` | RGB，帧号对齐。 |
| task 参数 | `task` | 每个 episode 保存语言任务。 |

转换后至少检查：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("<repo_id>", root="/path/to/lerobot_dataset")
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

对于已生成的 X-Trainer 视频数据集，可在不改动源数据的前提下创建相机方向校正副本。顶视和左手腕保持不变，右手腕上下加左右翻转：

```bash
python tools/transform_xtrainer_dataset_images.py \
  --input-root /data/xtrainer_dataset_original \
  --output-root /data/xtrainer_dataset_camera_aligned
```

可先增加 `--dry-run` 验证输入；输出目录已存在时必须显式指定 `--overwrite-output`。

---

## 7. X-Trainer 数据配置

关键文件：

```text
configs/robot_configs/xtrainer.yaml
configs/vla/xtrainer/xtrainer.yaml
configs/vla/norm_compute/post_data.yaml
```

训练前必须修改：

| 配置 | 说明 |
|---|---|
| `model.model_path` | LingBot-VLA base 模型目录。 |
| `model.tokenizer_path` | Qwen3-VL tokenizer/model 目录。 |
| `data.train_path` | LeRobot 数据集路径或数据清单。 |
| `data.norm_stats_file` | 本数据集对应的 norm stats。 |
| `train.output_dir` | checkpoint 输出目录。 |
| `train.align_params.depth.*` | MoGe/MoRGBD 权重路径。 |
| `train.align_params.video.*` | Video-DINO 权重与配置路径。 |

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

## 9. 全参数微调

### 9.1 默认训练策略

[`configs/vla/xtrainer/xtrainer.yaml`](configs/vla/xtrainer/xtrainer.yaml) 当前主要设置：

| 参数 | 默认值 |
|---|---:|
| 精度 | BF16；`enable_fp32: false` |
| 数据并行 | FSDP2 full shard |
| Gradient checkpointing | 开启 |
| Optimizer | Muon |
| Learning rate | `5e-5`，constant |
| Micro batch size | `1` |
| Gradient accumulation | `1` |
| Max steps | `20000` |
| Save interval | `5000` |
| Hugging Face 权重导出 | 开启，异步保存 |
| `torch.compile` | 默认关闭 |

`global_batch_size` 按下式计算：

```text
micro_batch_size * data_parallel_size * gradient_accumulation_steps
```

### 9.2 配置检查

先逐项修改 YAML 中的占位路径：

```bash
grep -nE '/path/to|\./models' configs/vla/xtrainer/xtrainer.yaml
```

确保命令没有输出未处理的 `/path/to/...`。`./models` 若保留，则对应资产必须实际位于仓库根目录 `models/`。

### 9.3 Smoke training

首次仅运行少量 step，并写入独立目录：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --train.output_dir /path/to/save_ckpt/xtrainer_smoke \
  --train.max_steps 10 \
  --train.save_steps 10
```

如果单机可见 GPU 数与上例不同，修改 `CUDA_VISIBLE_DEVICES`。smoke test 应确认：数据能读取、三路视频能解码、norm stats 能加载、loss 有限、反向传播无 OOM、最终能导出 checkpoint。

### 9.4 正式训练

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

### 9.5 Checkpoint 检查

用于推理的 checkpoint 目录必须包含一个或多个 `.safetensors` 文件。推理代码还会按以下关系寻找训练配置：

```text
Path(model_path).parent.parent.parent / "lingbotvla_cli.yaml"
```

启动服务前检查：

```bash
find /path/to/checkpoint -maxdepth 1 -name '*.safetensors' -print
```

若训练配置不在代码要求的位置，应调整 checkpoint 目录或训练产物布局，不能只复制 `.safetensors` 文件。

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
  --use-length 50
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model-path` | 必填 | 含 `.safetensors` 的 checkpoint 目录。 |
| `--robot` | `xtrainer` | `configs/robot_configs` 下的配置名。 |
| `--norm-path` | 配置中的路径 | 覆盖 norm stats。 |
| `--host` | `0.0.0.0` | 监听地址。 |
| `--port` | `8000` | HTTP/WebSocket 端口。 |
| `--use-length` | `50` | 每次返回的 action 数量。 |
| `--num-steps` | `10` | flow-matching denoising steps。 |
| `--step-mode` | 关闭 | 每次只返回一个 action；真机 chunk client 不使用。 |
| `--fp32` | 关闭 | 默认 BF16，启用后使用 FP32。 |
| `--compile` | 关闭 | 启用 `torch.compile`。先完成 eager smoke test。 |

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
  --host <POLICY_SERVER_IP> \
  --port 8000 \
  --task "<TRAINING_TASK_PROMPT>" \
  --camera-top-serial <TOP_SERIAL> \
  --camera-left-wrist-serial <LEFT_WRIST_SERIAL> \
  --camera-right-wrist-serial <RIGHT_WRIST_SERIAL> \
  --action-horizon 25 \
  --control-hz 10 \
  --max-steps 100
```

首次真实模型测试先使用 10Hz 和较小 `max-steps`；验证稳定后再逐步提高到默认 20Hz。

### 13.5 安全与平滑默认值

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--max-joint-delta` | `0.17` | 大幅目标变化触发平滑处理。 |
| `--ramp-step` | `0.01` | 平滑过渡步长。 |
| `--ramp-max-steps` | `100` | 平滑过渡最大步数。 |
| `--gripper-update-threshold` | `0.02` | 夹爪最小更新阈值。 |
| `--servo-step-limit` | `0.9` | follower 单步限制。 |
| `--max-switch-delta` | `0.12` | chunk 边界触发混合的阈值。 |
| `--switch-blend-steps` | `5` | chunk 边界混合步数。 |
| `--max-delta-per-step` | `0` | 最终逐步限幅；`0` 表示关闭。 |

客户端会拒绝错误形状、NaN 和 Inf，但无法判断数值有效的动作是否会在真实场景中碰撞。急停看护不能被软件检查替代。

### 13.6 异步预取

默认 `--prefetch-remaining 8`。建议按实测推理延迟调整：

```text
prefetch_remaining >= ceil(平均推理秒数 * control_hz) + 安全余量
```

示例：

```bash
python scripts/run_xtrainer_real.py \
  --host <POLICY_SERVER_IP> \
  --task "<TRAINING_TASK_PROMPT>" \
  --camera-top-serial <TOP_SERIAL> \
  --camera-left-wrist-serial <LEFT_WRIST_SERIAL> \
  --camera-right-wrist-serial <RIGHT_WRIST_SERIAL> \
  --action-horizon 50 \
  --control-hz 20 \
  --prefetch-remaining 20 \
  --max-switch-delta 0.12 \
  --switch-blend-steps 5 \
  --max-delta-per-step 0.05 \
  --max-steps 100
```

设置 `--prefetch-remaining 0` 可关闭预取。完整调参说明见 [`docs/xtrainer_real_async_prefetch.md`](docs/xtrainer_real_async_prefetch.md)。

---

## 14. 常见问题

### 14.1 环境脚本通过，但训练找不到模型

原因通常是下载脚本输出 `tools/models/`，YAML 却读取 `./models/`。统一所有模型、tokenizer、depth 和 video 路径。

### 14.2 LeRobot 数据加载失败

检查数据集格式是否为 v2.1、环境是否为固定的 `lerobot==0.4.2`、路径是否指向数据集根目录，以及五个必要 observation/action 字段是否存在。

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

调整 `prefetch-remaining`、`max-switch-delta`、`switch-blend-steps` 和 `max-delta-per-step`。先记录推理延迟，再按延迟计算预取点。

---

## 15. 关键文件索引

| 文件 | 作用 |
|---|---|
| `tools/create_environment` | 创建固定版本的训练环境。 |
| `tools/download_base_models.sh` | 下载 Qwen3-VL、LingBot-VLA 和 MoGe-2。 |
| `configs/robot_configs/xtrainer.yaml` | X-Trainer 字段、delta action 和相机映射。 |
| `configs/vla/xtrainer/xtrainer.yaml` | X-Trainer 全参训练配置。 |
| `scripts/compute_norm_stats.py` | 计算 normalization statistics。 |
| `tasks/vla/train_lingbotvla.py` | 训练入口。 |
| `scripts/open_loop_eval.py` | 离线开环评估。 |
| `scripts/serve_policy.py` | WebSocket policy server。 |
| `scripts/serve_mock_policy.py` | 不加载模型的保持姿态策略。 |
| `scripts/run_xtrainer_real.py` | 真机推理客户端。 |
| `tests/run_xtrainer_basic_control.py` | 逐关节和夹爪基础测试。 |
| `deploy/xtrainer_real/README.md` | 真机客户端专项说明。 |
| `docs/xtrainer_real_async_prefetch.md` | 异步 action chunk 预取调参。 |
| `lingbotvla/utils/lora_utils.py` | 通用 LoRA 工具；不等于 X-Trainer LoRA 已交付。 |

---

## 16. 最小验收清单

部署完成前逐项确认：

1. 记录了仓库 commit 和实际环境版本。
2. `tools/create_environment` 成功，PyTorch 能识别全部训练 GPU。
3. 基础模型、tokenizer、depth 和 video 路径与 YAML 一致。
4. follower、leader、夹爪和三路 RealSense 均通过独立检查。
5. 遥操作和 raw episode 采集正常。
6. raw 数据成功转换为 LeRobot v2.1。
7. 数据集样本含 14 维 state/action、三路图像和 task。
8. 数据集中无 NaN/Inf，三路图像与 state/action 对齐。
9. `assets/norm_stats/xtrainer.json` 由当前训练数据生成并可解析。
10. 全参 smoke training 完成，loss 有限且 checkpoint 可保存。
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
