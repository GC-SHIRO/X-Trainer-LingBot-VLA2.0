# X-Trainer 部署 LingBot-VLA 2.0 手册

版本：V2.0
更新日期：2026-10-24
适用代码：`GC-SHIRO/X-Trainer-LingBot-VLA2.0` `main`

---

## 1. 文档目标

本文用于在 Dobot X-Trainer 双臂平台上完成 LingBot-VLA 2.0 的基础模型准备、后训练（全参微调或冻结 VLM 微调）和真机部署。

主流程：

```text
基础模型下载 -> 训练环境部署 -> raw 示教数据转换为 LeRobot v3.0 -> 机器人配置映射
-> 计算归一化统计 -> 模型微调 -> 策略服务 -> 真机执行
```

本文所有命令默认在仓库根目录、Linux/Bash 环境下执行，路径请替换为实际值。

**本仓库不包含数据采集代码。** 准备好 X-Trainer 的 raw 原始示教记录后，使用本仓库的 `tools/convert_raw_to_lerobot_3.py` 转换为 LeRobot v3.0，具体命令见第 6 节。

## 2. 总体架构

### 2.1 端到端流程

```text
X-Trainer raw 示教数据（collect_data/<episode_id>/）
    -> tools/convert_raw_to_lerobot_3.py           原始记录转换
    -> LeRobot v3.0 数据集目录
    -> configs/robot_configs/xtrainer.yaml        特征映射
    -> scripts/compute_norm_stats.py              归一化统计
    -> assets/norm_stats/xtrainer.json
    -> tasks/vla/train_lingbotvla.py              全参微调或冻结 VLM 微调
    -> <output_dir>/checkpoints/global_step_N/hf_ckpt
    -> scripts/serve_policy.py                    WebSocket 策略服务
    -> scripts/run_xtrainer_real.py               真机客户端
```

### 2.2 项目分工

| 模块 | 位置 | 作用 |
|---|---|---|
| 训练入口 | `tasks/vla/train_lingbotvla.py` | LingBot-VLA 2.0 后训练主程序。 |
| 分布式启动 | `train.sh` | `torchrun` 封装，自动探测可见 GPU 数并写 `log.txt`。 |
| 归一化统计 | `scripts/compute_norm_stats.py` | 统计 state/action 的 mean/std，生成 JSON。 |
| 数据转换 | `tools/convert_raw_to_lerobot_3.py` | raw 原始记录转 LeRobot v3.0，三路相机对齐、坏帧过滤。 |
| 数据加载 | `lingbotvla/data/vla_data/` | LeRobot v3.0 加载、多数据集拼接、特征映射。 |
| 机器人配置 | `configs/robot_configs/*.yaml` | raw 特征名到统一特征空间的映射。 |
| VLA 训练配置 | `configs/vla/<robot>/<robot>.yaml` | 模型、数据、并行、优化器、MoE、对齐参数。 |
| 策略封装 | `deploy/lingbot_vla_v2_policy.py` | `LingbotVLAv2Server`，加载 checkpoint 与归一化器。 |
| 服务端 | `deploy/websocket_policy_server.py` | WebSocket + MessagePack 服务，提供 `/healthz`。 |
| 客户端 | `deploy/websocket_client_policy.py` | `WebsocketClientPolicy`，读取服务端 metadata。 |
| 真机环境 | `deploy/xtrainer_real/` | Dobot、RealSense、Feetech 封装与最小真机客户端环境。 |
| 环境构建 | `tools/create_train_env.sh` | 创建 Python 3.12 conda 环境并锁定依赖。 |
| 模型下载 | `download_base_models.sh` / `.ps1` | 下载 Qwen3-VL、LingBot-VLA v2.6B、MoGe-2。 |
| 配置说明 | `docs/config/lingbotvla_config_doc.md`、`configs/vla/Training_Config.md` | 全部训练参数的含义与取值建议。 |

### 2.3 关键数据契约

X-Trainer 的状态和真实动作均为 **14 维**，顺序固定：

| 索引 | 内容 |
|---|---|
| `0-5` | 左臂关节 1-6，弧度 |
| `6` | 左夹爪，归一化到 `[0,1]` |
| `7-12` | 右臂关节 1-6，弧度 |
| `13` | 右夹爪，归一化到 `[0,1]` |

视觉字段固定为 `observation.images.top`、`observation.images.left_wrist`、`observation.images.right_wrist`，语言输入为 `task` 字符串。

`configs/robot_configs/xtrainer.yaml` 把这 14 维拆成统一特征空间：

| 统一特征 | 原始切片 | 维度 | 说明 |
|---|---|---|---|
| `observation.state.arm.position` | `observation.state[0:6)` + `[7:13)` | 12 | 双臂关节。 |
| `observation.state.effector.position` | `observation.state[6:7)` + `[13:14)` | 2 | 左右夹爪。 |
| `action.arm.position` | `action[0:6)` + `[7:13)` | 12 | **`subtract_state: true`**，学习相对当前状态的增量。 |
| `action.effector.position` | `action[6:7)` + `[13:14)` | 2 | **`subtract_state: false`**，学习绝对目标值。 |

也就是说：**12 维机械臂动作是相对当前 state 的 delta，两个夹爪动作保持 absolute**。这一点必须与训练时的配置保持一致，不要在推理侧自行改成绝对关节角。

相机映射：

| 统一名称 | 原始键 |
|---|---|
| `observation.images.camera_top` | `observation.images.top` |
| `observation.images.camera_wrist_left` | `observation.images.left_wrist` |
| `observation.images.camera_wrist_right` | `observation.images.right_wrist` |

## 3. 硬件和软件前置条件

### 3.1 硬件组成

| 硬件 | 数量 | 用途 |
|---|---|---|
| Dobot follower 机械臂 | 2 | 执行动作。 |
| leader 主手 | 2 | 输入遥操作动作（采集阶段）。 |
| Feetech / STS 夹爪 | 2 | 左右夹爪。 |
| Intel RealSense | 3 | 顶部、左腕、右腕图像。 |
| GPU 训练机 | 1 | 微调与策略服务。 |
| 机器人控制机 | 1 | 运行真机客户端，连接机械臂与相机。 |

### 3.2 网络和设备约定

```text
左臂 follower: 192.168.5.1
右臂 follower: 192.168.5.2
左夹爪串口:   /dev/ttyUSB1   舵机 ID 21
右夹爪串口:   /dev/ttyUSB0   舵机 ID 22
```

```bash
ping 192.168.5.1
ping 192.168.5.2
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
```

相机接线约定（由 `deploy/xtrainer_real/environment.py` 决定）：

- 顶部相机 `cam_top` 与左腕相机 `cam_left_wrist` 挂在**左臂** follower 上。
- 右腕相机 `cam_right_wrist` 挂在**右臂** follower 上。

`/dev/ttyUSB*` 的实际编号会随插拔顺序变化，首次上机必须逐个确认，不能按枚举顺序假定左右。

### 3.3 系统与依赖建议

- 操作系统：Ubuntu（`tools/create_train_env.sh` 为 bash 脚本；`scripts/compute_norm_stats.py` 内部使用 `multiprocessing` 的 `fork` 启动方式，**不适用于 Windows**）。
- Python 3.12，PyTorch 2.8.0 + CUDA。训练环境构建脚本会断言 `torch.cuda.is_available()`。
- 需要可用的 NVIDIA 驱动；`train.sh` 通过 `nvidia-smi -L` 统计 GPU 数量。
- flash-attn 2.8.3（默认从 pip 安装，也可用本地 wheel）。
- 训练侧依赖见 `requirements.txt` 与 `requirements-depth.txt`；真机客户端依赖见 `deploy/xtrainer_real/requirements.txt`（`pyrealsense2`、`pyserial`、`websockets`、`msgpack` 等，无需 torch）。

## 4. 环境部署

在仓库根目录执行：

```bash
bash tools/create_train_env.sh
conda activate lingbotvla
```

脚本参数：

| 参数 | 作用 |
|---|---|
| `--env-name NAME` | 自定义环境名，默认 `lingbotvla`。 |
| `--recreate` | 删除并重建同名环境。 |
| `--resume` | 向已存在的环境继续安装。 |
| `--flash-attn-wheel PATH` | 从本地 wheel 安装 flash-attn，避免现场编译。 |

脚本会按顺序完成：创建 Python 3.12 环境、安装 torch 2.8.0 栈、安装 `requirements.txt`、安装 `numpydantic==1.9.0`、安装 flash-attn、以 `--no-deps` 安装 `lerobot v0.4.2`、以可编辑方式安装本仓库、安装 `requirements-depth.txt`、可编辑安装 `lingbot-depth` 与 `MoGe`，最后把 `huggingface_hub` 固定到 `0.34.0`。

脚本不安装显卡驱动、不下载模型，也不修改串口权限。`lerobot` 与深度子包是有意用 `--no-deps` 安装的，因此脚本结尾的 `pip check` 可能报告依赖元数据问题，属预期现象。

机器人控制机只需要轻量客户端依赖：

```bash
pip install -r deploy/xtrainer_real/requirements.txt
```

## 5. 基础模型下载

需要先有新版 Hugging Face CLI：

```bash
pip install -U huggingface_hub
```

然后在仓库根目录执行：

```bash
bash download_base_models.sh
```

Windows 下可使用 `download_base_models.ps1`。

脚本下载三个目标到 `models/`：

| 目标 | 来源 | 落地路径 |
|---|---|---|
| Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` | `models/Qwen3-VL-4B-Instruct` |
| LingBot-VLA v2.6B | `robbyant/lingbot-vla-v2-6b` | `models/lingbot-vla-v2-6b` |
| MoGe-2 ViT-B Normal | `Ruicheng/moge-2-vitb-normal` 的 `model.pt` | `models/MoRGBD/moge2-vitb-normal.pt` |

LingBot 仓库内已包含 LingBot-Depth 与 Video-DINO 教师，无需单独下载。如需替换来源：

```bash
LINGBOT_REPOSITORY=<owner/repository> bash download_base_models.sh
```

也可以用仓库内的通用脚本单独拉取：

```bash
python scripts/download_hf_model.py --repo_id <owner/model> --local_dir <目录>
python scripts/download_hf_data.py  --repo_id <owner/dataset> --local_dir <目录>
```

部署端加载 base model 时，可以用 `QWEN3VL_PATH` 环境变量覆盖训练配置里记录的 tokenizer 路径。

**重要：`train.sh` 会强制打开离线模式**（`HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`）。因此基础模型和数据集必须在启动训练前就落到本地，否则训练会直接失败。

## 6. 数据准备（LeRobot 数据集）

LingBot-VLA 2.0 通过 LeRobot 的 `LeRobotDataset` 读取数据。本仓库固定使用的 `lerobot==0.4.2` 环境需要 **LeRobot v3.0 数据格式**：raw 原始记录使用下面的新脚本直接转换；已有 v2.1 数据集则按第 6.2 节升级。多个数据集可以在加载时拼接，无需预先合并文件。

每个数据集条目可以是 Hugging Face repo id，也可以是本地 LeRobot 数据集目录。至少需要包含：

| 字段 | 形状 | 含义 |
|---|---|---|
| `observation.state` | `(14,)` | 左臂 6 关节 + 左夹爪 + 右臂 6 关节 + 右夹爪 |
| `action` | `(14,)` | 维度顺序与 `observation.state` 相同 |
| `observation.images.top` | image | 顶部相机 |
| `observation.images.left_wrist` | image | 左腕相机 |
| `observation.images.right_wrist` | image | 右腕相机 |
| `task` | text | 自然语言任务描述 |

### 6.1 从 raw 原始记录生成 LeRobot v3.0

输入目录中，每个 episode（一次完整示教记录）包含三路相机图像和逐帧的机器人状态、动作：

```text
collect_data/<episode_id>/
  topImg/<frame_id>.jpg
  leftImg/<frame_id>.jpg
  rightImg/<frame_id>.jpg
  observation/<frame_id>.pkl  # joint_positions：状态；control：动作
```

在已安装训练依赖的环境中执行：

```bash
python tools/convert_raw_to_lerobot_3.py \
  --raw-root /data/collect_data \
  --output-root /data/xtrainer_lerobot_v3 \
  --task "Insert the test tube on the desktop into the rack." \
  --fps 30
```

`--task` 替换为这批示教的实际任务描述，`--fps` 设置为采集帧率。脚本按帧编号对齐状态与三路相机，默认跳过损坏帧、保留至少 10 个有效帧的记录，并将图像编码为 MP4。可用 `--skip-first-frames N` 跳过每段记录开头的 N 帧，用 `--min-frames N` 调整最少有效帧数；`--no-videos` 可改为将图像直接存入数据文件。

输出直接采用 v3.0 布局，无需再运行 v2.1 转换工具：

```text
xtrainer_lerobot_v3/
  data/chunk-000/file-000.parquet
  videos/observation.images.top/chunk-000/file-000.mp4
  videos/observation.images.left_wrist/chunk-000/file-000.mp4
  videos/observation.images.right_wrist/chunk-000/file-000.mp4
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
```

每段示教分别保存数据和视频文件；`meta/episodes` 记录每段示教的文件位置、帧范围、视频时间范围及统计量。输出目录默认必须为空，只有显式传入 `--overwrite-output` 才会删除并重建已有输出；输入、输出目录不能相互包含。转换完成后，将后续的 `data.train_path` 指向 `/data/xtrainer_lerobot_v3`，完成机器人和训练配置后，继续执行第 9 节的训练归一化统计计算。

### 6.2 已有 LeRobot v2.1 数据集

已有 v2.1 数据不需要重新读取 raw，使用仓库内的升级工具：

```bash
python tools/convert_dataset_v21_to_v30.py \
  --root /data \
  --repo-id xtrainer_lerobot \
  --push-to-hub false
```

这里输入数据集位于 `/data/xtrainer_lerobot`，`--root` 填其父目录。成功后原路径保存 v3.0 数据，旧数据保留在 `/data/xtrainer_lerobot_old`。更多说明见 [v2.1 → v3.0 转换说明](tools/convert_dataset_v21_to_v30.md)。

### 6.3 单个数据集

把 `data.data_name` 设为机器人配置名（`xtrainer`），`data.train_path` 指向数据集目录：

```yaml
data:
  datasets_type: vla
  data_name: xtrainer
  train_path: /path/to/lerobot_dataset
  robot_config_root: ./configs/robot_configs
```

### 6.4 多个数据集

把 `data.data_name` 设为 `multi`，`data.train_path` 指向一个**文本清单文件**。每个非空行两列，用空格分隔：

```text
xtrainer /path/to/lerobot_dataset_a
xtrainer /path/to/lerobot_dataset_b
```

`MultiVLADataset` 会在运行时逐行实例化并拼接，替代了早先"先合并数据集"的做法。

可参考 `assets/training_data/robotwin.txt` 的格式。

## 7. 机器人配置（`configs/robot_configs/xtrainer.yaml`）

机器人配置负责把 raw LeRobot 字段映射到统一特征空间，完整内容如下：

```yaml
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state: {start: 0, end: 6}
        - observation.state: {start: 7, end: 13}

  - observation.state.effector.position:
      origin_keys:
        - observation.state: {start: 6, end: 7}
        - observation.state: {start: 13, end: 14}

actions:
  - action.arm.position:
      origin_keys:
        - action: {start: 0, end: 6}
        - action: {start: 7, end: 13}
      subtract_state: true

  - action.effector.position:
      origin_keys:
        - action: {start: 6, end: 7}
        - action: {start: 13, end: 14}
      subtract_state: false

images:
  - observation.images.camera_top:
      origin_keys: observation.images.top
  - observation.images.camera_wrist_left:
      origin_keys: observation.images.left_wrist
  - observation.images.camera_wrist_right:
      origin_keys: observation.images.right_wrist

norm_stats: assets/norm_stats/xtrainer.json
```

规则：

- `states` 与 `actions` 结构相同，同一 raw 张量的多个切片按顺序拼接。
- `subtract_state: true` 表示该动作特征是"相对当前 state 的增量"，`false` 表示绝对动作。真实机器人数据推荐机械臂用 `true`、夹爪用 `false`。
- `images` 支持简写：raw 键与目标键同名时可以直接写一行 `- observation.images.camera_top`。
- `norm_stats` 指向归一化统计 JSON，训练时会读取该字段。
- 机器人配置中出现的 `<joint_type>`（`observation.state.<joint_type>` / `action.<joint_type>`）必须已经在 VLA 训练配置的 `data.joints` 中声明；相机名必须出现在 `data.cameras` 中。使用未声明的类型会在运行时抛出 `ValueError`。

本仓库另附 `configs/robot_configs/xtrainer_clothes.yaml`，它与 `xtrainer.yaml` 的唯一区别是 `norm_stats` 指向 `assets/norm_stats/xtrainer_clothes.json`，用于区分不同任务的统计量。

## 8. 训练配置（`configs/vla/xtrainer/xtrainer.yaml`）

训练配置决定模型结构、数据声明和训练超参。X-Trainer 相关部分的关键值：

```yaml
model:
  model_path: ./models/lingbot-vla-v2-6b
  tokenizer_path: ./models/Qwen3-VL-4B-Instruct
  config_key: LingbotVLAV2Config
  moe_implementation: fused

data:
  datasets_type: vla
  data_name: xtrainer
  train_path: /path/to/lerobot_dataset
  robot_config_root: ./configs/robot_configs
  joints:
    - arm.position: 14
    - effector.position: 2
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
  norm_stats_file: assets/norm_stats/xtrainer.json
  num_workers: 8
  use_future_image: true

train:
  output_dir: /path/to/save_ckpt
  action_dim: 55
  max_action_dim: 55
  max_state_dim: 55
  optimizer: muon
  lr: 5.0e-5
  micro_batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 40000
  save_steps: 40000
  enable_fp32: true
  enable_resume: true
  use_wandb: false
  use_compile: true
  ckpt_manager: dcp
```

必须保持一致的约束：

- `data.joints` 与 `data.cameras` 必须覆盖机器人配置用到的全部关节类型和相机名。`joints` 的数值是统一特征空间的宽度，应**大于等于**原始切片拼接后的维度（`arm.position` 原始为 12，这里给 14 留出对齐余量；`effector.position` 原始为 2，这里就是 2）。
- `data.norm_stats_file` 必须指向真实存在的 JSON。
- `train.action_dim` / `max_action_dim` / `max_state_dim` 由预训练模型的动作头决定，X-Trainer 沿用 `55`，不要为了 14 维数据而调小。
- 训练和部署必须使用同一个 `config_key: LingbotVLAV2Config`，服务端会据此选择 Qwen3-VL 分支。

批量大小由代码自动推导，无需手填：

```text
global_batch_size = micro_batch_size × data_parallel_size(GPU 数) × gradient_accumulation_steps
dataloader_batch_size = global_batch_size / data_parallel_size
```

如果显式传入的 `global_batch_size` 与推导值不一致，训练会直接报错并给出期望值。`configs/vla/real_robot/real_robot.yaml` 是官方给出的真实机器人训练示例，补齐了 `end.position` 等其他关节类型和完整 `norm_type`，多关节平台应以它为参照。

更多参数含义见 `docs/config/lingbotvla_config_doc.md` 和 `configs/vla/Training_Config.md`。

## 9. 计算归一化统计

训练前必须为数据集计算归一化统计，否则 `--data.norm_stats_file` 指向的文件不存在。

单个数据集：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --data.data_name xtrainer \
  --data.train_path /path/to/lerobot_dataset \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/xtrainer.json \
  --data.data_ratio_for_norm_compute 1
```

多个数据集：把 `--data.data_name` 设为 `multi`，`--data.train_path` 指向第 6.2 节的清单文件：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --data.data_name multi \
  --data.train_path assets/training_data/xtrainer.txt \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/xtrainer.json \
  --data.data_ratio_for_norm_compute 1
```

相关参数：

| 参数 | 说明 |
|---|---|
| `--data.train_path` | 单个数据集目录，或 `.txt` 清单文件（清单必须以 `.txt` 结尾）。 |
| `--data.robot_name` | 可选。只在清单中筛选指定机器人配置名对应的条目。 |
| `--data.norm_path` | 统计结果输出路径，应与机器人配置的 `norm_stats` 一致。 |
| `--data.data_ratio_for_norm_compute` | 采样比例。`1` 表示用全量数据；小于 `1` 时按固定种子抽样。 |
| `--data.norm_merge_chunk_dim` | 是否在统计 action 时合并 chunk 维度，默认 `true`。 |
| `--train.micro_batch_size` | 统计过程的批大小。`configs/vla/norm_compute/post_data.yaml` 中预设为 `48`。 |

仓库同时提供了 `configs/vla/norm_compute/post_data.yaml` 作为专用统计配置（`data_name: multi`、`micro_batch_size: 48`），可以用它替代训练配置作为第一个参数。

脚本对 state 与 action 分别统计 mean/std，action 会按照 `subtract_state` 的实际取值决定是否按 delta 统计。多卡时各 rank 先本地统计，再 `all_gather_object` 由 rank 0 合并写盘。

生成后请确认输出 JSON 的路径与 `configs/robot_configs/xtrainer.yaml` 的 `norm_stats` 字段一致（当前仓库 `assets/norm_stats/` 中只有 `robotwin.json`，`xtrainer.json` 需要自己生成）。

## 10. 模型微调

### 10.1 全参微调启动命令

配置文件作为第一个位置参数传入，且必须以 `.yaml` 结尾；其余参数用点号路径覆盖配置项：

```bash
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --data.data_name xtrainer \
  --data.train_path /path/to/lerobot_dataset \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_stats_file assets/norm_stats/xtrainer.json \
  --train.output_dir /path/to/save_ckpt
```

先用 `--train.max_steps` 设一个小值（例如 10）跑通链路，确认数据、显存和保存都正常，再放大到正式步数。短跑只验证管线，不代表模型已经学会任务。

### 10.2 `train.sh` 的行为

`train.sh` 是一个 `torchrun` 封装：

- 未设置 `CUDA_VISIBLE_DEVICES` 时用 `nvidia-smi -L` 的行数作为进程数；设置了则按逗号分隔的数量计算。
- 支持 `NNODES`、`NODE_RANK`、`MASTER_ADDR`（默认 `0.0.0.0`）、`MASTER_PORT`（默认 `62500`）。
- 强制离线 HF 环境变量。
- 所有输出同时写入仓库根目录的 `log.txt`。

### 10.3 输出目录结构

```text
<output_dir>/
├── lingbotvla_cli.yaml                    # save_args 写出的完整训练配置
└── checkpoints/
    └── global_step_40000/
        ├── hf_ckpt/                       # *.safetensors，策略服务使用
        └── ...                            # dcp 分布式训练状态
```

`hf_ckpt` 目录是部署入口。服务端需要从 checkpoint 路径向上三级找到 `<output_dir>/lingbotvla_cli.yaml`，因此**不要把 `hf_ckpt` 单独拷走**，必须保持目录层级。

### 10.4 冻结 VLM 微调及注意事项

冻结 VLM 时，视觉编码器和语言模型保持预训练权重，继续训练动作专家及其余未冻结模块。使用相同的训练入口，在第 10.1 节命令中增加以下覆盖项，并将 `--train.output_dir` 改为新的输出目录：

```bash
--train.freeze_vision_encoder true \
--train.train_expert_only true \
--train.train_state_proj true \
--train.enable_resume false
```

也可以在 YAML 的 `train` 节点中设置这四个字段。当前 X-Trainer 配置的 `freeze_vision_encoder` 为 `false`，`train_expert_only` 默认也是 `false`，需要显式开启。

- **冻结视觉编码器与冻结整个 VLM 不同。** 仅设置 `freeze_vision_encoder: true` 时，语言模型仍参与训练；冻结整个 VLM 应同时设置 `train_expert_only: true`。不要仅用 `freeze_vit` 代替这两个开关。
- **`train_expert_only` 不表示只更新动作专家。** `train_state_proj` 默认是 `true`，状态投影层仍参与训练；动作侧投影层和已启用的深度/视频对齐模块等也不会被这两个开关自动冻结。如需冻结状态投影层，可额外设置 `train_state_proj: false`。
- **切换训练方式时使用新输出目录。** 从全参微调切换到冻结 VLM 微调时，不要直接恢复原运行的优化器状态，可通过 `model.model_path` 指定已有 `hf_ckpt` 作为权重起点。上述命令关闭自动续训；如配置中设置了 `train.load_checkpoint_path`，也应清除。后续对同一次冻结 VLM 训练续训时，保持冻结配置一致，再开启 `enable_resume`。
- **冻结不会关闭辅助损失。** `align_params` 中已启用的深度/视频教师和损失仍会执行，教师权重与未来帧数据仍须准备齐全。是否调整辅助损失应单独评估。
- **冻结后仍需加载 VLM 并执行前向计算。** 不应据此假定可以移除基础模型，或显存和耗时必然按可训练参数比例下降。先进行少量步数试跑，确认损失、显存和 checkpoint 保存正常，再开展正式训练。
- **数据与部署契约保持一致。** 仍需计算并使用匹配的归一化统计，保持机器人特征映射、动作维度和 checkpoint 目录层级一致。

## 11. 断点续训

`train.enable_resume: true` 时，训练启动会扫描 `<output_dir>/checkpoints` 下形如 `global_step_<N>` 的目录并恢复最近的检查点：

```bash
NNODES=2 NODE_RANK=<0-or-1> MASTER_ADDR=<rank0-ip> MASTER_PORT=62500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml \
  --data.data_name xtrainer \
  --data.train_path /path/to/lerobot_dataset \
  --data.norm_stats_file assets/norm_stats/xtrainer.json \
  --train.output_dir /path/to/save_ckpt \
  --train.enable_resume true
```

如果目录里有多个检查点，会按 `global_step_<N>` 的数值从新到旧依次尝试加载。也可以显式指定：

```bash
--train.load_checkpoint_path /path/to/save_ckpt/checkpoints/global_step_20000
```

续训时以 checkpoint 中保存的模型、优化器、调度器和步数为准。更换数据集或任务后，应重新评估训练结果是否仍具备可比性。

配置文件的首个位置参数除 `.yaml` / `.yml` 外也支持 `.json`；命令行传入的 `--group.field value` 优先级高于配置文件中的同名字段。

## 12. 启动策略服务

### 12.1 启动

```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
python scripts/serve_policy.py \
  --model-path /path/to/save_ckpt/checkpoints/global_step_40000/hf_ckpt \
  --robot xtrainer \
  --use-length 50 \
  --port 8000
```

服务启动时会加载 checkpoint 和 `configs/robot_configs/<robot>.yaml`，初始化特征映射与归一化器，并按 metadata 中的 `robot` 完成一次 `reset`。**客户端不需要在第一次推理前额外发送 reset 请求。**

参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-path` / `--model_path` | 必填 | 包含 `.safetensors` 的 checkpoint 目录。 |
| `--robot` / `--robo-name` / `--robo_name` | `xtrainer` | `configs/robot_configs` 下的机器人配置名，不含 `.yaml`。 |
| `--norm-path` / `--norm_path` | 无 | 覆盖归一化统计路径；不填则用训练配置的 `data.norm_stats_file`。 |
| `--host` | `0.0.0.0` | WebSocket 监听地址。 |
| `--port` | `8000` | WebSocket 监听端口。 |
| `--use-length` / `--use_length` | `50` | 每次模型前向后使用的动作数量。 |
| `--step-mode` | 关闭 | 每次请求仅返回一个动作；默认返回完整 action chunk。 |
| `--fp32` | 关闭 | 使用 FP32；默认使用 BF16。 |
| `--compile` | 关闭 | 使用 `torch.compile` 加速推理。 |

启动后会向客户端发送 metadata：

```json
{"model_type": "lingbot-vla-2.0", "robot": "xtrainer"}
```

真机客户端会校验 `model_type` 与 `robot`，不匹配则拒绝运行。

服务存活检查：

```bash
curl http://127.0.0.1:8000/healthz
```

客户端侧调用方式：

```python
from deploy.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="127.0.0.1", port=8000)
metadata = policy.get_server_metadata()
result = policy.infer(observation)
```

`observation` 的字段必须与所选机器人 YAML 中的 `origin_keys` 对齐。对默认的 `xtrainer` 配置：

- `observation.state`：形状 `(14,)`。
- `observation.images.top` / `observation.images.left_wrist` / `observation.images.right_wrist`：RGB 图像。
- `task`：自然语言任务描述。

策略服务返回 `action`，形状为 `(H, 14)`，并附带 `server_timing` 字段。

加载期间还需要满足的目录约束：

- checkpoint 目录内包含一个或多个 `.safetensors` 文件。
- checkpoint 路径向上三级存在训练时生成的 `lingbotvla_cli.yaml`。
- Qwen3-VL 基础模型路径从训练配置的 `tokenizer_path` 读取，可用 `QWEN3VL_PATH` 环境变量覆盖。

## 13. 启动真机任务

### 13.1 启动命令

```bash
# 终端 1
python scripts/serve_mock_policy.py --host 127.0.0.1 --port 8000 --horizon 50

# 终端 2
python scripts/run_xtrainer_real.py \
  --host 192.168.1.10 \
  --task "将桌面上的方块放入收纳盒" \
  --action-horizon 25 \
  --control-hz 20 \
  --max-steps 200
```

默认使用本机 X-Trainer 的三路相机序列号；若机器人接线不同，可通过三个
`--camera-*-serial` 参数覆盖。

**安全提示：客户端没有 `--execute` 之类的二次确认开关。** 参数校验通过后程序会立即连接机械臂、复位并开始下发动作。上机前必须清空工作空间、确认急停可触达，并先用很小的 `--max-steps` 试跑。

### 13.2 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | 必填 | 策略服务的 IP 或主机名。 |
| `--port` | `8000` | 必须与 `serve_policy.py` 的 `--port` 一致。 |
| `--task` | `pick up the object` | 发送给模型的任务文本，应与训练数据中的任务描述一致。中文可直接传入。 |
| `--action-horizon` | `25` | 每次策略响应最多执行的动作步数。 |
| `--control-hz` | `20.0` | 动作下发频率。 |
| `--max-steps` | `1000` | 整次任务最多执行的动作步总数。首次上机建议显著调小。 |
| `--left-robot-ip` | `192.168.5.1` | 左臂 IP。 |
| `--right-robot-ip` | `192.168.5.2` | 右臂 IP。 |
| `--left-gripper-port` | `/dev/ttyUSB1` | 左夹爪串口。 |
| `--right-gripper-port` | `/dev/ttyUSB0` | 右夹爪串口。 |
| `--left-gripper-id` | `21` | 左夹爪舵机 ID。 |
| `--right-gripper-id` | `22` | 右夹爪舵机 ID。 |
| `--left-gripper-servo-pos` | `2048,3052` | 左夹爪行程 `MIN,MAX`。 |
| `--right-gripper-servo-pos` | `2048,3052` | 右夹爪行程 `MIN,MAX`。 |
| `--camera-top-serial` | `409122273405` | 顶部相机序列号；可按现场硬件覆盖。 |
| `--camera-left-wrist-serial` | `412622272997` | 左腕相机序列号；可按现场硬件覆盖。 |
| `--camera-right-wrist-serial` | `412622271417` | 右腕相机序列号；可按现场硬件覆盖。 |
| `--camera-fps` | `30.0` | 相机帧率。 |
| `--render-height` | `224` | 送入模型的图像高度。 |
| `--render-width` | `224` | 送入模型的图像宽度。 |
| `--max-joint-delta` | `0.17` | 单步关节变化上限（弧度）。超过时改用插值平滑过渡。 |
| `--ramp-step` | `0.01` | 平滑插值的步长。 |
| `--ramp-max-steps` | `100` | 平滑插值的最大步数。 |
| `--gripper-update-threshold` | `0.02` | 夹爪目标的最小变化量，小于该值不发送。 |
| `--servo-step-limit` | `0.9` | 底层舵机单步限幅。 |

服务端的 `--use-length` 必须**不小于**客户端的 `--action-horizon`，并且不要向服务端传 `--step-mode`。

### 13.3 客户端行为

- `observation.state` 由左臂 7 维（6 关节 + 夹爪）与右臂 7 维拼接而成，与第 2.3 节顺序一致。
- 三路图像会先 `resize_with_pad` 到 `render_height × render_width`，再转成 uint8。
- 动作下发前会校验形状必须为 14 维且全部有限；两个夹爪目标会被裁剪到 `[0,1]`。
- 关节变化超过 `--max-joint-delta` 时不会直接跳变，而是按 `--ramp-step` 插值过去。
- 客户端只能拒绝格式错误、NaN 或 Inf 动作，**无法判断一个数值合法的动作对当前场景是否安全**。
