# X-Trainer 真机客户端

本目录提供在 X-Trainer 真机上运行 LingBot-VLA 2.0 的最小 Linux 客户端。Dobot、RealSense 和 Feetech SDK 代码搬运自 `X-Trainer-Pi0.5-JAX/examples/xtrainer_real`，客户端不依赖 OpenPI runtime 或 OpenPI 客户端包。

## 安装依赖

在机器人控制机上安装轻量客户端依赖：

```bash
pip install -r deploy/xtrainer_real/requirements.txt
```

来源仓库中的 Feetech `scservo_sdk` 文件没有单独的上游许可证声明。独立分发本目录前，请先确认原始 SDK 的再分发条款。

## 启动推理服务

使用默认 action chunk 模式启动服务端。服务端的 `--use-length` 必须不小于客户端的 `--action-horizon`。

```bash
python scripts/serve_policy.py \
  --model-path /path/to/checkpoint \
  --robot xtrainer \
  --use-length 50 \
  --port 8000
```

使用本客户端时，不要向服务端传入 `--step-mode`。

当服务端和机器人控制机不是同一台电脑时，服务端必须监听全部网卡：

```bash
python scripts/serve_policy.py \
  --model-path /path/to/checkpoint \
  --robot xtrainer \
  --host 0.0.0.0 \
  --use-length 50 \
  --port 8000
```

在控制机上先检查服务端地址（不要使用 `127.0.0.1`）：

```bash
curl http://192.168.1.10:8000/healthz
```

如果服务端启用了 UFW，只向机器人控制机 IP 放行 TCP 8000 端口：

```bash
sudo ufw allow from CONTROL_PC_IP to any port 8000 proto tcp
```

当前 WebSocket 协议没有 TLS 和服务端鉴权，不要把端口直接暴露到公网；跨网段时应使用可信局域网或 VPN。

## 启动真机客户端

```bash
python scripts/run_xtrainer_real.py \
  --host 192.168.1.10 \
  --task "fold the clothes" \
  --camera-top-serial TOP_SERIAL \
  --camera-left-wrist-serial LEFT_WRIST_SERIAL \
  --camera-right-wrist-serial RIGHT_WRIST_SERIAL \
  --action-horizon 50 \
  --control-hz 20
```

其中 `--host` 是推理服务器的局域网/VPN IP。模型服务器不需要连接机器人、串口或相机；这些硬件只连接运行本客户端的控制机。

默认硬件地址与 Pi0.5 客户端保持一致：

- 左臂：`192.168.5.1`，夹爪串口 `/dev/ttyUSB1`，舵机 ID `21`。
- 右臂：`192.168.5.2`，夹爪串口 `/dev/ttyUSB0`，舵机 ID `22`。
- 状态与动作顺序：左臂 6 个关节、左夹爪、右臂 6 个关节、右夹爪。

运行 `python scripts/run_xtrainer_real.py --help` 可查看全部安全阈值和硬件参数。

## 首次真机测试

清空机器人工作空间，并从较小的 `--max-steps` 开始测试。确认第一次策略响应的形状为 `(H, 14)`，机械臂目标是以弧度表示的绝对关节角，并且两个夹爪值均位于 `[0, 1]`。

客户端会在动作发送到机器人前拒绝格式错误、NaN 或 Inf 动作，但无法判断一个数值有效的动作对于当前场景是否具备实际安全性。

### 不启动模型服务的基础控制测试

先清空工作空间、确认每个关节正负 5 度范围内都不会碰撞，并站在急停按钮旁。该脚本不连接 WebSocket，也不加载模型，而是按照服务端相同的 `{"action": (H, 14)}` 数据契约生成动作。它会依次测试左臂 1–6、右臂 1–6，每个关节执行“当前角度 → +5° → 当前角度 → -5° → 当前角度”，随后依次开合左、右夹爪；任何时刻都只有一个测试目标发生变化：

```bash
python tests/run_xtrainer_basic_control.py \
  --camera-top-serial TOP_SERIAL \
  --camera-left-wrist-serial LEFT_WRIST_SERIAL \
  --camera-right-wrist-serial RIGHT_WRIST_SERIAL \
  --yes
```

三个相机序列号必须替换为运行本脚本的控制机通过 USB 直接识别到的真实 RealSense 序列号，不能填 `TOP_SERIAL` 这类占位符。脚本会先检查三台相机是否在本机可见；检查失败时不会连接、启用或移动机器人。

机械臂接口接收的是弧度制绝对目标，因此脚本会先读取当前 14 维 state，再把相对的正负 5 度换算成绝对目标。夹爪默认 `1.0` 为开、`0.0` 为合；如果实机方向相反，可交换 `--gripper-open` 和 `--gripper-close`。现有硬件接口在连接阶段本身还会依次初始化两个夹爪。脚本结束或异常退出时会尝试恢复初始姿态，然后断开设备并执行 `DisableRobot()`。

### 不加载模型的完整客户端演练

如果要连同 WebSocket、图像上传、action chunk 和控制循环一起测试，可先运行轻量假策略。它把客户端上报的当前状态原样重复为动作，因此只保持当前位置，不需要 GPU 或 checkpoint：

```bash
# 终端 1；同机测试保持默认 127.0.0.1，跨机测试改为 0.0.0.0
python scripts/serve_mock_policy.py --host 127.0.0.1 --port 8000 --horizon 50

# 终端 2
python scripts/run_xtrainer_real.py \
  --host 127.0.0.1 \
  --task "hardware integration test" \
  --camera-top-serial TOP_SERIAL \
  --camera-left-wrist-serial LEFT_WRIST_SERIAL \
  --camera-right-wrist-serial RIGHT_WRIST_SERIAL \
  --action-horizon 5 \
  --max-steps 10
```

跨机测试时，在假策略所在电脑监听 `0.0.0.0`，并把客户端的 `--host` 改成该电脑的实际 IP。确认以上测试通过后，再停止假策略并在同一 IP/端口启动真实 `serve_policy.py`。
