# X-Trainer 真机异步预取运行命令

本文档说明如何启动 LingBot-VLA 2.0 策略服务端，以及如何用异步 action chunk 预取模式运行 X-Trainer 真机客户端。

## 1. 启动策略服务端

在加载模型 checkpoint 的机器上运行：

```bash
python scripts/serve_policy.py \
  --model-path /path/to/checkpoint \
  --robot xtrainer \
  --host 0.0.0.0 \
  --use-length 50 \
  --port 8000
```

### 服务端参数说明

`--model-path /path/to/checkpoint`

训练好的 LingBot-VLA 2.0 checkpoint 路径。

`--robot xtrainer`

加载 X-Trainer 机器人配置。

`--host 0.0.0.0`

监听所有网卡。当策略服务端和机器人控制机不是同一台机器时，建议使用这个值。

`--use-length 50`

服务端每次推理返回的 action chunk 长度。这个值必须大于或等于客户端的 `--action-horizon`。

`--port 8000`

WebSocket 服务端端口。

真机异步预取客户端需要 action chunk，因此服务端不要使用 `--step-mode`。

## 2. 启动真机客户端

在机器人控制机上运行：

```bash
python scripts/run_xtrainer_real.py \
  --host 192.168.1.10 \
  --port 8000 \
  --task "fold the clothes" \
  --camera-top-serial TOP_SERIAL \
  --camera-left-wrist-serial LEFT_WRIST_SERIAL \
  --camera-right-wrist-serial RIGHT_WRIST_SERIAL \
  --action-horizon 50 \
  --control-hz 30 \
  --prefetch-remaining 28 \
  --max-switch-delta 0.12 \
  --switch-blend-steps 5 \
  --max-delta-per-step 0.05
```

请把 `192.168.1.10` 替换为策略服务端所在机器的实际 IP。请把三个相机序列号替换为机器人控制机上实际识别到的 RealSense 序列号。

## 3. 客户端参数说明

### 连接与任务参数

`--host 192.168.1.10`

策略服务端 IP 地址。如果服务端在另一台机器上，这里填写服务端的局域网或 VPN IP。只有服务端和客户端在同一台机器上时，才可以使用 `127.0.0.1`。

`--port 8000`

策略服务端端口，需要和服务端命令里的 `--port` 保持一致。

`--task "fold the clothes"`

发送给 VLA 模型的语言任务指令。实际运行时按任务内容修改。

### 相机参数

`--camera-top-serial TOP_SERIAL`

顶部相机序列号。

`--camera-left-wrist-serial LEFT_WRIST_SERIAL`

左腕相机序列号。

`--camera-right-wrist-serial RIGHT_WRIST_SERIAL`

右腕相机序列号。

### Action Chunk 与控制频率

`--action-horizon 50`

客户端每次最多执行多少步 action chunk。服务端的 `--use-length` 必须大于或等于这个值。

`--control-hz 30`

机器人控制频率。默认 `30`，表示每秒执行 30 个动作，也就是每步约 `0.033` 秒。

### 异步预取参数

`--prefetch-remaining 28`

当前 chunk 剩余多少步时，开始在后台推理下一段 action chunk。

推荐计算方式：

```text
prefetch_remaining >= ceil(模型平均推理耗时秒数 * control_hz) + 安全余量
```

示例：

```text
平均推理 0.4 秒，控制频率 30Hz -> 0.4 * 30 = 12，建议 --prefetch-remaining 16
平均推理 0.8 秒，控制频率 30Hz -> 0.8 * 30 = 24，建议 --prefetch-remaining 28
平均推理 1.2 秒，控制频率 30Hz -> 1.2 * 30 = 36，建议 --prefetch-remaining 40
```

如果这个值太小，下一段 chunk 可能来不及返回，客户端会短暂保持最后一个动作。如果这个值太大，下一段 chunk 使用的 observation 会更早，状态滞后会变大。

默认 `--prefetch-remaining 0`，即关闭异步预取；显式设置正数才会开启。

### Chunk 边界平滑参数

`--max-switch-delta 0.12`

切换 chunk 时，比较上一帧已经执行的动作和 `next_chunk[0]`。如果最大差值超过这个阈值，就触发边界平滑。

`--switch-blend-steps 5`

触发边界平滑后，用 next chunk 开头的多少步做线性过渡。

过渡效果大致是：

```text
上一帧动作 -> 平滑后的 next_chunk[0] -> 平滑后的 next_chunk[1] -> ... -> 正常 next_chunk 动作
```

### 单步动作限幅参数

`--max-delta-per-step 0.05`

每个控制步发送给机器人之前，对动作变化量做最终限幅。

设置为 `0` 表示关闭这个客户端侧限幅。

调参建议：

```text
动作太猛、边界抖动明显 -> 减小 --max-delta-per-step
动作太钝、跟随太慢 -> 增大 --max-delta-per-step，或者设为 0
```

## 4. 推荐初始参数

如果模型平均推理耗时约 `0.8` 秒，控制频率为 `30Hz`，可以先使用：

```bash
--action-horizon 50 \
--control-hz 30 \
--prefetch-remaining 28 \
--max-switch-delta 0.12 \
--switch-blend-steps 5 \
--max-delta-per-step 0.05
```

首次真机测试建议加小一点的 `--max-steps`，并确保急停按钮在手边：

```bash
python scripts/run_xtrainer_real.py \
  --host 192.168.1.10 \
  --port 8000 \
  --task "hardware integration test" \
  --camera-top-serial TOP_SERIAL \
  --camera-left-wrist-serial LEFT_WRIST_SERIAL \
  --camera-right-wrist-serial RIGHT_WRIST_SERIAL \
  --action-horizon 50 \
  --control-hz 30 \
  --prefetch-remaining 28 \
  --max-switch-delta 0.12 \
  --switch-blend-steps 5 \
  --max-delta-per-step 0.05 \
  --max-steps 100
```
