# X-Trainer LingBot-VLA 2.0 审查日志

**日期**：2026-09-17
**范围**：对照 `X-Trainer-SmolVLA`（67 提交）与 `X-Trainer-XVLA`（31 提交）两个已接近稳定版本的历史，审查 `X-Trainer-LingBot-VLA2.0` 的真机部署链路（`deploy/`、`scripts/`），决定哪些机制移植、哪些记录在案暂缓。

**方法**：逐提交读取实际 diff（不依赖提交信息），再与 LingBot 当前实现逐项比对。所有结论都落到具体 commit 和文件行。

---

## 1. 决策摘要

| 编号 | 项目 | 决定 |
| --- | --- | --- |
| A | 异步 action chunk 预取 | **删除**，不保留相关参数 |
| B | 客户端启动时调用 `policy.reset()` | **已实施** |
| C | 相机同步读取阻塞控制线程 | 暂缓（见 §4.1） |
| D | action chunk 拼接语义 | **已实施**（按 SmolVLA offset decay） |
| E | 图像传输压缩 | **已实施**（按 XVLA JPEG codec） |
| F | chunk 段内三点平滑 | 暂缓（见 §4.2） |
| G | 段尾 ease-out | 不做 |
| H | 观测/控制频率解耦 | 暂缓（见 §4.3） |
| I | 服务端 warmup | **已实施** |
| J | `smooth_reset` | **已实施**（范围小于原判断，见 §5） |
| K | 客户端读取 `server_timing` | **已实施** |
| L | 夹爪上报实测位置 | **已实施** |
| M | 数据集校验/转换器、LoRA | 不做 |
| N | 服务端 ensemble 死参数 | 暂缓（见 §4.4） |
| O | 段间 hold 策略 | **已实施**（按 XVLA 实测位姿） |

---

## 2. 关键分歧：预取该不该留

两个"稳定"仓库在这件事上分道扬镳，而 LingBot 原先停在两者都淘汰掉的形态上：

| 仓库 | 做法 | 原因 |
| --- | --- | --- |
| XVLA | `aa7b327` 整个删掉 prefetch（−190/+102） | 把新采样的 chunk 拼进正在执行的 chunk 会让真机中途换目标。改为：整段执行完 → 下发**实测位姿**作为 hold → 同步等下一段 |
| SmolVLA | 保留并修好 | `c113061` 明确删掉 0.3/0.7 重叠混合，改成 offset decay；`b6dd5a1`/`b0b7ec1` 修好单槽 latest-wins 协议与接收器死亡 bug |

LingBot 原本保留了 prefetch，且 `_apply_prefetched_chunk` 的 `replace` 模式会**直接丢弃当前未执行完的 remainder**——这正是 XVLA 删掉 prefetch 的理由。默认 `--prefetch-remaining 0` 虽已关闭预取，但参数和代码路径仍在。

**决定：整段删除预取代码与全部相关参数**，架构收敛到"完整执行当前 chunk → 同步请求下一段"，与 XVLA 的最终形态一致。

---

## 3. 已实施改动

### 3.1 删除预取（`scripts/run_xtrainer_real.py`）

删除的代码：`PrefetchResult`、`_infer_prefetch_chunk`、`_with_projected_state`、`_project_chunk_end_state`、`_apply_prefetched_chunk`、`_align_prefetched_chunk`、`_smooth_chunk_boundary`、`ThreadPoolExecutor` 预取执行器及相关 `Future` 状态。

删除的参数：`--prefetch-remaining`、`--prefetch-state-mode`、`--prefetch-apply-mode`、`--disable-prefetch-alignment`、`--prefetch-alignment-mode`、`--prefetch-alignment-search`、`--min-prefetch-actions`、`--max-switch-delta`、`--switch-blend-steps`。

主循环重写为：执行完整 chunk → 耗尽后同步 `policy.infer` → 取新 chunk。`run_xtrainer_real.py` 从 548 行降到约 300 行。

> 副作用（正向）：原先 `action_index >= len(action_chunk)` 之后的 `elif last_sent_action is not None and prefetch_future is not None` 分支不可达。该分支随重写消失，无需单独处理。

### 3.2 启动时调用 `policy.reset()`（B）

顺序：`environment.reset()`（机械臂到 reset pose）→ `policy.reset(robot_name)`。robot 名取自服务端 metadata 的 `robot` 字段。

服务端 `LingbotVLAv2Server.reset()` 会清空 `global_step`、`last_action_chunk`、`last_normalized_action_chunk` 并重建 `FeatureTransform`。`serve_policy.py` 在启动时已调用一次；但长驻服务端在多次运行之间不会再清，客户端现在每次运行开头都会清一次，与 SmolVLA（`db853c4`）、XVLA（首提交的 `await policy.reset()`）一致。

### 3.3 chunk 拼接改为 offset decay（D）

替换 `_smooth_chunk_boundary`（anchor 插值 + `max_switch_delta` 门控）为 SmolVLA `c113061` 的语义：

```text
切换时:  blend_offset = 上一个已下发动作 - 新 chunk 第一个动作
每一步:  progress = (index + 1) / blend_steps
         weight   = progress^2 * (3 - 2 * progress)
         target[关节] += (1 - weight) * blend_offset[关节]
```

只有常量偏移被衰减，新轨迹自身运动不受影响。首步保留约 92.6% 偏移，第 `blend_steps` 步精确回到模型目标。**夹爪列（索引 6、13）永不参与混合。** 新参数 `--chunk-blend-steps`（默认 6，`<=1` 关闭），不再有触发阈值门控——两个兄弟仓库都是无条件混合。

### 3.4 相机观测 JPEG 传输（E）

- 新增 `deploy/image_codec.py`：`encode_policy_images` / `decode_policy_images`，只处理 `observation.images.` 前缀的 ndarray，标记 `__lingbot_jpeg_rgb__`。
- 客户端（`deploy/websocket_client_policy.py`）：新增 `image_jpeg_quality` 参数（默认 85）。**只在服务端于握手元数据 `image_encodings` 中声明 `jpeg_rgb` 时才压缩**，对旧服务端自动回退为原始 ndarray。
- 服务端（`deploy/websocket_policy_server.py`）：在送入 policy 前解码，因此 `inference_callback` 记录到的仍是真实图像。
- `scripts/serve_policy.py` 与 `scripts/serve_mock_policy.py` 均声明 `image_encodings: ["raw_ndarray", "jpeg_rgb"]`。mock 服务端也声明，是为了让无模型真机测试顺带验证这条传输链路。
- 客户端新增 `--image-jpeg-quality`（默认 85，`0` 关闭）。

**实测压缩比**（本机 PIL 测得，非 cv2）：640×480 合成画面 q85 约 **28.7x**；64×96 纯噪声（JPEG 最坏情况）约 **3.5x**。三路 640×480 原始约 2.76 MB/请求。真实 RealSense 画面介于两者之间，文档统一表述为"十分之一量级"。

### 3.5 服务端 warmup（I）

`LingbotVLAv2Server.warmup(robo_name)`：用零 state 与三路零图像跑一次完整 `infer`，再 `reset()`。图像键从 `feature_transform.org_features["images"]` 读取，不硬编码相机名。`serve_policy.py` 新增 `--no-warmup`（默认开启 warmup）。目的：让首次真实请求不必承担 CUDA 核函数自动调优与惰性初始化的开销。

### 3.6 `smooth_reset`（J）

`XTrainerRealEnvironment.smooth_reset(goal)` 提为公开方法，`reset()` 委托给它，并在移动后按 `RESET_SETTLE_SECONDS = 0.2` 等待稳定，返回稳定后的实测关节角作为起始状态。保留 linspace 语义、保证落在目标点、夹爪与关节一同插值。

### 3.7 客户端读取 `server_timing`（K）

服务端本就在每个响应里写 `server_timing.infer_ms` / `prev_total_ms`，但客户端从不读取。现在每次拿到 chunk 后记录一行日志，使 chunk 之间的停顿可直接归因。

### 3.8 夹爪上报实测位置（L）

`DobotXTrainer._read_gripper_or_last_command()`：当 `read_gripper_position=True` 时读伺服实际位置，读失败则回退到上次下发值并告警，**不中断控制循环**。`XTrainerRealEnvironment` 两个 follower 均改为 `read_gripper_position=True`。

此前上报的是**上次下发值**，会掩盖夹爪被卡住、掉电或伺服故障——策略会一直被告知夹爪在它被要求的位置。

---

### 3.9 段间 hold 采用实测位姿（O）

删除预取后，每段 chunk 执行完到下一段返回之间是一段纯等待。此前客户端在这期间不发任何指令，机械臂靠 ServoJ 的 `servo_j_time = 0.03` 前瞻自然停在**最后下发目标**上。

chunk 耗尽时读一次观测，把其中的 `observation.state` 的机械臂关节位置作为 hold 下发，夹爪（索引 6、13）保留上一条下发目标，然后才发起下一次推理；**同一份观测**同时用于 hold 和推理请求，不额外多读一次相机。夹爪被物体挡住时，不能用实测开度覆盖抓紧目标，否则可能降低夹持力。

```python
observation = environment.get_observation()
if last_sent_action is not None:
    hold_action = _hold_action_from_observation(observation, last_sent_action)   # 实测关节 + 上一条夹爪目标
    environment.apply_action(hold_action)
    last_sent_action = hold_action.copy()
response = policy.infer(observation)
```

两个关键点：

- **为什么用实测位姿而不是重发上一个动作**（SmolVLA 的做法）：机械臂在段内被外力推动或下坠时，继续下发最后目标会一直"顶"着那个旧目标；下发实测位姿则停在它当前所在的位置，再让新 chunk 从那里起规划。
- **hold 会写入 `last_sent_action`**，因此下一段的 `blend_offset` 是从 hold 位置起算，而不是从上一段最后一步算——这正是 offset decay 能衔接上的前提。

首段没有 hold（`last_sent_action is None`，机械臂刚 reset 完）。新增 `--` 无相关开关：与 XVLA 一致，hold 是无条件行为。日志中记为 `source="hold"`。

---

## 4. 记录在案、本次不做

### 4.1 相机观测在控制线程同步读取

**问题**：`XTrainerRealEnvironment.get_observation()` 在控制线程内联读三路相机；`_read_camera_frame` 每路相机 `async_read(timeout_ms=50)` 重试最多 5 次、间隔 `sleep(0.005)`。最坏情况单次观测超过 75 ms，而 30 Hz 的控制周期只有 33 ms。相机的 `async_read` 实际是阻塞在 `wait_for_frames` 上，名字有误导性。

**参考做法**：

- SmolVLA `64e2c24`：采集移到单 worker `ThreadPoolExecutor`（`_ObservationWorker`），至多一个在飞，`age_steps >= action_horizon` 的过期帧直接丢弃。
- SmolVLA `58b9066`：取帧改用 `read_latest(max_age_ms=100)`，容忍 stale buffer，取代阻塞式 `read()`。

**实施时的硬约束**：后台线程读夹爪会与写操作竞争串口应答，**必须与串口事务锁一并移植**（SmolVLA `64e2c24` 在 `_request` 外加 `threading.Lock` 覆盖整段写+状态交换）。LingBot 的 `DobotGripperWrapper` 已有 `_io_lock`，但需确认它覆盖完整的写—读应答往返，而不只是单次调用。

**暂缓原因**：改动面覆盖采集、串口、控制循环三处，且需要与刚简化的控制流一起上机验证。建议作为独立一轮。

### 4.2 chunk 段内三点平滑

**参考做法**（SmolVLA `64e2c24`，默认强度 0.5）：

```python
result[1:-1, joints] = (1 - strength / 2) * src[1:-1, joints] \
                     + strength / 4 * (src[:-2, joints] + src[2:, joints])
```

只作用于段**内部**索引与关节列（`[0:6, 7:13]`），首尾动作与夹爪不动；强度 0.5 时卷积核为 0.125 / 0.75 / 0.125。LingBot 完全没有段内平滑。

**暂缓原因**：它与 §3.3 的 offset decay 作用在同一信号上，两者叠加的效果需要分开验证。建议先单独确认 offset decay 的收益，再决定是否叠加。

### 4.3 观测频率与控制频率解耦

**参考做法**（SmolVLA `b6dd5a1`）：`--observation-hz`（默认 10）限流观测提交，`next_observation_at = now + 1/observation_hz`；控制循环**即使超时也强制让出**（`await sleep_fn(max(remaining, 0.0))`），否则异步收发任务得不到调度。

**LingBot 现状评估**：这条对 LingBot 的价值有限。LingBot 客户端是同步实现（`websockets.sync`）而非 asyncio，不存在收发任务被饿死的问题；且每段 chunk 只观测一次，本来就没有观测过频。真正需要限流的前提是 §4.1 的后台采集落地。

**暂缓原因**：与 §4.1 绑定，单独做没有收益。

### 4.4 服务端 ensemble 死参数

`LingbotVLAv2Server.__init__` 声明了 `adaptive_ensemble_alpha=0.1` 与 `action_ensemble_horizon=8`，但全仓库**从未读取这两个属性**；类 docstring 声称 "support action ensemble"，与实际不符。

**值得注意的是**：SmolVLA 和 XVLA **都没有** temporal ensemble。把这组参数真正接上（对重叠的 action chunk 做指数加权集成）是 LingBot 相对两个兄弟仓库**唯一可以反超**的点——但它属于新特性而非缺陷修复。

**暂缓原因**：需要先有稳定的基线，否则无法判断集成带来的变化是改善还是掩盖问题。

### 4.6 拼接偏移取自"已下发"而非"已应用"动作

`_chunk_blend_offset` 用的是客户端上一步下发的动作。在默认配置下这与真机实际执行的动作一致（关节目标不被改写，夹爪列不参与混合）。

但当显式设置有限的 `--max-joint-delta` 时，`XTrainerRealEnvironment.apply_action` 会改走 `_move_smooth`，实际下发的是插值序列而非原始目标——此时偏移量会有小偏差。SmolVLA 通过让 `apply_action` 返回真正下发的动作来规避这一点。

**未改**：默认路径精确无误，且这是非默认安全开关下的次级影响。若要彻底消除，可让 `apply_action` / `_move_smooth` 返回最终下发动作（对现有调用方向后兼容）。

---

## 5. 分析勘误

**关于 `smooth_reset`（J）**：先前的判断是"LingBot `_move_smooth` 每步 sleep、`reset()` 后还 `sleep(0.2)`"，据此认为需要按 XVLA `96e564a` 从"带 pacing 的渐进循环"改成"linspace + `pace=False`"。

实际读取代码后：**LingBot 的 `_move_smooth` 本来就没有逐步 sleep**，它早就是 `np.linspace(start, goal, steps)` 且保证落在目标点——即 XVLA 改造**之后**的语义。XVLA 那次改动的对象是它自己带 pacing 的旧实现，LingBot 从来不是那样。

因此 J 的实际改动缩小为：公开 `smooth_reset`、`reset()` 委托给它、保留稳定等待、把返回值定为稳定后的实测关节角。**没有**语义层面的修复。

**保留的观察**：`_move_smooth` 把整个 linspace 连贯下发，而 ServoJ 是带 0.03 s 前瞻的伺服目标——控制器实际上只会执行最后一个目标。也就是说这条"平滑"路径对大范围移动并不提供真正的速度限制。两个兄弟仓库在 `pace=False` 之后有同样性质。真要限制速度，需要按控制周期节流下发，或依赖控制器侧限制。

---

## 6. 明确不做

| 项目 | 说明 |
| --- | --- |
| 段尾 ease-out（XVLA `09aa7ca`） | `weight = p + p² − p³`，让段尾关节速度衰减到 0。不做。 |
| 数据集校验器 / raw→LeRobot 转换器 | SmolVLA 有 `validate_dataset_v21.py` + `convert_raw_to_lerobot_2_1.py`，XVLA 另有并行版 `..._turbo.py`。LingBot 不含转换器（README 指向 Pi0.5 流水线）。不做。 |
| X-Trainer LoRA | `lingbotvla/utils/lora_utils.py` 存在但 `add_lora_to_model` 从未被调用，README 亦自认未交付。SmolVLA 有完整链路（LORA 配置 + `train_smolvla_lora.sh` + policy 侧 `PeftModel.from_pretrained`）。不做。 |
| `expert_visual_transform` 引用未定义的 `config.resize_imgs_with_padding` | `lingbotvla/data/vla_data/transform.py:490` 读取的配置键**全仓库没有定义**，今天无害只因为它被 `utils.py:14` import 但从未被调用。一旦接线立刻崩。不管。 |
| README 死链、不可达分支 | 不可达分支已随预取重写消失。`deploy/xtrainer_real/README.md` 一行仍指向不存在的文件，保留。 |
| 安全限位默认全关 | `max_joint_delta` / `servo_step_limit` / `max_delta_per_step` 默认 `inf`/`0`。两个兄弟仓库是**有意为之**，一致，不改。 |

---

## 7. 验证状态

| 项目 | 状态 |
| --- | --- |
| `tests/test_xtrainer_real_client.py` | 17 项通过（含新增的"预取参数已彻底移除"断言与 offset decay 语义断言） |
| 全部 `tests/test_*.py` | **32 项通过，9 项跳过** |
| `tests/test_image_codec.py` | **9 项全部跳过** —— 本机三个 python 环境都没有 cv2 |
| codec 逻辑 | 用 PIL 实现的 cv2 shim 另行验证通过（19 项全过）：往返形状/色彩、压缩比、不改动入参、非图像键不受影响、原始载荷解码返回原对象、越界质量/非 uint8/损坏 JPEG 均被拒绝 |
| 真机验证 | **未做。** 本轮所有改动都未上机 |

**必须强调的保留意见**：`tests/test_image_codec.py` 从未在真实 cv2 下运行过。shim 验证的是我写的部分（字典遍历、标记、非变异），`cv2.imencode/imdecode` 的调用形式是从 XVLA 已工作的代码逐字复制的。但没有真机或真 cv2 验证前，不应假定这条链路已经跑通。

**首次上机建议顺序**：

1. 先跑 `serve_mock_policy.py` + `tests/run_xtrainer_basic_control.py`，此时 mock 服务端已声明 JPEG，可顺带验证编解码链路。
2. 确认 `--image-jpeg-quality 0` 与 `85` 两种情况下行为一致（原始 ndarray 路径是回退基线）。
3. 再上真实模型，先用 10 Hz 和较小 `--max-steps`，观察新增的 `Server timing` 日志，确认 `infer_ms` 与 chunk 间停顿的对应关系。

---

## 8. 附：本轮改动清单

```
 README.md                                       |  49 +-   预取章节移除、拼接/传输/hold 说明
 deploy/lingbot_vla_v2_policy.py                 |  26 ++   warmup
 deploy/websocket_client_policy.py               |  31 +-   JPEG 协商
 deploy/websocket_policy_server.py               |   3 +-   解码
 deploy/xtrainer_real/environment.py             |  32 +-   smooth_reset、夹爪实测
 deploy/xtrainer_real/hardware/dobot_xtrainer.py |  21 +-   夹爪实测
 scripts/run_xtrainer_real.py                    | 463 +---  删除预取、offset decay、实测位姿 hold、reset、timing
 scripts/serve_mock_policy.py                    |   4 +    JPEG 声明
 scripts/serve_policy.py                         |  16 +-   metadata、warmup
 tests/test_xtrainer_real_client.py              | 315 +---  预取测试移除，blend/hold/timing 测试新增
 10 files changed, 412 insertions(+), 548 deletions(-)

 新增 deploy/image_codec.py            76 行
 新增 tests/test_image_codec.py       113 行
 新增 docs/xtrainer_review_log.md     249 行
```

分支 `xtrainer/review-fixes`，**未合并入 main**。
