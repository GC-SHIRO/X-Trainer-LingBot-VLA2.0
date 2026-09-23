# LeRobot v2.1 → v3.0 官方转换脚本

同目录的 `convert_dataset_v21_to_v30.py` 原样来自 Hugging Face LeRobot **v0.4.2**，保留 Apache-2.0 许可证头，未修改上游逻辑。

- 上游：https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py
- SHA256：`e95736e3039f962ea96629c8c8c664f3ccc6caee5bb3f2a7c872adf01b1bd636`
- 使用仓库 `tools/create_environment` 安装的 `lerobot==0.4.2` 环境，并确保 FFmpeg 可用。此脚本依赖 LeRobot 包，不是独立运行的单文件工具。

## 本地转换

假设数据集在 `/data/xtrainer_lerobot`，其中存在 `meta/info.json`，且 `codebase_version` 为 `v2.1`。在仓库根目录执行：

```bash
python tools/convert_dataset_v21_to_v30.py --root /data --repo-id xtrainer_lerobot --push-to-hub false
```

**这一版本实际使用 `Path(root) / repo_id` 定位数据。** 因此 `--root` 填父目录 `/data`，`--repo-id` 填子目录 `xtrainer_lerobot`；不要将数据集完整目录再次传给 `--root`。上述不含命名空间的 repo-id 仅用于已存在的本地目录和关闭上传的场景。

转换成功后：

- `/data/xtrainer_lerobot`：v3.0 数据，训练继续使用此绝对路径。
- `/data/xtrainer_lerobot_old`：原始 v2.1 数据。
- `/data/xtrainer_lerobot_v30`：转换期间的临时目录，成功后移动到原路径。

官方默认 `--push-to-hub true`，本地转换务必显式使用 `false`。若指定的本地目录不存在，脚本会尝试从 Hub 下载。运行前确认路径，并预留保存新旧两份数据的磁盘空间。

原版脚本会删除已有的 `_v30` 临时目录；当原目录与 `_old` 同时存在且通过版本校验时，会删除原目录并恢复 `_old` 后重新转换。重试前检查这几个目录，勿把其他数据放入这些同名目录。

若需要运行 `transform_xtrainer_dataset_images.py` 修正相机方向，请先在 v2.1 数据上完成，再执行本转换：图像修正工具依赖 v2.1 的视频目录布局。

## 验证

```bash
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; d = LeRobotDataset('xtrainer_lerobot', root='/data/xtrainer_lerobot'); print(d.meta.info['codebase_version'], len(d)); print(d[0].keys())"
```

应输出 `v3.0` 并成功读取第一帧。正式训练前仍需检查 episode 边界及多路视频对齐。本次引入只完成 Python 语法检查；当前本地环境缺少 LeRobot 及训练依赖，未执行真实数据转换。
