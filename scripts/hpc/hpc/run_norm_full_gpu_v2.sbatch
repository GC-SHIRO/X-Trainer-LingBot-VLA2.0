#!/bin/bash
#SBATCH --job-name=lingbot-norm-full-gpu-v2
#SBATCH --partition=sla100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data-sl/sl20230172/logs/lingbot-norm-full-gpu-v2-%j.out
#SBATCH --error=/data-sl/sl20230172/logs/lingbot-norm-full-gpu-v2-%j.err

set -eo pipefail

REPO=/data-sl/sl20230172/X-Trainer-LingBot-VLA2.0
CONFIG=/data-sl/sl20230172/tmp/xtrainer_norm_runtime.yaml
DATASET=/data-sl/sl20230172/DataSet/Sort_0803
OUTPUT="$REPO/assets/norm_stats/xtrainer.json"

cd "$REPO"
source "$REPO/activate_lingbot.sh"

test -f "$CONFIG" || {
    echo "Missing runtime config: $CONFIG"
    exit 1
}

test -d "$DATASET/data" || {
    echo "Missing dataset: $DATASET"
    exit 1
}

# 防止意外覆盖已经完成的正式统计文件。
test ! -e "$OUTPUT" || {
    echo "Output already exists; refusing to overwrite: $OUTPUT"
    exit 1
}

echo "host: $(hostname)"
echo "python: $(which python)"
echo "dataset: $DATASET"
echo "output: $OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# torchrun 会设置 LOCAL_RANK、RANK 和 WORLD_SIZE。
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  scripts/compute_norm_stats.py \
  "$CONFIG" \
  --data.data_name xtrainer \
  --data.train_path "$DATASET" \
  --data.robot_config_root configs/robot_configs \
  --data.norm_path "$OUTPUT" \
  --data.data_ratio_for_norm_compute 1.0 \
  --data.num_workers 24 \
  --train.micro_batch_size 256

echo "Full normalization completed: $OUTPUT"
