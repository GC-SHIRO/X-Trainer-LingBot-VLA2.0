#!/bin/bash
#SBATCH --job-name=sort-0812-full
#SBATCH --partition=sla100
#SBATCH --nodelist=fg03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=512G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data-sl/sl20230172/logs/sort-0812-full-%j.out
#SBATCH --error=/data-sl/sl20230172/logs/sort-0812-full-%j.err

set -eo pipefail

REPO=/data-sl/sl20230172/X-Trainer-LingBot-VLA2.0
DATASET=/data-sl/sl20230172/DataSet/Sort_0812_step3
SOURCE_NORM=$REPO/assets/norm_stats/xtrainer/sort_0812_step3.json
OUTPUT=/data-sl/sl20230172/lingbot_runs/sort_0812_step3_full_v1
RUN_NORM=$OUTPUT/norm_stats.json

cd "$REPO"
source "$REPO/activate_lingbot.sh"

test -d "$DATASET/data"
test -f "$SOURCE_NORM"
test -f models/lingbot-vla-v2-6b/model.safetensors.index.json
test -f models/Qwen3-VL-4B-Instruct/model.safetensors.index.json
test -f models/lingbot-vla-v2-6b/depth/model.pt
test -f models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth
test -f models/lingbot-vla-v2-6b/dino_video/config.yaml
test -f models/MoRGBD/moge2-vitb-normal.pt
grep -q 'relative_type: joint' configs/robot_configs/xtrainer.yaml
grep -Fq \
  'chunk_joint_mask = chunk_joint_mask & ~action_is_pad.unsqueeze(-1)' \
  lingbotvla/data/vla_data/utils.py

mkdir -p "$OUTPUT"

if test -f "$RUN_NORM"; then
  cmp -s "$SOURCE_NORM" "$RUN_NORM" || {
    echo "ERROR: existing run uses different norm stats: $RUN_NORM"
    exit 1
  }
else
  cp "$SOURCE_NORM" "$RUN_NORM"
fi

cp "$0" "$OUTPUT/run_train_sort_0812_step3_8gpu.sbatch"
git rev-parse HEAD > "$OUTPUT/git_commit.txt"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export TMPDIR=/data-sl/sl20230172/tmp/sort_train_${SLURM_JOB_ID}

mkdir -p "$TMPDIR"

echo "host: $(hostname)"
echo "commit: $(git rev-parse --short HEAD)"
echo "python: $(which python)"
echo "dataset: $DATASET"
echo "norm: $RUN_NORM"
echo "output: $OUTPUT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

torchrun \
  --nnodes=1 \
  --nproc-per-node=8 \
  --node-rank=0 \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  tasks/vla/train_lingbotvla.py \
  configs/vla/xtrainer/xtrainer.yaml \
  --data.data_name xtrainer \
  --data.train_path "$DATASET" \
  --data.robot_config_root configs/robot_configs \
  --data.norm_stats_file "$RUN_NORM" \
  --data.video_backend pyav \
  --data.num_workers 2 \
  --data.use_future_image true \
  --train.output_dir "$OUTPUT" \
  --train.max_steps 20000 \
  --train.save_steps 10000 \
  --train.save_epochs 0 \
  --train.enable_resume true \
  --train.save_hf_weights false \
  --train.async_save_hf_weights false \
  --train.use_wandb false \
  --train.enable_full_shard true \
  --train.enable_fp32 false \
  --train.enable_activation_offload true \
  --train.activation_gpu_limit 8 \
  --train.use_compile false \
  --train.moe_monitor_interval 1

echo "Sort training completed"