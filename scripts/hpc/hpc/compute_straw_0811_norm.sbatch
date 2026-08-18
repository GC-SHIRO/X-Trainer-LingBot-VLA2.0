#!/bin/bash
#SBATCH --job-name=straw-norm
#SBATCH --partition=sla100
#SBATCH --nodelist=fg02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data-sl/sl20230172/logs/straw-norm-%j.out
#SBATCH --error=/data-sl/sl20230172/logs/straw-norm-%j.err

set -euo pipefail

REPO=/data-sl/sl20230172/X-Trainer-LingBot-VLA2.0
DATASET=/data-sl/sl20230172/DataSet/Straw/Straw_0811
NORM=$REPO/assets/norm_stats/xtrainer/straw_0811.json

cd "$REPO"
source "$REPO/activate_lingbot.sh"

test -d "$DATASET"
test ! -e "$NORM" || {
  echo "ABORT: refusing to overwrite $NORM"
  exit 1
}

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

python scripts/compute_norm_stats.py \
  configs/vla/norm_compute/post_data.yaml \
  --data.data_name xtrainer \
  --data.train_path "$DATASET" \
  --data.robot_config_root configs/robot_configs \
  --data.norm_path "$NORM" \
  --data.data_ratio_for_norm_compute 1.0 \
  --data.num_workers 8 \
  --data.video_backend pyav \
  --train.micro_batch_size 32

echo "Straw norm stats completed: $NORM"