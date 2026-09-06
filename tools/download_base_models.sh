#!/bin/bash
set -euo pipefail

echo "=== Download LingBot-VLA 2.0 base models ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(dirname "$SCRIPT_DIR")/models"
mkdir -p "$MODELS_DIR"

if ! command -v hf >/dev/null 2>&1; then
  echo "The supported Hugging Face CLI, hf, was not found. Install it first: pip install -U huggingface_hub" >&2
  exit 1
fi

LINGBOT_REPOSITORY="${LINGBOT_REPOSITORY:-robbyant/lingbot-vla-v2-6b}"

# 1. Qwen3-VL-4B-Instruct (vision-language model base)
echo "Downloading Qwen3-VL-4B-Instruct..."
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$MODELS_DIR/Qwen3-VL-4B-Instruct"

# 2. LingBot-VLA v2.6B. This repository also includes LingBot-Depth and Video-DINO.
echo "Downloading LingBot-VLA v2.6B..."
hf download "$LINGBOT_REPOSITORY" \
  --local-dir "$MODELS_DIR/lingbot-vla-v2-6b"

# 3. MoGe-2 checkpoint expected by configs/vla/xtrainer/xtrainer.yaml.
DEPTH_DIR="$MODELS_DIR/MoRGBD"
mkdir -p "$DEPTH_DIR"
echo "Downloading MoGe-2 ViT-B Normal..."
hf download Ruicheng/moge-2-vitb-normal model.pt \
  --local-dir "$DEPTH_DIR"
mv -f "$DEPTH_DIR/model.pt" "$DEPTH_DIR/moge2-vitb-normal.pt"

echo "All base models were downloaded to $MODELS_DIR"
echo "Override the LingBot repository with: LINGBOT_REPOSITORY=<owner/repository> bash download_base_models.sh"
