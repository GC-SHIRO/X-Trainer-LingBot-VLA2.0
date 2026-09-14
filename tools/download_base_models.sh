#!/bin/bash
set -euo pipefail

# Download the LingBot-VLA 2.0 base models from Hugging Face or ModelScope.
#
#   bash tools/download_base_models.sh                      # Hugging Face (default)
#   bash tools/download_base_models.sh --source modelscope  # ModelScope
#   bash tools/download_base_models_modelscope.sh           # ModelScope wrapper
#
# Environment overrides:
#   MODEL_SOURCE=<hf|modelscope>              used when --source is omitted
#   MODELS_DIR=<path>                         output directory (default: <repo>/models)
#   QWEN_REPOSITORY=<owner/repository>        Qwen3-VL repository
#   LINGBOT_REPOSITORY=<owner/repository>     LingBot-VLA repository
#   MOGE_REPOSITORY=<owner/repository>        MoGe-2 repository on Hugging Face
#   MOGE_MODELSCOPE_REPOSITORY=<owner/repo>   MoGe-2 mirror on ModelScope (optional)

usage() {
  cat <<'EOF'
Usage: bash tools/download_base_models.sh [--source hf|modelscope]

  -s, --source <hf|modelscope>  Model hub to download from (default: hf, or $MODEL_SOURCE)
      --hf                      Shorthand for --source hf
      --modelscope              Shorthand for --source modelscope
  -h, --help                    Show this message

MoGe-2 ViT-B Normal has no official ModelScope release. With --source modelscope the
MoGe-2 checkpoint falls back to Hugging Face unless MOGE_MODELSCOPE_REPOSITORY points
at a ModelScope mirror.
EOF
}

SOURCE="${MODEL_SOURCE:-hf}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      SOURCE="$2"
      shift 2
      ;;
    --source=*)
      SOURCE="${1#*=}"
      shift
      ;;
    --hf|--huggingface)
      SOURCE="hf"
      shift
      ;;
    --modelscope|--ms)
      SOURCE="modelscope"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$SOURCE" in
  hf|huggingface) SOURCE="hf" ;;
  modelscope|ms|modelscope.cn) SOURCE="modelscope" ;;
  *)
    echo "Unsupported model source: $SOURCE (expected 'hf' or 'modelscope')" >&2
    exit 2
    ;;
esac

echo "=== Download LingBot-VLA 2.0 base models (source: $SOURCE) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default output is <repo>/models, which matches the ./models/... paths used by
# configs/vla/xtrainer/xtrainer.yaml. Override with MODELS_DIR=<path> if needed.
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
mkdir -p "$MODELS_DIR"

if [[ "$SOURCE" == "hf" ]]; then
  DEFAULT_QWEN_REPOSITORY="Qwen/Qwen3-VL-4B-Instruct"
  DEFAULT_LINGBOT_REPOSITORY="robbyant/lingbot-vla-v2-6b"
  DEFAULT_MOGE_REPOSITORY="Ruicheng/moge-2-vitb-normal"
else
  # ModelScope mirrors Qwen3-VL and LingBot-VLA, but not MoGe-2 ViT-B Normal.
  DEFAULT_QWEN_REPOSITORY="Qwen/Qwen3-VL-4B-Instruct"
  DEFAULT_LINGBOT_REPOSITORY="Robbyant/lingbot-vla-v2-6b"
  DEFAULT_MOGE_REPOSITORY="Ruicheng/moge-2-vitb-normal"
fi

QWEN_REPOSITORY="${QWEN_REPOSITORY:-$DEFAULT_QWEN_REPOSITORY}"
LINGBOT_REPOSITORY="${LINGBOT_REPOSITORY:-$DEFAULT_LINGBOT_REPOSITORY}"
MOGE_REPOSITORY="${MOGE_REPOSITORY:-$DEFAULT_MOGE_REPOSITORY}"
MOGE_MODELSCOPE_REPOSITORY="${MOGE_MODELSCOPE_REPOSITORY:-}"

MOGE_SOURCE="$SOURCE"
MOGE_REPO="$MOGE_REPOSITORY"
if [[ "$SOURCE" == "modelscope" && -z "$MOGE_MODELSCOPE_REPOSITORY" ]]; then
  MOGE_SOURCE="hf"
fi

NEED_HF="no"
NEED_MODELSCOPE="no"
if [[ "$SOURCE" == "hf" ]]; then NEED_HF="yes"; else NEED_MODELSCOPE="yes"; fi
if [[ "$MOGE_SOURCE" == "hf" ]]; then NEED_HF="yes"; else NEED_MODELSCOPE="yes"; fi

if [[ "$NEED_HF" == "yes" ]] && ! command -v hf >/dev/null 2>&1; then
  if [[ "$SOURCE" == "hf" ]]; then
    echo "The supported Hugging Face CLI, hf, was not found. Install it first: pip install -U huggingface_hub" >&2
  else
    echo "MoGe-2 ViT-B Normal is not published on ModelScope, so it is fetched from Hugging Face and needs the hf CLI." >&2
    echo "Install it with: pip install -U huggingface_hub, or set MOGE_MODELSCOPE_REPOSITORY=<owner/repository> to use a ModelScope mirror." >&2
  fi
  exit 1
fi

if [[ "$NEED_MODELSCOPE" == "yes" ]] && ! command -v modelscope >/dev/null 2>&1; then
  echo "The ModelScope CLI, modelscope, was not found. Install it first: pip install -U modelscope" >&2
  exit 1
fi

download_repo() {
  # download_repo <source> <repo_id> <local_dir>
  local source="$1" repo_id="$2" local_dir="$3"
  case "$source" in
    hf) hf download "$repo_id" --local-dir "$local_dir" ;;
    modelscope) modelscope download --model "$repo_id" --local-dir "$local_dir" ;;
  esac
}

download_file() {
  # download_file <source> <repo_id> <file> <local_dir>
  local source="$1" repo_id="$2" file="$3" local_dir="$4"
  case "$source" in
    hf) hf download "$repo_id" "$file" --local-dir "$local_dir" ;;
    modelscope) modelscope download --model "$repo_id" "$file" --local-dir "$local_dir" ;;
  esac
}

# 1. Qwen3-VL-4B-Instruct (vision-language model base)
echo "Downloading Qwen3-VL-4B-Instruct from $SOURCE..."
download_repo "$SOURCE" "$QWEN_REPOSITORY" "$MODELS_DIR/Qwen3-VL-4B-Instruct"

# 2. LingBot-VLA v2.6B. This repository also includes LingBot-Depth and Video-DINO.
echo "Downloading LingBot-VLA v2.6B from $SOURCE..."
download_repo "$SOURCE" "$LINGBOT_REPOSITORY" "$MODELS_DIR/lingbot-vla-v2-6b"

# 3. MoGe-2 checkpoint expected by configs/vla/xtrainer/xtrainer.yaml.
DEPTH_DIR="$MODELS_DIR/MoRGBD"
MODEL_FILE="$DEPTH_DIR/model.pt"
TARGET_FILE="$DEPTH_DIR/moge2-vitb-normal.pt"
mkdir -p "$DEPTH_DIR"
if [[ "$MOGE_SOURCE" != "$SOURCE" ]]; then
  echo "MoGe-2 ViT-B Normal is not published on ModelScope; downloading $MOGE_REPO from Hugging Face instead."
  echo "Set MOGE_MODELSCOPE_REPOSITORY=<owner/repository> to use a ModelScope mirror."
fi
echo "Downloading MoGe-2 ViT-B Normal from $MOGE_SOURCE..."
download_file "$MOGE_SOURCE" "$MOGE_REPO" model.pt "$DEPTH_DIR"

# The hub client can materialize model.pt as a symlink into its own cache, and a
# previous run may already have created the filename the configs expect. In that
# case `mv` exits non-zero with "are the same file", so compare the files first
# and keep this step idempotent.
if [[ -e "$MODEL_FILE" ]]; then
  if [[ "$MODEL_FILE" -ef "$TARGET_FILE" ]]; then
    echo "MoGe-2 already published as $TARGET_FILE; removing the duplicate name $MODEL_FILE."
    rm -f "$MODEL_FILE"
  else
    mv -f "$MODEL_FILE" "$TARGET_FILE"
  fi
fi

# Resolve a cache symlink into a real file so the checkpoint survives copying or
# moving the models directory to another disk.
if [[ -L "$TARGET_FILE" ]]; then
  echo "MoGe-2: replacing the symlink at $TARGET_FILE with a local copy of the weights."
  cp -Lf "$TARGET_FILE" "$TARGET_FILE.local" && mv -f "$TARGET_FILE.local" "$TARGET_FILE"
fi

if [[ ! -f "$TARGET_FILE" ]]; then
  echo "MoGe-2 checkpoint is missing at $TARGET_FILE" >&2
  exit 1
fi

echo "All base models were downloaded to $MODELS_DIR"
echo "  $MODELS_DIR/Qwen3-VL-4B-Instruct"
echo "  $MODELS_DIR/lingbot-vla-v2-6b"
echo "  $TARGET_FILE"
echo "These paths match the ./models/... entries in configs/vla/xtrainer/xtrainer.yaml."
echo "Hugging Face mode: bash tools/download_base_models.sh"
echo "ModelScope mode:   bash tools/download_base_models.sh --source modelscope"
echo "Override repositories with LINGBOT_REPOSITORY / MOGE_REPOSITORY / MOGE_MODELSCOPE_REPOSITORY."
