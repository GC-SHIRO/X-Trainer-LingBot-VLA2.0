#!/bin/bash
set -euo pipefail

# Convenience wrapper: download the LingBot-VLA 2.0 base models from ModelScope.
# Equivalent to `bash tools/download_base_models.sh --source modelscope`.
# Extra arguments are forwarded, so `--help` and the *_REPOSITORY overrides still work.
#
#   bash tools/download_base_models_modelscope.sh
#   bash tools/download_base_models_modelscope.sh --help
#
# Requires: pip install -U modelscope
# MoGe-2 ViT-B Normal is not published on ModelScope and is fetched from Hugging Face,
# so `pip install -U huggingface_hub` is also required unless MOGE_MODELSCOPE_REPOSITORY
# points at a ModelScope mirror.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/download_base_models.sh" --source modelscope "$@"
