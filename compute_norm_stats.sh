#!/usr/bin/env bash
set -euo pipefail

cd /workspace1/zhijun/RLinf
python -u compute_openpi_norm_stats.py \
  --model-path /workspace1/zhijun/hf_download/models/pi05_base \
  --config-name pi05_blockpap_mix \
  --batch-size 64 \
  --lerobot-home /workspace1/zhijun/mg_dataset \
  --repo-id /workspace1/zhijun/mg_dataset/BlockPAP-v1_Mix \
  --asset-id BlockPAP-v1_Mix \
  --fast-parquet-only \
  --hf-offline
