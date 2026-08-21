#!/usr/bin/env bash
# Step 2 of 2: plain pi0.5 SFT on MVTOKEN_22_27_04.
# GPUs 0,2,5,7 are selected via cluster.component_placement in the yaml.
# Run run_mvtoken_prepare.sh first.
set -euo pipefail

REPO=/workspace1/zhijun/RLinf
BASE_MODEL=/workspace1/zhijun/hf_download/models/pi05_base
REPO_ID=MVTOKEN_22_27_04
DATA_PARENT=/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/mix_22_27_04

cd "$REPO"

# ── preflight: fail loudly instead of training on un-normalised data ──────────
[ -d "${DATA_PARENT}/${REPO_ID}" ] || { echo "missing dataset ${DATA_PARENT}/${REPO_ID}; run run_mvtoken_prepare.sh" >&2; exit 1; }
[ -f "${BASE_MODEL}/${REPO_ID}/norm_stats.json" ] || { echo "missing norm stats ${BASE_MODEL}/${REPO_ID}/norm_stats.json; run run_mvtoken_prepare.sh" >&2; exit 1; }
echo "[sft] dataset + norm stats OK"

bash examples/sft/run_vla_sft.sh mvtoken_sft_openpi
