#!/usr/bin/env bash
# Step 1 of 2: prepare the MVTOKEN dataset for pi0.5 SFT.
#   (a) expose the lerobot dir under the repo_id name openpi expects
#   (b) compute global norm stats and drop them where the config looks for them
# Run this ONCE. It is idempotent.
set -euo pipefail

REPO=/workspace1/zhijun/RLinf
DATA_PARENT=/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/mix_22_27_04
REPO_ID=MVTOKEN_22_27_04
BASE_MODEL=/workspace1/zhijun/hf_download/models/pi05_base

cd "$REPO"

# ── (a) openpi resolves <HF_LEROBOT_HOME>/<repo_id>; the dataset dir is named
#        "lerobot", so expose it under the repo_id name. Symlink, not a move.
if [ ! -e "${DATA_PARENT}/${REPO_ID}" ]; then
    ln -s lerobot "${DATA_PARENT}/${REPO_ID}"
    echo "[prepare] symlinked ${DATA_PARENT}/${REPO_ID} -> lerobot"
else
    echo "[prepare] ${DATA_PARENT}/${REPO_ID} already exists, skipping symlink"
fi
ls -l "${DATA_PARENT}/${REPO_ID}/" | head -5

# ── (b) norm stats -> ${BASE_MODEL}/${REPO_ID}/norm_stats.json
#        --fast-parquet-only reads state/actions straight from parquet and skips
#        image decoding + transforms, which is what we want for a local dataset.
python -u compute_openpi_norm_stats.py \
  --model-path "${BASE_MODEL}" \
  --config-name pi05_mvtoken \
  --batch-size 64 \
  --lerobot-home "${DATA_PARENT}" \
  --repo-id "${DATA_PARENT}/${REPO_ID}" \
  --asset-id "${REPO_ID}" \
  --fast-parquet-only \
  --hf-offline

# ── verify: training silently falls back to un-normalised data if this is missing
STATS="${BASE_MODEL}/${REPO_ID}/norm_stats.json"
if [ -f "$STATS" ]; then
    echo "[prepare] OK: $STATS ($(stat -c%s "$STATS") bytes)"
    python - <<PY
import json
d = json.load(open("${STATS}"))
norm = d.get("norm_stats", d)
for k, v in norm.items():
    mean = v.get("mean"); std = v.get("std")
    if mean is None: continue
    print(f"  {k:10} dim={len(mean)}  mean[:3]={[round(x,4) for x in mean[:3]]}  std[:3]={[round(x,4) for x in std[:3]]}")
PY
else
    echo "[prepare] FAILED: $STATS not created" >&2
    exit 1
fi
