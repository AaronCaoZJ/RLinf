#!/bin/bash
cd /workspace1/zhijun/RLinf

# ── Config ────────────────────────────────────────────────────────────────────
GPU=0

CKPT_DIR=/workspace1/zhijun/RLinf/logs/20260317-14:18:32/new_norm_stats_stride2
STEP=15000
NORM_STATS=../hf_download/models/pi05_base
TRAJ_ID=random
BIAS=true
LOG_DIR=/workspace1/zhijun/RLinf/eval/blockpap
BIAS_TAG=$([ "${BIAS,,}" = "true" ] && echo "bias" || echo "no_bias")


# ── Run ───────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=${GPU} python toolkits/eval_scripts_openpi/blockpap_eval.py \
    --exp_name      pi05_${BIAS_TAG}_${TRAJ_ID}_${STEP}_stride2 \
    --log_dir       ${LOG_DIR} \
    --pretrained_path ${CKPT_DIR}/checkpoints/global_step_${STEP}/actor/model_state_dict/full_weights.pt \
    --norm_stats_path ${NORM_STATS} \
    --num_episodes  15 \
    --max_steps     600 \
    --action_chunk  8 \
    --num_steps     5 \
    --traj_id       ${TRAJ_ID} \
    --num_save_videos 15 \
    --log_interval  50 \
    --state_bias    ${BIAS}

# ── Base model baseline (uncomment to run) ────────────────────────────────────
# CUDA_VISIBLE_DEVICES=${GPU} python toolkits/eval_scripts_openpi/blockpap_eval.py \
#     --exp_name      pi05_base_eval \
#     --log_dir       ${LOG_DIR} \
#     --pretrained_path ${NORM_STATS} \
#     --norm_stats_path ${NORM_STATS} \
#     --num_episodes  5 \
#     --max_steps     600 \
#     --action_chunk  8 \
#     --num_steps     5 \
#     --traj_id       random \
#     --num_save_videos 5 \
#     --log_interval  100 \
#     --state_bias    false
