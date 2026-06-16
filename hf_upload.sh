CKPT_DIR=/workspace1/zhijun/RLinf/logs/20260509-12:33:15/real_3cam/checkpoints
# REPO=aaroncaozj/pi05_co-sft_blockpap_front-back
REPO=aaroncaozj/pi05_real-sft_sweep
TYPE=model
NUM_WORKERS=32

cd /workspace1/zhijun/RLinf

# Throughput-oriented settings (full-folder upload, no file filtering).
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Upload the norm stats
# python hf_upload.py \
#   --path /workspace1/zhijun/hf_download/models/pi05_base/SweepIntoDustpan-v1_Real \
#   --repo aaroncaozj/pi05_norm_stats_collection \
#   --type model \
#   --path-in-repo SweepIntoDustpan-v1_Real

# python hf_upload.py \
#   --path ${CKPT_DIR}/global_step_7500/actor/model_state_dict \
#   --repo ${REPO} \
#   --type ${TYPE} \
#   --path-in-repo global_step_7500/actor/model_state_dict

python hf_upload.py \
  --path ${CKPT_DIR}/global_step_20000/actor/model_state_dict \
  --repo ${REPO} \
  --type ${TYPE} \
  --path-in-repo global_step_20000/actor/model_state_dict

# python hf_upload.py \
#   --path /workspace1/zhijun/RLinf/logs/20260326-17:46:58/real_3cam/checkpoints/global_step_5000/actor/model_state_dict \
#   --repo ${REPO} \
#   --type ${TYPE} \
#   --path-in-repo global_step_5000/actor/model_state_dict

# python hf_upload.py \
#   --path ${CKPT_DIR}/global_step_15000 \
#   --repo ${REPO} \
#   --type ${TYPE} \
#   --path-in-repo stride2_global_step_15000 \
#   --num-workers ${NUM_WORKERS}

# Upload BlockStack dataset
# python hf_upload.py \
#   --path /workspace1/zhijun/mg_dataset/blockstack_cleaned_real \
#   --repo aaroncaozj/BlockStack-v1_Real \
#   --type dataset \
#   --num-workers ${NUM_WORKERS}

# python hf_upload.py \
#   --path /workspace1/zhijun/hf_download/datasets/0323_block_stack_neg \
#   --repo aaroncaozj/0323_blockstack_neg \
#   --type dataset \
#   --num-workers ${NUM_WORKERS}