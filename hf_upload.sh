# CKPT_DIR=/workspace1/zhijun/RLinf/logs/20260316-08:26:43/test_z_euler_bias/checkpoints
# REPO=aaroncaozj/pi05_aligned_co-sft_blockpap
# TYPE=model

cd /workspace1/zhijun/RLinf

# # Upload the norm stats
# python hf_upload.py \
#   --path /workspace1/zhijun/hf_download/models/pi05_base/BlockPAP-v1_Mix \
#   --repo aaroncaozj/pi05_norm_stats_collection \
#   --type model \
#   --path-in-repo BlockPAP-v1_Mix

# python hf_upload.py \
#   --path ${CKPT_DIR}/global_step_5000 \
#   --repo ${REPO} \
#   --type ${TYPE} \
#   --path-in-repo global_step_5000

python hf_upload.py \
  --path ${CKPT_DIR}/global_step_15000 \
  --repo ${REPO} \
  --type ${TYPE} \
  --path-in-repo global_step_15000
