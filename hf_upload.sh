CKPT_DIR=/workspace1/zhijun/RLinf/logs/20260314-06:23:04/test_new_gripper_data/checkpoints
REPO=aaroncaozj/pi05_co-sft_blockpap
TYPE=model

cd /workspace1/zhijun/RLinf

python hf_upload.py \
  --path ${CKPT_DIR}/global_step_25000 \
  --repo ${REPO} \
  --type ${TYPE} \
  --path-in-repo global_step_25000

python hf_upload.py \
  --path ${CKPT_DIR}/global_step_10000 \
  --repo ${REPO} \
  --type ${TYPE} \
  --path-in-repo global_step_10000
