#!/bin/bash
if [[ -d /workspace1/zhijun ]]; then
  BASE=/workspace1/zhijun
else
  BASE=/users/zhijun
fi

EXTRA_MOUNTS=""
if [[ -d /storage ]]; then
  EXTRA_MOUNTS+=" -v /storage:/storage"
fi

# docker run -it \
#   --gpus all \
#   --shm-size 128g \
#   --net=host \
#   --name zhijun_rlinf \
#   -v ${BASE}:${BASE} \
#   ${EXTRA_MOUNTS} \
#   -w ${BASE}/RLinf \
#   -e NVIDIA_DRIVER_CAPABILITIES=all \
#   -e HF_TOKEN=$HF_TOKEN \
#   -e HF_HOME=${BASE}/hf_download \
#   -e WANDB_API_KEY=$WANDB_API_KEY \
#   rlinf/rlinf:agentic-rlinf0.1-maniskill_libero \
#   /bin/bash

docker run -it \
  --device nvidia.com/gpu=0 \
  --device nvidia.com/gpu=1 \
  --device nvidia.com/gpu=2 \
  --device nvidia.com/gpu=3 \
  --device nvidia.com/gpu=4 \
  --device nvidia.com/gpu=5 \
  --shm-size 128g \
  --net=host \
  --name zhijun_rlinf \
  -v /workspace1/zhijun:/workspace1/zhijun \
  -v /storage:/storage \
  -w /workspace1/zhijun/RLinf \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME=/workspace1/zhijun/hf_download \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  rlinf/rlinf:agentic-rlinf0.2-maniskill_libero \
  /bin/bash
