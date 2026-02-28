#!/bin/bash
if [[ -d /workspace1/zhijun ]]; then
  BASE=/workspace1/zhijun
else
  BASE=/users/zhijun
fi

docker run -it \
  --gpus all \
  --shm-size 128g \
  --net=host \
  --name zhijun_rlinf \
  -v ${BASE}:${BASE} \
  -w ${BASE}/RLinf \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME=${BASE}/hf_download \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  rlinf/rlinf:agentic-rlinf0.1-maniskill_libero \
  /bin/bash