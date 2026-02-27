docker run -it \
  --gpus all \
  --shm-size 128g \
  --net=host \
  --name zhijun_rlinf \
  -v /workspace1/zhijun:/workspace1/zhijun \
  -w /workspace1/zhijun \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME=/workspace1/zhijun/hf_download \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 \
  /bin/bash