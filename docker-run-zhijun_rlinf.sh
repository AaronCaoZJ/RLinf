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

# ── GPU selection ────────────────────────────────────────────────────────────
# Default: every GPU the host reports (previously this was hard-coded to 0-5,
# which is why torch.cuda.device_count() was 6 inside the container).
#
# Pin a subset by setting GPUS (space- or comma-separated):
#     GPUS="0 2 5 7" bash docker-run-zhijun_rlinf.sh
#     GPUS=0,2,5,7   bash docker-run-zhijun_rlinf.sh
#     GPUS=all       bash docker-run-zhijun_rlinf.sh
#
# IMPORTANT: CUDA renumbers from 0 inside the container, in the order listed here.
#   GPUS="0 1 2 3 4 5 6 7" -> cuda:0..7  == physical 0..7   (identity, recommended)
#   GPUS="0 2 5 7"         -> cuda:0..3  == physical 0,2,5,7
# So with a subset you must use the CONTAINER-LOCAL indices in
# examples/sft/config/*.yaml `cluster.component_placement`, not the physical ones.
GPUS="${GPUS:-}"
GPUS="${GPUS//,/ }"
if [[ -z "${GPUS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
  else
    GPUS=all
  fi
fi

GPU_FLAGS=""
if [[ "${GPUS}" == "all" ]]; then
  GPU_FLAGS="--device nvidia.com/gpu=all"
else
  for g in ${GPUS}; do
    GPU_FLAGS+=" --device nvidia.com/gpu=${g}"
  done
fi
echo "[docker-run] exposing GPUs: ${GPUS}"

# Override to run a second container alongside an existing one.
CONTAINER_NAME="${CONTAINER_NAME:-zhijun_rlinf}"

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
  ${GPU_FLAGS} \
  --shm-size 128g \
  --net=host \
  --name ${CONTAINER_NAME} \
  -v /workspace1/zhijun:/workspace1/zhijun \
  -v /storage:/storage \
  -w /workspace1/zhijun/RLinf \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME=/workspace1/zhijun/hf_download \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  rlinf/rlinf:agentic-rlinf0.2-maniskill_libero \
  /bin/bash
