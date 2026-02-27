# REFERENCE
https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html

# DOCKER USAGE

## Download docker image
```bash
docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0
```

## Docker run container
⚠️ 只有第一次创建的时候运行这个命令
```bash
docker run -it \
  --gpus all \
  --shm-size 128g \
  --net=host \
  --name zhijun_rlinf \
  -v /workspace1/zhijun:/workspace1/zhijun \
  -w /workspace1/zhijun \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 \
  /bin/bash
```
如果使用了一下命令删除该容器，则要重新创建，原来在容器中配置环境变量，安装的软件会丢失
```bash
exit
docker stop zhijun_rlinf # 当然在删除之前需要停止该容器的运行
docker rm zhijun_rlinf # 注意删除容器会丢失安装的软件和配置的环境变量
```

## Continue with zhijun_rlinf container
```bash
docker start zhijun_rlinf
docker exec -it zhijun_rlinf /bin/bash
```

# QUICK START

## Switch to openpi env
```bash
source switch_env openpi
```

## Run SFT using libero_sft_openpi.yaml
```bash
bash examples/sft/run_embodiment_sft.sh arc_libero_sft_openpi # zhijun edited
bash examples/sft/run_embodiment_sft.sh libero_sft_openpi # og
```

## RL & Eval using libero_goal_ppo_openpi.yaml
```bash
bash examples/embodiment/eval_embodiment.sh arc_libero_goal_ppo_openpi LIBERO # zhijun edited
bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_openpi LIBERO # og
```



# CHECK LOGS
⚠️ server 上的 logs 时间似乎较真实时间慢 8h