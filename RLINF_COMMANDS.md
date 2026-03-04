# REFERENCE

https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html

# DOCKER USAGE

## Download docker image

```bash
# docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0
docker pull rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
```

## Docker run container

⚠️ 只有第一次创建的时候运行这个命令！

```bash
bash /workspace1/zhijun/RLinf/docker-run-zhijun_rlinf.sh
```
如果使用了以下命令删除该容器，则要重新创建，原来在容器中配置环境变量，安装的软件会丢失。

```bash
exit
docker stop zhijun_rlinf # 当然在删除之前需要停止该容器的运行
docker rm zhijun_rlinf # 注意删除容器会丢失安装的软件和配置的环境变量
```

## Continue with zhijun_rlinf container

```bash
bash /workspace1/zhijun/RLinf/docker-goon-zhijun_rlinf.sh
```

# QUICK START

## Switch to openpi env

```bash
source switch_env openpi
```

## Run SFT using libero_sft_openpi.yaml

```bash
export RANK=0  # set the rank of the current node
cd /path_to_RLinf/ray_utils
bash start_ray.sh

bash examples/sft/run_embodiment_sft.sh arc_libero_sft_openpi # zhijun edited
bash examples/sft/run_embodiment_sft.sh libero_sft_openpi # og
```

## RL & Eval using libero_goal_ppo_openpi.yaml

```bash
bash examples/embodiment/eval_embodiment.sh arc_libero_goal_ppo_openpi LIBERO # zhijun edited
bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_openpi LIBERO # og
```

# CHECK LOGS

⚠️ server 上的 logs 时间为协调世界时（UTC/GMT+0），较北京/新加坡时间慢 8h。

# REAL-TO-SIM

## BlockPAP-v1

同时渲染初始化位置和大约在第 12s 时的两个图像，并与真机照片对比。

`CAM_T` 当前默认使用手眼标定的内外参的平移向量 "og"，其中内参矩阵的光心 `cx, cy` 经过微调，也可以选用手动对齐机器人基座的 "0302"。

需要自定义的参数是 `TRAJ_ID`，物块初始化位置根据 `TRAJ_ID` 查找字典并构建目标环境。

```bash
python real_franka/real2sim_env/pick_and_place.py

# 多视角渲染
python real_franka/real2sim_env/multiview_render.py \
  --out-dir real_franka/real2sim_env/render/BlockPAP_multiview \
  --env-id BlockPAP-MultiView-v1 \
  --distance-scale 1.35
```

## Replay Action

HDF5 回放：

* 前 7 个关节（手臂）使用高刚度 PD 驱动（K=1e5）严格跟踪轨迹，避免跟踪过程的误差积累；
* 后 2 个关节（夹爪）使用物理仿真 PD 驱动，目标为 `joint_pos[:,7:9]/2`（干净的观测信号，0~0.04）。

更多 args 和全局变量设置，见具体代码。

```bash
python real_franka/real2sim_env/replay_traj.py \
  --traj 15

# delta 动作回放，效果应当一致
python real_franka/real2sim_env/replay_delta.py
  --traj 15

# 与真机视频叠加对比，在 host 环境中，docker 没有 ffmpeg
# TRAJ_LIST = [0, 15, 25, 40, 45]，可以修改处理哪几条视频
python /workspace1/zhijun/RLinf/real_franka/real2sim_env/overlay_videos.py
```