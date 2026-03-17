# 1. BlockPAP-v1 (`pick_and_place.py`)

## 🛠️ Simulation Environment Setup

### Task: "pick up the block and place it on the coaster"

当前成功判定：
- XY 对准：block 中心到 coaster 中心的平面距离 < COASTER_RADIUS = 0.043m
- Z 高度：block 中心 Z 落在目标 Z（TABLE_Z + COASTER_THICKNESS + BLOCK_HALF_SIZE[2]）± 0.02m
- 姿态约束：四元数计算得到倾角 block 倾角 ≤ 20°
- 动力学稳定：线速度 < 0.05 m/s；角速度 < 0.5 rad/s
- 释放约束：夹爪已打开（左指关节 + 右指关节 = qpos[7] + qpos[8] > 0.03 m）

相机与轨迹（5 条不同初始化物块位置的真机轨迹）关键参数：
- `CAM_T`: `og` / `0302` / `0303`
- `TRAJ_ID`: `0` / `15` / `25` / `40` / `45` / `random`

其中 TRAJ_ID = random 用于 MimicGen 随机化初始化物块的生成模式。

```bash
# 指定轨迹和相机矩阵，渲染 init 和第 12s 视频帧
python real_franka/real2sim_env/pick_and_place.py

# 多视角渲染
python real_franka/real2sim_env/multiview_render.py \
--out-dir real_franka/real2sim_env/render/BlockPAP_multiview \
--env-id BlockPAP-MultiView-v1 \
--distance-scale 1.35
```

### Replay hdf5

`replay_traj.py` 中的 `hybrid_step()` 分层仿真：
- 前 7 个关节（手臂）使用高刚度 PD 驱动（K=1e5）严格跟踪轨迹，避免跟踪过程的误差积累
- 后 2 个关节（夹爪）使用物理仿真 PD 驱动，目标为 `joint_pos[:,7:9]/2`（干净的观测信号，0~0.04）

```bash
python real_franka/real2sim_env/replay_traj.py \
--traj 15 \
--side-cam side_cam # side_cam 还没有调试对齐

# delta 动作回放，效果应当一致
python real_franka/real2sim_env/replay_delta.py \
--traj 15

# 与真机视频叠加对比，在 host 环境中，docker 没有 ffmpeg
# TRAJ_LIST = [0, 15, 25, 40, 45]，修改以指定处理哪几条视频
python /workspace1/zhijun/RLinf/real_franka/real2sim_env/overlay_videos.py
```

## 🪄 MimicGen Pipeline

使用 MimicGen 生成规模化仿真数据集是 RLinf-Co 中的做法，MimicGen 通过将任务分解为子任务，不同子任务使用不同的目标物作为参考，并对轨迹进行变换操作从而获得新的数据。需要注意的是，这种做法在子任务切换的时候会出现不自然的插值横移，且几乎是难以避免的。

### Prepare src data -> blockpap_cleaned_src.hdf5

其中 `clean_demo()` 参考 `data_convert.py` 中的 `detect_static_segments_advanced()` 对真实轨迹执行静止段检测与清洗：
- 用 obs_ee_pose（8D）的位置/旋转/夹爪变化量逐帧计算运动分数
- 检测到连续 ≥ min_static_frames（默认5帧）且运动分数低的片段，从 mask 中排除
- 将 mask 同步应用到所有字段：actions、states、eef_pose、block_pose、grasped 等
- 重新 latch grasped 信号（保证清洗后 grasped 仍单调变为 1）
- 重新验证 is_success

重放使用 `replay_traj.py` 中的 `hybrid_step()`（手臂高刚度 PD，夹爪物理仿真）。

```bash
python real_franka/real2sim_env/mg_prepare_src_data.py
```

```json
blockpap_cleaned_src.hdf5
├─ [data/]
│  ├─ [demo_0/]
│  │  ├─ actions  (323, 8)  float32  (0.010 MB)
│  │  ├─ [datagen_info/]
│  │  │  ├─ eef_pose  (323, 4, 4)  float32  (0.020 MB)
│  │  │  ├─ gripper_action  (323, 1)  float32  (0.001 MB)
│  │  │  ├─ [object_poses/]
│  │  │  │  ├─ block  (323, 4, 4)  float32  (0.020 MB)
│  │  │  │  └─ target_coaster  (323, 4, 4)  float32  (0.020 MB)
│  │  │  ├─ [subtask_term_signals/]
│  │  │  │  ├─ grasped  (323,)  int32  (0.001 MB)
│  │  │  │  └─ placed  (323,)  int32  (0.001 MB)
│  │  │  └─ target_pose  (323, 4, 4)  float32  (0.020 MB)
│  │  ├─ [obs/]
│  │  │  ├─ ee_pose  (323, 8)  float32  (0.010 MB)
│  │  │  └─ joint_pos  (323, 9)  float32  (0.011 MB)
│  │  └─ states  (323, 28)  float32  (0.035 MB)
│  ├─ [demo_1/]  (same as demo_0, omitted)
│  ├─ [demo_2/]  (same as demo_0, omitted)
│  ├─ [demo_3/]  (same as demo_0, omitted)
│  └─ [demo_4/]  (same as demo_0, omitted)
└─ [mask/]
   └─ all  (5,)  |S6  (0.000 MB)
```

### Generate MimicGen dataset -> LeRobot-v2.1 format

具体 pipeline（`mg_generate_blockpap_data.py`）：
1. 定义 2 个子任务 `grasp` 和 `place`，分别以 block 和 target_coaster 作为参考物
2. 生成循环，每次重置环境，随机初始化物块位置，根据 block 找到最近 src traj
3. 通过参考物变换源轨迹，任务执行期间使用 `subtask_term_signals/grasped` 作为子任务切换标志，`hybrid_step()` 步进仿真
4. env.evaulate() 最终判定，仿真环境设置与 `datagen info` 分别判定任务成功与否
5. 掐头去尾（开头静帧和瞬移帧/末尾静帧）的轨迹清洗策略，收集数据

```bash
bash real_franka/real2sim_env/mg_multi_gpu_run.sh
```

### Real data convert and dataset merge 

❗️ 注意 hdf5 中的图像和视频文件的命名对应：
- camera_left_color  (正前方全景, 对应 1.mp4) -> image
- camera_wrist_color (腕部俯视,  对应 0.mp4) -> wrist_image
- camera_front_color (实为是侧后相机，对应 2.mp4), 不使用

```bash
python real_franka/real_data_convert.py \
  --config /workspace1/zhijun/RLinf/real_franka/config/pick_and_place_config.json
```

```bash
python real_franka/merge_datasets.py
```

❗️ 最终数据集应与 RLinf-Co 示例数据集（Rlinf/RLCo-Example-Mix-Data）保持一致:
- 目前 Mix 数据集中 real 和 mimicgen 的 `episodes_stats` 字段顺序不同，但读取逻辑按字段名匹配，不依赖固定顺序，通常无影响
- 包含 episodes_stats（汇总所有 demo 的特征统计）
- parquet 文件中的 `index` 为全局连续递增，而非按每个 episode 从 0 重新计数。
- image 确定是 PIL 格式而不是 dist，避免训练报错
- 确保使用 `datasets==3.6.0`，`lerobot==0.1.0`，actions、state 字段的数据格式为 `Sequence`，而不是 `List`，若某些 parquet 的 HF metadata 仍是 `List` 而非 `Sequence`，执行：

  ```bash
  python real_franka/fix_parquet_metadata.py \
    /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/BlockPAP-v1_Mix
  ```

💡 Gripper 数据的三个层级：
1. 真实 HDF5 原始数据（EE_pose）的最后一维，表示两指间的总宽度 [0, 0.08]
2. 仿真环境中，per-finger = 0.04 = 完全张开，per-finger = 0.0 = 完全闭合，per-finger 约为 0.02 时则表示夹到了物体，被挡在了两指间总宽度 0.04 的位置
3. 数据转换时，针对 real data，以 0.04 为阈值，区分夹住和没夹住，分别二值化为 0 和 1；针对 MimicGen 生成的数据，同样以 0.04 为阈值，gripper = jpos[:, 7] + jpos[:, 8]，基于下一帧的 gripper >= 0.04 判定这一帧的 action 0 或 1

训练数据中，所有 gripper action 都是二值化 0 和 1，这与 state 的实际连续值不同，也与 ee_delta_pos 不同，在推理和验证时，传入 0，物理仿真就会时夹爪提供让宽度趋向 0 的力，进而抓住物体。

期望的数据集格式：

```json
BlockPAP-v1_Mix
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   └── chunk-001/
│       ├── episode_001000.parquet
│       └── ...
├── videos/
│   ├── chunk-000/
│   │   ├── observation.images.image/
│   │   │   ├── episode_000000.mp4
│   │   │   └── ...
│   │   └── observation.images.wrist_image/
│   │       ├── episode_000000.mp4
│   │       └── ...
│   └── chunk-001/
│       ├── observation.images.image/
│       │   └── ...
│       └── observation.images.wrist_image/
│           └── ...
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

### Utils

`mg_panda_kinematics.py`：通用的 Panda FK/IK，新增任务时按照以下方法书写，即可全部继承通用方法，避免重复造轮子。

```python
class MG_NewTask(PandaKinematicsMixin, MG_EnvInterface):
    def get_object_poses(self): ...
    def get_subtask_term_signals(self): ...
```

`mg_blockpap_interface.py`：继承 PandaKinematicsMixin 和标准的 MG_EnvInterface，获取 BlockPAP-v1 任务中的物块位姿、子任务信号、抓取检验。

`mg_blockpap_wrapper.py`：MimicGen 期望 env 是 robomimic 的 `EnvBase`，该 wrapper 负责把 ManiSkill（gymnasium 风格）接口翻译成 robomimic 风格。

```json
入口（谁在驱动）
MimicGen DataGenerator

任务接口层（回答“目标是什么、什么时候切子任务”）
MG_BlockPAP  <- 文件: mg_blockpap_interface.py
  - get_object_poses()              # BlockPAP 任务专用
  - get_subtask_term_signals()      # BlockPAP 任务专用
  - get_robot_eef_pose()            # 来自 PandaKinematicsMixin
  - target_pose_to_action()         # 来自 PandaKinematicsMixin
  - action_to_target_pose()         # 来自 PandaKinematicsMixin

环境适配层（回答“动作怎么在仿真里执行”）
EnvManiskillBlockPAP  <- 文件: mg_blockpap_wrapper.py
  - step()
    -> denormalize_joints()         # 来自 PandaKinematicsMixin
    -> hybrid_step()                # 来自 replay_traj
  - reset_to() / get_state()        # BlockPAP 专用（28D state）

`DataGenerator` 每步先通过 `MG_BlockPAP` 决定目标与子任务状态，再通过 `EnvManiskillBlockPAP.step()` 执行动作并推进仿真
```

## 已弃用！Generate MimicGen dataset -> HDF5

该路径先生成 MimicGen HDF5，再做可视化，不是当前 RLinf-Co 使用的 LeRobot v2.1 直出格式。

```bash
bash real_franka/real2sim_env/mg_run.sh

# 可以独立使用的视频渲染脚本，可以同时指定 demo_x 查看 external_cam_image 内容
bash real_franka/real2sim_env/mg_visualize.sh
```