# BlockPAP-v1

## Prepare data generation config

```bash
cd /workspace1/zhijun/RLinf
python real_franka/real2sim_env/mg_prepare_src_data.py
```

其中，`replay_and_extract()` 调用 `replay_traj.py` 中的高刚度 PD 分层控制（重放关节物理仿真夹爪）`hybrid_step()` 仿真步进

写入 MimicGen 格式的新 HDF5 文件：

```json
============================================================
  HDF5 结构: blockpap_src.hdf5
============================================================
├─ [data/]
│  ├─ [demo_0/]
│  │  ├─ actions  (434, 8)  float32  (0.013 MB)
│  │  ├─ [datagen_info/]
│  │  │  ├─ eef_pose  (434, 4, 4)  float32  (0.026 MB)
│  │  │  ├─ gripper_action  (434, 1)  float32  (0.002 MB)
│  │  │  ├─ [object_poses/]
│  │  │  │  ├─ block  (434, 4, 4)  float32  (0.026 MB)
│  │  │  │  └─ target_coaster  (434, 4, 4)  float32  (0.026 MB)
│  │  │  ├─ [subtask_term_signals/]
│  │  │  │  ├─ grasped  (434,)  int32  (0.002 MB)
│  │  │  │  ├─ lifted  (434,)  int32  (0.002 MB)
│  │  │  │  └─ placed  (434,)  int32  (0.002 MB)
│  │  │  └─ target_pose  (434, 4, 4)  float32  (0.026 MB)
│  │  ├─ [obs/]
│  │  │  ├─ ee_pose  (434, 8)  float32  (0.013 MB)
│  │  │  └─ joint_pos  (434, 9)  float32  (0.015 MB)
│  │  └─ states  (434, 28)  float32  (0.046 MB)
│  ├─ [demo_1/]  (same as demo_0, omitted)
│  ├─ [demo_2/]  (same as demo_0, omitted)
│  ├─ [demo_3/]  (same as demo_0, omitted)
│  └─ [demo_4/]  (same as demo_0, omitted)
└─ [mask/]
   └─ all  (5,)  |S6  (0.000 MB)
```

## Generate MimicGen dataset

```bash
bash real_franka/real2sim_env/mg_run.sh

# 如果需要单独渲染视频，或对生成 HDF5 中的 obs/{camera_name}_image 做检查
bash real_franka/real2sim_env/mg_visualize.sh
```

使用 MimicGen 生成规模化仿真数据集是 RLinf-Co 中的做法，MimicGen 通过将任务分解为子任务，不同子任务使用不同的目标物作为参考，并对轨迹进行变换操作从而获得新的数据。需要注意的是，这种做法在子任务切换的时候会出现不自然的插值横移，且几乎是难以避免的。

具体 pipeline：

1. 定义 2 个子任务 grasp 和 place，分别以 block 和 target_coaster 作为参考物
2. 加载 src data
3. 生成循环，每次重置环境，随机初始化 block 和 coaster 位置，找到最近邻源轨迹，通过参考物变换源轨迹，`hybrid_step()` 步进仿真，收集数据
4. env evaulate，得到的生成数据集 HDF5 包含：

```json
============================================================
  HDF5 结构: blockpap_gen.hdf5
============================================================
└─ [data/]
   ├─ [demo_0/]
   │  ├─ actions  (446, 8)  float32  (0.014 MB)
   │  ├─ [datagen_info/]
   │  │  ├─ eef_pose  (446, 4, 4)  float64  (0.054 MB)
   │  │  ├─ gripper_action  (446, 1)  float32  (0.002 MB)
   │  │  ├─ [object_poses/]
   │  │  │  ├─ block  (446, 4, 4)  float64  (0.054 MB)
   │  │  │  └─ target_coaster  (446, 4, 4)  float64  (0.054 MB)
   │  │  ├─ [subtask_term_signals/]
   │  │  │  ├─ grasped  (446,)  int64  (0.003 MB)
   │  │  │  └─ lifted  (446,)  int64  (0.003 MB)
   │  │  └─ target_pose  (446, 4, 4)  float64  (0.054 MB)
   │  ├─ [obs/]
   │  │  ├─ block_pos  (446, 3)  float32  (0.005 MB)
   │  │  ├─ block_quat  (446, 4)  float32  (0.007 MB)
   │  │  ├─ coaster_pos  (446, 3)  float32  (0.005 MB)
   │  │  ├─ ee_pos  (446, 3)  float32  (0.005 MB)
   │  │  ├─ ee_quat  (446, 4)  float32  (0.007 MB)
   │  │  ├─ external_cam_image  (446, 480, 640, 3)  uint8  (391.992 MB)
   │  │  └─ joint_pos  (446, 9)  float32  (0.015 MB)
   │  ├─ src_demo_inds  (2,)  int64  (0.000 MB)
   │  ├─ src_demo_labels  (446, 1)  int64  (0.003 MB)
   │  └─ states  (446, 28)  float32  (0.048 MB)
   ├─ [demo_1/]  (same as demo_0, omitted)
   ├─ [demo_2/]  (same as demo_0, omitted)
   ├─ [demo_3/]  (same as demo_0, omitted)
   ├─ [demo_4/]  (same as demo_0, omitted)
   ├─ [demo_5/]  (same as demo_0, omitted)
   ├─ [demo_6/]  (same as demo_0, omitted)
   ├─ [demo_7/]  (same as demo_0, omitted)
   ├─ [demo_8/]  (same as demo_0, omitted)
   └─ [demo_9/]  (same as demo_0, omitted)
```

## Utils

`mg_panda_kinematics.py`：通用的 Panda FK/IK，新增任务时按照以下方法书写，即可全部继承通用方法，避免重复造轮子。

```python
class MG_NewTask(PandaKinematicsMixin, MG_EnvInterface):
    def get_object_poses(self): ...
    def get_subtask_term_signals(self): ...
```

`mg_blockpap_interface.py`：继承 PandaKinematicsMixin 和标准的 MG_EnvInterface，获取 BlockPAP-v1 任务中的物块位姿、子任务信号、抓取检验。

`mg_bockpap_wrapper.py`：MimicGen 在调用 env 的时候期望的是 robomimic 的 EnvBase，wrapper 提供了 Maniskill 的 gymnasium 风格到 robomimic 风格的翻译功能。

```json
# 调用关系
MimicGen DataGenerator
└─ MG_BlockPAP (interface)
  ├─ get_robot_eef_pose()      ┐
  ├─ target_pose_to_action()   ├─ 全部来自 PandaKinematicsMixin
  ├─ action_to_target_pose()   ┘
  ├─ get_object_poses()        ← BlockPAP 专用
  └─ get_subtask_term_signals()← BlockPAP 专用

└─ EnvManiskillBlockPAP (wrapper)
  ├─ step() → denormalize_joints() [来自 mg_panda_kinematics]
  │         → hybrid_step() [来自 replay_traj]
  └─ reset_to() / get_state()  ← BlockPAP 专用（28D state）
```