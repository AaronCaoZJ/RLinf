# BlockPAP-v1

## Prepare data generation config

```bash
cd /workspace1/zhijun/RLinf
python real_franka/real2sim_env/mg_prepare_src_data.py
```

输出新 HDF5 文件包含：

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

使用 MimicGen 生成规模化仿真数据集是 RLinf-Co 中的做法，MimicGen 通过将任务分解为子任务，不同子任务使用不同的目标物作为参考，并对轨迹进行变换操作从而获得新的数据。需要注意的是，这种做法在子任务切换的时候会出现不自然的插值横移，且几乎是难以避免的。

```bash
bash real_franka/real2sim_env/mg_run.sh

# 如果需要单独渲染视频，或对生成 HDF5 中的 obs/external_cam_image 做检查
bash real_franka/real2sim_env/mg_visualize.sh
```

得到的生成数据集 HDF5 包含：

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
