# BlockPAP-v1

## Prepare data generation config

源轨迹处理的质量影响生成轨迹的效果，在这一步对真机数据的过滤静止帧和 episode 长度检查。

`clean_demo()` 对应参考的 `data_convert.py` 中的 `detect_static_segments_advanced()`，具体的做了：

* 用 obs_ee_pose（真机数据，8D）的位置/旋转/夹爪变化量逐帧计算运动分数
* 检测连续 ≥ min_static_frames（默认5帧）且运动分数低的片段，从 mask 中排除
* 将 mask 同步应用到所有字段：actions、states、eef_pose、block_pose、grasped 等
* 重新 latch grasped 信号（保证清洗后 grasped 仍单调变为1）
* 重新验证 is_success

```bash
cd /workspace1/zhijun/RLinf
python real_franka/real2sim_env/mg_prepare_src_data.py
```

其中，`replay_and_extract()` 调用 `replay_traj.py` 中的高刚度 PD 分层控制（重放关节物理仿真夹爪）`hybrid_step()` 仿真步进。

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

## Generate MimicGen dataset - LeRobot 2.1 .parquet data

使用 MimicGen 生成规模化仿真数据集是 RLinf-Co 中的做法，MimicGen 通过将任务分解为子任务，不同子任务使用不同的目标物作为参考，并对轨迹进行变换操作从而获得新的数据。需要注意的是，这种做法在子任务切换的时候会出现不自然的插值横移，且几乎是难以避免的。

具体 pipeline：

1. 定义 2 个子任务 grasp 和 place，分别以 block 和 target_coaster 作为参考物
2. 加载 src data
3. 生成循环，每次重置环境，随机初始化 block 和 coaster 位置，找到最近邻源轨迹，通过参考物变换源轨迹，`hybrid_step()` 步进仿真，收集数据
4. env evaulate

轨迹清洗策略：

* 开头固定跳过前 2 帧（head_min_skip=2，覆盖 MimicGen 的初始插值帧），继续往后检查，若发现 EE 位移 > 5 cm（jump_thresh=0.05）的帧则继续跳过，直到没有大跳变为止
* 末尾向前扫描，遇到连续 3 帧（tail_min_frames=3）位移 < 1 mm（tail_motion_thresh=0.001）则截断那段静止帧

```bash
bash real_franka/real2sim_env/mg_multi_gpu_run.sh
```

生成结果直接与 RLinf-Co 示例数据集（LeRobot 2.1）形式一致，其中包含 episodes_stats (所有 demo 中的 feature 统计量)，.parquet 文件的 index 是依次递增的，而不是每一集从 0 开始。
：

```json
blockpap_cleaned_mimicgen
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

```json
info.json
{
  "codebase_version": "v2.1",
  "robot_type": "franka_panda",
  "total_episodes": xxx,
  "total_frames": xxx,
  "total_tasks": 1,
  "total_videos": xxx,
  "total_chunks": xxx,
  "chunks_size": 500,
  "fps": 20.0,
  "splits": { "train": "0:xxx" },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": {
    "observation.images.image": {
      "names": [ "channel", "height", "width" ],
      "dtype": "video",
      "shape": [ 3, 480, 640 ],
      "info": {
        "video.height": 480,
        "video.width": 640,
        "video.codec": "libx264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "video.fps": 20.0,
        "video.channels": 3,
        "has_audio": false
      }
    },
    "observation.images.wrist_image": {
      "names": [ "channel", "height", "width" ],
      "dtype": "video",
      "shape": [ 3, 480, 640 ],
      "info": {
        "video.height": 480,
        "video.width": 640,
        "video.codec": "libx264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "video.fps": 20.0,
        "video.channels": 3,
        "has_audio": false
      }
    },
    "image": {
      "dtype": "image",
      "shape": [ 480, 640, 3 ],
      "names": [ "height", "width", "channel" ]
    },
    "wrist_image": {
      "dtype": "image",
      "shape": [ 480, 640, 3 ],
      "names": [ "height", "width", "channel" ]
    },
    "state": {
      "dtype": "float32",
      "shape": [ 7 ],
      "names": [ "ee_pose_and_gripper_width" ]
    },
    "actions": {
      "dtype": "float32",
      "shape": [ 7 ],
      "names": [ "delta_ee_pose_and_gripper_action" ]
    },
    "timestamp": {
      "dtype": "float32",
      "shape": [ 1 ],
      "names": null
    },
    "frame_index": {
      "dtype": "int64",
      "shape": [ 1 ],
      "names": null
    },
    "episode_index": {
      "dtype": "int64",
      "shape": [ 1 ],
      "names": null
    },
    "index": {
      "dtype": "int64",
      "shape": [ 1 ],
      "names": null
    },
    "task_index": {
      "dtype": "int64",
      "shape": [ 1 ],
      "names": null
    }
  }
}
```

```json
============================================================
  Parquet 结构: episode_000000.parquet
============================================================
  总行数 (timesteps): 159
  总列数            : 9

  image           image  640×480 RGB  total=13189.0 KB
  wrist_image     image  640×480 RGB  total=150.9 KB
  state           shape=(159,7)  dtype=float32  (0.018 MB)
  actions         shape=(159,7)  dtype=float32  (0.018 MB)
  timestamp       scalar  dtype=float  (0.7 KB)
  frame_index     scalar  dtype=int64  (1.4 KB)
  episode_index   scalar  dtype=int64  (1.4 KB)
  index           scalar  dtype=int64  (1.4 KB)
  task_index      scalar  dtype=int64  (1.4 KB)
```

## Merged Dataset

同理对原始 real 数据做数据集格式转换，并合并 real 和 mimicgen 数据集。

需要注意的是，real 和 mimicgen 两个 dataset 的 episode_stats key 顺序不同，但是实际读取使用的是字段匹配，而不是严格顺序，所以问题不大。

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

## 已弃用！Generate MimicGen dataset - HDF5

原始实现首先生成 HDF5，并在最后回放部分 demo 得到视频演示。

```bash
bash real_franka/real2sim_env/mg_run.sh

# 可以独立使用的视频渲染脚本，可以同时指定 demo_x 查看 external_cam_image 内容
bash real_franka/real2sim_env/mg_visualize.sh
```

因为与 RLinf-Co 的示例数据集需要的形式不符，已弃用。

输出 HDF5 包含：

```json
============================================================
  HDF5 结构: blockpap_multi_gpu_gen.hdf5
============================================================
└─ [data/]
   ├─ [demo_0/]
   │  ├─ actions  (318, 8)  float32  (0.010 MB)
   │  ├─ [datagen_info/]
   │  │  ├─ eef_pose  (318, 4, 4)  float64  (0.039 MB)
   │  │  ├─ gripper_action  (318, 1)  float32  (0.001 MB)
   │  │  ├─ [object_poses/]
   │  │  │  ├─ block  (318, 4, 4)  float64  (0.039 MB)
   │  │  │  └─ target_coaster  (318, 4, 4)  float64  (0.039 MB)
   │  │  ├─ [subtask_term_signals/]
   │  │  │  ├─ grasped  (318,)  int64  (0.002 MB)
   │  │  │  └─ lifted  (318,)  int64  (0.002 MB)
   │  │  └─ target_pose  (318, 4, 4)  float64  (0.039 MB)
   │  ├─ [obs/]
   │  │  ├─ block_pos  (318, 3)  float32  (0.004 MB)
   │  │  ├─ block_quat  (318, 4)  float32  (0.005 MB)
   │  │  ├─ coaster_pos  (318, 3)  float32  (0.004 MB)
   │  │  ├─ ee_pos  (318, 3)  float32  (0.004 MB)
   │  │  ├─ ee_quat  (318, 4)  float32  (0.005 MB)
   │  │  ├─ external_cam_image  (318, 480, 640, 3)  uint8  (279.492 MB)
   │  │  └─ joint_pos  (318, 9)  float32  (0.011 MB)
   │  ├─ src_demo_inds  (2,)  int64  (0.000 MB)
   │  ├─ src_demo_labels  (318, 1)  int64  (0.002 MB)
   │  └─ states  (318, 28)  float32  (0.034 MB)
```