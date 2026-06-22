# 🔥 Environment Setup and Quick Start

> Reference Doc: https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html

## Docker 

Create a container for the first time:

```bash
docker pull rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
bash /workspace1/zhijun/RLinf/docker-run-zhijun_rlinf.sh
```

Continue with the existing `zhijun_rlinf` container:

```bash
bash /workspace1/zhijun/RLinf/docker-goon-zhijun_rlinf.sh
```

After entering the container, you should automatically enter the /workspace1/zhijun/RLinf directory.

Now switch the virtual environment to `openpi`:

```bash
source switch_env openpi
```

To kill a container:

```bash
exit
docker stop zhijun_rlinf # 在删除前需停止运行
docker rm zhijun_rlinf # 删除容器会丢失配置的软件和环境变量
```

⚠️ Notice：
- The server log time is UTC (8 hours slower than Beijing time).
- If encounter permissions issue when modifing files in /workspace1, run alias command `fixperm` in the host terminal for showlab15 (`fixperm_0` for /users directory) to resolve.

## Examples in Embodied Scenarios

### Basic SFT（LIBERO）

```bash
bash examples/sft/run_vla_sft.sh arc_libero_sft_openpi # edited
bash examples/sft/run_vla_sft.sh libero_sft_openpi # og
```

🌟 重要参数解析：
- `action_horizon` 模型一次性预测的动作序列长度
- `num_action_chunks` 是模型实际执行的动作长度
- 

### Eval and RL（LIBERO）

```bash
bash examples/embodiment/eval_embodiment.sh arc_libero_goal_ppo_openpi LIBERO # edited
bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_openpi LIBERO # og
```

# 🧙 Real-to-Sim Co-Training

## 👉 [MIMICGEN_COMMANDS](real_franka/real2sim_env/MIMICGEN_COMMANDS.md) for Simulation Environment Setup, Dataset Generation and Processing

## Real+Sim Co-SFT（BlockPAP-v1_Mix）

首先，注册 `pi05_blockpap_mix`，复用 `franka_co_training_dataconfig.py`，并在 `rlinf/models/embodiment/openpi/dataconfig/__init__.py` 中增加申明：

```python
TrainConfig(
  name="pi05_blockpap_mix",
  model=pi0_config.Pi0Config( ... ),
  data=LeRobotFrankaEEDataConfig( ... ),
  weight_loader=weight_loaders.CheckpointWeightLoader( ... ),
  pytorch_weight_path="checkpoints/torch/pi05_base",
  batch_size=16,
)
```

❗️ 提前计算数据集的全局归一化信息 `norm_stats.json`，并放置在 cfg.model_path/repo_id="BlockPAP-v1_Mix"（不同的数据集分离）

```bash
bash compute_norm_stats.sh
```

❗️ `fsdp_vla_sft_worker.py` 新增 `_DistributedWeightedSampler` 类和 `_build_weighted_openpi_loader` 方法，支持以 `cfg.co_training_ratio` 比例在前 num_real_episodes 个真机数据和 `1 - cfg.co_training_ratio` 比例在剩余数据中抽取。

❗️ BlockPAP-v1 环境配置脚本和 wrapper 中的 `_obs_list_to_state_action` 增加了两个默认开启的本体感知的偏置，避免 MimicGen 生成数据与真机分布不一致，避免训练时需对样本读取做特别处理，更简洁。eval 时单独写了 `get_ee_state()`，他和 `_build_extracted_obs` 读取了同样的数据，但是因为是独立的函数，仍要使用 state_bias 控制。~~真机 state[5] 与仿真存在恒定的 45 度偏差，state[2] 有 10cm 误差， `fsdp_vla_sft_worker.py` 新增 `_SimYawBiasDataset` 类，为 num_real_episodes 后的数据增加偏置，以真机数据对齐，仿真 eval 时需要对 state 做同样的处理。~~


```yaml
# 当前开启这两个参数时，对 simulation data 硬编码了偏置处理
data.num_real_episodes: 50
data.co_training_ratio: 0.5 
# 可以选择开启隔帧构造数据，增大每一个 action 的大小，应当更好学
model.openpi.action_subsample_stride: 2
```

开启训练：

```bash
bash examples/sft/run_vla_sft.sh arc_mix_sft_openpi # mix
bash examples/sft/run_vla_sft.sh arc_mix_sft_resume_openpi # resume

bash examples/sft/run_vla_sft.sh arc_real_sft_openpi
```

仿真检验：

```bash
bash toolkits/eval_scripts_openpi/blockpap_eval.sh
bash toolkits/eval_scripts_openpi/batch_blockpap_eval.sh
```

## 👉 [pi-StepNFT Forked Repo](https://github.com/AaronCaoZJ/pi-StepNFT)

### 训练过程

每个 Global Step 包含四个阶段（假设使用 `num_gpu` 卡训练）：
- 权重同步：Actor 将最新参数广播给所有 Rollout Worker，确保采样策略与训练策略一致。
- Rollout 采样：`num_gpu` 个 env rank 平分 `total_num_envs` 个 env，按照 `max_steps_per_rollout_epoch` 执行 `rollout_epoch` 次，，以 `num_action_chunks` 将连续步切分为 chunk：
  ```
  chunk_per_env = max_steps_per_rollout_epoch ÷ num_action_chunks × rollout_epoch
  total_num_chunk = chunk_per_env × total_num_envs
  ```
- Advantage 计算（terminal-binary），采用 DPO 风格的二值偏好信号，不需要 critic 网络：
  ```
  任务成功 -> advantage = +1.0，广播到该轨迹的所有时间步
  任务失败 -> advantage = −1.0，同上
  ```
- Actor 更新：每个 actor rank 平分到 total_num_chunk / `num_gpu` 个 chunk 样本。每个 optimizer step 消耗所有 rank 各 `global_batch_size` / `num_gpu` 个 chunk 样本，梯度累积步数为 opt_chunk_per_rank / `micro_batch_size`。PPO 风格的多次更新，每次 actor 更新执行 `update_epoch` 次 opt step.
  ```
  每个 rank：total_num_chunk / global_batch_size × update_epoch 优化步
            total_num_chunk × update_epoch / (num_gpu × micro_batch_size) 梯度回传
  整个训练 × num_gpu
  ```
- 每 `save_interval` 个 global step 保存一次 checkpoint。
- 完整训练的总环境交互次数是 `total_num_envs` * `max_epochs` * `max_steps_per_rollout_epoch` * `rollout_epoch`

开启训练：

```bash
cd ../pi-StepNFT
bash examples/embodiment/run_embodiment.sh arc_maniskill_nft_actor_openpi_pi05 # edited maniskill baseline
bash examples/embodiment/run_embodiment.sh arc_blockpap_nft_actor_openpi_pi05 # blockpap nft rl
```

#TODO

## 多相机输入

### 各路径的 key 转换

| 来源 | 转换过程 | 最终 key（送入模型前） |
| :--- | :--- | :--- |
| SFT dataset (RLinf) | HF 数据集存 `back_image` → `RepackTransform` 加前缀 → `FrankaEEInputs` 读取 | `observation/back_image` |
| NFT rollout (pi-StepNFT) | env `extra_view_images` → `obs_processor` 写入 → `FrankaEEInputs` 读取 | `observation/back_image` |
| Eval 脚本 (RLinf) | `get_image(obs, "back_cam")` → 直接赋值 | `observation/back_image` |

> `left_wrist_0_rgb` 对应 back camera，`right_wrist_0_rgb` 对应 wrist camera，命名源自 pi0 原始模型 slot 约定。

### `num_images_in_input` 的实际作用

| 场景 | 作用 |
| :--- | :--- |
| SFT (RLinf) | `_LimitImageInputsDataset` 用此值过滤数据集 view 数量（真正起过滤作用） |
| NFT rollout (pi-StepNFT) | **不**控制模型输入，仅影响 value head 的 `prefix_mask`（`[True]*256*N + [False]*256*(3-N)`） |
| Eval | 不使用 |

需与实际使用的相机数量保持一致，否则 value head prefix_mask 会错误。

### NFT 多相机配置（pi-StepNFT，已实装）

```yaml
actor:
  model:
    openpi:
      num_images_in_input: 1          # 同步更新，供 value head prefix_mask 使用
      image_input_views: ["image"]    # 控制 obs_processor 写哪些 key → 模型实际输入
      # 可选：
      # ["image"]                             → 仅主相机（back_cam 数据被丢弃）
      # ["image", "back_image"]               → 主相机 + back_cam
      # ["image", "back_image", "wrist_image"] → 三相机
```

`image_input_views` 为空时退回旧行为（传所有可用 view）。

### BlockPAP 环境的相机注册

BlockPAP-v1 始终在 `_default_sensor_configs` 注册两台传感器：
- `external_cam`（主相机）→ env obs `main_images`
- `back_cam`（侧后方）→ env obs `extra_view_images`

无论 `image_input_views` 如何设置，env 都产出两路图像。`image_input_views` 只决定哪路数据被写入 `processed_obs` 并送入模型。

### 视频渲染是否显示 back 视角

由 `video_cfg.show_extra_views` 独立控制，与 `image_input_views` 无关（已加入 blockpap yaml）：

```yaml
env:
  train:
    video_cfg:
      show_extra_views: False   # False=只录主相机；True=主相机+back_cam 并排
```

### SFT 多相机配置（RLinf）

```yaml
data:
  image_input_views: ["image", "back_image"]  # _LimitImageInputsDataset 过滤用

actor:
  model:
    openpi:
      num_images_in_input: 2   # 需与 image_input_views 长度一致
```

### Eval 多相机配置（RLinf）

```bash
python blockpap_eval.py --extra_cam back_cam   # 启用 back camera
# 不加 --extra_cam 则只用主相机
```
