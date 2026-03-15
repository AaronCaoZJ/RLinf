# 🔥 Environment Setup and Quick Start

> Reference Doc: https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html

## Docker 

Create a container for the first time:

```bash
docker pull rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
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

👉 Notice：

- The server log time is UTC (8 hours slower than Beijing time).
- If encounter permissions issue when modifing files in /workspace1, run `fix_perm_15` in the host terminal (for showlab15) to resolve.

## Examples in Embodied Scenarios

### Basic SFT（LIBERO）

```bash
bash examples/sft/run_vla_sft.sh arc_libero_sft_openpi # edited
bash examples/sft/run_vla_sft.sh libero_sft_openpi # og
```

### Eval and RL（LIBERO）

```bash
bash examples/embodiment/eval_embodiment.sh arc_libero_goal_ppo_openpi LIBERO # edited
bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_openpi LIBERO # og
```

# 🧙 Real-to-Sim Co-Training

## Simulation Environment Setup, Dataset Generation and Processing 👉 [MIMICGEN_COMMANDS](real_franka/real2sim_env/MIMICGEN_COMMANDS.md)

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

❗️ 真机 state[5] 与仿真存在恒定的 45 度偏差，state[2] 有 10cm 误差， `fsdp_vla_sft_worker.py` 新增 `_SimYawBiasDataset` 类，为 num_real_episodes 后的数据增加偏置，以真机数据对齐，仿真 eval 时需要对 state 做同样的处理。

开启训练：

```bash
bash examples/sft/run_vla_sft.sh arc_mix_sft_openpi # mix
bash examples/sft/run_vla_sft.sh arc_mix_sft_resume_openpi # resume

```

仿真检验：

```bash
bash toolkits/eval_scripts_openpi/blockpap_eval.sh
```
