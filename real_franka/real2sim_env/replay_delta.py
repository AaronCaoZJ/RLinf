#!/usr/bin/env python3
"""
Delta-action trajectory replay: high-stiffness PD arm + physics gripper.

Same as replay_traj.py but uses incremental (delta) actions instead of
absolute joint position targets.

  arm_delta[t]  = joint_pos[t+1, :7] - joint_pos[t, :7]
  grip_delta[t] = (joint_pos[t+1, 7:9] - joint_pos[t, 7:9]) / 2

At each step the accumulated target is updated:
  current_arm  += arm_delta[t]
  current_grip += grip_delta[t]
and then set as the PD drive target.
"""
import argparse
import os

import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch

import pick_and_place  # noqa: F401 – register env


# ── Data loading ────────────────────────────────────────────────────────────

def load_joint_trajectory(h5_path: str, dataset_key: str = "auto") -> tuple[np.ndarray, str]:
    """Load joint trajectory from HDF5. Returns (T, D) array and key used."""
    with h5py.File(h5_path, "r") as f:
        if dataset_key != "auto":
            if dataset_key in f:
                return f[dataset_key][:], dataset_key
            if "observations" in f and dataset_key in f["observations"]:
                return f["observations"][dataset_key][:], f"observations/{dataset_key}"
            raise KeyError(f"Dataset '{dataset_key}' not found in {h5_path}")

        for group, ds in [("observations", "joint_pos"), ("observations", "full_joint_pos")]:
            if group in f and ds in f[group]:
                return f[group][ds][:], f"{group}/{ds}"
        for ds in ("joint_pos", "full_joint_pos"):
            if ds in f:
                return f[ds][:], ds

    raise KeyError("No joint trajectory found in HDF5")


def ensure_2d(qpos: np.ndarray) -> np.ndarray:
    if qpos.ndim == 3 and qpos.shape[1] == 1:
        qpos = qpos[:, 0, :]
    if qpos.ndim != 2:
        raise ValueError(f"Expected shape (T, D), got {qpos.shape}")
    return qpos


# ── Rendering helper ────────────────────────────────────────────────────────

def to_uint8_frame(frame) -> np.ndarray | None:
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    return frame


def capture_frame(env, base_env, side_cam_name: str | None = None) -> np.ndarray | None:
    """Render main view; optionally concatenate a side sensor camera."""
    main = to_uint8_frame(env.render())
    if main is None or side_cam_name is None:
        return main
    obs = base_env.get_obs()
    side = to_uint8_frame(obs["sensor_data"][side_cam_name]["rgb"])
    if side is None:
        return main
    if main.shape[0] != side.shape[0]:
        h = min(main.shape[0], side.shape[0])
        main, side = main[:h], side[:h]
    return np.concatenate([main, side], axis=1)


# ── Hybrid replay core ─────────────────────────────────────────────────────

def setup_hybrid_drives(base_env,
                        arm_stiffness: float = 1e5,
                        arm_damping: float = 1e3,
                        gripper_stiffness: float = 2000.0,
                        gripper_damping: float = 100.0,
                        gripper_force_limit: float = 500.0):
    active_joints = base_env.agent.robot.get_active_joints()
    for i, joint in enumerate(active_joints):
        if i < 7:
            joint.set_drive_properties(
                arm_stiffness, arm_damping, force_limit=1e6, mode="force"
            )
        else:
            joint.set_drive_properties(
                gripper_stiffness, gripper_damping,
                force_limit=gripper_force_limit, mode="force",
            )


def hybrid_step(base_env, arm_target: np.ndarray, gripper_target: np.ndarray,
                sim_steps: int = 5):
    robot = base_env.agent.robot
    device = base_env.device

    arm_t = torch.tensor(arm_target, device=device, dtype=torch.float32)
    grip_t = torch.tensor(gripper_target, device=device, dtype=torch.float32)

    all_targets = torch.cat([arm_t, grip_t]).unsqueeze(0)  # (1, 9)
    all_joints = robot.get_active_joints()
    robot.set_joint_drive_targets(all_targets, all_joints)

    for _ in range(sim_steps):
        base_env.scene.step()

    base_env.scene.update_render()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Delta-action replay: kinematic arm + physics gripper",
    )
    parser.add_argument("--traj", type=str, default="0",
                        choices=["0", "15", "25", "40", "45"],
                        help="轨迹 ID，自动推断 HDF5 路径和输出路径，并加载对应初始化配置")
    parser.add_argument("--h5", type=str, default=None,
                        help="HDF5 文件路径，默认根据 --traj 自动推断")
    parser.add_argument("--out", type=str, default=None,
                        help="输出视频路径，默认根据 --traj 自动推断")
    parser.add_argument("--dataset-key", type=str, default="auto")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--sim-steps", type=int, default=5,
                        help="Physics substeps per trajectory frame")
    parser.add_argument("--gripper-stiffness", type=float, default=2000.0)
    parser.add_argument("--gripper-damping", type=float, default=100.0)
    parser.add_argument("--gripper-force-limit", type=float, default=500.0)
    parser.add_argument("--side-view", action="store_true",
                        help="Concatenate left-side camera view to output video")
    parser.add_argument("--cam-t", type=str, default="og",
                        choices=["og", "0302", "0303"],
                        help="相机平移向量预设: 'og'、'0302'、'0303'")
    args = parser.parse_args()

    # 根据 --traj 自动推断路径
    _HDF5_BASE = "/storage/zhijun/real_franka/pick_and_place"
    _RENDER_BASE = "real_franka/real2sim_env/render"
    if args.h5 is None:
        args.h5 = f"{_HDF5_BASE}/episode_{args.traj}.hdf5"
    if args.out is None:
        args.out = f"{_RENDER_BASE}/traj{args.traj}/{args.cam_t}/replay_delta.mp4"

    if not os.path.exists(args.h5):
        raise FileNotFoundError(args.h5)

    # ── Load trajectory ──────────────────────────────────────────────────
    qpos_raw, used_key = load_joint_trajectory(args.h5, args.dataset_key)
    qpos = ensure_2d(np.asarray(qpos_raw, dtype=np.float32))
    print(f"Loaded: {qpos.shape} from '{used_key}'")

    if qpos.shape[1] < 9:
        raise ValueError(f"Need ≥9 dims, got {qpos.shape[1]}")
    qpos = qpos[:, :9]

    # ── Gripper: total-width → per-finger ──────────────────────────────────
    # joint_pos[:,7:9] stores total-width (0~0.08); /2 gives per-finger target.
    qpos[:, 7:9] = qpos[:, 7:9] / 2.0

    if args.max_frames > 0:
        qpos = qpos[: args.max_frames]

    # ── Compute deltas ──────────────────────────────────────────────────────
    # arm_deltas[t]  = qpos[t+1, :7] - qpos[t, :7]
    # grip_deltas[t] = qpos[t+1, 7:9] - qpos[t, 7:9]
    arm_deltas  = np.diff(qpos[:, :7],  axis=0)  # (T-1, 7)
    grip_deltas = np.diff(qpos[:, 7:9], axis=0)  # (T-1, 2)
    T = len(arm_deltas)

    print(f"Arm  delta range:    [{arm_deltas.min():.6f}, {arm_deltas.max():.6f}]")
    print(f"Grip delta range:    [{grip_deltas.min():.6f}, {grip_deltas.max():.6f}]")
    print(f"Total replay steps:  {T}")

    # ── 注入初始化配置 ───────────────────────────────────────────────────
    pick_and_place.TRAJ_ID = args.traj
    pick_and_place.USE_REF_12 = False   # 回放始终从 t=0 姿态出发
    print(f"TRAJ_ID={args.traj}, USE_REF_12=False → 使用 _TRAJ_CONFIGS['{args.traj}'] 初始化")

    # ── Create env ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env_id = "BlockPAP-v1"
    side_cam_name = None
    if args.side_view:
        import RLinf.real_franka.real2sim_env.multiview_render as multiview_render  # noqa: F401
        env_id = "BlockPAP-SideView-v1"
        side_cam_name = "cam_side_left"
    env = gym.make(env_id, obs_mode=args.obs_mode, render_mode=args.render_mode,
                   cam_t=args.cam_t)
    print(f"Camera _t preset: '{args.cam_t}'")
    obs, _ = env.reset()
    base_env = env.unwrapped

    # ── Configure hybrid drives ──────────────────────────────────────────
    setup_hybrid_drives(
        base_env,
        arm_stiffness=1e5,
        arm_damping=1e3,
        gripper_stiffness=args.gripper_stiffness,
        gripper_damping=args.gripper_damping,
        gripper_force_limit=args.gripper_force_limit,
    )
    print(f"Hybrid mode: arm PD (K=1e5, D=1e3), gripper PD "
          f"(K={args.gripper_stiffness}, D={args.gripper_damping}, "
          f"F={args.gripper_force_limit})")

    # ── Initialize to qpos[0] ───────────────────────────────────────────
    robot = base_env.agent.robot
    q0 = torch.tensor(qpos[0], device=base_env.device, dtype=torch.float32).unsqueeze(0)
    robot.set_qpos(q0)
    robot.set_qvel(torch.zeros((1, 9), device=base_env.device))
    robot.set_joint_drive_targets(q0, robot.get_active_joints())
    base_env.scene.step()
    base_env.scene.update_render()

    frames = [capture_frame(env, base_env, side_cam_name)]
    print(f"Replaying {T} delta steps (sim_steps={args.sim_steps}) → {args.out}")

    # ── Delta replay loop ────────────────────────────────────────────────
    # Accumulate from qpos[0]; apply delta[t] to get the target for step t+1.
    current_arm  = qpos[0, :7].copy()
    current_grip = qpos[0, 7:9].copy()

    for t in range(T):
        current_arm  += arm_deltas[t]
        current_grip += grip_deltas[t]
        hybrid_step(base_env, current_arm, current_grip, sim_steps=args.sim_steps)
        frame = capture_frame(env, base_env, side_cam_name)
        if frame is not None:
            frames.append(frame)

    # ── Save video ───────────────────────────────────────────────────────
    if not frames:
        raise RuntimeError("No frames captured – check render_mode / renderer")

    imageio.mimsave(args.out, frames, fps=args.fps)
    env.close()
    print(f"Done. {len(frames)} frames @ {args.fps} fps → {args.out}")


if __name__ == "__main__":
    main()
