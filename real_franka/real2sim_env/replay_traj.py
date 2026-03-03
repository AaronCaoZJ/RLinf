#!/usr/bin/env python3
"""
Hybrid trajectory replay: high-stiffness PD arm + physics gripper.

Replays joint trajectory from HDF5 in the BlockPAP-v1 ManiSkill environment.
  - Arm joints (0-6): very high-K PD drive (K=1e5) tracks targets through the physics
    solver — contacts are maintained continuously so friction can carry the block upward.
  - Gripper joints (7-8): normal PD drives; targets = joint_pos[:,7:9]/2 (observed
    total-width 0~0.08 → per-finger 0~0.04, clean signal with no anomalies).
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
    """
    Configure joint drives for hybrid replay:
      - Arm (joints 0-6): very high stiffness PD drive (K=1e5, D=1e3, F=1e6).
        The arm tracks set_joint_drive_targets almost perfectly (<0.3deg error)
        while going through the physics solver — so contacts are maintained
        continuously and friction can carry the block upward.
      - Gripper (joints 7-8): normal PD drives for physics grasping.
    """
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
    """
    One hybrid replay step:
      Set arm + gripper PD drive targets, then run sim_steps physics substeps.
      The arm's high-stiffness drive tracks arm_target with <0.3deg error while
      going through the physics solver — contacts are maintained continuously,
      and friction can carry the block upward as the arm rises.
    """
    robot = base_env.agent.robot
    device = base_env.device

    arm_t = torch.tensor(arm_target, device=device, dtype=torch.float32)
    grip_t = torch.tensor(gripper_target, device=device, dtype=torch.float32)

    # Set drive targets for all joints (arm: high-K tracking; gripper: squeeze)
    all_targets = torch.cat([arm_t, grip_t]).unsqueeze(0)  # (1, 9)
    all_joints = robot.get_active_joints()
    robot.set_joint_drive_targets(all_targets, all_joints)

    for _ in range(sim_steps):
        base_env.scene.step()

    base_env.scene.update_render()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid replay: kinematic arm + physics gripper",
    )
    parser.add_argument("--h5", type=str,
                        default="/storage/zhijun/real_franka/pick_and_place/episode_0.hdf5")
    parser.add_argument("--out", type=str,
                        default="real_franka/real2sim_env/render/BlockPAP-v1_traj_0.mp4")
    parser.add_argument("--dataset-key", type=str, default="auto")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--sim-steps", type=int, default=5,
                        help="Physics substeps per trajectory frame (more = smoother gripper)")
    parser.add_argument("--gripper-stiffness", type=float, default=2000.0)
    parser.add_argument("--gripper-damping", type=float, default=100.0,
                        help="PD damping for gripper joints (keep ~100, not 2000)")
    parser.add_argument("--gripper-force-limit", type=float, default=500.0)
    parser.add_argument("--side-view", action="store_true",
                        help="Concatenate left-side camera view to output video")
    parser.add_argument("--t", type=str, default="og",
                        choices=["og", "0302", "0303"],
                        help="相机平移向量预设: 'og'（2026-02-15）、'0302'（2026-03-02）、'0303'（2026-03-03）")
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(args.h5)

    # ── Load trajectory ──────────────────────────────────────────────────
    qpos_raw, used_key = load_joint_trajectory(args.h5, args.dataset_key)
    qpos = ensure_2d(np.asarray(qpos_raw, dtype=np.float32))
    print(f"Loaded: {qpos.shape} from '{used_key}'")

    if qpos.shape[1] < 9:
        raise ValueError(f"Need ≥9 dims, got {qpos.shape[1]}")
    qpos = qpos[:, :9]

    # ── Gripper targets: joint_pos[:,7:9]/2 ────────────────────────────────
    # joint_pos stores total-width (0~0.08); per-finger sim target = /2 (0~0.04).
    # During grasp the observed value ~0.039 → target 0.0195, so the PD spring
    # continuously squeezes the block. This signal is clean with no anomalies.
    qpos[:, 7:9] = qpos[:, 7:9] / 2.0
    print(f"Gripper targets (joint_pos/2): "
          f"min={qpos[:,7:9].min():.4f}, max={qpos[:,7:9].max():.4f}")

    if args.max_frames > 0:
        qpos = qpos[: args.max_frames]

    print(f"Arm range:     [{qpos[:, :7].min():.4f}, {qpos[:, :7].max():.4f}]")
    print(f"Gripper range: [{qpos[:, 7:9].min():.6f}, {qpos[:, 7:9].max():.6f}]")

    # ── Create env ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env_id = "BlockPAP-v1"
    side_cam_name = None
    if args.side_view:
        import RLinf.real_franka.real2sim_env.multiview_render as multiview_render  # noqa: F401
        env_id = "BlockPAP-SideView-v1"
        side_cam_name = "cam_side_left"
    env = gym.make(env_id, obs_mode=args.obs_mode, render_mode=args.render_mode,
                   cam_t=args.t)
    print(f"Camera _t preset: '{args.t}'")
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
    # set_qpos to place the robot, then set drive targets = q0 so the high-K
    # arm drive doesn't immediately try to pull away from the initial pose.
    robot = base_env.agent.robot
    q0 = torch.tensor(qpos[0], device=base_env.device, dtype=torch.float32).unsqueeze(0)
    robot.set_qpos(q0)
    robot.set_qvel(torch.zeros((1, 9), device=base_env.device))
    robot.set_joint_drive_targets(q0, robot.get_active_joints())
    base_env.scene.step()
    base_env.scene.update_render()

    frames = [capture_frame(env, base_env, side_cam_name)]
    print(f"Replaying {qpos.shape[0]} frames (sim_steps={args.sim_steps}) → {args.out}")
    if args.side_view:
        print(f"Side view enabled: {side_cam_name}")

    # ── Hybrid replay loop ───────────────────────────────────────────────
    for t in range(qpos.shape[0]):
        hybrid_step(base_env, qpos[t, :7], qpos[t, 7:9], sim_steps=args.sim_steps)
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
