#!/usr/bin/env python3
"""
Convert real Franka pick-and-place trajectories into a MimicGen-compatible
source HDF5 dataset.

Pipeline:
  1. Load each real robots HDF5 (joint_pos, ee_pose, etc.)
  2. Replay in ManiSkill BlockPAP-v1 simulation
  3. At each timestep extract: EEF pose, object poses, subtask termination
     signals, target EEF pose, gripper action, and full sim state
  4. Write to a single HDF5 in the format expected by MimicGen

Usage:
    cd /workspace1/zhijun/RLinf
    python real_franka/real2sim_env/prepare_mimicgen_data.py \
        --trajs 0 15 25 40 45 \
        --output /storage/zhijun/real_franka/mimicgen_src/blockpap_src.hdf5

The output HDF5 has the structure MimicGen expects:
    data/
      demo_0/
        actions            (T, 8)    - [abs_arm_joint_pos(7), gripper_cmd(1)]
        states             (T, ...)  - serialized sim states for replay
        obs/
          joint_pos        (T, 9)
          ee_pose          (T, 8)    - [x,y,z, qw,qx,qy,qz, gripper_width]
          camera_front_color (T, H, W, 3)  (optional, if --save-images)
        datagen_info/
          eef_pose                 (T, 4, 4)
          target_pose              (T, 4, 4)
          gripper_action           (T, 1)
          object_poses/
            block                  (T, 4, 4)
            target_coaster         (T, 4, 4)
          subtask_term_signals/
            grasped                (T,)
            placed                 (T,)
      demo_1/
        ...
    data (attrs):
      env_args             JSON string with env metadata
      total                number of demos
"""
import argparse
from collections import Counter
import json
import os
import sys

import gymnasium as gym
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Ensure the real2sim_env module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pick_and_place  # noqa: F401 – register BlockPAP-v1


# ─── Helpers ───────────────────────────────────────────────────────────────


def load_real_episode(h5_path):
    """Load a real robot episode HDF5 and return relevant arrays."""
    with h5py.File(h5_path, "r") as f:
        joint_pos = f["observations/joint_pos"][:]        # (T, 9)
        ee_pose = f["observations/ee_pose"][:]            # (T, 8) [x,y,z,qw,qx,qy,qz,gw]
        joint_vel = f["observations/joint_vel"][:]        # (T, 9)
        # Images (optional, only load if needed)
        images = {}
        if "observations/images" in f:
            for cam in f["observations/images"]:
                images[cam] = f[f"observations/images/{cam}"][:]
    return dict(
        joint_pos=joint_pos,
        ee_pose=ee_pose,
        joint_vel=joint_vel,
        images=images,
    )


def infer_gripper_cmd(joint_pos: np.ndarray, ee_pose: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    """
    Infer binary gripper command from continuous width signals.

    Priority:
      1) ee_pose[..., -1] (gripper width, if available)
      2) joint_pos[..., -1] (fallback)

    Returns:
      (T, 1) float32 command in {-1, +1}, where +1=open and -1=close.
    """
    if ee_pose.ndim == 2 and ee_pose.shape[1] >= 8:
        width = ee_pose[:, -1]
    elif joint_pos.ndim == 2 and joint_pos.shape[1] >= 1:
        width = joint_pos[:, -1]
    else:
        raise ValueError(
            f"Cannot infer gripper width from shapes joint_pos={joint_pos.shape}, ee_pose={ee_pose.shape}"
        )

    width = np.asarray(width, dtype=np.float32)
    gripper_cmd = np.where(width > threshold, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
    return gripper_cmd


def quat_to_rotmat(quat_wxyz):
    """Convert [w,x,y,z] quaternion to 3x3 rotation matrix."""
    # scipy uses [x,y,z,w]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return Rotation.from_quat(quat_xyzw).as_matrix()


def make_pose_4x4(pos, rot):
    """Build a 4x4 homogeneous transform from position (3,) and rotation (3,3)."""
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def sapien_pose_to_4x4(pose_p, pose_q):
    """
    Convert SAPIEN pose (p=[x,y,z], q=[w,x,y,z]) to 4x4 matrix.
    Works with tensors or numpy arrays.
    """
    if hasattr(pose_p, "cpu"):
        p = pose_p.cpu().numpy().flatten()[:3]
        q = pose_q.cpu().numpy().flatten()[:4]
    else:
        p = np.asarray(pose_p).flatten()[:3]
        q = np.asarray(pose_q).flatten()[:4]
    rot = quat_to_rotmat(q)  # q is [w,x,y,z]
    return make_pose_4x4(p, rot)


def compute_eef_pose_from_fk(base_env):
    """
    Get EEF pose (4x4) from the ManiSkill Panda's forward kinematics.
    Uses the panda_hand link (or TCP) pose.
    """
    robot = base_env.agent.robot
    # Get the end-effector link
    ee_link = base_env.agent.tcp  # or agent.robot.get_links()[-3] for panda_hand
    pose = ee_link.pose
    p = pose.p[0].cpu().numpy() if hasattr(pose.p, "cpu") else np.asarray(pose.p)
    q = pose.q[0].cpu().numpy() if hasattr(pose.q, "cpu") else np.asarray(pose.q)
    if p.ndim > 1:
        p = p[0]
    if q.ndim > 1:
        q = q[0]
    rot = quat_to_rotmat(q)  # SAPIEN quaternion is [w,x,y,z]
    return make_pose_4x4(p, rot)



def get_sim_state(base_env):
    """
    Serialize the ManiSkill sim state for state-based replay.
    Returns a flat numpy array of shape (28,):
      [robot_qpos(9), robot_qvel(9), cube_pos(3), cube_quat(4), coaster_pos(3)]

    Coaster position is included so that MimicGen can correctly restore the full
    scene state when replaying source demo segments during data generation.
    """
    robot = base_env.agent.robot
    qpos = robot.get_qpos()[0].cpu().numpy()   # (9,)
    qvel = robot.get_qvel()[0].cpu().numpy()   # (9,)

    cube_p    = base_env.cube.pose.p[0].cpu().numpy()   # (3,)
    cube_q    = base_env.cube.pose.q[0].cpu().numpy()   # (4,) [w,x,y,z]
    coaster_p = base_env.target.pose.p[0].cpu().numpy() # (3,)

    state = np.concatenate([qpos, qvel, cube_p, cube_q, coaster_p])  # 9+9+3+4+3 = 28
    return state


def reset_to_state(base_env, state):
    """Restore ManiSkill sim state from flat vector (28D or legacy 25D)."""
    import sapien
    from mani_skill.utils.structs.pose import Pose

    device = base_env.device
    qpos = torch.tensor(state[:9], device=device, dtype=torch.float32).unsqueeze(0)
    qvel = torch.tensor(state[9:18], device=device, dtype=torch.float32).unsqueeze(0)

    robot = base_env.agent.robot
    robot.set_qpos(qpos)
    robot.set_qvel(qvel)

    cube_p = state[18:21]
    cube_q = state[21:25]
    base_env.cube.set_pose(Pose.create_from_pq(
        p=torch.tensor(cube_p, device=device, dtype=torch.float32).unsqueeze(0)
    ))

    # Restore coaster pose (28D states only)
    if len(state) >= 28:
        coaster_z = base_env.TABLE_Z + base_env.COASTER_THICKNESS
        base_env.target.set_pose(sapien.Pose(
            p=[state[25], state[26], coaster_z],
            q=[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
        ))

    base_env.scene.step()
    base_env.scene.update_render()


# ─── Cleaning ─────────────────────────────────────────────────────────────


def _analyze_motion(ee_pose: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute per-frame motion score from ee_pose (T, 7or8)."""
    T = len(ee_pose)
    pos     = ee_pose[:, :3]
    rot     = ee_pose[:, 3:6] if ee_pose.shape[1] == 7 else ee_pose[:, 3:7]
    gripper = ee_pose[:, 6]   if ee_pose.shape[1] == 7 else ee_pose[:, 7]
    score = np.zeros(T)
    for i in range(T):
        s, e = max(0, i - window_size), min(T, i + window_size + 1)
        score[i] = (np.var(pos[s:e], axis=0).sum()
                    + np.var(rot[s:e], axis=0).sum()
                    + np.var(gripper[s:e]) * 10)
    return score


def _detect_static_segments(
    ee_pose: np.ndarray,
    pos_threshold: float     = 0.0005,
    rot_threshold: float     = 0.005,
    gripper_threshold: float = 0.0005,
    min_static_frames: int   = 5,
    motion_score_threshold: float = 0.0001,
) -> np.ndarray:
    """
    Return a boolean mask (True = keep).  Removes interior segments where the
    robot is nearly stationary for >= min_static_frames consecutive frames.
    """
    T = len(ee_pose)
    if T == 0:
        return np.ones(T, dtype=bool)

    score   = _analyze_motion(ee_pose)
    pos     = ee_pose[:, :3]
    rot     = ee_pose[:, 3:6] if ee_pose.shape[1] == 7 else ee_pose[:, 3:7]
    gripper = ee_pose[:, 6]   if ee_pose.shape[1] == 7 else ee_pose[:, 7]

    pos_d     = np.linalg.norm(np.diff(pos,     axis=0), axis=1)
    rot_d     = np.linalg.norm(np.diff(rot,     axis=0), axis=1)
    gripper_d = np.abs(np.diff(gripper, axis=0))

    is_static  = np.concatenate([[False],
                                  (pos_d < pos_threshold)
                                  & (rot_d < rot_threshold)
                                  & (gripper_d < gripper_threshold)])
    is_abnormal = is_static & (score < motion_score_threshold)

    mask = np.ones(T, dtype=bool)
    i = 0
    while i < T:
        if is_abnormal[i]:
            j = i
            while j < T and is_abnormal[j]:
                j += 1
            if (j - i) >= min_static_frames:
                if not (
                    (i > 0 and score[i - 1] > motion_score_threshold * 2)
                    and (j < T and score[j] > motion_score_threshold * 2)
                ):
                    mask[i:j] = False
            i = j
        else:
            i += 1
    return mask


def clean_demo(
    demo: dict,
    pos_threshold: float     = 0.0005,
    rot_threshold: float     = 0.005,
    gripper_threshold: float = 0.0005,
    min_static_frames: int   = 5,
    min_episode_length: int  = 20,
) -> dict | None:
    """
    Remove interior near-static segments from a source demo, then discard
    the demo if it becomes shorter than min_episode_length.

    Uses obs_ee_pose (real robot, T×8: [x,y,z,qw,qx,qy,qz,gw]) as the
    motion signal, matching real_data_convert.py's cleaning strategy.

    Returns cleaned demo dict, or None if too short after cleaning.
    """
    ee_pose = demo["obs_ee_pose"]   # (T, 8)
    mask    = _detect_static_segments(
        ee_pose,
        pos_threshold=pos_threshold,
        rot_threshold=rot_threshold,
        gripper_threshold=gripper_threshold,
        min_static_frames=min_static_frames,
    )

    kept = int(mask.sum())
    T    = len(mask)
    print(f"    [clean] traj {demo['traj_id']}: T={T} → {kept} "
          f"(removed {T - kept} static frames)")

    if kept < min_episode_length:
        print(f"    [clean] traj {demo['traj_id']}: discarded (too short: {kept} < {min_episode_length})")
        return None

    cleaned = {}
    for key in ("actions", "states", "obs_joint_pos", "obs_ee_pose",
                "eef_pose", "target_pose", "gripper_action",
                "block_pose", "coaster_pose"):
        cleaned[key] = demo[key][mask]

    # Re-latch grasped after removing frames
    grasped_raw = demo["grasped"][mask].copy()
    if grasped_raw.any():
        grasped_raw[np.argmax(grasped_raw):] = 1
    cleaned["grasped"] = grasped_raw
    cleaned["placed"]  = demo["placed"][mask]

    # Re-evaluate success
    T_new           = cleaned["actions"].shape[0]
    tail            = max(1, T_new // 10)
    grasp_any       = bool(cleaned["grasped"].any())
    placed_tail_any = bool(cleaned["placed"][-tail:].any())
    grasp_onset     = int(np.argmax(cleaned["grasped"])) if grasp_any else -1
    place_onset     = int(np.argmax(cleaned["placed"]))  if cleaned["placed"].any() else -1
    order_ok        = (place_onset >= 0) and (grasp_onset >= 0) and (grasp_onset <= place_onset)
    cleaned["is_success"] = bool(placed_tail_any and grasp_any and order_ok)
    cleaned["traj_id"]    = demo["traj_id"]
    return cleaned


# ─── Replay and extraction ────────────────────────────────────────────────


def replay_and_extract(traj_id, h5_base, cam_t="og", save_images=False, sim_steps=5):
    """
    Replay one real trajectory in ManiSkill and extract all MimicGen data.

    Returns a dict with all data needed for one demo in the MimicGen HDF5.
    """
    h5_path = os.path.join(h5_base, f"episode_{traj_id}.hdf5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    real_data = load_real_episode(h5_path)
    joint_pos = real_data["joint_pos"].astype(np.float32)
    ee_pose_raw = real_data["ee_pose"].astype(np.float32)  # (T, 8)
    T_len = joint_pos.shape[0]

    # Per-finger gripper targets
    joint_pos_sim = joint_pos.copy()
    joint_pos_sim[:, 7:9] = joint_pos_sim[:, 7:9] / 2.0

    # Inject trajectory config
    pick_and_place.TRAJ_ID = str(traj_id)
    pick_and_place.USE_REF_12 = False

    # Create environment
    env = gym.make("BlockPAP-v1", obs_mode="rgbd" if save_images else "state",
                   render_mode="rgb_array", cam_t=cam_t)
    obs, _ = env.reset()
    base_env = env.unwrapped

    # Setup hybrid PD drives (same as replay_traj.py)
    from replay_traj import setup_hybrid_drives, hybrid_step
    setup_hybrid_drives(base_env, arm_stiffness=1e5, arm_damping=1e3)

    # Initialize robot to qpos[0]
    robot = base_env.agent.robot
    q0 = torch.tensor(joint_pos_sim[0], device=base_env.device,
                       dtype=torch.float32).unsqueeze(0)
    robot.set_qpos(q0)
    robot.set_qvel(torch.zeros((1, 9), device=base_env.device))
    robot.set_joint_drive_targets(q0, robot.get_active_joints())
    base_env.scene.step()
    base_env.scene.update_render()

    # Storage
    eef_poses = []       # (T, 4, 4)
    block_poses = []     # (T, 4, 4)
    coaster_poses = []   # (T, 4, 4)
    sim_states = []      # (T, state_dim)
    sim_joint_pos = []   # (T, 9) actual sim joint pos
    success_flags = []   # (T,) from env.evaluate() – canonical success
    grasped_flags = []   # (T,) from sim gripper width – matches MG_BlockPAP._check_grasp


    # Record initial state
    def record_timestep():
        # EEF pose
        eef_pose = compute_eef_pose_from_fk(base_env)
        eef_poses.append(eef_pose)

        # Object poses
        cube_p = base_env.cube.pose.p[0].cpu().numpy()
        cube_q = base_env.cube.pose.q[0].cpu().numpy()
        block_pose = sapien_pose_to_4x4(cube_p, cube_q)
        block_poses.append(block_pose)

        coaster_p = base_env.target.pose.p[0].cpu().numpy()
        coaster_q = base_env.target.pose.q[0].cpu().numpy()
        coaster_pose = sapien_pose_to_4x4(coaster_p, coaster_q)
        coaster_poses.append(coaster_pose)

        # Sim state
        sim_states.append(get_sim_state(base_env))

        # Actual sim joint pos
        qpos_actual = robot.get_qpos()[0].cpu().numpy()
        sim_joint_pos.append(qpos_actual)

        # Success: use env.evaluate() – same logic as during MimicGen data generation
        eval_info = base_env.evaluate()
        success_flags.append(int(eval_info["success"][0].item()))

        # Grasped: sim gripper total width < 0.05 m, matches MG_BlockPAP._check_grasp()
        gripper_total = qpos_actual[7] + qpos_actual[8]
        is_grasped = gripper_total < 0.05
        grasped_flags.append(int(is_grasped))

    record_timestep()

    # Replay loop
    print(f"  Replaying traj {traj_id}: {T_len} timesteps...")
    for t in range(T_len):
        hybrid_step(base_env, joint_pos_sim[t, :7], joint_pos_sim[t, 7:9],
                    sim_steps=sim_steps)
        record_timestep()

    env.close()

    # Trim to match trajectory length (we recorded T_len+1 states, keep T_len)
    eef_poses = np.array(eef_poses[:T_len])
    block_poses = np.array(block_poses[:T_len])
    coaster_poses = np.array(coaster_poses[:T_len])
    sim_states = np.array(sim_states[:T_len])
    sim_joint_pos_arr = np.array(sim_joint_pos[:T_len])

    # Compute actions: normalized arm joint positions + gripper cmd
    # Actions: [norm_arm_joint_pos(7), gripper_cmd(1)] = 8D
    # Joint angles are normalized to [-1, 1] using Panda joint limits so the action format
    # matches what MimicGen expects (it clips all arm actions to [-1, 1] in waypoint.py:386).
    # env.step denormalizes back to radians before applying PD control.
    from mg_panda_kinematics import normalize_joints
    # Do not use joint_action (known noisy in some trajectories).
    # Infer open / close from continuous widths in ee_pose (preferred) or joint_pos.
    gripper_cmd = infer_gripper_cmd(joint_pos, ee_pose_raw, threshold=0.04)
    # The real robot may have been grasping at t=0 (inferred width -> close).
    # Force the trajectory to start with open gripper to avoid a spurious close
    # at the very first step of every generated demo.
    gripper_cmd[0] = 1.0
    norm_arm = np.stack(
        [normalize_joints(joint_pos_sim[t, :7]) for t in range(T_len)], axis=0
    )  # (T, 7) in [-1, 1]
    actions = np.concatenate([norm_arm, gripper_cmd], axis=1)  # (T, 8)

    # Target EEF poses: FK(joint_pos_sim[t+1]) = where the arm command at step t drives the robot.
    # Since high-stiffness PD tracks joint targets nearly perfectly,
    # eef_poses[t+1] (recorded AFTER command t) is the exact FK of the commanded position.
    target_poses_arr = np.zeros_like(eef_poses)
    target_poses_arr[:-1] = eef_poses[1:]
    target_poses_arr[-1] = eef_poses[-1]

    # Gripper action for MimicGen (the binary -1/+1 command)
    gripper_action_arr = gripper_cmd.astype(np.float32)

    # Subtask-1 term signal: "grasped" – derived from sim gripper width during replay,
    # using the same threshold as MG_BlockPAP._check_grasp() (0.05 m total width).
    # Once grasped becomes 1 it stays 1 (latch), consistent with MimicGen convention.
    grasped_raw = np.array(grasped_flags[:T_len], dtype=np.int32)
    grasped = grasped_raw.copy()
    if grasped_raw.any():
        grasped[np.argmax(grasped_raw):] = 1

    # Placed / success: derived from env.evaluate().
    # Hard success condition requires:
    # 1) final phase contains a placed signal,
    # 2) grasped occurs at least once,
    # 3) grasp onset is not later than place onset.
    placed = np.array(success_flags[:T_len], dtype=np.int32)
    tail = max(1, T_len // 10)

    grasp_any = bool(grasped.any())
    placed_tail_any = bool(placed[-tail:].any())

    grasp_onset = int(np.argmax(grasped)) if grasp_any else -1
    place_onset = int(np.argmax(placed)) if placed.any() else -1
    order_ok = (place_onset >= 0) and (grasp_onset >= 0) and (grasp_onset <= place_onset)

    is_success = bool(placed_tail_any and grasp_any and order_ok)

    return dict(
        actions=actions.astype(np.float32),
        states=sim_states.astype(np.float32),
        obs_joint_pos=sim_joint_pos_arr.astype(np.float32),
        obs_ee_pose=ee_pose_raw,
        eef_pose=eef_poses.astype(np.float32),
        target_pose=target_poses_arr.astype(np.float32),
        gripper_action=gripper_action_arr,
        block_pose=block_poses.astype(np.float32),
        coaster_pose=coaster_poses.astype(np.float32),
        grasped=grasped,
        placed=placed,
        is_success=is_success,
        traj_id=traj_id,
    )


# ─── Write HDF5 ──────────────────────────────────────────────────────────


def write_mimicgen_hdf5(all_demos, output_path, filter_success=True):
    """
    Write extracted demos into a single MimicGen-compatible HDF5.

    Args:
        all_demos:      list of demo dicts from replay_and_extract
        output_path:    path to output HDF5
        filter_success: if True, skip demos where is_success is False
    """
    if filter_success:
        kept   = [d for d in all_demos if d["is_success"]]
        dropped = [d for d in all_demos if not d["is_success"]]
        if dropped:
            print(f"\n  Skipping {len(dropped)} failed demo(s) "
                  f"(traj IDs: {[d['traj_id'] for d in dropped]})")
        all_demos = kept

    if not all_demos:
        raise RuntimeError("No successful demos to write. Check your trajectories.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")

        # env metadata (MimicGen reads this via robomimic FileUtils)
        env_args = {
            "env_name": "BlockPAP-v1",
            "type": 3,  # GYM_TYPE in robomimic EnvType enum (custom)
            "env_kwargs": {
                "env_name": "BlockPAP-v1",
                "robot_uids": "panda_high_friction",
            },
        }
        data_grp.attrs["env_args"] = json.dumps(env_args)
        data_grp.attrs["total"] = len(all_demos)

        for i, demo in enumerate(all_demos):
            ep_name = f"demo_{i}"
            ep_grp = data_grp.create_group(ep_name)

            T_len = demo["actions"].shape[0]

            # Core data required by MimicGen
            ep_grp.create_dataset("actions", data=demo["actions"])
            ep_grp.create_dataset("states", data=demo["states"])

            # Observations
            obs_grp = ep_grp.create_group("obs")
            obs_grp.create_dataset("joint_pos", data=demo["obs_joint_pos"])
            obs_grp.create_dataset("ee_pose", data=demo["obs_ee_pose"])

            # datagen_info - the core data MimicGen uses
            dg_grp = ep_grp.create_group("datagen_info")
            dg_grp.create_dataset("eef_pose", data=demo["eef_pose"])
            dg_grp.create_dataset("target_pose", data=demo["target_pose"])
            dg_grp.create_dataset("gripper_action", data=demo["gripper_action"])

            # Object poses
            obj_grp = dg_grp.create_group("object_poses")
            obj_grp.create_dataset("block", data=demo["block_pose"])
            obj_grp.create_dataset("target_coaster", data=demo["coaster_pose"])

            # Subtask termination signals
            # - grasped: end of subtask-1 (grasp), detected by gripper width
            # - placed:  informational only; final subtask needs no term signal
            sig_grp = dg_grp.create_group("subtask_term_signals")
            sig_grp.create_dataset("grasped", data=demo["grasped"])
            sig_grp.create_dataset("placed",  data=demo["placed"])

            # Store env interface info as attributes
            dg_grp.attrs["env_interface_name"] = "MG_BlockPAP"
            dg_grp.attrs["env_interface_type"] = "maniskill"

            # Store the original trajectory ID as metadata
            ep_grp.attrs["real_traj_id"] = str(demo["traj_id"])
            ep_grp.attrs["num_samples"] = T_len

            grasp_t = int(np.argmax(demo["grasped"])) if demo["grasped"].any() else -1
            place_t = int(np.argmax(demo["placed"]))  if demo["placed"].any()  else -1
            print(f"  Wrote {ep_name} (traj {demo['traj_id']}): T={T_len}, "
                  f"grasp={grasp_t}, place={place_t}, "
                  f"success={demo['is_success']}")

        # Create filter keys for all demos
        filter_grp = f.create_group("mask")
        all_demo_keys = [f"demo_{i}" for i in range(len(all_demos))]
        filter_grp.create_dataset("all", data=np.array(all_demo_keys, dtype="S"))

    print(f"\nWrote {len(all_demos)} demos → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert real Franka trajectories to MimicGen source HDF5"
    )
    parser.add_argument("--trajs", nargs="+", type=str, default=["0", "15", "25", "40", "45"],
                        help="Trajectory IDs to process")
    parser.add_argument("--h5-base", type=str,
                        default="/storage/zhijun/real_franka/pick_and_place",
                        help="Base directory for real episode HDF5 files")
    parser.add_argument("--output", type=str,
                        default="/workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_cleaned_src.hdf5",
                        help="Output MimicGen source HDF5 path")
    parser.add_argument("--cam-t", type=str, default="og",
                        help="Camera calibration preset")
    parser.add_argument("--sim-steps", type=int, default=5,
                        help="Physics substeps per frame during replay")
    parser.add_argument("--save-images", action="store_true",
                        help="Also save camera images in the HDF5 (large!)")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Write all demos including failed ones (default: filter out failures)")
    args = parser.parse_args()

    all_demos = []
    for traj_id in args.trajs:
        print(f"\n{'='*60}")
        print(f"Processing trajectory {traj_id}")
        print(f"{'='*60}")
        demo = replay_and_extract(
            traj_id=traj_id,
            h5_base=args.h5_base,
            cam_t=args.cam_t,
            save_images=args.save_images,
            sim_steps=args.sim_steps,
        )
        demo = clean_demo(demo)
        if demo is None:
            print(f"  [SKIP] traj {traj_id} discarded after cleaning")
            continue
        all_demos.append(demo)

    # Print per-trajectory success summary before filtering
    print(f"\n{'='*60}")
    print("Trajectory Success Summary")
    print(f"{'='*60}")

    def classify_demo(demo):
        """Return (status, reasons, grasp_onset, place_onset, tail)."""
        grasped = demo["grasped"]
        placed = demo["placed"]
        T_len = demo["actions"].shape[0]
        tail = max(1, T_len // 10)

        grasp_any = bool(grasped.any())
        placed_tail_any = bool(placed[-tail:].any())
        grasp_onset = int(np.argmax(grasped)) if grasp_any else -1
        place_onset = int(np.argmax(placed)) if placed.any() else -1
        order_ok = (place_onset >= 0) and (grasp_onset >= 0) and (grasp_onset <= place_onset)

        reasons = []
        if not grasp_any:
            reasons.append("no_grasp")
        if not placed_tail_any:
            reasons.append("no_place_in_tail")
        if grasp_any and (place_onset >= 0) and not order_ok:
            reasons.append("grasp_after_place")

        status = "OK" if (len(reasons) == 0 and demo["is_success"]) else "FAIL"
        return status, reasons, grasp_onset, place_onset, tail

    n_success = sum(d["is_success"] for d in all_demos)
    reason_counter = Counter()
    traj_counter = Counter()

    for d in all_demos:
        status, reasons, g, p, tail = classify_demo(d)
        if status == "OK":
            traj_counter["ok"] += 1
        else:
            traj_counter["fail"] += 1
            for r in reasons:
                reason_counter[r] += 1

        reason_txt = "-" if not reasons else ",".join(reasons)
        print(f"  traj {d['traj_id']}: T={d['actions'].shape[0]}, "
              f"grasp={g}, place={p}, tail={tail}, "
              f"[{status}] reason={reason_txt}")

    print(f"Success rate: {n_success}/{len(all_demos)}")
    print(f"Counts: OK={traj_counter['ok']}, FAIL={traj_counter['fail']}")
    if reason_counter:
        print("Failure reason breakdown:")
        for reason, cnt in sorted(reason_counter.items()):
            print(f"  - {reason}: {cnt}")

    write_mimicgen_hdf5(all_demos, args.output, filter_success=not args.keep_failed)

    print(f"\nOutput: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. (Optional) Manually annotate subtask boundaries:")
    print(f"     python mimicgen/scripts/annotate_subtasks.py --dataset {args.output}")
    print(f"  2. Use with MimicGen data generation (requires MG_BlockPAP env interface)")


if __name__ == "__main__":
    main()
