#!/usr/bin/env python3
"""
Robomimic-compatible environment wrapper for ManiSkill BlockPAP-v1.

This wrapper makes the ManiSkill pick-and-place environment compatible with
robomimic's EnvBase interface, which MimicGen relies on for:
  - Loading/saving simulation states via get_state() / reset_to()
  - Stepping with actions via step()
  - Evaluating success via is_success()
  - Serializing env config via serialize()

Usage:
    from mg_blockpap_wrapper import EnvManiskillBlockPAP

    env = EnvManiskillBlockPAP(
        env_name="BlockPAP-v1",
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )
    obs = env.reset()
    state = env.get_state()
    env.reset_to(state)
"""
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

from robomimic.envs.env_base import EnvBase, EnvType
import robomimic.utils.obs_utils as ObsUtils

def _denormalize_joints(norm7):
    """Denormalize 7D arm action from [-1, 1] → joint angles [rad]."""
    from mg_panda_kinematics import denormalize_joints
    return denormalize_joints(norm7)

# Ensure BlockPAP env is registered
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pick_and_place  # noqa: F401


class EnvManiskillBlockPAP(EnvBase):
    """
    Robomimic EnvBase wrapper for ManiSkill BlockPAP-v1.

    The action space is 8D: [norm_arm_joint_pos(7), gripper_cmd(1)]
      norm_arm_joint_pos: 7 Panda arm joints normalized to [-1, 1] via Panda joint limits
      gripper_cmd: +1 = open (0.04 m per finger), -1 = close (0 m)

    MimicGen clips arm actions to [-1, 1] (waypoint.py), so joint angles must be
    normalized. env.step denormalizes before applying PD control.
    """

    def __init__(
        self,
        env_name="BlockPAP-v1",
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
        postprocess_visual_obs=False,
        cam_t="og",
        **kwargs,
    ):
        self._env_name = env_name
        self._cam_t = cam_t
        self._init_kwargs = dict(
            env_name=env_name,
            cam_t=cam_t,
            **kwargs,
        )

        obs_mode = "rgbd" if (use_image_obs or use_depth_obs) else "state"
        render_mode = "rgb_array" if (render or render_offscreen) else None

        self.env = gym.make(
            env_name,
            obs_mode=obs_mode,
            render_mode=render_mode,
            cam_t=cam_t,
        )
        self._base_env = self.env.unwrapped

        # Set up high-stiffness PD drives (same as replay_traj.py)
        from replay_traj import setup_hybrid_drives
        setup_hybrid_drives(self._base_env)

        self._use_image_obs = use_image_obs
        self._use_depth_obs = use_depth_obs
        self._postprocess_visual_obs = postprocess_visual_obs

        # Track current trajectory config (for multi-traj support)
        self._current_traj_id = getattr(pick_and_place, "TRAJ_ID", "0")

    def step(self, action):
        """
        Step the environment with an 8D action.
        Action: [norm_arm_joint_pos(7), gripper_cmd(1)]
          norm_arm_joint_pos: normalized to [-1, 1] using Panda joint limits
          gripper_cmd: +1 = open (0.04 m per finger), -1 = close (0 m)

        MimicGen clips arm actions to [-1, 1] (waypoint.py:386), so we expect
        normalized joint angles here and denormalize before applying PD control.
        """
        from replay_traj import hybrid_step
        arm_targets = _denormalize_joints(np.asarray(action[:7], dtype=np.float32))
        gripper_cmd = float(action[7])
        # Per-finger target: Panda finger range is 0~0.04 m
        gripper_width = 0.04 if gripper_cmd > 0 else 0.0
        gripper_targets = np.array([gripper_width, gripper_width], dtype=np.float32)
        hybrid_step(self._base_env, arm_targets, gripper_targets, sim_steps=5)

        obs = self._base_env.get_obs()
        obs_dict = self._obs_to_dict(obs)
        info = self._base_env.evaluate()
        success = bool(info.get("success", [False])[0])
        return obs_dict, float(success), False, info

    def _obs_to_dict(self, obs):
        """Convert ManiSkill observation to robomimic-style dict."""
        di = {}

        # Low-dimensional state
        robot = self._base_env.agent.robot
        qpos = robot.get_qpos()
        if hasattr(qpos, "cpu"):
            qpos = qpos[0].cpu().numpy()
        di["joint_pos"] = np.asarray(qpos).flatten().astype(np.float32)

        # EEF pose
        tcp = self._base_env.agent.tcp
        p = tcp.pose.p
        q = tcp.pose.q
        if hasattr(p, "cpu"):
            p = p[0].cpu().numpy()
            q = q[0].cpu().numpy()
        di["ee_pos"] = np.asarray(p).flatten()[:3].astype(np.float32)
        di["ee_quat"] = np.asarray(q).flatten()[:4].astype(np.float32)

        # Object poses
        cube_p = self._base_env.cube.pose.p
        cube_q = self._base_env.cube.pose.q
        if hasattr(cube_p, "cpu"):
            cube_p = cube_p[0].cpu().numpy()
            cube_q = cube_q[0].cpu().numpy()
        di["block_pos"] = np.asarray(cube_p).flatten()[:3].astype(np.float32)
        di["block_quat"] = np.asarray(cube_q).flatten()[:4].astype(np.float32)

        coaster_p = self._base_env.target.pose.p
        if hasattr(coaster_p, "cpu"):
            coaster_p = coaster_p[0].cpu().numpy()
        di["coaster_pos"] = np.asarray(coaster_p).flatten()[:3].astype(np.float32)

        # Images (optional)
        if self._use_image_obs and isinstance(obs, dict) and "sensor_data" in obs:
            for cam_name, cam_data in obs["sensor_data"].items():
                img = cam_data["rgb"]
                if hasattr(img, "cpu"):
                    img = img[0].cpu().numpy()
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                di[f"{cam_name}_image"] = img

        return di

    def reset(self):
        obs, _ = self.env.reset()
        return self._obs_to_dict(obs)

    def reset_to(self, state, step_sim=True):
        """
        Reset environment to a specific state.

        Args:
            state: dict with "states" key containing flattened state vector
                   [robot_qpos(9), robot_qvel(9), cube_pos(3), cube_quat(4), coaster_pos(3)] = 28D
                   (Legacy 25D states without coaster_pos are also accepted.)
            step_sim: if True (default), advance physics one step after restoring state.
                      Set False during video replay so gravity doesn't disturb the pose.
        """
        if state is None or "states" not in state:
            return None

        s = state["states"]
        device = self._base_env.device
        robot = self._base_env.agent.robot

        # Restore robot state
        qpos = torch.tensor(s[:9], device=device, dtype=torch.float32).unsqueeze(0)
        qvel = torch.tensor(s[9:18], device=device, dtype=torch.float32).unsqueeze(0)
        robot.set_qpos(qpos)
        robot.set_qvel(qvel)
        # Set drive targets to restored qpos so the high-stiffness PD drive holds
        # position during scene.step() instead of pulling toward stale targets.
        robot.set_joint_drive_targets(qpos, robot.get_active_joints())

        # Restore cube pose
        import sapien
        self._base_env.cube.set_pose(sapien.Pose(p=s[18:21], q=s[21:25]))

        # Restore coaster pose (present in 28D states from generation mode)
        if len(s) >= 28:
            base_env = self._base_env
            coaster_z = base_env.TABLE_Z + base_env.COASTER_THICKNESS
            import numpy as _np
            self._base_env.target.set_pose(sapien.Pose(
                p=[s[25], s[26], coaster_z],
                q=[_np.cos(_np.pi / 4), 0, _np.sin(_np.pi / 4), 0],
            ))

        if step_sim:
            self._base_env.scene.step()
        self._base_env.scene.update_render()

        obs = self._base_env.get_obs()
        return self._obs_to_dict(obs)

    def render(self, mode="human", height=None, width=None, camera_name=None):
        if mode == "rgb_array":
            frame = self.env.render()
            if hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()
            frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            return frame
        return None

    def get_observation(self, di=None):
        obs = self._base_env.get_obs()
        return self._obs_to_dict(obs)

    def get_state(self):
        """
        Get current simulation state as dict.
        Returns: {"states": np.array of shape (28,)}
          [robot_qpos(9), robot_qvel(9), cube_pos(3), cube_quat(4), coaster_pos(3)]
        Coaster position is included because it is randomized in generation mode.
        """
        robot = self._base_env.agent.robot
        qpos = robot.get_qpos()[0].cpu().numpy()
        qvel = robot.get_qvel()[0].cpu().numpy()

        cube_p = self._base_env.cube.pose.p[0].cpu().numpy()
        cube_q = self._base_env.cube.pose.q[0].cpu().numpy()

        coaster_p = self._base_env.target.pose.p[0].cpu().numpy()

        state_vec = np.concatenate([qpos, qvel, cube_p, cube_q, coaster_p]).astype(np.float32)
        return {"states": state_vec}

    def get_reward(self):
        info = self._base_env.evaluate()
        if "success" in info:
            return float(info["success"][0])
        return 0.0

    def get_goal(self):
        return {}

    def set_goal(self, **kwargs):
        pass

    def is_done(self):
        return False

    def is_success(self):
        info = self._base_env.evaluate()
        success = bool(info["success"][0]) if "success" in info else False
        return {"task": success}

    def serialize(self):
        return {
            "env_name": self._env_name,
            "type": self.type,
            "env_kwargs": self._init_kwargs,
        }

    @property
    def rollout_exceptions(self):
        return (RuntimeError,)

    @property
    def base_env(self):
        return self._base_env

    @property
    def action_dimension(self):
        return 8  # [abs_arm_joint_pos(7), gripper_cmd(1)]

    @property
    def name(self):
        return self._env_name

    @property
    def type(self):
        return EnvType.GYM_TYPE  # type 3

    @classmethod
    def create_for_data_processing(
        cls,
        env_name,
        camera_names=None,
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
        render=None,
        render_offscreen=None,
        use_image_obs=None,
        use_depth_obs=False,
        **kwargs,
    ):
        has_camera = camera_names is not None and len(camera_names) > 0

        obs_modality_specs = {
            "obs": {
                "low_dim": ["joint_pos", "ee_pos", "ee_quat", "block_pos",
                            "block_quat", "coaster_pos"],
                "rgb": [],
            }
        }
        if has_camera:
            obs_modality_specs["obs"]["rgb"] = [f"{cn}_image" for cn in camera_names]

        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        return cls(
            env_name=env_name,
            render=False if render is None else render,
            render_offscreen=has_camera if render_offscreen is None else render_offscreen,
            use_image_obs=has_camera if use_image_obs is None else use_image_obs,
            use_depth_obs=use_depth_obs,
            postprocess_visual_obs=False,
            **kwargs,
        )
