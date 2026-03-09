#!/usr/bin/env python3
"""
MimicGen environment interface for ManiSkill BlockPAP-v1 (pick-and-place).

Task-specific code only. All Panda FK/IK and action-space conversions
are inherited from PandaKinematicsMixin (mg_panda_kinematics.py).

The task has two subtasks:
  1. Grasp the block  (reference object: "block")
  2. Place on coaster (reference object: "target_coaster", final subtask)

Usage:
    import mg_blockpap_interface  # registers MG_BlockPAP

    from mimicgen.env_interfaces.base import make_interface
    env_interface = make_interface(
        name="MG_BlockPAP",
        interface_type="maniskill",
        env=maniskill_base_env,
    )
"""
import numpy as np

from mimicgen.env_interfaces.base import MG_EnvInterface
from mg_panda_kinematics import PandaKinematicsMixin, sapien_pose_to_mat


class MG_BlockPAP(PandaKinematicsMixin, MG_EnvInterface):
    """
    MimicGen interface for BlockPAP-v1.

    Inherits all FK/IK and action-space conversions from PandaKinematicsMixin.
    Only task-specific methods are defined here.
    """
    INTERFACE_TYPE = "maniskill"

    def get_object_poses(self):
        """
        Poses of task-relevant objects.

        Returns:
            dict mapping object name → 4x4 pose matrix:
              "block"         – the movable block to pick up
              "target_coaster" – the static coaster where block is placed
        """
        return dict(
            block=sapien_pose_to_mat(self.env.cube.pose),
            target_coaster=sapien_pose_to_mat(self.env.target.pose),
        )

    def get_subtask_term_signals(self):
        """
        Binary completion flags for each subtask.

        "grasped": gripper closed AND TCP within 10 cm of block
        "lifted":  grasped AND block center z > 0.12 m
                   (block rests at z = TABLE_Z + BLOCK_HALF_Z = 0.06 m;
                    threshold 0.12 m ≈ 6 cm above rest)
        """
        grasped = self._check_grasp()

        lifted = False
        if grasped:
            cube_p = self.env.cube.pose.p
            cube_z = (float(cube_p[0, 2].cpu().numpy())
                      if hasattr(cube_p, "cpu")
                      else float(np.asarray(cube_p).flatten()[2]))
            lifted = cube_z > 0.12

        return dict(grasped=int(grasped), lifted=int(lifted))

    def _check_grasp(self):
        """
        True when the robot is grasping the block.

        Conditions (both must hold):
          1. Total finger width < 5 cm  (gripper mostly closed)
          2. TCP distance to block < 10 cm  (prevents false positives on approach)
        """
        robot = self.env.agent.robot
        qpos = robot.get_qpos()
        qpos = (qpos[0].cpu().numpy() if hasattr(qpos, "cpu")
                else np.asarray(qpos).flatten())

        if qpos[7] + qpos[8] >= 0.05:
            return False

        tcp_p = self.env.agent.tcp.pose.p
        cube_p = self.env.cube.pose.p
        if hasattr(tcp_p, "cpu"):
            tcp_p = tcp_p[0].cpu().numpy()
            cube_p = cube_p[0].cpu().numpy()
        else:
            tcp_p = np.asarray(tcp_p).flatten()[:3]
            cube_p = np.asarray(cube_p).flatten()[:3]

        return float(np.linalg.norm(tcp_p - cube_p)) < 0.10
