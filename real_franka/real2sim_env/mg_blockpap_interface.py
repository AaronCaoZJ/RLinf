#!/usr/bin/env python3
"""
MimicGen environment interface for ManiSkill BlockPAP-v1 (pick-and-place).

This module provides the MG_BlockPAP class that bridges between MimicGen's
data generation pipeline and the ManiSkill BlockPAP-v1 environment.

The task has two subtasks:
  1. Grasp the block  (reference object: "block")
  2. Place on coaster (reference object: "target_coaster", final subtask)

Usage:
    # Register the interface (import this module)
    import mg_blockpap_interface

    # Then use MimicGen's make_interface
    from mimicgen.env_interfaces.base import make_interface
    env_interface = make_interface(
        name="MG_BlockPAP",
        interface_type="maniskill",
        env=maniskill_base_env,
    )
"""
import numpy as np
from scipy.spatial.transform import Rotation

from mimicgen.env_interfaces.base import MG_EnvInterface


# Franka Panda joint limits (radians) for the 7 arm joints.
# Source: official Franka Panda URDF specification.
PANDA_JOINT_LIMITS_MIN = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
PANDA_JOINT_LIMITS_MAX = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)


def normalize_joints(q7: np.ndarray) -> np.ndarray:
    """Map 7D joint angles [rad] → [-1, 1] using Panda joint limits."""
    q7 = np.asarray(q7, dtype=np.float32)
    return (2.0 * (q7 - PANDA_JOINT_LIMITS_MIN)
            / (PANDA_JOINT_LIMITS_MAX - PANDA_JOINT_LIMITS_MIN) - 1.0)


def denormalize_joints(norm7: np.ndarray) -> np.ndarray:
    """Map [-1, 1] normalized values → 7D joint angles [rad]."""
    norm7 = np.asarray(norm7, dtype=np.float32)
    return ((norm7 + 1.0) / 2.0
            * (PANDA_JOINT_LIMITS_MAX - PANDA_JOINT_LIMITS_MIN)
            + PANDA_JOINT_LIMITS_MIN)


def _make_pose(pos, rot):
    """Build 4x4 homogeneous transform from (3,) position and (3,3) rotation."""
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def _unmake_pose(pose):
    """Extract position (3,) and rotation (3,3) from 4x4 pose."""
    return pose[:3, 3].copy(), pose[:3, :3].copy()


def _sapien_pose_to_mat(pose):
    """
    Convert a SAPIEN Pose to 4x4 matrix.
    SAPIEN uses [w,x,y,z] for quaternion.
    """
    p = pose.p
    q = pose.q
    if hasattr(p, "cpu"):
        p = p[0].cpu().numpy()
        q = q[0].cpu().numpy()
    else:
        p = np.asarray(p).flatten()[:3]
        q = np.asarray(q).flatten()[:4]

    # SAPIEN [w,x,y,z] → scipy [x,y,z,w]
    quat_xyzw = [q[1], q[2], q[3], q[0]]
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    return _make_pose(p, rot)


def _quat_multiply(q1, q2):
    """Multiply two quaternions in [w,x,y,z] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


class MG_BlockPAP(MG_EnvInterface):
    """
    MimicGen environment interface for BlockPAP-v1 (ManiSkill pick-and-place).

    The action space for MimicGen data generation uses delta-EEF pose:
      action = [delta_x, delta_y, delta_z, delta_ax, delta_ay, delta_az, gripper]
    where delta_a* is axis-angle rotation delta.

    The interface converts between this action space and 4x4 pose matrices
    using the robot's forward kinematics via ManiSkill's TCP link.
    """
    INTERFACE_TYPE = "maniskill"

    def get_robot_eef_pose(self):
        """
        Get current robot end-effector pose as 4x4 matrix.

        Uses pinocchio FK on current joint angles — the same FK used by
        action_to_target_pose().  This ensures get_robot_eef_pose() and
        action_to_target_pose() live in the SAME coordinate frame.

        MimicGen's interpolation (interpolate_from_last_target_pose=True)
        calls action_to_target_pose(last_action) to get the interpolation
        start pose, then passes waypoints computed from that frame to
        target_pose_to_action() (which again uses pinocchio IK).  If
        get_robot_eef_pose() used a DIFFERENT frame (e.g. GPU tcp.pose),
        MimicGen would see a positional jump at every subtask boundary,
        causing the observed pauses and erratic motion.

        GPU physics (scene.step, rendering) is completely unaffected —
        pinocchio is pure CPU math used only for planning arithmetic.
        """
        robot = self.env.agent.robot
        qpos = robot.get_qpos()
        qpos_np = (qpos[0].cpu().numpy() if hasattr(qpos, "cpu")
                   else np.asarray(qpos).flatten())
        # FK with current arm joints; result is in world frame (same as GPU)
        return self._fk_arm_qpos(qpos_np[:7])

    def _get_pinocchio(self):
        """
        Return cached (pinocchio_model, tcp_link_index).

        Auto-calibrates the TCP link index by comparing pinocchio FK against
        SAPIEN's tcp.pose position for every articulation link, choosing the
        index with the smallest error.  This is robust against SAPIEN/pinocchio
        link-ordering differences and does not affect GPU physics or rendering.
        """
        if hasattr(self, "_pmodel") and hasattr(self, "_tcp_pino_idx"):
            return self._pmodel, self._tcp_pino_idx

        robot = self.env.agent.robot
        self._pmodel = robot.create_pinocchio_model()
        tcp = self.env.agent.tcp
        tcp_name = tcp.name
        links = robot.get_links()
        link_names = [l.name for l in links]
        # print(f"[pino] tcp='{tcp_name}'  links({len(links)}): {link_names}")

        # Current joint configuration for FK calibration
        qpos_np = (robot.get_qpos()[0].cpu().numpy()
                   if hasattr(robot.get_qpos(), "cpu")
                   else np.asarray(robot.get_qpos()).flatten())

        # GPU tcp.pose is valid right after env.reset() + scene.step()
        p_actual = tcp.pose.p
        if hasattr(p_actual, "cpu"):
            p_actual = p_actual[0].cpu().numpy()
        else:
            p_actual = np.asarray(p_actual).flatten()[:3]

        # Run pinocchio FK and find which link index best matches tcp.pose
        self._pmodel.compute_forward_kinematics(qpos_np.astype(np.float64))
        best_idx, min_err = 0, float("inf")
        for i in range(len(links)):
            try:
                p_pino = _sapien_pose_to_mat(
                    self._pmodel.get_link_pose(i))[:3, 3]
                err = float(np.linalg.norm(p_pino - p_actual))
                if err < min_err:
                    min_err, best_idx = err, i
            except Exception:
                continue

        name_at_best = link_names[best_idx] if best_idx < len(link_names) else "?"
        # print(f"[pino] auto-calibrated idx={best_idx} ('{name_at_best}'), "
        #       f"pos_err={min_err*100:.2f} cm")
        if min_err > 0.02:
            print(f"[pino] WARNING: FK pos error {min_err*100:.1f} cm > 2 cm")

        self._tcp_pino_idx = best_idx
        return self._pmodel, self._tcp_pino_idx

    def _fk_arm_qpos(self, arm_qpos_7: np.ndarray) -> np.ndarray:
        """
        Forward kinematics: 7D arm joint positions → 4x4 EEF pose.

        Uses pinocchio (pure kinematic math, CPU-side) instead of reading
        SAPIEN's link pose buffer.

        WHY: In GPU-accelerated SAPIEN, link.pose is only updated after
        scene.step().  Calling set_qpos() then reading tcp.pose without a
        scene.step() returns the pose from the *previous* physics step, which
        makes FK wrong and causes the scipy IK to converge to garbage solutions.
        Pinocchio computes FK directly from the URDF, independent of the GPU
        physics state.
        """
        pmodel, tcp_idx = self._get_pinocchio()

        robot = self.env.agent.robot
        curr_np = (robot.get_qpos()[0].cpu().numpy()
                   if hasattr(robot.get_qpos(), "cpu")
                   else np.asarray(robot.get_qpos()).flatten())

        full_qpos = curr_np.copy().astype(np.float64)
        full_qpos[:7] = arm_qpos_7

        pmodel.compute_forward_kinematics(full_qpos)
        pose = pmodel.get_link_pose(tcp_idx)
        return _sapien_pose_to_mat(pose)

    def _ik_eef_pose(self, target_pose: np.ndarray) -> np.ndarray:
        """
        Inverse kinematics: 4x4 EEF target pose → 7D arm joint positions.

        Uses pinocchio IK with a convergence check.  Falls back to scipy
        L-BFGS-B (which now uses the correct pinocchio-based FK) if pinocchio
        IK does not converge.
        """
        import sapien
        pmodel, tcp_idx = self._get_pinocchio()

        robot = self.env.agent.robot
        curr_np = (robot.get_qpos()[0].cpu().numpy()
                   if hasattr(robot.get_qpos(), "cpu")
                   else np.asarray(robot.get_qpos()).flatten())

        target_pos = target_pose[:3, 3]
        q_xyzw = Rotation.from_matrix(target_pose[:3, :3]).as_quat()
        target_quat_wxyz = np.array(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
        )
        target_sapien_pose = sapien.Pose(
            p=target_pos.tolist(), q=target_quat_wxyz.tolist()
        )

        # IK diagnostics counters
        if not hasattr(self, "_ik_pino_ok"):
            self._ik_pino_ok = 0
            self._ik_pino_fail = 0
            self._ik_scipy_count = 0

        # active_qmask: 1=active, 0=fixed, shape (ndof,) int32
        n_dof = len(curr_np)
        active_mask = np.zeros(n_dof, dtype=np.int32)
        active_mask[:7] = 1

        pino_q7 = None
        try:
            result = pmodel.compute_inverse_kinematics(
                tcp_idx,
                target_sapien_pose,
                initial_qpos=curr_np.astype(np.float64),
                active_qmask=active_mask,
            )
            # API returns (qpos, success, error_vec)
            if isinstance(result, (tuple, list)):
                qpos_result = result[0]
                success = bool(result[1]) if len(result) > 1 else True
                if not success:
                    self._ik_pino_fail += 1
                    raise RuntimeError("pinocchio IK did not converge")
            else:
                qpos_result = result

            pino_q7 = np.asarray(qpos_result, dtype=np.float32)[:7]

            # Verify FK residual of pinocchio solution
            fk_result = self._fk_arm_qpos(pino_q7)
            pos_err = float(np.linalg.norm(fk_result[:3, 3] - target_pos))
            if pos_err > 0.05:  # > 5cm residual = bad solution
                self._ik_pino_fail += 1
                total = self._ik_pino_ok + self._ik_pino_fail + self._ik_scipy_count
                # if total % 20 == 0:
                #     print(f"[IK] pino residual {pos_err*100:.1f}cm > 5cm, "
                #           f"falling to scipy  [ok={self._ik_pino_ok} "
                #           f"fail={self._ik_pino_fail} scipy={self._ik_scipy_count}]")
                raise RuntimeError(f"pinocchio IK residual too large: {pos_err:.4f}m")

            self._ik_pino_ok += 1
            total = self._ik_pino_ok + self._ik_pino_fail + self._ik_scipy_count
            # if total % 50 == 0:
            #     print(f"[IK] step {total}: pino_ok={self._ik_pino_ok} "
            #           f"pino_fail={self._ik_pino_fail} scipy={self._ik_scipy_count} "
            #           f"last_residual={pos_err*100:.1f}cm")
            return pino_q7

        except Exception as e:
            self._ik_scipy_count += 1
            total = self._ik_pino_ok + self._ik_pino_fail + self._ik_scipy_count
            # if self._ik_scipy_count <= 3 or total % 20 == 0:
            #     print(f"[IK] scipy fallback #{self._ik_scipy_count} "
            #           f"(reason: {e}) target_pos={target_pos.round(3)}")
            return self._ik_scipy(target_pose, curr_np[:7])

    def _ik_scipy(self, target_pose: np.ndarray, initial_q7: np.ndarray) -> np.ndarray:
        """Numerical IK via L-BFGS-B minimisation of FK residual (with joint limits)."""
        from scipy.optimize import minimize

        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]

        def cost(q7):
            fk = self._fk_arm_qpos(q7)
            pos_err = fk[:3, 3] - target_pos
            rot_err = fk[:3, :3] - target_rot
            return float(np.dot(pos_err, pos_err) + 0.01 * np.sum(rot_err ** 2))

        bounds = list(zip(
            PANDA_JOINT_LIMITS_MIN.astype(np.float64),
            PANDA_JOINT_LIMITS_MAX.astype(np.float64),
        ))
        res = minimize(
            cost, initial_q7.astype(np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        return res.x.astype(np.float32)

    def target_pose_to_action(self, target_pose, relative=True):
        """
        4x4 target EEF pose → normalized 7D arm action in [-1, 1].

        MimicGen clips arm actions to [-1, 1] after adding noise (waypoint.py:386),
        so we must return normalized joint angles. ``env.step`` denormalizes before
        passing to the PD controller.

        The ``relative`` argument is ignored (absolute joint-position control).
        Gripper command is NOT included here; MimicGen appends it separately
        via ``action_to_gripper_action`` before calling ``env.step``.

        Returns:
            action (np.ndarray): shape (7,) normalized arm joint positions in [-1, 1]
        """
        q7 = self._ik_eef_pose(target_pose)
        return normalize_joints(q7)

    def action_to_target_pose(self, action, relative=True):
        """
        8D action [norm_arm(7), gripper_cmd(1)] → 4x4 EEF pose via FK.

        Denormalizes from [-1, 1] to joint angles [rad] before running FK.
        The ``relative`` argument is ignored (absolute joint-position control).

        Returns:
            target_pose (np.ndarray): shape (4, 4)
        """
        q7 = denormalize_joints(action[:7])
        return self._fk_arm_qpos(q7)

    def action_to_gripper_action(self, action):
        """
        Extract gripper action from the full action vector.
        Convention: last dimension is gripper command.
        """
        return action[-1:]

    def get_object_poses(self):
        """
        Get poses of task-relevant objects.

        Returns dict mapping object names to 4x4 pose matrices:
          - "block": the movable block to be picked up
          - "target_coaster": the static coaster where block should be placed
        """
        return dict(
            block=_sapien_pose_to_mat(self.env.cube.pose),
            target_coaster=_sapien_pose_to_mat(self.env.target.pose),
        )

    def get_subtask_term_signals(self):
        """
        Get binary flags for subtask completion.

        Signals:
          "grasped": block is grasped (gripper closed and near block)
          "lifted":  grasped AND block center z > 0.12 m
                     (block rests at z = TABLE_Z + BLOCK_HALF_Z = 0.03 + 0.03 = 0.06 m;
                      threshold 0.12 m ≈ 6 cm above rest, indicating a clear lift)
        """
        grasped = self._check_grasp()

        lifted = False
        if grasped:
            cube_p = self.env.cube.pose.p
            if hasattr(cube_p, "cpu"):
                cube_z = float(cube_p[0, 2].cpu().numpy())
            else:
                cube_z = float(np.asarray(cube_p).flatten()[2])
            lifted = cube_z > 0.12

        return dict(
            grasped=int(grasped),
            lifted=int(lifted),
        )

    def _check_grasp(self):
        """
        Check if the robot is grasping the block.
        Two conditions must both be true:
          1. Gripper is mostly closed (total finger width < 5 cm)
          2. TCP is physically close to the block (< 10 cm)
        Condition 2 prevents false positives when the gripper closes
        during the approach phase before reaching the block.
        """
        robot = self.env.agent.robot
        qpos = robot.get_qpos()
        if hasattr(qpos, "cpu"):
            qpos = qpos[0].cpu().numpy()
        else:
            qpos = np.asarray(qpos).flatten()

        gripper_width = qpos[7] + qpos[8]
        if gripper_width >= 0.05:
            return False

        # Also require TCP to be near the block
        tcp_p = self.env.agent.tcp.pose.p
        cube_p = self.env.cube.pose.p
        if hasattr(tcp_p, "cpu"):
            tcp_p = tcp_p[0].cpu().numpy()
            cube_p = cube_p[0].cpu().numpy()
        else:
            tcp_p = np.asarray(tcp_p).flatten()[:3]
            cube_p = np.asarray(cube_p).flatten()[:3]

        return float(np.linalg.norm(tcp_p - cube_p)) < 0.10
