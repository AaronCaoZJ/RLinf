#!/usr/bin/env python3
"""
Common Panda kinematics for MimicGen environment interfaces.

Provides:
  - PANDA_JOINT_LIMITS_MIN/MAX  : official Franka Panda URDF joint limits
  - normalize_joints()          : [rad] → [-1, 1]
  - denormalize_joints()        : [-1, 1] → [rad]
  - PandaKinematicsMixin        : pinocchio FK/IK methods for MG_EnvInterface subclasses

Usage:
    class MG_MyTask(PandaKinematicsMixin, MG_EnvInterface):
        INTERFACE_TYPE = "maniskill"

        def get_object_poses(self): ...
        def get_subtask_term_signals(self): ...
"""
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Joint limits (Franka Panda official URDF)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def make_pose(pos, rot):
    """Build 4x4 homogeneous transform from (3,) position and (3,3) rotation."""
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def sapien_pose_to_mat(pose):
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
    return make_pose(p, rot)


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class PandaKinematicsMixin:
    """
    Mixin for MG_EnvInterface subclasses running on a ManiSkill Panda robot.

    Provides pinocchio-based FK/IK and the standard MimicGen action-space
    conversions (absolute normalized joint positions).

    Expects `self.env` to be the ManiSkill base environment (set by
    MG_EnvInterface.__init__).
    """

    # ------------------------------------------------------------------
    # EEF pose (for MimicGen planning)
    # ------------------------------------------------------------------

    def get_robot_eef_pose(self):
        """
        Get current EEF pose as 4x4 matrix via pinocchio FK.

        Uses pinocchio instead of GPU tcp.pose so that this method and
        action_to_target_pose() live in the same frame. If they differed,
        MimicGen would see a positional jump at every subtask boundary
        (interpolate_from_last_target_pose path).
        """
        robot = self.env.agent.robot
        qpos = robot.get_qpos()
        qpos_np = (qpos[0].cpu().numpy() if hasattr(qpos, "cpu")
                   else np.asarray(qpos).flatten())
        return self._fk_arm_qpos(qpos_np[:7])

    # ------------------------------------------------------------------
    # Pinocchio FK
    # ------------------------------------------------------------------

    def _get_pinocchio(self):
        """
        Return cached (pinocchio_model, tcp_link_index).

        Auto-calibrates the TCP link index by comparing pinocchio FK against
        SAPIEN's tcp.pose for every articulation link.  Robust against
        SAPIEN/pinocchio link-ordering differences.
        """
        if hasattr(self, "_pmodel") and hasattr(self, "_tcp_pino_idx"):
            return self._pmodel, self._tcp_pino_idx

        robot = self.env.agent.robot
        self._pmodel = robot.create_pinocchio_model()

        qpos_np = (robot.get_qpos()[0].cpu().numpy()
                   if hasattr(robot.get_qpos(), "cpu")
                   else np.asarray(robot.get_qpos()).flatten())

        p_actual = self.env.agent.tcp.pose.p
        if hasattr(p_actual, "cpu"):
            p_actual = p_actual[0].cpu().numpy()
        else:
            p_actual = np.asarray(p_actual).flatten()[:3]

        self._pmodel.compute_forward_kinematics(qpos_np.astype(np.float64))
        best_idx, min_err = 0, float("inf")
        for i in range(len(robot.get_links())):
            try:
                p_pino = sapien_pose_to_mat(
                    self._pmodel.get_link_pose(i))[:3, 3]
                err = float(np.linalg.norm(p_pino - p_actual))
                if err < min_err:
                    min_err, best_idx = err, i
            except Exception:
                continue

        if min_err > 0.02:
            print(f"[pino] WARNING: FK pos error {min_err*100:.1f} cm > 2 cm")

        self._tcp_pino_idx = best_idx
        return self._pmodel, self._tcp_pino_idx

    def _fk_arm_qpos(self, arm_qpos_7: np.ndarray) -> np.ndarray:
        """
        Forward kinematics: 7D arm joint positions → 4x4 EEF pose.

        Uses pinocchio (pure CPU) instead of reading SAPIEN's GPU link.pose,
        which is only updated after scene.step().
        """
        pmodel, tcp_idx = self._get_pinocchio()

        robot = self.env.agent.robot
        curr_np = (robot.get_qpos()[0].cpu().numpy()
                   if hasattr(robot.get_qpos(), "cpu")
                   else np.asarray(robot.get_qpos()).flatten())

        full_qpos = curr_np.copy().astype(np.float64)
        full_qpos[:7] = arm_qpos_7

        pmodel.compute_forward_kinematics(full_qpos)
        return sapien_pose_to_mat(pmodel.get_link_pose(tcp_idx))

    # ------------------------------------------------------------------
    # Pinocchio IK (with scipy L-BFGS-B fallback)
    # ------------------------------------------------------------------

    def _ik_eef_pose(self, target_pose: np.ndarray) -> np.ndarray:
        """
        Inverse kinematics: 4x4 EEF target pose → 7D arm joint positions.

        Tries pinocchio IK first; falls back to scipy L-BFGS-B if pinocchio
        does not converge or the FK residual exceeds 5 cm.
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

        if not hasattr(self, "_ik_pino_ok"):
            self._ik_pino_ok = 0
            self._ik_pino_fail = 0
            self._ik_scipy_count = 0

        n_dof = len(curr_np)
        active_mask = np.zeros(n_dof, dtype=np.int32)
        active_mask[:7] = 1

        try:
            result = pmodel.compute_inverse_kinematics(
                tcp_idx,
                target_sapien_pose,
                initial_qpos=curr_np.astype(np.float64),
                active_qmask=active_mask,
            )
            if isinstance(result, (tuple, list)):
                qpos_result = result[0]
                success = bool(result[1]) if len(result) > 1 else True
                if not success:
                    self._ik_pino_fail += 1
                    raise RuntimeError("pinocchio IK did not converge")
            else:
                qpos_result = result

            pino_q7 = np.asarray(qpos_result, dtype=np.float32)[:7]

            pos_err = float(np.linalg.norm(
                self._fk_arm_qpos(pino_q7)[:3, 3] - target_pos))
            if pos_err > 0.05:
                self._ik_pino_fail += 1
                raise RuntimeError(f"pinocchio IK residual too large: {pos_err:.4f}m")

            self._ik_pino_ok += 1
            return pino_q7

        except Exception as e:
            self._ik_scipy_count += 1
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

    # ------------------------------------------------------------------
    # MimicGen action-space conversions
    # ------------------------------------------------------------------

    def target_pose_to_action(self, target_pose, relative=True):
        """
        4x4 target EEF pose → normalized 7D arm action in [-1, 1].

        ``relative`` is ignored (absolute joint-position control).
        Gripper command is appended separately by MimicGen via
        action_to_gripper_action().
        """
        return normalize_joints(self._ik_eef_pose(target_pose))

    def action_to_target_pose(self, action, relative=True):
        """
        8D action [norm_arm(7), gripper_cmd(1)] → 4x4 EEF pose via FK.

        ``relative`` is ignored (absolute joint-position control).
        """
        return self._fk_arm_qpos(denormalize_joints(action[:7]))

    def action_to_gripper_action(self, action):
        """Extract gripper command (last element) from the full action vector."""
        return action[-1:]
