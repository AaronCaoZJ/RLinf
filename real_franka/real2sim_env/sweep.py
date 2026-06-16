import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import sapien
from scipy.spatial.transform import Rotation
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import Panda
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building.ground import build_ground
from sapien.physx import PhysxMaterial


# ── Replay dataset control ────────────────────────────────────────────────────
# Set REPLAY_EPISODE_ID to an episode int before env creation to initialize the
# robot from the corresponding HDF5 frame, enabling real-to-sim replay.
REPLAY_EPISODE_ID = None
DATASET_DIR = "/workspace1/zhijun/hf_download/datasets/0109_sweep"


try:
    @register_agent()
    class PandaHighFriction(Panda):
        uid = "panda_high_friction"
        urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=100.0, dynamic_friction=100.0, restitution=0.0)
            ),
            link=dict(
                panda_leftfinger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
                panda_rightfinger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            ),
        )

        @property
        def ee_pose_at_robot_base(self):
            return self.robot.pose.inv() * self.tcp.pose
except Exception:
    from .pick_and_place import PandaHighFriction  # already registered


@register_env("Sweep-v1", max_episode_steps=3000)
class SweepEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda_high_friction", "panda", "panda_wristcam"]
    agent: PandaHighFriction

    # ── Camera calibration (front camera, RealSense D435, 2026-02-15) ──────
    _R = np.array([
        [ 0.02816316,  0.21788680, -0.97556762],
        [ 0.99959024, -0.00114196,  0.02860160],
        [ 0.00511786, -0.97597338, -0.21782968],
    ])
    _t = np.array([1.1002696, -0.00701879, 0.2589829])
    _K = np.array([
        [607.875,   0.0,   348.961],
        [  0.0,  607.719, 270.486],
        [  0.0,    0.0,     1.0  ],
    ])

    # ── Scene parameters (same layout as pick_and_place) ────────────────────
    TABLE_Z = 0.03
    _TABLE_CENTER_X = 0.501
    _TABLE_HALF = (0.30, 0.60, 0.0175)
    TABLE_STATIC_FRICTION = 1.0
    TABLE_DYNAMIC_FRICTION = 1.0
    TABLE_RESTITUTION = 0.0
    BASE_PEDESTAL_SIZE = 0.95

    # ── Broom (gray box substitute): xyz = 10 × 2 × 5 cm ───────────────────
    BROOM_HALF_SIZE = [0.05, 0.01, 0.025]
    BROOM_DENSITY = 100.0
    BROOM_STATIC_FRICTION = 0.5
    BROOM_DYNAMIC_FRICTION = 0.5

    # ── Green small block: 2 × 2 × 2 cm ────────────────────────────────────
    BLOCK_HALF_SIZE = [0.01, 0.01, 0.01]
    BLOCK_DENSITY = 50.0
    BLOCK_STATIC_FRICTION = 0.5
    BLOCK_DYNAMIC_FRICTION = 0.5

    # ── Dustpan dimensions ───────────────────────────────────────────────────
    # x-direction 13 cm wide, y-direction 10 cm deep
    # bottom: ~0.75 cm thick (average of 1 cm at open end, 0.5 cm at closed end)
    # three walls: 2 cm tall, 1 cm thick; open on -Y side (left from camera view)
    DUSTPAN_X_HALF = 0.065          # 13 cm total
    DUSTPAN_Y_HALF = 0.05           # 10 cm total
    DUSTPAN_BOTTOM_HALF_Z = 0.00375 # 0.75 cm average thickness
    DUSTPAN_WALL_HALF_Z = 0.01      # 2 cm tall walls
    DUSTPAN_WALL_HALF_T = 0.005     # 1 cm thick walls

    # ── Initial positions ────────────────────────────────────────────────────
    # Green block: traj0 block_xy from pick_and_place
    _BLOCK_XY   = (_TABLE_CENTER_X + 0.04, -0.12)
    # Broom: table centre
    _BROOM_XY   = (_TABLE_CENTER_X, -0.05)
    # Dustpan: table centre + 10 cm to the right (right from camera = +Y in world)
    _DUSTPAN_XY = (_TABLE_CENTER_X, 0.05)

    # traj0 qpos_ref_0 from pick_and_place (7 arm joints)
    _QPOS_REF = [
        0.03920901566743851, -0.6867635250091553, -0.009805346839129925,
        -2.6944820880889893, -0.013050149194896221, 2.007251739501953, 0.8208261132240295,
    ]

    def __init__(self, *args, robot_uids="panda_high_friction", **kwargs):
        kwargs.setdefault("enable_shadow", True)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ── Camera helpers ───────────────────────────────────────────────────────

    def _make_cam_pose(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        # OpenCV (Z-fwd, X-right, Y-down) → SAPIEN (X-fwd, Y-left, Z-up)
        T_cv_to_sapien = np.array([
            [ 0, -1,  0,  0],
            [ 0,  0, -1,  0],
            [ 1,  0,  0,  0],
            [ 0,  0,  0,  1],
        ])
        T_s = T @ T_cv_to_sapien
        p = T_s[:3, 3]
        q_xyzw = Rotation.from_matrix(T_s[:3, :3]).as_quat()
        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        return sapien.Pose(p=p, q=q_wxyz)

    @property
    def _default_sensor_configs(self):
        pose = self._make_cam_pose(self._R, self._t)
        return [
            CameraConfig("external_cam", pose, 640, 480,
                         near=0.01, far=10, intrinsic=self._K),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = self._make_cam_pose(self._R, self._t)
        return CameraConfig("render_camera", pose, 640, 480,
                            near=0.01, far=100, intrinsic=self._K,
                            shader_pack="default")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.35, 0.35, 0.35])
        # Front light: from camera side (-X world), slight downward tilt
        self.scene.add_directional_light(
            [-1.0, 0.0, -0.3], [1.2, 1.15, 1.1],
            shadow=True, shadow_scale=5, shadow_map_size=4096,
        )

    # ── Scene construction ───────────────────────────────────────────────────

    def _load_scene(self, options: dict):
        build_ground(self.scene, floor_width=10, altitude=-self.BASE_PEDESTAL_SIZE)

        # Robot base pedestal
        px_min = -self.BASE_PEDESTAL_SIZE / 2
        px_max = 1.2
        phx = (px_max - px_min) / 2
        pcx = (px_max + px_min) / 2
        phy = self.BASE_PEDESTAL_SIZE / 2
        phz = self.BASE_PEDESTAL_SIZE / 2

        ped_b = self.scene.create_actor_builder()
        ped_b.add_box_collision(half_size=[phx, phy, phz])
        ped_mat = sapien.render.RenderMaterial()
        ped_mat.base_color = [0.2, 0.2, 0.2, 1.0]
        ped_mat.roughness = 0.2
        ped_mat.metallic = 1.0
        ped_b.add_box_visual(half_size=[phx, phy, phz], material=ped_mat)
        ped_b.set_initial_pose(sapien.Pose(p=[pcx, 0, -phz]))
        self._base_pedestal = ped_b.build_kinematic(name="robot_base_pedestal")

        # Table
        tbl_b = self.scene.create_actor_builder()
        tbl_phys = PhysxMaterial(
            static_friction=self.TABLE_STATIC_FRICTION,
            dynamic_friction=self.TABLE_DYNAMIC_FRICTION,
            restitution=self.TABLE_RESTITUTION,
        )
        tbl_b.add_box_collision(half_size=self._TABLE_HALF, material=tbl_phys)
        tbl_mat = sapien.render.RenderMaterial()
        tbl_mat.base_color = [0.85, 0.78, 0.65, 1.0]
        tbl_mat.roughness = 0.6
        tbl_mat.metallic = 0.0
        tbl_b.add_box_visual(half_size=self._TABLE_HALF, material=tbl_mat)
        tbl_b.set_initial_pose(
            sapien.Pose(p=[self._TABLE_CENTER_X, 0, self.TABLE_Z - self._TABLE_HALF[2]])
        )
        self._table = tbl_b.build_kinematic(name="table")

        # Broom: gray box 10 × 2 × 5 cm
        brm_b = self.scene.create_actor_builder()
        brm_phys = PhysxMaterial(
            static_friction=self.BROOM_STATIC_FRICTION,
            dynamic_friction=self.BROOM_DYNAMIC_FRICTION,
            restitution=0.0,
        )
        brm_b.add_box_collision(
            half_size=self.BROOM_HALF_SIZE,
            material=brm_phys,
            density=self.BROOM_DENSITY,
        )
        brm_mat = sapien.render.RenderMaterial()
        brm_mat.base_color = [0.45, 0.45, 0.45, 1.0]
        brm_mat.roughness = 0.7
        brm_mat.metallic = 0.0
        brm_b.add_box_visual(half_size=self.BROOM_HALF_SIZE, material=brm_mat)
        brm_b.set_initial_pose(sapien.Pose(p=[
            self._BROOM_XY[0], self._BROOM_XY[1],
            self.TABLE_Z + self.BROOM_HALF_SIZE[2],
        ]))
        self.broom = brm_b.build(name="broom")

        # Green small block: 2 × 2 × 2 cm
        blk_b = self.scene.create_actor_builder()
        blk_phys = PhysxMaterial(
            static_friction=self.BLOCK_STATIC_FRICTION,
            dynamic_friction=self.BLOCK_DYNAMIC_FRICTION,
            restitution=0.0,
        )
        blk_b.add_box_collision(
            half_size=self.BLOCK_HALF_SIZE,
            material=blk_phys,
            density=self.BLOCK_DENSITY,
        )
        blk_mat = sapien.render.RenderMaterial()
        blk_mat.base_color = [0.0, 0.8, 0.2, 1.0]
        blk_mat.roughness = 0.5
        blk_mat.metallic = 0.0
        blk_b.add_box_visual(half_size=self.BLOCK_HALF_SIZE, material=blk_mat)
        blk_b.set_initial_pose(sapien.Pose(p=[
            self._BLOCK_XY[0], self._BLOCK_XY[1],
            self.TABLE_Z + self.BLOCK_HALF_SIZE[2],
        ]))
        self.block = blk_b.build(name="block")

        # White dustpan (compound kinematic actor)
        self._build_dustpan()

    def _build_dustpan(self):
        """
        Compound kinematic actor for the dustpan.

        Layout (local frame, anchor = dustpan xy-centre at table surface z):
          x-axis: 13 cm wide (±6.5 cm)
          y-axis: 10 cm deep (±5 cm)
          open on -Y side (left from front camera = toward the broom/block)
          closed on +Y, -X, +X sides with 1 cm-thick walls, 2 cm tall
          bottom plate ~0.75 cm thick (average of the 1 cm→0.5 cm wedge)
        """
        cx, cy = self._DUSTPAN_XY
        dx  = self.DUSTPAN_X_HALF         # 0.065
        dy  = self.DUSTPAN_Y_HALF         # 0.050
        bz  = self.DUSTPAN_BOTTOM_HALF_Z  # 0.00375
        wz  = self.DUSTPAN_WALL_HALF_Z    # 0.010
        wt  = self.DUSTPAN_WALL_HALF_T    # 0.005

        white = sapien.render.RenderMaterial()
        white.base_color = [0.95, 0.95, 0.95, 1.0]
        white.roughness = 0.3
        white.metallic = 0.0

        phys = PhysxMaterial(static_friction=0.5, dynamic_friction=0.5, restitution=0.0)

        b = self.scene.create_actor_builder()

        # Bottom plate: z from 0 to 2*bz
        b.add_box_collision(half_size=[dx, dy, bz],
                            pose=sapien.Pose(p=[0, 0, bz]), material=phys)
        b.add_box_visual(half_size=[dx, dy, bz],
                         pose=sapien.Pose(p=[0, 0, bz]), material=white)

        wall_cz = 2 * bz + wz  # centre z of all three walls

        # Back wall: closed +Y end
        b.add_box_collision(half_size=[dx, wt, wz],
                            pose=sapien.Pose(p=[0, dy, wall_cz]), material=phys)
        b.add_box_visual(half_size=[dx, wt, wz],
                         pose=sapien.Pose(p=[0, dy, wall_cz]), material=white)

        # Left side wall: −X end (full y span)
        b.add_box_collision(half_size=[wt, dy + wt, wz],
                            pose=sapien.Pose(p=[-dx, 0, wall_cz]), material=phys)
        b.add_box_visual(half_size=[wt, dy + wt, wz],
                         pose=sapien.Pose(p=[-dx, 0, wall_cz]), material=white)

        # Right side wall: +X end (full y span)
        b.add_box_collision(half_size=[wt, dy + wt, wz],
                            pose=sapien.Pose(p=[dx, 0, wall_cz]), material=phys)
        b.add_box_visual(half_size=[wt, dy + wt, wz],
                         pose=sapien.Pose(p=[dx, 0, wall_cz]), material=white)

        b.set_initial_pose(sapien.Pose(p=[cx, cy, self.TABLE_Z]))
        self.dustpan = b.build_kinematic(name="dustpan")

    # ── Episode initialisation ───────────────────────────────────────────────

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            n = len(env_idx)

            if REPLAY_EPISODE_ID is not None:
                import h5py as _h5py
                h5_path = os.path.join(DATASET_DIR, f"episode_{REPLAY_EPISODE_ID}.hdf5")
                with _h5py.File(h5_path, "r") as _f:
                    qpos0 = np.array(_f["observations/joint_pos"][0], dtype=np.float32)
                qpos0[7:9] /= 2.0  # total-width -> per-finger
                init_qpos = torch.tensor(qpos0, device=self.device).repeat(n, 1)
            else:
                init_qpos = torch.tensor(
                    self._QPOS_REF + [0.04, 0.04], device=self.device
                ).repeat(n, 1)

            self.agent.robot.set_qpos(init_qpos)
            self.agent.robot.set_qvel(torch.zeros((n, 9), device=self.device))

            self._table.set_pose(
                sapien.Pose(p=[self._TABLE_CENTER_X, 0, self.TABLE_Z - self._TABLE_HALF[2]])
            )

            # Broom at traj0 block position
            brm = torch.zeros((n, 3), device=self.device)
            brm[:, 0] = self._BROOM_XY[0]
            brm[:, 1] = self._BROOM_XY[1]
            brm[:, 2] = self.TABLE_Z + self.BROOM_HALF_SIZE[2]
            self.broom.set_pose(Pose.create_from_pq(p=brm))

            # Green block at table centre
            blk = torch.zeros((n, 3), device=self.device)
            blk[:, 0] = self._BLOCK_XY[0]
            blk[:, 1] = self._BLOCK_XY[1]
            blk[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2]
            self.block.set_pose(Pose.create_from_pq(p=blk))

            # Dustpan is kinematic – reset to its fixed world pose
            self.dustpan.set_pose(
                sapien.Pose(p=[self._DUSTPAN_XY[0], self._DUSTPAN_XY[1], self.TABLE_Z])
            )

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _is_block_in_dustpan(self) -> torch.Tensor:
        """True when the green block centre lies within the dustpan interior."""
        bp = self.block.pose.p      # (B, 3)
        dp = self.dustpan.pose.p    # (B, 3)

        rel_x = bp[:, 0] - dp[:, 0]
        rel_y = bp[:, 1] - dp[:, 1]
        rel_z = bp[:, 2] - dp[:, 2]

        bx = self.BLOCK_HALF_SIZE[0]
        by = self.BLOCK_HALF_SIZE[1]
        dx = self.DUSTPAN_X_HALF
        dy = self.DUSTPAN_Y_HALF
        floor_z = 2 * self.DUSTPAN_BOTTOM_HALF_Z
        top_z   = floor_z + 2 * self.DUSTPAN_WALL_HALF_Z + 0.01  # small tolerance

        x_in = torch.abs(rel_x) < (dx - bx)
        # -Y side is open; block must be fully past the open edge (rel_y > -dy+by)
        y_in = (rel_y > -dy + by) & (rel_y < dy - by)
        z_in = (rel_z > floor_z - 0.005) & (rel_z < top_z)

        return x_in & y_in & z_in

    def evaluate(self):
        in_pan = self._is_block_in_dustpan()
        lin_vel = self.block.linear_velocity
        vel_ok = torch.norm(lin_vel, dim=1) < 0.05
        return {
            "success": in_pan & vel_ok,
            "block_in_dustpan": in_pan,
        }

    # ── Reward ───────────────────────────────────────────────────────────────

    def compute_dense_reward(self, obs, action, info):
        tcp_pos   = self.agent.tcp.pose.p   # (B, 3)
        broom_pos = self.broom.pose.p       # (B, 3)
        block_pos = self.block.pose.p       # (B, 3)
        dustpan_pos = self.dustpan.pose.p   # (B, 3)

        dist_tcp_broom   = torch.norm(tcp_pos - broom_pos, dim=1)
        dist_block_dustpan = torch.norm(block_pos[:, :2] - dustpan_pos[:, :2], dim=1)

        # Stage 1 – approach broom
        approach = (1.0 - torch.tanh(5.0 * dist_tcp_broom)) * 0.1
        # Stage 2 – sweep block toward dustpan
        sweep = (1.0 - torch.tanh(3.0 * dist_block_dustpan)) * 0.5
        # Stage 3 – success
        success = info["success"].float() * 3.0

        return approach + sweep + success

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    # ── Observation helpers ──────────────────────────────────────────────────

    def get_language_instruction(self):
        return ["use the broom to sweep the green block into the dustpan"] * self.num_envs

    def _build_extracted_obs(self, raw_obs: dict) -> dict:
        obs_image = None
        if isinstance(raw_obs, dict):
            sd = raw_obs.get("sensor_data", {})
            cam = sd.get("external_cam", {})
            if "rgb" in cam:
                obs_image = cam["rgb"].to(torch.uint8)

        if obs_image is None:
            frame = self.render()
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 \
                    else np.clip(frame, 0, 255).astype(np.uint8)
            obs_image = torch.from_numpy(frame).to(torch.uint8).unsqueeze(0)

        return {
            "main_images": obs_image,
            "task_descriptions": self.get_language_instruction(),
        }

    def reset(self, seed=None, options=None):
        raw_obs, infos = super().reset(seed=seed, options=options)
        infos["extracted_obs"] = self._build_extracted_obs(raw_obs)
        return raw_obs, infos

    def step(self, action):
        raw_obs, reward, terminations, truncations, infos = super().step(action)
        infos["extracted_obs"] = self._build_extracted_obs(raw_obs)
        return raw_obs, reward, terminations, truncations, infos


def _to_uint8_frame(frame) -> np.ndarray | None:
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


def _setup_hybrid_drives(base_env,
                         arm_stiffness: float = 1e5,
                         arm_damping: float = 1e3,
                         gripper_stiffness: float = 2000.0,
                         gripper_damping: float = 100.0,
                         gripper_force_limit: float = 500.0):
    for i, joint in enumerate(base_env.agent.robot.get_active_joints()):
        if i < 7:
            joint.set_drive_properties(arm_stiffness, arm_damping, force_limit=1e6, mode="force")
        else:
            joint.set_drive_properties(
                gripper_stiffness, gripper_damping,
                force_limit=gripper_force_limit, mode="force",
            )


def _hybrid_step(base_env, arm_target: np.ndarray, gripper_target: np.ndarray,
                 sim_steps: int = 5):
    robot = base_env.agent.robot
    device = base_env.device
    arm_t = torch.tensor(arm_target, device=device, dtype=torch.float32)
    grip_t = torch.tensor(gripper_target, device=device, dtype=torch.float32)
    all_targets = torch.cat([arm_t, grip_t]).unsqueeze(0)
    robot.set_joint_drive_targets(all_targets, robot.get_active_joints())
    for _ in range(sim_steps):
        base_env.scene.step()
    base_env.scene.update_render()


if __name__ == "__main__":
    import argparse
    import gymnasium as gym
    import h5py
    import imageio

    parser = argparse.ArgumentParser(description="Sweep-v1 simulation")
    parser.add_argument("--replay", type=int, default=0, metavar="EPISODE_ID",
                        help="Replay a real-robot episode from the dataset (default: 0)")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR,
                        help="Path to the sweep HDF5 dataset directory")
    parser.add_argument("--out", type=str, default=None,
                        help="Output video path (default: ./render/Sweep/replay_epN.mp4)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--sim-steps", type=int, default=5,
                        help="Physics substeps per replay frame")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo mode with random actions instead of replay")
    args = parser.parse_args()

    if not args.demo:
        # ── Replay mode ────────────────────────────────────────────────────────
        # Update module-level variables so _initialize_episode loads dataset qpos
        REPLAY_EPISODE_ID = args.replay
        DATASET_DIR = args.dataset_dir

        h5_path = os.path.join(DATASET_DIR, f"episode_{args.replay}.hdf5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Episode not found: {h5_path}")
        with h5py.File(h5_path, "r") as _f:
            qpos_raw = _f["observations/joint_pos"][:]  # (T, 9)
        qpos = qpos_raw.astype(np.float32)
        qpos[:, 7:9] /= 2.0  # total-width -> per-finger
        T = qpos.shape[0]
        print(f"Episode {args.replay}: {T} frames")
        print(f"Arm range:     [{qpos[:, :7].min():.4f}, {qpos[:, :7].max():.4f}]")
        print(f"Gripper range: [{qpos[:, 7:9].min():.6f}, {qpos[:, 7:9].max():.6f}]")

        env = gym.make("Sweep-v1", obs_mode="rgb", render_mode="rgb_array")
        obs, _ = env.reset()
        base_env = env.unwrapped

        _setup_hybrid_drives(base_env, arm_stiffness=1e5, arm_damping=1e3)

        # Re-initialize to qpos[0] and set drive targets to avoid initial jerk
        robot = base_env.agent.robot
        q0 = torch.tensor(qpos[0], device=base_env.device, dtype=torch.float32).unsqueeze(0)
        robot.set_qpos(q0)
        robot.set_qvel(torch.zeros((1, 9), device=base_env.device))
        robot.set_joint_drive_targets(q0, robot.get_active_joints())
        base_env.scene.step()
        base_env.scene.update_render()

        frames = []
        f0 = _to_uint8_frame(env.render())
        if f0 is not None:
            frames.append(f0)

        print(f"Replaying {T} frames (sim_steps={args.sim_steps})...")
        for t in range(T):
            _hybrid_step(base_env, qpos[t, :7], qpos[t, 7:9], sim_steps=args.sim_steps)
            frame = _to_uint8_frame(env.render())
            if frame is not None:
                frames.append(frame)
            if base_env.evaluate()["success"][0]:
                print(f"  SUCCESS at frame {t}")

        out_path = args.out or f"./render/Sweep/replay_ep{args.replay}.mp4"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if frames:
            imageio.mimsave(out_path, frames, fps=args.fps)
            print(f"Saved {len(frames)} frames → {out_path}")
        else:
            print("No frames captured.")
        env.close()

    else:
        # ── Demo mode (random actions) ─────────────────────────────────────────
        env = gym.make("Sweep-v1", obs_mode="rgb", render_mode="rgb_array")
        obs, _ = env.reset()
        base = env.unwrapped

        print("Sensor cameras:", list(obs["sensor_data"].keys()))
        print(f"Table top z     : {base.TABLE_Z}")
        print(f"Table centre x  : {base._TABLE_CENTER_X}")
        print(f"Broom position  : {base._BROOM_XY}")
        print(f"Block position  : {base._BLOCK_XY}")
        print(f"Dustpan position: {base._DUSTPAN_XY}")

        def get_frame(o):
            img = o["sensor_data"]["external_cam"]["rgb"]
            if img.ndim == 4:
                img = img[0]
            img = img.cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            return img

        frames = [get_frame(obs)]
        for _ in range(60):
            o, _, done, trunc, _ = env.step(env.action_space.sample())
            frames.append(get_frame(o))
            if done or trunc:
                break

        os.makedirs("./render/Sweep", exist_ok=True)
        imageio.imwrite("./render/Sweep/screenshot.png", frames[0])
        imageio.mimsave("./render/Sweep/demo.mp4", frames, fps=20)
        print("Saved screenshot and demo video to render/Sweep/")
        env.close()
