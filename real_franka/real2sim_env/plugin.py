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
DATASET_DIR = "/storage/zhijun/real_franka/pick_and_place"


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


@register_env("Plugin-v1", max_episode_steps=3000)
class PluginEnv(BaseEnv):

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

    # ── Scene parameters (identical to pick_and_place) ───────────────────────
    TABLE_Z = 0.03
    _TABLE_CENTER_X = 0.501
    _TABLE_HALF = (0.30, 0.60, 0.0175)
    TABLE_STATIC_FRICTION = 1.0
    TABLE_DYNAMIC_FRICTION = 1.0
    TABLE_RESTITUTION = 0.0
    BASE_PEDESTAL_SIZE = 0.95

    # ── Block: identical to pick_and_place (4 × 4 × 6 cm, orange) ──────────
    BLOCK_HALF_SIZE = [0.02, 0.02, 0.03]
    BLOCK_DENSITY = 200.0
    BLOCK_STATIC_FRICTION = 0.5
    BLOCK_DYNAMIC_FRICTION = 0.5
    BLOCK_RESTITUTION = 0.0

    # ── Socket ("peg") geometry ──────────────────────────────────────────────
    # Outer footprint: 8 × 8 cm  →  half = 0.04
    # Inner hole:      5 × 5 cm  →  half = 0.025
    # Wall thickness:  (8-5)/2   =  1.5 cm → half = 0.0075
    # Total height:    2 cm      →  half = 0.01
    # Base thickness:  0.5 cm    →  half = 0.0025
    # Wall height above base:    2 - 0.5 = 1.5 cm → half = 0.0075
    SOCKET_OUTER_HALF = 0.04
    SOCKET_INNER_HALF = 0.024
    SOCKET_WALL_HALF_T = 0.0075   # (outer - inner) / 2
    SOCKET_BASE_HALF_Z = 0.0025   # base plate half-thickness (0.5 cm)
    SOCKET_WALL_HALF_Z = 0.0075   # wall half-height above base (1.5 cm)

    # ── Initial positions from pick_and_place traj15 ─────────────────────────
    # Block: traj15 block_xy
    _BLOCK_XY  = (_TABLE_CENTER_X + 0.083, -0.15)
    # Socket: traj15 coaster_xy (socket sits where the coaster was)
    _SOCKET_XY = (_TABLE_CENTER_X + 0.0,   0.035)

    # traj15 qpos_ref_0
    _QPOS_REF = [
        0.04231681674718857, -0.46236130595207214, -0.018441837280988693,
        -2.6430587768554688, -0.002512579783797264, 2.179812431335449, 0.8177617788314819,
    ]

    def __init__(self, *args, robot_uids="panda_high_friction", **kwargs):
        kwargs.setdefault("enable_shadow", True)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ── Camera helpers ───────────────────────────────────────────────────────

    def _make_cam_pose(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
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
        # Front light: from camera side (-X world = from screen toward scene), slight downward tilt
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
        _wood_tex = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            "..", "..", "rlinf", "envs", "maniskill", "assets",
            "carrot", "more_table", "textures", "006.png",
        ))
        tbl_mat.base_color_texture = sapien.render.RenderTexture2D(
            filename=_wood_tex, mipmap_levels=4, srgb=True
        )
        tbl_mat.roughness = 0.55
        tbl_mat.metallic = 0.0
        tbl_mat.specular = 0.5
        tbl_b.add_box_visual(half_size=self._TABLE_HALF, material=tbl_mat)
        tbl_b.set_initial_pose(
            sapien.Pose(p=[self._TABLE_CENTER_X, 0, self.TABLE_Z - self._TABLE_HALF[2]])
        )
        self._table = tbl_b.build_kinematic(name="table")

        # Block: same as pick_and_place (orange-red, 4×4×6 cm)
        blk_b = self.scene.create_actor_builder()
        blk_phys = PhysxMaterial(
            static_friction=self.BLOCK_STATIC_FRICTION,
            dynamic_friction=self.BLOCK_DYNAMIC_FRICTION,
            restitution=self.BLOCK_RESTITUTION,
        )
        blk_b.add_box_collision(
            half_size=self.BLOCK_HALF_SIZE,
            material=blk_phys,
            density=self.BLOCK_DENSITY,
        )
        blk_mat = sapien.render.RenderMaterial()
        blk_mat.base_color = [0.82, 0.22, 0.06, 1.0]  # orange-red, same as pick_and_place
        blk_mat.roughness = 0.5
        blk_mat.metallic = 0.0
        blk_mat.specular = 0.6
        blk_b.add_box_visual(half_size=self.BLOCK_HALF_SIZE, material=blk_mat)
        blk_b.set_initial_pose(sapien.Pose(p=[
            self._BLOCK_XY[0], self._BLOCK_XY[1],
            self.TABLE_Z + self.BLOCK_HALF_SIZE[2],
        ]))
        self.cube = blk_b.build(name="cube")

        # Socket (kinematic compound actor)
        self._build_socket()

    def _build_socket(self):
        """
        White square socket (peg hole):
          outer 8×8 cm, inner hole 5×5 cm, total height 2 cm, base 0.5 cm thick.

        Local frame (anchor = socket xy-centre at table surface):
          Base plate:  8×8×0.5 cm, z=[0, 0.005]
          Four walls:  1.5 cm thick, 1.5 cm tall above base, z=[0.005, 0.020]
            +X wall spans full 8 cm in Y
            -X wall spans full 8 cm in Y
            +Y wall fills the inner gap in X (5 cm span)
            -Y wall fills the inner gap in X (5 cm span)
          Hole (open top): 5×5 cm centred in the socket
        """
        cx, cy = self._SOCKET_XY
        oh  = self.SOCKET_OUTER_HALF   # 0.04
        ih  = self.SOCKET_INNER_HALF   # 0.025
        wt  = self.SOCKET_WALL_HALF_T  # 0.0075
        bz  = self.SOCKET_BASE_HALF_Z  # 0.0025
        wz  = self.SOCKET_WALL_HALF_Z  # 0.0075

        white = sapien.render.RenderMaterial()
        white.base_color = [0.95, 0.95, 0.95, 1.0]
        white.roughness = 0.25
        white.metallic = 0.0
        white.specular = 0.5

        phys = PhysxMaterial(static_friction=0.5, dynamic_friction=0.5, restitution=0.0)

        b = self.scene.create_actor_builder()

        # Base plate: 8×8 cm × 0.5 cm, z=[0, 2*bz]
        base_pose = sapien.Pose(p=[0, 0, bz])
        b.add_box_collision(half_size=[oh, oh, bz], pose=base_pose, material=phys)
        b.add_box_visual(half_size=[oh, oh, bz], pose=base_pose, material=white)

        wall_cz = 2 * bz + wz  # 0.005 + 0.0075 = 0.0125

        # +X wall (x = ih → oh), spans full 8 cm in Y
        b.add_box_collision(half_size=[wt, oh, wz],
                            pose=sapien.Pose(p=[ih + wt, 0, wall_cz]), material=phys)
        b.add_box_visual(half_size=[wt, oh, wz],
                         pose=sapien.Pose(p=[ih + wt, 0, wall_cz]), material=white)

        # -X wall (x = -oh → -ih), spans full 8 cm in Y
        b.add_box_collision(half_size=[wt, oh, wz],
                            pose=sapien.Pose(p=[-(ih + wt), 0, wall_cz]), material=phys)
        b.add_box_visual(half_size=[wt, oh, wz],
                         pose=sapien.Pose(p=[-(ih + wt), 0, wall_cz]), material=white)

        # +Y wall (y = ih → oh), fills inner X gap only (5 cm span)
        b.add_box_collision(half_size=[ih, wt, wz],
                            pose=sapien.Pose(p=[0, ih + wt, wall_cz]), material=phys)
        b.add_box_visual(half_size=[ih, wt, wz],
                         pose=sapien.Pose(p=[0, ih + wt, wall_cz]), material=white)

        # -Y wall (y = -oh → -ih), fills inner X gap only (5 cm span)
        b.add_box_collision(half_size=[ih, wt, wz],
                            pose=sapien.Pose(p=[0, -(ih + wt), wall_cz]), material=phys)
        b.add_box_visual(half_size=[ih, wt, wz],
                         pose=sapien.Pose(p=[0, -(ih + wt), wall_cz]), material=white)

        b.set_initial_pose(sapien.Pose(p=[cx, cy, self.TABLE_Z]))
        self.socket = b.build_kinematic(name="socket")

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

            # Block at traj15 block position
            blk = torch.zeros((n, 3), device=self.device)
            blk[:, 0] = self._BLOCK_XY[0]
            blk[:, 1] = self._BLOCK_XY[1]
            blk[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2]
            self.cube.set_pose(Pose.create_from_pq(p=blk))

            # Socket at traj15 coaster position (kinematic, fixed)
            self.socket.set_pose(
                sapien.Pose(p=[self._SOCKET_XY[0], self._SOCKET_XY[1], self.TABLE_Z])
            )

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _is_block_inserted(self) -> torch.Tensor:
        """True when the block is inside the socket hole and lowered below the rim."""
        bp = self.cube.pose.p    # (B, 3)
        sp = self.socket.pose.p  # (B, 3)

        rel_x = bp[:, 0] - sp[:, 0]
        rel_y = bp[:, 1] - sp[:, 1]
        rel_z = bp[:, 2] - sp[:, 2]

        bx, by, bz = self.BLOCK_HALF_SIZE
        ih = self.SOCKET_INNER_HALF
        socket_top_z = 2 * self.SOCKET_BASE_HALF_Z + 2 * self.SOCKET_WALL_HALF_Z  # 0.020

        # Block XY centre within hole with clearance
        x_in = torch.abs(rel_x) < (ih - bx)
        y_in = torch.abs(rel_y) < (ih - by)
        # Block bottom (rel_z - bz) below socket rim top
        z_in = (rel_z - bz) < socket_top_z

        return x_in & y_in & z_in

    def evaluate(self):
        inserted = self._is_block_inserted()

        # Require block to be nearly stationary (not bouncing)
        lin_vel = self.cube.linear_velocity
        ang_vel = self.cube.angular_velocity
        vel_ok = (torch.norm(lin_vel, dim=1) < 0.05) & (torch.norm(ang_vel, dim=1) < 0.5)

        # Gripper must be open (block released)
        qpos = self.agent.robot.get_qpos()
        gripper_open = (qpos[:, 7] + qpos[:, 8]) > 0.03

        return {
            "success": inserted & vel_ok & gripper_open,
            "block_inserted": inserted,
        }

    # ── Reward ───────────────────────────────────────────────────────────────

    def _is_block_grasped(self) -> torch.Tensor:
        qpos = self.agent.robot.get_qpos()
        gripper_width = qpos[:, 7] + qpos[:, 8]
        cube_z = self.cube.pose.p[:, 2]
        cube_lifted = cube_z > self.TABLE_Z + self.BLOCK_HALF_SIZE[2] + 0.015
        gripper_gripping = gripper_width < 0.05
        return cube_lifted & gripper_gripping

    def compute_dense_reward(self, obs, action, info):
        tcp_pos   = self.agent.tcp.pose.p
        cube_pos  = self.cube.pose.p
        socket_pos = self.socket.pose.p

        is_grasped   = self._is_block_grasped()
        is_grasped_f = is_grasped.float()

        dist_tcp_cube    = torch.norm(tcp_pos - cube_pos, dim=1)
        dist_cube_socket = torch.norm(cube_pos - socket_pos, dim=1)

        approach   = (1.0 - is_grasped_f) * (1.0 - torch.tanh(5.0 * dist_tcp_cube)) * 0.1
        grasp      = is_grasped_f * 1.0
        transport  = is_grasped_f * (1.0 - torch.tanh(5.0 * dist_cube_socket)) * 0.5
        success    = info["success"].float() * 3.0

        return approach + grasp + transport + success

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    # ── Observation helpers ──────────────────────────────────────────────────

    def get_language_instruction(self):
        return ["pick up the block and insert it into the socket"] * self.num_envs

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

    parser = argparse.ArgumentParser(description="Plugin-v1 simulation")
    parser.add_argument("--replay", type=int, default=15, metavar="EPISODE_ID",
                        help="Replay a pick-and-place episode in the plugin env (default: 15)")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR,
                        help="Path to the pick-and-place HDF5 dataset directory")
    parser.add_argument("--out", type=str, default=None,
                        help="Output video path (default: ./render/Plugin/replay_epN.mp4)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--sim-steps", type=int, default=5,
                        help="Physics substeps per replay frame")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo mode with random actions instead of replay")
    args = parser.parse_args()

    if not args.demo:
        # ── Replay mode: replay pick_and_place traj in Plugin-v1 env ──────────
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

        env = gym.make("Plugin-v1", obs_mode="rgb", render_mode="rgb_array")
        obs, _ = env.reset()
        base_env = env.unwrapped

        print(f"Block position  : {base_env._BLOCK_XY}")
        print(f"Socket position : {base_env._SOCKET_XY}")

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

        out_path = args.out or f"./render/Plugin/replay_ep{args.replay}.mp4"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if frames:
            imageio.mimsave(out_path, frames, fps=args.fps)
            print(f"Saved {len(frames)} frames → {out_path}")
        else:
            print("No frames captured.")
        env.close()

    else:
        # ── Demo mode (random actions, --demo flag) ────────────────────────────
        env = gym.make("Plugin-v1", obs_mode="rgb", render_mode="rgb_array")
        obs, _ = env.reset()
        base = env.unwrapped

        print("Sensor cameras:", list(obs["sensor_data"].keys()))
        print(f"Table top z       : {base.TABLE_Z}")
        print(f"Block position    : {base._BLOCK_XY}")
        print(f"Socket position   : {base._SOCKET_XY}")
        print(f"Socket outer      : {base.SOCKET_OUTER_HALF*2*100:.0f} cm")
        print(f"Socket inner hole : {base.SOCKET_INNER_HALF*2*100:.0f} cm")
        print(f"Block XY fit gap  : {(base.SOCKET_INNER_HALF - base.BLOCK_HALF_SIZE[0])*1000:.1f} mm per side")

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

        os.makedirs("./render/Plugin", exist_ok=True)
        imageio.imwrite("./render/Plugin/screenshot.png", frames[0])
        imageio.mimsave("./render/Plugin/demo.mp4", frames, fps=20)
        print("Saved to render/Plugin/")
        env.close()
