"""BlockStack-v1（多轨迹版本）—— 将白色方块放到灰色方块顶部。

与 blockstack_env.py 相比，本文件新增：
  - 5 条轨迹初始配置（traj "12"–"16"，block_xy 待手动填写）
  - 后方相机（back_cam）视角
  - _initialize_episode 三种模式（与 pick_and_place.py 保持一致）：
      1. episode_id → 确定性（固定轨迹）
      2. TRAJ_ID 全局变量为已知轨迹 → Replay/prepare 脚本模式
      3. 否则（TRAJ_ID=="random"）→ RL 训练随机模式
  - build_blockstack_states_from_env 辅助函数
  - __main__ 对两个相机、t0/t12 各保存截图+对比+视频

场景参数（参照图中真实环境）：
  - 灰色方块 (gray_block)：4 cm × 4 cm × 4 cm，作为底座（kinematic 目标）
  - 白色方块 (white_block)：4 cm × 4 cm × 4 cm，需要被抓取并放置
  - 夹爪：Panda + 2 cm 指尖延伸（panda_v2_extended.urdf，TCP 偏移 0.1234 m）

坐标系：世界系 = 机器人底座系（Panda 底座固定在原点）。
"""

import os
import numpy as np
import torch
import sapien
from scipy.spatial.transform import Rotation
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import Panda
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building.ground import build_ground
from sapien.physx import PhysxMaterial


USE_REF_12 = False   # False = t0 初始姿态；True = t12 中间姿态（由外部脚本覆盖）
TRAJ_ID    = "12"    # 轨迹 ID，可选 "12"–"16"；设为 "random" 启用 RL 随机模式
RENDER_BASE_DIR = "real_franka/real2sim_env/render"


# ── 辅助函数 ─────────────────────────────────────────────────────────────────────

def build_blockstack_states_from_env(base_env, apply_bias: bool = False):
    """Build 7D BlockStack proprio state from env internals.

    Returns:
        states: (N, 7) float32 [x, y, z, euler_x, euler_y, euler_z, gripper_total_width]
        gripper_total_width: (N,) float32 in meters [0, 0.08]
    """
    ee_pose_T = base_env.agent.ee_pose_at_robot_base.to_transformation_matrix()
    if torch.is_tensor(ee_pose_T):
        ee_pose_T = ee_pose_T.detach().cpu().numpy()
    ee_pose_T = np.asarray(ee_pose_T)
    if ee_pose_T.ndim == 2:
        ee_pose_T = ee_pose_T[None, ...]

    pos   = ee_pose_T[:, :3, 3].astype(np.float32)
    euler = Rotation.from_matrix(ee_pose_T[:, :3, :3]).as_euler("xyz").astype(np.float32)

    qpos = base_env.agent.robot.get_qpos()
    if torch.is_tensor(qpos):
        qpos = qpos.detach().cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim == 1:
        qpos = qpos[None, ...]

    gripper_total_width = (qpos[:, 7] + qpos[:, 8]).astype(np.float32)
    states = np.concatenate([pos, euler, gripper_total_width[:, None]], axis=1)

    if apply_bias:
        states = states.copy()
        states[:, 2] += 0.1
        states[:, 5] += -np.pi / 4

    return states, gripper_total_width


# ── 延伸夹爪机器人 ───────────────────────────────────────────────────────────────

@register_agent()
class PandaExtendedGripper(Panda):
    """Panda with 2 cm finger extension + high-friction gripper pads.

    Uses panda_v2_extended.urdf where:
      - panda_hand_tcp_joint xyz = 0 0 0.1234  (+20 mm vs. stock 0.1034)
      - Each finger has an extra 20 mm collision box appended beyond the
        rubber tip (center at z = 64.5 mm from finger joint origin).
    """
    uid = "panda_extended_gripper"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2_extended.urdf"
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


# ── 环境 ─────────────────────────────────────────────────────────────────────────

@register_env("BlockStack-v1", max_episode_steps=3000)
class BlockStackEnv(BaseEnv):
    """将白色方块堆叠到灰色方块顶部的 Real2Sim 环境（多轨迹版本）。

    初始状态：两块方块放置在桌面上（位置按轨迹配置）。
    成功条件：白色方块稳定放置于灰色方块正上方，且夹爪已张开。
    """

    SUPPORTED_ROBOTS = ["panda_extended_gripper"]
    agent: PandaExtendedGripper

    # ── 正面相机标定参数（2026-02-15）────────────────────────────────────────────
    # og 原始旋转矩阵
    _R_og = np.array([
        [ 0.02816316,  0.21788680, -0.97556762],
        [ 0.99959024, -0.00114196,  0.02860160],
        [ 0.00511786, -0.97597338, -0.21782968],
    ])
    # 场景补偿：绕世界 Z 轴旋转 -2.5°
    _R  = (Rotation.from_euler("z", -2.5, degrees=True).as_matrix() @ _R_og)
    _t  = np.array([1.10002696, -0.00701879, 0.25898290])

    # ── 侧后方相机标定参数（2026-01-11）──────────────────────────────────────────
    _R2 = np.array([
        [ 0.49248784, -0.28922532,  0.82085592],
        [-0.86866200, -0.10517417,  0.48411230],
        [-0.05368470, -0.95146577, -0.30303605],
    ])
    _t2 = np.array([0.05115059, -0.33471246, 0.25403179])

    # 内参（两个相机共用 RealSense D435 内参）
    _K = np.array([
        [607.875,   0.0,   348.961],
        [  0.0,  607.719, 270.486],
        [  0.0,    0.0,     1.0  ],
    ])

    # ── 场景参数 ──────────────────────────────────────────────────────────────────
    TABLE_Z          = 0.03
    _TABLE_CENTER_X  = 0.501
    _TABLE_HALF      = (0.30, 0.60, 0.0175)
    TABLE_STATIC_FRICTION  = 1.0
    TABLE_DYNAMIC_FRICTION = 1.0
    TABLE_RESTITUTION      = 0.0

    # 方块：4 cm × 4 cm × 4 cm
    BLOCK_HALF_SIZE        = [0.02, 0.02, 0.02]
    BLOCK_DENSITY          = 200.0
    BLOCK_STATIC_FRICTION  = 0.5
    BLOCK_DYNAMIC_FRICTION = 0.5
    BLOCK_RESTITUTION      = 0.0

    BASE_PEDESTAL_SIZE = 0.95

    # 灰色方块（底座目标）固定位置
    _GRAY_BLOCK_XY = (0.510, 0.15)

    # ── 各轨迹初始化参数 ──────────────────────────────────────────────────────────
    # block_xy:    白色方块初始位置 (x, y)，需手动测量后填入
    # qpos_ref_0:  t=0  时刻真实关节角（弧度，7 个关节）
    # qpos_ref_12: t=12 时刻真实关节角（弧度，7 个关节）
    _TRAJ_CONFIGS = {
        "12": {
            "block_xy":    (0.501, -0.12),   # TODO: 手动测量后填入
            "qpos_ref_0":  [0.014266801066696644, -0.5575937032699585, -0.020573778077960014, -2.6751248836517334, 8.969002374215052e-05, 2.101120710372925, 0.8732436299324036],
            "qpos_ref_12": [0.07514475286006927, 0.14888109266757965, 0.025265969336032867, -2.2255454063415527, 8.305629307869822e-05, 2.3443357944488525, 0.9800084233283997],
        },
        "13": {
            "block_xy":    (0.501, -0.12),   # TODO: 手动测量后填入
            "qpos_ref_0":  [0.01700538583099842, -0.7167431116104126, -0.008280294016003609, -2.650082588195801, 0.009025879204273224, 1.9203811883926392, 0.792972207069397],
            "qpos_ref_12": [0.27984341979026794, 0.028549158945679665, 0.010437212884426117, -2.446746349334717, 0.013412341475486755, 2.4676806926727295, 1.181433916091919],
        },
        "14": {
            "block_xy":    (0.501, -0.12),   # TODO: 手动测量后填入
            "qpos_ref_0":  [0.0031628001015633345, -0.7538919448852539, 0.01381409727036953, -2.801128625869751, 0.007255664095282555, 2.0222644805908203, 0.8338807225227356],
            "qpos_ref_12": [0.0595947690308094, 0.11300260573625565, 0.015491015277802944, -2.262223720550537, 0.0065317400731146336, 2.3801426887512207, 1.0007210969924927],
        },
        "15": {
            "block_xy":    (0.501, -0.12),   # TODO: 手动测量后填入
            "qpos_ref_0":  [0.0156794972717762, -0.5837724208831787, -0.012542960233986378, -2.735301971435547, -0.015894290059804916, 2.115887403488159, 0.8262093663215637],
            "qpos_ref_12": [0.090180903673172, 0.27683764696121216, 0.040512364357709885, -2.0962936878204346, -0.015820803120732307, 2.3434646129608154, 0.9916989803314209],
        },
        "16": {
            "block_xy":    (0.501, -0.12),   # TODO: 手动测量后填入
            "qpos_ref_0":  [-0.041274722665548325, -0.7419406771659851, 0.03363127261400223, -2.7966506481170654, 0.009078307077288628, 2.0164427757263184, 0.7496281862258911],
            "qpos_ref_12": [0.21015943586826324, 0.2664976418018341, 0.042176347225904465, -2.170553207397461, 0.00908923801034689, 2.4183542728424072, 1.1711584329605103],
        },
    }

    def __init__(self, *args, robot_uids="panda_extended_gripper", **kwargs):
        kwargs.setdefault("enable_shadow", True)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ── 相机辅助 ──────────────────────────────────────────────────────────────────

    def _make_cam_pose(self, R, t):
        """OpenCV 旋转矩阵 + 平移向量 → SAPIEN Pose。"""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = t
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
        pose_front = self._make_cam_pose(self._R,  self._t)
        pose_side  = self._make_cam_pose(self._R2, self._t2)
        return [
            CameraConfig("external_cam", pose_front, 640, 480,
                         near=0.01, far=10, intrinsic=self._K),
            CameraConfig("back_cam",     pose_side,  640, 480,
                         near=0.01, far=10, intrinsic=self._K),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = self._make_cam_pose(self._R, self._t)
        return CameraConfig("render_camera", pose, 640, 480,
                            near=0.01, far=100, intrinsic=self._K,
                            shader_pack="default")

    # ── 加载场景 ──────────────────────────────────────────────────────────────────

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.25, 0.25, 0.28])
        self.scene.add_directional_light(
            [0.6, 0.4, -1.0], [1.1, 1.05, 1.0],
            shadow=True, shadow_scale=5, shadow_map_size=4096,
        )
        self.scene.add_directional_light([-1.0, 0.2, -0.5], [0.35, 0.38, 0.45])
        self.scene.add_point_light([0.5, 0.0, 1.2], [1.8, 1.7, 1.6], shadow=False)

    def _load_scene(self, options: dict):
        build_ground(self.scene, floor_width=10, altitude=-self.BASE_PEDESTAL_SIZE)

        # 机器人基座台（黑色金属台面）
        ped_x_min = -self.BASE_PEDESTAL_SIZE / 2
        ped_x_max = 1.2
        ped_hx = (ped_x_max - ped_x_min) / 2
        ped_cx = (ped_x_max + ped_x_min) / 2
        ped_hy = self.BASE_PEDESTAL_SIZE / 2
        ped_hz = self.BASE_PEDESTAL_SIZE / 2
        ped_builder = self.scene.create_actor_builder()
        ped_builder.add_box_collision(half_size=[ped_hx, ped_hy, ped_hz])
        ped_mat = sapien.render.RenderMaterial()
        ped_mat.base_color = [0.2, 0.2, 0.2, 1.0]
        ped_mat.roughness  = 0.2
        ped_mat.metallic   = 1.0
        ped_mat.specular   = 0.9
        ped_builder.add_box_visual(half_size=[ped_hx, ped_hy, ped_hz], material=ped_mat)
        ped_builder.set_initial_pose(sapien.Pose(p=[ped_cx, 0, -ped_hz]))
        self._base_pedestal = ped_builder.build_kinematic(name="robot_base_pedestal")

        # 桌面（浅灰色铝合金）
        table_builder = self.scene.create_actor_builder()
        table_phys = PhysxMaterial(
            static_friction=self.TABLE_STATIC_FRICTION,
            dynamic_friction=self.TABLE_DYNAMIC_FRICTION,
            restitution=self.TABLE_RESTITUTION,
        )
        table_builder.add_box_collision(half_size=self._TABLE_HALF, material=table_phys)
        table_mat = sapien.render.RenderMaterial()
        table_mat.base_color = [0.75, 0.75, 0.75, 1.0]
        table_mat.roughness  = 0.4
        table_mat.metallic   = 0.6
        table_mat.specular   = 0.5
        table_builder.add_box_visual(half_size=self._TABLE_HALF, material=table_mat)
        table_builder.set_initial_pose(
            sapien.Pose(p=[self._TABLE_CENTER_X, 0, self.TABLE_Z - self._TABLE_HALF[2]])
        )
        self._table = table_builder.build_kinematic(name="table")

        block_phys = PhysxMaterial(
            static_friction=self.BLOCK_STATIC_FRICTION,
            dynamic_friction=self.BLOCK_DYNAMIC_FRICTION,
            restitution=self.BLOCK_RESTITUTION,
        )

        # 白色方块（需抓取）
        wb = self.scene.create_actor_builder()
        wb.add_box_collision(half_size=self.BLOCK_HALF_SIZE, material=block_phys,
                             density=self.BLOCK_DENSITY)
        white_mat = sapien.render.RenderMaterial()
        white_mat.base_color = [0.95, 0.95, 0.95, 1.0]
        white_mat.roughness  = 0.4
        white_mat.metallic   = 0.0
        white_mat.specular   = 0.3
        wb.add_box_visual(half_size=self.BLOCK_HALF_SIZE, material=white_mat)
        wb.set_initial_pose(sapien.Pose(p=[
            self._TRAJ_CONFIGS["12"]["block_xy"][0],
            self._TRAJ_CONFIGS["12"]["block_xy"][1],
            self.TABLE_Z + self.BLOCK_HALF_SIZE[2],
        ]))
        self.white_block = wb.build(name="white_block")

        # 灰色方块（底座目标）
        gb = self.scene.create_actor_builder()
        gb.add_box_collision(half_size=self.BLOCK_HALF_SIZE, material=block_phys,
                             density=self.BLOCK_DENSITY)
        gray_mat = sapien.render.RenderMaterial()
        gray_mat.base_color = [0.45, 0.45, 0.48, 1.0]
        gray_mat.roughness  = 0.5
        gray_mat.metallic   = 0.2
        gray_mat.specular   = 0.4
        gb.add_box_visual(half_size=self.BLOCK_HALF_SIZE, material=gray_mat)
        gb.set_initial_pose(sapien.Pose(p=[
            self._GRAY_BLOCK_XY[0],
            self._GRAY_BLOCK_XY[1],
            self.TABLE_Z + self.BLOCK_HALF_SIZE[2],
        ]))
        self.gray_block = gb.build(name="gray_block")

    # ── 回合初始化 ────────────────────────────────────────────────────────────────

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # ── 初始化模式选择 ────────────────────────────────────────────────────
            # 优先级：1) options["episode_id"] → 确定性选固定轨迹
            #          2) TRAJ_ID 全局变量为已知轨迹 → Replay/prepare 脚本模式
            #          3) 否则（TRAJ_ID == "random"）→ RL 训练随机模式
            episode_id = options.get("episode_id", None)
            traj_keys  = list(self._TRAJ_CONFIGS.keys())

            if episode_id is not None:
                # ── 确定性模式（来自 ManiskillEnv use_fixed_reset_state_ids）──────
                traj_idx = int(episode_id[0].item()) % len(traj_keys)
                traj_cfg = self._TRAJ_CONFIGS[traj_keys[traj_idx]]
                qpos_rad = traj_cfg["qpos_ref_12"] if USE_REF_12 else traj_cfg["qpos_ref_0"]
                block_x, block_y = traj_cfg["block_xy"]
            elif TRAJ_ID in self._TRAJ_CONFIGS:
                # ── Replay / prepare 模式：使用固定轨迹配置 ──────────────────────
                traj_cfg = self._TRAJ_CONFIGS[TRAJ_ID]
                qpos_rad = traj_cfg["qpos_ref_12"] if USE_REF_12 else traj_cfg["qpos_ref_0"]
                block_x, block_y = traj_cfg["block_xy"]
            else:
                # ── RL 训练 / MimicGen 生成模式（TRAJ_ID == "random"）────────────
                # 白色方块：在桌面合理范围内随机采样
                block_x = torch.tensor(
                    self._TABLE_CENTER_X + np.random.uniform(-0.10, 0.10, size=b),
                    device=self.device, dtype=torch.float32,
                )
                block_y = torch.tensor(
                    np.random.uniform(-0.15, -0.03, size=b),
                    device=self.device, dtype=torch.float32,
                )

                # 向量化最近轨迹查找：为每个 env 独立找最近 qpos
                traj_vals = list(self._TRAJ_CONFIGS.values())
                traj_bxy  = torch.tensor(
                    [c["block_xy"] for c in traj_vals],
                    device=self.device, dtype=torch.float32,
                )  # (T, 2)
                env_bxy  = torch.stack([block_x, block_y], dim=1)  # (b, 2)
                dists    = ((env_bxy[:, None] - traj_bxy[None]) ** 2).sum(-1)  # (b, T)
                best_idx = dists.argmin(dim=1)  # (b,)

                all_qpos = torch.tensor(
                    [c["qpos_ref_0"] for c in traj_vals],
                    device=self.device, dtype=torch.float32,
                )  # (T, 7)
                qpos_per_env = all_qpos[best_idx]  # (b, 7)
                self._nearest_traj_id = traj_keys[best_idx[0].item()]

                init_qpos = torch.cat(
                    [qpos_per_env, torch.full((b, 2), 0.04, device=self.device)], dim=1
                )  # (b, 9)
                self.agent.robot.set_qpos(init_qpos)
                self.agent.robot.set_qvel(torch.zeros((b, 9), device=self.device))

                self._table.set_pose(
                    sapien.Pose(p=[self._TABLE_CENTER_X, 0,
                                   self.TABLE_Z - self._TABLE_HALF[2]])
                )

                white_xyz = torch.zeros((b, 3), device=self.device)
                white_xyz[:, 0] = block_x
                white_xyz[:, 1] = block_y
                white_xyz[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2]
                self.white_block.set_pose(Pose.create_from_pq(p=white_xyz))

                gray_xyz = torch.zeros((b, 3), device=self.device)
                gray_xyz[:, 0] = self._GRAY_BLOCK_XY[0]
                gray_xyz[:, 1] = self._GRAY_BLOCK_XY[1]
                gray_xyz[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2]
                self.gray_block.set_pose(Pose.create_from_pq(p=gray_xyz))
                return

            # 确定性 / Replay 模式的公共初始化路径
            init_qpos = torch.tensor(qpos_rad + [0.04, 0.04], device=self.device)
            self.agent.robot.set_qpos(init_qpos.repeat(b, 1))
            self.agent.robot.set_qvel(torch.zeros((b, 9), device=self.device))

            self._table.set_pose(
                sapien.Pose(p=[self._TABLE_CENTER_X, 0,
                               self.TABLE_Z - self._TABLE_HALF[2]])
            )

            white_xyz = torch.zeros((b, 3), device=self.device)
            white_xyz[:, 0] = block_x
            white_xyz[:, 1] = block_y
            white_xyz[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2]
            self.white_block.set_pose(Pose.create_from_pq(p=white_xyz))

            gray_xyz = torch.zeros((b, 3), device=self.device)
            gray_xyz[:, 0] = self._GRAY_BLOCK_XY[0]
            gray_xyz[:, 1] = self._GRAY_BLOCK_XY[1]
            gray_xyz[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2]
            self.gray_block.set_pose(Pose.create_from_pq(p=gray_xyz))

    # ── 成功判断 ──────────────────────────────────────────────────────────────────

    def evaluate(self):
        white_pos = self.white_block.pose.p   # (B, 3)
        white_q   = self.white_block.pose.q   # (B, 4) [w,x,y,z]
        gray_pos  = self.gray_block.pose.p    # (B, 3)

        # XY：白块中心需在灰块正上方（容差 ±1 cm）
        dist_xy = torch.norm(white_pos[:, :2] - gray_pos[:, :2], dim=1)
        xy_ok   = dist_xy < 0.01

        # Z：白块中心 ≈ 灰块中心 + 两倍半边长（容差 ±1 cm）
        expected_z = gray_pos[:, 2] + 2 * self.BLOCK_HALF_SIZE[2]
        z_ok = torch.abs(white_pos[:, 2] - expected_z) < 0.01

        # 白块姿态保持竖直（倾斜 < 20°）
        x = white_q[:, 1]
        y = white_q[:, 2]
        up_z       = 1.0 - 2.0 * (x * x + y * y)
        upright_ok = up_z > np.cos(np.deg2rad(20.0))

        # 白块静止
        lin_vel   = self.white_block.linear_velocity
        ang_vel   = self.white_block.angular_velocity
        vel_ok    = torch.norm(lin_vel, dim=1) < 0.05
        angvel_ok = torch.norm(ang_vel, dim=1) < 0.5

        # 夹爪已张开
        qpos = self.agent.robot.get_qpos()
        gripper_open = (qpos[:, 7] + qpos[:, 8]) > 0.03

        return {
            "success": xy_ok & z_ok & upright_ok & vel_ok & angvel_ok & gripper_open
        }

    # ── 奖励 ──────────────────────────────────────────────────────────────────────

    def compute_dense_reward(self, obs, action, info):
        white_pos = self.white_block.pose.p
        gray_pos  = self.gray_block.pose.p
        target    = gray_pos.clone()
        target[:, 2] = gray_pos[:, 2] + 2 * self.BLOCK_HALF_SIZE[2]
        dist   = torch.norm(white_pos - target, dim=1)
        reward = 1 - torch.tanh(5 * dist)
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0

    # ── RLinf 接口 ────────────────────────────────────────────────────────────────

    def get_language_instruction(self):
        return ["stack the white cube on the gray cube"] * self.num_envs

    def _build_extracted_obs(self, raw_obs: dict) -> dict:
        """构建 extracted_obs，包含正面相机、侧后相机、本体感知和任务描述。

        Returns:
            dict with keys:
                main_images: (num_envs, H, W, 3) uint8, external_cam RGB
                back_images: (num_envs, H, W, 3) uint8, back_cam RGB
                states:      (num_envs, 7) float32, [ee_pos(3), ee_euler(3), gripper(1)]
                task_descriptions: list[str]
        """
        def _get_cam_image(cam_name):
            obs_image = None
            if isinstance(raw_obs, dict):
                sd  = raw_obs.get("sensor_data", {})
                cam = sd.get(cam_name, {})
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
            return obs_image

        main_image = _get_cam_image("external_cam")
        back_image = _get_cam_image("back_cam")

        states_np, _ = build_blockstack_states_from_env(self, apply_bias=False)
        proprioception = torch.from_numpy(states_np).to(main_image.device)

        return {
            "main_images":       main_image,
            "back_images":       back_image,
            "states":            proprioception,
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


# ── 快速验证 + 保存截图/视频 ───────────────────────────────────────────────────────

RENDER_SAVE_DIR = os.path.join(
    os.path.dirname(__file__), "render", "blockstack_multitraj"
)

if __name__ == "__main__":
    import gymnasium as gym
    import imageio

    env = gym.make("BlockStack-v1", obs_mode="rgb", render_mode="rgb_array")
    obs, _ = env.reset()
    base_env = env.unwrapped

    print("Sensor cameras:", list(obs["sensor_data"].keys()))
    print(f"\n[场景] 世界系 = 机器人底座系")
    print(f"  机器人底座 : (0, 0, 0)")
    print(f"  桌面顶部   : z = {base_env.TABLE_Z}")
    print(f"  桌子中心   : x = {base_env._TABLE_CENTER_X}")

    for name, cam_obj in base_env._sensors.items():
        params = cam_obj.get_params()
        K_sim  = params["intrinsic_cv"]
        E_sim  = params["extrinsic_cv"]
        if hasattr(K_sim, "cpu"):
            K_sim = K_sim[0].cpu().numpy()
            E_sim = E_sim[0].cpu().numpy()
        R_E, t_E = E_sim[:3, :3], E_sim[:3, 3]
        p_cam = -R_E.T @ t_E
        print(f"[{name}] fx={K_sim[0,0]:.3f} fy={K_sim[1,1]:.3f} "
              f"cx={K_sim[0,2]:.3f} cy={K_sim[1,2]:.3f}")
        print(f"  相机世界位置={p_cam}")

    def get_frame(obs, cam_name):
        img = obs["sensor_data"][cam_name]["rgb"]
        if len(img.shape) == 4:
            img = img[0]
        img = img.cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return img

    def center_crop(img, th, tw):
        ih, iw = img.shape[:2]
        top  = (ih - th) // 2
        left = (iw - tw) // 2
        return img[top:top + th, left:left + tw]

    def save_cam_outputs(init_frame, save_dir, cam_name, ref_path, step_frames_fn):
        """渲染截图 + 对比图 + demo视频，存入 save_dir。"""
        os.makedirs(save_dir, exist_ok=True)
        frame = init_frame
        imageio.imwrite(os.path.join(save_dir, "BlockStack-v1_screenshot.png"), frame)
        print(f"[{cam_name}] Screenshot saved -> {save_dir}")

        compare_path = os.path.join(save_dir, "BlockStack-v1_compare.png")
        if os.path.exists(ref_path):
            ref_img = imageio.imread(ref_path)
            if ref_img.ndim == 2:
                ref_img = np.stack([ref_img, ref_img, ref_img], axis=-1)
            if ref_img.shape[-1] == 4:
                ref_img = ref_img[..., :3]
            h1, w1 = frame.shape[:2]
            h2, w2 = ref_img.shape[:2]
            h, w   = min(h1, h2), min(w1, w2)
            compare = (0.5 * center_crop(frame, h, w).astype(np.float32)
                     + 0.5 * center_crop(ref_img, h, w).astype(np.float32)).astype(np.uint8)
            imageio.imwrite(compare_path, compare)
            print(f"[{cam_name}] Compare image saved: {compare_path}")
        else:
            print(f"[{cam_name}] 参考图不存在，跳过对比图: {ref_path}")

        frames = step_frames_fn(cam_name)
        imageio.mimsave(os.path.join(save_dir, "BlockStack-v1_demo.mp4"), frames, fps=20)
        print(f"[{cam_name}] Video saved -> {save_dir}")

    # 依次渲染 t=0 和 t=12 两种情况
    for use_ref_12 in [False, True]:
        USE_REF_12 = use_ref_12
        _REF_LABEL = "12" if USE_REF_12 else "0"

        obs, _ = env.reset()

        white_p = base_env.white_block.pose.p[0].cpu().numpy()
        gray_p  = base_env.gray_block.pose.p[0].cpu().numpy()
        print(f"\n[t={_REF_LABEL}s 初始化]")
        print(f"  白色方块 : ({white_p[0]:.4f}, {white_p[1]:.4f}, {white_p[2]:.4f})")
        print(f"  灰色方块 : ({gray_p[0]:.4f},  {gray_p[1]:.4f},  {gray_p[2]:.4f})")

        _CAMS = ["external_cam", "back_cam"]
        video_frames = {cam: [get_frame(obs, cam)] for cam in _CAMS}
        for _ in range(60):
            action = env.action_space.sample()
            o, _, done, trunc, _ = env.step(action)
            for cam in _CAMS:
                video_frames[cam].append(get_frame(o, cam))
            if done or trunc:
                break

        def _step_frames(cam_name):
            return video_frames[cam_name]

        _REF_BASE = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "data_inspector", "BlockStack_ref_screenshot",
            )
        )

        # 正面相机
        front_dir = f"{RENDER_SAVE_DIR}/traj{TRAJ_ID}/{_REF_LABEL}"
        front_ref = f"{_REF_BASE}/front/BlockStack_traj{TRAJ_ID}_t{_REF_LABEL}.png"
        save_cam_outputs(video_frames["external_cam"][0], front_dir, "external_cam",
                         front_ref, _step_frames)

        # 后方相机
        back_dir = f"{RENDER_SAVE_DIR}/back_cam/traj{TRAJ_ID}/{_REF_LABEL}"
        back_ref = f"{_REF_BASE}/back/BlockStack_traj{TRAJ_ID}_t{_REF_LABEL}.png"
        save_cam_outputs(video_frames["back_cam"][0], back_dir, "back_cam",
                         back_ref, _step_frames)

    env.close()
