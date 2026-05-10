# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
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
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from sapien.physx import PhysxMaterial


@register_agent()
class PandaHighFriction(Panda):
    """Panda with high-friction finger pads to reliably grasp the block."""
    uid = "panda_high_friction"
    # Override gripper contact material: bump static/dynamic friction from 2.0 → 10.0
    # to match the block friction and prevent finger slipping during lift.
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

# front camera
# 2026-02-15 19:16:38 - __main__ - INFO: 旋转矩阵是:
#  [[ 0.02816316  0.2178868  -0.97556762]
#  [ 0.99959024 -0.00114196  0.0286016 ]
#  [ 0.00511786 -0.97597338 -0.21782968]]
# 2026-02-15 19:16:38 - __main__ - INFO: 平移向量是:
#  [[ 1.10002696]
#  [-0.00701879]
#  [ 0.2589829 ]]
# 2026-02-15 19:16:38 - __main__ - INFO: 四元数是：
#  [ 0.55837595  0.54509737 -0.43449658 -0.44977537]

# back camera
# 2026-01-11 15:39:15 - __main__ - INFO: 旋转矩阵是:
#  [[ 0.49248784 -0.28922532  0.82085592]
#  [-0.868662   -0.10517417  0.4841123 ]
#  [-0.0536847  -0.95146577 -0.30303605]]
# 2026-01-11 15:39:15 - __main__ - INFO: 平移向量是:
#  [[ 0.05115059]
#  [-0.33471246]
#  [ 0.25403179]]
# 2026-01-11 15:39:15 - __main__ - INFO: 四元数是：
#  [ 0.68932903 -0.41993274  0.27823114 -0.52064326]

# Principal Point         : 333.961, 246.486
# Focal Length            : 607.875, 607.719
# Distortion Model        : Inverse Brown Conrady
# Distortion Coefficients : [0,0,0,0,0]
# Show stream intrinsics again?[y/n]: y

USE_REF_12 = False  # 运行时由 __main__ 循环覆盖（False=t0, True=t12）
CAM_T = "og"  # 相机平移向量预设，选择 "og"、"0302" 或 "0303"，需与 RENDER_BASE_DIR 中的子目录一致
TRAJ_ID = "0"  # 轨迹 ID，选择 0、15、25、40 或 45；设为 "random" 启用 MimicGen 生成随机模式
RENDER_BASE_DIR = "real_franka/real2sim_env/render"
TABLE_TEX_ID   = "006"      # 桌面纹理 ID（001-021 / "white" / "black"）；由 CLI --table-tex 覆盖
AMBIENT_PRESET = "default"  # 环境光预设：default / warm / cool / bright / dim
BG_PRESET      = "none"     # 背景墙预设：none / white / gray / dark / blue
CAM_JITTER_RAD = None       # 相机随机偏转（roll, pitch, yaw）单位弧度；None = 不偏转


def build_blockpap_states_from_env(base_env, apply_bias: bool = True):
    """Build 7D BlockPAP proprio state from env internals.

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

    pos = ee_pose_T[:, :3, 3].astype(np.float32)
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
        # Match SFT training convention for simulation data.
        states = states.copy()
        states[:, 2] += 0.1
        states[:, 5] += -np.pi / 4

    return states, gripper_total_width

@register_env("BlockPAP-v1", max_episode_steps=3000)
class PickAndPlaceEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda_high_friction", "panda", "panda_wristcam"]
    agent: PandaHighFriction

    # ── 相机平移向量预设 ────────────────────────────────────────────────────
    _T_PRESETS = {
        "og":   np.array([1.1002696, -0.00701879, 0.2589829]),   # 原始标定
        "0302": np.array([1.1602696, -0.03301879, 0.3189829]),   # 2026-03-02
        "0303": np.array([1.1102696, -0.00701879, 0.2789829]),   # 2026-03-03
    }
    _t = _T_PRESETS["og"]  # 默认使用原始标定

    def __init__(self, *args, robot_uids="panda_high_friction", cam_t: str = "og", **kwargs):
        """
        cam_t: 相机平移向量预设名，可选 'og'（默认）、'0302'、'0303'。
               对应 _T_PRESETS 中的三套标定结果。
        """
        kwargs.setdefault("enable_shadow", True)
        if cam_t not in self._T_PRESETS:
            raise ValueError(f"cam_t='{cam_t}' 不在预设中，可选: {list(self._T_PRESETS)}")
        self._t = self._T_PRESETS[cam_t]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ── 标定参数（RealSense D435, 2026-02-15）─────────────────────────────
    # 正面相机
    _R = np.array([
        [ 0.02816316,  0.21788680, -0.97556762],
        [ 0.99959024, -0.00114196,  0.02860160],
        [ 0.00511786, -0.97597338, -0.21782968],
    ])

    #TODO 
    # 侧后方相机参数还没有验证；
    # 腕部相机参数还没有标定；
    # 暂时先使用一个 front 相机 (external_cam) 视角，符合 RLinf-Co 做法
    # 侧后方相机
    _R2 = np.array([
        [ 0.49248784, -0.28922532,  0.82085592],
        [-0.86866200, -0.10517417,  0.48411230],
        [-0.05368470, -0.95146577, -0.30303605],
    ])
    _t2 = np.array([0.05115059, -0.33471246, 0.25403179])

    _K = np.array([
        [607.875,   0.0,   348.961],
        [  0.0,  607.719, 270.486],
        [  0.0,    0.0,     1.0  ],
    ])
    _K2 = np.array([
        [607.875,   0.0,   359.961],
        [  0.0,  607.719, 242.486],
        [  0.0,    0.0,     1.0  ],
    ])

    # ── 场景参数 ────────────────────────────────────────────────────────────
    # 桌面顶部世界 z = 30 mm（桌底 -5 mm，桌厚 35 mm）
    TABLE_Z = 0.03   # 单位：米
    # 桌面中心 x = 201 mm (底座到桌边实测) + 300 mm (桌子半深)
    _TABLE_CENTER_X = 0.501
    # 桌面尺寸（半X(前后) × 半Y(左右) × 半厚）
    _TABLE_HALF = (0.30, 0.60, 0.0175)   # 60cm深 × 120cm宽 × 3.5cm厚
    # 桌面接触材质：提高摩擦，和物块配对更稳定
    TABLE_STATIC_FRICTION = 1.0
    TABLE_DYNAMIC_FRICTION = 1.0
    TABLE_RESTITUTION = 0.0

    # 杯垫
    # _COASTER_CENTER_X = _TABLE_CENTER_X - 0.01
    # _COASTER_CENTER_Y = 0.059
    COASTER_RADIUS = 0.043
    COASTER_THICKNESS = 0.002

    # 长方体物块
    # _BLOCK_CENTER_X = _TABLE_CENTER_X - 0.076
    # _BLOCK_CENTER_Y = -0.15
    # 长方体半尺寸：4cm × 4cm × 6cm 竖放
    BLOCK_HALF_SIZE = [0.02, 0.02, 0.03]
    # 长方体物理参数：更重 + 更大摩擦 + 无回弹，减少被弹飞/打滑
    BLOCK_DENSITY = 200.0
    BLOCK_STATIC_FRICTION = 0.5
    BLOCK_DYNAMIC_FRICTION = 0.5
    BLOCK_RESTITUTION = 0.0

    # 机器人基座台：0.95m 正方体（上缘 z=0，下缘 z=-0.95）
    BASE_PEDESTAL_SIZE = 0.95

    # ── 各轨迹初始化参数 ──────────────────────────────────────────────────────
    # block_xy:   物块初始位置 (x, y)
    # coaster_xy: 杯垫中心位置 (x, y)
    # qpos_ref_0:  t=0  时刻真实关节角（弧度，7 个关节）
    # qpos_ref_12: t=12 时刻真实关节角（弧度，7 个关节）
    _TRAJ_CONFIGS = {
        "0": {
            "block_xy":    (_TABLE_CENTER_X - 0.076, -0.15),
            "coaster_xy":  (_TABLE_CENTER_X - 0.01, 0.059),
            "qpos_ref_0":  [0.03920901566743851, -0.6867635250091553, -0.009805346839129925, -2.6944820880889893, -0.013050149194896221, 2.007251739501953, 0.8208261132240295],
            "qpos_ref_12": [-0.29462647438049316, 0.26739823818206787, -0.04917740449309349, -2.537435531616211, -0.04237804561853409, 2.7909481525421143, 0.4803294539451599],
        },
        "15": {
            "block_xy":    (_TABLE_CENTER_X + 0.083, -0.15),
            "coaster_xy":  (_TABLE_CENTER_X - 0.01, 0.03),
            "qpos_ref_0":  [0.04231681674718857, -0.46236130595207214, -0.018441837280988693, -2.6430587768554688, -0.002512579783797264, 2.179812431335449, 0.8177617788314819],
            "qpos_ref_12": [-0.15030790865421295, 0.5153473019599915, -0.09237745404243469, -2.0401413440704346, -0.007849287241697311, 2.5943570137023926, 0.589148759841919],
        },
        "25": {
            "block_xy":    (_TABLE_CENTER_X + 0.04, -0.1),
            "coaster_xy":  (_TABLE_CENTER_X - 0.01, 0.02),
            "qpos_ref_0":  [0.017783386632800102, -0.5367491841316223, 0.02235139161348343, -2.7003209590911865, 0.019186807796359062, 2.1525814533233643, 0.8190732598304749],
            "qpos_ref_12": [-0.1456894725561142, 0.3910311460494995, -0.04914682358503342, -2.324347734451294, 0.019173461943864822, 2.7607336044311523, 0.6215252876281738],
        },
        "40": {
            "block_xy":    (_TABLE_CENTER_X - 0.076, -0.03),
            "coaster_xy":  (_TABLE_CENTER_X - 0.01, 0.04),
            "qpos_ref_0":  [0.020625591278076172, -0.5485104322433472, 0.0025699941907078028, -2.711549758911133, 0.013794232159852982, 2.171576976776123, 0.8035998940467834],
            "qpos_ref_12": [0.05512106418609619, 0.10421822965145111, -0.004373287782073021, -2.563603639602661, 0.013808992691338062, 2.692765712738037, 0.974073588848114],
        },
        "45": {
            "block_xy":    (_TABLE_CENTER_X + 0.099, -0.025),
            "coaster_xy":  (_TABLE_CENTER_X - 0.01, 0.04),
            "qpos_ref_0":  [0.044930748641490936, -0.413775235414505, -0.015586936846375465, -2.6283504962921143, -0.015863671898841858, 2.2137093544006348, 0.811760663986206],
            "qpos_ref_12": [0.002559863729402423, 0.3720402419567108, -0.027471385896205902, -2.115872859954834, -0.015866732224822044, 2.4757132530212402, 0.8062286376953125],
        },
    }

    def _make_cam_pose(self, R, t):
        """
        使用严格的 4x4 齐次变换矩阵生成相机位姿，避免 look_at 造成的微小偏差。

        Args:
            R: (3,3) OpenCV 旋转矩阵（相机→世界）
            t: (3,)  OpenCV 平移向量（相机原点在世界坐标系中）
        """
        # 1. 构造 OpenCV 约定下，相机在世界坐标系（机器人基座）中的变换矩阵
        T_cam_to_world = np.eye(4)
        T_cam_to_world[:3, :3] = R
        T_cam_to_world[:3, 3] = t

        # 2. 坐标系对齐：OpenCV (Z前, X右, Y下) -> SAPIEN (X前, Y左, Z上)
        T_cv_to_sapien = np.array([
            [ 0, -1,  0,  0],
            [ 0,  0, -1,  0],
            [ 1,  0,  0,  0],
            [ 0,  0,  0,  1]
        ])
        T_sapien_cam_to_world = T_cam_to_world @ T_cv_to_sapien

        # 3. 提取平移和 SAPIEN 格式的四元数 [w, x, y, z]
        p = T_sapien_cam_to_world[:3, 3]
        q_xyzw = Rotation.from_matrix(T_sapien_cam_to_world[:3, :3]).as_quat()
        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        return sapien.Pose(p=p, q=q_wxyz)

    @property
    def _default_sensor_configs(self):
        # external_cam 应用随机偏转（CAM_JITTER_RAD = [roll, pitch, yaw] 弧度）
        R_front = self._R
        if CAM_JITTER_RAD is not None:
            R_jitter = Rotation.from_euler('xyz', CAM_JITTER_RAD).as_matrix()
            R_front = self._R @ R_jitter
        pose_front = self._make_cam_pose(R_front, self._t)
        pose_side  = self._make_cam_pose(self._R2, self._t2)
        return [
            CameraConfig("external_cam", pose_front, 640, 480,
                         near=0.01, far=10, intrinsic=self._K),
            CameraConfig("back_cam",     pose_side,  640, 480,
                         near=0.01, far=10, intrinsic=self._K2),
        ]

    @property
    def _default_human_render_camera_configs(self):
        R_front = self._R
        if CAM_JITTER_RAD is not None:
            R_jitter = Rotation.from_euler('xyz', CAM_JITTER_RAD).as_matrix()
            R_front = self._R @ R_jitter
        pose = self._make_cam_pose(R_front, self._t)
        return CameraConfig("render_camera", pose, 640, 480,
                            near=0.01, far=100, intrinsic=self._K,
                            shader_pack="default")

    def _load_agent(self, options: dict):
        # 机器人底座固定在世界原点
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    # 环境光预设表（RGB 强度）
    _AMBIENT_TABLE = {
        "default": [0.25, 0.25, 0.28],   # 原始偏冷白
        "warm":    [0.38, 0.30, 0.18],   # 暖黄调
        "cool":    [0.18, 0.22, 0.40],   # 冷蓝调
        "bright":  [0.55, 0.55, 0.58],   # 明亮漫射
        "dim":     [0.10, 0.10, 0.12],   # 昏暗低调
    }

    def _load_lighting(self, options: dict):
        """写实光照：环境光 + 主方向光（含阴影）+ 补光"""
        # 环境光：由 AMBIENT_PRESET 全局变量控制
        ambient = self._AMBIENT_TABLE.get(AMBIENT_PRESET, self._AMBIENT_TABLE["default"])
        self.scene.set_ambient_light(ambient)
        # 主光：右前上方，模拟自然侧光，开启阴影
        self.scene.add_directional_light(
            [0.6, 0.4, -1.0], [1.1, 1.05, 1.0],
            shadow=True, shadow_scale=5, shadow_map_size=4096
        )
        # 补光：左侧柔和蓝调，减少死角
        self.scene.add_directional_light([-1.0, 0.2, -0.5], [0.35, 0.38, 0.45])
        # 顶部点光：模拟天花板灯
        self.scene.add_point_light([0.5, 0.0, 1.2], [1.8, 1.7, 1.6], shadow=False)

    def _load_scene(self, options: dict):
        # 地面以机器人基座（世界原点）为参考：z = -0.95
        build_ground(self.scene, floor_width=10, altitude=-self.BASE_PEDESTAL_SIZE)

        # # 在 x = -3 位置添加一堵灰色墙
        # wall_builder = self.scene.create_actor_builder()
        # wall_half_size = [0.02, 10.0, 5.0]
        # wall_builder.add_box_collision(half_size=wall_half_size)
        # wall_mat = sapien.render.RenderMaterial()
        # wall_mat.base_color = [0.6, 0.6, 0.6, 1.0]
        # wall_mat.roughness = 0.8
        # wall_mat.metallic = 0.0
        # wall_builder.add_box_visual(half_size=wall_half_size, material=wall_mat)
        # wall_builder.set_initial_pose(sapien.Pose(p=[-3.0, 0, -self.BASE_PEDESTAL_SIZE + wall_half_size[2]]))
        # self._back_wall = wall_builder.build_kinematic(name="x_minus3_wall")

        # 机器人基座台：x 负方向保持不变，x 正方向延伸到 +1.2m；上缘 z=0，下缘 z=-0.95
        pedestal_x_min = -self.BASE_PEDESTAL_SIZE / 2
        pedestal_x_max = 1.2
        pedestal_half_x = (pedestal_x_max - pedestal_x_min) / 2
        pedestal_center_x = (pedestal_x_max + pedestal_x_min) / 2
        pedestal_half_y = self.BASE_PEDESTAL_SIZE / 2
        pedestal_half_z = self.BASE_PEDESTAL_SIZE / 2

        pedestal_builder = self.scene.create_actor_builder()
        pedestal_builder.add_box_collision(half_size=[pedestal_half_x, pedestal_half_y, pedestal_half_z])
        pedestal_mat = sapien.render.RenderMaterial()
        pedestal_mat.base_color = [0.2, 0.2, 0.2, 1.0]
        pedestal_mat.roughness = 0.2
        pedestal_mat.metallic = 1.0
        pedestal_mat.specular = 0.9
        pedestal_builder.add_box_visual(
            half_size=[pedestal_half_x, pedestal_half_y, pedestal_half_z],
            material=pedestal_mat,
        )
        pedestal_builder.set_initial_pose(sapien.Pose(p=[pedestal_center_x, 0, -pedestal_half_z]))
        self._base_pedestal = pedestal_builder.build_kinematic(name="robot_base_pedestal")

        # 桌面中心在机器人前方 _TABLE_CENTER_X（木质纹理 + PBR）
        # TABLE_TEX_ID：001-021 = 文件纹理；"white" = 纯白；"black" = 纯黑
        table_builder = self.scene.create_actor_builder()
        table_phys_mat = PhysxMaterial(
            static_friction=self.TABLE_STATIC_FRICTION,
            dynamic_friction=self.TABLE_DYNAMIC_FRICTION,
            restitution=self.TABLE_RESTITUTION,
        )
        table_builder.add_box_collision(half_size=self._TABLE_HALF, material=table_phys_mat)
        table_mat = sapien.render.RenderMaterial()
        if TABLE_TEX_ID == "white":
            table_mat.base_color = [0.95, 0.95, 0.95, 1.0]
            table_mat.roughness  = 0.25
            table_mat.metallic   = 0.0
            table_mat.specular   = 0.6
        elif TABLE_TEX_ID == "black":
            table_mat.base_color = [0.04, 0.04, 0.04, 1.0]
            table_mat.roughness  = 0.20
            table_mat.metallic   = 0.0
            table_mat.specular   = 0.8
        else:
            _WOOD_TEX = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "rlinf", "envs", "maniskill", "assets",
                    "carrot", "more_table", "textures", f"{TABLE_TEX_ID}.png",
                )
            )
            table_mat.base_color_texture = sapien.render.RenderTexture2D(
                filename=_WOOD_TEX, mipmap_levels=4, srgb=True
            )
            table_mat.roughness = 0.55
            table_mat.metallic  = 0.0
            table_mat.specular  = 0.5
        table_builder.add_box_visual(half_size=self._TABLE_HALF, material=table_mat)
        table_builder.set_initial_pose(sapien.Pose(p=[self._TABLE_CENTER_X, 0,
                                                       self.TABLE_Z - self._TABLE_HALF[2]]))
        self._table = table_builder.build_kinematic(name="table")

        # 橙红色长方体（竖放），带高光
        cube_builder = self.scene.create_actor_builder()
        cube_phys_mat = PhysxMaterial(
            static_friction=self.BLOCK_STATIC_FRICTION,
            dynamic_friction=self.BLOCK_DYNAMIC_FRICTION,
            restitution=self.BLOCK_RESTITUTION,
        )
        cube_builder.add_box_collision(
            half_size=self.BLOCK_HALF_SIZE,
            material=cube_phys_mat,
            density=self.BLOCK_DENSITY,
        )
        cube_mat = sapien.render.RenderMaterial()
        cube_mat.base_color = [0.82, 0.22, 0.06, 1.0]  # 橙红色
        cube_mat.roughness  = 0.5
        cube_mat.metallic   = 0.0
        cube_mat.specular   = 0.6
        cube_builder.add_box_visual(half_size=self.BLOCK_HALF_SIZE, material=cube_mat)
        cube_builder.set_initial_pose(sapien.Pose(p=[self._TABLE_CENTER_X, 0,
                                                     self.TABLE_Z + self.BLOCK_HALF_SIZE[2]]))
        self.cube = cube_builder.build(name="cube")

        # 绿色圆形杯垫（平放）
        # build_cylinder 默认沿 x 轴，需要绕 y 轴旋转 90 度使其”平放”在桌面上 (轴向变为 Z)
        self.target = actors.build_cylinder(
            self.scene,
            radius=self.COASTER_RADIUS,
            half_length=self.COASTER_THICKNESS,
            color=[0.0, 0.8, 0.2, 1.0],
            name="target_coaster",
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[self._TABLE_CENTER_X, 0.0, self.TABLE_Z + self.COASTER_THICKNESS],
                q = [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]  # 绕 Y 轴旋转 90 度
            ),
        )

        # 背景墙：仅视觉，BG_PRESET 全局变量控制颜色
        _BG_COLORS = {
            "none":  None,
            "white": [0.95, 0.95, 0.95, 1.0],
            "gray":  [0.55, 0.55, 0.55, 1.0],
            "dark":  [0.12, 0.12, 0.15, 1.0],
            "blue":  [0.18, 0.28, 0.50, 1.0],
        }
        bg_color = _BG_COLORS.get(BG_PRESET)
        if bg_color is not None:
            wall_builder = self.scene.create_actor_builder()
            wall_mat = sapien.render.RenderMaterial()
            wall_mat.base_color = bg_color
            wall_mat.roughness = 0.9
            wall_mat.metallic = 0.0
            # 墙体：x=-0.8，宽 5m，高 3m，厚 4cm
            wall_half = [0.02, 2.5, 1.5]
            wall_builder.add_box_visual(half_size=wall_half, material=wall_mat)
            wall_builder.set_initial_pose(
                sapien.Pose(p=[-0.8, 0.0, 0.3])
            )
            self._bg_wall = wall_builder.build_kinematic(name="bg_wall")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # ── 初始化模式选择 ──────────────────────────────────────────────
            # 优先级：1) options["episode_id"] → 确定性选固定轨迹
            #          2) TRAJ_ID 全局变量为已知轨迹 → Replay/prepare 脚本模式
            #          3) 否则（RL 训练）→ 随机采样
            episode_id = options.get("episode_id", None)
            traj_keys = list(self._TRAJ_CONFIGS.keys())

            if episode_id is not None:
                # ── 确定性模式（来自 ManiskillEnv use_fixed_reset_state_ids）──
                traj_idx = int(episode_id[0].item()) % len(traj_keys)
                traj_cfg = self._TRAJ_CONFIGS[traj_keys[traj_idx]]
                qpos_rad   = traj_cfg["qpos_ref_12"] if USE_REF_12 else traj_cfg["qpos_ref_0"]
                block_x,   block_y   = traj_cfg["block_xy"]
                coaster_x, coaster_y = traj_cfg["coaster_xy"]
            elif TRAJ_ID in self._TRAJ_CONFIGS:
                # ── Replay / prepare 模式：使用固定轨迹配置 ─────────────────
                traj_cfg = self._TRAJ_CONFIGS[TRAJ_ID]
                qpos_rad = traj_cfg["qpos_ref_12"] if USE_REF_12 else traj_cfg["qpos_ref_0"]
                block_x,   block_y   = traj_cfg["block_xy"]
                coaster_x, coaster_y = traj_cfg["coaster_xy"]
            else:
                # ── RL 训练 / MimicGen 生成模式（TRAJ_ID == "random"）────────
                # 每个 env 独立采样，保证同 GPU 内不同 env、不同 GPU、不同 epoch 场景各异
                # 物块：X ∈ [TABLE_CENTER_X - 0.10, TABLE_CENTER_X + 0.10]
                #        Y ∈ [-0.15, -0.03]
                block_x = torch.tensor(
                    self._TABLE_CENTER_X + np.random.uniform(-0.10, 0.10, size=b),
                    device=self.device, dtype=torch.float32,
                )
                block_y = torch.tensor(
                    np.random.uniform(-0.15, -0.03, size=b),
                    device=self.device, dtype=torch.float32,
                )
                # 杯垫：X ∈ [TABLE_CENTER_X - 0.03, TABLE_CENTER_X + 0.01]
                #        Y ∈ [0.03, 0.05]
                coaster_x = torch.tensor(
                    self._TABLE_CENTER_X + np.random.uniform(-0.03, 0.01, size=b),
                    device=self.device, dtype=torch.float32,
                )
                coaster_y = torch.tensor(
                    np.random.uniform(0.03, 0.05, size=b),
                    device=self.device, dtype=torch.float32,
                )

                # 向量化最近轨迹查找：为每个 env 独立找最近 qpos
                traj_vals = list(self._TRAJ_CONFIGS.values())
                traj_keys = list(self._TRAJ_CONFIGS.keys())
                traj_bxy = torch.tensor(
                    [c["block_xy"] for c in traj_vals],
                    device=self.device, dtype=torch.float32,
                )  # (T, 2)
                traj_cxy = torch.tensor(
                    [c["coaster_xy"] for c in traj_vals],
                    device=self.device, dtype=torch.float32,
                )  # (T, 2)
                env_bxy = torch.stack([block_x, block_y], dim=1)    # (b, 2)
                env_cxy = torch.stack([coaster_x, coaster_y], dim=1)  # (b, 2)
                dists = (
                    ((env_bxy[:, None] - traj_bxy[None]) ** 2).sum(-1)
                    + ((env_cxy[:, None] - traj_cxy[None]) ** 2).sum(-1)
                )  # (b, T)
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

                cube_xyz = torch.zeros((b, 3), device=self.device)
                cube_xyz[:, 0] = block_x
                cube_xyz[:, 1] = block_y
                cube_xyz[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2]
                self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

                coaster_xyz = torch.zeros((b, 3), device=self.device)
                coaster_xyz[:, 0] = coaster_x
                coaster_xyz[:, 1] = coaster_y
                coaster_xyz[:, 2] = self.TABLE_Z + self.COASTER_THICKNESS
                coaster_q = torch.tensor(
                    [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
                    device=self.device, dtype=torch.float32,
                ).expand(b, -1)
                self.target.set_pose(Pose.create_from_pq(p=coaster_xyz, q=coaster_q))
                return

            init_qpos = torch.tensor(qpos_rad + [0.04, 0.04], device=self.device)
            self.agent.robot.set_qpos(init_qpos.repeat(b, 1))
            self.agent.robot.set_qvel(torch.zeros((b, 9), device=self.device))

            self._table.set_pose(
                sapien.Pose(p=[self._TABLE_CENTER_X, 0,
                               self.TABLE_Z - self._TABLE_HALF[2]])
            )

            cube_xyz = torch.zeros((b, 3), device=self.device)
            cube_xyz[:, 0] = block_x
            cube_xyz[:, 1] = block_y
            cube_xyz[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2]
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

            coaster_pos = [coaster_x, coaster_y, self.TABLE_Z + self.COASTER_THICKNESS]
            self.target.set_pose(
                sapien.Pose(
                    p=coaster_pos,
                    q=[np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]
                )
            )

    def _is_cube_grasped(self) -> torch.Tensor:
        """检测 cube 是否被夹爪抓住：cube 离桌面静止位置 > 1.5cm 且夹爪足够闭合。"""
        qpos = self.agent.robot.get_qpos()          # (B, 9)
        gripper_width = qpos[:, 7] + qpos[:, 8]    # 两指总宽度，open≈0.08m
        cube_z = self.cube.pose.p[:, 2]
        # cube 离开桌面（静止 z = TABLE_Z + BLOCK_HALF[2]），抬起 > 1.5cm
        cube_lifted = cube_z > self.TABLE_Z + self.BLOCK_HALF_SIZE[2] + 0.015
        # 方块宽 4cm，夹住时两指总宽 < 5cm
        gripper_gripping = gripper_width < 0.05
        return cube_lifted & gripper_gripping

    def evaluate(self):
        cube_pos   = self.cube.pose.p
        cube_quat  = self.cube.pose.q  # [w, x, y, z]
        target_pos = self.target.pose.p
        dist_xy    = torch.norm(cube_pos[:, :2] - target_pos[:, :2], dim=1)
        # XY: block centre must be within coaster radius
        # Z:  block centre should be near coaster top + block half-height (±2 cm tolerance)
        expected_z = target_pos[:, 2] + self.COASTER_THICKNESS + self.BLOCK_HALF_SIZE[2]
        xy_ok = dist_xy < self.COASTER_RADIUS
        z_ok  = torch.abs(cube_pos[:, 2] - expected_z) < 0.02

        # Upright check: local +Z axis should stay close to world +Z.
        # For quaternion [w, x, y, z], the world z-component of local z-axis is:
        # up_z = 1 - 2 * (x^2 + y^2). Require tilt <= 20 deg.
        x = cube_quat[:, 1]
        y = cube_quat[:, 2]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        upright_ok = up_z > np.cos(np.deg2rad(20.0))

        # Block must be stationary: prevents spurious success when the block is
        # momentarily on the coaster but still moving / about to topple.
        lin_vel = self.cube.linear_velocity   # (B, 3)
        ang_vel = self.cube.angular_velocity  # (B, 3)
        vel_ok    = torch.norm(lin_vel, dim=1) < 0.05   # < 5 cm/s
        angvel_ok = torch.norm(ang_vel, dim=1) < 0.5    # < 0.5 rad/s

        # Gripper must be open: robot has released the block before success fires.
        qpos = self.agent.robot.get_qpos()  # (B, 9): joints 7&8 are fingers
        gripper_open = (qpos[:, 7] + qpos[:, 8]) > 0.03  # combined > 3 cm

        # ── Grasp state tracking ────────────────────────────────────────────
        is_grasped = self._is_cube_grasped()
        if not hasattr(self, "consecutive_grasp"):
            self.consecutive_grasp = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
        self.consecutive_grasp[is_grasped] += 1
        self.consecutive_grasp[~is_grasped] = 0

        return {
            "success": xy_ok & z_ok & upright_ok & vel_ok & angvel_ok & gripper_open,
            "is_cube_grasped": is_grasped,
            "consecutive_grasp": self.consecutive_grasp.clone().float(),
        }

    def compute_dense_reward(self, obs, action, info):
        """
        分阶段奖励设计（配合 use_rel_reward=True 使用增量信号）：

          Stage 1 – 靠近物块  (未抓取时)：approach_reward ∈ [0, 0.1]
          Stage 2 – 抓取保持  (已抓取时)：grasp_reward = 1.0 (绝对值，持续保持)
          Stage 3 – 搬运      (已抓取时)：transport_reward ∈ [0, 0.5]，cube 越近 coaster 越高
          Stage 4 – 成功             ：success_reward = 3.0

        各关键时刻的 reward_diff（use_rel_reward 下的实际信号）：
          抓取瞬间：≈ +1.0  (grasp 从 0 跳到 1.0，approach 消失)
          搬运中：  > 0     (transport 随 cube 靠近递增)
          成功瞬间：≈ +1.5  (3.0 - grasp 1.0 - transport≤0.5 > 0 ✓)
          掉落时：  大负值  (grasp+transport 突然归零)
        """
        cube_pos   = self.cube.pose.p     # (B, 3)
        target_pos = self.target.pose.p   # (B, 3)
        tcp_pos    = self.agent.tcp.pose.p  # (B, 3)

        is_grasped = info.get(
            "is_cube_grasped",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        is_grasped_f = is_grasped.float()

        dist_tcp_to_cube    = torch.norm(tcp_pos - cube_pos, dim=1)
        dist_cube_to_target = torch.norm(cube_pos - target_pos, dim=1)

        # Stage 1: 靠近物块（未抓取阶段提供引导信号）
        approach_reward = (1.0 - is_grasped_f) * (
            1.0 - torch.tanh(5.0 * dist_tcp_to_cube)
        ) * 0.1

        # Stage 2: 持续抓取（绝对值保持，使 reward_diff 在抓取瞬间产生正 spike）
        grasp_reward = is_grasped_f * 1.0

        # Stage 3: 搬运（抓取状态下 cube 越靠近 coaster reward 越高）
        transport_reward = is_grasped_f * (
            1.0 - torch.tanh(5.0 * dist_cube_to_target)
        ) * 0.5

        # Stage 4: 成功（3.0 > grasp 1.0 + transport≤0.5，保证成功时 diff > 0）
        success_reward = info["success"].float() * 3.0

        return approach_reward + grasp_reward + transport_reward + success_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    # ── RLinf 接口：供 ManiskillEnv (wrap_obs_mode=raw) 使用 ────────────────

    def get_language_instruction(self):
        return ["pick up the block and place it on the coaster"] * self.num_envs

    def _build_extracted_obs(self, raw_obs: dict) -> dict:
        """构建 extracted_obs，格式与官方 panda_put_on_in_scene 保持一致。

        Returns:
            dict with keys:
                main_images:       (num_envs, H, W, 3) uint8, external_cam RGB
                extra_view_images: (num_envs, 1, H, W, 3) uint8, back_cam RGB (if available)
                states:            (num_envs, 7) float32, [ee_pos(3), ee_euler(3), gripper(1)]
                task_descriptions: list[str]
        """
        obs_image = None
        back_image = None
        if isinstance(raw_obs, dict):
            sensor_data = raw_obs.get("sensor_data", None)
            if isinstance(sensor_data, dict):
                cam_data = sensor_data.get("external_cam", None)
                if isinstance(cam_data, dict) and "rgb" in cam_data:
                    obs_image = cam_data["rgb"].to(torch.uint8)
                back_data = sensor_data.get("back_cam", None)
                if isinstance(back_data, dict) and "rgb" in back_data:
                    back_image = back_data["rgb"].to(torch.uint8).unsqueeze(1)  # [B, 1, H, W, 3]

        if obs_image is None:
            # In state obs_mode, raw_obs may not contain sensor_data.
            # Fall back to offscreen render to keep MimicGen extracted_obs compatible.
            frame = self.render()
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255.0).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            obs_image = torch.from_numpy(frame).to(torch.uint8).unsqueeze(0)

        states_np, _ = build_blockpap_states_from_env(self, apply_bias=True)
        proprioception = torch.from_numpy(states_np).to(obs_image.device)

        return {
            "main_images": obs_image,
            "extra_view_images": back_image,
            "states": proprioception,
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


if __name__ == "__main__":
    import argparse
    import gymnasium as gym
    import imageio
    import os

    # ── CLI 参数解析 ────────────────────────────────────────────────────────
    # 必须在 gym.make 之前设置全局变量，因为 _load_scene/_load_lighting 在初始化时读取它们
    _parser = argparse.ArgumentParser(description="BlockPAP-v1 场景预览")
    _parser.add_argument("--table-tex", default=TABLE_TEX_ID,
                         help="桌面纹理 ID（001-021），默认 006")
    _parser.add_argument("--ambient", default=AMBIENT_PRESET,
                         help="环境光预设：default/warm/cool/bright/dim")
    _parser.add_argument("--bg", default=BG_PRESET,
                         help="背景墙预设：none/white/gray/dark/blue")
    _parser.add_argument("--out-dir", default=None,
                         help="快速预览模式：只保存截图到此目录，跳过视频生成")
    _parser.add_argument("--cam-jitter", action="store_true",
                         help="给 external_cam 施加 ±2° 随机 roll/pitch/yaw 偏转")
    _args = _parser.parse_args()

    # 覆盖模块级全局变量（模块级赋值无需 global 声明）
    TABLE_TEX_ID   = _args.table_tex
    AMBIENT_PRESET = _args.ambient
    BG_PRESET      = _args.bg
    if _args.cam_jitter:
        CAM_JITTER_RAD = np.random.uniform(-np.deg2rad(2), np.deg2rad(2), 3)
        print(f"[cam jitter] roll={np.rad2deg(CAM_JITTER_RAD[0]):.2f}°  "
              f"pitch={np.rad2deg(CAM_JITTER_RAD[1]):.2f}°  "
              f"yaw={np.rad2deg(CAM_JITTER_RAD[2]):.2f}°")

    env = gym.make("BlockPAP-v1", obs_mode="rgb", render_mode="rgb_array", cam_t=CAM_T)
    obs, _ = env.reset()
    base_env = env.unwrapped

    print("Sensor cameras:", list(obs["sensor_data"].keys()))
    print(f"\n[场景] 世界系 = 机器人底座系")
    print(f"  机器人底座 : (0, 0, 0)")
    print(f"  相机位置   : {base_env._t}  ← 直接来自标定 _t")
    print(f"  桌面顶部   : z = {base_env.TABLE_Z}")
    print(f"  桌子中心   : x = {base_env._TABLE_CENTER_X}")
    print(f"  相机离桌面 : {base_env._t[2] - base_env.TABLE_Z:.6f} m")

    for name, cam_obj in base_env._sensors.items():
        params  = cam_obj.get_params()
        K_sim   = params["intrinsic_cv"]
        E_sim   = params["extrinsic_cv"]
        if hasattr(K_sim, "cpu"):
            K_sim = K_sim[0].cpu().numpy()
            E_sim = E_sim[0].cpu().numpy()
        R_E, t_E = E_sim[:3, :3], E_sim[:3, 3]
        p_cam = -R_E.T @ t_E
        print(f"[{name}] fx={K_sim[0,0]:.6f} fy={K_sim[1,1]:.6f} "
              f"cx={K_sim[0,2]:.6f} cy={K_sim[1,2]:.6f}")
        print(f" 相机世界位置={p_cam}")

    def get_frame(obs, cam_name):
        img = obs["sensor_data"][cam_name]["rgb"]
        if len(img.shape) == 4:
            img = img[0]
        img = img.cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return img

    # ── 快速预览模式（--out-dir 指定时）────────────────────────────────────
    if _args.out_dir is not None:
        out_dir = _args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        frame = get_frame(obs, "external_cam")
        imageio.imwrite(os.path.join(out_dir, "screenshot.png"), frame)
        print(f"✅ Screenshot → {out_dir}/screenshot.png  "
              f"[tex={TABLE_TEX_ID} amb={AMBIENT_PRESET} bg={BG_PRESET}]")
        env.close()
        import sys; sys.exit(0)

    def center_crop(img, th, tw):
        ih, iw = img.shape[:2]
        top = (ih - th) // 2
        left = (iw - tw) // 2
        return img[top:top + th, left:left + tw]

    def save_cam_outputs(init_frame, save_dir, cam_name, ref_path, step_frames_fn):
        """渲染截图 + 对比图 + demo视频，存入 save_dir。
        init_frame: 已经拷贝为 numpy 的初始帧（避免 GPU tensor 视图被后续 step 覆盖）
        """
        os.makedirs(save_dir, exist_ok=True)
        frame = init_frame
        imageio.imwrite(os.path.join(save_dir, "BlockPAP-v1_screenshot.png"), frame)
        print(f"✅ [{cam_name}] Screenshot saved → {save_dir}")

        compare_path = os.path.join(save_dir, "BlockPAP-v1_compare.png")
        if os.path.exists(ref_path):
            ref_img = imageio.imread(ref_path)
            if ref_img.ndim == 2:
                ref_img = np.stack([ref_img, ref_img, ref_img], axis=-1)
            if ref_img.shape[-1] == 4:
                ref_img = ref_img[..., :3]
            h1, w1 = frame.shape[:2]
            h2, w2 = ref_img.shape[:2]
            h, w = min(h1, h2), min(w1, w2)
            compare = (0.5 * center_crop(frame, h, w).astype(np.float32)
                     + 0.5 * center_crop(ref_img, h, w).astype(np.float32)).astype(np.uint8)
            imageio.imwrite(compare_path, compare)
            print(f"✅ [{cam_name}] Compare image saved: {compare_path}")
        else:
            print(f"⚠️ [{cam_name}] 参考图不存在，跳过对比图: {ref_path}")

        frames = step_frames_fn(cam_name)
        imageio.mimsave(os.path.join(save_dir, "BlockPAP-v1_demo.mp4"), frames, fps=20)
        print(f"✅ [{cam_name}] Video saved → {save_dir}")

    # 依次渲染 t=0 和 t=12 两种情况
    for use_ref_12 in [False, True]:
        USE_REF_12 = use_ref_12   # 更新模块级变量，env.reset() 会读取新值
        _REF_LABEL = "12" if USE_REF_12 else "0"

        obs, _ = env.reset()

        cube_p    = base_env.cube.pose.p[0].cpu().numpy()
        coaster_p = base_env.target.pose.p[0].cpu().numpy()
        print(f"\n[t={_REF_LABEL}s 初始化]")
        print(f"  物块位置 : ({cube_p[0]:.4f}, {cube_p[1]:.4f}, {cube_p[2]:.4f})")
        print(f"  杯垫位置 : ({coaster_p[0]:.4f}, {coaster_p[1]:.4f}, {coaster_p[2]:.4f})")

        # GPU tensor 是原地更新的视图，每步立即转 numpy 拷贝，两个相机共用同一段轨迹
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
                "..", "data_inspector", "BlockPAP_ref_screenshot",
            )
        )

        # ── 正面相机 → render/traj... ────────────────────────────────────
        front_dir = f"{RENDER_BASE_DIR}/BlockPAP/traj{TRAJ_ID}/{CAM_T}/{_REF_LABEL}"
        front_ref = f"{_REF_BASE}/front/BlockPAP_traj{TRAJ_ID}_t{_REF_LABEL}.png"
        # video_frames[cam][0] 是 reset 后立即拷贝的初始帧，用于截图和对比图
        save_cam_outputs(video_frames["external_cam"][0], front_dir, "external_cam", front_ref, _step_frames)

        # ── 后方相机 → render/back_cam/traj... ────────────────────────
        back_dir  = f"{RENDER_BASE_DIR}/BlockPAP/back_cam/traj{TRAJ_ID}/{CAM_T}/{_REF_LABEL}"
        back_ref  = f"{_REF_BASE}/back/BlockPAP_traj{TRAJ_ID}_t{_REF_LABEL}.png"
        save_cam_outputs(video_frames["back_cam"][0], back_dir, "back_cam", back_ref, _step_frames)

    env.close()