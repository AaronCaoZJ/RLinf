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

# Principal Point         : 333.961, 246.486
# Focal Length            : 607.875, 607.719
# Distortion Model        : Inverse Brown Conrady
# Distortion Coefficients : [0,0,0,0,0]
# Show stream intrinsics again?[y/n]: y

# ── 对比模式开关 ────────────────────────────────────────────────────────────
# True  → 使用 12 秒参考帧（ref_12），与 BlockPAP_ref_12.png 叠加，保存到 render/12
# False → 使用  0 秒参考帧（ref_0），与 BlockPAP_ref_0.png  叠加，保存到 render/0
USE_REF_12 = True
# USE_REF_12 = False
CAM_T = "og"  # 相机平移向量预设，选择 "og"、"0302" 或 "0303"，需与 RENDER_BASE_DIR 中的子目录一致
RENDER_BASE_DIR = "real_franka/real2sim_env/render"

@register_env("BlockPAP-v1", max_episode_steps=600)
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
    _R = np.array([
        [ 0.02816316,  0.21788680, -0.97556762],
        [ 0.99959024, -0.00114196,  0.02860160],
        [ 0.00511786, -0.97597338, -0.21782968],
    ])

    _K = np.array([
        [607.875,   0.0,   333.961],
        [  0.0,  607.719, 246.486],
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
    TABLE_STATIC_FRICTION = 50.0
    TABLE_DYNAMIC_FRICTION = 50.0
    TABLE_RESTITUTION = 0.0

    # 杯垫
    _COASTER_CENTER_X = _TABLE_CENTER_X - 0.01
    _COASTER_CENTER_Y = 0.059
    COASTER_RADIUS = 0.043
    COASTER_HALF_THICKNESS = 0.002

    # 长方体物块
    _BLOCK_CENTER_X = _TABLE_CENTER_X - 0.076
    _BLOCK_CENTER_Y = -0.15
    # 长方体半尺寸：4cm × 4cm × 6cm 竖放
    BLOCK_HALF_SIZE = [0.02, 0.02, 0.03]
    # 长方体物理参数：更重 + 更大摩擦 + 无回弹，减少被弹飞/打滑
    BLOCK_DENSITY = 200.0
    BLOCK_STATIC_FRICTION = 100.0
    BLOCK_DYNAMIC_FRICTION = 100.0
    BLOCK_RESTITUTION = 0.0

    # 机器人基座台：0.95m 正方体（上缘 z=0，下缘 z=-0.95）
    BASE_PEDESTAL_SIZE = 0.95

    def _make_cam_pose(self):
        """
        使用严格的 4x4 齐次变换矩阵生成相机位姿，避免 look_at 造成的微小偏差
        """
        # 1. 构造 OpenCV 约定下，相机在世界坐标系（机器人基座）中的变换矩阵
        T_cam_to_world = np.eye(4)
        T_cam_to_world[:3, :3] = self._R
        T_cam_to_world[:3, 3] = self._t

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
        pose = self._make_cam_pose()
        return [CameraConfig("external_cam", pose, 640, 480,
                             near=0.01, far=10, intrinsic=self._K)]

    @property
    def _default_human_render_camera_configs(self):
        pose = self._make_cam_pose()
        return CameraConfig("render_camera", pose, 640, 480,
                            near=0.01, far=100, intrinsic=self._K,
                            shader_pack="default")

    def _load_agent(self, options: dict):
        # 机器人底座固定在世界原点
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_lighting(self, options: dict):
        """写实光照：环境光 + 主方向光（含阴影）+ 补光"""
        # 降低环境光，让阴影更立体
        self.scene.set_ambient_light([0.25, 0.25, 0.28])
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
        _WOOD_TEX = "/workspace1/zhijun/RLinf/rlinf/envs/maniskill/assets/carrot/more_table/textures/006.png"
        table_builder = self.scene.create_actor_builder()
        table_phys_mat = PhysxMaterial(
            static_friction=self.TABLE_STATIC_FRICTION,
            dynamic_friction=self.TABLE_DYNAMIC_FRICTION,
            restitution=self.TABLE_RESTITUTION,
        )
        table_builder.add_box_collision(half_size=self._TABLE_HALF, material=table_phys_mat)
        table_mat = sapien.render.RenderMaterial()
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
        # build_cylinder 默认沿 x 轴，需要绕 y 轴旋转 90 度使其“平放”在桌面上 (轴向变为 Z)
        self.target = actors.build_cylinder(
            self.scene,
            radius=self.COASTER_RADIUS,
            half_length=self.COASTER_HALF_THICKNESS,
            color=[0.0, 0.8, 0.2, 1.0],
            name="target_coaster",
            body_type="static",
            initial_pose=sapien.Pose(
                p=[self._TABLE_CENTER_X, 0.0, self.TABLE_Z + self.COASTER_HALF_THICKNESS],
                q = [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]  # 绕 Y 轴旋转 90 度
            ),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            
            # 真实 Panda 关节角（度）→ 弧度，最后两个为夹爪开合
            # qpos_deg = [0.2, -21.6, -0.9, -149.1, 0.0, 125.2, 47.3]
            # qpos_rad = np.deg2rad(qpos_deg).tolist()
            _qpos_ref_0  = [0.03920901566743851, -0.6867635250091553, -0.009805346839129925, -2.6944820880889893, -0.013050149194896221, 2.007251739501953, 0.8208261132240295]
            _qpos_ref_12 = [-0.29462647438049316, 0.26739823818206787, -0.04917740449309349, -2.537435531616211, -0.04237804561853409, 2.7909481525421143, 0.4803294539451599]
            qpos_rad = _qpos_ref_12 if USE_REF_12 else _qpos_ref_0
            init_qpos = torch.tensor(qpos_rad + [0.04, 0.04], device=self.device)
            # init_qpos = torch.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785, 0.04, 0.04], device=self.device)
            self.agent.robot.set_qpos(init_qpos.repeat(b, 1))
            self.agent.robot.set_qvel(torch.zeros((b, 9), device=self.device))

            self._table.set_pose(
                sapien.Pose(p=[self._TABLE_CENTER_X, 0,
                               self.TABLE_Z - self._TABLE_HALF[2]])
            )
            
            # 暂时注释：随机方块初始位置逻辑
            # cx = self._TABLE_CENTER_X
            # cy = -0.12
            # d = 0.06
            # candidate_points = torch.tensor([
            #     [cx, cy],
            #     [cx + d, cy + d],
            #     [cx + d, cy - d],
            #     [cx - d, cy + d],
            #     [cx - d, cy - d]
            # ], device=self.device)
            # random_indices = torch.randint(0, 5, (b,), device=self.device)
            # selected_xy = candidate_points[random_indices]
            # cube_xyz = torch.zeros((b, 3), device=self.device)
            # cube_xyz[:, :2] = selected_xy
            # cube_xyz[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2]

            # 固定方块初始位置：
            cube_xyz = torch.zeros((b, 3), device=self.device)
            cube_xyz[:, 0] = self._BLOCK_CENTER_X
            cube_xyz[:, 1] = self._BLOCK_CENTER_Y
            cube_xyz[:, 2] = self.TABLE_Z + self.BLOCK_HALF_SIZE[2] # Z轴高度保持在桌面上
            
            
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

            coaster_pos = [self._COASTER_CENTER_X, self._COASTER_CENTER_Y, self.TABLE_Z + self.COASTER_HALF_THICKNESS]
            self.target.set_pose(
                sapien.Pose(
                    p=coaster_pos,
                    q = [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]
                )
            )

    def evaluate(self):
        cube_pos   = self.cube.pose.p
        target_pos = self.target.pose.p
        dist_xy    = torch.norm(cube_pos[:, :2] - target_pos[:, :2], dim=1)
        # 允许 Z 轴存在一定误差
        on_target  = (dist_xy < 0.06) & (cube_pos[:, 2] < self.TABLE_Z + 0.06)
        return {"success": on_target}

    def compute_dense_reward(self, obs, action, info):
        cube_pos   = self.cube.pose.p
        target_pos = self.target.pose.p
        dist       = torch.norm(cube_pos - target_pos, dim=1)
        reward     = 1 - torch.tanh(5 * dist)
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0


if __name__ == "__main__":
    import gymnasium as gym
    import imageio
    import os

    _REF_LABEL = "12" if USE_REF_12 else "0"
    SAVE_DIR = f"{RENDER_BASE_DIR}/{CAM_T}/{_REF_LABEL}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    env = gym.make("BlockPAP-v1", obs_mode="rgb", render_mode="rgb_array", cam_t=CAM_T)
    obs, _ = env.reset()
    print("Sensor cameras:", list(obs["sensor_data"].keys()))

    base_env = env.unwrapped

    # 打印 reset 后最终的初始化位置（只打印一次）
    cube_p = base_env.cube.pose.p[0].cpu().numpy()
    coaster_p = base_env.target.pose.p[0].cpu().numpy()
    print(f"\n[初始化]")
    print(f"  物块位置 : ({cube_p[0]:.4f}, {cube_p[1]:.4f}, {cube_p[2]:.4f})")
    print(f"  杯垫位置 : ({coaster_p[0]:.4f}, {coaster_p[1]:.4f}, {coaster_p[2]:.4f})")

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

    def get_frame(obs):
        cam_name = list(obs["sensor_data"].keys())[0]
        img = obs["sensor_data"][cam_name]["rgb"]
        if len(img.shape) == 4:
            img = img[0]
        img = img.cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return img

    frame = get_frame(obs)
    imageio.imwrite(os.path.join(SAVE_DIR, "BlockPAP-v1_screenshot.png"), frame)
    print(f"\n✅ Screenshot saved")

    # 叠加对比图：当前渲染图(50%) + 参考图(50%)
    ref_path = f"/workspace1/zhijun/RLinf/real_franka/data_inspector/BlockPAP_ref_{_REF_LABEL}.png"
    compare_path = os.path.join(SAVE_DIR, "BlockPAP-v1_compare.png")
    if os.path.exists(ref_path):
        ref_img = imageio.imread(ref_path)

        # 统一通道为 RGB
        if ref_img.ndim == 2:
            ref_img = np.stack([ref_img, ref_img, ref_img], axis=-1)
        if ref_img.shape[-1] == 4:
            ref_img = ref_img[..., :3]

        # 尺寸不一致时做中心裁剪到公共尺寸
        h1, w1 = frame.shape[:2]
        h2, w2 = ref_img.shape[:2]
        h = min(h1, h2)
        w = min(w1, w2)

        def center_crop(img, th, tw):
            ih, iw = img.shape[:2]
            top = (ih - th) // 2
            left = (iw - tw) // 2
            return img[top:top + th, left:left + tw]

        frame_aligned = center_crop(frame, h, w)
        ref_aligned = center_crop(ref_img, h, w)

        compare = (0.5 * frame_aligned.astype(np.float32) + 0.5 * ref_aligned.astype(np.float32)).astype(np.uint8)
        imageio.imwrite(compare_path, compare)
        print(f"✅ Compare image saved: {compare_path}")
    else:
        print(f"⚠️ 参考图不存在，跳过对比图生成: {ref_path}")

    frames = [frame]
    for _ in range(60):
        action = env.action_space.sample()
        obs, _, done, trunc, _ = env.step(action)
        frames.append(get_frame(obs))
        if done or trunc:
            break

    imageio.mimsave(os.path.join(SAVE_DIR, "BlockPAP-v1_demo.mp4"), frames, fps=20)
    print(f"✅ Video saved")
    env.close()