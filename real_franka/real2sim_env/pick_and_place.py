# real2sim_env/pick_and_place.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors


@register_env("BlockPAP-v1", max_episode_steps=200)
class PickAndPlaceEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # 长方体半尺寸：5cm × 5cm × 8cm 竖放
    BLOCK_HALF_SIZE = [0.025, 0.025, 0.04]
    # 杯垫半径和厚度
    COASTER_RADIUS = 0.06
    COASTER_HALF_THICKNESS = 0.002

    @property
    def _default_sensor_configs(self):
        # 正视角：camera 在 +X 方向，朝 -X 看，桌面和机器人正对
        # robot 在 x=-0.615 面向 +X，桌面 z=0
        pose = sapien_utils.look_at(eye=[0.8, 0.0, 0.4], target=[0.0, 0.0, 0.15])
        return [CameraConfig("external_cam", pose, 640, 480, np.deg2rad(55), 0.01, 10)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.8, 0.0, 0.4], target=[0.0, 0.0, 0.15])
        return CameraConfig("render_camera", pose, 640, 480, np.deg2rad(55), 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # 橙色长方体（竖放）
        self.cube = actors.build_box(
            self.scene,
            half_sizes=self.BLOCK_HALF_SIZE,
            color=[1.0, 0.5, 0.0, 1.0],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.BLOCK_HALF_SIZE[2]]),
        )

        # 绿色圆形杯垫（平放，axis 沿 Z）
        self.target = actors.build_cylinder(
            self.scene,
            radius=self.COASTER_RADIUS,
            half_length=self.COASTER_HALF_THICKNESS,
            color=[0.0, 0.8, 0.2, 1.0],
            name="target_coaster",
            body_type="static",
            initial_pose=sapien.Pose(p=[0.15, 0.1, self.COASTER_HALF_THICKNESS]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 随机化长方体位置（桌面左侧区域）
            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, 0] = torch.rand(b) * 0.15 - 0.10   # x: -0.10~0.05
            cube_xyz[:, 1] = torch.rand(b) * 0.20 - 0.10   # y: -0.10~0.10
            cube_xyz[:, 2] = self.BLOCK_HALF_SIZE[2]         # z: 桌面 + 半高
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

            # 杯垫固定位置（桌面右侧）
            self.target.set_pose(sapien.Pose(p=[0.15, 0.1, self.COASTER_HALF_THICKNESS]))

    def evaluate(self):
        cube_pos = self.cube.pose.p
        target_pos = self.target.pose.p
        dist_xy = torch.norm(cube_pos[:, :2] - target_pos[:, :2], dim=1)
        on_target = (dist_xy < 0.06) & (cube_pos[:, 2] < 0.06)
        return {"success": on_target}

    def compute_dense_reward(self, obs, action, info):
        cube_pos = self.cube.pose.p
        target_pos = self.target.pose.p
        dist = torch.norm(cube_pos - target_pos, dim=1)
        reward = 1 - torch.tanh(5 * dist)
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0


# 工具函数：从真实标定结果导入相机外参
def load_camera_extrinsics_from_calibration(T_cam_to_base: np.ndarray) -> sapien.Pose:
    """T_cam_to_base: 4x4 变换矩阵，从真实标定获得，返回 sapien.Pose"""
    from scipy.spatial.transform import Rotation
    R = T_cam_to_base[:3, :3]
    t = T_cam_to_base[:3, 3]
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])
    return sapien.Pose(p=t, q=quat_wxyz)


if __name__ == "__main__":
    import gymnasium as gym
    import imageio
    import os

    # 1. 定义并自动创建保存路径
    SAVE_DIR = "real_franka/real2sim_env/render"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 初始化环境
    env = gym.make("BlockPAP-v1", obs_mode="rgb", render_mode="rgb_array")
    obs, _ = env.reset()
    print("Sensor cameras:", list(obs["sensor_data"].keys()))

    def get_frame(obs):
        # 获取第一个相机的 RGB 数据并处理 batch 维度
        cam_name = list(obs["sensor_data"].keys())[0]
        img = obs["sensor_data"][cam_name]["rgb"]
        
        if len(img.shape) == 4: # [batch, h, w, c]
            img = img[0]
            
        img = img.cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return img

    # 2. 保存初始截图
    frame = get_frame(obs)
    screenshot_path = os.path.join(SAVE_DIR, "BlockPAP-v1_screenshot.png")
    imageio.imwrite(screenshot_path, frame)
    print(f"✅ Screenshot saved to: {screenshot_path}")

    # 3. 运行随机策略并录制视频
    frames = [frame]
    for _ in range(60):
        action = env.action_space.sample()
        obs, _, done, trunc, _ = env.step(action)
        frames.append(get_frame(obs))
        if done or trunc:
            break
            
    video_path = os.path.join(SAVE_DIR, "BlockPAP-v1_demo.mp4")
    imageio.mimsave(video_path, frames, fps=20)
    print(f"✅ Video saved to: {video_path}")
    
    env.close()