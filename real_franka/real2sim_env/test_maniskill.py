import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import gymnasium as gym
import imageio
import mani_skill.envs  # 注册所有内置环境

# 1. 定义并创建保存路径
SAVE_DIR = "real_franka/real2sim_env/render"
os.makedirs(SAVE_DIR, exist_ok=True)

env_id = sys.argv[1] if len(sys.argv) > 1 else "PickCube-v1"

# 如果没指定环境，打印示例
if len(sys.argv) == 1:
    all_envs = [e for e in gym.envs.registry.keys() if "mani" in e.lower() or "pick" in e.lower() or "stack" in e.lower()]
    print(f"Available ManiSkill envs (sample): {all_envs[:10]}")

# 初始化环境
env = gym.make(env_id, obs_mode="rgb", render_mode="rgb_array")
obs, _ = env.reset()

print(f"Env: {env_id}")
print(f"Action space: {env.action_space}")
print(f"Sensor cameras: {list(obs['sensor_data'].keys())}")

def get_frame(obs):
    # 获取第一个相机的 RGB 数据
    cam_name = list(obs["sensor_data"].keys())[0]
    img = obs["sensor_data"][cam_name]["rgb"]
    
    # 处理不同维度的 tensor 输出
    if len(img.shape) == 4:  # [batch, h, w, c]
        img = img[0]
    
    img = img.cpu().numpy()
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img

# 2. 保存截图
frame = get_frame(obs)
screenshot_path = os.path.join(SAVE_DIR, f"{env_id}_screenshot.png")
imageio.imwrite(screenshot_path, frame)
print(f"✅ Screenshot saved to: {screenshot_path} ({frame.shape})")

# 3. 运行并保存视频
frames = [frame]
for _ in range(60):
    action = env.action_space.sample()
    obs, _, done, trunc, _ = env.step(action)
    frames.append(get_frame(obs))
    if done or trunc:
        break

video_path = os.path.join(SAVE_DIR, f"{env_id}_demo.mp4")
imageio.mimsave(video_path, frames, fps=20)
print(f"✅ Video saved to: {video_path} ({len(frames)} frames)")

env.close()