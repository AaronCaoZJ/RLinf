import argparse
import os

import gymnasium as gym
import imageio
import numpy as np
import sapien
from scipy.spatial.transform import Rotation

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env

from pick_and_place import PickAndPlaceEnv


def make_look_at_pose(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    x_axis = target - eye
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(up, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    q_xyzw = Rotation.from_matrix(rotation).as_quat()
    q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    return sapien.Pose(p=eye.tolist(), q=q_wxyz)


@register_env("BlockPAP-MultiView-v1", max_episode_steps=200)
class PickAndPlaceMultiViewEnv(PickAndPlaceEnv):
    DISTANCE_SCALE = 2.2

    @property
    def _default_sensor_configs(self):
        target = np.array([self._TABLE_CENTER_X, 0.0, self.TABLE_Z + 0.04], dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        camera_poses = {
            "front": [1.30, 0.00, 0.42],
            "left": [0.85, 0.95, 0.55],
            "right": [0.85, -0.95, 0.55],
            "top": [0.52, 0.00, 1.45],
        }

        scaled_poses = {}
        for name, eye in camera_poses.items():
            eye = np.asarray(eye, dtype=np.float64)
            scaled_eye = target + (eye - target) * float(self.DISTANCE_SCALE)
            scaled_poses[name] = scaled_eye.tolist()

        origin_side_distance = 0.9 * float(self.DISTANCE_SCALE)
        scaled_poses["origin_left"] = [origin[0], origin[1] + origin_side_distance, 0.35]
        scaled_poses["origin_right"] = [origin[0], origin[1] - origin_side_distance, 0.35]

        configs = []
        for name, eye in scaled_poses.items():
            look_target = origin.tolist() if name in {"origin_left", "origin_right"} else target
            configs.append(
                CameraConfig(
                    uid=f"cam_{name}",
                    pose=make_look_at_pose(eye=eye, target=look_target),
                    width=640,
                    height=480,
                    near=0.01,
                    far=20,
                    intrinsic=self._K,
                )
            )
        return configs


def tensor_img_to_uint8(image_tensor):
    image = image_tensor
    if len(image.shape) == 4:
        image = image[0]
    image = image.cpu().numpy()
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="real_franka/real2sim_env/render/BlockPAP_multiview")
    parser.add_argument("--env-id", type=str, default="BlockPAP-MultiView-v1")
    parser.add_argument("--distance-scale", type=float, default=1.35, help="视角距离缩放 (>1 更远)")
    args = parser.parse_args()

    PickAndPlaceMultiViewEnv.DISTANCE_SCALE = max(0.2, args.distance_scale)

    os.makedirs(args.out_dir, exist_ok=True)

    env = gym.make(args.env_id, obs_mode="rgb", render_mode="rgb_array")
    obs, _ = env.reset()

    camera_names = list(obs["sensor_data"].keys())
    print("Available cameras:", camera_names)

    rendered = {}
    for cam_name in camera_names:
        frame = tensor_img_to_uint8(obs["sensor_data"][cam_name]["rgb"])
        rendered[cam_name] = frame
        save_path = os.path.join(args.out_dir, f"{cam_name}.png")
        imageio.imwrite(save_path, frame)
        print(f"Saved: {save_path}")

    ordered = [
        name
        for name in [
            "cam_front",
            "cam_left",
            "cam_right",
            "cam_top",
            "cam_origin_left",
            "cam_origin_right",
        ]
        if name in rendered
    ]
    if len(ordered) == 6:
        top_row = np.concatenate([rendered["cam_front"], rendered["cam_left"], rendered["cam_right"]], axis=1)
        bottom_row = np.concatenate(
            [rendered["cam_top"], rendered["cam_origin_left"], rendered["cam_origin_right"]], axis=1
        )
        grid = np.concatenate([top_row, bottom_row], axis=0)
        grid_path = os.path.join(args.out_dir, "multiview_grid.png")
        imageio.imwrite(grid_path, grid)
        print(f"Saved: {grid_path}")

    env.close()


if __name__ == "__main__":
    main()
