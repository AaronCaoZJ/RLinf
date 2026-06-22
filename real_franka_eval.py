# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi0.5 real-robot evaluation on a Franka arm (BlockPAP task, PyTorch inference).

State / action convention (must match training in mg_generate_blockpap_data.py):
  state  (7,): [tcp_x, tcp_y, tcp_z, euler_x, euler_y, euler_z, gripper_total_width]
               - TCP pose in robot BASE (world) frame
               - Euler XYZ extrinsic (scipy convention)
               - gripper_total_width in metres [0, 0.08]

  action (7,): [Δx, Δy, Δz, Δeuler_x, Δeuler_y, Δeuler_z, gripper_norm]
               - delta EE pose in world frame
               - gripper_norm in [0, 1]: 0 = fully closed, 1 = fully open

Usage example:
    cd /workspace1/zhijun/RLinf
    python toolkits/eval_scripts_openpi/blockpap_real_eval.py \\
        --config_name pi05_blockpap_mix \\
        --pretrained_path logs/20260311-12:27:57/test_data_alpha_ratio/checkpoints/global_step_4000/actor/model_state_dict/full_weights.pt \\
        --nuc_ip 192.168.1.143 \\
        --external_camera_serial 123456789 \\
        --num_episodes 5 --action_chunk 8 --num_steps 5
"""

import collections
import datetime
import os
import pathlib
import queue
import select
import sys
import termios
import threading
import time
import tty

import cv2
import imageio
import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Path setup ────────────────────────────────────────────────────────────────
_OPENPI_ROOT = pathlib.Path(__file__).resolve().parent.parent / "openpi"
# franka_interface.py lives here (standalone, not a package)
sys.path.insert(0, str(_OPENPI_ROOT / "examples" / "lab"))
# openpi.lab_utils lives under src/
sys.path.insert(0, str(_OPENPI_ROOT / "src"))

from scipy.spatial.transform import Rotation

from franka_interface import FrankaInterface, MockRobot
import openpi.lab_utils.camera_utils as camera_utils

from toolkits.eval_scripts_openpi import setup_logger
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT

# ── Constants ─────────────────────────────────────────────────────────────────

TASK_DESCRIPTION = "pick up the block and place it on the coaster"
# TASK_DESCRIPTION = "stack the gray cube on the white cube"

# Safety limits for delta actions
MAX_POSITION_DELTA = 0.05   # 5 cm per step
MAX_ROTATION_DELTA = 0.2    # ~11 degrees per step

# Gripper geometry
GRIPPER_MAX_WIDTH = 0.08    # total width (both fingers), metres
GRIPPER_CLOSE_THRESHOLD = 0.04  # total width below this → treat as closed

# 曝光在相机启动后同步设置，按 serial number 查找
# D405 (218622): Stereo Module，单位 1μs
# D435 (317222, 336222): RGB Camera，单位 100μs
PER_CAMERA_EXPOSURE = {
    '218622273043': {'exposure': 2200, 'gain': 64},
    '317222075319': {'exposure': 50,   'gain': 64},
    '336222073740': {'exposure': 50,   'gain': 64},
    '327122079691': {'exposure': 60,   'gain': 64},
}


# ── Attention map capture & visualization ─────────────────────────────────────

class AttentionCapture:
    """Hook-based attention weight capture for Pi0.5 inference.

    Registers a forward hook on the custom ``eager_attention_forward`` call
    inside ``PaliGemmaWithExpertModel.compute_layer_complete``.  Because the
    custom forward concatenates QKV from both PaliGemma and Gemma-Expert,
    the captured attention covers the full cross-modal sequence (image tokens
    + language tokens + action tokens).

    Usage::

        cap = AttentionCapture(policy._model, layer_indices=[-1])
        cap.enable()
        policy.infer(obs)
        attn_maps = cap.get()   # list of (batch, heads, seq, seq)
        cap.disable()
    """

    def __init__(self, model, layer_indices=None):
        self._model = model
        # Which transformer layers to capture (default: last layer only).
        num_layers = len(model.paligemma_with_expert.paligemma.language_model.layers)
        if layer_indices is None:
            layer_indices = [-1]
        self._layer_indices = set(
            idx % num_layers if idx < 0 else idx for idx in layer_indices
        )
        self._hooks = []
        self._captured = {}  # layer_idx -> attn_weights tensor

    # -- hook on the eager_attention_forward call inside compute_layer_complete --
    # We monkey-patch compute_layer_complete to store attn_weights.

    def enable(self):
        """Install hooks."""
        from transformers.models.gemma import modeling_gemma as _gemma

        self._orig_eager = _gemma.eager_attention_forward
        capture = self  # closure reference

        def hooked_eager(module, query, key, value, attention_mask, scaling, **kw):
            attn_output, attn_weights = capture._orig_eager(
                module, query, key, value, attention_mask, scaling, **kw
            )
            # identify layer index from module
            # store by id(module) first, resolve later
            capture._last_attn = attn_weights.detach()
            return attn_output, attn_weights

        _gemma.eager_attention_forward = hooked_eager

        # Register hooks on selected transformer layers.
        for idx in self._layer_indices:
            layer = self._model.paligemma_with_expert.paligemma.language_model.layers[idx]

            def _make_hook(layer_idx):
                def _hook(_mod, _inp, _out):
                    if hasattr(capture, '_last_attn'):
                        capture._captured[layer_idx] = capture._last_attn
                return _hook

            h = layer.self_attn.register_forward_hook(_make_hook(idx))
            self._hooks.append(h)

    def disable(self):
        """Remove hooks and restore original function."""
        from transformers.models.gemma import modeling_gemma as _gemma
        if hasattr(self, '_orig_eager'):
            _gemma.eager_attention_forward = self._orig_eager
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    def get(self):
        """Return captured attention dict {layer_idx: (batch, heads, seq, seq)}."""
        return dict(self._captured)

    def clear(self):
        self._captured.clear()


def make_attn_overlay(attn_weights, num_img_tokens_per_cam, img_h, img_w,
                      cam_images_rgb):
    """Create attention heatmap overlays for each camera view.

    Args:
        attn_weights: (heads, seq, seq) — single-batch attention from one layer.
        num_img_tokens_per_cam: number of patch tokens per camera (e.g. 256).
        img_h, img_w: original camera image size (for overlay).
        cam_images_rgb: list of 3 RGB uint8 HWC images [ext, back, wrist].

    Returns:
        list of 3 overlay images (uint8 HWC BGR, suitable for cv2.imshow).
    """
    n_cams = len(cam_images_rgb)
    n_patches = num_img_tokens_per_cam
    patch_grid = int(n_patches ** 0.5)  # e.g. 16 for 256 patches

    # attn_weights: (heads, seq_q, seq_k)
    # During denoise_step with KV cache, seq_q = suffix length (action tokens),
    # seq_k = prefix (image+lang) + suffix.  Action tokens attend to image
    # tokens at positions [0 .. n_cams*n_patches-1] in the KV dimension.
    # Average over heads and query positions (all action tokens).
    attn_avg = attn_weights.float().mean(dim=0).mean(dim=0)  # (seq_k,)

    overlays = []
    for cam_idx in range(n_cams):
        start = cam_idx * n_patches
        end = start + n_patches
        if end > len(attn_avg):
            # Not enough tokens — skip
            overlays.append(cam_images_rgb[cam_idx][:, :, ::-1].copy())
            continue
        cam_attn = attn_avg[start:end].cpu().numpy()
        cam_attn = cam_attn.reshape(patch_grid, patch_grid)
        # Normalize to [0, 255]
        cam_attn = cam_attn - cam_attn.min()
        if cam_attn.max() > 0:
            cam_attn = cam_attn / cam_attn.max()
        heatmap = cv2.resize((cam_attn * 255).astype(np.uint8),
                             (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        base_bgr = cam_images_rgb[cam_idx][:, :, ::-1].copy()
        base_bgr = cv2.resize(base_bgr, (img_w, img_h))
        overlay = cv2.addWeighted(base_bgr, 0.5, heatmap_color, 0.5, 0)
        overlays.append(overlay)

    return overlays


# ── Policy loading ────────────────────────────────────────────────────────────

def _load_blockpap_policy(args):
    """Load pi0.5 PyTorch policy from a .pt or safetensors checkpoint."""
    import safetensors.torch
    import openpi.policies.policy as _policy
    import openpi.transforms as transforms
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import checkpoints as _checkpoints

    config = _CONFIGS_DICT[args.config_name]
    pt_path = pathlib.Path(args.pretrained_path)

    if pt_path.is_file() and pt_path.suffix == ".pt":
        model = pi0_pytorch.PI0Pytorch(config=config.model)
        state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): {missing[:3]} …")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm_stats_dir = args.norm_stats_path
    elif pt_path.is_file() and pt_path.suffix == ".safetensors":
        # Direct path to a .safetensors file
        model = pi0_pytorch.PI0Pytorch(config=config.model)
        safetensors.torch.load_model(model, str(pt_path), strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm_stats_dir = args.norm_stats_path
    else:
        # Directory containing model.safetensors
        weight_path = str(pt_path / "model.safetensors")
        model = pi0_pytorch.PI0Pytorch(config=config.model)
        safetensors.torch.load_model(model, weight_path, strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm_stats_dir = str(pt_path)

    data_config = config.data.create(config.assets_dirs, config.model)
    asset_id = data_config.asset_id
    if asset_id is None:
        raise ValueError("asset_id is None; cannot locate norm stats.")

    norm_stats = _checkpoints.load_norm_stats(norm_stats_dir, asset_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = _policy.Policy(
        model,
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": args.num_steps},
        metadata=config.policy_metadata,
        is_pytorch=True,
        pytorch_device=device,
    )
    return policy


# ── Robot state helpers ───────────────────────────────────────────────────────

def get_robot_state(robot, target_ee_pose=None):
    """Read 7-D state from the real robot.

    Args:
        target_ee_pose: If provided (7D: [x,y,z,qx,qy,qz,qw]), use this as the
            EE pose instead of calling get_ee_pose(). Using the last commanded
            target pose rather than the measured pose reduces accumulation of
            sensor noise and controller tracking error.
            On the first step pass None to get the real measured pose.

    Returns:
        state (7,): [tcp_x, tcp_y, tcp_z, euler_x, euler_y, euler_z, gripper_total_width]
    """
    if target_ee_pose is not None:
        ee_pose = target_ee_pose.astype(np.float32)
    else:
        ee_pose = robot.get_ee_pose().astype(np.float32)   # [x, y, z, qx, qy, qz, qw]

    position  = ee_pose[:3]
    quat_xyzw = ee_pose[3:]                               # scipy expects [qx, qy, qz, qw]
    euler = Rotation.from_quat(quat_xyzw).as_euler("xyz").astype(np.float32)
    gripper_width = float(robot.get_gripper_position()[0])  # always read from hardware

    return np.concatenate([position, euler, [gripper_width]], dtype=np.float32)


def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Euler XYZ (scipy extrinsic) → quaternion [qx, qy, qz, qw]."""
    return Rotation.from_euler("xyz", euler).as_quat().astype(np.float32)


# ── Camera display ────────────────────────────────────────────────────────────

class CameraDisplay:
    """File-based camera display compatible with openpi camera_viewer.py.

    Docker has no display server, so we write frames to a shared directory and
    let the user run camera_viewer.py on the HOST machine to view them:

        # On the HOST (outside Docker):
        python /home/showlab/Users/zhijun/openpi/examples/lab/camera_viewer.py
    """

    FRAME_DIR  = "/home/showlab/Users/zhijun/eval/real_eval/tmp"
    FRAME_FILE = os.path.join(FRAME_DIR, "combined_frame.npy")
    STEP_FILE  = os.path.join(FRAME_DIR, "step.txt")
    INFO_FILE  = os.path.join(FRAME_DIR, "info.txt")

    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.running = False
        self.quit_requested = False

    def start(self):
        os.makedirs(self.FRAME_DIR, exist_ok=True)
        self.running = True
        self._thread.start()
        print(f"  [Display] Writing frames to: {self.FRAME_DIR}")
        print(f"  [Display] On the HOST run:")
        print(f"    FRAME_DIR={self.FRAME_DIR} python /home/showlab/Users/zhijun/openpi/examples/lab/camera_viewer.py")

    def update(self, frame_bgr: np.ndarray, t: int, gripper_state: str,
               action: np.ndarray, wrist_frame_bgr: np.ndarray | None = None,
               back_frame_bgr: np.ndarray | None = None):
        """Push frame(s) (BGR) to writer thread. Drops if busy."""
        if not self.running:
            return
        wrist_copy = wrist_frame_bgr.copy() if wrist_frame_bgr is not None else None
        back_copy  = back_frame_bgr.copy()  if back_frame_bgr  is not None else None
        try:
            self._q.put_nowait((frame_bgr.copy(), t, gripper_state, action.copy(), wrist_copy, back_copy))
        except queue.Full:
            pass

    def _loop(self):
        while self.running:
            try:
                frame_bgr, t, gripper_state, action, wrist_bgr, back_bgr = self._q.get(timeout=0.3)
            except queue.Empty:
                continue

            H, W = frame_bgr.shape[:2]
            pos_mm  = float(np.linalg.norm(action[:3]) * 1000)
            rot_deg = float(np.degrees(np.linalg.norm(action[3:6])))

            def _overlay(img, label):
                out = img.copy()
                cv2.putText(out, label, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2)
                return out

            ext_disp = _overlay(frame_bgr, f"External  step={t}  gripper={gripper_state}")
            cv2.putText(ext_disp, f"|Dpos|={pos_mm:.2f}mm  |Drot|={rot_deg:.2f}deg",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 0), 1)

            panels = [ext_disp]
            if wrist_bgr is not None:
                wrist_disp = _overlay(cv2.resize(wrist_bgr, (W, H)), "Wrist")
                panels.append(wrist_disp)
            if back_bgr is not None:
                back_disp = _overlay(cv2.resize(back_bgr, (W, H)), "Back")
                panels.append(back_disp)
            disp = np.hstack(panels)

            # Atomic write (temp → rename)
            tmp = self.FRAME_FILE + ".tmp.npy"
            np.save(tmp, disp)
            os.replace(tmp, self.FRAME_FILE)
            with open(self.STEP_FILE, "w") as f:
                f.write(str(t))
            with open(self.INFO_FILE, "w") as f:
                f.write(f"gripper={gripper_state}  |Dpos|={pos_mm:.2f}mm  |Drot|={rot_deg:.2f}deg")

    def stop(self):
        self.running = False
        self._thread.join(timeout=2)
        # clean up shared files
        for p in (self.FRAME_FILE, self.STEP_FILE, self.INFO_FILE):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ── Gripper controller ────────────────────────────────────────────────────────

class GripperController:
    """Binary gripper controller with hysteresis and cooldown."""

    def __init__(self, cooldown_steps: int = 10):
        self.cooldown_steps = cooldown_steps
        self.last_state: bool | None = None   # True = closed
        self.steps_since_cmd = 0

    def reset(self):
        self.last_state = None
        self.steps_since_cmd = 0

    def step(self, robot, gripper_norm: float):
        """Send gripper command when state changes and cooldown allows.

        Args:
            gripper_norm: [0, 1] where 0 = fully closed, 1 = fully open.
        """
        target_width = float(np.clip(gripper_norm, 0.0, 1.0)) * GRIPPER_MAX_WIDTH
        should_close = target_width < GRIPPER_CLOSE_THRESHOLD

        self.steps_since_cmd += 1

        # Initialise state from hardware on first call
        if self.last_state is None:
            hw_width = float(robot.get_gripper_position()[0])
            self.last_state = hw_width < GRIPPER_CLOSE_THRESHOLD
            print(
                f"  [GRIPPER] Initialised: "
                f"{'CLOSED' if self.last_state else 'OPEN'} "
                f"(hw_width={hw_width*1000:.1f} mm)"
            )

        prev_ok = getattr(robot, "get_gripper_prev_cmd_success", lambda: True)()
        if (
            should_close != self.last_state
            and self.steps_since_cmd >= self.cooldown_steps
            and prev_ok
        ):
            print(
                f"  [GRIPPER] {'OPEN' if self.last_state else 'CLOSED'} → "
                f"{'CLOSED' if should_close else 'OPEN'}"
            )
            robot.control_gripper(should_close)
            self.last_state = should_close
            self.steps_since_cmd = 0
        elif should_close != self.last_state:
            print(
                f"  [GRIPPER] cooldown "
                f"({self.steps_since_cmd}/{self.cooldown_steps} steps)"
            )


# ── Keyboard listener ─────────────────────────────────────────────────────────

class KeyboardListener:
    """Non-blocking single-key listener (no Enter required).

    Terminal is put in raw mode only while an episode step-loop is running,
    so that the inter-episode ``input()`` calls continue to work normally.

    Usage::

        kb = KeyboardListener()
        kb.start()
        # … for each episode:
        kb.activate()          # raw mode ON  – call before the step loop
        try:
            for t in range(max_steps):
                if kb.terminate_requested:
                    break
        finally:
            kb.deactivate()    # raw mode OFF – call after the step loop
        kb.stop()              # in the outer finally block
    """

    def __init__(self):
        self._terminate = False
        self._active = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
        self._old_settings = None
        self._is_tty = sys.stdin.isatty()

    def start(self):
        if not self._is_tty:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            if not self._active.wait(timeout=0.1):
                continue
            try:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch.lower() == 't':
                        self._terminate = True
            except Exception:
                pass

    def activate(self):
        """Enable raw-mode key capture. Call just before the step loop."""
        if not self._is_tty:
            return
        self._terminate = False
        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._active.set()

    def deactivate(self):
        """Disable key capture and restore cooked mode. Call after the step loop."""
        if not self._is_tty:
            return
        self._active.clear()
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
            self._old_settings = None

    @property
    def terminate_requested(self) -> bool:
        return self._terminate

    def stop(self):
        self._running = False
        self.deactivate()
        if self._thread is not None:
            self._thread.join(timeout=1)


# ── Action execution ──────────────────────────────────────────────────────────

def execute_delta_action(robot, model_action: np.ndarray, current_state: np.ndarray):
    """Apply a delta EEF action to the real robot.

    Args:
        model_action (7,): [Δx, Δy, Δz, Δeuler_x, Δeuler_y, Δeuler_z, gripper_norm]
        current_state (7,): current robot state from get_robot_state()

    Returns:
        target_ee_pose (7,): [x, y, z, qx, qy, qz, qw] sent to robot.
    """
    delta_pos = np.clip(model_action[:3], -MAX_POSITION_DELTA, MAX_POSITION_DELTA)
    delta_rot = np.clip(model_action[3:6], -MAX_ROTATION_DELTA, MAX_ROTATION_DELTA)

    if not (np.allclose(model_action[:3], delta_pos) and
            np.allclose(model_action[3:6], delta_rot)):
        print(
            f"  [WARN] Action clipped: "
            f"Δpos {np.round(model_action[:3], 4)} → {np.round(delta_pos, 4)}  "
            f"Δrot {np.round(model_action[3:6], 4)} → {np.round(delta_rot, 4)}"
        )

    target_pos = current_state[:3] + delta_pos
    target_euler = current_state[3:6] + delta_rot
    target_quat = _euler_to_quat(target_euler)

    target_ee_pose = np.concatenate([target_pos, target_quat]).astype(np.float32)
    robot.update_desired_ee_pose(target_ee_pose)
    return target_ee_pose


# ── Camera helper ─────────────────────────────────────────────────────────────

def setup_camera(args):
    """Create and return (external_cam, wrist_cam, back_cam).

    wrist_cam and back_cam are None if the corresponding serial is not provided.
    """
    if args.use_mock_camera:
        print("  Using MOCK camera (no hardware)")
        ext_cam = camera_utils.MockCamera(width=640, height=480)
        wrist_cam = camera_utils.MockCamera(width=640, height=480) if args.wrist_camera_serial else None
        back_cam = camera_utils.MockCamera(width=640, height=480) if args.back_camera_serial else None
        return ext_cam, wrist_cam, back_cam

    devices = camera_utils.list_realsense_devices()
    print(f"  Found {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        print(f"    [{i}] {dev['name']}  serial={dev['serial_number']}")

    if not devices:
        raise RuntimeError(
            "No RealSense cameras found. "
            "Use --use_mock_camera for testing without hardware."
        )

    def _init_cam(serial, label):
        """Create a RealSenseCamera and apply per-camera exposure if configured."""
        cam = camera_utils.RealSenseCamera(
            serial_number=serial,
            width=640,
            height=480,
            fps=30,
        )
        if args.use_per_camera_exposure and serial in PER_CAMERA_EXPOSURE:
            exp_cfg = PER_CAMERA_EXPOSURE[serial]
            _set_camera_exposure(cam, exp_cfg['exposure'], exp_cfg['gain'])
        else:
            print(f"  {label} camera: using auto-exposure")
        print(f"  {label} camera (serial: {serial}) ready.")
        return cam

    ext_cam = _init_cam(args.external_camera_serial, "External")

    wrist_cam = None
    if args.wrist_camera_serial:
        wrist_cam = _init_cam(args.wrist_camera_serial, "Wrist")

    back_cam = None
    if args.back_camera_serial:
        back_cam = _init_cam(args.back_camera_serial, "Back")

    return ext_cam, wrist_cam, back_cam


def _set_camera_exposure(cam, exposure_us: int, gain: int):
    """Disable auto-exposure and set manual exposure/gain on a RealSenseCamera.

    Args:
        exposure_us: Exposure time in microseconds (e.g. 6000 for D435).
                     D435 color sensor range: ~1 – 165000 μs.
        gain:        Analog gain (e.g. 64).
    """
    try:
        import pyrealsense2 as rs
        profile = cam.pipeline.get_active_profile()
        device = profile.get_device()
        sensors = device.query_sensors()
        # D435: index 0 = Stereo Module, index 1 = RGB Camera
        # D405: index 0 = Stereo Module (provides colour), no index 1
        # Find the sensor that actually has the exposure option.
        color_sensor = None
        for s in sensors:
            if s.supports(rs.option.exposure):
                # Prefer the dedicated RGB Camera sensor if available (D435);
                # fall back to Stereo Module for D405.
                if "RGB" in s.get_info(rs.camera_info.name):
                    color_sensor = s
                    break
                if color_sensor is None:
                    color_sensor = s
        if color_sensor is None:
            print(f"  [Camera][WARN] No sensor with exposure control found")
            return
        sensor_name = color_sensor.get_info(rs.camera_info.name)
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, float(exposure_us))
        color_sensor.set_option(rs.option.gain, float(gain))
        print(f"  [Camera] Manual exposure set on {sensor_name}: "
              f"exposure={exposure_us} μs  gain={gain}")
    except Exception as e:
        print(f"  [Camera][WARN] Failed to set exposure: {e}")


# ── Main evaluation loop ─────────────────────────────────────────────────────

def main(args):
    logger = setup_logger(args.exp_name, args.log_dir)
    np.random.seed(args.seed)

    # ── Policy ──────────────────────────────────────────────────────────────
    logger.info("Loading pi0.5 policy …")
    policy = _load_blockpap_policy(args)
    logger.info("Policy loaded.")

    # ── Attention capture (optional) ────────────────────────────────────────
    attn_cap = None
    if args.save_attn_map:
        attn_cap = AttentionCapture(policy._model, layer_indices=[-1])
        attn_cap.enable()
        logger.info("Attention capture enabled (last layer).")

    # ── Camera ──────────────────────────────────────────────────────────────
    logger.info("Initialising camera(s) …")
    cam, wrist_cam, back_cam = setup_camera(args)
    logger.info(f"Camera(s) ready. wrist={'yes' if wrist_cam else 'no'}  back={'yes' if back_cam else 'no'}")

    # ── Robot ────────────────────────────────────────────────────────────────
    if args.use_mock_robot:
        logger.info("Using MOCK robot (no hardware)")
        robot = MockRobot(ip=args.nuc_ip, port=args.nuc_port)
    else:
        logger.info(f"Connecting to robot at {args.nuc_ip}:{args.nuc_port} …")
        robot = FrankaInterface(ip=args.nuc_ip, port=args.nuc_port)

    Kx  = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
    Kxd = np.array([ 37.0,  37.0,  37.0,  2.0,  2.0,  2.0])

    gripper_ctrl = GripperController(cooldown_steps=args.gripper_cooldown_steps)
    dt = 1.0 / args.control_frequency

    # ── Camera display ───────────────────────────────────────────────────────
    display = None
    if args.show_camera:
        display = CameraDisplay()
        display.start()

    # ── Keyboard listener ────────────────────────────────────────────────────
    kb = KeyboardListener()
    kb.start()

    # ── Episode loop ─────────────────────────────────────────────────────────
    session_video_dir: pathlib.Path | None = None  # set on first episode

    try:
        for episode_idx in range(args.num_episodes):
            logger.info(f"\n─── Episode {episode_idx + 1}/{args.num_episodes} ───")
            logger.info("Move robot to start pose, then press Enter to begin …")
            try:
                input()
            except EOFError:
                pass

            # Start controller here so it doesn't time out while waiting for Enter
            logger.info("Starting Cartesian impedance controller …")
            robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
            logger.info("Controller started.")

            # Initialise session video directory on the first episode
            if session_video_dir is None and args.save_video:
                session_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                session_video_dir = (
                    pathlib.Path(args.log_dir) / args.exp_name / session_ts
                )
                session_video_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"  Video directory: {session_video_dir}")

            policy.reset()
            gripper_ctrl.reset()
            action_plan = collections.deque()
            frames = []
            target_pose = None   # None → use real get_ee_pose() on first step

            kb.activate()
            logger.info("  [KB] Press 't' to terminate and skip to next episode.")
            for t in range(args.max_steps):
                step_start = time.time()

                # ── 1. Robot state ───────────────────────────────────────────
                # Normally use the last commanded target pose to avoid sensor
                # noise.  Every state_refresh_interval steps, fall back to the
                # real measured pose to prevent drift accumulation.
                use_real = (
                    target_pose is None
                    or (args.state_refresh_interval > 0
                        and t % args.state_refresh_interval == 0)
                )
                state = get_robot_state(robot, None if use_real else target_pose)
                if use_real and target_pose is not None:
                    # Also sync target_pose so the next delta is applied on
                    # top of the real pose, not the stale commanded one.
                    target_pose = robot.get_ee_pose().astype(np.float32)

                # ── 2. Camera image(s) ───────────────────────────────────────
                ret, frame, _ = cam.read()
                if not ret:
                    logger.warning(f"  [t={t}] External camera read failed, skipping step")
                    continue
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # uint8 HWC RGB

                wrist_img = None
                ret_w, wrist_frame = False, None
                if wrist_cam is not None:
                    ret_w, wrist_frame, _ = wrist_cam.read()
                    if ret_w:
                        wrist_img = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
                    else:
                        logger.warning(f"  [t={t}] Wrist camera read failed, skipping wrist input")

                back_img = None
                ret_b, back_frame = False, None
                if back_cam is not None:
                    ret_b, back_frame, _ = back_cam.read()
                    if ret_b:
                        back_img = cv2.cvtColor(back_frame, cv2.COLOR_BGR2RGB)
                    else:
                        logger.warning(f"  [t={t}] Back camera read failed, skipping back input")

                if args.save_video:
                    frames.append(img.copy())

                # ── 3. Policy inference (re-plan when queue is empty) ────────
                if not action_plan:
                    observation = {
                        "observation/image": img,
                        "observation/state": state,
                        "prompt": args.task_description,
                    }
                    if wrist_img is not None:
                        observation["observation/wrist_image"] = wrist_img
                    if back_img is not None:
                        observation["observation/back_image"] = back_img
                    inference_start = time.time()
                    action_chunk = policy.infer(observation)["actions"]
                    inference_ms = (time.time() - inference_start) * 1000

                    assert action_chunk.ndim == 2 and action_chunk.shape[1] == 7, (
                        f"Unexpected action shape: {action_chunk.shape}"
                    )
                    action_plan.extend(action_chunk[: args.action_chunk])
                    logger.info(
                        f"  [t={t}] NEW CHUNK  size={len(action_plan)}  "
                        f"inference={inference_ms:.1f} ms"
                    )

                    # ── 3b. Attention map overlay (optional) ──────────────
                    if attn_cap is not None:
                        captured = attn_cap.get()
                        if captured:
                            layer_idx = max(captured.keys())
                            aw = captured[layer_idx]  # (batch, heads, seq_q, seq_k)
                            if aw.dim() == 4:
                                aw = aw[0]  # remove batch → (heads, seq_q, seq_k)
                            cam_imgs = [
                                img,
                                back_img if back_img is not None else np.zeros_like(img),
                                wrist_img if wrist_img is not None else np.zeros_like(img),
                            ]
                            overlays = make_attn_overlay(
                                aw, num_img_tokens_per_cam=256,
                                img_h=480, img_w=640,
                                cam_images_rgb=cam_imgs,
                            )
                            attn_dir = (
                                pathlib.Path(args.log_dir) / args.exp_name / "attn_maps"
                            )
                            attn_dir.mkdir(parents=True, exist_ok=True)
                            combined = np.concatenate(overlays, axis=1)
                            save_path = attn_dir / f"ep{episode_idx:02d}_t{t:04d}.jpg"
                            cv2.imwrite(str(save_path), combined)
                        attn_cap.clear()

                model_action = action_plan.popleft()   # (7,)

                # ── 4. Debug logging ─────────────────────────────────────────
                if t % args.log_interval == 0:
                    logger.info(
                        f"  [t={t}] action={np.round(model_action, 5).tolist()}"
                    )
                    logger.info(
                        f"  [t={t}] state ={np.round(state, 4).tolist()}"
                    )

                # ── 5. Execute action ────────────────────────────────────────
                target_pose = execute_delta_action(robot, model_action, state)
                gripper_ctrl.step(robot, model_action[6])

                # ── 6. Camera display ────────────────────────────────────────
                if display is not None:
                    gripper_str = "CLOSED" if gripper_ctrl.last_state else "OPEN"
                    display.update(frame, t, gripper_str, model_action,
                                   wrist_frame_bgr=wrist_frame if wrist_cam is not None and ret_w else None,
                                   back_frame_bgr=back_frame if back_cam is not None and ret_b else None)
                    if display.quit_requested:
                        logger.info("  User pressed q, stopping episode.")
                        break

                # ── 6.5 Keyboard terminate ───────────────────────────────────
                if kb.terminate_requested:
                    logger.info("  User pressed 't', skipping to next episode.")
                    break

                # ── 7. Timing ────────────────────────────────────────────────
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                elif t % args.log_interval == 0:
                    logger.warning(
                        f"  [t={t}] step overran: {elapsed*1000:.1f} ms "
                        f"(target {dt*1000:.1f} ms)"
                    )

            kb.deactivate()
            logger.info(f"  Episode {episode_idx + 1} done ({t + 1} steps).")

            # ── Save video ───────────────────────────────────────────────────
            if args.save_video and frames and session_video_dir is not None:
                out_path = session_video_dir / f"ep{episode_idx + 1:03d}.mp4"
                subsampled = frames[:: args.video_temp_subsample]
                imageio.mimwrite(
                    str(out_path), subsampled,
                    fps=max(1, 30 // args.video_temp_subsample)
                )
                logger.info(f"  Video saved: {out_path}  ({len(subsampled)} frames)")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")

    finally:
        if attn_cap is not None:
            attn_cap.disable()
        kb.stop()
        if display is not None:
            display.stop()
        logger.info("Cleaning up …")
        try:
            robot.terminate_current_policy()
        except Exception:
            pass  # "no controller running" is expected if episode ended naturally
        try:
            robot.close()
        except Exception as e:
            logger.warning(f"  Robot close error: {e}")
        try:
            cam.release()
        except Exception as e:
            logger.warning(f"  Camera cleanup error: {e}")
        if wrist_cam is not None:
            try:
                wrist_cam.release()
            except Exception as e:
                logger.warning(f"  Wrist camera cleanup error: {e}")
        if back_cam is not None:
            try:
                back_cam.release()
            except Exception as e:
                logger.warning(f"  Back camera cleanup error: {e}")
        logger.info("Done.")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate pi0.5 (PyTorch .pt) on real Franka for BlockPAP task."
    )

    # ── Logging ──
    parser.add_argument("--log_dir", type=str, default="/home/showlab/Users/zhijun/eval/real_eval")
    parser.add_argument("--exp_name", type=str, default="blockpap_real_eval")

    # ── Policy ──
    parser.add_argument("--config_name", type=str, default="pi05_blockpap_mix",
                        help="OpenPI data config name (must match training)")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to full_weights.pt or safetensors directory")
    parser.add_argument("--norm_stats_path", type=str,
                        default="../my_ckpt/models/pi05_base",
                        help="Directory containing <asset_id>/norm_stats.json")
    parser.add_argument("--num_steps", type=int, default=5,
                        help="Flow-matching denoising steps")
    parser.add_argument("--action_chunk", type=int, default=8,
                        help="Execute this many actions before re-planning")
    parser.add_argument("--task_description", type=str,
                        default=TASK_DESCRIPTION)

    # ── Episode ──
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=400)

    # ── Robot ──
    parser.add_argument("--nuc_ip", type=str, default="192.168.1.143")
    parser.add_argument("--nuc_port", type=int, default=4242)
    parser.add_argument("--control_frequency", type=float, default=10.0,
                        help="Hz – target control loop rate")
    parser.add_argument("--gripper_cooldown_steps", type=int, default=5)
    parser.add_argument("--state_refresh_interval", type=int, default=0,
                        help="Every N control steps, replace target_pose with real "
                             "measured pose to prevent drift accumulation. "
                             "0 = never refresh (always use target_pose after step 0).")
    parser.add_argument("--use_mock_robot", action="store_true",
                        help="Use MockRobot (no hardware) for testing")

    # ── Camera ──
    parser.add_argument("--external_camera_serial", type=str, default=None,
                        help="RealSense serial number for external camera")
    parser.add_argument("--wrist_camera_serial", type=str, default=None,
                        help="RealSense serial number for wrist camera. "
                             "If set, wrist image is fed to the model's left_wrist_0_rgb slot.")
    parser.add_argument("--use_mock_camera", action="store_true",
                        help="Use MockCamera (no hardware) for testing")
    parser.add_argument("--back_camera_serial", type=str, default=None,
                        help="RealSense serial number for back camera. "
                             "If set, back image is fed to the model's observation/back_image slot.")
    parser.add_argument("--use_per_camera_exposure", action="store_true",
                        help="Apply manual exposure from PER_CAMERA_EXPOSURE dict. "
                             "Default: auto-exposure (same as before).")
    parser.add_argument("--show_camera", action="store_true",
                        help="Show live camera preview window (requires display)")

    # ── Video ──
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--video_temp_subsample", type=int, default=1,
                        help="Save every Nth frame")

    # ── Attention map ──
    parser.add_argument("--save_attn_map", action="store_true",
                        help="Save attention heatmap overlays for each re-plan step.")

    # ── Misc ──
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
