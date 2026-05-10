import h5py
import numpy as np

with h5py.File('/storage/zhijun/real_franka/pick_and_place/episode_0.hdf5', 'r') as f:
    # 查看文件结构
    f.visit(print)

with h5py.File('/storage/zhijun/real_franka/pick_and_place/episode_0.hdf5', 'r') as f:
    timestamp = f['timestamp'][:]  # 或具体路径
    dt = np.diff(timestamp)
    freq = 1.0 / np.mean(dt)
    print(f"平均频率: {freq:.1f} Hz")
    print(f"平均时间间隔: {np.mean(dt)*1000:.1f} ms")
