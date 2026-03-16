import h5py
import numpy as np
import cv2
from pathlib import Path

# 1. HDF5 文件路径：下面这个要按你自己的实际路径改一下
#   root_data_dir 在 config_libero.yaml 里是：
#   /home/lhy/Code/ICRA/openvla/LIBERO/datasets
#   自己补上 libero_goal 和具体任务名/文件名
h5_path = Path(
    "/home/lhy/Code/ICRA/openvla/LIBERO/libero/datasets/libero_goal/turn_on_the_stove_demo.hdf5"
)

assert h5_path.is_file(), f"{h5_path} 不存在，先确认路径"

with h5py.File(h5_path, "r") as f:
    # 2. 读出一帧 agentview_rgb
    # 你在 myHDF5 里看到路径是 /data/demo_0/obs/agentview_rgb
    dset = f["/data/demo_0/obs/eye_in_hand_rgb"]  # agentview_rgb / eye_in_hand_rgb
    print("dataset shape:", dset.shape)  # 应该是 (80, 128, 128, 3)

    frame0 = dset[0]  # 第 0 帧，shape: (128,128,3)，uint8
    
# 3. 保存一张图，看看方向
out_path = Path("train_agentview_raw_from_h5.png")
# HDF5 里一般是 RGB，OpenCV 用 BGR，顺手转一下
frame_bgr = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(out_path), frame_bgr)
print(f"✅ 已保存训练图像到: {out_path.resolve()}")
