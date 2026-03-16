#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：加载模型，从数据集读取第一帧并计算 loss

使用方法:
    python scripts/test_model_loss.py --checkpoint /path/to/checkpoint.ckpt
"""

import argparse
import sys
import os
import h5py
import glob
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from beast.models.beast_florence_siren1 import SirenVLA


# =============================================================================
# 图像归一化参数 (与训练时一致)
# =============================================================================
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)


def get_hdf5_files(data_dir: str):
    """获取所有 HDF5 文件列表"""
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    return hdf5_files


def load_first_frame_from_file(hdf5_path: str, action_seq_len: int = 30, print_structure: bool = False):
    """
    从单个 HDF5 文件加载第一帧
    
    Args:
        hdf5_path: HDF5 文件路径
        action_seq_len: 动作序列长度 (chunk size)
        print_structure: 是否打印数据结构
    
    Returns:
        batch: 包含图像、状态、动作的字典
    """
    with h5py.File(hdf5_path, 'r') as f:
        # 打印数据集结构 (只打印一次)
        if print_structure:
            print("\n=== HDF5 Structure ===")
            def _print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  {name}: {obj.shape}, dtype={obj.dtype}")
            f.visititems(_print_structure)
            print("======================\n")
        
        # 加载图像 (H, W, C) -> (1, 1, C, H, W)
        cam_high = f['observations/images/cam_high'][0]  # 第一帧
        cam_left = f['observations/images/cam_left_wrist'][0]
        cam_right = f['observations/images/cam_right_wrist'][0]
        
        # 加载状态和动作
        qpos = f['observations/qpos'][0]  # 第一帧状态
        actions = f['action'][:action_seq_len]  # 前 action_seq_len 帧动作
        
    return {
        'cam_high': cam_high,
        'cam_left_wrist': cam_left,
        'cam_right_wrist': cam_right,
        'qpos': qpos,
        'actions': actions,
    }


def preprocess_image(img_hwc: np.ndarray, device: str = 'cuda') -> torch.Tensor:
    """
    预处理图像 (与训练时一致)
    
    Args:
        img_hwc: HDF5 中读取的图像 (H, W, C), uint8
        
    Returns:
        tensor: (1, 1, C, H, W) 归一化后的图像
    """
    # HWC -> CHW
    img = torch.from_numpy(img_hwc).permute(2, 0, 1).float()  # (C, H, W)
    
    # Resize 到 224x224
    img = F.interpolate(
        img.unsqueeze(0), 
        size=(224, 224), 
        mode='bilinear', 
        align_corners=False, 
        antialias=True
    )  # (1, C, H, W)
    
    # ScaleImageTensor: 除以 255
    img = img / 255.0
    
    # CLIP 归一化
    mean = CLIP_MEAN.squeeze(0).squeeze(0).to(img.device)  # (3, 1, 1)
    std = CLIP_STD.squeeze(0).squeeze(0).to(img.device)
    img = (img - mean) / std
    
    # 添加 batch 和 time 维度: (1, C, H, W) -> (1, 1, C, H, W)
    img = img.unsqueeze(0)
    
    return img.to(device)


def normalize_actions(actions: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    """
    归一化动作到 [-1, 1]
    """
    return 2 * (actions - action_min) / (action_max - action_min) - 1


def main():
    parser = argparse.ArgumentParser(description="Test model loss on first frame")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="/home/lhy/act/data/corn",
        help="Path to data directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--action-seq-len",
        type=int,
        default=10,
        help="Action sequence length (chunk size)"
    )
    args = parser.parse_args()
    
    device = args.device
    
    # ==========================================================================
    # 1. 加载模型
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Loading model from checkpoint...")
    print(f"{'='*60}")
    
    model = SirenVLA.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        strict=False,
    )

    # Load EMA weights if they exist in the checkpoint
    print("Checking for EMA weights...")
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if "callbacks" in checkpoint_data and "EMA" in checkpoint_data["callbacks"]:
        ema_callback_state = checkpoint_data["callbacks"]["EMA"]
        if "ema_weights" in ema_callback_state:
            ema_weights_list = ema_callback_state["ema_weights"]
            
            # Get the original state dict from checkpoint to match parameter names
            original_state_dict = checkpoint_data.get("state_dict", {})
            
            # Create EMA weights dict using original parameter names
            ema_weights_dict = {}
            matched_count = 0
            total_count = len(original_state_dict)
            
            for param_name, ema_weight in zip(original_state_dict.keys(), ema_weights_list):
                if param_name in model.state_dict():
                    current_param = model.state_dict()[param_name]
                    if ema_weight.shape == current_param.shape:
                        ema_weights_dict[param_name] = ema_weight
                        matched_count += 1
                    else:
                        print(f"Warning: Shape mismatch for {param_name}: "
                              f"EMA shape {ema_weight.shape} vs current shape {current_param.shape}")
                else:
                    print(f"Warning: Parameter {param_name} not found in current model")
            
            # Load EMA weights into the model
            if ema_weights_dict:
                missing_keys, unexpected_keys = model.load_state_dict(ema_weights_dict, strict=False)
                print(f"Successfully loaded {matched_count}/{total_count} EMA weights from checkpoint!")
                if missing_keys:
                    print(f"Missing keys (using original weights): {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
            else:
                print("Warning: No compatible EMA weights found!")
        else:
            print("Warning: No EMA weights found in checkpoint!")
    else:
        print("Warning: No EMA callback found in checkpoint!")

    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully!")
    print(f"  - Action dim: {model.action_dim}")
    print(f"  - Chunk size: {model.chunk_size}")
    print(f"  - Action mode: {getattr(model, "action_mode", "unknown")}")
    
    # 获取动作归一化参数
    action_min = model.action_min.cpu().numpy()
    action_max = model.action_max.cpu().numpy()
    print(f"  - Action min: {action_min}")
    print(f"  - Action max: {action_max}")
    
    # ==========================================================================
    # 2. 加载数据并计算所有 episode 的 loss
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Loading data from: {args.data_dir}")
    print(f"{'='*60}")
    
    hdf5_files = get_hdf5_files(args.data_dir)
    num_episodes = len(hdf5_files)
    print(f"Found {num_episodes} episodes")
    
    # 存储每个 episode 的 loss
    all_losses = []
    all_per_dim_mse = []
    
    for ep_idx, hdf5_path in enumerate(hdf5_files):
        # 加载数据 (只在第一个 episode 打印结构)
        data = load_first_frame_from_file(
            hdf5_path, 
            args.action_seq_len, 
            print_structure=(ep_idx == 0)
        )
        
        # 预处理图像
        rgb_static = preprocess_image(data['cam_high'], device)
        rgb_left_wrist = preprocess_image(data['cam_left_wrist'], device)
        rgb_right_wrist = preprocess_image(data['cam_right_wrist'], device)
        
        # 归一化动作
        gt_actions = data['actions'].astype(np.float32)
        
        # 处理相对动作 (Relative Action)
        if model.action_mode != "absolute":
            # 1. 获取前一帧动作 (对于首帧，使用当前状态)
            # 由于 load_first_frame_from_file 只加载了前 action_seq_len 帧
            # 这里的 prev_actions 逻辑需要稍微简化处理，或者假设我们只测第一帧生成的序列
            
            # 为了严谨，我们需要加载更多上下文。但对于 "第一帧预测"，
            # 上一帧动作应该是该 Episode 的初始状态 (qpos) 或 0
            
            # 构造 prev_actions: (T, D)
            # t=0 时: prev_action = current_state (根据 DataModule 逻辑)
            # t>0 时: prev_action = action_{t-1}
            
            prev_actions = np.zeros_like(gt_actions)
            prev_actions[1:] = gt_actions[:-1]
            prev_actions[0] = data['qpos']  # t=0 使用初始状态作为"上一帧动作"
            
            # 计算 delta
            delta_actions = gt_actions - prev_actions
            
            # 保持夹爪维度为绝对动作 (Indices 6 and 13)
            # [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
            delta_actions[:, 6] = gt_actions[:, 6]
            delta_actions[:, 13] = gt_actions[:, 13]
            
            gt_actions = delta_actions

        gt_actions_norm = normalize_actions(gt_actions, action_min, action_max)
        gt_actions_tensor = torch.from_numpy(gt_actions_norm).float().unsqueeze(0).to(device)
        
        # 构建 batch
        batch = {
            "rgb_obs": {
                "rgb_static": rgb_static,
                "rgb_left_wrist": rgb_left_wrist,
                "rgb_right_wrist": rgb_right_wrist,
            },
            "rgb_static": rgb_static,
            "rgb_left_wrist": rgb_left_wrist,
            "rgb_right_wrist": rgb_right_wrist,
            "lang_text": ["pick up the soft object"],
            "actions": gt_actions_tensor,
        }
        
        # 前向传播
        with torch.no_grad():
            pred_actions = model._generate_actions_siren(batch)
            
            # 对齐长度
            T = min(pred_actions.shape[1], gt_actions_tensor.shape[1])
            pred_actions_t = pred_actions[:, :T, :]
            gt_actions_t = gt_actions_tensor[:, :T, :]
            
            # 计算 loss
            loss = F.mse_loss(pred_actions_t, gt_actions_t)
            per_dim_mse = ((pred_actions_t - gt_actions_t) ** 2).mean(dim=(0, 1))
            
            all_losses.append(loss.item())
            all_per_dim_mse.append(per_dim_mse.cpu().numpy())
        
        # 进度显示
        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            print(f"  Episode {ep_idx + 1:3d}/{num_episodes}: MSE = {loss.item():.6f}")
    
    # ==========================================================================
    # 3. 计算并打印统计结果
    # ==========================================================================
    all_losses = np.array(all_losses)
    all_per_dim_mse = np.array(all_per_dim_mse)  # (num_episodes, 14)
    
    print(f"\n{'='*60}")
    print(f"RESULTS (Average over {num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Mean MSE Loss: {all_losses.mean():.6f}")
    print(f"  Std MSE Loss:  {all_losses.std():.6f}")
    print(f"  Min MSE Loss:  {all_losses.min():.6f}")
    print(f"  Max MSE Loss:  {all_losses.max():.6f}")
    print(f"  Mean RMSE:     {np.sqrt(all_losses.mean()):.6f}")
    
    # 打印逐维度平均误差
    mean_per_dim_mse = all_per_dim_mse.mean(axis=0)
    print(f"\n  Per-dimension Mean MSE:")
    dim_names = [
        "L_joint1", "L_joint2", "L_joint3", "L_joint4", "L_joint5", "L_joint6", "L_gripper",
        "R_joint1", "R_joint2", "R_joint3", "R_joint4", "R_joint5", "R_joint6", "R_gripper"
    ]
    for i, (name, mse) in enumerate(zip(dim_names, mean_per_dim_mse)):
        print(f"    [{i:2d}] {name:10s}: {mse:.6f}")
    
    # 找出 loss 最大和最小的 episode
    worst_ep = np.argmax(all_losses)
    best_ep = np.argmin(all_losses)
    print(f"\n  Best episode:  {best_ep} (MSE = {all_losses[best_ep]:.6f})")
    print(f"  Worst episode: {worst_ep} (MSE = {all_losses[worst_ep]:.6f})")


if __name__ == "__main__":
    main()
