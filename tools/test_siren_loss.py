#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：从数据集随机采样数据，使用训练好的 SirenVLA 模型计算 loss
支持 delta_first, relative, absolute 动作模式

使用方法:
    python scripts/test_siren_loss.py \
        --checkpoint /data1/lhy/beast_log/aloha_siren/2026-01-21/23-30-23/checkpoints/epoch=39_siren_loss=0.0006.ckpt \
        --data_dir /data1/lhy/traindata/pick_pineapple \
        --num_samples 200 \
        --batch_size 16 \
        --save_dir ./test_results \
        --action_mode delta_first \
        --arm_mode left \
        --chunk_size 50
"""

import argparse
import sys
import os
import h5py
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from beast.models.beast_florence_siren1 import SirenVLA


# =============================================================================
# 图像归一化参数 (与训练时一致 - CLIP)
# =============================================================================
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)


# =============================================================================
# 动作归一化参数 (从 config_aloha_siren.yaml 复制)
# =============================================================================
ACTION_STATS = {
    "delta_first": {
        "min": np.array([
            -0.68288, -1.846657, -1.619449, -0.782573, -1.177313, -0.881812, -0.001,
            0.0, 0.0, 0.0, 0.0, -0.027579, -0.003838, -0.0013
        ]),
        "max": np.array([
            0.663326, 1.52389, 1.784504, 0.782154, 1.103926, 1.185965, 0.0761,
            0.052803, 0.016362, 0.0, 0.0, 0.002215, 0.003977, 0.006
        ])
    },
    "relative": {
        "min": np.array([
            -0.058525, -0.080504, -0.071049, -0.058996, -0.067561, -0.064874, -0.001,
            0.0, 0.0, 0.0, 0.0, -0.009158, -0.002756, -0.0013
        ]),
        "max": np.array([
            0.048756, 0.079266, 0.076963, 0.067072, 0.088476, 0.079248, 0.0761,
            0.011949, 0.004535, 0.0, 0.0, 0.002128, 0.003018, 0.006
        ])
    },
    "absolute": {
        "min": np.array([
            -0.757087, -0.001867, -2.282286, -1.019323, -0.277238, -0.877346, -0.001,
            0.023759, 0.003628, 0.005094, -0.012821, 0.118009, -0.042127, -0.0013
        ]),
        "max": np.array([
            0.131406, 2.519681, 0.011949, 0.34349, 1.205904, 1.15488, 0.0761,
            0.088616, 0.023916, 0.008931, 0.039825, 0.290355, -0.035167, 0.006
        ])
    }
}


def get_hdf5_files(data_dir: str) -> List[str]:
    """获取所有 HDF5 文件列表"""
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    return hdf5_files


def get_episode_length(hdf5_path: str) -> int:
    """获取单个 episode 的长度"""
    with h5py.File(hdf5_path, 'r') as f:
        return len(f['action'])


def load_sample(
    hdf5_path: str,
    start_idx: int,
    action_seq_len: int,
    action_mode: str,
    arm_mode: str,
    camera_keys: Dict[str, str],
) -> Dict:
    """
    从 HDF5 文件加载单个样本
    
    Args:
        hdf5_path: HDF5 文件路径
        start_idx: 起始帧索引
        action_seq_len: 动作序列长度 (chunk size)
        action_mode: 动作模式 (absolute/relative/delta_first)
        arm_mode: 机械臂模式 (dual/left/right)
        camera_keys: 相机键名映射
    
    Returns:
        包含图像、状态、动作的字典
    """
    # 根据 arm_mode 确定切片
    if arm_mode == "left":
        arm_slice = slice(0, 7)
        gripper_indices = [6]
    elif arm_mode == "right":
        arm_slice = slice(7, 14)
        gripper_indices = [6]  # 切片后的索引
    else:  # dual
        arm_slice = slice(0, 14)
        gripper_indices = [6, 13]
    
    with h5py.File(hdf5_path, 'r') as f:
        ep_length = len(f['action'])
        
        # 计算动作索引
        action_indices = [min(start_idx + i, ep_length - 1) for i in range(action_seq_len)]
        
        # 读取动作
        actions_full = f['action'][action_indices].astype(np.float32)
        actions = actions_full[:, arm_slice]
        
        # 根据动作模式处理
        if action_mode == "delta_first":
            # delta_first: action_t = action_t - state_0 (chunk起始状态)
            state_0 = f['observations/qpos'][action_indices[0]].astype(np.float32)[arm_slice]
            delta_actions = actions - state_0[np.newaxis, :]
            
            # 夹爪维度保持绝对值
            for g_idx in gripper_indices:
                delta_actions[:, g_idx] = actions[:, g_idx]
            
            actions = delta_actions
            
        elif action_mode == "relative":
            # relative: action_t = action_t - action_{t-1}
            prev_actions = np.empty_like(actions)
            prev_actions[1:] = actions[:-1]
            
            # 第一帧的 prev_action
            first_prev_idx = max(action_indices[0] - 1, 0)
            prev_action_full = f['action'][first_prev_idx].astype(np.float32)
            prev_actions[0] = prev_action_full[arm_slice]
            
            delta_actions = actions - prev_actions
            
            # 首帧特殊处理
            if action_indices[0] == 0:
                state_0 = f['observations/qpos'][0].astype(np.float32)[arm_slice]
                delta_actions[0] = actions[0] - state_0
            
            # 夹爪维度保持绝对值
            for g_idx in gripper_indices:
                delta_actions[:, g_idx] = actions[:, g_idx]
            
            actions = delta_actions
        # else: absolute 模式保持不变
        
        # 读取图像
        rgb_obs = {}
        for out_key, src_key in camera_keys.items():
            img = f[src_key][start_idx]  # (H, W, C)
            rgb_obs[out_key] = img
        
        # 读取状态
        robot_obs = f['observations/qpos'][start_idx].astype(np.float32)[arm_slice]
        
    return {
        'rgb_obs': rgb_obs,
        'robot_obs': robot_obs,
        'actions': actions,
        'start_idx': start_idx,
        'hdf5_path': hdf5_path,
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
    mean = CLIP_MEAN.squeeze(0).squeeze(0).to(img.device)
    std = CLIP_STD.squeeze(0).squeeze(0).to(img.device)
    img = (img - mean) / std
    
    # 添加 time 维度: (1, C, H, W) -> (1, 1, C, H, W)
    img = img.unsqueeze(1)
    
    return img.to(device)


def normalize_actions(actions: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    """归一化动作到 [-1, 1]"""
    # 避免除零
    range_vals = action_max - action_min
    range_vals = np.where(range_vals < 1e-6, 1.0, range_vals)
    return 2 * (actions - action_min) / range_vals - 1


def denormalize_actions(actions_norm: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    """反归一化动作"""
    return (actions_norm + 1) / 2 * (action_max - action_min) + action_min


def build_batch(
    samples: List[Dict],
    action_min: np.ndarray,
    action_max: np.ndarray,
    device: str,
    lang_text: str,
    camera_keys: List[str],
) -> Dict:
    """
    将多个样本组装成 batch
    """
    batch_size = len(samples)
    
    # 处理图像
    rgb_obs = {}
    for cam_key in camera_keys:
        imgs = []
        for sample in samples:
            img = preprocess_image(sample['rgb_obs'][cam_key], device)
            imgs.append(img)
        rgb_obs[cam_key] = torch.cat(imgs, dim=0)  # (B, 1, C, H, W)
    
    # 处理动作 (归一化)
    actions_list = []
    for sample in samples:
        actions_norm = normalize_actions(sample['actions'], action_min, action_max)
        actions_list.append(actions_norm)
    
    actions = torch.from_numpy(np.stack(actions_list, axis=0)).float().to(device)  # (B, T, D)
    
    # 构建 batch
    batch = {
        "rgb_obs": rgb_obs,
        "lang_text": [lang_text] * batch_size,
        "actions": actions,
    }
    
    # 添加独立的相机键 (模型可能需要)
    for cam_key in camera_keys:
        batch[cam_key] = rgb_obs[cam_key]
    
    return batch


def compute_loss_and_metrics(
    model: torch.nn.Module,
    batch: Dict,
    samples: List[Dict],
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> Dict:
    """
    计算 loss 和各种指标
    """
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # 模型预测
        pred_actions = model._generate_actions_siren(batch)  # (B, T, D)
        gt_actions = batch["actions"]
        
        # 对齐长度
        T = min(pred_actions.shape[1], gt_actions.shape[1])
        pred_actions = pred_actions[:, :T, :]
        gt_actions = gt_actions[:, :T, :]
        
        # 计算 MSE loss (归一化空间)
        mse_loss = F.mse_loss(pred_actions, gt_actions)
        
        # 逐样本 loss
        per_sample_mse = ((pred_actions - gt_actions) ** 2).mean(dim=(1, 2))  # (B,)
        
        # 逐维度 MSE
        per_dim_mse = ((pred_actions - gt_actions) ** 2).mean(dim=(0, 1))  # (D,)
        
        # 逐时间步 MSE
        per_step_mse = ((pred_actions - gt_actions) ** 2).mean(dim=(0, 2))  # (T,)
        
        # 反归一化后计算真实空间的误差
        pred_np = pred_actions.cpu().numpy()
        gt_np = gt_actions.cpu().numpy()
        
        pred_denorm = denormalize_actions(pred_np, action_min, action_max)
        gt_denorm = denormalize_actions(gt_np, action_min, action_max)
        
        # 真实空间的 MAE
        mae_real = np.abs(pred_denorm - gt_denorm).mean()
        per_dim_mae_real = np.abs(pred_denorm - gt_denorm).mean(axis=(0, 1))
        
    return {
        "mse_loss": mse_loss.item(),
        "per_sample_mse": per_sample_mse.cpu().numpy(),
        "per_dim_mse": per_dim_mse.cpu().numpy(),
        "per_step_mse": per_step_mse.cpu().numpy(),
        "mae_real": mae_real,
        "per_dim_mae_real": per_dim_mae_real,
        "pred_actions": pred_np,
        "gt_actions": gt_np,
        "pred_denorm": pred_denorm,
        "gt_denorm": gt_denorm,
    }


def plot_results(
    all_metrics: Dict,
    save_dir: str,
    arm_mode: str,
    action_mode: str,
):
    """绘制结果图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 维度名称
    if arm_mode == "left":
        dim_names = ["L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_j6", "L_grip"]
    elif arm_mode == "right":
        dim_names = ["R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_j6", "R_grip"]
    else:
        dim_names = ["L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_j6", "L_grip",
                     "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_j6", "R_grip"]
    
    # 1. 逐维度 MSE (归一化空间)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(dim_names)), all_metrics['mean_per_dim_mse'])
    ax.set_xticks(range(len(dim_names)))
    ax.set_xticklabels(dim_names, rotation=45)
    ax.set_ylabel('MSE (Normalized)')
    ax.set_title(f'Per-Dimension MSE ({action_mode} mode)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_dim_mse.png'), dpi=150)
    plt.close()
    
    # 2. 逐维度 MAE (真实空间)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(dim_names)), all_metrics['mean_per_dim_mae_real'])
    ax.set_xticks(range(len(dim_names)))
    ax.set_xticklabels(dim_names, rotation=45)
    ax.set_ylabel('MAE (Real Space)')
    ax.set_title(f'Per-Dimension MAE in Real Space ({action_mode} mode)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_dim_mae_real.png'), dpi=150)
    plt.close()
    
    # 3. 逐时间步 MSE
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_metrics['mean_per_step_mse'], marker='o', markersize=3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('MSE (Normalized)')
    ax.set_title(f'Per-Step MSE over Chunk ({action_mode} mode)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_step_mse.png'), dpi=150)
    plt.close()
    
    # 4. Loss 分布直方图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_metrics['all_sample_mse'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(all_metrics['mean_mse'], color='r', linestyle='--', label=f'Mean: {all_metrics["mean_mse"]:.6f}')
    ax.set_xlabel('MSE Loss')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Sample MSE Losses ({action_mode} mode)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_distribution.png'), dpi=150)
    plt.close()
    
    # 5. 预测 vs GT 对比 (随机选几个样本的某个维度)
    if 'example_pred' in all_metrics and 'example_gt' in all_metrics:
        n_examples = min(4, len(all_metrics['example_pred']))
        fig, axes = plt.subplots(n_examples, 2, figsize=(14, 3 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_examples):
            pred = all_metrics['example_pred'][i]  # (T, D)
            gt = all_metrics['example_gt'][i]
            
            # 左图: 第一个关节维度
            axes[i, 0].plot(gt[:, 0], 'b-', label='GT', linewidth=2)
            axes[i, 0].plot(pred[:, 0], 'r--', label='Pred', linewidth=2)
            axes[i, 0].set_ylabel(dim_names[0])
            axes[i, 0].legend()
            axes[i, 0].set_title(f'Sample {i+1} - {dim_names[0]}')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 右图: 夹爪维度 (最后一个)
            axes[i, 1].plot(gt[:, -1], 'b-', label='GT', linewidth=2)
            axes[i, 1].plot(pred[:, -1], 'r--', label='Pred', linewidth=2)
            axes[i, 1].set_ylabel(dim_names[-1])
            axes[i, 1].legend()
            axes[i, 1].set_title(f'Sample {i+1} - {dim_names[-1]}')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pred_vs_gt_examples.png'), dpi=150)
        plt.close()
    
    print(f"Plots saved to {save_dir}")


def print_chunk_details(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    gt_actions_norm: np.ndarray,
    pred_actions_norm: np.ndarray,
    dim_names: List[str],
    action_mode: str,
):
    """
    详细打印一个 chunk 的内容
    
    Args:
        gt_actions: GT动作 (真实空间) (T, D)
        pred_actions: 预测动作 (真实空间) (T, D)
        gt_actions_norm: GT动作 (归一化空间) (T, D)
        pred_actions_norm: 预测动作 (归一化空间) (T, D)
        dim_names: 维度名称列表
        action_mode: 动作模式
    """
    T, D = gt_actions.shape
    
    print(f"\n{'='*100}")
    print(f"CHUNK DETAILS ({action_mode} mode)")
    print(f"{'='*100}")
    print(f"Chunk size: {T} steps, Action dim: {D}")
    
    # 打印表头
    print(f"\n[Ground Truth Actions (Real Space)]")
    header = f"{'Step':>4} | " + " | ".join([f"{name:>10}" for name in dim_names])
    print(header)
    print("-" * len(header))
    
    # 打印每个时间步的 GT 动作
    for t in range(min(T, 10)):  # 只打印前10步
        row = f"{t:>4} | " + " | ".join([f"{gt_actions[t, d]:>10.6f}" for d in range(D)])
        print(row)
    if T > 10:
        print(f"  ... (省略 {T-10} 步)")
        # 打印最后几步
        for t in range(max(10, T-3), T):
            row = f"{t:>4} | " + " | ".join([f"{gt_actions[t, d]:>10.6f}" for d in range(D)])
            print(row)
    
    # 打印预测动作
    print(f"\n[Predicted Actions (Real Space)]")
    print(header)
    print("-" * len(header))
    
    for t in range(min(T, 10)):
        row = f"{t:>4} | " + " | ".join([f"{pred_actions[t, d]:>10.6f}" for d in range(D)])
        print(row)
    if T > 10:
        print(f"  ... (省略 {T-10} 步)")
        for t in range(max(10, T-3), T):
            row = f"{t:>4} | " + " | ".join([f"{pred_actions[t, d]:>10.6f}" for d in range(D)])
            print(row)
    
    # 打印误差
    print(f"\n[Absolute Error (Real Space): |Pred - GT|]")
    print(header)
    print("-" * len(header))
    
    errors = np.abs(pred_actions - gt_actions)
    for t in range(min(T, 10)):
        row = f"{t:>4} | " + " | ".join([f"{errors[t, d]:>10.6f}" for d in range(D)])
        print(row)
    if T > 10:
        print(f"  ... (省略 {T-10} 步)")
        for t in range(max(10, T-3), T):
            row = f"{t:>4} | " + " | ".join([f"{errors[t, d]:>10.6f}" for d in range(D)])
            print(row)
    
    # 打印归一化空间的动作 (用于验证loss计算)
    print(f"\n[Ground Truth Actions (Normalized Space, for loss computation)]")
    header_norm = f"{'Step':>4} | " + " | ".join([f"{name:>8}" for name in dim_names])
    print(header_norm)
    print("-" * len(header_norm))
    
    for t in range(min(T, 5)):
        row = f"{t:>4} | " + " | ".join([f"{gt_actions_norm[t, d]:>8.4f}" for d in range(D)])
        print(row)
    if T > 5:
        print(f"  ...")
    
    print(f"\n[Predicted Actions (Normalized Space)]")
    print(header_norm)
    print("-" * len(header_norm))
    
    for t in range(min(T, 5)):
        row = f"{t:>4} | " + " | ".join([f"{pred_actions_norm[t, d]:>8.4f}" for d in range(D)])
        print(row)
    if T > 5:
        print(f"  ...")
    
    # 汇总统计
    print(f"\n[Summary Statistics]")
    print(f"  GT range (real):   min={gt_actions.min():.6f}, max={gt_actions.max():.6f}")
    print(f"  Pred range (real): min={pred_actions.min():.6f}, max={pred_actions.max():.6f}")
    print(f"  GT range (norm):   min={gt_actions_norm.min():.4f}, max={gt_actions_norm.max():.4f}")
    print(f"  Pred range (norm): min={pred_actions_norm.min():.4f}, max={pred_actions_norm.max():.4f}")
    
    # 逐维度误差统计
    print(f"\n[Per-Dimension Error Summary]")
    print(f"  {'Dim':<12} {'MAE(real)':<12} {'MSE(norm)':<12} {'Max Err':<12}")
    print(f"  {'-'*48}")
    for d, name in enumerate(dim_names):
        mae_real = np.abs(pred_actions[:, d] - gt_actions[:, d]).mean()
        mse_norm = ((pred_actions_norm[:, d] - gt_actions_norm[:, d]) ** 2).mean()
        max_err = np.abs(pred_actions[:, d] - gt_actions[:, d]).max()
        print(f"  {name:<12} {mae_real:<12.6f} {mse_norm:<12.6f} {max_err:<12.6f}")
    
    # 逐时间步误差统计
    print(f"\n[Per-Step MSE (Normalized Space)]")
    per_step_mse = ((pred_actions_norm - gt_actions_norm) ** 2).mean(axis=1)
    print(f"  Step 0:  {per_step_mse[0]:.6f}")
    print(f"  Step {T//4}: {per_step_mse[T//4]:.6f}")
    print(f"  Step {T//2}: {per_step_mse[T//2]:.6f}")
    print(f"  Step {T-1}: {per_step_mse[-1]:.6f}")
    print(f"  Mean:    {per_step_mse.mean():.6f}")


def save_chunk_to_csv(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    gt_actions_norm: np.ndarray,
    pred_actions_norm: np.ndarray,
    dim_names: List[str],
    save_path: str,
):
    """
    保存 chunk 数据到 CSV 文件
    
    Args:
        gt_actions: GT动作 (真实空间) (T, D)
        pred_actions: 预测动作 (真实空间) (T, D)
        gt_actions_norm: GT动作 (归一化空间) (T, D)
        pred_actions_norm: 预测动作 (归一化空间) (T, D)
        dim_names: 维度名称列表
        save_path: 保存路径
    """
    T, D = gt_actions.shape
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 写表头
        header = ['Step']
        for name in dim_names:
            header.extend([f'GT_{name}', f'Pred_{name}', f'Error_{name}'])
        header.extend(['Step_MSE_norm', 'Step_MAE_real'])
        writer.writerow(header)
        
        # 写每一行数据
        for t in range(T):
            row = [t]
            for d in range(D):
                gt_val = gt_actions[t, d]
                pred_val = pred_actions[t, d]
                error = abs(pred_val - gt_val)
                row.extend([f'{gt_val:.6f}', f'{pred_val:.6f}', f'{error:.6f}'])
            
            # 该时间步的整体 MSE 和 MAE
            step_mse = ((pred_actions_norm[t] - gt_actions_norm[t]) ** 2).mean()
            step_mae = np.abs(pred_actions[t] - gt_actions[t]).mean()
            row.extend([f'{step_mse:.6f}', f'{step_mae:.6f}'])
            
            writer.writerow(row)
        
        # 写汇总统计行
        writer.writerow([])
        writer.writerow(['=== Summary Statistics ==='])
        
        # 逐维度统计
        writer.writerow(['Dimension', 'MAE_real', 'MSE_norm', 'Max_Error'])
        for d, name in enumerate(dim_names):
            mae_real = np.abs(pred_actions[:, d] - gt_actions[:, d]).mean()
            mse_norm = ((pred_actions_norm[:, d] - gt_actions_norm[:, d]) ** 2).mean()
            max_err = np.abs(pred_actions[:, d] - gt_actions[:, d]).max()
            writer.writerow([name, f'{mae_real:.6f}', f'{mse_norm:.6f}', f'{max_err:.6f}'])
        
        # 总体统计
        writer.writerow([])
        total_mse = ((pred_actions_norm - gt_actions_norm) ** 2).mean()
        total_mae = np.abs(pred_actions - gt_actions).mean()
        total_max_err = np.abs(pred_actions - gt_actions).max()
        writer.writerow(['Total', f'{total_mae:.6f}', f'{total_mse:.6f}', f'{total_max_err:.6f}'])
    
    print(f"  CSV saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test SirenVLA model loss on dataset samples")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--data_dir", "-d", type=str, required=True, help="Path to data directory")
    parser.add_argument("--episode_idx", "-e", type=int, default=0, help="Episode index to test (default: 0)")
    parser.add_argument("--start_frame", "-s", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--chunk_size", type=int, default=50, help="Action sequence length (chunk size)")
    parser.add_argument("--action_mode", type=str, default="delta_first", choices=["absolute", "relative", "delta_first"])
    parser.add_argument("--arm_mode", type=str, default="left", choices=["dual", "left", "right"])
    parser.add_argument("--save_dir", type=str, default="./test_results", help="Directory to save results")
    parser.add_argument("--lang_text", type=str, default="Pick up the pineapple toy and put it on the plate")
    args = parser.parse_args()
    
    device = args.device
    
    # 根据 arm_mode 选择相机配置
    camera_keys = {"rgb_static": "observations/images/cam_high"}
    if args.arm_mode == "left" or args.arm_mode == "dual":
        camera_keys["rgb_left_wrist"] = "observations/images/cam_left_wrist"
    if args.arm_mode == "right" or args.arm_mode == "dual":
        camera_keys["rgb_right_wrist"] = "observations/images/cam_right_wrist"
    
    # 获取动作归一化参数
    action_stats = ACTION_STATS[args.action_mode]
    if args.arm_mode == "left":
        action_min = action_stats["min"][:7]
        action_max = action_stats["max"][:7]
        dim_names = ["L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_j6", "L_grip"]
    elif args.arm_mode == "right":
        action_min = action_stats["min"][7:]
        action_max = action_stats["max"][7:]
        dim_names = ["R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_j6", "R_grip"]
    else:
        action_min = action_stats["min"]
        action_max = action_stats["max"]
        dim_names = ["L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_j6", "L_grip",
                     "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_j6", "R_grip"]
    
    # ==========================================================================
    # 1. 加载模型
    # ==========================================================================
    print(f"\n{'='*70}")
    print("Loading SirenVLA model...")
    print(f"{'='*70}")
    
    model = SirenVLA.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        strict=False,
    )
    
    # 加载 EMA 权重
    print("Checking for EMA weights...")
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if "callbacks" in checkpoint_data and "EMA" in checkpoint_data["callbacks"]:
        ema_callback_state = checkpoint_data["callbacks"]["EMA"]
        if "ema_weights" in ema_callback_state:
            ema_weights_list = ema_callback_state["ema_weights"]
            original_state_dict = checkpoint_data.get("state_dict", {})
            
            ema_weights_dict = {}
            matched_count = 0
            
            for param_name, ema_weight in zip(original_state_dict.keys(), ema_weights_list):
                if param_name in model.state_dict():
                    current_param = model.state_dict()[param_name]
                    if ema_weight.shape == current_param.shape:
                        ema_weights_dict[param_name] = ema_weight
                        matched_count += 1
            
            if ema_weights_dict:
                model.load_state_dict(ema_weights_dict, strict=False)
                print(f"✓ Loaded {matched_count}/{len(original_state_dict)} EMA weights")
        else:
            print("⚠ No EMA weights found in checkpoint")
    else:
        print("⚠ No EMA callback found in checkpoint")
    
    model.eval()
    model.to(device)
    
    # 覆盖模型的 action_mode 为命令行指定的值
    # (因为旧 checkpoint 没有保存 action_mode，默认是 absolute)
    original_action_mode = model.action_mode
    model.action_mode = args.action_mode
    
    print(f"Model Info:")
    print(f"  - Action dim: {model.action_dim}")
    print(f"  - Chunk size: {model.chunk_size}")
    print(f"  - Action mode (from checkpoint): {original_action_mode}")
    print(f"  - Action mode (override to): {model.action_mode}")
    print(f"  - Using normalization params for: {args.action_mode}")
    
    # ==========================================================================
    # 2. 加载单个样本 (第一个 episode 的第一帧)
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"Loading single sample from {args.data_dir}")
    print(f"{'='*70}")
    
    hdf5_files = get_hdf5_files(args.data_dir)
    print(f"Found {len(hdf5_files)} episodes")
    
    if args.episode_idx >= len(hdf5_files):
        print(f"⚠ Episode index {args.episode_idx} out of range, using 0")
        args.episode_idx = 0
    
    hdf5_path = hdf5_files[args.episode_idx]
    ep_length = get_episode_length(hdf5_path)
    print(f"Using episode {args.episode_idx}: {hdf5_path}")
    print(f"Episode length: {ep_length} frames")
    print(f"Start frame: {args.start_frame}")
    print(f"Chunk size: {args.chunk_size}")
    
    # 加载样本
    sample = load_sample(
        hdf5_path,
        args.start_frame,
        args.chunk_size,
        args.action_mode,
        args.arm_mode,
        camera_keys,
    )
    
    print(f"\nSample loaded:")
    print(f"  - Robot obs shape: {sample['robot_obs'].shape}")
    print(f"  - Actions shape: {sample['actions'].shape}")
    print(f"  - Cameras: {list(sample['rgb_obs'].keys())}")
    
    # ==========================================================================
    # 3. 构建 batch 并推理
    # ==========================================================================
    print(f"\n{'='*70}")
    print("Running inference...")
    print(f"{'='*70}")
    
    # 构建单样本 batch
    batch = build_batch(
        [sample],
        action_min,
        action_max,
        device,
        args.lang_text,
        list(camera_keys.keys()),
    )
    
    with torch.no_grad():
        # 模型预测
        pred_actions_norm = model._generate_actions_siren(batch)  # (1, T, D)
        gt_actions_norm = batch["actions"]  # (1, T, D)
        
        # 对齐长度
        T = min(pred_actions_norm.shape[1], gt_actions_norm.shape[1])
        pred_actions_norm = pred_actions_norm[:, :T, :]
        gt_actions_norm = gt_actions_norm[:, :T, :]
        
        # 计算 MSE loss
        mse_loss = F.mse_loss(pred_actions_norm, gt_actions_norm)
        
        # 转换为 numpy
        pred_norm_np = pred_actions_norm[0].cpu().numpy()  # (T, D)
        gt_norm_np = gt_actions_norm[0].cpu().numpy()  # (T, D)
        
        # 反归一化
        pred_real = denormalize_actions(pred_norm_np, action_min, action_max)
        gt_real = denormalize_actions(gt_norm_np, action_min, action_max)
    
    # ==========================================================================
    # 4. 打印 chunk 详细内容
    # ==========================================================================
    print_chunk_details(
        gt_actions=gt_real,
        pred_actions=pred_real,
        gt_actions_norm=gt_norm_np,
        pred_actions_norm=pred_norm_np,
        dim_names=dim_names,
        action_mode=args.action_mode,
    )
    
    # ==========================================================================
    # 5. 保存 CSV 文件
    # ==========================================================================
    episode_name = os.path.basename(hdf5_path).replace('.hdf5', '')
    csv_dir = "/home/lhy/code/beast_calvin/test_results/single_sample_delta_first_left"
    os.makedirs(csv_dir, exist_ok=True)
    csv_save_path = os.path.join(
        csv_dir,
        f"chunk_comparison_{episode_name}_frame{args.start_frame}.csv"
    )
    save_chunk_to_csv(
        gt_actions=gt_real, 
        pred_actions=pred_real, 
        gt_actions_norm=gt_norm_np, 
        pred_actions_norm=pred_norm_np, 
        dim_names=dim_names,
        save_path=csv_save_path
    )
    
    # ==========================================================================
    # 6. 最终 Loss 汇总
    # ==========================================================================
    print(f"\n{'='*100}")
    print("FINAL LOSS SUMMARY")
    print(f"{'='*100}")
    print(f"  MSE Loss (Normalized Space): {mse_loss.item():.6f}")
    print(f"  RMSE (Normalized Space):     {np.sqrt(mse_loss.item()):.6f}")
    print(f"  MAE (Real Space):            {np.abs(pred_real - gt_real).mean():.6f}")
    print(f"  Max Error (Real Space):      {np.abs(pred_real - gt_real).max():.6f}")
    
    # 保存可视化
    save_dir = os.path.join(args.save_dir, f"single_sample_{args.action_mode}_{args.arm_mode}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制每个维度的对比图
    fig, axes = plt.subplots(len(dim_names), 1, figsize=(14, 3 * len(dim_names)))
    for d, name in enumerate(dim_names):
        axes[d].plot(gt_real[:, d], 'b-', label='GT', linewidth=2)
        axes[d].plot(pred_real[:, d], 'r--', label='Pred', linewidth=2)
        axes[d].set_ylabel(name)
        axes[d].legend()
        axes[d].grid(True, alpha=0.3)
        axes[d].set_title(f'{name}: MAE={np.abs(pred_real[:, d] - gt_real[:, d]).mean():.6f}')
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'single_chunk_comparison.png'), dpi=150)
    plt.close()
    
    print(f"\n  Plot saved to: {save_dir}/single_chunk_comparison.png")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
