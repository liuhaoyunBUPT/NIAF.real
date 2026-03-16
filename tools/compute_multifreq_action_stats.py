#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算多频率数据的动作归一化范围

用法:
    python compute_multifreq_action_stats.py --data_dir /path/to/data --chunk_size 200
"""
import os
import glob
import argparse
import numpy as np
import h5py
from typing import List, Tuple


def load_episode_data(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载一个episode的关节数据"""
    with h5py.File(hdf5_path, 'r') as f:
        is_multifreq = f.attrs.get('multifreq', False)
        
        if is_multifreq:
            qpos = np.array(f['joints/qpos'][:], dtype=np.float32)
            actions = np.array(f['actions/data'][:], dtype=np.float32)
        else:
            qpos = np.array(f['observations/qpos'][:], dtype=np.float32)
            actions = np.array(f['action'][:], dtype=np.float32)
    
    return qpos, actions


def compute_stats(data_dir: str, chunk_size: int = 200, arm_mode: str = "dual"):
    """计算动作统计信息"""
    
    # 查找所有episode
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {data_dir}")
    
    print(f"Found {len(hdf5_files)} episodes")
    
    # 确定arm切片
    if arm_mode == "left":
        arm_slice = slice(0, 7)
    elif arm_mode == "right":
        arm_slice = slice(7, 14)
    else:
        arm_slice = slice(0, 14)
    
    # 收集所有数据
    all_absolute = []
    all_relative = []
    all_delta_first = []
    
    for hdf5_path in hdf5_files:
        qpos, actions = load_episode_data(hdf5_path)
        
        # 应用arm切片
        actions = actions[:, arm_slice]
        n_frames = len(actions)
        
        # Absolute: 原始动作
        all_absolute.append(actions)
        
        # Relative: a_t = s_{t+1} - s_t
        relative = np.zeros_like(actions)
        relative[:-1] = actions[1:] - actions[:-1]
        relative[-1] = relative[-2]
        all_relative.append(relative)
        
        # Delta_first: 对于每个有效窗口，a_t = s_t - s_0
        for start in range(0, n_frames - chunk_size + 1, chunk_size // 4):  # 滑动窗口
            chunk = actions[start:start + chunk_size]
            delta = chunk - chunk[0:1]  # 相对于chunk起始
            all_delta_first.append(delta)
    
    # 合并所有数据
    all_absolute = np.concatenate(all_absolute, axis=0)
    all_relative = np.concatenate(all_relative, axis=0)
    all_delta_first = np.concatenate(all_delta_first, axis=0)
    
    print(f"\nData shapes:")
    print(f"  Absolute: {all_absolute.shape}")
    print(f"  Relative: {all_relative.shape}")
    print(f"  Delta_first: {all_delta_first.shape}")
    
    # 计算统计量 (使用百分位数避免异常值)
    def compute_minmax(data, percentile=99):
        """使用百分位数计算min/max，避免异常值"""
        low = np.percentile(data, 100 - percentile, axis=0)
        high = np.percentile(data, percentile, axis=0)
        # 稍微扩展范围
        margin = (high - low) * 0.05
        return low - margin, high + margin
    
    abs_min, abs_max = compute_minmax(all_absolute)
    rel_min, rel_max = compute_minmax(all_relative)
    delta_min, delta_max = compute_minmax(all_delta_first)
    
    # 输出YAML格式
    print("\n" + "="*60)
    print("YAML格式配置 (复制到config文件):")
    print("="*60)
    
    def print_yaml_list(name, values):
        print(f"\n{name}:")
        for v in values:
            print(f"- {v}")
    
    print_yaml_list("action_min_absolute", abs_min.tolist())
    print_yaml_list("action_max_absolute", abs_max.tolist())
    print_yaml_list("action_min_relative", rel_min.tolist())
    print_yaml_list("action_max_relative", rel_max.tolist())
    print_yaml_list("action_min_delta_first", delta_min.tolist())
    print_yaml_list("action_max_delta_first", delta_max.tolist())
    
    # 速度统计 (如果有)
    print("\n" + "="*60)
    print("速度统计 (用于参考):")
    print("="*60)
    
    all_qvel = []
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, 'r') as f:
            is_multifreq = f.attrs.get('multifreq', False)
            if is_multifreq and 'joints/qvel' in f:
                qvel = np.array(f['joints/qvel'][:], dtype=np.float32)
                all_qvel.append(qvel[:, arm_slice])
    
    if len(all_qvel) > 0:
        all_qvel = np.concatenate(all_qvel, axis=0)
        vel_min, vel_max = compute_minmax(all_qvel)
        print(f"\nVelocity (rad/s):")
        print(f"  Shape: {all_qvel.shape}")
        print(f"  Min: {vel_min}")
        print(f"  Max: {vel_max}")
        print(f"  Mean abs: {np.mean(np.abs(all_qvel), axis=0)}")
    else:
        print("\n(No velocity data found)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录路径')
    parser.add_argument('--chunk_size', type=int, default=200,
                       help='动作序列长度')
    parser.add_argument('--arm_mode', type=str, default='dual',
                       choices=['dual', 'left', 'right'],
                       help='机械臂模式')
    args = parser.parse_args()
    
    compute_stats(args.data_dir, args.chunk_size, args.arm_mode)


if __name__ == '__main__':
    main()
