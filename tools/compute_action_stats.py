#!/usr/bin/env python3
"""
统计ALOHA数据集每个维度动作的最大值和最小值

支持三种动作模式:
- absolute: 绝对动作，直接统计 action 数值
- relative: 相对动作，统计 action_t - action_{t-1}，夹爪维度(6, 13)保持绝对值
- delta_first: 相对chunk起始动作，统计 action_t - state_0，夹爪维度(6, 13)保持绝对值

用法:
    # 统计绝对动作
    python scripts/compute_action_stats.py --data_dir /path/to/data --mode absolute
    
    # 统计相对动作
    python scripts/compute_action_stats.py --data_dir /path/to/data --mode relative
    
    # 统计delta_first动作
    python scripts/compute_action_stats.py --data_dir /path/to/data --mode delta_first
    
    # 同时统计所有模式
    python scripts/compute_action_stats.py --data_dir /path/to/data --mode all
"""
import os
import sys
import glob
import argparse
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


def compute_absolute_action_stats(
    hdf5_files: List[str], 
    action_key: str = "action",
    percentiles: List[float] = [5, 95]
) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """
    统计绝对动作的最大值、最小值和百分位数
    
    Args:
        hdf5_files: HDF5文件路径列表
        action_key: 动作键名
        percentiles: 要统计的百分位数列表
        
    Returns:
        (action_min, action_max, percentile_values): 每个维度的最小值、最大值和百分位数
    """
    all_actions = []
    
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, 'r') as f:
            actions = f[action_key][:].astype(np.float32)  # (T, action_dim)
            all_actions.append(actions)
    
    # 合并所有数据
    all_actions = np.concatenate(all_actions, axis=0)
    
    # 计算统计量
    action_min = all_actions.min(axis=0)
    action_max = all_actions.max(axis=0)
    
    # 计算百分位数
    percentile_values = {}
    for p in percentiles:
        percentile_values[p] = np.percentile(all_actions, p, axis=0)
    
    return action_min, action_max, percentile_values


def compute_relative_action_stats(
    hdf5_files: List[str], 
    action_key: str = "action",
    state_key: str = "observations/qpos",
    abs_action_min: np.ndarray = None,
    abs_action_max: np.ndarray = None,
    abs_percentile_values: Dict[float, np.ndarray] = None,
    percentiles: List[float] = [5, 95],
) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """
    统计相对动作的最大值、最小值和百分位数
    相对动作: delta_t = action_t - action_{t-1}
    首帧特殊处理: a_0 = action_0 - state_0
    夹爪维度(索引6和13)保持绝对值，使用绝对动作的统计范围
    
    Args:
        hdf5_files: HDF5文件路径列表
        action_key: 动作键名
        state_key: 状态键名
        abs_action_min: 绝对动作的最小值（用于夹爪维度）
        abs_action_max: 绝对动作的最大值（用于夹爪维度）
        abs_percentile_values: 绝对动作的百分位数（用于夹爪维度）
        percentiles: 要统计的百分位数列表
        
    Returns:
        (action_min, action_max, percentile_values): 每个维度的最小值、最大值和百分位数
    """
    all_delta_actions = []
    
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, 'r') as f:
            actions = f[action_key][:].astype(np.float32)  # (T, action_dim)
            states = f[state_key][:].astype(np.float32)    # (T, state_dim)
            
            if len(actions) < 1:
                continue
            
            # 计算相对动作: action_t - action_{t-1}
            # 首帧特殊处理: a_0 = action_0 - state_0
            if len(actions) == 1:
                # 只有一帧，使用 action_0 - state_0
                delta_actions = actions - states
            else:
                # 首帧: action_0 - state_0
                # 其他帧: action_t - action_{t-1}
                prev_actions = np.concatenate([states[0:1], actions[:-1]], axis=0)
                delta_actions = actions - prev_actions  # (T, action_dim)
            
            # 夹爪维度(索引6和13)保持绝对值，不计算相对动作
            # 动作格式: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
            delta_actions[:, 6] = actions[:, 6]   # 左夹爪用绝对值
            delta_actions[:, 13] = actions[:, 13]  # 右夹爪用绝对值
            
            all_delta_actions.append(delta_actions)
    
    if len(all_delta_actions) == 0:
        raise ValueError("没有找到有效的数据用于统计")
    
    # 合并所有数据
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)
    
    # 计算统计量
    action_min = all_delta_actions.min(axis=0)
    action_max = all_delta_actions.max(axis=0)
    
    # 计算百分位数
    percentile_values = {}
    for p in percentiles:
        percentile_values[p] = np.percentile(all_delta_actions, p, axis=0)
    
    # 夹爪维度(索引6和13)使用绝对动作的统计范围
    if abs_action_min is not None and abs_action_max is not None:
        action_min[6] = abs_action_min[6]
        action_max[6] = abs_action_max[6]
        action_min[13] = abs_action_min[13]
        action_max[13] = abs_action_max[13]
    
    # 夹爪维度的百分位数也使用绝对动作的百分位数
    if abs_percentile_values is not None:
        for p in percentiles:
            if p in abs_percentile_values:
                percentile_values[p][6] = abs_percentile_values[p][6]
                percentile_values[p][13] = abs_percentile_values[p][13]
    
    return action_min, action_max, percentile_values


def compute_delta_first_action_stats(
    hdf5_files: List[str], 
    action_key: str = "action",
    state_key: str = "observations/qpos",
    abs_action_min: np.ndarray = None,
    abs_action_max: np.ndarray = None,
    abs_percentile_values: Dict[float, np.ndarray] = None,
    percentiles: List[float] = [5, 95],
    chunk_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """
    统计delta_first模式动作的最大值、最小值和百分位数
    delta_first模式: action_t - state_0 (chunk起始时刻的状态)
    夹爪维度(索引6和13)保持绝对值，使用绝对动作的统计范围
    
    Args:
        hdf5_files: HDF5文件路径列表
        action_key: 动作键名
        state_key: 状态键名
        abs_action_min: 绝对动作的最小值（用于夹爪维度）
        abs_action_max: 绝对动作的最大值（用于夹爪维度）
        abs_percentile_values: 绝对动作的百分位数（用于夹爪维度）
        percentiles: 要统计的百分位数列表
        chunk_size: action chunk的大小，用于模拟训练时的采样
        
    Returns:
        (action_min, action_max, percentile_values): 每个维度的最小值、最大值和百分位数
    """
    all_delta_actions = []
    
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, 'r') as f:
            actions = f[action_key][:].astype(np.float32)  # (T, action_dim)
            states = f[state_key][:].astype(np.float32)    # (T, state_dim)
            
            ep_length = len(actions)
            if ep_length < 1:
                continue
            
            # 模拟训练时的采样：遍历所有可能的chunk起始位置
            for start_idx in range(ep_length):
                # 获取chunk起始时刻的状态
                state_0 = states[start_idx]
                
                # 获取从start_idx开始的chunk_size个动作
                end_idx = min(start_idx + chunk_size, ep_length)
                chunk_actions = actions[start_idx:end_idx]
                
                # 计算delta_first: action_t - state_0
                delta_actions = chunk_actions - state_0[np.newaxis, :]
                
                # 夹爪维度(索引6和13)保持绝对值
                delta_actions[:, 6] = chunk_actions[:, 6]
                delta_actions[:, 13] = chunk_actions[:, 13]
                
                all_delta_actions.append(delta_actions)
    
    if len(all_delta_actions) == 0:
        raise ValueError("没有找到有效的数据用于统计")
    
    # 合并所有数据
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)
    
    # 计算统计量
    action_min = all_delta_actions.min(axis=0)
    action_max = all_delta_actions.max(axis=0)
    
    # 计算百分位数
    percentile_values = {}
    for p in percentiles:
        percentile_values[p] = np.percentile(all_delta_actions, p, axis=0)
    
    # 夹爪维度(索引6和13)使用绝对动作的统计范围
    if abs_action_min is not None and abs_action_max is not None:
        action_min[6] = abs_action_min[6]
        action_max[6] = abs_action_max[6]
        action_min[13] = abs_action_min[13]
        action_max[13] = abs_action_max[13]
    
    # 夹爪维度的百分位数也使用绝对动作的百分位数
    if abs_percentile_values is not None:
        for p in percentiles:
            if p in abs_percentile_values:
                percentile_values[p][6] = abs_percentile_values[p][6]
                percentile_values[p][13] = abs_percentile_values[p][13]
    
    return action_min, action_max, percentile_values


def format_stats_for_yaml(
    action_min: np.ndarray, 
    action_max: np.ndarray, 
    mode: str, 
    margin: float = 0.1,
    percentile_values: Dict[float, np.ndarray] = None,
    use_percentile: bool = False,
    percentiles: List[float] = [5, 95]
) -> Dict:
    """
    将统计结果格式化为YAML配置格式
    
    Args:
        action_min: 每个维度的最小值
        action_max: 每个维度的最大值
        mode: 动作模式 (absolute/relative)
        margin: 边界余量百分比，用于扩展范围避免边界值（仅在不使用百分位数时生效）
        percentile_values: 百分位数字典
        use_percentile: 是否使用百分位数作为范围
        percentiles: 百分位数列表，用于确定上下界
        
    Returns:
        配置字典
    """
    if use_percentile and percentile_values is not None:
        # 使用百分位数作为范围
        pct_low = min(percentiles)
        pct_high = max(percentiles)
        action_min_final = percentile_values[pct_low]
        action_max_final = percentile_values[pct_high]
    else:
        # 添加边界余量
        range_values = action_max - action_min
        action_min_final = action_min - margin * range_values
        action_max_final = action_max + margin * range_values
    
    # 转换为Python列表（YAML兼容）
    config = {
        f"action_min_{mode}": [float(f"{x:.6f}") for x in action_min_final],
        f"action_max_{mode}": [float(f"{x:.6f}") for x in action_max_final],
    }
    
    return config


def print_stats(
    action_min: np.ndarray, 
    action_max: np.ndarray, 
    mode: str,
    percentile_values: Dict[float, np.ndarray] = None,
    percentiles: List[float] = [5, 95]
):
    """打印统计结果"""
    dim_names = [
        "left_joint_1", "left_joint_2", "left_joint_3", 
        "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
        "right_joint_1", "right_joint_2", "right_joint_3", 
        "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper"
    ]
    
    # 构建百分位数表头
    pct_headers = ""
    if percentile_values is not None:
        pct_headers = " ".join([f"{p}%".center(12) for p in percentiles])
    
    print(f"\n{'='*80}")
    print(f"动作统计结果 - {mode.upper()} 模式")
    print(f"{'='*80}")
    
    if percentile_values is not None:
        print(f"{'维度':<20} {'最小值':>12} {pct_headers} {'最大值':>12} {'范围':>12}")
        print(f"{'-'*80}")
        
        for i, (name, min_val, max_val) in enumerate(zip(dim_names, action_min, action_max)):
            range_val = max_val - min_val
            pct_vals = " ".join([f"{percentile_values[p][i]:>12.6f}" for p in percentiles])
            print(f"{name:<20} {min_val:>12.6f} {pct_vals} {max_val:>12.6f} {range_val:>12.6f}")
    else:
        print(f"{'维度':<20} {'最小值':>12} {'最大值':>12} {'范围':>12}")
        print(f"{'-'*60}")
        
        for i, (name, min_val, max_val) in enumerate(zip(dim_names, action_min, action_max)):
            range_val = max_val - min_val
            print(f"{name:<20} {min_val:>12.6f} {max_val:>12.6f} {range_val:>12.6f}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="统计ALOHA数据集动作的最大值和最小值")
    parser.add_argument("--data_dir", type=str, default="/data1/lhy/traindata/pick_pineapple", 
                       help="数据目录路径")
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["absolute", "relative", "delta_first", "all"],
                       help="动作模式: absolute(绝对), relative(相对), delta_first(相对chunk起始), all(统计所有模式)")
    parser.add_argument("--action_key", type=str, default="action", help="HDF5中的动作键名")
    parser.add_argument("--margin", type=float, default=0.01, help="边界余量百分比（仅在不使用百分位数时生效）")
    parser.add_argument("--percentiles", type=float, nargs="+", default=[5, 95],
                       help="要统计的百分位数列表 (默认: 5 95)")
    parser.add_argument("--use_percentile", action="store_true", 
                       help="使用百分位数作为YAML配置的范围（而不是min/max+margin）")
    parser.add_argument("--chunk_size", type=int, default=50, 
                       help="delta_first模式的chunk大小 (默认: 50)")
    parser.add_argument("--output", type=str, default=None, help="输出YAML配置文件路径")
    
    args = parser.parse_args()
    percentiles = sorted(args.percentiles)
    
    # 查找所有HDF5文件
    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.hdf5")))
    if len(hdf5_files) == 0:
        print(f"错误: 在 {args.data_dir} 中没有找到 episode_*.hdf5 文件")
        sys.exit(1)
    
    print(f"找到 {len(hdf5_files)} 个episode文件")
    print(f"百分位数: {percentiles}")
    print(f"使用百分位数作为范围: {args.use_percentile}")
    
    config = {}
    abs_min, abs_max, abs_pct = None, None, None
    
    # 统计绝对动作 (必须先统计，用于其他模式的夹爪维度)
    if args.mode in ["absolute", "all"]:
        print("\n正在统计绝对动作...")
        abs_min, abs_max, abs_pct = compute_absolute_action_stats(
            hdf5_files, args.action_key, percentiles
        )
        print_stats(abs_min, abs_max, "absolute", abs_pct, percentiles)
        config.update(format_stats_for_yaml(
            abs_min, abs_max, "absolute", args.margin,
            abs_pct, args.use_percentile, percentiles
        ))
    
    # 统计相对动作
    if args.mode in ["relative", "all"]:
        # 如果只统计相对动作，需要先获取绝对动作的夹爪范围
        if abs_min is None or abs_max is None:
            print("\n正在统计绝对动作（用于夹爪维度）...")
            abs_min, abs_max, abs_pct = compute_absolute_action_stats(
                hdf5_files, args.action_key, percentiles
            )
        
        print("\n正在统计相对动作...")
        rel_min, rel_max, rel_pct = compute_relative_action_stats(
            hdf5_files, args.action_key, 
            abs_action_min=abs_min, 
            abs_action_max=abs_max,
            abs_percentile_values=abs_pct,
            percentiles=percentiles
        )
        print_stats(rel_min, rel_max, "relative", rel_pct, percentiles)
        config.update(format_stats_for_yaml(
            rel_min, rel_max, "relative", args.margin,
            rel_pct, args.use_percentile, percentiles
        ))
    
    # 统计delta_first动作
    if args.mode in ["delta_first", "all"]:
        # 如果需要，先获取绝对动作的夹爪范围
        if abs_min is None or abs_max is None:
            print("\n正在统计绝对动作（用于夹爪维度）...")
            abs_min, abs_max, abs_pct = compute_absolute_action_stats(
                hdf5_files, args.action_key, percentiles
            )
        
        print(f"\n正在统计delta_first动作 (chunk_size={args.chunk_size})...")
        df_min, df_max, df_pct = compute_delta_first_action_stats(
            hdf5_files, args.action_key,
            abs_action_min=abs_min,
            abs_action_max=abs_max,
            abs_percentile_values=abs_pct,
            percentiles=percentiles,
            chunk_size=args.chunk_size,
        )
        print_stats(df_min, df_max, "delta_first", df_pct, percentiles)
        config.update(format_stats_for_yaml(
            df_min, df_max, "delta_first", args.margin,
            df_pct, args.use_percentile, percentiles
        ))
    
    # 输出配置
    pct_low, pct_high = min(percentiles), max(percentiles)
    range_mode = f"{pct_low}%-{pct_high}% 百分位数" if args.use_percentile else f"min/max + {args.margin*100}% margin"
    print("\n" + "="*60)
    print(f"YAML配置内容 (使用 {range_mode}):")
    print("可直接复制到 conf/datamodule/aloha.yaml")
    print("="*60)
    print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    
    # 保存到文件
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"\n配置已保存到: {output_path}")


if __name__ == "__main__":
    main()
