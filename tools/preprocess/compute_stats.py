"""
统计ALOHA数据集每个维度动作的归一化参数（最大值和最小值）

支持三种动作模式:
- absolute: 绝对动作，直接统计 action 数值
- relative: 相对动作，统计 action_t - action_{t-1}，夹爪维度(6, 13)保持绝对值
- delta_first: 相对chunk起始动作，统计 action_t - state_0，夹爪维度(6, 13)保持绝对值

归一化范围计算方式:
- 通过 --range 参数指定百分位数范围，如 [0, 100] 或 [0.1, 99.9]
- [0, 100]: 使用全部数据的 min/max
- [0.1, 99.9]: 使用 0.1% 和 99.9% 百分位数，舍弃两端异常值

用法:
    # 使用全部数据的 min/max (不舍弃异常值)
    python scripts/compute_action_stats.py --data_dir /path/to/data --mode all --range 0 100
    
    # 使用百分位数，舍弃两端 0.1% 的异常值
    python scripts/compute_action_stats.py --data_dir /path/to/data --mode all --range 0.1 99.9
    
    # 只统计 delta_first 模式
    python scripts/compute_action_stats.py --data_dir /path/to/data --mode delta_first --range 1 99
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
    pct_range: Tuple[float, float] = (0, 100),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统计绝对动作的归一化范围
    
    Args:
        hdf5_files: HDF5文件路径列表
        action_key: 动作键名
        pct_range: 百分位数范围 (low, high)，如 (0, 100) 或 (0.1, 99.9)
        
    Returns:
        (action_min, action_max): 每个维度的最小值和最大值
    """
    all_actions = []
    
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, 'r') as f:
            actions = f[action_key][:].astype(np.float32)  # (T, action_dim)
            all_actions.append(actions)
    
    # 合并所有数据
    all_actions = np.concatenate(all_actions, axis=0)
    
    # 根据百分位数范围计算 min/max
    pct_low, pct_high = pct_range
    if pct_low == 0 and pct_high == 100:
        # 使用全部数据的 min/max
        action_min = all_actions.min(axis=0)
        action_max = all_actions.max(axis=0)
    else:
        # 使用百分位数
        action_min = np.percentile(all_actions, pct_low, axis=0)
        action_max = np.percentile(all_actions, pct_high, axis=0)
    
    return action_min, action_max


def compute_relative_action_stats(
    hdf5_files: List[str], 
    action_key: str = "action",
    state_key: str = "observations/qpos",
    abs_action_min: np.ndarray = None,
    abs_action_max: np.ndarray = None,
    pct_range: Tuple[float, float] = (0, 100),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统计相对动作的归一化范围
    相对动作: delta_t = action_t - action_{t-1}
    首帧特殊处理: a_0 = action_0 - state_0
    夹爪维度(索引6和13)保持绝对值，使用绝对动作的统计范围
    
    Args:
        hdf5_files: HDF5文件路径列表
        action_key: 动作键名
        state_key: 状态键名
        abs_action_min: 绝对动作的最小值（用于夹爪维度）
        abs_action_max: 绝对动作的最大值（用于夹爪维度）
        pct_range: 百分位数范围 (low, high)
        
    Returns:
        (action_min, action_max): 每个维度的最小值和最大值
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
                delta_actions = actions - states
            else:
                prev_actions = np.concatenate([states[0:1], actions[:-1]], axis=0)
                delta_actions = actions - prev_actions  # (T, action_dim)
            
            # 夹爪维度(索引6和13)保持绝对值
            delta_actions[:, 6] = actions[:, 6]
            delta_actions[:, 13] = actions[:, 13]
            
            all_delta_actions.append(delta_actions)
    
    if len(all_delta_actions) == 0:
        raise ValueError("没有找到有效的数据用于统计")
    
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)
    
    # 根据百分位数范围计算 min/max
    pct_low, pct_high = pct_range
    if pct_low == 0 and pct_high == 100:
        action_min = all_delta_actions.min(axis=0)
        action_max = all_delta_actions.max(axis=0)
    else:
        action_min = np.percentile(all_delta_actions, pct_low, axis=0)
        action_max = np.percentile(all_delta_actions, pct_high, axis=0)
    
    # 夹爪维度使用绝对动作的统计范围
    if abs_action_min is not None and abs_action_max is not None:
        action_min[6] = abs_action_min[6]
        action_max[6] = abs_action_max[6]
        action_min[13] = abs_action_min[13]
        action_max[13] = abs_action_max[13]
    
    return action_min, action_max


def compute_delta_first_action_stats(
    hdf5_files: List[str], 
    action_key: str = "action",
    state_key: str = "observations/qpos",
    abs_action_min: np.ndarray = None,
    abs_action_max: np.ndarray = None,
    pct_range: Tuple[float, float] = (0, 100),
    chunk_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统计delta_first模式动作的归一化范围
    delta_first模式: action_t - state_0 (chunk起始时刻的状态)
    夹爪维度(索引6和13)保持绝对值，使用绝对动作的统计范围
    
    Args:
        hdf5_files: HDF5文件路径列表
        action_key: 动作键名
        state_key: 状态键名
        abs_action_min: 绝对动作的最小值（用于夹爪维度）
        abs_action_max: 绝对动作的最大值（用于夹爪维度）
        pct_range: 百分位数范围 (low, high)
        chunk_size: action chunk的大小
        
    Returns:
        (action_min, action_max): 每个维度的最小值和最大值
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
                state_0 = states[start_idx]
                end_idx = min(start_idx + chunk_size, ep_length)
                chunk_actions = actions[start_idx:end_idx]
                
                # 计算delta_first: action_t - state_0
                delta_actions = chunk_actions - state_0[np.newaxis, :]
                
                # 夹爪维度保持绝对值
                delta_actions[:, 6] = chunk_actions[:, 6]
                delta_actions[:, 13] = chunk_actions[:, 13]
                
                all_delta_actions.append(delta_actions)
    
    if len(all_delta_actions) == 0:
        raise ValueError("没有找到有效的数据用于统计")
    
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)
    
    # 根据百分位数范围计算 min/max
    pct_low, pct_high = pct_range
    if pct_low == 0 and pct_high == 100:
        action_min = all_delta_actions.min(axis=0)
        action_max = all_delta_actions.max(axis=0)
    else:
        action_min = np.percentile(all_delta_actions, pct_low, axis=0)
        action_max = np.percentile(all_delta_actions, pct_high, axis=0)
    
    # 夹爪维度使用绝对动作的统计范围
    if abs_action_min is not None and abs_action_max is not None:
        action_min[6] = abs_action_min[6]
        action_max[6] = abs_action_max[6]
        action_min[13] = abs_action_min[13]
        action_max[13] = abs_action_max[13]
    
    return action_min, action_max


def format_stats_for_yaml(action_min: np.ndarray, action_max: np.ndarray, mode: str) -> Dict:
    """将统计结果格式化为YAML配置格式"""
    return {
        f"action_min_{mode}": [float(f"{x:.6f}") for x in action_min],
        f"action_max_{mode}": [float(f"{x:.6f}") for x in action_max],
    }


def print_stats(action_min: np.ndarray, action_max: np.ndarray, mode: str):
    """打印统计结果"""
    dim_names = [
        "left_joint_1", "left_joint_2", "left_joint_3", 
        "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
        "right_joint_1", "right_joint_2", "right_joint_3", 
        "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper"
    ]
    
    print(f"\n{'='*70}")
    print(f"动作统计结果 - {mode.upper()} 模式")
    print(f"{'='*70}")
    print(f"{'维度':<20} {'最小值':>12} {'最大值':>12} {'范围':>12}")
    print(f"{'-'*70}")
    
    for i, (name, min_val, max_val) in enumerate(zip(dim_names, action_min, action_max)):
        range_val = max_val - min_val
        print(f"{name:<20} {min_val:>12.6f} {max_val:>12.6f} {range_val:>12.6f}")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="统计ALOHA数据集动作的归一化参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用全部数据 (不舍弃异常值)
  python scripts/compute_action_stats.py --data_dir /path/to/data --range 0 100
  
  # 舍弃两端 0.1%% 的异常值
  python scripts/compute_action_stats.py --data_dir /path/to/data --range 0.1 99.9
  
  # 舍弃两端 1%% 的异常值
  python scripts/compute_action_stats.py --data_dir /path/to/data --range 1 99
        """
    )
    parser.add_argument("--data_dir", type=str, default='/data1/lhy/traindata/cup', help="数据目录路径")
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["absolute", "relative", "delta_first", "all"],
                       help="动作模式 (默认: all)")
    parser.add_argument("--range", type=float, nargs=2, default=[0, 100], metavar=("LOW", "HIGH"),
                       help="百分位数范围 [LOW, HIGH]。[0,100]=全部数据，[0.1,99.9]=舍弃两端0.1%%异常值 (默认: 0 100)")
    parser.add_argument("--chunk_size", type=int, default=50, 
                       help="delta_first模式的chunk大小 (默认: 50)")
    parser.add_argument("--action_key", type=str, default="action", help="HDF5中的动作键名")
    parser.add_argument("--state_key", type=str, default="observations/qpos", help="HDF5中的状态键名")
    parser.add_argument("--output", type=str, default=None, help="输出YAML配置文件路径")
    
    args = parser.parse_args()
    pct_range = tuple(sorted(args.range))  # 确保 low < high
    
    # 验证百分位数范围
    if pct_range[0] < 0 or pct_range[1] > 100:
        print(f"错误: 百分位数范围必须在 [0, 100] 之间，当前值: {pct_range}")
        sys.exit(1)
    
    # 查找所有HDF5文件
    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.hdf5")))
    if len(hdf5_files) == 0:
        print(f"错误: 在 {args.data_dir} 中没有找到 episode_*.hdf5 文件")
        sys.exit(1)
    
    print(f"找到 {len(hdf5_files)} 个episode文件")
    if pct_range == (0, 100):
        print(f"归一化范围: 使用全部数据的 min/max")
    else:
        print(f"归一化范围: 使用 {pct_range[0]}% ~ {pct_range[1]}% 百分位数 (舍弃两端 {pct_range[0]}% 异常值)")
    
    config = {}
    abs_min, abs_max = None, None
    
    # 统计绝对动作 (必须先统计，用于其他模式的夹爪维度)
    if args.mode in ["absolute", "all"]:
        print("\n正在统计绝对动作...")
        abs_min, abs_max = compute_absolute_action_stats(
            hdf5_files, args.action_key, pct_range
        )
        print_stats(abs_min, abs_max, "absolute")
        config.update(format_stats_for_yaml(abs_min, abs_max, "absolute"))
    
    # 统计相对动作
    if args.mode in ["relative", "all"]:
        if abs_min is None:
            print("\n正在统计绝对动作（用于夹爪维度）...")
            abs_min, abs_max = compute_absolute_action_stats(
                hdf5_files, args.action_key, pct_range
            )
        
        print("\n正在统计相对动作...")
        rel_min, rel_max = compute_relative_action_stats(
            hdf5_files, args.action_key, args.state_key,
            abs_action_min=abs_min, abs_action_max=abs_max,
            pct_range=pct_range
        )
        print_stats(rel_min, rel_max, "relative")
        config.update(format_stats_for_yaml(rel_min, rel_max, "relative"))
    
    # 统计delta_first动作
    if args.mode in ["delta_first", "all"]:
        if abs_min is None:
            print("\n正在统计绝对动作（用于夹爪维度）...")
            abs_min, abs_max = compute_absolute_action_stats(
                hdf5_files, args.action_key, pct_range
            )
        
        print(f"\n正在统计delta_first动作 (chunk_size={args.chunk_size})...")
        df_min, df_max = compute_delta_first_action_stats(
            hdf5_files, args.action_key, args.state_key,
            abs_action_min=abs_min, abs_action_max=abs_max,
            pct_range=pct_range, chunk_size=args.chunk_size,
        )
        print_stats(df_min, df_max, "delta_first")
        config.update(format_stats_for_yaml(df_min, df_max, "delta_first"))
    
    # 输出配置
    print("\n" + "="*60)
    print("YAML配置内容 (可直接保存到 configs/action_stats/<dataset>.yaml):")
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
