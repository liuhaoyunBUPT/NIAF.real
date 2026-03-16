#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALOHA训练数据集chunk可视化工具

从构建好的AlohaMultiEpisodeDataset中选取一个chunk，
按照训练时的方式进行归一化处理，然后绘制三种动作模式的图
支持三种动作模式：absolute, relative, delta_first

使用方法:
    # 使用配置文件中的归一化参数
    python plot_aloha_chunk.py --config ../conf/config_aloha_siren.yaml --idx 100
    
    # 随机选取多个chunk
    python plot_aloha_chunk.py --config ../conf/config_aloha_siren.yaml -n 5
"""

import os
import sys
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头环境支持
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beast.datasets.aloha_dataset import AlohaMultiEpisodeDataset
from beast.utils.transforms import NormalizeActions


def get_joint_names(arm_mode: str) -> list:
    """根据arm_mode返回关节名称列表"""
    if arm_mode == "left":
        return [f"left_j{i}" for i in range(6)] + ["left_gripper"]
    elif arm_mode == "right":
        return [f"right_j{i}" for i in range(6)] + ["right_gripper"]
    else:  # dual
        return (
            [f"left_j{i}" for i in range(6)] + ["left_gripper"] +
            [f"right_j{i}" for i in range(6)] + ["right_gripper"]
        )


def save_separated_joint_plots(
    x_axis: np.ndarray,
    data: np.ndarray,
    joint_names: list,
    title: str,
    output_filename: str,
    ylabel_prefix: str = "",
):
    """
    绘制垂直排列的关节数据子图
    归一化后数据范围应在[-1, 1]之间
    """
    num_joints = data.shape[1]
    
    # 创建垂直排列的子图
    fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(12, 2 * num_joints), sharex=True)
    
    if num_joints == 1:
        axes = [axes]
    
    for i, joint_name in enumerate(joint_names):
        ax = axes[i]
        ax.plot(x_axis, data[:, i], color='#1f77b4', linewidth=1.5)
        
        # 设置Y轴标签
        label = f"{ylabel_prefix}{joint_name}" if ylabel_prefix else joint_name
        ax.set_ylabel(label, rotation=0, labelpad=50, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Y轴零线
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.1)
        
        # 夹爪不固定Y轴范围，其他关节固定为[-1, 1]
        if 'gripper' not in joint_name.lower():
            ax.set_ylim(-1, 1)
        
        # 移除多余边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 设置底部标签和标题
    axes[-1].set_xlabel('Frame Index', fontsize=12)
    fig.suptitle(title, fontsize=14, y=0.99)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"成功生成: {output_filename}")


def get_normalization_params(cfg, action_mode: str, arm_mode: str):
    """
    从配置中获取归一化参数
    
    Args:
        cfg: OmegaConf配置对象
        action_mode: 动作模式 (absolute, relative, delta_first)
        arm_mode: 机械臂模式 (dual, left, right)
    
    Returns:
        (min_vec, max_vec): 归一化参数
    """
    if action_mode == "absolute":
        min_vec = list(cfg.action_min_absolute)
        max_vec = list(cfg.action_max_absolute)
    elif action_mode == "relative":
        min_vec = list(cfg.action_min_relative)
        max_vec = list(cfg.action_max_relative)
    elif action_mode == "delta_first":
        min_vec = list(cfg.action_min_delta_first)
        max_vec = list(cfg.action_max_delta_first)
    else:
        raise ValueError(f"Unknown action_mode: {action_mode}")
    
    # 根据arm_mode切片
    if arm_mode == "left":
        min_vec = min_vec[:7]
        max_vec = max_vec[:7]
    elif arm_mode == "right":
        min_vec = min_vec[7:14]
        max_vec = max_vec[7:14]
    
    return min_vec, max_vec


def main():
    parser = argparse.ArgumentParser(
        description="从ALOHA训练数据集中选取chunk，按训练方式归一化后绘图"
    )
    parser.add_argument('--config', '-cfg', default='/home/lhy/config_siren.yaml',
                        help="训练配置文件路径 (如 ../conf/config_aloha_siren.yaml)")
    parser.add_argument('--data_dir', '-d', default=None, 
                        help="数据目录，默认使用配置文件中的root_data_dir")
    parser.add_argument('--arm_mode', '-a', default=None,
                        choices=['dual', 'left', 'right'],
                        help="机械臂模式，默认使用配置文件中的arm_mode")
    parser.add_argument('--chunk_len', '-c', type=int, default=None,
                        help="chunk长度，默认使用配置文件中的chunk_size")
    parser.add_argument('--idx', '-i', type=int, default=None,
                        help="指定数据集索引，默认随机选择")
    parser.add_argument('--num_samples', '-n', type=int, default=1,
                        help="随机采样的chunk数量，默认1个")
    parser.add_argument('--output_dir', '-o', default=None,
                        help="输出目录，默认为脚本所在目录下的plots子目录")
    parser.add_argument('--seed', type=int, default=None,
                        help="随机种子")
    parser.add_argument('--max_episodes', type=int, default=None,
                        help="限制加载的episode数量（调试用）")
    
    args = parser.parse_args()
    
    # 加载配置文件
    print(f"加载配置文件: {args.config}")
    cfg = OmegaConf.load(args.config)
    
    # 从配置文件或命令行参数获取设置
    data_dir = args.data_dir if args.data_dir else cfg.root_data_dir
    arm_mode = args.arm_mode if args.arm_mode else cfg.arm_mode
    chunk_len = args.chunk_len if args.chunk_len else cfg.chunk_size
    
    print(f"数据目录: {data_dir}")
    print(f"机械臂模式: {arm_mode}")
    print(f"Chunk长度: {chunk_len}")
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # 设置输出目录
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "plots")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取关节名称
    joint_names = get_joint_names(arm_mode)
    
    # 生成横轴（帧索引）
    frame_indices = np.arange(chunk_len)
    
    # 动作模式列表
    action_modes = ['absolute', 'relative', 'delta_first']
    
    # 首先构建一次数据集以获取总样本数
    print(f"\n构建数据集以获取总样本数...")
    temp_dataset = AlohaMultiEpisodeDataset(
        data_dir=data_dir,
        action_seq_len=chunk_len,
        obs_seq_len=1,
        subsample_rate=1,
        action_mode='absolute',
        arm_mode=arm_mode,
        lazy_load=True,
        max_episodes=args.max_episodes,
        action_transforms=None,
    )
    total_samples = len(temp_dataset)
    print(f"数据集大小: {total_samples} 个有效chunk")
    
    # 确定要处理的sample索引列表
    if args.idx is not None:
        sample_indices = [args.idx]
        if args.idx >= total_samples:
            print(f"错误: idx {args.idx} 超出范围 [0, {total_samples-1}]")
            sys.exit(1)
    else:
        if args.num_samples > total_samples:
            print(f"警告: num_samples ({args.num_samples}) 大于总样本数 ({total_samples})")
            sample_indices = list(range(total_samples))
        else:
            sample_indices = random.sample(range(total_samples), args.num_samples)
        sample_indices.sort()
    
    print(f"\n将处理 {len(sample_indices)} 个chunk: {sample_indices}")
    print(f"将为每个chunk生成三种动作模式的归一化后图表")
    
    # 打印归一化参数
    print(f"\n归一化参数:")
    for mode in action_modes:
        min_vec, max_vec = get_normalization_params(cfg, mode, arm_mode)
        print(f"  {mode}:")
        print(f"    min: {[f'{x:.4f}' for x in min_vec]}")
        print(f"    max: {[f'{x:.4f}' for x in max_vec]}")
    
    # 外层循环：遍历每个chunk
    for chunk_num, sample_idx in enumerate(sample_indices, 1):
        print(f"\n{'#'*60}")
        print(f"# 处理 Chunk {chunk_num}/{len(sample_indices)}: sample_idx={sample_idx}")
        print(f"{'#'*60}")
        
        # 内层循环：对每个chunk生成所有动作模式的图
        for action_mode in action_modes:
            print(f"\n{'='*60}")
            print(f"处理动作模式: {action_mode}")
            print(f"{'='*60}")
            
            # 获取该模式的归一化参数
            min_vec, max_vec = get_normalization_params(cfg, action_mode, arm_mode)
            
            # 创建归一化变换（与训练时一致）
            action_transforms = NormalizeActions(min_vec, max_vec)
            
            # 构建数据集（带归一化，与训练时完全一致）
            dataset = AlohaMultiEpisodeDataset(
                data_dir=data_dir,
                action_seq_len=chunk_len,
                obs_seq_len=1,
                subsample_rate=1,
                action_mode=action_mode,
                arm_mode=arm_mode,
                lazy_load=True,
                max_episodes=args.max_episodes,
                action_transforms=action_transforms,  # 使用训练时的归一化
            )
            
            print(f"选择数据集索引: {sample_idx}")
            
            # 从数据集获取一个样本
            sample = dataset[sample_idx]
            
            # 提取数据（已经归一化）
            robot_obs = sample["robot_obs"].numpy()  # (1, num_joints)
            actions = sample["actions"].numpy()  # (chunk_len, num_joints)
            
            print(f"robot_obs shape: {robot_obs.shape}")
            print(f"actions shape: {actions.shape}")
            print(f"actions range: [{actions.min():.4f}, {actions.max():.4f}]")
            
            # 生成文件名前缀
            prefix = f"idx{sample_idx}_{action_mode}_{arm_mode}_normalized"
        
            # ==================== 绘图 ====================
            print("开始绘图...")
            
            # 绘制归一化后的动作图
            action_title_map = {
                "absolute": "Normalized Action (Absolute)",
                "relative": "Normalized Action (Relative)",
                "delta_first": "Normalized Action (Delta First)",
            }
            save_separated_joint_plots(
                x_axis=frame_indices,
                data=actions,
                joint_names=joint_names,
                title=f"{action_title_map[action_mode]} - Dataset idx {sample_idx}",
                output_filename=os.path.join(output_dir, f"{prefix}_action.png"),
                ylabel_prefix="",
            )
            
            # 起始状态信息（只在第一个chunk的第一个模式打印）
            if chunk_num == 1 and action_mode == action_modes[0]:
                print(f"\n起始状态 (robot_obs):")
                for i, name in enumerate(joint_names):
                    print(f"  {name}: {robot_obs[0, i]:.4f}")
            
            print(f"图表已保存: {prefix}_action.png")
    
    print(f"\n{'#'*60}")
    print(f"# 全部完成")
    print(f"{'#'*60}")
    print(f"所有图表已保存到: {output_dir}")
    print(f"处理的chunk索引: {sample_indices}")
    print(f"每个chunk生成的动作模式: {action_modes}")
    print(f"总共生成图片数量: {len(sample_indices) * len(action_modes)}")


if __name__ == "__main__":
    main()
