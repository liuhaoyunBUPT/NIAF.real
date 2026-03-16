#!/usr/bin/env python3
"""
测试 delta_first 动作模式下的数据处理流程

验证内容:
1. 训练时: 原始动作 -> delta_first变换 -> 归一化 -> 模型输入
2. 推理时: 模型输出 -> 反归一化 -> delta_first逆变换 -> 绝对动作
3. 端到端验证: 绝对动作应该与原始动作一致

使用方法:
    python scripts/test_delta_first_pipeline.py --data-dir /path/to/data
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import h5py
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))


def load_episode_data(hdf5_path: str, arm_mode: str = "left"):
    """加载一个 episode 的数据"""
    with h5py.File(hdf5_path, 'r') as f:
        actions = f['action'][:].astype(np.float32)  # (T, 14)
        states = f['observations/qpos'][:].astype(np.float32)  # (T, 14)
    
    # 根据 arm_mode 切片
    if arm_mode == "left":
        actions = actions[:, :7]
        states = states[:, :7]
        gripper_indices = [6]
    elif arm_mode == "right":
        actions = actions[:, 7:14]
        states = states[:, 7:14]
        gripper_indices = [6]
    else:  # dual
        gripper_indices = [6, 13]
    
    return actions, states, gripper_indices


def apply_delta_first_transform(actions: np.ndarray, state_0: np.ndarray, gripper_indices: list):
    """
    应用 delta_first 变换 (训练时)
    
    delta_actions[t] = actions[t] - state_0
    夹爪维度保持绝对值
    """
    delta_actions = actions - state_0[np.newaxis, :]
    
    # 夹爪维度保持绝对值
    for g_idx in gripper_indices:
        delta_actions[:, g_idx] = actions[:, g_idx]
    
    return delta_actions


def normalize_actions(actions: np.ndarray, action_min: np.ndarray, action_max: np.ndarray):
    """
    归一化动作到 [-1, 1]
    
    normalized = 2 * (action - min) / (max - min) - 1
    """
    return 2 * (actions - action_min) / (action_max - action_min) - 1


def denormalize_actions(normalized: np.ndarray, action_min: np.ndarray, action_max: np.ndarray):
    """
    反归一化动作
    
    action = (normalized + 1) / 2 * (max - min) + min
    """
    return (normalized + 1) / 2 * (action_max - action_min) + action_min


def inverse_delta_first_transform(delta_actions: np.ndarray, current_state: np.ndarray, gripper_indices: list):
    """
    逆 delta_first 变换 (推理时)
    
    absolute_actions[t] = delta_actions[t] + current_state
    夹爪维度保持不变（已经是绝对值）
    """
    joint_indices = [i for i in range(delta_actions.shape[-1]) if i not in gripper_indices]
    
    absolute_actions = delta_actions.copy()
    absolute_actions[:, joint_indices] = delta_actions[:, joint_indices] + current_state[joint_indices]
    
    return absolute_actions


def test_single_chunk(
    actions: np.ndarray,
    states: np.ndarray,
    start_idx: int,
    chunk_size: int,
    gripper_indices: list,
    action_min: np.ndarray,
    action_max: np.ndarray,
    verbose: bool = True,
):
    """
    测试单个 chunk 的端到端流程
    
    返回:
        max_error: 最大误差
        mean_error: 平均误差
    """
    end_idx = min(start_idx + chunk_size, len(actions))
    actual_chunk_size = end_idx - start_idx
    
    # 原始动作
    original_actions = actions[start_idx:end_idx]  # (chunk_size, action_dim)
    
    # chunk 起始状态
    state_0 = states[start_idx]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"测试 Chunk: start_idx={start_idx}, chunk_size={actual_chunk_size}")
        print(f"{'='*60}")
        print(f"State_0 (chunk 起始状态): {state_0}")
        print(f"原始动作 (第一帧): {original_actions[0]}")
        print(f"原始动作 (最后帧): {original_actions[-1]}")
    
    # ========== 训练时的处理流程 ==========
    # 1. delta_first 变换
    delta_actions = apply_delta_first_transform(original_actions, state_0, gripper_indices)
    
    if verbose:
        print(f"\n[训练流程] delta_first 变换后:")
        print(f"  delta_actions (第一帧): {delta_actions[0]}")
        print(f"  delta_actions (最后帧): {delta_actions[-1]}")
    
    # 2. 归一化
    normalized_actions = normalize_actions(delta_actions, action_min, action_max)
    
    if verbose:
        print(f"\n[训练流程] 归一化后 (应在 [-1, 1] 范围内):")
        print(f"  normalized (第一帧): {normalized_actions[0]}")
        print(f"  normalized (最后帧): {normalized_actions[-1]}")
        print(f"  min: {normalized_actions.min():.4f}, max: {normalized_actions.max():.4f}")
        
        # 检查是否超出 [-1, 1] 范围
        if normalized_actions.min() < -1.0 or normalized_actions.max() > 1.0:
            print(f"  ⚠️ 警告: 归一化后超出 [-1, 1] 范围!")
    
    # ========== 推理时的处理流程 ==========
    # 假设模型完美预测，输出与输入相同的归一化动作
    model_output = normalized_actions  # 模拟模型输出
    
    # 3. 反归一化
    denormalized_actions = denormalize_actions(model_output, action_min, action_max)
    
    if verbose:
        print(f"\n[推理流程] 反归一化后:")
        print(f"  denormalized (第一帧): {denormalized_actions[0]}")
        print(f"  denormalized (最后帧): {denormalized_actions[-1]}")
    
    # 验证反归一化的正确性
    denorm_error = np.abs(denormalized_actions - delta_actions).max()
    if verbose:
        print(f"  反归一化误差: {denorm_error:.10f}")
    
    # 4. 逆 delta_first 变换
    # 推理时 current_state 就是 chunk 起始状态
    current_state = state_0
    reconstructed_actions = inverse_delta_first_transform(denormalized_actions, current_state, gripper_indices)
    
    if verbose:
        print(f"\n[推理流程] 逆 delta_first 变换后 (应与原始动作一致):")
        print(f"  reconstructed (第一帧): {reconstructed_actions[0]}")
        print(f"  reconstructed (最后帧): {reconstructed_actions[-1]}")
    
    # ========== 验证端到端正确性 ==========
    error = np.abs(reconstructed_actions - original_actions)
    max_error = error.max()
    mean_error = error.mean()
    
    if verbose:
        print(f"\n[验证结果]")
        print(f"  最大误差: {max_error:.10f}")
        print(f"  平均误差: {mean_error:.10f}")
        
        if max_error < 1e-6:
            print(f"  ✅ 端到端验证通过!")
        else:
            print(f"  ❌ 端到端验证失败!")
            print(f"  原始动作 vs 重建动作 差异:")
            for t in range(min(5, actual_chunk_size)):
                diff = reconstructed_actions[t] - original_actions[t]
                print(f"    t={t}: {diff}")
    
    return max_error, mean_error


def test_normalization_range(
    data_dir: str,
    arm_mode: str,
    action_min: np.ndarray,
    action_max: np.ndarray,
    chunk_size: int = 50,
    max_episodes: int = 10,
):
    """
    测试归一化范围是否足够覆盖所有数据
    """
    import glob
    
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))[:max_episodes]
    
    if len(hdf5_files) == 0:
        print(f"❌ 未找到 HDF5 文件: {data_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"测试归一化范围覆盖度")
    print(f"{'='*60}")
    print(f"数据目录: {data_dir}")
    print(f"Episode 数量: {len(hdf5_files)}")
    print(f"Arm mode: {arm_mode}")
    print(f"Chunk size: {chunk_size}")
    
    all_delta_actions = []
    
    # 根据 arm_mode 确定夹爪索引
    if arm_mode == "left":
        gripper_indices = [6]
    elif arm_mode == "right":
        gripper_indices = [6]
    else:
        gripper_indices = [6, 13]
    
    for hdf5_path in hdf5_files:
        actions, states, _ = load_episode_data(hdf5_path, arm_mode)
        
        # 对每个可能的 chunk 起始位置计算 delta_first
        for start_idx in range(0, len(actions) - chunk_size + 1, chunk_size // 2):
            end_idx = min(start_idx + chunk_size, len(actions))
            chunk_actions = actions[start_idx:end_idx]
            state_0 = states[start_idx]
            
            delta_actions = apply_delta_first_transform(chunk_actions, state_0, gripper_indices)
            all_delta_actions.append(delta_actions)
    
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)
    
    # 统计实际数据范围
    actual_min = all_delta_actions.min(axis=0)
    actual_max = all_delta_actions.max(axis=0)
    
    print(f"\n配置的归一化范围:")
    print(f"  action_min: {action_min}")
    print(f"  action_max: {action_max}")
    
    print(f"\n实际数据范围:")
    print(f"  actual_min: {actual_min}")
    print(f"  actual_max: {actual_max}")
    
    # 检查是否超出范围
    min_overflow = actual_min < action_min
    max_overflow = actual_max > action_max
    
    if min_overflow.any() or max_overflow.any():
        print(f"\n⚠️ 警告: 实际数据超出配置的归一化范围!")
        for i in range(len(action_min)):
            if min_overflow[i]:
                print(f"  维度 {i}: 实际最小值 {actual_min[i]:.6f} < 配置最小值 {action_min[i]:.6f}")
            if max_overflow[i]:
                print(f"  维度 {i}: 实际最大值 {actual_max[i]:.6f} > 配置最大值 {action_max[i]:.6f}")
    else:
        print(f"\n✅ 归一化范围覆盖所有数据!")
    
    # 检查归一化后的范围
    normalized = normalize_actions(all_delta_actions, action_min, action_max)
    print(f"\n归一化后的数据范围:")
    print(f"  min: {normalized.min():.4f}")
    print(f"  max: {normalized.max():.4f}")
    
    if normalized.min() < -1.0 or normalized.max() > 1.0:
        overflow_ratio = ((normalized < -1.0) | (normalized > 1.0)).sum() / normalized.size * 100
        print(f"  ⚠️ 有 {overflow_ratio:.2f}% 的数据超出 [-1, 1] 范围")


def main():
    parser = argparse.ArgumentParser(description="测试 delta_first 动作模式")
    parser.add_argument("--data-dir", type=str, default="/data1/lhy/traindata/cup",
                        help="数据目录路径")
    parser.add_argument("--arm-mode", type=str, default="left", choices=["dual", "left", "right"],
                        help="机械臂模式")
    parser.add_argument("--chunk-size", type=int, default=50, help="Chunk 大小")
    parser.add_argument("--episode-idx", type=int, default=0, help="测试的 episode 索引")
    parser.add_argument("--num-chunks", type=int, default=3, help="测试的 chunk 数量")
    parser.add_argument("--test-range", action="store_true", help="测试归一化范围覆盖度")
    
    args = parser.parse_args()
    
    # delta_first 归一化参数 (从 config_aloha_siren.yaml 复制)
    # 格式: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    action_min_delta_first_full = np.array([
        -0.6635, -1.856722, -1.517506, -0.526146, -1.445514, -0.480338, -0.0011,
        0.0, -0.000593, 0.0, 0.0, -0.002215, -0.00314, -0.0007
    ], dtype=np.float32)
    
    action_max_delta_first_full = np.array([
        0.732386, 1.282797, 1.798808, 0.476256, 1.155525, 0.532844, 0.0714,
        0.065136, 0.010152, 0.0, 0.0, 0.003122, 0.002808, -0.0005
    ], dtype=np.float32)
    
    # 根据 arm_mode 切片
    if args.arm_mode == "left":
        action_min = action_min_delta_first_full[:7]
        action_max = action_max_delta_first_full[:7]
        gripper_indices = [6]
    elif args.arm_mode == "right":
        action_min = action_min_delta_first_full[7:14]
        action_max = action_max_delta_first_full[7:14]
        gripper_indices = [6]
    else:
        action_min = action_min_delta_first_full
        action_max = action_max_delta_first_full
        gripper_indices = [6, 13]
    
    print("=" * 60)
    print("delta_first 动作模式测试")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"机械臂模式: {args.arm_mode}")
    print(f"Chunk 大小: {args.chunk_size}")
    print(f"归一化范围:")
    print(f"  action_min: {action_min}")
    print(f"  action_max: {action_max}")
    
    # 测试归一化范围覆盖度
    if args.test_range:
        test_normalization_range(
            args.data_dir, args.arm_mode,
            action_min, action_max,
            args.chunk_size
        )
        return
    
    # 加载测试数据
    import glob
    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.hdf5")))
    
    if len(hdf5_files) == 0:
        print(f"❌ 未找到 HDF5 文件: {args.data_dir}")
        return
    
    if args.episode_idx >= len(hdf5_files):
        print(f"❌ Episode 索引超出范围: {args.episode_idx} >= {len(hdf5_files)}")
        return
    
    hdf5_path = hdf5_files[args.episode_idx]
    print(f"\n测试文件: {hdf5_path}")
    
    actions, states, _ = load_episode_data(hdf5_path, args.arm_mode)
    print(f"Episode 长度: {len(actions)} 帧")
    print(f"动作维度: {actions.shape[-1]}")
    
    # 测试多个 chunk
    all_max_errors = []
    all_mean_errors = []
    
    for i in range(args.num_chunks):
        start_idx = i * (args.chunk_size // 2)
        if start_idx + args.chunk_size > len(actions):
            break
        
        max_error, mean_error = test_single_chunk(
            actions, states,
            start_idx, args.chunk_size,
            gripper_indices,
            action_min, action_max,
            verbose=True
        )
        all_max_errors.append(max_error)
        all_mean_errors.append(mean_error)
    
    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"测试的 Chunk 数量: {len(all_max_errors)}")
    print(f"最大误差: {max(all_max_errors):.10f}")
    print(f"平均误差: {np.mean(all_mean_errors):.10f}")
    
    if max(all_max_errors) < 1e-6:
        print(f"\n✅ 所有测试通过! delta_first 数据处理流程正确。")
    else:
        print(f"\n❌ 测试失败! 存在数据处理问题。")


if __name__ == "__main__":
    main()
