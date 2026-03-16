#!/usr/bin/env python3
"""
分析HDF5文件中各臂静止不动的数据分布
统计有多少数据的动作变化量接近0（静止状态）

================================================================================
【静止/运动 判定逻辑说明】
================================================================================

1. 计算 Delta Action（动作变化量）:
   ---------------------------------------------------------------
   delta_action[t] = action[t] - action[t-1]
   
   - 对于 t=0（首帧）: delta_action[0] = action[0] - state[0]
   - 对于 t>0: delta_action[t] = action[t] - action[t-1]
   
   delta_action 表示相邻两帧之间的动作变化量（弧度）

2. 判定单个维度静止/运动:
   ---------------------------------------------------------------
   if |delta_action[t, dim]| < threshold:
       该维度在 t 时刻是【静止】的
   else:
       该维度在 t 时刻是【运动】的
   
   例如 threshold=0.01 (约0.57度):
   - |delta| = 0.005 < 0.01 → 静止 (变化量太小，视为没动)
   - |delta| = 0.02 > 0.01  → 运动 (有明显变化)

3. 判定整个臂静止/运动:
   ---------------------------------------------------------------
   【左臂完全静止】: 6个关节(dim 0-5)全部满足 |delta| < threshold
   【左臂有运动】:   6个关节中至少1个满足 |delta| >= threshold
   
   代码实现:
   left_static_all = np.all(|delta[:, 0:6]| < threshold, axis=1)  # 全部静止
   left_moving_any = ~left_static_all  # 任一运动 = 非全部静止

4. 阈值参考:
   ---------------------------------------------------------------
   - 0.001 rad ≈ 0.057° (非常严格，微小抖动也算运动)
   - 0.005 rad ≈ 0.29°  (较严格)
   - 0.01 rad  ≈ 0.57°  (常用，约半度)
   - 0.05 rad  ≈ 2.86°  (宽松，只有明显运动才算)

================================================================================
"""

import h5py
import numpy as np
import glob
import os
from collections import defaultdict


def analyze_static_distribution(data_dir: str, thresholds: list = [0.001, 0.005, 0.01, 0.05]):
    """
    分析各臂静止不动的数据分布
    
    Args:
        data_dir: HDF5文件所在目录
        thresholds: 判断静止的阈值列表（delta绝对值小于此值视为静止）
    """
    files = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
    print(f'Found {len(files)} HDF5 files in {data_dir}')
    
    if not files:
        print("No HDF5 files found!")
        return
    
    # 收集所有delta动作数据
    all_delta_actions = []
    total_frames = 0
    
    for f_path in files:
        with h5py.File(f_path, 'r') as f:
            if 'action' not in f:
                continue
                
            actions = f['action'][:].astype(np.float32)
            total_frames += actions.shape[0]
            
            # 计算delta action
            if 'observations/qpos' in f:
                states = f['observations/qpos'][:].astype(np.float32)
            else:
                states = None
            
            if len(actions) < 1:
                continue
            
            if states is not None:
                if len(actions) == 1:
                    delta_actions = actions - states
                else:
                    prev_actions = np.concatenate([states[0:1], actions[:-1]], axis=0)
                    delta_actions = actions - prev_actions
                
                # 夹爪维度保持绝对值（这里我们分析时也考虑夹爪的变化）
                # 但为了分析静止状态，我们需要计算夹爪的delta
                gripper_delta = np.zeros_like(delta_actions[:, [6, 13]])
                if len(actions) > 1:
                    gripper_delta[1:] = actions[1:, [6, 13]] - actions[:-1, [6, 13]]
                    gripper_delta[0] = actions[0, [6, 13]] - states[0, [6, 13]]
                else:
                    gripper_delta[0] = actions[0, [6, 13]] - states[0, [6, 13]]
                
                # 用夹爪delta替换
                delta_actions[:, 6] = gripper_delta[:, 0]
                delta_actions[:, 13] = gripper_delta[:, 1]
                
                all_delta_actions.append(delta_actions)
    
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)
    
    dim_names = [
        '左臂关节0', '左臂关节1', '左臂关节2', '左臂关节3', '左臂关节4', '左臂关节5', '左夹爪',
        '右臂关节0', '右臂关节1', '右臂关节2', '右臂关节3', '右臂关节4', '右臂关节5', '右夹爪'
    ]
    
    # ==================== 基本统计 ====================
    print(f'\n{"="*80}')
    print(f'【Delta Action 基本统计】')
    print(f'{"="*80}')
    print(f'总帧数: {total_frames}')
    print(f'动作维度: {all_delta_actions.shape[1]}')
    
    print(f'\n各维度 delta 统计:')
    print(f'{"维度":^6} {"名称":^12} {"均值":^12} {"标准差":^12} {"最小值":^12} {"最大值":^12}')
    print('-' * 78)
    
    for i in range(all_delta_actions.shape[1]):
        data = all_delta_actions[:, i]
        name = dim_names[i] if i < len(dim_names) else f'维度{i}'
        print(f'{i:^6} {name:^12} {data.mean():+12.6f} {data.std():12.6f} {data.min():+12.6f} {data.max():+12.6f}')
    
    # ==================== 静止状态分析 ====================
    print(f'\n{"="*80}')
    print(f'【静止状态分析】(|delta| < threshold 视为静止)')
    print(f'{"="*80}')
    
    # 按不同阈值统计
    for threshold in thresholds:
        print(f'\n--- 阈值: {threshold} rad (≈{np.degrees(threshold):.2f}°) ---')
        print(f'判定规则: |delta_action| < {threshold} 视为静止')
        print(f'{"维度":^6} {"名称":^12} {"静止帧数":^12} {"静止比例":^12} {"运动帧数":^12} {"运动比例":^12}')
        print('-' * 78)
        
        for i in range(all_delta_actions.shape[1]):
            data = all_delta_actions[:, i]
            static_mask = np.abs(data) < threshold
            static_count = static_mask.sum()
            static_ratio = static_count / len(data) * 100
            moving_count = len(data) - static_count
            moving_ratio = 100 - static_ratio
            name = dim_names[i] if i < len(dim_names) else f'维度{i}'
            print(f'{i:^6} {name:^12} {static_count:^12} {static_ratio:^11.1f}% {moving_count:^12} {moving_ratio:^11.1f}%')
    
    # ==================== 左臂 vs 右臂对比 ====================
    print(f'\n{"="*80}')
    print(f'【左臂 vs 右臂 静止状态对比】')
    print(f'{"="*80}')
    
    left_arm_indices = list(range(6))   # 左臂关节 0-5
    right_arm_indices = list(range(7, 13))  # 右臂关节 7-12
    
    for threshold in thresholds:
        print(f'\n--- 阈值: {threshold} rad (≈{np.degrees(threshold):.2f}°) ---')
        print(f'【判定规则】')
        print(f'  单维度: |delta_action[t, dim]| < {threshold} → 静止')
        print(f'  左臂整体: 6个关节(dim0-5)全部静止 → 左臂静止')
        print(f'  右臂整体: 6个关节(dim7-12)全部静止 → 右臂静止')
        
        # ========== 判定公式 ==========
        # 左臂：所有6个关节都静止才算静止
        # left_static_all[t] = (|delta[t,0]|<th) AND (|delta[t,1]|<th) AND ... AND (|delta[t,5]|<th)
        left_static_all = np.all(np.abs(all_delta_actions[:, left_arm_indices]) < threshold, axis=1)
        # 左臂：任意一个关节运动就算运动
        # left_moving_any = NOT(left_static_all)
        left_moving_any = ~left_static_all
        
        # 右臂：所有6个关节都静止才算静止
        right_static_all = np.all(np.abs(all_delta_actions[:, right_arm_indices]) < threshold, axis=1)
        # 右臂：任意一个关节运动就算运动
        right_moving_any = ~right_static_all
        
        print(f'\n  左臂 (dim 0-5):')
        print(f'    完全静止帧数: {left_static_all.sum():>8} ({left_static_all.mean()*100:>6.2f}%)')
        print(f'    有运动帧数:   {left_moving_any.sum():>8} ({left_moving_any.mean()*100:>6.2f}%)')
        
        print(f'\n  右臂 (dim 7-12):')
        print(f'    完全静止帧数: {right_static_all.sum():>8} ({right_static_all.mean()*100:>6.2f}%)')
        print(f'    有运动帧数:   {right_moving_any.sum():>8} ({right_moving_any.mean()*100:>6.2f}%)')
        
        # 组合情况
        both_static = left_static_all & right_static_all
        both_moving = left_moving_any & right_moving_any
        left_only_moving = left_moving_any & right_static_all
        right_only_moving = left_static_all & right_moving_any
        
        print(f'\n  组合情况:')
        print(f'    双臂都静止:     {both_static.sum():>8} ({both_static.mean()*100:>6.2f}%)')
        print(f'    双臂都运动:     {both_moving.sum():>8} ({both_moving.mean()*100:>6.2f}%)')
        print(f'    仅左臂运动:     {left_only_moving.sum():>8} ({left_only_moving.mean()*100:>6.2f}%)')
        print(f'    仅右臂运动:     {right_only_moving.sum():>8} ({right_only_moving.mean()*100:>6.2f}%)')
    
    # ==================== 运动幅度分布 ====================
    print(f'\n{"="*80}')
    print(f'【运动幅度分布 (|delta| 的分布)】')
    print(f'{"="*80}')
    
    # 计算各维度的运动幅度分位数
    percentiles = [50, 75, 90, 95, 99]
    
    print(f'\n各维度 |delta| 分位数:')
    header = f'{"维度":^6} {"名称":^12} ' + ' '.join([f'{p}%'.center(10) for p in percentiles])
    print(header)
    print('-' * (20 + 11 * len(percentiles)))
    
    for i in range(all_delta_actions.shape[1]):
        abs_delta = np.abs(all_delta_actions[:, i])
        pct_values = [np.percentile(abs_delta, p) for p in percentiles]
        name = dim_names[i] if i < len(dim_names) else f'维度{i}'
        values_str = ' '.join([f'{v:10.6f}' for v in pct_values])
        print(f'{i:^6} {name:^12} {values_str}')
    
    # ==================== 左右臂运动量对比 ====================
    print(f'\n{"="*80}')
    print(f'【左臂 vs 右臂 运动量对比】')
    print(f'{"="*80}')
    
    # 计算每帧的L2范数运动量
    left_motion = np.linalg.norm(all_delta_actions[:, left_arm_indices], axis=1)
    right_motion = np.linalg.norm(all_delta_actions[:, right_arm_indices], axis=1)
    
    print(f'\n每帧 L2 范数运动量统计:')
    print(f'  左臂: mean={left_motion.mean():.6f}, std={left_motion.std():.6f}, max={left_motion.max():.6f}')
    print(f'  右臂: mean={right_motion.mean():.6f}, std={right_motion.std():.6f}, max={right_motion.max():.6f}')
    print(f'  右臂/左臂 比例: {right_motion.mean() / (left_motion.mean() + 1e-10):.2f}x')
    
    # L2范数的分位数
    print(f'\n每帧 L2 范数分位数:')
    print(f'  {"":^8} ' + ' '.join([f'{p}%'.center(10) for p in percentiles]))
    left_pct = [np.percentile(left_motion, p) for p in percentiles]
    right_pct = [np.percentile(right_motion, p) for p in percentiles]
    print(f'  {"左臂":^8} ' + ' '.join([f'{v:10.6f}' for v in left_pct]))
    print(f'  {"右臂":^8} ' + ' '.join([f'{v:10.6f}' for v in right_pct]))
    
    # ==================== 按Episode分析 ====================
    print(f'\n{"="*80}')
    print(f'【按Episode分析静止比例】(阈值=0.01)')
    print(f'{"="*80}')
    
    threshold = 0.001
    episode_stats = []
    
    for f_path in files:
        with h5py.File(f_path, 'r') as f:
            if 'action' not in f or 'observations/qpos' not in f:
                continue
                
            actions = f['action'][:].astype(np.float32)
            states = f['observations/qpos'][:].astype(np.float32)
            
            if len(actions) <= 1:
                continue
            
            prev_actions = np.concatenate([states[0:1], actions[:-1]], axis=0)
            delta_actions = actions - prev_actions
            
            # 计算左右臂静止比例
            left_static = np.all(np.abs(delta_actions[:, :6]) < threshold, axis=1).mean()
            right_static = np.all(np.abs(delta_actions[:, 7:13]) < threshold, axis=1).mean()
            
            episode_stats.append({
                'file': os.path.basename(f_path),
                'frames': len(actions),
                'left_static': left_static,
                'right_static': right_static
            })
    
    print(f'\n{"Episode":<30} {"帧数":>8} {"左臂静止%":>12} {"右臂静止%":>12} {"差异":>10}')
    print('-' * 75)
    
    for stat in episode_stats[:20]:  # 只显示前20个
        diff = stat['left_static'] - stat['right_static']
        print(f'{stat["file"]:<30} {stat["frames"]:>8} {stat["left_static"]*100:>11.1f}% {stat["right_static"]*100:>11.1f}% {diff*100:>+9.1f}%')
    
    if len(episode_stats) > 20:
        print(f'... (还有 {len(episode_stats) - 20} 个episode)')
    
    # 汇总
    avg_left_static = np.mean([s['left_static'] for s in episode_stats])
    avg_right_static = np.mean([s['right_static'] for s in episode_stats])
    print(f'\n平均值: 左臂静止 {avg_left_static*100:.1f}%, 右臂静止 {avg_right_static*100:.1f}%')
    
    return all_delta_actions


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='分析HDF5数据中各臂静止不动的数据分布')
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='/home/lhy/act/data/pickcorn',
        help='HDF5文件目录 (default: /home/lhy/act/data/pickcorn)'
    )
    parser.add_argument(
        '--thresholds', '-t',
        type=float,
        nargs='+',
        default=[0.001, 0.005, 0.01, 0.05],
        help='判断静止的阈值列表 (default: 0.001 0.005 0.01 0.05)'
    )
    
    args = parser.parse_args()
    analyze_static_distribution(args.data_dir, thresholds=args.thresholds)


# 使用示例:
# python analyze_static.py -d /home/lhy/act/data/pickcorn
# python analyze_static.py -d /home/lhy/act/data/pickcorn -t 0.001 0.01 0.1
