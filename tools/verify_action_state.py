#!/usr/bin/env python3
"""
验证 ALOHA 数据集中 action 和 qpos 的关系
检查是否满足 a_t = s_{t+1} (t时刻的动作是t+1时刻的状态)
"""
import h5py
import numpy as np
import glob
import os
import argparse


def verify_episode(hdf5_path: str, verbose: bool = False):
    """验证单个episode"""
    with h5py.File(hdf5_path, 'r') as f:
        qpos = f['observations/qpos'][:]
        action = f['action'][:]
        
        print(f"\n{'='*60}")
        print(f"文件: {os.path.basename(hdf5_path)}")
        print(f"qpos shape: {qpos.shape}")
        print(f"action shape: {action.shape}")
        
        # 检查长度
        T = min(len(action), len(qpos) - 1)
        
        # 计算 action[t] 和 qpos[t+1] 的差异
        diffs = []
        for t in range(T):
            diff = np.abs(action[t] - qpos[t + 1])
            diffs.append(diff)
            
            if verbose and t < 10:
                print(f"  t={t}: max|a[t] - s[t+1]| = {diff.max():.6f}, mean = {diff.mean():.6f}")
        
        diffs = np.array(diffs)
        max_diff = diffs.max()
        mean_diff = diffs.mean()
        
        # 同时检查 action[t] 和 qpos[t] 的差异 (不符合约定的情况)
        diffs_same = []
        for t in range(T):
            diff_same = np.abs(action[t] - qpos[t])
            diffs_same.append(diff_same)
        
        diffs_same = np.array(diffs_same)
        max_diff_same = diffs_same.max()
        mean_diff_same = diffs_same.mean()
        
        print(f"\n对比分析:")
        print(f"  a[t] vs s[t+1] (期望的约定): max={max_diff:.6f}, mean={mean_diff:.6f}")
        print(f"  a[t] vs s[t]   (同一时刻):   max={max_diff_same:.6f}, mean={mean_diff_same:.6f}")
        
        if max_diff < 0.01:
            print(f"  ✅ 数据符合 a_t = s_{{t+1}} 的约定")
            return True
        elif max_diff_same < 0.01:
            print(f"  ⚠️ 数据使用 a_t = s_t 的约定 (动作和状态同一时刻)")
            return False
        else:
            print(f"  ❓ 数据约定不明确，需要进一步分析")
            return None


def main():
    parser = argparse.ArgumentParser(description="验证ALOHA数据集的动作-状态关系")
    parser.add_argument("--data_dir", type=str, default="/home/lhy/act/data/picksoft",
                        help="数据目录路径")
    parser.add_argument("--max_episodes", type=int, default=3,
                        help="最多验证的episode数量")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细信息")
    args = parser.parse_args()
    
    # 查找所有episode文件
    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.hdf5")))
    
    if not hdf5_files:
        print(f"未找到HDF5文件: {args.data_dir}/episode_*.hdf5")
        return
    
    print(f"找到 {len(hdf5_files)} 个episode文件")
    print(f"验证前 {min(args.max_episodes, len(hdf5_files))} 个...")
    
    results = []
    for hdf5_path in hdf5_files[:args.max_episodes]:
        result = verify_episode(hdf5_path, verbose=args.verbose)
        results.append(result)
    
    # 总结
    print(f"\n{'='*60}")
    print("总结:")
    符合约定 = sum(1 for r in results if r is True)
    不符合约定 = sum(1 for r in results if r is False)
    不明确 = sum(1 for r in results if r is None)
    
    print(f"  符合 a_t=s_{{t+1}}: {符合约定}/{len(results)}")
    print(f"  使用 a_t=s_t:      {不符合约定}/{len(results)}")
    print(f"  不明确:           {不明确}/{len(results)}")
    
    if 符合约定 == len(results):
        print("\n✅ 所有数据都符合 a_t = s_{t+1} 的约定，当前代码是正确的！")
    elif 不符合约定 == len(results):
        print("\n⚠️ 数据使用 a_t = s_t 的约定，需要修改数据集代码！")
        print("   建议: 将 action 改为使用 qpos[t+1:] 作为目标")


if __name__ == "__main__":
    main()
