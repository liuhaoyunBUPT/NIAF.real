#!/usr/bin/env python
"""
调试脚本：检查图像和action数据大小是否合理
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np


def print_separator(title: str = ""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


def analyze_batch(batch: dict, batch_idx: int = 0):
    """分析一个batch的数据"""
    print_separator(f"Batch {batch_idx} 数据分析")
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"\n[{key}]")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Device: {value.device}")
            print(f"  Min: {value.min().item():.4f}")
            print(f"  Max: {value.max().item():.4f}")
            print(f"  Mean: {value.float().mean().item():.4f}")
            print(f"  Std: {value.float().std().item():.4f}")
            
            # 检查是否有 NaN 或 Inf
            if torch.isnan(value).any():
                print(f"  ⚠️  包含 NaN 值!")
            if torch.isinf(value).any():
                print(f"  ⚠️  包含 Inf 值!")
                
        elif isinstance(value, str):
            print(f"\n[{key}]: {value[:100]}..." if len(value) > 100 else f"\n[{key}]: {value}")
        elif isinstance(value, list):
            print(f"\n[{key}]: List of {len(value)} items")
            if len(value) > 0 and isinstance(value[0], str):
                print(f"  First item: {value[0][:50]}...")
        else:
            print(f"\n[{key}]: {type(value)}")


def visualize_images(batch: dict, save_path: str = "debug_images.png"):
    """可视化batch中的图像"""
    image_keys = [k for k in batch.keys() if 'image' in k.lower() or 'rgb' in k.lower() or 'img' in k.lower()]
    
    if not image_keys:
        print("没有找到图像数据")
        return
    
    print_separator("图像可视化")
    
    for key in image_keys:
        img = batch[key]
        if isinstance(img, torch.Tensor):
            print(f"\n正在可视化 [{key}]...")
            
            # 取第一个样本
            if img.dim() == 5:  # [B, T, C, H, W]
                img = img[0, 0]  # 取第一个batch的第一帧
            elif img.dim() == 4:  # [B, C, H, W]
                img = img[0]
            
            # 转换为 numpy
            if img.shape[0] in [1, 3]:  # CHW
                img = img.permute(1, 2, 0).cpu().numpy()
            else:
                img = img.cpu().numpy()
            
            # 归一化到 [0, 1]
            if img.max() > 1:
                img = img / 255.0
            img = np.clip(img, 0, 1)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"{key}\nShape: {batch[key].shape}")
            plt.axis('off')
            
            save_file = f"debug_{key}.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"  保存到: {save_file}")
            plt.close()


def visualize_actions(batch: dict, save_path: str = "debug_actions.png"):
    """可视化action数据"""
    action_keys = [k for k in batch.keys() if 'action' in k.lower() or 'act' in k.lower()]
    
    if not action_keys:
        print("没有找到动作数据")
        return
    
    print_separator("动作数据可视化")
    
    for key in action_keys:
        action = batch[key]
        if isinstance(action, torch.Tensor):
            print(f"\n正在可视化 [{key}]...")
            
            # 取第一个样本
            if action.dim() == 3:  # [B, T, A]
                action = action[0].cpu().numpy()  # [T, A]
            elif action.dim() == 2:  # [B, A] 或 [T, A]
                action = action.cpu().numpy()
            else:
                action = action.cpu().numpy()
            
            plt.figure(figsize=(12, 6))
            
            if action.ndim == 2:
                for i in range(min(action.shape[1], 14)):  # 最多显示14个维度
                    plt.plot(action[:, i], label=f'dim_{i}')
                plt.xlabel('Time Step')
                plt.ylabel('Action Value')
                plt.title(f"{key}\nShape: {batch[key].shape}")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.bar(range(len(action)), action)
                plt.xlabel('Dimension')
                plt.ylabel('Value')
                plt.title(f"{key}\nShape: {batch[key].shape}")
            
            plt.tight_layout()
            save_file = f"debug_{key}.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"  保存到: {save_file}")
            plt.close()


def check_data_reasonableness(batch: dict):
    """检查数据是否合理"""
    print_separator("数据合理性检查")
    
    issues = []
    
    # 检查图像
    for key in batch.keys():
        if 'image' in key.lower() or 'rgb' in key.lower():
            img = batch[key]
            if isinstance(img, torch.Tensor):
                if img.max() > 255:
                    issues.append(f"[{key}] 图像值超过255: max={img.max().item()}")
                if img.min() < 0:
                    issues.append(f"[{key}] 图像值为负: min={img.min().item()}")
                if img.shape[-2] < 100 or img.shape[-1] < 100:
                    issues.append(f"[{key}] 图像尺寸过小: {img.shape}")
    
    # 检查动作
    for key in batch.keys():
        if 'action' in key.lower():
            action = batch[key]
            if isinstance(action, torch.Tensor):
                if action.abs().max() > 10:
                    issues.append(f"[{key}] 动作值可能过大: max_abs={action.abs().max().item():.4f}")
                if torch.isnan(action).any():
                    issues.append(f"[{key}] 动作包含NaN")
    
    if issues:
        print("\n⚠️  发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 数据看起来合理!")
    
    return issues


@hydra.main(version_base=None, config_path="../conf", config_name="config_aloha_siren")
def main(cfg: DictConfig):
    """主函数"""
    print_separator("ALOHA Siren 数据调试脚本")
    
    # 打印配置
    print("\n关键配置:")
    print(f"  数据路径: {cfg.root_data_dir}")
    print(f"  Batch Size: {cfg.batch_size}")
    print(f"  动作维度: {cfg.act_dim}")
    print(f"  本体感知维度: {cfg.proprio_dims}")
    print(f"  动作序列长度: {cfg.chunk_size}")
    
    # 初始化数据模块
    print_separator("初始化数据模块")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup('fit')
    
    # 获取训练数据加载器
    train_loader_dict = datamodule.train_dataloader()
    
    # train_dataloader 返回的是字典 {"lang": DataLoader(...)}
    if isinstance(train_loader_dict, dict):
        print(f"\nDataLoader keys: {list(train_loader_dict.keys())}")
        train_loader = train_loader_dict["lang"]
    else:
        train_loader = train_loader_dict
    
    print(f"\n训练集大小: {len(datamodule.train_dataset)}")
    print(f"Batch数量: {len(train_loader)}")
    
    # 获取一个batch
    print_separator("加载第一个Batch")
    batch = next(iter(train_loader))
    
    # 分析batch
    analyze_batch(batch, 0)
    
    # 可视化
    visualize_images(batch)
    visualize_actions(batch)
    
    # 检查合理性
    check_data_reasonableness(batch)
    
    print_separator("调试完成")
    print("\n你可以在以下位置设置断点进行更详细的调试:")
    print("  1. datamodule 的 __getitem__ 方法")
    print("  2. model 的 forward 方法")
    print("  3. model 的 training_step 方法")
    
    # 在这里设置断点可以交互式检查数据
    breakpoint()  # <-- 在这里设置断点！


if __name__ == "__main__":
    main()
