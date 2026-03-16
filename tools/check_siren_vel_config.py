#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检测 Siren 速度监督训练配置是否正确
检查项:
1. arm_mode 与 action_dim 是否匹配
2. 数据维度与模型维度是否一致
3. 相机配置是否与 arm_mode 对应
4. 归一化参数是否正确切片
5. 前向传播是否产生 NaN
6. Loss 计算是否正常
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def check_config_consistency(cfg):
    """检查配置一致性"""
    errors = []
    warnings = []
    
    arm_mode = cfg.get("arm_mode", "dual")
    act_dim = cfg.get("act_dim", 14)
    proprio_dims = cfg.get("proprio_dims", 14)
    
    # 检查 arm_mode 与维度
    expected_dim = 7 if arm_mode in ["left", "right"] else 14
    
    logger.info(f"[Config Check] arm_mode: {arm_mode}")
    logger.info(f"[Config Check] act_dim: {act_dim} (expected: {expected_dim})")
    logger.info(f"[Config Check] proprio_dims: {proprio_dims}")
    
    # 注意: 训练脚本会动态修改 act_dim，这里只是警告
    if act_dim != expected_dim:
        warnings.append(
            f"act_dim={act_dim} 与 arm_mode={arm_mode} 不匹配 (期望 {expected_dim})。"
            f"训练脚本会自动修正，但配置文件中的值可能造成混淆。"
        )
    
    # 检查相机配置
    use_cam_high = cfg.get("use_cam_high", True)
    use_cam_left_wrist = cfg.get("use_cam_left_wrist", True)
    use_cam_right_wrist = cfg.get("use_cam_right_wrist", False)
    
    logger.info(f"[Config Check] Cameras: high={use_cam_high}, left={use_cam_left_wrist}, right={use_cam_right_wrist}")
    
    if arm_mode == "left" and use_cam_right_wrist and not use_cam_left_wrist:
        errors.append(f"arm_mode=left 但启用了右腕相机而非左腕相机，这会导致视野不匹配！")
    elif arm_mode == "right" and use_cam_left_wrist and not use_cam_right_wrist:
        errors.append(f"arm_mode=right 但启用了左腕相机而非右腕相机，这会导致视野不匹配！")
    
    # 检查归一化参数
    action_min = cfg.get("action_min_delta_first", None)
    action_max = cfg.get("action_max_delta_first", None)
    
    if action_min is not None and action_max is not None:
        min_len = len(action_min)
        max_len = len(action_max)
        logger.info(f"[Config Check] action_min_delta_first length: {min_len}")
        logger.info(f"[Config Check] action_max_delta_first length: {max_len}")
        
        if min_len != max_len:
            errors.append(f"action_min_delta_first 长度 ({min_len}) 与 action_max_delta_first ({max_len}) 不一致！")
        
        # 检查是否有零范围（会导致除零）
        for i, (mn, mx) in enumerate(zip(action_min, action_max)):
            if abs(mx - mn) < 1e-8:
                warnings.append(f"维度 {i} 的归一化范围几乎为零: min={mn}, max={mx}")
    
    return errors, warnings


def check_data_model_consistency(cfg):
    """检查数据和模型维度一致性"""
    errors = []
    
    logger.info("\n" + "="*60)
    logger.info("Creating DataModule...")
    logger.info("="*60)
    
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup('fit')
    
    logger.info("\n" + "="*60)
    logger.info("Creating Model...")
    logger.info("="*60)
    
    model = hydra.utils.instantiate(cfg.model)
    
    # 获取一个 batch
    train_loader = datamodule.train_dataloader()['lang']
    batch = next(iter(train_loader))
    
    # 检查维度
    actions_shape = batch['actions'].shape
    logger.info(f"\n[Dimension Check] Batch actions shape: {actions_shape}")
    
    if 'joint_vel' in batch:
        vel_shape = batch['joint_vel'].shape
        logger.info(f"[Dimension Check] Batch joint_vel shape: {vel_shape}")
    else:
        vel_shape = None
        logger.warning("[Dimension Check] No joint_vel in batch!")
    
    model_action_dim = model.action_dim
    model_arm_mode = getattr(model, 'arm_mode', 'unknown')
    logger.info(f"[Dimension Check] Model action_dim: {model_action_dim}")
    logger.info(f"[Dimension Check] Model arm_mode: {model_arm_mode}")
    
    # 检查维度是否匹配
    batch_action_dim = actions_shape[-1]
    if batch_action_dim != model_action_dim:
        errors.append(
            f"维度不匹配！数据 action_dim={batch_action_dim}, 模型 action_dim={model_action_dim}。"
            f"这会导致 loss 计算时广播错误，产生 NaN！"
        )
    
    if vel_shape is not None:
        batch_vel_dim = vel_shape[-1]
        if batch_vel_dim != model_action_dim:
            errors.append(
                f"速度维度不匹配！数据 vel_dim={batch_vel_dim}, 模型 action_dim={model_action_dim}。"
            )
    
    return errors, model, datamodule, batch


def check_forward_and_loss(model, batch, device='cuda'):
    """检查前向传播和 loss 计算"""
    errors = []
    warnings = []
    
    logger.info("\n" + "="*60)
    logger.info("Forward & Loss Check...")
    logger.info("="*60)
    
    model = model.to(device)
    model.train()
    
    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    batch[k][kk] = vv.to(device)
    
    # Import forward function
    from beast.models.beast_florence_siren_loss import (
        _siren_forward_with_model_in,
        _d1_dtau_from_actions_and_coords,
        _d1d2d3_dtau_from_actions_and_coords,
    )
    
    model.modality_scope = 'lang'
    
    # 1. Forward pass
    logger.info("\n[Forward] Running Siren forward...")
    try:
        actions_pred, coords_in = _siren_forward_with_model_in(model, batch)
        logger.info(f"[Forward] actions_pred shape: {actions_pred.shape}")
        logger.info(f"[Forward] actions_pred range: [{actions_pred.min().item():.4f}, {actions_pred.max().item():.4f}]")
        
        if torch.isnan(actions_pred).any():
            errors.append("actions_pred 包含 NaN！")
        if torch.isinf(actions_pred).any():
            errors.append("actions_pred 包含 Inf！")
    except Exception as e:
        errors.append(f"Forward pass 失败: {e}")
        return errors, warnings
    
    # 2. Action loss
    actions_gt = batch['actions']
    T = min(model.chunk_size, actions_gt.shape[1])
    
    logger.info(f"\n[Loss] Computing action loss...")
    logger.info(f"[Loss] actions_pred[:, :T, :] shape: {actions_pred[:, :T, :].shape}")
    logger.info(f"[Loss] actions_gt[:, :T, :] shape: {actions_gt[:, :T, :].shape}")
    
    try:
        loss_act = F.mse_loss(actions_pred[:, :T, :], actions_gt[:, :T, :])
        logger.info(f"[Loss] action_loss: {loss_act.item():.6f}")
        
        if torch.isnan(loss_act):
            errors.append("action_loss 是 NaN！")
        if torch.isinf(loss_act):
            errors.append("action_loss 是 Inf！")
    except Exception as e:
        errors.append(f"Action loss 计算失败: {e}")
    
    # 3. Velocity derivatives
    logger.info(f"\n[Derivatives] Computing velocity derivatives...")
    try:
        d1 = _d1_dtau_from_actions_and_coords(
            actions_pred, coords_in, create_graph=True, keep_graph=True
        )
        logger.info(f"[Derivatives] d1 (velocity) range: [{d1.min().item():.4f}, {d1.max().item():.4f}]")
        
        if torch.isnan(d1).any():
            errors.append("一阶导数 d1 包含 NaN！")
        if torch.isinf(d1).any():
            errors.append("一阶导数 d1 包含 Inf！")
    except Exception as e:
        errors.append(f"一阶导数计算失败: {e}")
        d1 = None
    
    # 4. Jerk derivatives (3rd order)
    logger.info(f"\n[Derivatives] Computing jerk derivatives (3rd order)...")
    try:
        d1_, d2, d3 = _d1d2d3_dtau_from_actions_and_coords(
            actions_pred, coords_in, create_graph=True, keep_graph=False
        )
        logger.info(f"[Derivatives] d2 (acceleration) range: [{d2.min().item():.4f}, {d2.max().item():.4f}]")
        logger.info(f"[Derivatives] d3 (jerk) range: [{d3.min().item():.4f}, {d3.max().item():.4f}]")
        
        if torch.isnan(d3).any():
            errors.append("三阶导数 d3 (jerk) 包含 NaN！")
        if torch.isinf(d3).any():
            errors.append("三阶导数 d3 (jerk) 包含 Inf！")
        
        # Jerk 值很大是正常的，但太大可能导致数值问题
        jerk_max = d3.abs().max().item()
        if jerk_max > 1e8:
            warnings.append(f"Jerk 绝对值很大 ({jerk_max:.2e})，可能导致数值不稳定")
    except Exception as e:
        errors.append(f"高阶导数计算失败: {e}")
    
    # 5. Velocity loss (if joint_vel available)
    if 'joint_vel' in batch:
        logger.info(f"\n[Loss] Computing velocity loss...")
        try:
            vel_gt = batch['joint_vel']
            
            # Get scale factors
            vel_cfg = model.vel_cfg
            from beast.models.beast_florence_siren_loss import _get_stats_minmax, _delta_first_scale_s, _scale_tau_to_t
            
            stats = _get_stats_minmax(vel_cfg.delta_first_action_stats)
            if stats is None:
                errors.append("delta_first_action_stats 无法解析！")
            else:
                K = actions_pred.shape[1]
                scale_t = _scale_tau_to_t(fps=float(vel_cfg.fps), K=K)
                s = _delta_first_scale_s(stats, device=d1.device, dtype=d1.dtype)
                
                vel_pred = d1[..., :7] * float(scale_t) * s
                logger.info(f"[Loss] vel_pred range: [{vel_pred.min().item():.4f}, {vel_pred.max().item():.4f}] rad/s")
                logger.info(f"[Loss] vel_gt range: [{vel_gt.min().item():.4f}, {vel_gt.max().item():.4f}] rad/s")
                
                use = min(vel_pred.shape[1], vel_gt.shape[1], T)
                use = max(1, use - 1)
                
                loss_vel = F.mse_loss(vel_pred[:, :use, :], vel_gt[:, :use, :])
                logger.info(f"[Loss] velocity_loss: {loss_vel.item():.6f}")
                
                if torch.isnan(loss_vel):
                    errors.append("velocity_loss 是 NaN！")
                if torch.isinf(loss_vel):
                    errors.append("velocity_loss 是 Inf！")
        except Exception as e:
            errors.append(f"Velocity loss 计算失败: {e}")
    
    # 6. Jerk loss
    logger.info(f"\n[Loss] Computing jerk loss...")
    try:
        if 'd3' in dir() and d3 is not None:
            jerk_squared = d3[:, :T-1, :7].pow(2)
            loss_jerk = jerk_squared.mean()
            logger.info(f"[Loss] jerk_loss (raw): {loss_jerk.item():.2e}")
            
            if torch.isnan(loss_jerk):
                errors.append("jerk_loss 是 NaN！")
            if torch.isinf(loss_jerk):
                errors.append("jerk_loss 是 Inf！")
    except Exception as e:
        errors.append(f"Jerk loss 计算失败: {e}")
    
    return errors, warnings


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("Siren Velocity Supervision Config Checker")
    logger.info("="*60)
    
    # 加载配置
    logger.info("\nLoading configuration...")
    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(Path(__file__).absolute().parents[1] / 'conf'),
        version_base='1.2'
    )
    cfg = compose(config_name='config_aloha_siren_vel')
    
    all_errors = []
    all_warnings = []
    
    # 1. 配置一致性检查
    logger.info("\n" + "="*60)
    logger.info("Step 1: Configuration Consistency Check")
    logger.info("="*60)
    errors, warnings = check_config_consistency(cfg)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    # 2. 数据模型一致性检查
    logger.info("\n" + "="*60)
    logger.info("Step 2: Data-Model Consistency Check")
    logger.info("="*60)
    errors, model, datamodule, batch = check_data_model_consistency(cfg)
    all_errors.extend(errors)
    
    # 如果有严重错误，停止
    if any("维度不匹配" in e for e in all_errors):
        logger.error("\n" + "!"*60)
        logger.error("发现严重维度不匹配错误，无法继续测试！")
        logger.error("!"*60)
        for e in all_errors:
            logger.error(f"  ❌ {e}")
        return 1
    
    # 3. 前向传播和 loss 检查
    logger.info("\n" + "="*60)
    logger.info("Step 3: Forward & Loss Check")
    logger.info("="*60)
    errors, warnings = check_forward_and_loss(model, batch)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    # 输出结果
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    
    if all_warnings:
        logger.warning(f"\n⚠️  Warnings ({len(all_warnings)}):")
        for w in all_warnings:
            logger.warning(f"  ⚠️  {w}")
    
    if all_errors:
        logger.error(f"\n❌ Errors ({len(all_errors)}):")
        for e in all_errors:
            logger.error(f"  ❌ {e}")
        logger.error("\n" + "!"*60)
        logger.error("检测到错误！请修复后再开始训练。")
        logger.error("!"*60)
        return 1
    else:
        logger.info("\n" + "✓"*60)
        logger.info("✅ All checks passed! Configuration is correct.")
        logger.info("✓"*60)
        logger.info("\n可以安全地开始训练:")
        logger.info("  python beast/training_aloha_siren_vel.py")
        return 0


if __name__ == "__main__":
    exit(main())
