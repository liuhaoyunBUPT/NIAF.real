# scripts/test_siren_vel_loss.py
"""
测试速度监督 Siren 模型的各个 loss 数量级
用于确定合理的 loss 权重配置
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    # ========== 配置参数 ==========
    config_path = Path(__file__).parents[1] / "conf"
    data_dir = "/data1/lhy/traindata/pick_pineapple"
    num_batches = 5
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Device: {device}")
    logger.info(f"Data dir: {data_dir}")
    
    # ========== 加载配置 ==========
    logger.info("Loading config...")
    with initialize_config_dir(version_base=None, config_dir=str(config_path.absolute())):
        cfg = compose(config_name="config_aloha_siren_vel")
    
    # 覆盖一些配置用于测试
    cfg.root_data_dir = data_dir
    cfg.batch_size = batch_size
    cfg.datamodule.batch_size = batch_size
    cfg.datamodule.num_workers = 0  # 测试时用单进程
    cfg.datamodule.val_split = 0.0  # 不需要验证集
    
    logger.info(f"fps: {cfg.fps}")
    logger.info(f"act_loss_weight: {cfg.act_loss_weight}")
    logger.info(f"vel_loss_weight: {cfg.vel_loss_weight}")
    logger.info(f"jerk_loss_weight: {cfg.jerk_loss_weight}")
    
    # ========== 根据 arm_mode 调整维度 ==========
    arm_mode = cfg.get("arm_mode", "dual")
    if arm_mode == "left":
        cfg.act_dim = 7
        cfg.proprio_dims = 7
    elif arm_mode == "right":
        cfg.act_dim = 7
        cfg.proprio_dims = 7
    logger.info(f"arm_mode: {arm_mode}, act_dim: {cfg.act_dim}")
    
    # ========== 初始化数据模块 ==========
    logger.info("Initializing datamodule...")
    import hydra
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("fit")
    
    train_loader = datamodule.train_dataloader()["lang"]
    logger.info(f"Train dataset size: {len(datamodule.train_dataset)}")
    
    # ========== 初始化模型 ==========
    logger.info("Initializing model...")
    
    # 手动构建模型参数
    from beast.models.beast_florence_siren_loss import SirenVLAWithVel
    
    # 获取 delta_first_action_stats
    delta_first_action_stats = datamodule.delta_first_action_stats
    logger.info(f"delta_first_action_stats: {delta_first_action_stats}")
    
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    model.eval()
    
    logger.info("Model initialized successfully!")
    
    # ========== 收集 loss 统计 ==========
    act_losses = []
    vel_losses = []
    jerk_losses = []
    
    # 导入必要的函数
    from beast.models.beast_florence_siren_loss import (
        _siren_forward_with_model_in,
        _d1_dtau_from_actions_and_coords,
        _d1d2d3_dtau_from_actions_and_coords,
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {num_batches} batches to estimate loss magnitudes...")
    logger.info(f"{'='*60}\n")
    
    with torch.enable_grad():  # 需要 grad 来计算导数
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            # 移动数据到设备
            dataset_batch = batch
            for k, v in dataset_batch.items():
                if isinstance(v, torch.Tensor):
                    dataset_batch[k] = v.to(device)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            dataset_batch[k][kk] = vv.to(device)
            
            actions_gt = dataset_batch["actions"].to(device)
            joint_vel_gt = dataset_batch["joint_vel"].to(device)
            B, T, act_dim = actions_gt.shape
            
            logger.info(f"Batch {batch_idx + 1}: B={B}, T={T}, act_dim={act_dim}")
            logger.info(f"  actions_gt range: [{actions_gt.min().item():.4f}, {actions_gt.max().item():.4f}]")
            logger.info(f"  joint_vel_gt range: [{joint_vel_gt.min().item():.4f}, {joint_vel_gt.max().item():.4f}] rad/s")
            
            # Siren 前向
            actions_pred, coords_in = _siren_forward_with_model_in(model, dataset_batch)
            
            # 1. 位置 loss
            loss_act = F.mse_loss(actions_pred[:, :T, :], actions_gt[:, :T, :])
            act_losses.append(loss_act.item())
            
            # 2. 速度 loss
            dactions_dtau = _d1_dtau_from_actions_and_coords(
                actions_pred, coords_in,
                create_graph=False, keep_graph=True,
            )
            vel_pred = model._pred_joint_vel_rad_s_from_dactions_dtau(dactions_dtau)
            
            use = min(vel_pred.shape[1], joint_vel_gt.shape[1], T)
            use = max(1, use - 1)
            loss_vel = F.mse_loss(vel_pred[:, :use, :], joint_vel_gt[:, :use, :])
            vel_losses.append(loss_vel.item())
            
            logger.info(f"  vel_pred range: [{vel_pred.min().item():.4f}, {vel_pred.max().item():.4f}] rad/s")
            
            # 3. Jerk loss (正则项)
            _d1, _d2, d3actions_dtau3 = _d1d2d3_dtau_from_actions_and_coords(
                actions_pred, coords_in,
                create_graph=False, keep_graph=False,
            )
            jerk_pred = model._pred_joint_jerk_rad_s3_from_d3actions_dtau3(d3actions_dtau3)
            use_jerk = min(jerk_pred.shape[1], T)
            use_jerk = max(1, use_jerk - 1)
            loss_jerk = torch.mean(jerk_pred[:, :use_jerk, :].pow(2))
            jerk_losses.append(loss_jerk.item())
            
            logger.info(f"  loss_act:  {loss_act.item():.6f}")
            logger.info(f"  loss_vel:  {loss_vel.item():.6f}")
            logger.info(f"  loss_jerk: {loss_jerk.item():.6f}")
            logger.info("")
    
    # ========== 统计分析 ==========
    act_losses = np.array(act_losses)
    vel_losses = np.array(vel_losses)
    jerk_losses = np.array(jerk_losses)
    
    logger.info(f"\n{'='*60}")
    logger.info("Loss Statistics (over {} batches):".format(num_batches))
    logger.info(f"{'='*60}")
    logger.info(f"Action Loss:   mean={act_losses.mean():.6f}, std={act_losses.std():.6f}, "
                f"min={act_losses.min():.6f}, max={act_losses.max():.6f}")
    logger.info(f"Velocity Loss: mean={vel_losses.mean():.6f}, std={vel_losses.std():.6f}, "
                f"min={vel_losses.min():.6f}, max={vel_losses.max():.6f}")
    logger.info(f"Jerk Loss:     mean={jerk_losses.mean():.6f}, std={jerk_losses.std():.6f}, "
                f"min={jerk_losses.min():.6f}, max={jerk_losses.max():.6f}")
    
    # ========== 权重建议 ==========
    logger.info(f"\n{'='*60}")
    logger.info("Weight Recommendations:")
    logger.info(f"{'='*60}")
    
    # 目标：让各个 loss 项对总 loss 的贡献在相近的数量级
    # 以 action loss 为基准
    act_mean = act_losses.mean()
    vel_mean = vel_losses.mean()
    jerk_mean = jerk_losses.mean()
    
    # 计算使各项 loss 贡献相等的权重
    if vel_mean > 0:
        vel_weight_equal = act_mean / vel_mean
    else:
        vel_weight_equal = 0.1
    
    if jerk_mean > 0:
        jerk_weight_equal = act_mean / jerk_mean
    else:
        jerk_weight_equal = 0.01
    
    logger.info(f"\nTo make losses contribute equally to total loss:")
    logger.info(f"  act_loss_weight:  1.0 (baseline)")
    logger.info(f"  vel_loss_weight:  {vel_weight_equal:.4f} (to match action loss magnitude)")
    logger.info(f"  jerk_loss_weight: {jerk_weight_equal:.2e} (to match action loss magnitude)")
    
    # 实际建议配置 - 基于 loss 量级计算
    logger.info(f"\n" + "="*60)
    logger.info("RECOMMENDED CONFIGURATIONS:")
    logger.info("="*60)
    
    # 配置1: 保守配置 - velocity 贡献约 10%, jerk 贡献约 5%
    vel_w1 = 0.1 * act_mean / vel_mean if vel_mean > 0 else 0.01
    jerk_w1 = 0.05 * act_mean / jerk_mean if jerk_mean > 0 else 1e-12
    total1 = act_mean + vel_w1 * vel_mean + jerk_w1 * jerk_mean
    logger.info(f"\n[Conservative] (recommended for initial training)")
    logger.info(f"  act_loss_weight:  1.0")
    logger.info(f"  vel_loss_weight:  {vel_w1:.4f}")
    logger.info(f"  jerk_loss_weight: {jerk_w1:.2e}")
    logger.info(f"  Expected total loss: {total1:.4f}")
    logger.info(f"    - Action: {100*act_mean/total1:.1f}%, Vel: {100*vel_w1*vel_mean/total1:.1f}%, Jerk: {100*jerk_w1*jerk_mean/total1:.1f}%")
    
    # 配置2: 中等配置 - velocity 贡献约 20%, jerk 贡献约 10%
    vel_w2 = 0.2 * act_mean / vel_mean if vel_mean > 0 else 0.02
    jerk_w2 = 0.1 * act_mean / jerk_mean if jerk_mean > 0 else 1e-11
    total2 = act_mean + vel_w2 * vel_mean + jerk_w2 * jerk_mean
    logger.info(f"\n[Moderate] (balanced)")
    logger.info(f"  act_loss_weight:  1.0")
    logger.info(f"  vel_loss_weight:  {vel_w2:.4f}")
    logger.info(f"  jerk_loss_weight: {jerk_w2:.2e}")
    logger.info(f"  Expected total loss: {total2:.4f}")
    logger.info(f"    - Action: {100*act_mean/total2:.1f}%, Vel: {100*vel_w2*vel_mean/total2:.1f}%, Jerk: {100*jerk_w2*jerk_mean/total2:.1f}%")
    
    # 配置3: 强速度监督 - velocity 贡献约 30%, jerk 贡献约 15%
    vel_w3 = 0.3 * act_mean / vel_mean if vel_mean > 0 else 0.03
    jerk_w3 = 0.15 * act_mean / jerk_mean if jerk_mean > 0 else 1e-11
    total3 = act_mean + vel_w3 * vel_mean + jerk_w3 * jerk_mean
    logger.info(f"\n[Strong Velocity] (emphasize smooth velocity)")
    logger.info(f"  act_loss_weight:  1.0")
    logger.info(f"  vel_loss_weight:  {vel_w3:.4f}")
    logger.info(f"  jerk_loss_weight: {jerk_w3:.2e}")
    logger.info(f"  Expected total loss: {total3:.4f}")
    logger.info(f"    - Action: {100*act_mean/total3:.1f}%, Vel: {100*vel_w3*vel_mean/total3:.1f}%, Jerk: {100*jerk_w3*jerk_mean/total3:.1f}%")
    
    logger.info(f"\n{'='*60}")
    logger.info("Test completed successfully!")
    logger.info(f"{'='*60}")
    
    return {
        "act_loss_mean": act_mean,
        "vel_loss_mean": vel_mean,
        "jerk_loss_mean": jerk_mean,
        "suggested_weights": {
            "act_loss_weight": 1.0,
            "vel_loss_weight": vel_w1,  # Conservative
            "jerk_loss_weight": jerk_w1,
        }
    }


if __name__ == "__main__":
    main()
