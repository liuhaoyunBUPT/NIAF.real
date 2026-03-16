#!/usr/bin/env python3
"""
SIREN 调制参数分析脚本

用途：
    1. 统计 base_params 的数值分布
    2. 跑一次 forward 获取 contextualized_wtokens
    3. 统计 delta_b_raw 和 u_raw 的分布
    4. 分析调制后参数与调制前参数的比较
    5. 提供 bias_mod_scale 设置建议

使用方法：
    python analyze_siren_modulation.py --ckpt_path <path_to_checkpoint> --config_name config_calvin_siren

作者：GitHub Copilot
"""

import os
import sys
from pathlib import Path
import argparse
from collections import OrderedDict
from typing import Dict, Tuple
import numpy as np

import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

import src.models.niaf as models_m
from src.models.base import create_bidirectional_mask


def load_model_and_data(ckpt_path: str, config_path: str):
    """加载模型和数据"""
    from omegaconf import OmegaConf
    
    print(f"[INFO] 加载配置文件: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # 解析 defaults 中引用的子配置文件
    config_dir = Path(config_path).parent
    
    # 加载 datamodule 配置
    if "datamodule" in cfg and isinstance(cfg.datamodule, str):
        dm_path = config_dir / "datamodule" / f"{cfg.datamodule}.yaml"
        if dm_path.exists():
            dm_cfg = OmegaConf.load(dm_path)
            cfg.datamodule = dm_cfg
    
    # 加载 model 配置
    if "model" in cfg and isinstance(cfg.model, str):
        model_path = config_dir / "model" / f"{cfg.model}.yaml"
        if model_path.exists():
            model_cfg = OmegaConf.load(model_path)
            cfg.model = model_cfg
    
    print(f"[INFO] 加载检查点: {ckpt_path}")
    
    # 从检查点直接加载模型
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # 获取模型类
    if "hyper_parameters" in checkpoint and "model" in checkpoint["hyper_parameters"]:
        model_cfg = checkpoint["hyper_parameters"]["model"]
        target = model_cfg.get("_target_", "src.models.niaf.NIAF")
    else:
        target = "src.models.niaf.NIAF"
    
    model_class_name = target.split(".")[-1]
    model_class = getattr(models_m, model_class_name)
    model = model_class.load_from_checkpoint(ckpt_path, map_location='cpu')
    model.eval()
    
    # 加载数据模块 - 使用 Hydra 初始化
    with hydra.initialize(version_base=None, config_path="../conf"):
        # 根据配置文件名确定使用哪个配置
        config_name = Path(config_path).stem
        full_cfg = hydra.compose(config_name=config_name)
    
    datamodule = hydra.utils.instantiate(full_cfg.datamodule)
    datamodule.setup(stage="fit")
    
    return model, datamodule, full_cfg


def analyze_base_params(model) -> Dict[str, Dict]:
    """
    分析 base_params 的数值分布
    
    返回：
        各层 base_params 的统计信息
    """
    print("\n" + "="*80)
    print("📊 Base Parameters 统计分析")
    print("="*80)
    
    stats = {}
    
    for name, param in model.base_params.items():
        data = param.detach().cpu().numpy()
        
        layer_stats = {
            "shape": data.shape,
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "abs_mean": float(np.mean(np.abs(data))),
            "abs_max": float(np.max(np.abs(data))),
            "percentiles": {
                "1%": float(np.percentile(data, 1)),
                "5%": float(np.percentile(data, 5)),
                "25%": float(np.percentile(data, 25)),
                "50%": float(np.percentile(data, 50)),
                "75%": float(np.percentile(data, 75)),
                "95%": float(np.percentile(data, 95)),
                "99%": float(np.percentile(data, 99)),
            }
        }
        stats[name] = layer_stats
        
        param_type = "bias" if "bias" in name else "weight"
        print(f"\n┌─ {name} ({param_type})")
        print(f"│  Shape: {data.shape}")
        print(f"│  Mean: {layer_stats['mean']:.6f}, Std: {layer_stats['std']:.6f}")
        print(f"│  Min: {layer_stats['min']:.6f}, Max: {layer_stats['max']:.6f}")
        print(f"│  |Abs| Mean: {layer_stats['abs_mean']:.6f}, |Abs| Max: {layer_stats['abs_max']:.6f}")
        print(f"└  Percentiles: 5%={layer_stats['percentiles']['5%']:.4f}, "
              f"50%={layer_stats['percentiles']['50%']:.4f}, "
              f"95%={layer_stats['percentiles']['95%']:.4f}")
    
    return stats


@torch.no_grad()
def analyze_modulation_forward(model, batch: Dict) -> Dict[str, Dict]:
    """
    进行一次 forward，分析调制参数的分布
    
    返回：
        delta_b_raw, u_raw, 调制后参数等统计信息
    """
    print("\n" + "="*80)
    print("🔄 Forward Pass 调制参数分析")
    print("="*80)
    
    device = next(model.parameters()).device
    default_dtype = next(model.parameters()).dtype
    
    # 1. Encoder 编码多模态输入特征
    features, encoder_attn_mask = model.compute_input_features(batch)
    B = features.size(0)
    
    print(f"\n[INFO] Batch size: {B}")
    print(f"[INFO] Features shape: {features.shape}")
    
    # 2. Decoder 处理参数查询 token
    decoder_input_embeds = model.param_queries.expand(B, -1, -1)
    decoder_outputs = model.vlm.get_decoder()(
        inputs_embeds=decoder_input_embeds,
        encoder_hidden_states=features,
        encoder_attention_mask=encoder_attn_mask,
        attention_mask=create_bidirectional_mask(B, model.param_queries.size(1), device),
    )
    contextualized_wtokens = decoder_outputs[0]
    
    print(f"[INFO] Contextualized tokens shape: {contextualized_wtokens.shape}")
    
    # 3. 统计 contextualized_wtokens
    ctx_data = contextualized_wtokens.detach().cpu().numpy()
    print(f"\n┌─ Contextualized Tokens 统计")
    print(f"│  Mean: {np.mean(ctx_data):.6f}, Std: {np.std(ctx_data):.6f}")
    print(f"│  Min: {np.min(ctx_data):.6f}, Max: {np.max(ctx_data):.6f}")
    print(f"└  |Abs| Mean: {np.mean(np.abs(ctx_data)):.6f}")
    
    # 4. 分析各层调制参数
    modulation_stats = {
        "bias_modulation": {},
        "weight_modulation": {},
        "contextualized_tokens": {
            "mean": float(np.mean(ctx_data)),
            "std": float(np.std(ctx_data)),
            "min": float(np.min(ctx_data)),
            "max": float(np.max(ctx_data)),
        }
    }
    
    print("\n" + "-"*80)
    print("📐 Bias 调制分析 (加性调制: b = b_base + scale * delta_b_raw)")
    print("-"*80)
    
    for name, shape in model.hypo_param_shapes.items():
        if "bias" in name:
            start, end = model.btoken_rng[name]
            token_slice = contextualized_wtokens[:, start:end, :]
            delta_b_raw = model.btoken_postfc[name](token_slice).squeeze(1)
            
            delta_b_np = delta_b_raw.detach().cpu().numpy()
            b_base = model.base_params[name].detach().cpu().numpy()
            b_modulated = b_base + model.bias_mod_scale * delta_b_np
            
            # 计算调制的影响
            modulation_magnitude = model.bias_mod_scale * np.abs(delta_b_np)
            relative_change = modulation_magnitude / (np.abs(b_base) + 1e-8)
            
            layer_stats = {
                "delta_b_raw": {
                    "mean": float(np.mean(delta_b_np)),
                    "std": float(np.std(delta_b_np)),
                    "min": float(np.min(delta_b_np)),
                    "max": float(np.max(delta_b_np)),
                    "abs_mean": float(np.mean(np.abs(delta_b_np))),
                },
                "b_base": {
                    "mean": float(np.mean(b_base)),
                    "abs_mean": float(np.mean(np.abs(b_base))),
                },
                "b_modulated": {
                    "mean": float(np.mean(b_modulated)),
                    "abs_mean": float(np.mean(np.abs(b_modulated))),
                },
                "modulation_magnitude": {
                    "mean": float(np.mean(modulation_magnitude)),
                    "max": float(np.max(modulation_magnitude)),
                },
                "relative_change": {
                    "mean": float(np.mean(relative_change)),
                    "max": float(np.max(relative_change)),
                },
            }
            modulation_stats["bias_modulation"][name] = layer_stats
            
            print(f"\n┌─ {name}")
            print(f"│  delta_b_raw (经 Tanh，范围 [-1, 1]):")
            print(f"│    Mean: {layer_stats['delta_b_raw']['mean']:.6f}, "
                  f"Std: {layer_stats['delta_b_raw']['std']:.6f}")
            print(f"│    Min: {layer_stats['delta_b_raw']['min']:.6f}, "
                  f"Max: {layer_stats['delta_b_raw']['max']:.6f}")
            print(f"│    |Abs| Mean: {layer_stats['delta_b_raw']['abs_mean']:.6f}")
            print(f"│  ")
            print(f"│  调制后 (scale={model.bias_mod_scale}):")
            print(f"│    调制量 = scale * |delta_b_raw|")
            print(f"│    Mean 调制量: {layer_stats['modulation_magnitude']['mean']:.6f}")
            print(f"│    Max 调制量: {layer_stats['modulation_magnitude']['max']:.6f}")
            print(f"│  ")
            print(f"│  与 base bias 比较:")
            print(f"│    |b_base| Mean: {layer_stats['b_base']['abs_mean']:.6f}")
            print(f"│    相对变化 Mean: {layer_stats['relative_change']['mean']*100:.2f}%")
            print(f"└    相对变化 Max: {layer_stats['relative_change']['max']*100:.2f}%")
    
    print("\n" + "-"*80)
    print("📐 Weight 调制分析 (乘性调制: w = w_base * (1 + 0.5 * u_raw))")
    print("-"*80)
    
    for name, shape in model.hypo_param_shapes.items():
        if "bias" not in name:
            start, end = model.wtoken_rng[name]
            token_slice = contextualized_wtokens[:, start:end, :]
            u_raw = model.wtoken_postfc[name](token_slice)
            u = 1.0 + 0.5 * u_raw
            
            u_raw_np = u_raw.detach().cpu().numpy()
            u_np = u.detach().cpu().numpy()
            w_base = model.base_params[name].detach().cpu().numpy()
            
            # 乘性调制的相对变化就是 u_raw
            layer_stats = {
                "u_raw": {
                    "mean": float(np.mean(u_raw_np)),
                    "std": float(np.std(u_raw_np)),
                    "min": float(np.min(u_raw_np)),
                    "max": float(np.max(u_raw_np)),
                    "abs_mean": float(np.mean(np.abs(u_raw_np))),
                },
                "u (1 + 0.5 * u_raw)": {
                    "mean": float(np.mean(u_np)),
                    "min": float(np.min(u_np)),
                    "max": float(np.max(u_np)),
                },
                "w_base": {
                    "abs_mean": float(np.mean(np.abs(w_base))),
                },
                "relative_scale_change": {
                    "mean": float(np.mean(np.abs(u_raw_np))) * 100,  # 百分比
                    "max": float(np.max(np.abs(u_raw_np))) * 100,
                },
            }
            modulation_stats["weight_modulation"][name] = layer_stats
            
            print(f"\n┌─ {name}")
            print(f"│  u_raw (经 Tanh，范围 [-1, 1]):")
            print(f"│    Mean: {layer_stats['u_raw']['mean']:.6f}, "
                  f"Std: {layer_stats['u_raw']['std']:.6f}")
            print(f"│    Min: {layer_stats['u_raw']['min']:.6f}, "
                  f"Max: {layer_stats['u_raw']['max']:.6f}")
            print(f"│    |Abs| Mean: {layer_stats['u_raw']['abs_mean']:.6f}")
            print(f"│  ")
            print(f"│  缩放因子 u = 1 + 0.5 * u_raw:")
            print(f"│    Mean: {layer_stats['u (1 + 0.5 * u_raw)']['mean']:.6f}")
            print(f"│    Range: [{layer_stats['u (1 + 0.5 * u_raw)']['min']:.4f}, "
                  f"{layer_stats['u (1 + 0.5 * u_raw)']['max']:.4f}]")
            print(f"│  ")
            print(f"│  相对尺度变化:")
            print(f"│    Mean: ±{layer_stats['relative_scale_change']['mean']:.2f}%")
            print(f"└    Max: ±{layer_stats['relative_scale_change']['max']:.2f}%")
    
    return modulation_stats


def analyze_siren_frequency_impact(model, bias_mod_scale: float) -> Dict:
    """
    分析 SIREN 频率 (omega=30) 对调制的影响
    
    SIREN 使用 sin(30 * x) 作为激活函数，这意味着：
    - bias 调制会直接影响相位
    - 相位变化 = 30 * (bias 变化)
    """
    print("\n" + "="*80)
    print("🌊 SIREN 频率影响分析 (omega = 30)")
    print("="*80)
    
    omega = 30.0
    
    print(f"\n[背景知识]")
    print(f"  SIREN 激活: sin(omega * (W*x + b)), omega = {omega}")
    print(f"  当前 bias_mod_scale = {bias_mod_scale}")
    print(f"  delta_b_raw 范围: [-1, 1] (经过 Tanh)")
    print(f"  实际 bias 调制范围: [{-bias_mod_scale:.4f}, {bias_mod_scale:.4f}]")
    
    # 相位影响分析
    max_phase_shift = omega * bias_mod_scale
    
    print(f"\n[相位影响]")
    print(f"  最大相位偏移 = omega * max_bias_mod = {omega} * {bias_mod_scale} = {max_phase_shift:.4f} rad")
    print(f"  相位偏移范围: [{-max_phase_shift:.4f}, {max_phase_shift:.4f}] rad")
    print(f"  换算成度数: [{-max_phase_shift * 180 / np.pi:.2f}°, {max_phase_shift * 180 / np.pi:.2f}°]")
    print(f"  占整周期比例: {max_phase_shift / (2 * np.pi) * 100:.2f}%")
    
    # 建议
    print(f"\n[建议]")
    if max_phase_shift < 0.3:
        print(f"  ⚠️ 相位偏移较小 ({max_phase_shift:.4f} rad ≈ {max_phase_shift * 180 / np.pi:.1f}°)")
        print(f"     调制能力可能不足，建议适当增大 bias_mod_scale")
        suggested_scale = 0.5 / omega  # 目标: 约 0.5 rad 偏移
        print(f"     建议值: {suggested_scale:.4f} (对应 ~0.5 rad 或 ~30° 相位偏移)")
    elif max_phase_shift > 3.14:
        print(f"  ⚠️ 相位偏移过大 ({max_phase_shift:.4f} rad ≈ {max_phase_shift * 180 / np.pi:.1f}°)")
        print(f"     可能导致训练不稳定")
        suggested_scale = 1.0 / omega  # 目标: 约 1 rad 偏移
        print(f"     建议值: {suggested_scale:.4f} (对应 ~1 rad 或 ~57° 相位偏移)")
    else:
        print(f"  ✅ 相位偏移在合理范围 ({max_phase_shift:.4f} rad ≈ {max_phase_shift * 180 / np.pi:.1f}°)")
    
    return {
        "omega": omega,
        "bias_mod_scale": bias_mod_scale,
        "max_phase_shift_rad": float(max_phase_shift),
        "max_phase_shift_deg": float(max_phase_shift * 180 / np.pi),
        "phase_shift_ratio": float(max_phase_shift / (2 * np.pi)),
    }


def provide_recommendations(base_stats: Dict, modulation_stats: Dict, frequency_impact: Dict):
    """
    根据分析结果提供 bias_mod_scale 设置建议
    """
    print("\n" + "="*80)
    print("💡 综合建议")
    print("="*80)
    
    omega = frequency_impact["omega"]
    current_scale = frequency_impact["bias_mod_scale"]
    
    # 收集所有 bias 层的调制信息
    bias_layers = modulation_stats.get("bias_modulation", {})
    
    if not bias_layers:
        print("\n[警告] 未找到 bias 调制统计信息")
        return
    
    print(f"\n[当前设置]")
    print(f"  bias_mod_scale = {current_scale}")
    print(f"  SIREN omega = {omega}")
    
    # 分析各层的调制利用率
    print(f"\n[调制利用率分析]")
    avg_delta_b_abs = []
    for name, stats in bias_layers.items():
        delta_b_abs_mean = stats["delta_b_raw"]["abs_mean"]
        avg_delta_b_abs.append(delta_b_abs_mean)
        utilization = delta_b_abs_mean * 100  # Tanh 输出范围 [-1,1]，利用率
        print(f"  {name}: |delta_b_raw| mean = {delta_b_abs_mean:.4f} "
              f"(利用率 ~{utilization:.1f}% of Tanh range)")
    
    overall_utilization = np.mean(avg_delta_b_abs)
    
    print(f"\n[综合评估]")
    print(f"  平均 |delta_b_raw|: {overall_utilization:.4f}")
    
    # 根据利用率和相位偏移给出建议
    max_phase = frequency_impact["max_phase_shift_rad"]
    
    if overall_utilization < 0.3:
        print(f"  📊 调制值较小，模型可能欠调制")
        if max_phase < 0.5:
            recommended_scale = current_scale * 2
            print(f"  💡 建议: 增大 bias_mod_scale 到 {recommended_scale:.4f}")
            print(f"     理由: 当前相位偏移能力不足")
    elif overall_utilization > 0.7:
        print(f"  📊 调制值较大，模型充分利用调制能力")
        if max_phase > 2.0:
            recommended_scale = current_scale * 0.5
            print(f"  💡 建议: 减小 bias_mod_scale 到 {recommended_scale:.4f}")
            print(f"     理由: 相位偏移可能过大，训练可能不稳定")
        else:
            print(f"  ✅ 当前设置看起来合理")
    else:
        print(f"  ✅ 调制利用率适中")
        if 0.3 <= max_phase <= 1.5:
            print(f"  ✅ 当前 bias_mod_scale = {current_scale} 设置合理")
        else:
            print(f"  💡 可以尝试微调 bias_mod_scale")
    
    print(f"\n[常用参考值]")
    print(f"  omega=30 时的建议范围:")
    print(f"    保守值: {0.01:.4f} (相位偏移 ~17°)")
    print(f"    中等值: {1/omega:.4f} ≈ 0.033 (相位偏移 ~57°)")
    print(f"    激进值: {np.pi/(2*omega):.4f} ≈ 0.052 (相位偏移 ~90°)")


def run_multi_batch_analysis(model, dataloader, num_batches: int = 10) -> Dict:
    """
    在多个 batch 上运行分析，获得更稳定的统计结果
    """
    print(f"\n" + "="*80)
    print(f"📊 多 Batch 统计分析 (num_batches={num_batches})")
    print("="*80)
    
    device = next(model.parameters()).device
    
    all_delta_b = {name: [] for name in model.hypo_param_shapes if "bias" in name}
    all_u_raw = {name: [] for name in model.hypo_param_shapes if "bias" not in name}
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # 移动数据到设备
        if isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for k2 in batch[key]:
                        if isinstance(batch[key][k2], torch.Tensor):
                            batch[key][k2] = batch[key][k2].to(device)
        
        with torch.no_grad():
            features, encoder_attn_mask = model.compute_input_features(batch)
            B = features.size(0)
            
            decoder_input_embeds = model.param_queries.expand(B, -1, -1)
            decoder_outputs = model.vlm.get_decoder()(
                inputs_embeds=decoder_input_embeds,
                encoder_hidden_states=features,
                encoder_attention_mask=encoder_attn_mask,
                attention_mask=create_bidirectional_mask(B, model.param_queries.size(1), device),
            )
            contextualized_wtokens = decoder_outputs[0]
            
            # 收集 bias 调制值
            for name, shape in model.hypo_param_shapes.items():
                if "bias" in name:
                    start, end = model.btoken_rng[name]
                    token_slice = contextualized_wtokens[:, start:end, :]
                    delta_b_raw = model.btoken_postfc[name](token_slice).squeeze(1)
                    all_delta_b[name].append(delta_b_raw.cpu())
                else:
                    start, end = model.wtoken_rng[name]
                    token_slice = contextualized_wtokens[:, start:end, :]
                    u_raw = model.wtoken_postfc[name](token_slice)
                    all_u_raw[name].append(u_raw.cpu())
    
    # 汇总统计
    print(f"\n[Bias 调制 delta_b_raw 汇总统计]")
    for name, tensors in all_delta_b.items():
        if tensors:
            combined = torch.cat(tensors, dim=0).numpy()
            print(f"  {name}:")
            print(f"    Samples: {combined.shape[0]}, Features: {combined.shape[1] if len(combined.shape) > 1 else 1}")
            print(f"    Mean: {np.mean(combined):.6f}, Std: {np.std(combined):.6f}")
            print(f"    |Abs| Mean: {np.mean(np.abs(combined)):.6f}")
            print(f"    Percentiles: 5%={np.percentile(combined, 5):.4f}, "
                  f"50%={np.percentile(combined, 50):.4f}, "
                  f"95%={np.percentile(combined, 95):.4f}")
    
    print(f"\n[Weight 调制 u_raw 汇总统计]")
    for name, tensors in all_u_raw.items():
        if tensors:
            combined = torch.cat(tensors, dim=0).numpy()
            print(f"  {name}:")
            print(f"    Mean: {np.mean(combined):.6f}, Std: {np.std(combined):.6f}")
            print(f"    |Abs| Mean: {np.mean(np.abs(combined)):.6f}")
            print(f"    Percentiles: 5%={np.percentile(combined, 5):.4f}, "
                  f"50%={np.percentile(combined, 50):.4f}, "
                  f"95%={np.percentile(combined, 95):.4f}")
    
    return {"delta_b": all_delta_b, "u_raw": all_u_raw}


def main():
    parser = argparse.ArgumentParser(description="SIREN 调制参数分析")
    parser.add_argument("--ckpt_path", type=str, 
                        default='/home/lhy/Code/ICRA/beast_calvin/logs/runs/2026-01-05/19-11-32/saved_models/epoch=39_eval_lh/avg_seq_len=0.75.ckpt',
                        help="检查点路径")
    parser.add_argument("--config_path", type=str, 
                        default="/home/lhy/Code/ICRA/beast_calvin/conf/config_libero.yaml",
                        help="配置文件完整路径")
    parser.add_argument("--num_batches", type=int, default=8, 
                        help="用于多 batch 统计的批次数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备 (cuda/cpu)")
    args = parser.parse_args()
    
    print("="*80)
    print("🔬 SIREN 调制参数分析工具")
    print("="*80)
    print(f"检查点: {args.ckpt_path}")
    print(f"配置: {args.config_path}")
    print(f"设备: {args.device}")
    print(f"批次数: {args.num_batches}")
    
    # 加载模型和数据
    model, datamodule, cfg = load_model_and_data(args.ckpt_path, args.config_path)
    model = model.to(args.device)
    model.eval()
    
    # 1. 分析 base_params
    base_stats = analyze_base_params(model)
    
    # 2. 获取一个 batch 进行 forward 分析
    train_loader_dict = datamodule.train_dataloader()
    
    # HulcDataModule 返回的是 dict{key: DataLoader}，我们需要 'lang' 的 DataLoader
    if isinstance(train_loader_dict, dict):
        # 找到 lang dataset 的 loader
        if 'lang' in train_loader_dict:
            train_loader = train_loader_dict['lang']
        else:
            # 取第一个 loader
            train_loader = list(train_loader_dict.values())[0]
        print(f"[INFO] 使用 DataLoader key: {list(train_loader_dict.keys())}")
    else:
        train_loader = train_loader_dict
    
    batch = next(iter(train_loader))
    
    # 移动数据到设备
    def move_to_device(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(move_to_device(v, device) for v in data)
        else:
            return data
    
    batch = move_to_device(batch, args.device)
    
    # 3. 分析单次 forward
    modulation_stats = analyze_modulation_forward(model, batch)
    
    # 4. 分析 SIREN 频率影响
    frequency_impact = analyze_siren_frequency_impact(model, model.bias_mod_scale)
    
    # 5. 多 batch 统计
    run_multi_batch_analysis(model, train_loader, args.num_batches)
    
    # 6. 提供综合建议
    provide_recommendations(base_stats, modulation_stats, frequency_impact)
    
    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
