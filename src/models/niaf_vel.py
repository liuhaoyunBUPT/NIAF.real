from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.niaf import NIAF
from src.models.base import create_bidirectional_mask


logger = logging.getLogger(__name__)


def _get_stats_minmax(stats_cfg: Any, action_dim: int = 7) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """获取动作归一化统计参数 (min, max)。
    
    Args:
        stats_cfg: 包含 min/max 的配置对象或字典
        action_dim: 动作维度，用于切片归一化参数
    
    Returns:
        (min_array, max_array) 或 None
    """
    if stats_cfg is None:
        return None
    if isinstance(stats_cfg, dict):
        mn = stats_cfg.get("min", None)
        mx = stats_cfg.get("max", None)
    else:
        mn = getattr(stats_cfg, "min", None)
        mx = getattr(stats_cfg, "max", None)
    if mn is None or mx is None:
        return None
    mn = np.asarray(mn, dtype=np.float32).reshape(-1)[:action_dim]
    mx = np.asarray(mx, dtype=np.float32).reshape(-1)[:action_dim]
    if mn.shape[0] != action_dim or mx.shape[0] != action_dim:
        return None
    return mn, mx


def _delta_first_scale_s(stats: Tuple[np.ndarray, np.ndarray], device: torch.device, dtype: torch.dtype, action_dim: int = 7) -> torch.Tensor:
    """计算 delta_first 模式下的缩放因子 s = (max - min) / 2。
    
    Args:
        stats: (min_array, max_array) 归一化参数
        device: 目标设备
        dtype: 目标数据类型
        action_dim: 动作维度
    
    Returns:
        shape (1, 1, action_dim) 的缩放因子张量
    """
    mn, mx = stats
    s = (np.asarray(mx, dtype=np.float32) - np.asarray(mn, dtype=np.float32)) / 2.0
    return torch.as_tensor(s, device=device, dtype=dtype).view(1, 1, action_dim)


def _scale_tau_to_t(*, fps: float, K: int) -> float:
    if K <= 1:
        raise ValueError(f"chunk_size K must be >=2, got K={K}")
    return 2.0 * float(fps) / float(K - 1)


def _d1_dtau_from_actions_and_coords(
    actions_pred: torch.Tensor,
    coords_in: torch.Tensor,
    *,
    create_graph: bool,
    keep_graph: bool = False,
) -> torch.Tensor:
    """Compute dactions/dtau for all action dimensions from a single Siren forward.
    
    Args:
        actions_pred: 预测动作 (B, K, action_dim)
        coords_in: Siren 输入坐标 (B, K, 1)
        create_graph: 是否创建计算图（用于高阶导数或反向传播）
        keep_graph: 是否在最后一个导数计算后保留图
    
    Returns:
        d1: 一阶导数 (B, K, action_dim)
    """
    action_dim = int(actions_pred.shape[-1])
    num_joints = action_dim  # 计算所有维度的导数
    d1 = torch.zeros_like(actions_pred)
    for j in range(num_joints):
        grad_j = torch.autograd.grad(
            actions_pred[..., j].sum(),
            coords_in,
            retain_graph=(True if keep_graph else (j != num_joints - 1)),
            create_graph=create_graph,
            allow_unused=False,
        )[0]
        d1[..., j] = grad_j.squeeze(-1)
    return d1


def _d1d2d3_dtau_from_actions_and_coords(
    actions_pred: torch.Tensor,
    coords_in: torch.Tensor,
    *,
    create_graph: bool,
    keep_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 1st/2nd/3rd derivatives wrt tau for all action dimensions.

    Args:
        actions_pred: 预测动作 (B, K, action_dim)
        coords_in: Siren 输入坐标 (B, K, 1)
        create_graph: 是否创建计算图
        keep_graph: 是否在最后一个导数计算后保留图
    
    Returns: (d1, d2, d3) with same shape as actions_pred.
    """
    action_dim = int(actions_pred.shape[-1])
    num_joints = action_dim  # 计算所有维度的导数
    d1 = torch.zeros_like(actions_pred)
    d2 = torch.zeros_like(actions_pred)
    d3 = torch.zeros_like(actions_pred)

    for j in range(num_joints):
        d1_j = torch.autograd.grad(
            actions_pred[..., j].sum(),
            coords_in,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        d2_j = torch.autograd.grad(
            d1_j.sum(),
            coords_in,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        d3_j = torch.autograd.grad(
            d2_j.sum(),
            coords_in,
            retain_graph=(True if keep_graph else (j != num_joints - 1)),
            create_graph=create_graph,
            allow_unused=False,
        )[0]

        d1[..., j] = d1_j.squeeze(-1)
        d2[..., j] = d2_j.squeeze(-1)
        d3[..., j] = d3_j.squeeze(-1)

    return d1, d2, d3


def _siren_forward_with_model_in(
    model: NIAF,
    dataset_batch: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run NIAF Siren branch and return (actions_pred, coords_in).

    Important: SingleBVPNet internally clones+detaches input coords and returns the leaf
    tensor as `siren_out['model_in']`. We must differentiate w.r.t. that tensor.
    """
    device = next(model.parameters()).device
    default_dtype = next(model.parameters()).dtype
    use_fp32 = bool(getattr(model, "fp32", False))

    if not hasattr(model, "modality_scope"):
        setattr(model, "modality_scope", "default")
    else:
        model.modality_scope = getattr(model, "modality_scope", "default") or "default"

    features, encoder_attn_mask = model.compute_input_features(dataset_batch)
    B = int(features.size(0))

    param_queries = model.param_queries
    decoder_input_embeds = param_queries.expand(B, -1, -1)
    decoder_outputs = model.vlm.get_decoder()(
        inputs_embeds=decoder_input_embeds,
        encoder_hidden_states=features,
        encoder_attention_mask=encoder_attn_mask,
        attention_mask=create_bidirectional_mask(B, param_queries.size(1), device),
    )
    contextualized_wtokens = decoder_outputs[0]

    wtoks_for_mod = contextualized_wtokens.float() if use_fp32 else contextualized_wtokens

    amp_off = nullcontext()
    if device.type == "cuda":
        amp_off = torch.amp.autocast("cuda", enabled=False)

    hypo_params: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, shape in model.hypo_param_shapes.items():
        if "bias" in name:
            start, end = model.btoken_rng[name]
            token_slice = wtoks_for_mod[:, start:end, :]
            if use_fp32:
                with amp_off:
                    delta_b_raw = model.btoken_postfc[name](token_slice).squeeze(1)
            else:
                delta_b_raw = model.btoken_postfc[name](token_slice).squeeze(1)
            b_base = (model.base_params[name].float() if use_fp32 else model.base_params[name]).unsqueeze(0)
            b_modulated = b_base + float(model.bias_mod_scale) * delta_b_raw
            hypo_params[name] = b_modulated
        else:
            start, end = model.wtoken_rng[name]
            token_slice = wtoks_for_mod[:, start:end, :]
            if use_fp32:
                with amp_off:
                    u_raw = model.wtoken_postfc[name](token_slice)
            else:
                u_raw = model.wtoken_postfc[name](token_slice)
            u = 1.0 + u_raw
            g = int(u.shape[1])
            out_features = int(shape[0])
            u_repeated = u.repeat_interleave(out_features // g, dim=1)
            w_base = (model.base_params[name].float() if use_fp32 else model.base_params[name]).unsqueeze(0)
            w_modulated = w_base * u_repeated
            w_modulated = w_modulated + 1e-7
            hypo_params[name] = w_modulated

    final_hypo_params = OrderedDict((n.replace("_", "."), v) for n, v in hypo_params.items())

    K = int(getattr(model, "chunk_size", 10))
    coords_dtype = torch.float32 if use_fp32 else default_dtype
    coords = model._make_time_coords(B, K, device, coords_dtype)
    model_input = {"coords": coords}

    if use_fp32:
        fp32_params = OrderedDict((k, v.float()) for k, v in final_hypo_params.items())
        with amp_off:
            siren_out = model.hypo_net(model_input, params=fp32_params)
    else:
        siren_out = model.hypo_net(model_input, params=final_hypo_params)

    coords_in = siren_out["model_in"]
    actions_pred = siren_out["model_out"]
    return actions_pred, coords_in


def actions_and_d1d2d3_dtau_from_sirenvla(
    model: NIAF,
    dataset_batch: Dict[str, Any],
    *,
    create_graph: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 1st/2nd/3rd derivatives wrt normalized time tau for the 7 arm joints.

    Returns:
        actions_pred: (B, K, action_dim)
        d1: (B, K, action_dim)
        d2: (B, K, action_dim)
        d3: (B, K, action_dim)
    """
    actions_pred, coords_in = _siren_forward_with_model_in(model, dataset_batch)
    d1, d2, d3 = _d1d2d3_dtau_from_actions_and_coords(
        actions_pred,
        coords_in,
        create_graph=create_graph,
        keep_graph=False,
    )
    return actions_pred, d1, d2, d3


def actions_and_dactions_dtau_from_sirenvla(
    model: NIAF,
    dataset_batch: Dict[str, Any],
    *,
    create_graph: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run NIAF Siren branch and compute analytic dactions/dtau via autograd.

    This function intentionally does NOT modify NIAF code; it mirrors
    `NIAF._generate_actions_siren` but keeps the internal Siren coords tensor
    (`siren_out['model_in']`) to differentiate w.r.t. normalized time tau.

    Returns:
        actions_pred: (B, K, action_dim)
        dactions_dtau: (B, K, action_dim)
    """
    actions_pred, coords_in = _siren_forward_with_model_in(model, dataset_batch)

    dactions_dtau = _d1_dtau_from_actions_and_coords(
        actions_pred,
        coords_in,
        create_graph=create_graph,
        keep_graph=False,
    )
    return actions_pred, dactions_dtau


@dataclass
class VelSupervisionConfig:
    fps: float = 30.0
    act_loss_weight: float = 1.0  # 位置 loss 权重
    vel_loss_weight: float = 0.0
    jerk_loss_weight: float = 0.0
    action_mode: str = "delta_first"
    delta_first_action_stats: Any = None


class NIAFVel(NIAF):
    """NIAF subclass that adds joint velocity supervision.

    - Expects dataset batch to include `joint_vel` in rad/s with shape (B, K, 7).
    - Only supports `delta_first` action mode.
    - Computes analytic dactions/dtau in normalized action space and converts to rad/s by
      cancelling both the tau->time scaling and per-joint min/max normalization scaling.
    - Supports arm_mode to automatically adjust action dimensions.
    """

    def __init__(
        self,
        *args,
        fps: float = 30.0,
        act_loss_weight: float = 1.0,
        vel_loss_weight: float = 0.0,
        jerk_loss_weight: float = 0.0,
        delta_first_action_stats: Any = None,
        # 以下参数传递给父类 NIAF 用于反归一化
        action_mode: str = "delta_first",
        action_min: Any = None,
        action_max: Any = None,
        # arm_mode 用于自动调整动作维度
        arm_mode: str = "dual",  # "dual", "left", "right"
        **kwargs,
    ) -> None:
        # 强制使用 delta_first 模式
        if action_mode != "delta_first":
            logger.warning(f"NIAFVel 只支持 delta_first 模式，忽略 action_mode={action_mode}")
            action_mode = "delta_first"
        
        # 保存 arm_mode
        self.arm_mode = arm_mode
        
        # 根据 arm_mode 调整 action_dim 和归一化参数
        original_action_dim = kwargs.get('action_dim', 14)
        if arm_mode == "left":
            effective_action_dim = 7
            if action_min is not None:
                action_min = list(action_min)[:7]
            if action_max is not None:
                action_max = list(action_max)[:7]
            if delta_first_action_stats is not None:
                if isinstance(delta_first_action_stats, dict):
                    delta_first_action_stats = {
                        'min': list(delta_first_action_stats.get('min', []))[:7],
                        'max': list(delta_first_action_stats.get('max', []))[:7],
                    }
            logger.info(f"arm_mode=left: action_dim adjusted from {original_action_dim} to {effective_action_dim}")
        elif arm_mode == "right":
            effective_action_dim = 7
            if action_min is not None:
                action_min = list(action_min)[7:14]
            if action_max is not None:
                action_max = list(action_max)[7:14]
            if delta_first_action_stats is not None:
                if isinstance(delta_first_action_stats, dict):
                    delta_first_action_stats = {
                        'min': list(delta_first_action_stats.get('min', []))[7:14],
                        'max': list(delta_first_action_stats.get('max', []))[7:14],
                    }
            logger.info(f"arm_mode=right: action_dim adjusted from {original_action_dim} to {effective_action_dim}")
        else:  # dual
            effective_action_dim = original_action_dim
            logger.info(f"arm_mode=dual: action_dim={effective_action_dim}")
        
        # 更新 kwargs 中的 action_dim
        kwargs['action_dim'] = effective_action_dim
        
        # 传递归一化参数给父类
        super().__init__(
            *args,
            action_mode=action_mode,
            action_min=action_min,
            action_max=action_max,
            **kwargs,
        )
        
        # 速度监督配置
        self.vel_cfg = VelSupervisionConfig(
            fps=float(fps),
            act_loss_weight=float(act_loss_weight),
            vel_loss_weight=float(vel_loss_weight),
            jerk_loss_weight=float(jerk_loss_weight),
            action_mode="delta_first",  # 强制 delta_first
            delta_first_action_stats=delta_first_action_stats,
        )
        self._logged_vel_scale_factors = False

    def _pred_joint_vel_rad_s_from_dactions_dtau(self, dactions_dtau: torch.Tensor) -> torch.Tensor:
        """将归一化时间导数转换为物理速度 (rad/s)。
        
        Args:
            dactions_dtau: 动作对归一化时间的导数 (B, K, action_dim)
        
        Returns:
            物理速度 (B, K, action_dim)，单位 rad/s
        """
        if self.vel_cfg.action_mode != "delta_first":
            raise ValueError(
                f"NIAFVel only supports action_mode=delta_first for vel supervision, "
                f"got {self.vel_cfg.action_mode}"
            )
        
        action_dim = dactions_dtau.shape[-1]
        stats = _get_stats_minmax(self.vel_cfg.delta_first_action_stats, action_dim=action_dim)
        if stats is None:
            raise ValueError(f"delta_first_action_stats (min/max) must be provided for vel supervision, need {action_dim} dims")

        _, K, _ = dactions_dtau.shape
        scale_t = _scale_tau_to_t(fps=float(self.vel_cfg.fps), K=int(K))
        vel_norm = dactions_dtau * float(scale_t)  # 所有维度
        s = _delta_first_scale_s(stats, device=vel_norm.device, dtype=vel_norm.dtype, action_dim=action_dim)
        return vel_norm * s

    def _pred_joint_jerk_rad_s3_from_d3actions_dtau3(self, d3actions_dtau3: torch.Tensor) -> torch.Tensor:
        """将归一化时间三阶导数转换为物理 jerk (rad/s³)。
        
        Args:
            d3actions_dtau3: 动作对归一化时间的三阶导数 (B, K, action_dim)
        
        Returns:
            物理 jerk (B, K, action_dim)，单位 rad/s³
        """
        if self.vel_cfg.action_mode != "delta_first":
            raise ValueError(
                f"NIAFVel only supports action_mode=delta_first for jerk regularization, "
                f"got {self.vel_cfg.action_mode}"
            )
        
        action_dim = d3actions_dtau3.shape[-1]
        stats = _get_stats_minmax(self.vel_cfg.delta_first_action_stats, action_dim=action_dim)
        if stats is None:
            raise ValueError(f"delta_first_action_stats (min/max) must be provided for jerk regularization, need {action_dim} dims")

        _, K, _ = d3actions_dtau3.shape
        scale_t = _scale_tau_to_t(fps=float(self.vel_cfg.fps), K=int(K))
        jerk_norm = d3actions_dtau3 * float(scale_t) ** 3  # 所有维度
        s = _delta_first_scale_s(stats, device=jerk_norm.device, dtype=jerk_norm.dtype, action_dim=action_dim)
        return jerk_norm * s

    def predict_actions_and_joint_vel(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        actions_pred, coords_in = _siren_forward_with_model_in(self, batch)
        dactions_dtau = _d1_dtau_from_actions_and_coords(
            actions_pred,
            coords_in,
            create_graph=False,
            keep_graph=False,
        )
        vel_pred = self._pred_joint_vel_rad_s_from_dactions_dtau(dactions_dtau)
        return actions_pred, vel_pred

    def forward_with_joint_vel(self, obs: Dict[str, Any], goal: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference API: return (actions_pred, joint_vel_pred_rad_s).

        This mirrors NIAF.forward()'s input structure so rollout/eval code can call it
        without changing existing observation plumbing.
        """
        rgb_static = obs["rgb_obs"]["rgb_static"]
        rgb_gripper = obs["rgb_obs"].get("rgb_gripper", None)
        lang_text = goal.get("lang_text", "")

        batch: Dict[str, Any] = {
            "rgb_obs": {"rgb_static": rgb_static},
            "lang_text": [lang_text],
        }
        if rgb_gripper is not None:
            batch["rgb_obs"]["rgb_gripper"] = rgb_gripper

        # We need autograd enabled to compute dactions/dtau, but don't need higher-order graphs.
        with torch.enable_grad():
            actions_pred, vel_pred = self.predict_actions_and_joint_vel(batch)
        return actions_pred, vel_pred

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0

        modalities = self._iter_modalities(batch)
        for modality_scope, dataset_batch in modalities.items():
            self.modality_scope = modality_scope

            actions_gt = dataset_batch["actions"].to(self.device)
            B, T_gt, _ = actions_gt.shape
            T = min(self.chunk_size, T_gt)

            # Single Siren forward: reuse for action/velocity/jerk to reduce VRAM.
            actions_pred, coords_in = _siren_forward_with_model_in(self, dataset_batch)
            
            loss_act = F.mse_loss(actions_pred[:, :T, :], actions_gt[:, :T, :])

            # 使用 act_loss_weight 加权位置 loss
            loss = float(self.vel_cfg.act_loss_weight) * loss_act

            # Velocity supervision (optional)
            if float(self.vel_cfg.vel_loss_weight) > 0.0:
                if "joint_vel" not in dataset_batch:
                    raise KeyError("Velocity supervision enabled but batch is missing key 'joint_vel'.")

                # Debug/verification: print velocity scaling factors once (rank-0) to compare
                # with offline_check_siren_velocity_scale.py.
                if not self._logged_vel_scale_factors:
                    trainer = getattr(self, "trainer", None)
                    is_global_zero = True
                    if trainer is not None and hasattr(trainer, "is_global_zero"):
                        is_global_zero = bool(trainer.is_global_zero)
                    if is_global_zero:
                        stats = _get_stats_minmax(self.vel_cfg.delta_first_action_stats)
                        if stats is None:
                            logger.warning(
                                "[vel_scale] delta_first_action_stats missing; cannot log s=(max-min)/2."
                            )
                        else:
                            mn, mx = stats
                            s = (mx - mn) / 2.0
                            K = int(actions_pred.shape[1])
                            scale_t = _scale_tau_to_t(fps=float(self.vel_cfg.fps), K=K)
                            combined = s * float(scale_t)
                            logger.info(
                                "[vel_scale] action_mode=%s fps=%.6f K=%d scale_t(dτ/dt)=%.8f\n"
                                "[vel_scale] s=(max-min)/2 (rad per norm): %s\n"
                                "[vel_scale] scale_t*s (rad/s per norm): %s",
                                str(self.vel_cfg.action_mode),
                                float(self.vel_cfg.fps),
                                K,
                                float(scale_t),
                                np.array2string(s, precision=6, floatmode="fixed"),
                                np.array2string(combined, precision=6, floatmode="fixed"),
                            )
                    self._logged_vel_scale_factors = True

                # Compute analytic derivatives with graph so vel loss can backprop.
                # IMPORTANT: keep_graph=True so we can still backprop through action_loss and vel_loss.
                dactions_dtau = _d1_dtau_from_actions_and_coords(
                    actions_pred,
                    coords_in,
                    create_graph=True,
                    keep_graph=True,
                )
                vel_pred = self._pred_joint_vel_rad_s_from_dactions_dtau(dactions_dtau)

                vel_gt = dataset_batch["joint_vel"].to(self.device)
                use = min(vel_pred.shape[1], vel_gt.shape[1], T)
                use = max(1, use - 1)  # avoid boundary; supervise first K-1

                loss_vel = F.mse_loss(vel_pred[:, :use, :], vel_gt[:, :use, :])
                
                loss = loss + float(self.vel_cfg.vel_loss_weight) * loss_vel

                # 安全 logging: 避免 NaN 传播到 wandb
                vel_loss_value = loss_vel.detach()
                if not torch.isnan(vel_loss_value) and not torch.isinf(vel_loss_value):
                    self.log(
                        "train/vel_loss",
                        vel_loss_value,
                        on_step=True,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=B,
                    )

            # Jerk regularization (optional, no GT): penalize 3rd derivative magnitude in rad/s^3.
            if float(self.vel_cfg.jerk_loss_weight) > 0.0:
                # Compute higher-order derivatives with graph for backprop.
                # IMPORTANT: keep_graph=True so the main backward can still traverse this graph.
                _d1, _d2, d3actions_dtau3 = _d1d2d3_dtau_from_actions_and_coords(
                    actions_pred,
                    coords_in,
                    create_graph=True,
                    keep_graph=True,
                )
                jerk_pred = self._pred_joint_jerk_rad_s3_from_d3actions_dtau3(d3actions_dtau3)
                use = min(jerk_pred.shape[1], T)
                use = max(1, use - 1)
                loss_jerk = torch.mean(jerk_pred[:, :use, :].pow(2))
                
                loss = loss + float(self.vel_cfg.jerk_loss_weight) * loss_jerk

                # 安全 logging
                jerk_loss_value = loss_jerk.detach()
                if not torch.isnan(jerk_loss_value) and not torch.isinf(jerk_loss_value):
                    self.log(
                        "train/jerk_loss",
                        jerk_loss_value,
                        on_step=True,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=B,
                    )

            total_loss += loss
            total_bs += B

            # 安全 logging for act_loss
            act_loss_value = loss_act.detach()
            if not torch.isnan(act_loss_value) and not torch.isinf(act_loss_value):
                self.log(
                    "train/act_loss",
                    act_loss_value,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=B,
                )

        total_loss = total_loss / max(1, len(modalities))
        
        # 安全 logging for total_loss
        if not torch.isnan(total_loss) and not torch.isinf(total_loss):
            self.log(
                "train/total_loss",
                total_loss,
                on_step=True,
                on_epoch=True,
            sync_dist=True,
            batch_size=total_bs,
        )
        return total_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0

        modalities = self._iter_modalities(batch)
        for modality_scope, dataset_batch in modalities.items():
            self.modality_scope = modality_scope

            actions_gt = dataset_batch["actions"].to(self.device)
            B, T_gt, _ = actions_gt.shape
            T = min(self.chunk_size, T_gt)

            need_vel = float(self.vel_cfg.vel_loss_weight) > 0.0 and "joint_vel" in dataset_batch
            need_jerk = float(self.vel_cfg.jerk_loss_weight) > 0.0
            need_derivs = bool(need_vel or need_jerk)

            if need_derivs:
                # Lightning evaluation loop runs under global no_grad. We must compute the forward
                # pass itself with grad enabled, otherwise actions_pred/coords_in will have no grad_fn.
                with torch.enable_grad():
                    actions_pred, coords_in = _siren_forward_with_model_in(self, dataset_batch)
            else:
                actions_pred, coords_in = _siren_forward_with_model_in(self, dataset_batch)

            # Detach for pure logging losses (no backward in validation).
            actions_pred_det = actions_pred.detach()
            loss_act = F.mse_loss(actions_pred_det[:, :T, :], actions_gt[:, :T, :])

            # 使用 act_loss_weight 加权位置 loss
            loss = float(self.vel_cfg.act_loss_weight) * loss_act
            if need_vel:
                with torch.enable_grad():
                    dactions_dtau = _d1_dtau_from_actions_and_coords(
                        actions_pred,
                        coords_in,
                        create_graph=False,
                        keep_graph=False,
                    )
                    vel_pred = self._pred_joint_vel_rad_s_from_dactions_dtau(dactions_dtau)
                vel_pred = vel_pred.detach()
                vel_gt = dataset_batch["joint_vel"].to(self.device)
                use = min(vel_pred.shape[1], vel_gt.shape[1], T)
                use = max(1, use - 1)
                loss_vel = F.mse_loss(vel_pred[:, :use, :], vel_gt[:, :use, :])
                loss = loss + float(self.vel_cfg.vel_loss_weight) * loss_vel
                self.log("val/vel_loss", loss_vel, sync_dist=True, batch_size=B)

            if need_jerk:
                with torch.enable_grad():
                    _d1, _d2, d3actions_dtau3 = _d1d2d3_dtau_from_actions_and_coords(
                        actions_pred,
                        coords_in,
                        create_graph=False,
                        keep_graph=False,
                    )
                    jerk_pred = self._pred_joint_jerk_rad_s3_from_d3actions_dtau3(d3actions_dtau3)
                jerk_pred = jerk_pred.detach()
                use = min(jerk_pred.shape[1], T)
                use = max(1, use - 1)
                loss_jerk = torch.mean(jerk_pred[:, :use, :].pow(2))
                loss = loss + float(self.vel_cfg.jerk_loss_weight) * loss_jerk
                self.log("val/jerk_loss", loss_jerk, sync_dist=True, batch_size=B)

            total_loss += loss
            total_bs += B
            self.log("val/act_loss", loss_act, sync_dist=True, batch_size=B)

        total_loss = total_loss / max(1, len(modalities))
        self.log("val/total_loss", total_loss, sync_dist=True, batch_size=total_bs)
        return {"val_loss": total_loss}
