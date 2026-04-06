"""BEASTVel: continuous B-spline control-point regression with pos/vel supervision.

Loss:
    total_loss = lambda_pos * pos_loss + lambda_vel * vel_loss

Notes:
- Replaces token CE supervision with learnable query embeddings + regression head.
- Uses one query per B-spline control point.
- Supports action_mode=delta_first only.
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import create_bidirectional_mask
from src.models.beast import BEAST

logger = logging.getLogger(__name__)


class BEASTVel(BEAST):
    """BEAST variant with continuous control-point prediction and pos/vel L2 losses."""

    def __init__(
        self,
        *args,
        fps: float = 30.0,
        lambda_pos: float = 1.0,
        lambda_vel: float = 0.1,
        action_mode: str = "delta_first",
        arm_mode: str = "dual",
        query_init_std: float = 0.01,
        bound_warmup_steps: int = 500,
        **kwargs,
    ):
        if action_mode != "delta_first":
            raise ValueError(f"BEASTVel only supports action_mode='delta_first', got {action_mode}")
        if query_init_std <= 0:
            raise ValueError(f"query_init_std must be > 0, got {query_init_std}")
        if bound_warmup_steps < 0:
            raise ValueError(f"bound_warmup_steps must be >= 0, got {bound_warmup_steps}")

        super().__init__(*args, **kwargs)
        self.fps = float(fps)
        self.lambda_pos = float(lambda_pos)
        self.lambda_vel = float(lambda_vel)
        self.action_mode = action_mode
        self.arm_mode = arm_mode
        self.bound_warmup_steps = int(bound_warmup_steps)

        d_model = int(self.vlm.config.text_config.d_model)
        self.num_control_points = int(self.num_dof * self.num_basis)

        # One learnable query per control point.
        self.param_queries = nn.Parameter(
            torch.randn(1, self.num_control_points, d_model) * float(query_init_std)
        )
        self.param_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _iter_modalities(batch: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if isinstance(batch, dict) and "actions" in batch:
            return {"default": batch}
        if isinstance(batch, dict) and all(isinstance(v, dict) and "actions" in v for v in batch.values()):
            return batch
        if isinstance(batch, dict):
            return {"default": batch}
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _predict_control_params(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Predict continuous B-spline control parameters from learnable queries."""
        features, encoder_attn_mask = self.compute_input_features(batch)
        bsz = int(features.shape[0])

        decoder_inputs = self.param_queries.expand(bsz, -1, -1)
        bidirectional_mask = create_bidirectional_mask(
            batch_size=bsz,
            seq_length=decoder_inputs.shape[1],
            device=self.device,
        )

        decoder_outputs = self.vlm.get_decoder()(
            inputs_embeds=decoder_inputs,
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=bidirectional_mask,
            use_cache=False,
        )

        query_states = decoder_outputs[0]
        params_norm = torch.tanh(self.param_head(query_states).squeeze(-1))

        w_min = self.action_tokenizer.w_min.to(device=query_states.device, dtype=query_states.dtype)
        w_max = self.action_tokenizer.w_max.to(device=query_states.device, dtype=query_states.dtype)
        params_pred = (params_norm + 1.0) * 0.5 * (w_max - w_min) + w_min
        return params_pred

    def _maybe_update_param_bounds(self, actions_gt: torch.Tensor) -> None:
        """Warmup-time bound adaptation using GT trajectories.

        In continuous regression mode, tokenizer encode() is no longer part of the
        forward path, so w_min/w_max would otherwise stay at default values.
        """
        if not bool(getattr(self, "update_w_bound", False)):
            return

        # If bounds are precomputed on fit start, keep them fixed during training.
        if bool(getattr(self, "precompute_w_bound", False)):
            return

        warmup_steps = int(getattr(self, "bound_warmup_steps", 0))
        if warmup_steps <= 0:
            return

        if int(getattr(self, "global_step", 0)) >= warmup_steps:
            return

        with torch.no_grad():
            # Side effect only: updates tokenizer w_min/w_max via update_bounds=True.
            self.action_tokenizer.encode(actions_gt.detach(), update_bounds=True)

    def compute_llm_outputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "actions" not in batch:
            raise ValueError("batch is missing 'actions'")

        actions_gt = batch["actions"].to(self.device)
        params_pred = self._predict_control_params(batch)

        actions_pred = self.action_tokenizer.reconstruct_traj_from_params(
            params_pred,
            times=None,
        )

        use_t_pos = min(actions_pred.shape[1], actions_gt.shape[1])
        pos_loss = F.mse_loss(actions_pred[:, :use_t_pos, :], actions_gt[:, :use_t_pos, :])

        if self.lambda_vel > 0.0:
            if "joint_vel" not in batch:
                raise KeyError("velocity supervision is enabled but batch is missing 'joint_vel'")
            vel_gt = batch["joint_vel"].to(self.device)
            vel_pred_norm = self.action_tokenizer.reconstruct_traj_vel_from_params(
                params_pred,
                times=None,
                execution_hz=self.fps,
            )
            vel_pred = self.denormalize_velocities(vel_pred_norm)
            vel_gt = vel_gt.to(dtype=vel_pred.dtype)
            use_t_vel = min(vel_pred.shape[1], vel_gt.shape[1])
            use_t_vel = max(1, use_t_vel - 1)
            vel_loss = F.mse_loss(vel_pred[:, :use_t_vel, :], vel_gt[:, :use_t_vel, :])
        else:
            vel_loss = torch.zeros((), device=self.device, dtype=pos_loss.dtype)

        total_loss = self.lambda_pos * pos_loss + self.lambda_vel * vel_loss

        return {
            "total_loss": total_loss,
            "pos_loss": pos_loss,
            "vel_loss": vel_loss,
        }

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        modalities = self._iter_modalities(batch)

        total_loss = torch.zeros((), device=self.device)
        pos_loss = torch.zeros((), device=self.device)
        vel_loss = torch.zeros((), device=self.device)
        total_bs = 0

        for modality_scope, dataset_batch in modalities.items():
            self.modality_scope = modality_scope

            actions_gt = dataset_batch["actions"].to(self.device)
            self._maybe_update_param_bounds(actions_gt)

            out = self.compute_llm_outputs(dataset_batch)

            bsz = int(dataset_batch["actions"].shape[0])
            total_bs += bsz

            total_loss = total_loss + out["total_loss"]
            pos_loss = pos_loss + out["pos_loss"].detach()
            vel_loss = vel_loss + out["vel_loss"].detach()

        n_mod = max(1, len(modalities))
        total_loss = total_loss / n_mod
        pos_loss = pos_loss / n_mod
        vel_loss = vel_loss / n_mod

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/pos_loss", pos_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/vel_loss", vel_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=total_bs)

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        modalities = self._iter_modalities(batch)

        total_loss = torch.zeros((), device=self.device)
        pos_loss = torch.zeros((), device=self.device)
        vel_loss = torch.zeros((), device=self.device)
        total_bs = 0

        with torch.no_grad():
            for modality_scope, dataset_batch in modalities.items():
                self.modality_scope = modality_scope
                out = self.compute_llm_outputs(dataset_batch)

                bsz = int(dataset_batch["actions"].shape[0])
                total_bs += bsz

                total_loss = total_loss + out["total_loss"]
                pos_loss = pos_loss + out["pos_loss"]
                vel_loss = vel_loss + out["vel_loss"]

        n_mod = max(1, len(modalities))
        total_loss = total_loss / n_mod
        pos_loss = pos_loss / n_mod
        vel_loss = vel_loss / n_mod

        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("val/pos_loss", pos_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("val/vel_loss", vel_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

        return {"val_loss": total_loss}

    @torch.no_grad()
    def forward(self, obs: Dict[str, Any], goal: Dict[str, Any]) -> torch.Tensor:
        """Inference: predict continuous control points and reconstruct denormalized actions."""
        rgb_obs_batch = {}
        for k in self.rgb_obs_keys:
            if k in obs["rgb_obs"]:
                rgb_obs_batch[k] = obs["rgb_obs"][k]
        batch = {
            "rgb_obs": rgb_obs_batch,
            "lang_text": [goal["lang_text"]],
        }

        params_pred = self._predict_control_params(batch)
        actions_normalized = self.action_tokenizer.reconstruct_traj_from_params(
            params_pred,
            times=None,
        )
        return self.denormalize_actions(actions_normalized)

    @torch.no_grad()
    def forward_with_velocity(
        self, obs: Dict[str, Any], goal: Dict[str, Any], execution_hz: float = 30.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference: return denormalized actions and physical joint velocities (rad/s)."""
        rgb_obs_batch = {}
        for k in self.rgb_obs_keys:
            if k in obs["rgb_obs"]:
                rgb_obs_batch[k] = obs["rgb_obs"][k]
        batch = {
            "rgb_obs": rgb_obs_batch,
            "lang_text": [goal["lang_text"]],
        }

        params_pred = self._predict_control_params(batch)
        actions_norm = self.action_tokenizer.reconstruct_traj_from_params(
            params_pred,
            times=None,
        )
        vel_norm = self.action_tokenizer.reconstruct_traj_vel_from_params(
            params_pred,
            times=None,
            execution_hz=execution_hz,
        )

        actions = self.denormalize_actions(actions_norm)
        velocities = self.denormalize_velocities(vel_norm)
        return actions, velocities
