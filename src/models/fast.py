"""
FAST: FAST tokenizer model.

Key features:
- Inherits from BASE (Florence-2 VLM backbone)
- Uses FAST tokenizer for action tokenization
- Autoregressive decoding for variable-length action tokens
- Multi-camera support via rgb_obs_keys
- Action denormalization for deployment
"""

import logging
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig

from src.models.base import BASE
from src.models.tokenizers.fast_tokenizer import FAST_Tokenizer

logger = logging.getLogger(__name__)


def create_causal_mask(batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal (autoregressive) attention mask for decoder.

    In a causal mask, each token can only attend to previous tokens and itself,
    preventing information flow from future tokens.
    """
    mask = torch.triu(
        torch.ones((seq_length, seq_length), device=device) * float('-inf'),
        diagonal=1
    )
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return mask


def create_mixed_attention_mask(
    batch_size: int,
    prefix_length: int,
    suffix_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Creates a mixed attention mask: bidirectional for prefix, causal for suffix.
    """
    total_length = prefix_length + suffix_length
    mask = torch.zeros((total_length, total_length), device=device)

    if suffix_length > 0:
        suffix_mask = torch.triu(
            torch.ones((suffix_length, suffix_length), device=device) * float('-inf'),
            diagonal=1
        )
        mask[prefix_length:, prefix_length:] = suffix_mask

    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return mask


class FAST(BASE):
    """
    FAST model with FAST tokenizer, inheriting from BASE.

    This model uses:
    - Florence-2 VLM as the backbone (encoder-decoder architecture)
    - FAST tokenizer for converting actions to/from discrete tokens
    - Autoregressive decoding for generating action tokens
    - Action denormalization for real-world deployment
    """

    def __init__(
        self,
        *args,
        max_action_tokens: int = 256,
        # 动作归一化配置 (由 action_stats 注入)
        action_min: List[float] = None,
        action_max: List[float] = None,
        # FAST tokenizer
        fast_tokenizer: DictConfig = None,
        # arm_mode for denormalization
        arm_mode: str = "left",
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.max_action_tokens = max_action_tokens
        self.arm_mode = arm_mode

        # 动作归一化 buffer (推理时反归一化)
        if action_min is None or action_max is None:
            raise ValueError("需要配置 action_min 和 action_max")
        self.register_buffer('action_min', torch.tensor(action_min, dtype=torch.float32))
        self.register_buffer('action_max', torch.tensor(action_max, dtype=torch.float32))

        # Initialize FAST tokenizer
        if fast_tokenizer is not None:
            self.action_tokenizer = hydra.utils.instantiate(fast_tokenizer)
        else:
            self.action_tokenizer = FAST_Tokenizer(
                num_dof=self.action_dim,
                seq_len=self.act_window_size,
            )

        # Update tokenizer with VLM vocab size
        self.action_tokenizer.update_vlm_vocab_size(self.vlm_vocab_size)

        # Set VLM tokenizer for prefix/suffix encoding
        self.action_tokenizer.set_vlm_tokenizer(self.tokenizer)

        print(f"FAST tokenizer: {self.action_tokenizer.fast_vocab_size} action tokens "
              f"mapped to VLM tokens [{self.vlm_vocab_size - self.action_tokenizer.fast_vocab_size + 1}, {self.vlm_vocab_size}]")

        # Store action dimensions for decoding
        self.num_dof = self.action_dim
        self.seq_len = self.act_window_size

        # Store special token IDs
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.decoder_start_token_id = self.vlm.config.text_config.decoder_start_token_id

    # ---------------------------------------------------------------------
    #                            工具函数
    # ---------------------------------------------------------------------
    def denormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """
        将归一化的动作 [-1, 1] 反归一化到原始范围 [action_min, action_max]。

        公式: action = (normalized + 1) / 2 * (max - min) + min

        支持维度自动适配：根据 arm_mode 切片 action_min/action_max。
        """
        action_min = self.action_min.to(normalized_actions.device)
        action_max = self.action_max.to(normalized_actions.device)

        output_dim = normalized_actions.shape[-1]
        stats_dim = action_min.shape[-1]

        if output_dim != stats_dim:
            arm_mode = getattr(self, 'arm_mode', 'left')
            if arm_mode == 'right':
                action_min = action_min[-output_dim:]
                action_max = action_max[-output_dim:]
            else:
                action_min = action_min[:output_dim]
                action_max = action_max[:output_dim]

        actions = (normalized_actions + 1) / 2 * (action_max - action_min) + action_min
        return actions

    # ---------------------------------------------------------------------
    #                          训练流程
    # ---------------------------------------------------------------------
    def token_prediction_accuracy(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> float:
        """Computes token-level prediction accuracy, optionally with a mask."""
        if mask is not None:
            correct = ((preds == targets) & mask).sum().item()
            total = mask.sum().item()
        else:
            correct = (preds == targets).sum().item()
            total = targets.numel()

        return 100.0 * correct / total if total > 0 else 0.0

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """Lightning training step with autoregressive loss"""
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        training_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0
        token_predict_accuracy = 0.0

        for modality_scope, dataset_batch in batch.items():
            self.modality_scope = modality_scope
            llm_output_dict = self.compute_llm_outputs(dataset_batch)

            act_loss = llm_output_dict["reconstruct_action_mse"]
            predict_accuracy = llm_output_dict["token_predict_accuracy"]
            llm_loss = llm_output_dict["llm_loss"]

            action_loss = action_loss + act_loss
            total_bs = total_bs + len(dataset_batch["actions"])
            training_loss = training_loss + llm_loss
            token_predict_accuracy = token_predict_accuracy + predict_accuracy

        token_predict_accuracy = token_predict_accuracy / len(batch)
        action_loss = action_loss / len(batch)
        training_loss = training_loss / len(batch)

        self._log_training_metrics(
            llm_loss=training_loss,
            token_pred_acc=token_predict_accuracy,
            reconstruct_mse=action_loss,
            total_bs=total_bs,
            token_acc_prefix=llm_output_dict.get("token_acc_prefix"),
            token_acc_action=llm_output_dict.get("token_acc_action"),
            token_acc_suffix=llm_output_dict.get("token_acc_suffix"),
        )

        return training_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Lightning validation step"""
        output = {}
        with torch.no_grad():
            llm_output_dict = self.compute_llm_outputs(batch)

            self._log_validation_metrics(
                llm_loss=llm_output_dict["llm_loss"],
                token_pred_acc=llm_output_dict["token_predict_accuracy"],
                reconstruct_mse=llm_output_dict["reconstruct_action_mse"],
                token_acc_prefix=llm_output_dict.get("token_acc_prefix"),
                token_acc_action=llm_output_dict.get("token_acc_action"),
                token_acc_suffix=llm_output_dict.get("token_acc_suffix"),
            )

            output["validation_loss"] = llm_output_dict["llm_loss"] / len(batch)
            return output

    def compute_llm_outputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute LLM outputs with autoregressive training.
        """
        features, encoder_attn_mask = self.compute_input_features(batch)

        if "actions" not in batch.keys():
            raise ValueError("Actions required for training")

        # Encode actions to FAST tokens
        action_tokens, token_lengths, _ = self.action_tokenizer.encode(
            batch["actions"],
            return_lengths=True
        )

        # Build complete training sequence with prefix + action + suffix
        decoder_input_ids, labels, loss_mask = self.action_tokenizer.build_training_sequence(
            action_tokens=action_tokens,
            token_lengths=token_lengths,
            pad_token_id=self.pad_token_id,
        )

        batch_size, seq_len = labels.shape

        # Create causal attention mask for autoregressive decoding
        causal_mask = create_causal_mask(
            batch_size=batch_size,
            seq_length=seq_len,
            device=self.device
        )

        # Run decoder with teacher forcing
        decoder_outputs = self.vlm.get_decoder()(
            input_ids=decoder_input_ids,
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=causal_mask,
            use_cache=False,
        )

        # Get logits
        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)

        # Compute cross-entropy loss with masking
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            lm_logits.view(-1, self.vlm.config.vocab_size),
            labels.view(-1),
        ).view(batch_size, seq_len)

        # Apply loss mask
        masked_loss = per_token_loss * loss_mask.float()
        total_loss = masked_loss.sum() / loss_mask.sum()

        # Compute token prediction accuracy
        pred_tokens = torch.argmax(lm_logits, dim=-1)
        token_predict_accuracy = self.token_prediction_accuracy(
            pred_tokens, labels, loss_mask
        )

        # Split token accuracy into prefix / action / suffix
        prefix_len = len(self.action_tokenizer.prefix_token_ids)
        suffix_len = len(self.action_tokenizer.suffix_token_ids)

        prefix_mask = torch.zeros_like(loss_mask)
        suffix_mask = torch.zeros_like(loss_mask)
        action_mask = torch.zeros_like(loss_mask)

        for i in range(batch_size):
            action_len = int(token_lengths[i].item())
            valid_len = prefix_len + action_len + suffix_len
            if valid_len <= 0:
                continue
            prefix_mask[i, :prefix_len] = True
            action_mask[i, prefix_len:prefix_len + action_len] = True
            suffix_mask[i, prefix_len + action_len:valid_len] = True

        prefix_mask = prefix_mask & loss_mask
        action_mask = action_mask & loss_mask
        suffix_mask = suffix_mask & loss_mask

        token_acc_prefix = self.token_prediction_accuracy(pred_tokens, labels, prefix_mask)
        token_acc_action = self.token_prediction_accuracy(pred_tokens, labels, action_mask)
        token_acc_suffix = self.token_prediction_accuracy(pred_tokens, labels, suffix_mask)

        # Compute reconstruction error
        with torch.no_grad():
            extracted_action_tokens = self.action_tokenizer.extract_action_tokens_from_generated(
                pred_tokens
            )
            reconstructed = self.action_tokenizer.decode(
                extracted_action_tokens,
                action_horizon=self.seq_len,
                action_dim=self.num_dof
            )
            action_mse = F.mse_loss(reconstructed, batch["actions"])

        return {
            'llm_loss': total_loss,
            'token_predict_accuracy': token_predict_accuracy,
            'token_acc_prefix': token_acc_prefix,
            'token_acc_action': token_acc_action,
            'token_acc_suffix': token_acc_suffix,
            'reconstruct_action_mse': action_mse,
        }

    def llm_generate_autoregressive(self, batch: Dict, max_length: int = None) -> torch.Tensor:
        """
        Generate action tokens autoregressively.
        """
        if max_length is None:
            max_length = self.max_action_tokens

        features, encoder_attn_mask = self.compute_input_features(batch)
        batch_size = features.shape[0]

        # Get prefix tokens from tokenizer
        prefix_tokens = torch.tensor(
            self.action_tokenizer.prefix_token_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)

        # Get stopping criteria
        suffix_first_token = self.action_tokenizer.suffix_token_ids[0]
        eos_token = self.eos_token_id

        # Start with BOS token + prefix tokens
        bos_token = torch.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=torch.long,
            device=self.device
        )
        generated = torch.cat([bos_token, prefix_tokens], dim=1)

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Process the prefix tokens to build cache
        decoder_outputs = self.vlm.get_decoder()(
            input_ids=generated,
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            use_cache=True,
        )
        past_key_values = decoder_outputs.past_key_values

        # Get first action token
        next_logits = self.vlm.language_model.get_output_embeddings()(
            decoder_outputs.last_hidden_state[:, -1:]
        )
        next_logits = next_logits + self.vlm.language_model.final_logits_bias.to(next_logits.device)
        next_token = torch.argmax(next_logits, dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)

        # Continue generating action tokens
        for step in range(max_length - 1):
            last_token = generated[:, -1]
            just_finished = (last_token == suffix_first_token) | (last_token == eos_token)
            finished = finished | just_finished

            if finished.all():
                break

            decoder_outputs = self.vlm.get_decoder()(
                input_ids=generated[:, -1:],
                encoder_hidden_states=features,
                encoder_attention_mask=encoder_attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = decoder_outputs.past_key_values

            next_logits = self.vlm.language_model.get_output_embeddings()(
                decoder_outputs.last_hidden_state[:, -1:]
            )
            next_logits = next_logits + self.vlm.language_model.final_logits_bias.to(next_logits.device)

            next_token = torch.argmax(next_logits, dim=-1)

            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, eos_token),
                next_token
            )

            generated = torch.cat([generated, next_token], dim=-1)

        return generated

    # ---------------------------------------------------------------------
    #                          推理接口
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Forward pass for inference.
        Returns denormalized actions.
        """
        # 构建 batch
        batch = {
            "rgb_obs": {},
            "lang_text": [goal["lang_text"]] if isinstance(goal["lang_text"], str) else goal["lang_text"]
        }

        for cam_key in self.rgb_obs_keys:
            if cam_key in obs.get("rgb_obs", {}):
                batch["rgb_obs"][cam_key] = obs["rgb_obs"][cam_key]
            elif cam_key in obs:
                batch["rgb_obs"][cam_key] = obs[cam_key]

        # Generate complete token sequence autoregressively
        generated_tokens = self.llm_generate_autoregressive(batch)

        # Extract action tokens from generated sequence
        action_tokens = self.action_tokenizer.extract_action_tokens_from_generated(
            generated_tokens
        )

        # Decode FAST tokens to continuous actions (normalized)
        actions_normalized = self.action_tokenizer.decode(
            action_tokens,
            action_horizon=self.seq_len,
            action_dim=self.num_dof
        )

        # 反归一化到原始动作空间
        actions = self.denormalize_actions(actions_normalized)

        return actions

    # ---------------------------------------------------------------------
    #                          日志方法
    # ---------------------------------------------------------------------
    def _log_training_metrics(
        self,
        llm_loss,
        token_pred_acc,
        reconstruct_mse,
        total_bs,
        token_acc_prefix=None,
        token_acc_action=None,
        token_acc_suffix=None,
    ):
        """Log training metrics"""
        self.log("train/llm_loss", llm_loss, on_step=False, on_epoch=True,
                sync_dist=True, batch_size=total_bs)
        self.log("train/token_pred_acc", token_pred_acc, on_step=False, on_epoch=True,
                sync_dist=True, batch_size=total_bs)
        if token_acc_prefix is not None:
            self.log("train/token_acc_prefix", token_acc_prefix, on_step=False, on_epoch=True,
                    sync_dist=True, batch_size=total_bs)
        if token_acc_action is not None:
            self.log("train/token_acc_action", token_acc_action, on_step=False, on_epoch=True,
                    sync_dist=True, batch_size=total_bs)
        if token_acc_suffix is not None:
            self.log("train/token_acc_suffix", token_acc_suffix, on_step=False, on_epoch=True,
                    sync_dist=True, batch_size=total_bs)
        self.log("train/reconstruct_mse", reconstruct_mse, on_step=False, on_epoch=True,
                sync_dist=True, batch_size=total_bs)

    def _log_validation_metrics(
        self,
        llm_loss,
        token_pred_acc,
        reconstruct_mse,
        token_acc_prefix=None,
        token_acc_action=None,
        token_acc_suffix=None,
    ):
        """Log validation metrics"""
        self.log(f"val/{self.modality_scope}_llm_loss", llm_loss, sync_dist=True)
        self.log("val/token_pred_acc", token_pred_acc, sync_dist=True)
        if token_acc_prefix is not None:
            self.log("val/token_acc_prefix", token_acc_prefix, sync_dist=True)
        if token_acc_action is not None:
            self.log("val/token_acc_action", token_acc_action, sync_dist=True)
        if token_acc_suffix is not None:
            self.log("val/token_acc_suffix", token_acc_suffix, sync_dist=True)
        self.log("val/reconstruct_mse", reconstruct_mse, sync_dist=True)

    def get_model_info(self) -> Dict[str, any]:
        """获取模型配置信息。"""
        return {
            "model_type": "FAST",
            "action_dim": self.action_dim,
            "seq_len": self.seq_len,
            "rgb_obs_keys": self.rgb_obs_keys,
        }
