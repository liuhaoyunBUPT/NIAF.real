"""
OFT: OpenVLA-OFT model implementation.

This module extends BEAST to support:
1. Flexible multi-camera configurations (rgb_obs_keys)
2. Action denormalization for relative/absolute actions
3. Camera flip options
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from src.models.base import BASE, create_bidirectional_mask

logger = logging.getLogger(__name__)


class OFT(BASE):
    """
    OFT: OpenVLA-OFT model with multi-camera and denormalization support.
    
    This model uses Florence-2 to encode observations, then uses "empty" query embeddings
    passed to the decoder (which adds positional encoding) to generate action embeddings.
    These embeddings are projected via a multi-layer MLP to continuous actions (regression).
    
    Key features:
    1. Decoder Input: Empty query embeddings (zeros) + internal PE.
    2. Output Head: MLP mapping hidden_dim -> action_dim.
    3. Loss: L1 Loss for actions.
    4. Multi-camera support.
    5. Action denormalization for real-world deployment.
    """
    
    def __init__(
        self,
        *args,
        # OFT specific config
        oft_mlp_depth: int = 4,
        # 动作归一化配置 (由 action_stats 注入)
        action_min: List[float] = None,
        action_max: List[float] = None,
        **kwargs
    ):
        """
        初始化 OFT 模型。

        参数:
            oft_mlp_depth: OFT MLP 层数
            action_min: 当前动作模式对应的最小值 (逐维度)
            action_max: 当前动作模式对应的最大值 (逐维度)
        """
        super().__init__(*args, **kwargs)
        
        # 动作归一化 buffer (推理时反归一化)
        if action_min is None or action_max is None:
            raise ValueError("需要配置 action_min 和 action_max")
        self.register_buffer('action_min', torch.tensor(action_min, dtype=torch.float32))
        self.register_buffer('action_max', torch.tensor(action_max, dtype=torch.float32))

        # OFT MLP 配置
        self.oft_mlp_depth = oft_mlp_depth
        
        hidden_dim = self.vlm.config.text_config.d_model
        self.hidden_dim = hidden_dim
        
        # MLP Construction: Linear -> GELU -> ... -> Linear
        layers = []
        for i in range(oft_mlp_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        
        # Last layer projects to action_dim
        layers.append(nn.Linear(hidden_dim, self.action_dim))
        
        self.action_head = nn.Sequential(*layers)
        
        # Initialize MLP weights
        self._init_mlp_weights()
        
    def _init_mlp_weights(self):
        """Initialize MLP weights using Xavier initialization."""
        for m in self.action_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------
    #                            工具函数
    # ---------------------------------------------------------------------
    def denormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """
        将归一化的动作 [-1, 1] 反归一化到原始范围 [action_min, action_max]。
        
        公式: action = (normalized + 1) / 2 * (max - min) + min
        
        支持维度自动适配：当模型输出维度与统计量维度不匹配时，
        自动根据输出维度切片 action_min/action_max。

        参数:
            normalized_actions: 归一化的动作张量，形状为 (..., action_dim)

        返回:
            反归一化后的动作张量
        """
        # 确保 action_min 和 action_max 在正确的设备上
        action_min = self.action_min.to(normalized_actions.device)
        action_max = self.action_max.to(normalized_actions.device)
        
        # 自动适配维度：如果模型输出维度与统计量维度不匹配，进行切片
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
                
        # 反归一化: (y + 1) / 2 * (max - min) + min
        actions = (normalized_actions + 1) / 2 * (action_max - action_min) + action_min
        return actions

    # ---------------------------------------------------------------------
    #                          训练流程
    # ---------------------------------------------------------------------
    def compute_llm_outputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training.
        Override BEAST method to use OFT regression logic.
        """
        features, encoder_attn_mask = self.compute_input_features(batch)
        
        batch_size = features.shape[0]
        device = features.device
        
        # Ensure 'actions' is in batch for supervision
        if "actions" not in batch:
            raise ValueError("Batch must contain 'actions' for training.")
            
        target_actions = batch["actions"]  # (B, T, D)
        seq_len = target_actions.shape[1]
        
        # Decoder inputs: empty query embeddings (B, T, H)
        dtype = self.vlm.dtype
        empty_queries = torch.zeros((batch_size, seq_len, self.hidden_dim), device=device, dtype=dtype)
        
        # Bidirectional attention mask for queries
        decoder_bidirectional_mask = create_bidirectional_mask(batch_size, seq_len, device)
        
        # Forward pass through Decoder
        decoder_outputs = self.vlm.get_decoder()(
            inputs_embeds=empty_queries,
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=decoder_bidirectional_mask,
            use_cache=False 
        )
        
        # decoder_outputs.last_hidden_state: (B, T, H)
        last_hidden_state = decoder_outputs[0]
        
        # Project to Actions
        pred_actions = self.action_head(last_hidden_state)  # (B, T, D)
        
        # Compute Loss (L1)
        loss = F.l1_loss(pred_actions, target_actions)
        
        return {
            'llm_loss': loss,
            'token_predict_accuarcy': 0.0,
            'reconstruct_action_mse': loss.detach(),
            'pred_actions': pred_actions
        }

    def llm_generates(self, batch: Dict) -> torch.Tensor:
        """
        Inference generation.
        Returns action sequence directly (normalized).
        """
        features, encoder_attn_mask = self.compute_input_features(batch)
        batch_size = features.shape[0]
        device = features.device
        dtype = self.vlm.dtype
        
        seq_len = self.act_window_size
        
        empty_queries = torch.zeros((batch_size, seq_len, self.hidden_dim), device=device, dtype=dtype)
        decoder_bidirectional_mask = create_bidirectional_mask(batch_size, seq_len, device)
        
        decoder_outputs = self.vlm.get_decoder()(
            inputs_embeds=empty_queries,
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=decoder_bidirectional_mask,
            use_cache=False
        )
        
        last_hidden_state = decoder_outputs[0]
        pred_actions = self.action_head(last_hidden_state)  # (B, T, D)
        
        return pred_actions

    # ---------------------------------------------------------------------
    #                          推理接口
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Forward pass for inference.
        Supports flexible multi-camera configurations for ALOHA.
        Returns denormalized actions.
        """
        # 构建 batch，支持灵活的相机配置
        batch = {
            "rgb_obs": {},
            "lang_text": [goal["lang_text"]] if isinstance(goal["lang_text"], str) else goal["lang_text"]
        }
        
        # 根据配置添加相机
        if hasattr(self, 'rgb_obs_keys') and self.rgb_obs_keys:
            for cam_key in self.rgb_obs_keys:
                if cam_key in obs.get("rgb_obs", {}):
                    batch["rgb_obs"][cam_key] = obs["rgb_obs"][cam_key]
                elif cam_key in obs:
                    batch["rgb_obs"][cam_key] = obs[cam_key]
        else:
            # 默认 Calvin 风格的相机键
            if "rgb_obs" in obs:
                batch["rgb_obs"] = obs["rgb_obs"]
            else:
                if "rgb_static" in obs:
                    batch["rgb_obs"]["rgb_static"] = obs["rgb_static"]
                if "rgb_gripper" in obs:
                    batch["rgb_obs"]["rgb_gripper"] = obs["rgb_gripper"]
        
        # 生成归一化动作
        pred_actions_normalized = self.llm_generates(batch)
        
        # 反归一化到原始动作空间
        pred_actions = self.denormalize_actions(pred_actions_normalized)
        
        return pred_actions


    # ---------------------------------------------------------------------
    #                        调试辅助方法
    # ---------------------------------------------------------------------
    def get_model_info(self) -> Dict[str, any]:
        """
        获取模型配置信息。
        """
        return {
            "model_type": "OFT",
            "oft_mlp_depth": self.oft_mlp_depth,
            "action_dim": self.action_dim,
        }
