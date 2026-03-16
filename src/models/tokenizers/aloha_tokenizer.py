# src/models/tokenizers/aloha_tokenizer.py
"""
ALOHA双臂动作分词器
支持14维双臂动作 (7+7) 的编码和解码
"""
import torch
import torch.nn as nn
from typing import List, Optional

from src.models.tokenizers.base_tokenizer import TokenizerBase


class AlohaActionTokenizer(TokenizerBase):
    """
    ALOHA双臂动作分词器
    - 输入: actions [B, T, 14]
    - 输出: tokens [B, 14, num_basis]
    
    使用简单的时间重采样方法将动作序列编码为固定长度的token序列
    """
    
    def __init__(
        self,
        num_dof: int = 14,
        num_basis: int = 10,
        seq_len: int = 20,
        vocab_size: int = 256,
        action_min: Optional[List[float]] = None,
        action_max: Optional[List[float]] = None,
        init_pos: bool = False,
        gripper_dof: int = 2,  # 双臂有2个夹爪
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        
        self.num_dof = int(num_dof)
        self.joint_dof = self.num_dof
        self.num_basis = int(num_basis)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.init_pos = bool(init_pos)
        self.gripper_dof = gripper_dof
        self.device = device
        
        # 动作范围
        if action_min is None:
            action_min = [-2.0] * self.num_dof
        if action_max is None:
            action_max = [2.0] * self.num_dof
            
        # 注册为buffer以便自动移动到正确设备
        self.register_buffer("w_min", torch.tensor(action_min, dtype=torch.float32))
        self.register_buffer("w_max", torch.tensor(action_max, dtype=torch.float32))
        
        # VLM词表映射
        self.vlm_vocab_size = None
        self.llm_token_offset = 0
        
    def update_vlm_vocab_size(self, vlm_vocab_size: int):
        """更新VLM词表大小"""
        self.vlm_vocab_size = int(vlm_vocab_size)
        self.llm_token_offset = max(0, self.vlm_vocab_size - self.vocab_size)
    
    @property
    def action_min(self) -> torch.Tensor:
        return self.w_min
    
    @property
    def action_max(self) -> torch.Tensor:
        return self.w_max
    
    def _resample_time(self, actions: torch.Tensor) -> torch.Tensor:
        """
        将动作序列重采样到固定长度
        actions: [B, T, D] -> samples: [B, D, num_basis]
        """
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        B, T, D = actions.shape
        
        if T == self.num_basis:
            samples = actions
        else:
            idx = torch.linspace(0, T - 1, self.num_basis, device=actions.device)
            idx = idx.round().long().clamp(0, T - 1)
            samples = actions[:, idx, :]
            
        return samples.transpose(1, 2).contiguous()  # [B, D, num_basis]
    
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        将动作归一化到 [-1, 1] 范围
        actions: [..., D]
        """
        w_min = self.w_min.to(actions.device)
        w_max = self.w_max.to(actions.device)
        return 2.0 * (actions - w_min) / (w_max - w_min + 1e-8) - 1.0
    
    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        将归一化动作还原到原始范围
        actions: [..., D]
        """
        w_min = self.w_min.to(actions.device)
        w_max = self.w_max.to(actions.device)
        return (actions + 1.0) / 2.0 * (w_max - w_min) + w_min
    
    def encode(self, actions: torch.Tensor, update_bounds: bool = False) -> tuple:
        """
        将动作序列编码为token
        actions: [B, T, D] -> tokens: [B, (num_basis * D)]
        
        返回: (tokens, params_dict) 与 BSpline_Tokenizer 接口对齐
        """
        # 重采样到固定长度
        samples = self._resample_time(actions)  # [B, D, num_basis]
        # 归一化
        normalized = self.normalize_actions(samples.transpose(1, 2)).transpose(1, 2)
        # 量化到离散token
        tokens = ((normalized + 1.0) / 2.0 * (self.vocab_size - 1)).round().long()
        tokens = tokens.clamp(0, self.vocab_size - 1)
        
        # 重排为 [B, (num_basis * D)] 格式，与 BSpline_Tokenizer 对齐
        # BSpline: 'b (d t) -> b (t d)' 即 [B, num_basis, num_dof]
        tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], -1)  # [B, num_basis * D]
        
        # 返回格式与 BSpline_Tokenizer 对齐
        params_dict = {'params': normalized.reshape(normalized.shape[0], -1)}  # [B, D * num_basis]
        return tokens, params_dict
    
    def tokens_to_llm_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        将 MP tokens 转换为 LLM tokens
        使用反向映射: llm_token = vlm_vocab_size - 1 - mp_token
        """
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab size is not set. Call update_vlm_vocab_size() first.")
        return self.vlm_vocab_size - 1 - tokens
    
    def llm_tokens_to_mp_tokens(self, llm_tokens: torch.Tensor) -> torch.Tensor:
        """
        将 LLM tokens 转换回 MP tokens
        """
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab size is not set.")
        return self.vlm_vocab_size - 1 - llm_tokens
    
    def reconstruct_from_llm_tokens(self, llm_tokens: torch.Tensor, times=None, init_p=None, **kwargs) -> torch.Tensor:
        """
        从 LLM tokens 重建动作序列
        llm_tokens: [B, num_basis * D] -> actions: [B, T, D]
        """
        # 转换为 MP tokens
        mp_tokens = self.llm_tokens_to_mp_tokens(llm_tokens)
        mp_tokens = mp_tokens.clamp(0, self.vocab_size - 1)
        
        # 重排为 [B, D, num_basis]
        B = mp_tokens.shape[0]
        mp_tokens = mp_tokens.reshape(B, self.num_basis, self.num_dof).transpose(1, 2)  # [B, D, num_basis]
        
        # 反量化
        normalized = mp_tokens.float() / (self.vocab_size - 1) * 2.0 - 1.0
        
        # 反归一化
        samples = self.denormalize_actions(normalized.transpose(1, 2)).transpose(1, 2)  # [B, D, num_basis]
        
        # 重采样到目标序列长度
        D, N = samples.shape[1], samples.shape[2]
        if N == self.seq_len:
            actions = samples.transpose(1, 2)
        else:
            idx = torch.linspace(0, N - 1, self.seq_len, device=samples.device)
            idx = idx.round().long().clamp(0, N - 1)
            actions = samples[:, :, idx].transpose(1, 2)
        
        return actions  # [B, T, D]
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        前向传播：编码然后解码，用于验证
        """
        tokens = self.encode(actions)
        return self.decode(tokens)
