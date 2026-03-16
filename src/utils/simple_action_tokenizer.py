# beast/utils/simple_action_tokenizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch


@dataclass
class SimpleActionTokenizer:
    """
    最快跑通版 tokenizer：
    - 把 action chunk [B, T, D] 直接量化成 token
    - token layout 按 (D, T) 展平 => 序列长度 = D*T
    - tokens_to_llm_tokens 做一个 vocab 尾部映射，避免与普通文本 token 强冲突

    你只要确保 dataloader 输出 actions 已经大致归一化到 [low, high]（默认 [-1, 1]）即可。
    """
    num_dof: int
    num_basis: int                # 这里用 chunk_size (T)
    vocab_size: int = 256
    low: float = -1.0
    high: float = 1.0

    def __post_init__(self):
        self.joint_dof = self.num_dof
        self._vlm_vocab_size: Optional[int] = None
        # 兼容 BEAST 的 precompute_w_bound（不启用也没关系）
        self.w_min = torch.full((self.num_dof * self.num_basis,), self.low, dtype=torch.float32)
        self.w_max = torch.full((self.num_dof * self.num_basis,), self.high, dtype=torch.float32)

    def update_vlm_vocab_size(self, vlm_vocab_size: int):
        self._vlm_vocab_size = int(vlm_vocab_size)

    # ---------- helpers ----------
    def _quantize(self, x: torch.Tensor) -> torch.LongTensor:
        x = torch.clamp(x, self.low, self.high)
        # [low, high] -> [0, vocab_size-1]
        y = (x - self.low) / (self.high - self.low + 1e-8)
        y = torch.round(y * (self.vocab_size - 1)).long()
        return torch.clamp(y, 0, self.vocab_size - 1)

    def _dequantize(self, t: torch.LongTensor) -> torch.FloatTensor:
        t = torch.clamp(t, 0, self.vocab_size - 1).float()
        y = t / (self.vocab_size - 1)
        x = y * (self.high - self.low) + self.low
        return x

    def _llm_offset(self) -> int:
        if self._vlm_vocab_size is None:
            raise RuntimeError("call update_vlm_vocab_size() before tokens_to_llm_tokens()")
        if self._vlm_vocab_size < self.vocab_size + 10:
            raise RuntimeError(f"vlm_vocab_size({self._vlm_vocab_size}) too small for vocab_size({self.vocab_size})")
        # 把动作 token 映射到 VLM vocab 的尾部区域
        return self._vlm_vocab_size - self.vocab_size

    # ---------- BEAST required API ----------
    def encode(self, actions: torch.Tensor, update_bounds: bool = False) -> Tuple[torch.LongTensor, Dict]:
        """
        actions: [B, T, D]
        return:
          action_tokens: [B, D*T]  (flattened)
          params: dict (for compatibility)
        """
        assert actions.dim() == 3, f"expected [B,T,D], got {actions.shape}"
        B, T, D = actions.shape
        if D != self.num_dof:
            raise ValueError(f"action_dim mismatch: got D={D}, tokenizer.num_dof={self.num_dof}")
        if T != self.num_basis:
            raise ValueError(f"chunk_size mismatch: got T={T}, tokenizer.num_basis={self.num_basis}")

        # [B,T,D] -> [B,D,T]
        w = actions.transpose(1, 2).contiguous()
        tokens = self._quantize(w)              # [B,D,T]
        tokens = tokens.view(B, D * T)          # [B, D*T]
        return tokens, {"params": w}

    def decode(self, action_tokens: torch.LongTensor) -> torch.Tensor:
        """
        action_tokens: [B, D*T] or [D*T]
        return actions: [B, T, D]
        """
        if action_tokens.dim() == 1:
            action_tokens = action_tokens.unsqueeze(0)
        B, N = action_tokens.shape
        D, T = self.num_dof, self.num_basis
        if N != D * T:
            raise ValueError(f"token length mismatch: got {N}, expected {D*T}")
        w = action_tokens.view(B, D, T)
        w = self._dequantize(w)                 # [B,D,T]
        actions = w.transpose(1, 2).contiguous()# [B,T,D]
        return actions

    def tokens_to_llm_tokens(self, action_tokens: torch.LongTensor) -> torch.LongTensor:
        """
        输入可以是 [B,D*T] 或 [B,D,T]，输出必须是 [B,seq_len]
        """
        if action_tokens.dim() == 3:
            B, D, T = action_tokens.shape
            action_tokens = action_tokens.view(B, D * T)
        offset = self._llm_offset()
        return action_tokens + offset

    def llm_tokens_to_tokens(self, llm_tokens: torch.LongTensor) -> torch.LongTensor:
        offset = self._llm_offset()
        return torch.clamp(llm_tokens - offset, 0, self.vocab_size - 1)

    def reconstruct_from_llm_tokens(self, llm_pred_tokens: torch.LongTensor, times=None) -> torch.Tensor:
        """
        llm_pred_tokens: [B, seq_len] (argmax 后)
        return: [B, T, D]
        """
        tokens = self.llm_tokens_to_tokens(llm_pred_tokens)
        return self.decode(tokens)

    # 兼容 precompute（你不开 pre_compute_w_bound 就不会用到）
    def compute_weights(self, act_chunks: torch.Tensor) -> torch.Tensor:
        # act_chunks: [B, T, D] -> flatten as pseudo-weights
        B, T, D = act_chunks.shape
        w = act_chunks.transpose(1, 2).contiguous().view(B, D * T)
        return w
