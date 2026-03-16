"""
FAST Tokenizer for action tokenization in VLA models.

This module implements the FAST (Fast Action tokenization using Subword Tokenization) 
tokenizer for converting continuous robot actions into discrete tokens. Unlike B-spline 
tokenizer which produces fixed-length tokens, FAST uses BPE (Byte Pair Encoding) and 
produces variable-length tokens, requiring autoregressive decoding.

Reference: Physical Intelligence's Pi0-FAST implementation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import einops
import contextlib
import io
from typing import Optional, Tuple, Union, List
from functools import wraps

from transformers import AutoProcessor

from src.models.tokenizers.base_tokenizer import TokenizerBase


def autocast_float32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            return fn(*args, **kwargs)
    return wrapped


class FAST_Tokenizer(TokenizerBase):
    """
    FAST Tokenizer for encoding continuous actions into discrete tokens.
    
    Unlike B-spline tokenizer which produces fixed-length tokens (suitable for 
    bidirectional attention + learnable queries), FAST uses BPE-based tokenization
    resulting in variable-length tokens, requiring autoregressive decoding.
    
    Token Mapping Strategy (same as BSpline):
    =========================================
    Maps FAST tokens to the END of VLM vocabulary by overwriting the last 
    fast_vocab_size positions. This is consistent with BSpline tokenizer.
    
    Mapping: vlm_token = vlm_vocab_size - 1 - fast_token
    FAST tokens [0, fast_vocab_size-1] -> VLM tokens [vlm_vocab_size-1, vlm_vocab_size-fast_vocab_size]
    
    Note: This overwrites Florence-2's special tokens (<loc_xxx>, <cap>, etc.),
    which is acceptable if not using detection/grounding features.
    
    Args:
        num_dof: Number of degrees of freedom (action dimension).
        seq_len: Action sequence length (time horizon).
        fast_tokenizer_path: Path to the FAST tokenizer model on HuggingFace.
        fast_vocab_size: FAST tokenizer vocabulary size (default 2048).
        device: Device to run the tokenizer on.
    """

    def __init__(
        self,
        num_dof: int = 7,
        seq_len: int = 50,
        fast_tokenizer_path: str = "physical-intelligence/fast",
        fast_vocab_size: int = 2048,  # FAST tokenizer vocab size
        device: str = "cuda",
        suppress_decode_prints: bool = True,
    ):
        super().__init__()
        
        self.num_dof = num_dof  # action dimension
        self.seq_len = seq_len  # action horizon / time steps
        self.device = device
        
        # FAST vocabulary size (number of unique action tokens)
        self.fast_vocab_size = fast_vocab_size
        self.suppress_decode_prints = suppress_decode_prints
        
        # Initialize FAST tokenizer from HuggingFace
        self._fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, 
            trust_remote_code=True
        )
        
        # VLM vocabulary size - will be updated by the model
        self.vlm_vocab_size = None
        
        # VLM tokenizer reference - set by the model for prefix/suffix encoding
        self._vlm_tokenizer = None
        
        # Prefix and suffix tokens (like Pi0-FAST's "Action: " and "|" + EOS)
        self.action_prefix_text = "Action:"
        self.action_suffix_text = "|"
        self.prefix_token_ids = None  # Will be set when vlm_tokenizer is available
        self.suffix_token_ids = None  # Includes EOS token
        self.eos_token_id = None
        
        # Track statistics for normalization (actions should be in [-1, 1])
        self.register_buffer("action_min", -torch.ones(num_dof))
        self.register_buffer("action_max", torch.ones(num_dof))

    def update_vlm_vocab_size(self, vlm_vocab_size: int):
        """
        Update the VLM vocabulary size for token mapping.
        
        Uses the same mapping strategy as BSpline tokenizer:
        Maps action tokens to the end of VLM vocabulary.
        
        Args:
            vlm_vocab_size: VLM vocabulary size (config.vocab_size - 1, same as BSpline)
        """
        self.vlm_vocab_size = vlm_vocab_size
    
    def set_vlm_tokenizer(self, vlm_tokenizer):
        """
        Set the VLM tokenizer for encoding prefix/suffix text tokens.
        
        This enables creating complete action token sequences with:
        - Prefix: "Action:" text tokens
        - Action: FAST action tokens mapped to VLM vocabulary
        - Suffix: "|" + EOS token
        
        Args:
            vlm_tokenizer: The VLM's text tokenizer (e.g., Florence-2's tokenizer)
        """
        self._vlm_tokenizer = vlm_tokenizer
        self.eos_token_id = vlm_tokenizer.eos_token_id
        
        # Encode prefix text "Action:"
        prefix_encoding = vlm_tokenizer.encode(
            self.action_prefix_text, 
            add_special_tokens=False
        )
        self.prefix_token_ids = prefix_encoding
        
        # Encode suffix text "|" + EOS
        suffix_encoding = vlm_tokenizer.encode(
            self.action_suffix_text, 
            add_special_tokens=False
        )
        # Add EOS token at the end
        self.suffix_token_ids = suffix_encoding + [self.eos_token_id]
        
        print(f"FAST Tokenizer prefix tokens: {self.prefix_token_ids} ('{self.action_prefix_text}')")
        print(f"FAST Tokenizer suffix tokens: {self.suffix_token_ids} ('{self.action_suffix_text}' + EOS)")
        
    @torch.no_grad()
    def encode(
        self, 
        actions: torch.Tensor, 
        update_bounds: bool = False,
        return_lengths: bool = False
    ) -> Union[Tuple[torch.Tensor, dict], Tuple[torch.Tensor, torch.Tensor, dict]]:
        """
        Encode continuous actions into discrete FAST tokens.
        
        Args:
            actions: Continuous actions tensor of shape (batch_size, seq_len, num_dof)
                     Actions should be normalized to [-1, 1] range.
            update_bounds: Whether to update action bounds (not used for FAST).
            return_lengths: Whether to return the actual token lengths.
            
        Returns:
            tokens: Discrete tokens of shape (batch_size, max_token_len)
            lengths: (Optional) Actual token lengths for each sample (batch_size,)
            params_dict: Dictionary containing additional info (empty for FAST)
        """
        device = actions.device
        dtype = actions.dtype
        batch_size = actions.shape[0]
        
        # Ensure actions are on CPU and numpy for FAST tokenizer
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy().astype(np.float32)
        else:
            actions_np = actions.astype(np.float32)
        
        # Tokenize using FAST tokenizer
        # FAST tokenizer expects shape (batch_size, seq_len, action_dim)
        all_tokens = []
        max_len = 0
        
        for i in range(batch_size):
            # FAST tokenizer expects a batch, so wrap single sample
            tokens = self._fast_tokenizer(actions_np[i:i+1])[0]
            all_tokens.append(tokens)
            max_len = max(max_len, len(tokens))
        
        # Pad tokens to same length
        # Note: We use a special value for padding since 0 is a valid FAST token.
        # We use -1 for padding which will be clipped to 0 if accidentally decoded,
        # but the lengths tensor should be used to identify valid tokens.
        PAD_VALUE = -1  # Not a valid FAST token, used for internal padding
        padded_tokens = np.full((batch_size, max_len), PAD_VALUE, dtype=np.int64)
        lengths = np.zeros(batch_size, dtype=np.int64)
        
        for i, tokens in enumerate(all_tokens):
            lengths[i] = len(tokens)
            padded_tokens[i, :len(tokens)] = tokens

        # Keep -1 padding in the returned tensor. Downstream code should use token_lengths
        # when constructing labels; decode() will filter negative values.
        
        tokens_tensor = torch.from_numpy(padded_tokens).to(device)
        lengths_tensor = torch.from_numpy(lengths).to(device)
        
        params_dict = {
            'original_actions': actions,
            'token_lengths': lengths_tensor,
        }
        
        if return_lengths:
            return tokens_tensor, lengths_tensor, params_dict
        return tokens_tensor, params_dict

    def tokens_to_llm_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Map FAST tokens to VLM vocabulary tokens.
        
        Same mapping strategy as BSpline tokenizer:
        Maps to the end of VLM vocabulary.
        
        Mapping: vlm_token = vlm_vocab_size - 1 - fast_token
        FAST tokens [0, fast_vocab_size-1] -> VLM tokens [vlm_vocab_size-1, vlm_vocab_size-fast_vocab_size]
        
        Args:
            tokens: FAST tokens of shape (batch_size, seq_len)
            
        Returns:
            VLM tokens of shape (batch_size, seq_len)
        """
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab size is not set. Call update_vlm_vocab_size first.")
        
        # Ensure tokens are in-range before mapping.
        # - Negative values (e.g., -1 padding) are treated as 0.
        # - Values >= fast_vocab_size are clamped.
        tokens = torch.clamp(tokens, min=0, max=self.fast_vocab_size - 1)

        # Same mapping as BSpline: vlm_token = vlm_vocab_size - 1 - token
        llm_tokens = self.vlm_vocab_size - 1 - tokens
        return llm_tokens

    def llm_tokens_to_fast_tokens(self, llm_tokens: torch.Tensor) -> torch.Tensor:
        """
        Map VLM vocabulary tokens back to FAST tokens.
        
        Note: Tokens that are not valid action tokens (e.g., prefix/suffix/padding)
        will be mapped to -1 to indicate they should be filtered out.
        
        Args:
            llm_tokens: VLM tokens of shape (batch_size, seq_len)
            
        Returns:
            FAST tokens of shape (batch_size, seq_len)
            Invalid tokens are marked with -1
        """
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab size is not set. Call update_vlm_vocab_size first.")
        
        # Reverse mapping: fast_token = vlm_vocab_size - 1 - vlm_token
        fast_tokens = self.vlm_vocab_size - 1 - llm_tokens
        
        # Mark out-of-range tokens as -1 (invalid/padding marker)
        # Valid FAST tokens should be in [0, fast_vocab_size - 1]
        invalid_mask = (fast_tokens < 0) | (fast_tokens >= self.fast_vocab_size)
        fast_tokens = torch.where(invalid_mask, torch.tensor(-1, device=fast_tokens.device), fast_tokens)
        
        return fast_tokens

    def build_training_sequence(
        self, 
        action_tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build complete training sequence with prefix and suffix tokens.
        
        Following Pi0-FAST approach:
        - Prefix: "Action:" text tokens
        - Body: Action tokens (mapped to VLM vocabulary)
        - Suffix: "|" + EOS token
        
        Complete sequence: [prefix_tokens] + [action_tokens] + [suffix_tokens]
        
        Args:
            action_tokens: FAST action tokens (batch_size, max_action_len)
            token_lengths: Actual lengths of action tokens for each sample
            pad_token_id: Padding token ID for the VLM
            
        Returns:
            decoder_input_ids: Input for decoder (shifted right for teacher forcing)
            labels: Target token IDs for loss computation
            loss_mask: Boolean mask for valid positions
        """
        if self._vlm_tokenizer is None:
            raise ValueError("VLM tokenizer not set. Call set_vlm_tokenizer first.")
        
        device = action_tokens.device
        batch_size = action_tokens.shape[0]
        
        # Convert action tokens to VLM vocabulary
        action_llm_tokens = self.tokens_to_llm_tokens(action_tokens)
        
        # Get prefix and suffix lengths
        prefix_len = len(self.prefix_token_ids)
        suffix_len = len(self.suffix_token_ids)
        
        # Compute total sequence lengths for each sample
        # total_len = prefix_len + action_len + suffix_len
        total_lengths = prefix_len + token_lengths + suffix_len
        max_total_len = total_lengths.max().item()
        
        # Initialize tensors
        labels = torch.full(
            (batch_size, max_total_len), 
            pad_token_id, 
            dtype=torch.long, 
            device=device
        )
        loss_mask = torch.zeros(batch_size, max_total_len, dtype=torch.bool, device=device)
        
        # Build sequences for each sample
        prefix_tensor = torch.tensor(self.prefix_token_ids, dtype=torch.long, device=device)
        suffix_tensor = torch.tensor(self.suffix_token_ids, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            action_len = token_lengths[i].item()
            
            # Fill prefix
            labels[i, :prefix_len] = prefix_tensor
            
            # Fill action tokens
            labels[i, prefix_len:prefix_len + action_len] = action_llm_tokens[i, :action_len]
            
            # Fill suffix
            suffix_start = prefix_len + action_len
            labels[i, suffix_start:suffix_start + suffix_len] = suffix_tensor
            
            # Mark valid positions for loss
            valid_len = prefix_len + action_len + suffix_len
            loss_mask[i, :valid_len] = True
        
        # Create decoder input (shift right for teacher forcing)
        # Input: [decoder_start] + [prefix] + [action_tokens] + [suffix without last]
        decoder_input_ids = torch.full_like(labels, pad_token_id)
        decoder_input_ids[:, 0] = self._vlm_tokenizer.bos_token_id or self.eos_token_id
        decoder_input_ids[:, 1:] = labels[:, :-1]
        
        return decoder_input_ids, labels, loss_mask
    
    def extract_action_tokens_from_generated(
        self, 
        generated_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract action tokens from generated sequence by removing prefix/suffix.
        
        Looks for the prefix tokens, then extracts tokens until suffix/EOS is found.
        
        Args:
            generated_tokens: Generated token sequence (batch_size, seq_len)
            
        Returns:
            FAST action tokens (batch_size, max_action_len)
        """
        device = generated_tokens.device
        batch_size = generated_tokens.shape[0]
        
        # If VLM tokenizer is not set, we can't find prefix/suffix
        # In this case, assume all tokens (except first BOS) are action tokens
        if self._vlm_tokenizer is None or self.prefix_token_ids is None:
            # Convert all tokens to FAST tokens
            return self.llm_tokens_to_fast_tokens(generated_tokens)
        
        prefix_len = len(self.prefix_token_ids)
        suffix_first_token = self.suffix_token_ids[0]  # "|" token
        
        all_action_tokens = []
        max_action_len = 0
        
        for i in range(batch_size):
            seq = generated_tokens[i]
            
            # Find where action tokens start (after prefix)
            # Prefix might not be at the start if there's a BOS token
            start_idx = 0
            found_prefix = False
            for j in range(len(seq) - prefix_len + 1):
                if seq[j:j + prefix_len].tolist() == self.prefix_token_ids:
                    start_idx = j + prefix_len
                    found_prefix = True
                    break
            
            # If prefix not found, skip the first token (BOS) and use all remaining
            if not found_prefix:
                start_idx = 1 if len(seq) > 1 else 0
            
            # Find where action tokens end (at suffix or EOS)
            end_idx = len(seq)
            for j in range(start_idx, len(seq)):
                token_val = seq[j].item()
                if token_val == suffix_first_token or token_val == self.eos_token_id:
                    end_idx = j
                    break
            
            # Extract action tokens (in VLM vocabulary space)
            if end_idx > start_idx:
                action_llm_tokens = seq[start_idx:end_idx]
                # Convert VLM tokens to FAST tokens
                action_fast_tokens = self.llm_tokens_to_fast_tokens(action_llm_tokens.unsqueeze(0))[0]
            else:
                # No valid action tokens found, create empty tensor
                # Use empty tensor - will return zeros after decode anyway
                action_fast_tokens = torch.tensor([], dtype=torch.long, device=device)
            
            all_action_tokens.append(action_fast_tokens)
            max_action_len = max(max_action_len, len(action_fast_tokens))
        
        # Pad to same length - use -1 as padding marker (will be clipped in decode)
        if max_action_len == 0:
            # No valid tokens found at all, return special marker
            # FAST decode will handle this gracefully
            return torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
            
        padded_tokens = torch.full((batch_size, max_action_len), -1, dtype=torch.long, device=device)
        for i, tokens in enumerate(all_action_tokens):
            if len(tokens) > 0:
                padded_tokens[i, :len(tokens)] = tokens
        
        return padded_tokens

    @torch.no_grad()
    def decode(
        self, 
        tokens: torch.Tensor,
        action_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode FAST tokens back to continuous actions.
        
        Args:
            tokens: FAST tokens of shape (batch_size, seq_len)
            action_horizon: Target action sequence length (defaults to self.seq_len)
            action_dim: Target action dimension (defaults to self.num_dof)
            
        Returns:
            Continuous actions of shape (batch_size, action_horizon, action_dim)
        """
        if action_horizon is None:
            action_horizon = self.seq_len
        if action_dim is None:
            action_dim = self.num_dof
            
        device = tokens.device
        batch_size = tokens.shape[0]
        
        # Convert to numpy for FAST tokenizer
        if isinstance(tokens, torch.Tensor):
            tokens_np = tokens.cpu().numpy()
        else:
            tokens_np = tokens
        
        # Decode using FAST tokenizer
        actions_list = []
        for i in range(batch_size):
            # Get token list
            tokens_row = tokens_np[i].tolist()
            
            # Remove padding: we use -1 as padding marker (not 0, since 0 is valid)
            # Filter out negative values which indicate padding
            valid_tokens = [t for t in tokens_row if t >= 0]
            
            # Clamp remaining tokens to valid range [0, fast_vocab_size - 1]
            valid_tokens = [max(0, min(t, self.fast_vocab_size - 1)) for t in valid_tokens]
            
            # Ensure we have at least some tokens
            if len(valid_tokens) == 0:
                valid_tokens = [0]  # Fallback to a single zero token
            
            try:
                if self.suppress_decode_prints:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        actions = self._fast_tokenizer.decode(
                            [valid_tokens],
                            time_horizon=action_horizon,
                            action_dim=action_dim
                        )[0]
                else:
                    actions = self._fast_tokenizer.decode(
                        [valid_tokens],
                        time_horizon=action_horizon,
                        action_dim=action_dim
                    )[0]
                
                # Verify shape
                if actions.shape != (action_horizon, action_dim):
                    actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
                    
                actions_list.append(actions)
            except Exception as e:
                # Fallback to zeros if decoding fails
                # This can happen during early training when model outputs are random
                actions_list.append(np.zeros((action_horizon, action_dim), dtype=np.float32))
        
        actions = np.stack(actions_list, axis=0)
        return torch.from_numpy(actions).to(device).float()

    def reconstruct_from_llm_tokens(
        self, 
        llm_tokens: torch.Tensor, 
        times: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Reconstruct continuous actions from VLM tokens.
        
        Args:
            llm_tokens: VLM tokens of shape (batch_size, seq_len)
            times: Not used for FAST tokenizer (kept for API compatibility)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Continuous actions of shape (batch_size, seq_len, num_dof)
        """
        # Convert VLM tokens to FAST tokens
        fast_tokens = self.llm_tokens_to_fast_tokens(llm_tokens)
        
        # Decode FAST tokens to continuous actions
        return self.decode(
            fast_tokens, 
            action_horizon=kwargs.get('action_horizon', self.seq_len),
            action_dim=kwargs.get('action_dim', self.num_dof)
        )

    @torch.no_grad()
    def reconstruct_traj(
        self, 
        tokens: torch.Tensor, 
        times: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Reconstruct trajectory from FAST tokens.
        
        Args:
            tokens: FAST tokens of shape (batch_size, seq_len)
            times: Not used for FAST tokenizer
            **kwargs: Additional arguments
            
        Returns:
            Continuous actions of shape (batch_size, seq_len, num_dof)
        """
        return self.decode(tokens, **kwargs)

    def compute_reconstruction_error(self, raw_traj: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error for the tokenizer.
        
        Args:
            raw_traj: Original trajectory of shape (batch_size, seq_len, num_dof)
            
        Returns:
            Mean squared error between original and reconstructed trajectory
        """
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(0)
        
        tokens, _ = self.encode(raw_traj)
        reconstructed = self.reconstruct_traj(tokens)
        
        error = torch.mean((raw_traj - reconstructed) ** 2)
        return error

    def get_start_token_id(self) -> int:
        """
        Get the start token ID for autoregressive decoding.
        
        For FAST tokenizer with Florence-2, we use a special start token.
        This should be mapped to VLM vocabulary.
        
        Returns:
            Start token ID in VLM vocabulary
        """
        # Use a token that indicates "start of action sequence"
        # This could be customized based on the VLM tokenizer
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab size is not set.")
        
        # Use the decoder_start_token_id from Florence-2 (typically 2)
        return 2  # EOS token acts as decoder start in Florence-2
    
    def get_end_token_id(self) -> int:
        """
        Get the end token ID for autoregressive decoding.
        
        Returns:
            End token ID in VLM vocabulary
        """
        return 2  # EOS token in Florence-2
