"""
Precomputed SIREN Tokenizer for BEAST framework.
This tokenizer loads pre-trained SIREN weights from files instead of training SIREN networks on-the-fly.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import einops
from functools import wraps

from src.models.tokenizers.base_tokenizer import TokenizerBase
from src.models.tokenizers.utils import discrete_to_continuous, normalize_tensor
from siren_pytorch import SirenNet


def autocast_float32(fn):
    """Decorator to ensure float32 precision for SIREN operations."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled=False):
            return fn(*args, **kwargs)
    return wrapped


class PrecomputedSirenTokenizer(TokenizerBase):
    """
    Precomputed SIREN Tokenizer that loads pre-trained weights from files.
    This tokenizer skips the SIREN training process and directly uses pre-computed tokens.
    """

    def __init__(self, 
                 action_dim=7,
                 chunk_size=20,
                 siren_hidden_dim=64,
                 siren_num_layers=2,
                 siren_w0_initial=30.0,
                 siren_w0=30.0,
                 vocab_size=256,
                 precomputed_tokens_path=None,
                 device="cuda"):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.siren_hidden_dim = siren_hidden_dim
        self.siren_num_layers = siren_num_layers
        self.siren_w0_initial = siren_w0_initial
        self.siren_w0 = siren_w0
        self.vocab_size = vocab_size
        self.device = device
        self.precomputed_tokens_path = precomputed_tokens_path
        
        # Calculate total number of SIREN parameters
        self._calculate_siren_params_count()
        
        # Initialize bounds for weight normalization
        self.register_buffer("w_min", -0.1 * torch.ones(self.total_siren_params))
        self.register_buffer("w_max", 0.1 * torch.ones(self.total_siren_params))
        
        self.vlm_vocab_size = None
        
        # Create coordinate tensor for SIREN input
        self.coords = torch.linspace(0, 1, chunk_size).unsqueeze(-1).to(device)
        
        # Add compatibility attributes for BEAST
        self.num_dof = action_dim
        self.num_basis = 1
        self.seq_length = chunk_size
        self.duration = 1.0
        
        # Load precomputed tokens if path is provided
        self.precomputed_tokens = None
        if precomputed_tokens_path and os.path.exists(precomputed_tokens_path):
            self._load_precomputed_tokens()
        
        print(f"Precomputed SIREN Tokenizer initialized:")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - SIREN hidden dim: {siren_hidden_dim}")
        print(f"  - SIREN layers: {siren_num_layers}")
        print(f"  - Total SIREN params: {self.total_siren_params}")
        print(f"  - Vocab size: {vocab_size}")
        print(f"  - Precomputed tokens loaded: {self.precomputed_tokens is not None}")

    def _calculate_siren_params_count(self):
        """Calculate the total number of parameters in the SIREN network."""
        temp_model = SirenNet(
            dim_in=1,
            dim_hidden=self.siren_hidden_dim,
            dim_out=self.action_dim,
            num_layers=self.siren_num_layers,
            w0_initial=self.siren_w0_initial,
            w0=self.siren_w0,
            use_bias=True,
            final_activation=nn.Identity()
        )
        
        self.total_siren_params = sum(p.numel() for p in temp_model.parameters())
        del temp_model

    def _load_precomputed_tokens(self):
        """Load precomputed tokens from file."""
        print(f"Loading precomputed tokens from: {self.precomputed_tokens_path}")
        self.precomputed_tokens = np.load(self.precomputed_tokens_path)
        print(f"Loaded tokens shape: {self.precomputed_tokens.shape}")

    def _get_siren_model(self):
        """Create a SIREN model instance."""
        return SirenNet(
            dim_in=1,
            dim_hidden=self.siren_hidden_dim,
            dim_out=self.action_dim,
            num_layers=self.siren_num_layers,
            w0_initial=self.siren_w0_initial,
            w0=self.siren_w0,
            use_bias=True,
            final_activation=nn.Identity()
        )

    def _get_model_weights_vector(self, model):
        """Extract flattened weights from a SIREN model."""
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def _load_weights_to_model(self, model, weights_vector):
        """Load flattened weights into a SIREN model."""
        start_idx = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data = weights_vector[start_idx:start_idx + param_size].view(param.shape)
            start_idx += param_size

    def update_vlm_vocab_size(self, vlm_vocab_size):
        """Update VLM vocabulary size for token conversion."""
        self.vlm_vocab_size = vlm_vocab_size

    def update_weights_bounds_per_batch(self, weights):
        """Update weight bounds based on a batch of weights."""
        batch_min = weights.min(dim=0)[0]
        batch_max = weights.max(dim=0)[0]
        
        smaller_mask = batch_min < (self.w_min - 1e-4)
        larger_mask = batch_max > (self.w_max + 1e-4)
        
        if torch.any(smaller_mask):
            self.w_min[smaller_mask] = batch_min[smaller_mask]
        if torch.any(larger_mask):
            self.w_max[larger_mask] = batch_max[larger_mask]

    @torch.no_grad()
    @autocast_float32
    def encode(self, trajs, update_bounds=False):
        """
        Get precomputed tokens for action trajectories.
        
        Args:
            trajs (torch.Tensor): Action trajectories of shape (batch, time, action_dim)
            update_bounds (bool): Whether to update weight bounds (ignored for precomputed)
            
        Returns:
            tuple: (tokens, weights_dict) where tokens are precomputed
        """
        if self.precomputed_tokens is None:
            raise ValueError("Precomputed tokens not loaded. Please provide precomputed_tokens_path.")
        
        if len(trajs.shape) == 2:
            trajs = trajs.unsqueeze(0)
        
        batch_size = trajs.shape[0]
        
        # For now, we assume the batch indices correspond to precomputed token indices
        # In a real implementation, you might need to map trajectory indices to token indices
        if batch_size > len(self.precomputed_tokens):
            raise ValueError(f"Batch size {batch_size} exceeds number of precomputed tokens {len(self.precomputed_tokens)}")
        
        # Get precomputed tokens for this batch
        tokens = torch.from_numpy(self.precomputed_tokens[:batch_size]).to(self.device)
        
        # Convert tokens back to weights for compatibility
        weights = discrete_to_continuous(
            tokens, 
            min_val=self.w_min, 
            max_val=self.w_max, 
            num_bins=self.vocab_size
        )
        
        weights_dict = {'params': weights}
        
        return tokens, weights_dict

    def decode(self, tokens):
        """
        Decode discrete tokens back to continuous SIREN weights.
        
        Args:
            tokens (torch.Tensor): Discrete tokens
            
        Returns:
            torch.Tensor: Continuous SIREN weights
        """
        params = discrete_to_continuous(
            tokens, 
            min_val=self.w_min, 
            max_val=self.w_max, 
            num_bins=self.vocab_size
        )
        return params

    @torch.no_grad()
    @autocast_float32
    def reconstruct_traj(self, tokens, times=None, **kwargs):
        """
        Reconstruct action trajectories from SIREN weight tokens.
        
        Args:
            tokens (torch.Tensor): SIREN weight tokens
            times (torch.Tensor, optional): Time points for reconstruction
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Reconstructed action trajectories
        """
        # Decode tokens to weights
        weights = self.decode(tokens)
        
        batch_size = weights.shape[0]
        reconstructed_trajs = []
        
        for i in range(batch_size):
            # Create SIREN model and load weights
            model = self._get_siren_model().to(self.device)
            self._load_weights_to_model(model, weights[i])
            
            # Generate coordinates
            if times is None:
                coords = self.coords
            else:
                coords = times[i] if len(times.shape) == 3 else times
            
            # Forward pass to get reconstructed trajectory
            model.eval()
            with torch.no_grad():
                reconstructed_traj = model(coords)
            
            reconstructed_trajs.append(reconstructed_traj)
        
        return torch.stack(reconstructed_trajs, dim=0)

    def tokens_to_llm_tokens(self, tokens):
        """Convert SIREN tokens to LLM tokens."""
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab size is not set.")
        
        if len(tokens.shape) == 3:
            tokens = einops.rearrange(tokens, 'b t d -> b (t d)')
        
        llm_tokens = self.vlm_vocab_size - 1 - tokens
        return llm_tokens

    def llm_tokens_to_mp_tokens(self, llm_tokens):
        """Convert LLM tokens back to SIREN tokens."""
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab is not set.")
        
        tokens = self.vlm_vocab_size - 1 - llm_tokens
        if len(tokens.shape) == 2:
            tokens = einops.rearrange(tokens, 'b (t d) -> b t d', t=1, d=self.total_siren_params)
        return tokens

    def reconstruct_from_llm_tokens(self, llm_tokens, times=None, **kwargs):
        """Reconstruct trajectories from LLM tokens."""
        tokens = self.llm_tokens_to_mp_tokens(llm_tokens)
        return self.reconstruct_traj(tokens, times=times, **kwargs)

    @torch.no_grad()
    @autocast_float32
    def encode_continuous(self, trajs, update_bounds=False):
        """
        Encode trajectories to continuous SIREN weights (without discretization).
        
        Args:
            trajs (torch.Tensor): Action trajectories
            update_bounds (bool): Whether to update weight bounds
            
        Returns:
            tuple: (continuous_weights, weights_dict)
        """
        tokens, weights_dict = self.encode(trajs, update_bounds=update_bounds)
        continuous_weights = weights_dict['params']
        
        # Normalize weights
        normalized_weights = normalize_tensor(
            continuous_weights, 
            w_min=self.w_min, 
            w_max=self.w_max
        )
        
        return normalized_weights, weights_dict

    @torch.no_grad()
    def reconstruct_traj_continuous(self, params, times=None, **kwargs):
        """
        Reconstruct trajectories from continuous SIREN weights.
        
        Args:
            params (torch.Tensor): Continuous SIREN weights
            times (torch.Tensor, optional): Time points for reconstruction
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Reconstructed action trajectories
        """
        batch_size = params.shape[0]
        reconstructed_trajs = []
        
        for i in range(batch_size):
            # Create SIREN model and load weights
            model = self._get_siren_model().to(self.device)
            self._load_weights_to_model(model, params[i])
            
            # Generate coordinates
            if times is None:
                coords = self.coords
            else:
                coords = times[i] if len(times.shape) == 3 else times
            
            # Forward pass to get reconstructed trajectory
            model.eval()
            with torch.no_grad():
                reconstructed_traj = model(coords)
            
            reconstructed_trajs.append(reconstructed_traj)
        
        return torch.stack(reconstructed_trajs, dim=0)

    def compute_reconstruction_error(self, raw_traj, **kwargs):
        """
        Compute reconstruction error between original and reconstructed trajectories.
        
        Args:
            raw_traj (torch.Tensor): Original action trajectories
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Mean squared error
        """
        # Encode and reconstruct
        tokens, _ = self.encode(raw_traj)
        reconstructed = self.reconstruct_traj(tokens)
        
        # Compute MSE
        mse = torch.nn.functional.mse_loss(reconstructed, raw_traj)
        return mse

    @autocast_float32
    def visualize_reconstruction_error(self, raw_traj, save_path=None):
        """
        Visualize reconstruction error between original and reconstructed trajectories.
        
        Args:
            raw_traj (torch.Tensor): Original action trajectories
            save_path (str, optional): Path to save visualization
        """
        # Encode and reconstruct
        tokens, _ = self.encode(raw_traj)
        reconstructed = self.reconstruct_traj(tokens)
        
        # Convert to numpy for plotting
        raw_np = raw_traj.cpu().numpy()
        recon_np = reconstructed.cpu().numpy()
        
        # Create visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(min(7, len(axes))):
            axes[i].plot(raw_np[0, :, i], label='Original', alpha=0.7)
            axes[i].plot(recon_np[0, :, i], label='Reconstructed', alpha=0.7)
            axes[i].set_title(f'Dimension {i}')
            axes[i].legend()
            axes[i].grid(True)
        
        if len(axes) > 7:
            axes[7].plot(raw_np[0, :, :].flatten(), label='Original', alpha=0.7)
            axes[7].plot(recon_np[0, :, :].flatten(), label='Reconstructed', alpha=0.7)
            axes[7].set_title('All Dimensions')
            axes[7].legend()
            axes[7].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close() 