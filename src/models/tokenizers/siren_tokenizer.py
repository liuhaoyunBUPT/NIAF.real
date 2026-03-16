import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import einops
from functools import wraps

from siren_pytorch import SirenNet
from src.models.tokenizers.utils import continuous_to_discrete, discrete_to_continuous, normalize_tensor, denormalize_tensor
from src.models.tokenizers.base_tokenizer import TokenizerBase

from torch.amp import autocast


def autocast_float32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            return fn(*args, **kwargs)
    return wrapped


class SirenTokenizer(TokenizerBase):
    """
    SIREN Tokenizer for BEAST framework.
    Encodes action sequences to SIREN network weights and decodes weights back to actions.
    """

    def __init__(self, 
                 action_dim=7,
                 chunk_size=20,
                 siren_hidden_dim=16,
                 siren_num_layers=2,
                 siren_w0_initial=30.0,
                 siren_w0=30.0,
                 siren_learning_rate=5e-5,
                 siren_training_steps=5000,
                 vocab_size=256,
                 use_bpe=False,
                 device="cuda"):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.siren_hidden_dim = siren_hidden_dim
        self.siren_num_layers = siren_num_layers
        self.siren_w0_initial = siren_w0_initial
        self.siren_w0 = siren_w0
        self.siren_learning_rate = siren_learning_rate
        self.siren_training_steps = siren_training_steps
        self.vocab_size = vocab_size
        self.use_bpe = use_bpe
        self.device = device
        
        # Calculate total number of SIREN parameters
        self._calculate_siren_params_count()
        
        # Initialize bounds for weight normalization
        self.register_buffer("w_min", -0.1 * torch.ones(self.total_siren_params, device=self.device))
        self.register_buffer("w_max", 0.1 * torch.ones(self.total_siren_params, device=self.device))
        
        self.vlm_vocab_size = None
        
        # Create coordinate tensor for SIREN input
        self.coords = torch.linspace(0, 1, chunk_size).unsqueeze(-1).to(device)
        
        # Add compatibility attributes for BEAST
        self.num_dof = action_dim  # For compatibility with BEAST
        self.num_basis = 1  # SIREN doesn't use basis functions like B-spline, but BEAST expects this
        self.seq_length = chunk_size  # For compatibility
        self.duration = 1.0  # For compatibility
        
        print(f"SIREN Tokenizer initialized:")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - SIREN hidden dim: {siren_hidden_dim}")
        print(f"  - SIREN layers: {siren_num_layers}")
        print(f"  - Total SIREN params: {self.total_siren_params}")
        print(f"  - Vocab size: {vocab_size}")

    def _calculate_siren_params_count(self):
        """Calculate the total number of parameters in the SIREN network."""
        # Create a temporary SIREN model to count parameters
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

    def _train_siren_on_chunk(self, action_chunk):
        """Train SIREN network on a single action chunk and return weights."""
        model = self._get_siren_model().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.siren_learning_rate)
        criterion = nn.MSELoss()
        
        # Prepare data
        coords = einops.repeat(self.coords, 't one -> b t one', b=action_chunk.shape[0])
        ground_truth = action_chunk.float().to(self.device)
        
        # Training loop
        for step in range(self.siren_training_steps):
            model.train()
            optimizer.zero_grad()
            
            predicted_actions = model(coords)
            loss = criterion(predicted_actions, ground_truth)
            
            loss.backward()
            optimizer.step()
        
        # Extract weights
        weights = self._get_model_weights_vector(model)
        return weights, model

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
        Encode action trajectories to SIREN weight tokens.
        
        Args:
            trajs (torch.Tensor): Action trajectories of shape (batch, time, action_dim)
            update_bounds (bool): Whether to update weight bounds
            
        Returns:
            tuple: (tokens, weights_dict) where tokens are discretized weights
        """
        if len(trajs.shape) == 2:
            trajs = trajs.unsqueeze(0)
        
        batch_size = trajs.shape[0]
        all_weights = []
        
        # Process each trajectory in the batch
        for i in range(batch_size):
            # Extract chunk from trajectory (assuming trajectory is longer than chunk_size)
            if trajs.shape[1] >= self.chunk_size:
                chunk = trajs[i, :self.chunk_size]
            else:
                # Pad if trajectory is shorter than chunk_size
                chunk = torch.zeros(self.chunk_size, self.action_dim, device=self.device)
                chunk[:trajs.shape[1]] = trajs[i]
            
            chunk = chunk.unsqueeze(0)  # Add batch dimension
            
            # Train SIREN and get weights
            with torch.enable_grad():
                weights, _ = self._train_siren_on_chunk(chunk)
            all_weights.append(weights)
        
        # Stack weights from all batches
        weights = torch.stack(all_weights, dim=0)
        
        if update_bounds:
            self.update_weights_bounds_per_batch(weights)
        
        # Clamp weights to bounds
        unclamped_weights = weights
        weights = torch.clamp(unclamped_weights, min=self.w_min, max=self.w_max)
        
        # Convert to discrete tokens
        tokens = continuous_to_discrete(
            weights, 
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
        Reconstruct trajectories from continuous normalized weights.
        
        Args:
            params (torch.Tensor): Normalized continuous weights
            times (torch.Tensor, optional): Time points
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Reconstructed trajectories
        """
        # Denormalize weights
        denormalized_params = denormalize_tensor(
            params, 
            w_min=self.w_min, 
            w_max=self.w_max
        )
        
        # Use the same reconstruction logic as discrete tokens
        return self.reconstruct_traj(denormalized_params, times=times, **kwargs)

    def compute_reconstruction_error(self, raw_traj, **kwargs):
        """Compute reconstruction error between original and reconstructed trajectories."""
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(0)
        
        tokens, _ = self.encode(raw_traj)
        reconstructed_trajs = self.reconstruct_traj(tokens)
        
        # Ensure same length
        min_len = min(raw_traj.shape[1], reconstructed_trajs.shape[1])
        error = torch.mean((raw_traj.to(reconstructed_trajs.device)[:, :min_len] - reconstructed_trajs[:, :min_len]) ** 2)
        
        return error

    @autocast_float32
    def visualize_reconstruction_error(self, raw_traj, save_path=None):
        """Visualize reconstruction error between original and reconstructed trajectories."""
        raw_traj = raw_traj.to(torch.float32)
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(0)
        
        tokens, _ = self.encode(raw_traj, update_bounds=True)
        reconstructed_trajs = self.reconstruct_traj(tokens)
        
        # Convert to numpy for plotting
        pos = reconstructed_trajs.detach().cpu().numpy()
        raw_traj_np = raw_traj.detach().cpu().numpy()
        
        # Ensure same length
        min_len = min(pos.shape[1], raw_traj_np.shape[1])
        pos = pos[:, :min_len]
        raw_traj_np = raw_traj_np[:, :min_len]
        
        x_vals = np.linspace(0, 1, min_len)
        batch_size, time_steps, dof = raw_traj_np.shape
        
        # Plot for each sample in batch
        for sample_idx in range(batch_size):
            fig, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)
            
            for i in range(dof):
                axes[i].plot(x_vals, pos[sample_idx, :, i], marker='o', 
                           label='SIREN Reconstruct', linestyle='-', color='b')
                axes[i].plot(x_vals, raw_traj_np[sample_idx, :, i], marker='*', 
                           label='Ground Truth', linestyle='--', color='r')
                axes[i].set_ylabel(f"DOF {i + 1}")
                axes[i].grid(True)
                axes[i].legend(loc="best")
            
            axes[-1].set_xlabel("Normalized Time")
            plt.suptitle(f"SIREN Tokenizer - Sample {sample_idx} in Batch")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                save_filename = os.path.join(save_path, f"siren_sample_{sample_idx}.png")
                plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            
            plt.show()
            plt.close(fig) 