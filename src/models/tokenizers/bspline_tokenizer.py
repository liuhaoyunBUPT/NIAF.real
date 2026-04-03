import os
import torch

from addict import Dict

from mp_pytorch.mp import MPFactory
from mp_pytorch.demo import get_mp_utils

from src.models.tokenizers.utils import continuous_to_discrete, discrete_to_continuous, normalize_tensor, denormalize_tensor
from src.models.tokenizers.base_tokenizer import TokenizerBase
import mp_pytorch.util as mp_utils

import numpy as np
import matplotlib.pyplot as plt
import einops

from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from torch.amp import autocast

from functools import wraps


def autocast_float32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            return fn(*args, **kwargs)

    return wrapped


class BSpline_Tokenizer(TokenizerBase):

    def __init__(self, num_dof=1, num_basis=10, duration=2 * torch.pi, seq_len=50, vocab_size=256,
                 degree_p=4, gripper_zero_order=True, gripper_dof=1, gripper_indices=[6],
                 init_cond_order=0, end_cond_order=0, init_pos = True,
                 use_bpe=False, device="cuda"):
        super().__init__()

        self.dt = 0.01  # 100 Hz, fixed for now
        if not gripper_zero_order:
            gripper_dof = 0
        self.joint_dof = num_dof - gripper_dof
        self.gripper_dof = gripper_dof
        self.bspline_config = Dict()
        self.bspline_config.mp_type = "uni_bspline"
        self.bspline_config.device = device
        self.bspline_config.num_dof = self.joint_dof
        self.bspline_config.tau = duration
        self.bspline_config.mp_args.num_basis = num_basis
        self.bspline_config.mp_args.degree_p = degree_p
        self.bspline_config.mp_args.init_condition_order = init_cond_order
        self.bspline_config.mp_args.end_condition_order = end_cond_order
        self.bspline_config.mp_args.dt = 0.01
        # self.bspline_config.mp_args.weights_scale = 0.01
        self.init_pos = init_pos

        self.mp = MPFactory.init_mp(**self.bspline_config)

        self.gripper_mp = None
        self.gripper_indices = gripper_indices

        ### TODO: Now we just assume that the gripper always at the end of the DoF

        if gripper_zero_order:
            self.gripper_mp_config = Dict()
            self.gripper_mp_config.mp_type = "uni_bspline"
            self.gripper_mp_config.device = device
            self.gripper_mp_config.num_dof = gripper_dof
            self.gripper_mp_config.tau = duration
            self.gripper_mp_config.mp_args.num_basis = num_basis
            self.gripper_mp_config.mp_args.degree_p = 0
            self.gripper_mp = MPFactory.init_mp(**self.gripper_mp_config)
            print(f"Gripper MP initialized with {num_basis} basis functions")

        self.device = device
        self.num_dof = self.joint_dof + self.gripper_dof
        self.num_basis = num_basis

        self.vocab_size = vocab_size

        self.duration = duration
        self.seq_length = seq_len

        self.use_bpe = use_bpe

        self.times = mp_utils.tensor_linspace(0, duration, seq_len).to(device)

        self.register_buffer("w_min", - 0.02 * torch.ones((num_dof * num_basis)))
        self.register_buffer("w_max", 0.02 * torch.ones((num_dof * num_basis)))
        self.vlm_vocab_size = None

    def fit_bpe(self, demos):

        demos = demos.reshape(-1, self.num_dof * self.num_basis)
        max_token = demos.max()
        min_token = demos.min()

        bpe = ByteLevelBPETokenizer()
        alpha_bet = [chr(i) for i in range(max_token - min_token + 1)]
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=[],
            initial_alphabet=alpha_bet,
            max_token_length=10000,
        )

        def _token_iter():
            for tokens in demos:
                string = "".join(map(chr, tokens))
                yield string

        bpe._tokenizer.train_from_iterator(_token_iter(), trainer=trainer)

        self.bpe = PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False)

        print("BPE training complete")

    def update_vlm_vocab_size(self, vlm_vocab_size):
        self.vlm_vocab_size = vlm_vocab_size

    @torch.no_grad()
    @autocast_float32
    def compute_weights(self, demos):
        times = einops.repeat(self.times, 't -> b t', b=demos.shape[0])
        weights = self.mp.learn_mp_params_from_trajs(times, demos)['params']
        return weights

    def update_weights_bounds(self, demos):
        times = einops.repeat(self.times, 't -> b t', b=demos.shape[0])
        weights = self.mp.learn_mp_params_from_trajs(times, demos)['params']
        self.w_min = weights.min(dim=0)[0]
        self.w_max = weights.max(dim=0)[0]

    def update_weights_bounds_per_batch(self, weights):
        weights = weights.reshape(-1, self.num_dof * self.num_basis)
        batch_min = weights.min(dim=0)[0]
        batch_max = weights.max(dim=0)[0]
        smaller_mask = batch_min < (self.w_min - 1e-4)
        larger_mask = batch_max > (self.w_max + 1e-4)
        if torch.any(smaller_mask):
            self.w_min[smaller_mask] = batch_min[smaller_mask]
        if torch.any(larger_mask):
            self.w_max[larger_mask] = batch_max[larger_mask]

    def update_times(self, times):
        self.times = times

    @torch.no_grad()
    @autocast_float32
    def encode(self, trajs, update_bounds=False):

        trajs = trajs.to(torch.float32)
        times = einops.repeat(self.times, 't -> b t', b=trajs.shape[0])
        params_dict = self.mp.learn_mp_params_from_trajs(times, trajs[..., :self.joint_dof])
        if self.gripper_mp is not None:
            gripper_params_dict = self.gripper_mp.learn_mp_params_from_trajs(times, trajs[..., -self.gripper_dof:])
            params_dict['params'] = torch.cat([params_dict['params'], gripper_params_dict['params']], dim=-1)
        if update_bounds:
            self.update_weights_bounds_per_batch(params_dict['params'])
        unclampled_params = params_dict['params']
        params = torch.clamp(unclampled_params, min=self.w_min, max=self.w_max)
        tokens = continuous_to_discrete(params, min_val=self.w_min, max_val=self.w_max, num_bins=self.vocab_size)
        # tokens = einops.rearrange(tokens, 'b (d t) -> b t d', t=self.num_basis, d=self.num_dof)
        tokens = einops.rearrange(tokens, 'b (d t) -> b (t d)', t=self.num_basis, d=self.num_dof)
        return tokens, params_dict

    def tokens_to_llm_tokens(self, tokens):
        if len(tokens.shape) == 3:
            tokens = einops.rearrange(tokens, 'b t d -> b (t d)')
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab size is not set.")
        llm_tokens = self.vlm_vocab_size - 1 - tokens
        return llm_tokens

    def llm_tokens_to_mp_tokens(self, llm_tokens):
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab is not set.")
        tokens = self.vlm_vocab_size - 1 - llm_tokens
        if len(tokens.shape) == 2:
            tokens = einops.rearrange(tokens, 'b (t d) -> b t d', t=self.num_basis, d=self.num_dof)
        return tokens

    def reconstruct_from_llm_tokens(self, llm_tokens, times=None, **kwargs):
        tokens = self.llm_tokens_to_mp_tokens(llm_tokens)
        return self.reconstruct_traj(tokens, times=times, **kwargs)

    def reconstruct_vel_from_llm_tokens(self, llm_tokens, times=None, execution_hz=30.0, **kwargs):
        tokens = self.llm_tokens_to_mp_tokens(llm_tokens)
        return self.reconstruct_traj_vel(
            tokens,
            times=times,
            execution_hz=execution_hz,
            **kwargs,
        )

    @torch.no_grad()
    @autocast_float32
    def encode_continuous(self, trajs, update_bounds=False):
        times = einops.repeat(self.times, 't -> b t', b=trajs.shape[0])
        params_dict = self.mp.learn_mp_params_from_trajs(times, trajs[..., :self.joint_dof])
        if self.gripper_mp is not None:
            gripper_params_dict = self.gripper_mp.learn_mp_params_from_trajs(times, trajs[..., -self.gripper_dof:])
            params_dict['params'] = torch.cat([params_dict['params'], gripper_params_dict['params']], dim=-1)
        if update_bounds:
            self.update_weights_bounds_per_batch(params_dict['params'])
        tokens = params_dict['params']
        tokens = normalize_tensor(tokens, w_min=self.w_min, w_max=self.w_max)
        # tokens = einops.rearrange(tokens, 'b (d t) -> b t d', t=self.num_basis, d=self.num_dof)
        return tokens, params_dict

    def decode(self, tokens):
        params = discrete_to_continuous(tokens, min_val=self.w_min, max_val=self.w_max, num_bins=self.vocab_size)
        return params

    @torch.no_grad()
    @autocast_float32
    def reconstruct_traj(self, tokens, times=None, **kwargs):
        # params = self.decode(tokens.reshape(-1, self.num_dof * self.num_basis))

        # 1. 调整词元形状：如果输入的词元形状是 (batch, time, dof)，则将其重排为 (batch, dof * time)。
        # 这是为了匹配后续处理所期望的扁平化格式。
        if len(tokens.shape) == 3:
            tokens = einops.rearrange(tokens, "b t d -> b (d t)")

        # 2. 词元到连续值：调用 decode 方法，将离散的整数词元（如0-255）反量化为连续的B样条权重（params）。
        # 这个过程是 continuous_to_discrete 的逆操作。
        params = self.decode(tokens)
        
        # 3. 设置时间点：如果未提供用于生成轨迹的时间点向量，则使用默认的预设时间点。
        if times is None:
            times = einops.repeat(self.times, 't -> b t', b=params.shape[0])
        
        # 4. (可选) 设置初始位置：检查是否需要将轨迹的起点强制设置为当前的机器人位置。
        # 这是为了确保生成的动作从当前状态平滑地开始。
        if self.init_pos and kwargs.get("init_p") is not None:
            # 将扁平化的权重重塑为 (batch, num_basis, num_dof) 以便修改。
            _params = einops.rearrange(params, "b (d t) -> b t d", t=self.num_basis, d=self.num_dof)
            # 将第一个B样条控制点（通常决定了轨迹的起始位置）替换为传入的当前机器人位置。
            _params[:, 0, :self.joint_dof] = kwargs["init_p"][:, :self.joint_dof]
            # 将修改后的权重重新扁平化。
            params = einops.rearrange(_params, "b t d -> b (d t)")
        
        # 5. 生成关节轨迹：使用B样条运动基元（self.mp）和解码出的权重（params）来计算关节部分的轨迹。
        self.mp.update_inputs(times=times, params=params[..., :self.joint_dof * self.num_basis])
        pos = self.mp.get_traj_pos()
        
        # 6. (可选) 生成夹爪轨迹：如果配置了单独的夹爪运动基元。
        if self.gripper_mp is not None:
            # 提取属于夹爪的权重。
            gripper_params = params[..., -self.gripper_dof * self.num_basis:]
            # 使用夹爪的运动基元生成其轨迹。
            self.gripper_mp.update_inputs(times=times, params=gripper_params)
            gripper_pos = self.gripper_mp.get_traj_pos()
            # 将关节轨迹和夹爪轨迹拼接在一起。
            pos = torch.cat([pos, gripper_pos], dim=-1)

        # 7. 返回最终的、完整的、连续的动作轨迹。
        return pos

    @torch.no_grad()
    @autocast_float32
    def reconstruct_traj_vel(self, tokens, times=None, execution_hz=30.0, **kwargs):
        if execution_hz <= 0:
            raise ValueError(f"execution_hz must be > 0, got {execution_hz}")

        if len(tokens.shape) == 3:
            tokens = einops.rearrange(tokens, "b t d -> b (d t)")

        params = self.decode(tokens)

        if times is None:
            times = einops.repeat(self.times, 't -> b t', b=params.shape[0])

        if self.init_pos and kwargs.get("init_p") is not None:
            _params = einops.rearrange(params, "b (d t) -> b t d", t=self.num_basis, d=self.num_dof)
            _params[:, 0, :self.joint_dof] = kwargs["init_p"][:, :self.joint_dof]
            params = einops.rearrange(_params, "b t d -> b (d t)")

        if not hasattr(self.mp, "get_traj_vel"):
            raise RuntimeError("B-spline derivative API get_traj_vel is unavailable in mp backend")

        self.mp.update_inputs(times=times, params=params[..., :self.joint_dof * self.num_basis])
        vel = self.mp.get_traj_vel()
        if vel is None:
            raise RuntimeError("B-spline derivative API get_traj_vel returned None")

        if times.shape[-1] <= 1:
            raise ValueError("At least 2 time points are required to scale velocity")
        dt_tokenizer = torch.mean(times[..., 1:] - times[..., :-1])
        dt_exec = 1.0 / float(execution_hz)
        vel = vel * (dt_tokenizer / dt_exec)

        if self.gripper_mp is not None:
            gripper_params = params[..., -self.gripper_dof * self.num_basis:]
            self.gripper_mp.update_inputs(times=times, params=gripper_params)

            gripper_degree = getattr(getattr(self.gripper_mp, "basis_gn", None), "degree_p", None)
            if gripper_degree == 0:
                # Zero-order B-spline is piecewise constant, so velocity is zero in control intervals.
                gripper_vel = torch.zeros(
                    vel.shape[0],
                    vel.shape[1],
                    self.gripper_dof,
                    dtype=vel.dtype,
                    device=vel.device,
                )
            else:
                if not hasattr(self.gripper_mp, "get_traj_vel"):
                    raise RuntimeError("B-spline derivative API get_traj_vel is unavailable in gripper mp backend")
                gripper_vel = self.gripper_mp.get_traj_vel()
                if gripper_vel is None:
                    raise RuntimeError("B-spline derivative API get_traj_vel returned None for gripper mp")
                gripper_vel = gripper_vel * (dt_tokenizer / dt_exec)

            vel = torch.cat([vel, gripper_vel], dim=-1)

        return vel

    @torch.no_grad()
    def reconstruct_traj_continuous(self, params, times=None, **kwargs):
        # params = einops.rearrange(params, "b t d -> b (d t)")
        params = denormalize_tensor(params, w_min=self.w_min, w_max=self.w_max)
        if times is None:
            times = einops.repeat(self.times, 't -> b t', b=params.shape[0])
        if self.init_pos and kwargs.get("init_p") is not None:
            _params = einops.rearrange(params, "b (d t) -> b t d", t=self.num_basis, d=self.num_dof)
            _params[:, 0, :self.joint_dof] = kwargs["init_p"][:, :self.joint_dof]
            params = einops.rearrange(_params, "b t d -> b (d t)")
        self.mp.update_inputs(times=times, params=params[..., :self.joint_dof * self.num_basis])
        pos = self.mp.get_traj_pos()
        if self.gripper_mp is not None:
            gripper_params = params[..., -self.gripper_dof * self.num_basis:]
            self.gripper_mp.update_inputs(times=times, params=gripper_params)
            gripper_pos = self.gripper_mp.get_traj_pos()
            pos = torch.cat([pos, gripper_pos], dim=-1)
        return pos

    def compute_reconstruction_error(self, raw_traj):
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(-1)
        tokens, _ = self.encode(raw_traj)
        reconstruct_trajs = self.reconstruct_traj(tokens)
        error = torch.mean((raw_traj - reconstruct_trajs) ** 2)
        return error

    @autocast_float32
    def visualize_reconstruction_error(self, raw_traj):
            raw_traj = raw_traj.to(torch.float32)
            if len(raw_traj.shape) == 2:
                raw_traj = raw_traj.unsqueeze(0)
            tokens, params_dict = self.encode(raw_traj, update_bounds=True)
            pos = self.reconstruct_traj(tokens)
            pos = pos.detach().cpu().numpy()
            raw_traj = raw_traj.detach().cpu().numpy()
            x_vals = np.linspace(0, self.duration, raw_traj.shape[1])

            batch_size, time_steps, dof = raw_traj.shape
            # Plot both generated and ground truth sine waves
            for sample_idx in range(batch_size):
                fig, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)

                for i in range(dof):
                    axes[i].plot(x_vals, pos[sample_idx, :, i], marker='o', label='reconstruct', linestyle='-',
                                 color='b')
                    axes[i].plot(x_vals, raw_traj[sample_idx, :, i], marker='*', label='ground_truth', linestyle='--',
                                 color='r')
                    axes[i].set_ylabel(f"DOF {i + 1}")
                    axes[i].grid(True)
                    axes[i].legend(loc="best")

                axes[-1].set_xlabel("Timesteps")
                plt.suptitle(f"Visualization of Sample {sample_idx} in Batch")
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

    @autocast_float32
    def visualize_reconstruction_error_with_llm_tokenizer(self, raw_traj,
                                                          save_path=None):
            raw_traj = raw_traj.to(torch.float32)
            if len(raw_traj.shape) == 2:
                raw_traj = raw_traj.unsqueeze(0)
            tokens, params_dict = self.encode(raw_traj, update_bounds=True)
            llm_tokens = self.tokens_to_llm_tokens(tokens)
            # reconstruct the trajectory from the llm tokens
            pos = self.reconstruct_from_llm_tokens(llm_tokens)
            pos = pos.detach().cpu().numpy()
            raw_traj = raw_traj.detach().cpu().numpy()
            x_vals = np.linspace(0, self.duration, raw_traj.shape[1])

            batch_size, time_steps, dof = raw_traj.shape
            # Plot both generated and ground truth sine waves
            for sample_idx in range(batch_size):
                fig, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)

                for i in range(dof):
                    axes[i].plot(x_vals, pos[sample_idx, :, i], marker='o', label='reconstruct', linestyle='-',
                                 color='b')
                    axes[i].plot(x_vals, raw_traj[sample_idx, :, i], marker='*', label='ground_truth', linestyle='--',
                                 color='r')
                    axes[i].set_ylabel(f"DOF {i + 1}")
                    axes[i].grid(True)
                    axes[i].legend(loc="best")

                axes[-1].set_xlabel("Timesteps")
                plt.suptitle(f"Visualization of Sample {sample_idx} in Batch")
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                # Save the figure with the specified naming format
                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    save_filename = os.path.join(save_path, f"sample_{sample_idx}.png")
                    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
                # Close the figure to free memory (important when processing many plots)
                plt.show()
                plt.close(fig)
    
    @autocast_float32
    def visualize_reconstruction_error_with_cont_tokenizer(self, raw_traj,
                                                          save_path=None):
            raw_traj = raw_traj.to(torch.float32)
            if len(raw_traj.shape) == 2:
                raw_traj = raw_traj.unsqueeze(0)
            continous_tokens, _ = self.encode_continuous(raw_traj, update_bounds=True)
            # reconstruct the trajectory from the llm tokens
            pos = self.reconstruct_traj_continuous(continous_tokens)
            pos = pos.detach().cpu().numpy()
            raw_traj = raw_traj.detach().cpu().numpy()
            x_vals = np.linspace(0, self.duration, raw_traj.shape[1])

            batch_size, time_steps, dof = raw_traj.shape
            # Plot both generated and ground truth sine waves
            for sample_idx in range(batch_size):
                fig, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)

                for i in range(dof):
                    axes[i].plot(x_vals, pos[sample_idx, :, i], marker='o', label='reconstruct', linestyle='-',
                                 color='b')
                    axes[i].plot(x_vals, raw_traj[sample_idx, :, i], marker='*', label='ground_truth', linestyle='--',
                                 color='r')
                    axes[i].set_ylabel(f"DOF {i + 1}")
                    axes[i].grid(True)
                    axes[i].legend(loc="best")

                axes[-1].set_xlabel("Timesteps")
                plt.suptitle(f"Visualization of Sample {sample_idx} in Batch")
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                # Save the figure with the specified naming format
                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    save_filename = os.path.join(save_path, f"sample_{sample_idx}.png")
                    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
                # Close the figure to free memory (important when processing many plots)
                plt.show()
                # plt.close(fig)

