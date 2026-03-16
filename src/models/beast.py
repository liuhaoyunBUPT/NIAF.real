"""
BEAST模型 - MP tokenizer动作编码/解码
"""
import logging
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from src.models.base import BASE, create_bidirectional_mask

logger = logging.getLogger(__name__)


class BEAST(BASE):
    """
    BEAST模型实现

    继承BASE基类,使用MP tokenizer进行动作序列的编码和解码
    支持双臂机器人的多相机配置和单臂/双臂模式切换
    """

    def __init__(
        self,
        *args,
        # 动作归一化配置 (由 action_stats 注入)
        action_min: List[float] = None,
        action_max: List[float] = None,
        # MP tokenizer
        mp_tokenizer: DictConfig = None,
        update_w_bound: bool = False,
        pre_compute_w_bound: bool = False,
        pre_compute_w_bound_steps: int = 50000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # 动作归一化 buffer (推理时反归一化)
        if action_min is None or action_max is None:
            raise ValueError("需要配置 action_min 和 action_max")
        self.register_buffer('action_min', torch.tensor(action_min, dtype=torch.float32))
        self.register_buffer('action_max', torch.tensor(action_max, dtype=torch.float32))

        # MP Tokenizer
        if mp_tokenizer is None:
            raise ValueError("BEAST 模型需要配置 mp_tokenizer")
        self.action_tokenizer = hydra.utils.instantiate(mp_tokenizer)
        self.num_dof = self.action_tokenizer.num_dof
        self.num_basis = self.action_tokenizer.num_basis
        self.action_tokenizer.update_vlm_vocab_size(self.vlm_vocab_size)

        self.update_w_bound = update_w_bound
        self.precompute_w_bound = pre_compute_w_bound
        self.precompute_w_bound_steps = pre_compute_w_bound_steps

        logger.info(f"Initialized BEAST with rgb_obs_keys={self.rgb_obs_keys}")
        logger.info(f"Action tokenizer: num_dof={self.num_dof}, num_basis={self.num_basis}")

    # -----------------------------------------------------------------
    #                     训练 / 验证逻辑
    # -----------------------------------------------------------------
    def compute_llm_outputs(self, batch: Dict) -> torch.Tensor:
        """使用 MP tokenizer 对动作进行编码, 通过 VLM decoder 预测动作 token"""
        features, encoder_attn_mask = self.compute_input_features(batch)

        if "actions" not in batch:
            raise ValueError("batch 缺少 'actions' 键")

        # MP tokenizer encode
        action_tokens, params = self.action_tokenizer.encode(
            batch["actions"], update_bounds=self.update_w_bound
        )

        llm_label_ids = self.action_tokenizer.tokens_to_llm_tokens(action_tokens)

        # 中性输入序列
        input_tokens = self.action_tokenizer.vocab_size // 2 * torch.ones_like(
            llm_label_ids, dtype=torch.long, device=self.device
        )
        llm_input_ids = self.action_tokenizer.tokens_to_llm_tokens(input_tokens)

        bidirectional_mask = create_bidirectional_mask(
            batch_size=llm_label_ids.shape[0],
            seq_length=llm_label_ids.shape[1],
            device=self.device,
        )

        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids,
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=bidirectional_mask,
            use_cache=True,
        )

        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)

        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            lm_logits.view(-1, self.vlm.config.vocab_size),
            llm_label_ids.view(-1),
        )

        # 重建精度指标
        pred_tokens = torch.argmax(lm_logits, dim=-1)
        token_predict_accuracy = self.token_prediction_accuracy(pred_tokens, llm_label_ids)
        reconstruct_traj = self.action_tokenizer.reconstruct_from_llm_tokens(pred_tokens, times=None)
        action_mse = F.mse_loss(reconstruct_traj, batch["actions"])

        return {
            'llm_loss': masked_lm_loss,
            'token_predict_accuarcy': token_predict_accuracy,
            'reconstruct_action_mse': action_mse,
        }

    def llm_generates(self, batch: Dict) -> torch.Tensor:
        """推理时生成动作 token"""
        features, encoder_attn_mask = self.compute_input_features(batch)

        input_tokens = self.action_tokenizer.vocab_size // 2 * torch.ones(
            (1, self.num_dof, self.num_basis), dtype=torch.long, device=self.device
        )
        llm_input_ids = self.action_tokenizer.tokens_to_llm_tokens(input_tokens)

        bidirectional_mask = create_bidirectional_mask(
            batch_size=llm_input_ids.shape[0],
            seq_length=llm_input_ids.shape[1],
            device=self.device,
        )

        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids,
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=bidirectional_mask,
            use_cache=True,
        )

        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)

        return torch.argmax(lm_logits, dim=-1)

    # -----------------------------------------------------------------
    #                        推理
    # -----------------------------------------------------------------
    @torch.no_grad()
    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        推理：生成并反归一化动作序列

        Args:
            obs: 观测字典
            goal: 目标字典 (含 lang_text)

        Returns:
            反归一化后的动作序列
        """
        # 构建 batch
        rgb_obs_batch = {}
        for k in self.rgb_obs_keys:
            if k in obs["rgb_obs"]:
                rgb_obs_batch[k] = obs["rgb_obs"][k]
        batch = {
            "rgb_obs": rgb_obs_batch,
            "lang_text": [goal["lang_text"]],
        }

        llm_action_tokens = self.llm_generates(batch)
        actions_normalized = self.action_tokenizer.reconstruct_from_llm_tokens(
            llm_action_tokens, times=None
        )

        return self.denormalize_actions(actions_normalized)

    # -----------------------------------------------------------------
    #                 MP tokenizer 归一化器预计算
    # -----------------------------------------------------------------
    def on_fit_start(self):
        if self.precompute_w_bound:
            self.precompute_mp_normalizer()
            if self.trainer.world_size > 1 and dist.is_initialized():
                dist.broadcast(self.action_tokenizer.w_min, src=0)
                dist.broadcast(self.action_tokenizer.w_max, src=0)
                logger.info(f"mp weight normalizer set in rank {self.global_rank}")
        else:
            logger.info("mp normalizer not precomputed, using default [-1, 1]")

    @rank_zero_only
    def precompute_mp_normalizer(self):
        logger.info("precompute mp normalizer")

        dataloader = self.trainer.datamodule.train_dataloader()
        params = []
        for batch in tqdm(
            dataloader["lang"],
            desc=f"Rank_{self.global_rank}, precomputing weight normalizer of MP",
            unit="batch",
        ):
            act_chunks = batch["actions"][..., :self.action_tokenizer.joint_dof]
            act_chunks = act_chunks.to(self.device)
            param = self.action_tokenizer.compute_weights(act_chunks)
            params.append(param.cpu().numpy())
            if len(params) > self.precompute_w_bound_steps:
                logger.info(f"Rank_{self.global_rank}, precomputed enough samples")
                break

        params = np.concatenate(params, axis=0)
        params_min = np.quantile(params, 0.01, 0)
        params_max = np.quantile(params, 0.99, 0)

        params_min = torch.from_numpy(params_min).to(self.action_tokenizer.w_min.device)
        params_max = torch.from_numpy(params_max).to(self.action_tokenizer.w_max.device)

        self.action_tokenizer.w_min[:self.action_tokenizer.joint_dof * self.num_basis] = params_min
        self.action_tokenizer.w_max[:self.action_tokenizer.joint_dof * self.num_basis] = params_max

        logger.info("mp_normalizer computed and set on rank 0")

    # -----------------------------------------------------------------
    #                        反归一化
    # -----------------------------------------------------------------
    def denormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """
        将归一化的动作 [-1, 1] 反归一化到原始范围 [action_min, action_max]。

        公式: action = (normalized + 1) / 2 * (max - min) + min

        支持维度自动适配：当模型输出维度与统计量维度不匹配时，
        根据 arm_mode 切片 action_min/action_max。
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
