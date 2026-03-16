'''
TODO:
0. 并看一下训练好的base parameter中，调制参数的大小，例如偏置调制强度是否有必要设置为0.02
1. 使用large模型
2. GT使用绝对动作
3. 可视化base parameter对应的动作形状
4. Finer
5. 可视化bad case
6. 选一个动作变化剧烈的task，可视化调制前后动作函数在时域和频域的变化情况
'''
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List
from omegaconf import DictConfig
from src.models.base import BASE, create_bidirectional_mask
from .siren import modules as siren_modules


class NIAF(BASE):
    """
    BEAST 的 Siren 实现版本。

    本模型使用一个 VLM 的解码器作为 Meta-Learner，来生成 Siren 的参数。
    该 Siren 网络将归一化的时间坐标 t 映射为连续的动作序列。

    核心流程:
        1. VLM Encoder: 将图像和语言指令编码为多模态特征序列。
        2. VLM Decoder:
           - 输入: 一组可学习的参数查询 token (Parameter Query tokens)。
           - 上下文: Encoder 输出的上下文特征序列。
           - 输出: 经过上下文信息调节后的参数表征 (Contextualized Parameter tokens)。
        3. 参数生成:
           - 输出的 weight 与 bias 表征经线性头投影，得到用于调制 Siren 网络 weight 和 bias 的向量。
        4. 动作解码:
           - 使用生成的参数配置 Siren 网络。
           - 将时间坐标序列 t (范围 [-1, 1]) 输入 Siren，解码出连续的动作序列 a_t。
    """

    def __init__(
        self, 
        *args, 
        siren: DictConfig | None = None, 
        # 动作模式配置
        action_mode: str = "delta_first",
        # 动作归一化配置 (由 action_stats 注入)
        action_min: List[float] = None,
        action_max: List[float] = None,
        **kwargs
    ) -> None:
        """
        初始化 NIAF 模型。

        参数:
            siren: SIREN 网络配置
            action_mode: 动作模式 ("absolute", "relative", "delta_first")
            action_min: 当前动作模式对应的最小值 (逐维度)
            action_max: 当前动作模式对应的最大值 (逐维度)
        """
        super().__init__(*args, **kwargs)
        
        # =====================================================================
        # 保存子类特有的 hyperparameters (确保推理时能从 checkpoint 恢复)
        # =====================================================================
        self.save_hyperparameters(
            'siren',
            'action_mode',
            'action_min',
            'action_max',
        )
        
        # 1. 读取与校验配置
        if siren is None:
            raise ValueError("缺少 siren 配置")
        self.siren_cfg = siren
        
        # 验证 action_mode 有效性
        valid_modes = ["absolute", "relative", "delta_first"]
        if action_mode not in valid_modes:
            raise ValueError(f"Invalid action_mode: {action_mode}. Must be one of {valid_modes}")
        self.action_mode = action_mode
        
        # 动作归一化 buffer (推理时反归一化)
        if action_min is None or action_max is None:
            raise ValueError(f"使用 {action_mode} 动作模式时需要配置 action_min 和 action_max")
        self.register_buffer('action_min_buf', torch.tensor(action_min, dtype=torch.float32))
        self.register_buffer('action_max_buf', torch.tensor(action_max, dtype=torch.float32))

        # Siren 相关超参数
        self.chunk_size: int = int(self.siren_cfg.get("chunk_size", 20))                    # 预测的动作序列长度
        self.n_groups_per_layer: int = int(self.siren_cfg.get("n_groups_per_layer", 8))     # weight 参数分组数
        self.siren_activation: str = str(self.siren_cfg.get("siren_activation", "sine"))    # Siren 激活函数类型（sine / relu）
        self.bias_mod_scale: float = float(self.siren_cfg.get("bias_mod_scale", 0.02))      # bias 调制强度
        self.fp32: bool = bool(self.siren_cfg.get("fp32", True))                              # 在 Siren 的 forward 中使用FP32

        # Siren 网络结构参数
        siren_hidden_dim: int = int(self.siren_cfg.get("siren_hidden_dim", 64))             # Siren隐藏层维度
        siren_num_layers: int = int(self.siren_cfg.get("siren_num_layers", 3))              # Siren隐藏层数量

        # 2. 定义 Hypo-Net (Siren)
        self.hypo_net = siren_modules.SingleBVPNet(
            out_features=self.action_dim,                   # 输出动作维度
            type=self.siren_activation,                     # 激活函数类型
            in_features=1,                                  # 输入时间坐标维度
            hidden_features=siren_hidden_dim,               # 隐藏层维度
            num_hidden_layers=siren_num_layers,             # 隐藏层数量
        )
        self.hypo_param_shapes: Dict[str, torch.Size] = { name.replace(".", "_"): p.shape for name, p in self.hypo_net.meta_named_parameters() }    # 记录 Siren 各层参数的形状信息
        self._param_name_map: Dict[str, str] = { name: name.replace(".", "_") for name, _ in self.hypo_net.meta_named_parameters() }                # 构建参数名映射表，用于兼容旧版本的 weight 加载

        # 3. 分配参数查询 token
        d_model: int = int(self.vlm.config.text_config.d_model)     # VLM解码器的特征维度
        n_wtokens = 0                                               # token 总数
        self.wtoken_rng = {}                                        # weight 参数的 token 索引范围
        self.btoken_rng = {}                                        # bias 参数的 token 索引范围
        
        for name, shape in self.hypo_param_shapes.items():
            if "bias" in name:
                # 每层 bias 分配一个 token
                self.btoken_rng[name] = (n_wtokens, n_wtokens + 1)                  # 记录 token 索引范围
                n_wtokens += 1                                                      # 累加 token 计数
            else:
                # weight 采用分组方式分配 token
                out_features = int(shape[0])                                        # 输出神经元数量
                g = min(self.n_groups_per_layer, out_features)                      # 该层的分组数
                if out_features % g != 0:
                    raise ValueError( f" weight  {name} 的 out_features={out_features} 必须能被 n_groups_per_layer={g} 整除" )
                self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)                  # 记录 token 索引范围
                n_wtokens += g                                                      # 累加 token 计数

        # 初始化参数查询 token, 使用小标准差的随机初始化
        self.param_queries = nn.Parameter(torch.randn(1, n_wtokens, d_model) * 0.01)

        # 4. 定义基础参数 (base_params) 与投影头
        self.base_params = nn.ParameterDict()                       # 存储 Siren 的基础参数
        self.wtoken_postfc = nn.ModuleDict()                        # weight 调制投影头
        self.btoken_postfc = nn.ModuleDict()                        # bias 调制投影头

        for name, shape in self.hypo_param_shapes.items():
            self.base_params[name] = nn.Parameter(self._init_siren_param(name, shape))  # 初始化基础参数
            
            if "bias" in name:
                # bias 调制投影头
                # 输入: (B, 1, d_model)
                # 输出: (B, 1, bias_dim)
                bias_dim = int(shape[0])
                self.btoken_postfc[name] = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, bias_dim),
                    nn.Tanh()
                )
            else:
                # weight 调制投影头
                # 输入: (B, g, d_model)
                # 输出: (B, g, in_features) 每组预测一个 in_features 维的调制向量
                output_dim = int(shape[1])
                self.wtoken_postfc[name] = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, output_dim),
                    nn.Tanh()
                )

    # ---------------------------------------------------------------------
    #                          参数与加载辅助
    # ---------------------------------------------------------------------
    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """
        自定义 weight 加载逻辑,兼容使用旧命名方式(含'.')的模型存档。

        参数:
            state_dict: 待加载的状态字典
            prefix: 参数名前缀
            其他参数: PyTorch标准加载参数
        """
        new_state_dict = state_dict.copy()
        # 将旧版本的参数名(含'.')转换为新版本(含'_')
        for key in list(new_state_dict.keys()):
            for old_base_key, new_base_key in self._param_name_map.items():
                old_full_key = f"{prefix}base_params.{old_base_key}"
                new_full_key = f"{prefix}base_params.{new_base_key}"
                if key == old_full_key:
                    new_state_dict[new_full_key] = new_state_dict.pop(key)

        super()._load_from_state_dict(
            new_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _init_siren_param(self, name: str, shape: torch.Size) -> torch.Tensor:
        """
        初始化 Siren 的基础参数。

        Siren 初始化规则:
          - 第一层 weight : U(-1/in, 1/in)
          - 其他层 weight : U(-√(6/in)/ω₀, √(6/in)/ω₀) 其中ω₀=30
          - 所有 bias : 小随机值 ~N(0, 0.01²)
        
        ReLU 初始化规则: Kaiming初始化

        参数:
            name: 参数名称
            shape: 参数形状

        返回:
            初始化后的参数张量
        """
        if self.siren_activation == "sine":
            # bias 初始化
            if "bias" in name:
                return torch.randn(shape) * 0.01            #  bias 初始化为小随机值,为相位提供初始多样性
            
            # weight 初始化
            w = torch.empty(shape)
            fan_in = int(shape[1])                          # 输入维度
            omega0 = 30.0                                   # Siren 频率参数
            
            if "net_0_0_weight" in name:                    # 第一层使用特殊的初始化范围
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / omega0
            
            nn.init.uniform_(w, -bound, +bound)
            return w
        
        # ReLU 或其他激活函数的初始化
        # bias 初始化 
        if "bias" in name:
            w_name = name.replace("bias", "weight")
            if w_name in self.hypo_param_shapes:
                w_shape = self.hypo_param_shapes[w_name]
                fan_in = int(w_shape[1])
                bound = 1.0 / math.sqrt(fan_in)
                return torch.empty(shape).uniform_(-bound, bound)
            return torch.zeros(shape)

        # weight 使用 Kaiming 初始化
        w = torch.empty(shape)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        return w


    # ---------------------------------------------------------------------
    #                            工具函数
    # ---------------------------------------------------------------------
    def denormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """
        将归一化的动作 [-1, 1] 反归一化到原始范围 [action_min, action_max]。
        
        公式: action = (normalized + 1) / 2 * (max - min) + min
        
        支持维度自动适配：当模型输出维度与统计量维度不匹配时，
        根据 arm_mode 切片 action_min/action_max。

        参数:
            normalized_actions: 归一化的动作张量，形状为 (..., action_dim)

        返回:
            反归一化后的动作张量
        """
        action_min = self.action_min_buf.to(normalized_actions.device)
        action_max = self.action_max_buf.to(normalized_actions.device)
        
        # 自动适配维度
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

    @staticmethod
    def _make_time_coords(
        batch_size: int, steps: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        生成归一化到 [-1, 1] 区间的时间坐标序列。

        参数:
            batch_size: 批次大小
            steps: 时间步数
            device: 设备
            dtype: 数据类型

        返回:
            时间坐标张量,形状为 (B, T, 1)
        """
        t = torch.linspace(-1.0, 1.0, steps=steps, device=device, dtype=dtype)  # 在[-1,1]区间均匀采样
        coords = t.view(1, steps, 1).expand(batch_size, -1, -1)                # 扩展到批次维度
        return coords

    @staticmethod
    def _iter_modalities(batch: Dict) -> Dict[str, Dict]:
        """
        提供统一接口,用于迭代处理单模态或多模态的batch数据。

        支持的输入格式:
            1. 单模态: {"actions": ...}
            2. 多模态: {"rgb": {"actions": ...}, "depth": {"actions": ...}}

        参数:
            batch: 输入批次数据

        返回:
            模态名到对应数据的字典映射
        """
        # 检查是否为单模态格式
        if isinstance(batch, dict) and "actions" in batch:
            return {"default": batch}
        
        # 检查是否为多模态格式
        if isinstance(batch, dict) and all(isinstance(v, dict) and "actions" in v for v in batch.values()):
            return batch
        
        # 默认情况
        if isinstance(batch, dict):
            return {"default": batch}
        
        raise TypeError(f"不支持的 batch 数据结构: {type(batch)}")


    # ---------------------------------------------------------------------
    #                          生成动作主流程
    # ---------------------------------------------------------------------
    def _generate_actions_siren(self, batch: Dict) -> torch.Tensor:
        """
        根据多模态输入生成一个时间窗口内的动作序列。

        流程:
            1. 通过 VLM Encoder 编码输入特征
            2. 通过 VLM Decoder 处理参数查询 token
            3. 根据上下文化的 token 生成 Siren 参数调制值
            4. 应用调制得到最终的 Siren 参数
            5. 使用 Siren 解码时间坐标为动作序列

        参数:
            batch: 包含观测和任务信息的批次数据

        返回:
            预测的动作序列，形状为 (B, chunk_size, action_dim)
        """
        device = self.device
        default_dtype = next(self.parameters()).dtype

        # 1. Encoder 编码多模态输入特征
        features, encoder_attn_mask = self.compute_input_features(batch)        # 提取编码器特征 (B, N_data, d_model)
        B = features.size(0)                                                    # 批次大小

        # 2. Decoder 处理参数查询 token
        decoder_input_embeds = self.param_queries.expand(B, -1, -1)             # 扩展参数查询到批次维度 (B, N_q, d_model)
        decoder_outputs = self.vlm.get_decoder()(
            inputs_embeds=decoder_input_embeds,                                 # 参数查询 token
            encoder_hidden_states=features,                                     # Encoder 输出的多模态特征
            encoder_attention_mask=encoder_attn_mask,                           # Encoder 注意力掩码
            attention_mask=create_bidirectional_mask(B, self.param_queries.size(1), device),  # Decoder 双向注意力掩码
        )
        contextualized_wtokens = decoder_outputs[0]                             # 上下文化的参数 token (B, N_q, d_model)

        # 3. 调制生成 Siren 参数
        hypo_params: OrderedDict[str, torch.Tensor] = OrderedDict()
        
        for name, shape in self.hypo_param_shapes.items():
            if "bias" in name:
                # 加性调制生成 bias 参数
                start, end = self.btoken_rng[name]                              # 对应的 token 索引范围
                token_slice = contextualized_wtokens[:, start:end, :]           # 提取 bias 对应的上下文 token (B, 1, d_model)
                delta_b_raw = self.btoken_postfc[name](token_slice).squeeze(1)  # 投影为调制值并去掉中间维度 (B, bias_dim)
                b_base = self.base_params[name].unsqueeze(0)                    # 获取基础 bias 并扩展批次维度 (B, bias_dim)
                b_modulated = b_base + self.bias_mod_scale * delta_b_raw        # 加性调制: b = b_base + scale*Δb
                hypo_params[name] = b_modulated                                 # 保存调制后的 bias 参数

            else:
                # 乘性调制生成 weight 参数
                start, end = self.wtoken_rng[name]                              # 对应的 token 索引范围
                token_slice = contextualized_wtokens[:, start:end, :]           # 提取 weight 对应的上下文 token (B, g, d_model)
                u_raw = self.wtoken_postfc[name](token_slice)                   # 投影为各组的调制向量 (B, g, in_features)
                u = 1.0 + u_raw                                                 # 计算缩放因子: u = 1 + u_raw
                g = u.shape[1]                                                  # 分组数
                out_features = int(shape[0])                                    # 输出神经元数量
                u_repeated = u.repeat_interleave(out_features // g, dim=1)      # 扩展分组缩放因子到完整维度 (B, out_features, in_features)
                w_base = self.base_params[name].unsqueeze(0)                    # 获取基础 weight 并扩展批次维度 (1, out_features, in_features)
                w_modulated = w_base * u_repeated                               # 乘性调制: w = w_base ⊙ u
                w_modulated = w_modulated + 1e-7                                # 避免数值问题
                hypo_params[name] = w_modulated                                 # 保存调制后的 weight 参数

        # 4. 恢复Siren期望的参数名格式('_' → '.')
        final_hypo_params = OrderedDict(
            (name.replace("_", "."), val) for name, val in hypo_params.items()
        )

        # 5. 生成时间坐标并通过Siren解码为动作序列
        coords = self._make_time_coords(B, self.chunk_size, device, default_dtype)  # 生成时间坐标 (B, T, 1)
        model_input = {"coords": coords}                                            # 构造 Siren 的时间序列输入

        if self.fp32:
            fp32_params = OrderedDict((k, v.float()) for k, v in final_hypo_params.items())     # 转换为FP32
            with torch.amp.autocast('cuda', enabled=False):                                        # 禁用自动混合精度
                actions_pred = self.hypo_net(model_input, params=fp32_params)["model_out"]      # Siren inference (B, T, action_dim)
        else:
            actions_pred = self.hypo_net(model_input, params=final_hypo_params)["model_out"]

        return actions_pred


    # ---------------------------------------------------------------------
    #                         Lightning训练流程
    # ---------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """
        训练步骤,计算多模态输入的MSE损失。

        参数:
            batch: 训练批次数据
            batch_idx: 批次索引

        返回:
            总损失值
        """
        total_loss = torch.tensor(0.0, device=self.device)                      # 累积总损失
        total_mse = torch.tensor(0.0, device=self.device)                       # 累积 MSE
        total_bs = 0                                                            # 累积批次大小

        # 迭代处理所有模态的数据
        modalities = self._iter_modalities(batch)
        for modality_scope, dataset_batch in modalities.items():
            self.modality_scope = modality_scope                                # 设置当前模态
            actions_gt = dataset_batch["actions"].to(self.device)               # 获取真值动作 (B, T, action_dim)
            B, T_gt, _ = actions_gt.shape
            T = min(self.chunk_size, T_gt)                                      # 取预测长度和真值长度的较小值

            # 生成预测动作
            actions_pred = self._generate_actions_siren(dataset_batch)          # (B, chunk_size, action_dim)
            # 计算MSE损失
            mse = F.mse_loss(actions_pred, actions_gt[:, :T, :])

            total_loss += mse
            total_mse += mse.detach()
            total_bs += B

        # 计算平均损失
        total_loss /= len(modalities)
        avg_mse = total_mse / len(modalities)

        # 记录训练指标
        self.log("train/siren_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/siren_mse", avg_mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        验证步骤,计算验证集上的MSE损失。

        参数:
            batch: 验证批次数据
            batch_idx: 批次索引

        返回:
            包含验证损失的字典
        """
        total_loss = torch.tensor(0.0, device=self.device)                      # 累积总损失
        total_mse = torch.tensor(0.0, device=self.device)                       # 累积MSE
        total_bs = 0                                                            # 累积批次大小

        # 迭代处理所有模态的数据
        modalities = self._iter_modalities(batch)
        for modality_scope, dataset_batch in modalities.items():
            self.modality_scope = modality_scope                                # 设置当前模态
            actions_gt = dataset_batch["actions"].to(self.device)               # 获取真值动作 (B, T, action_dim)
            B, T_gt, _ = actions_gt.shape
            T = min(self.chunk_size, T_gt)                                      # 取预测长度和真值长度的较小值

            # 生成预测动作
            actions_pred = self._generate_actions_siren(dataset_batch)          # (B, chunk_size, action_dim)
            # 计算MSE损失
            mse = F.mse_loss(actions_pred, actions_gt[:, :T, :])

            total_loss += mse
            total_mse += mse.detach()
            total_bs += B

        # 计算平均损失
        total_loss /= len(modalities)
        avg_mse = total_mse / len(modalities)

        # 记录验证指标
        self.log("val/siren_loss", total_loss, sync_dist=True, batch_size=total_bs)
        self.log("val/siren_mse", avg_mse, sync_dist=True, batch_size=total_bs)

        return {"val_loss": total_loss, "val_mse": avg_mse}

    # ---------------------------------------------------------------------
    #                        推理接口
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        标准前向传播接口,用于生成完整的动作序列。
        支持灵活的多相机配置（ALOHA/LIBERO/CALVIN等）。

        参数:
            obs: 观测字典
                - 新格式（ALOHA）: obs['rgb_static'], obs['rgb_left_wrist'], obs['rgb_right_wrist']
                - 旧格式: obs['rgb_obs']['rgb_static'], obs['rgb_obs']['rgb_gripper']
            goal: 目标字典,需包含 goal['lang_text']

        返回:
            预测的动作序列,形状为 (1, chunk_size, action_dim)
        """
        lang_text = goal.get("lang_text", "")                                   # 语言指令
        
        batch = {"lang_text": [lang_text], "rgb_obs": {}}
        for cam_key in self.rgb_obs_keys:
            if cam_key in obs:
                batch["rgb_obs"][cam_key] = obs[cam_key]
            elif "rgb_obs" in obs and cam_key in obs["rgb_obs"]:
                batch["rgb_obs"][cam_key] = obs["rgb_obs"][cam_key]

        # 推理生成动作序列 (归一化空间 [-1, 1])
        actions_normalized = self._generate_actions_siren(batch)
        
        # 反归一化到原始动作空间
        actions = self.denormalize_actions(actions_normalized)
        
        return actions


    # ---------------------------------------------------------------------
    #                        调试辅助方法
    # ---------------------------------------------------------------------
    def get_modulation_stats(self) -> Dict[str, float]:
        """
        获取当前模型的调制参数统计信息。

        返回:
            包含各层调制强度的字典
        """
        return {
            "bias_mod_scale": self.bias_mod_scale,
        }