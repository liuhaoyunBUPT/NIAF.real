# src/datasets/data_module.py
"""
机器人训练数据模块
支持直接读取HDF5格式的数据文件
"""
from __future__ import annotations

import os
from typing import Dict, Any, Optional, List
import logging

import hydra
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.datasets.episode_dataset import (
    MultiEpisodeDataset,
    collate_fn,
)
from src.utils.transforms import NormalizeActions

logger = logging.getLogger(__name__)


def _build_transforms(stage_cfg: DictConfig) -> Dict[str, Any]:
    """
    构建数据变换
    stage_cfg: transforms.train 或 transforms.val
    """
    out = {}
    if stage_cfg is None:
        return out
        
    for k, ops in stage_cfg.items():
        op_list = [hydra.utils.instantiate(op) for op in ops]
        
        def _apply(x, op_list=op_list):
            for op in op_list:
                x = op(x)
            return x
        out[k] = _apply
    return out


class NIAFDataModule(pl.LightningDataModule):
    """
    ALOHA双臂机器人数据模块
    
    直接从HDF5文件加载数据，支持：
    - 三个相机视角 (cam_high, cam_left_wrist, cam_right_wrist)
    - 14维动作 (7+7 双臂)
    - 14维状态 (qpos)
    - 懒加载模式（节省内存）
    """
    
    def __init__(
        self,
        root_data_dir: str,
        action_seq_len: int = 20,
        batch_size: int = 16,
        num_workers: int = 8,
        camera_keys: Dict[str, str] = None,
        state_key: str = "observations/qpos",
        action_key: str = "action",
        transforms: DictConfig = None,
        val_split: float = 0.1,
        seed: int = 42,
        subsample_rate: int = 1,
        lang_text: str = "perform the manipulation task",
        max_episodes: Optional[int] = None,
        lazy_load: bool = True,
        action_mode: str = "delta_first",  # "absolute", "relative", "delta_first"
        arm_mode: str = "dual",  # "dual", "left", "right"
        # 相机开关
        use_cam_high: bool = True,
        use_cam_left_wrist: bool = True,
        use_cam_right_wrist: bool = True,
        # 速度监督相关
        vel_key: str = None,  # HDF5中速度数据的键，如 "observations/qvel"
        return_joint_vel: bool = False,  # 是否返回关节速度
        # 逐维度动作归一化范围
        action_min_absolute: List[float] = None,
        action_max_absolute: List[float] = None,
        action_min_relative: List[float] = None,
        action_max_relative: List[float] = None,
        action_min_delta_first: List[float] = None,
        action_max_delta_first: List[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.root_data_dir = root_data_dir
        self.action_seq_len = action_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.subsample_rate = subsample_rate
        self.lang_text = lang_text
        self.max_episodes = max_episodes
        self.lazy_load = lazy_load
        self.action_mode = action_mode
        self.arm_mode = arm_mode
        
        # 速度监督相关
        self.vel_key = vel_key
        self.return_joint_vel = return_joint_vel
        
        # 逐维度动作归一化参数
        # 格式: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        self.action_min_absolute = action_min_absolute
        self.action_max_absolute = action_max_absolute
        self.action_min_relative = action_min_relative
        self.action_max_relative = action_max_relative
        self.action_min_delta_first = action_min_delta_first
        self.action_max_delta_first = action_max_delta_first
        
        logger.info(f"Action mode: {self.action_mode}")
        logger.info(f"Arm mode: {arm_mode}")
        
        # 根据相机开关构建相机映射
        camera_keys = {}
        if use_cam_high:
            camera_keys["rgb_static"] = "observations/images/cam_high"
        if use_cam_left_wrist:
            camera_keys["rgb_left_wrist"] = "observations/images/cam_left_wrist"
        if use_cam_right_wrist:
            camera_keys["rgb_right_wrist"] = "observations/images/cam_right_wrist"
        
        if len(camera_keys) == 0:
            raise ValueError("At least one camera must be enabled!")
        
        logger.info(f"Active cameras: {list(camera_keys.keys())}")
        self.camera_keys = camera_keys
        self.state_key = state_key
        self.action_key = action_key
        
        # 数据变换配置
        self.transforms_cfg = transforms
        
        self.train_dataset = None
        self.val_dataset = None
    
    @property
    def delta_first_action_stats(self) -> Dict[str, List[float]]:
        """
        返回 delta_first 动作归一化统计信息，供模型配置引用
        根据 arm_mode 自动切片
        """
        if self.action_min_delta_first is None or self.action_max_delta_first is None:
            return None
        
        min_vec = list(self.action_min_delta_first)
        max_vec = list(self.action_max_delta_first)
        
        # 根据 arm_mode 切片
        if self.arm_mode == "left":
            min_vec = min_vec[:7]
            max_vec = max_vec[:7]
        elif self.arm_mode == "right":
            min_vec = min_vec[7:14]
            max_vec = max_vec[7:14]
        
        return {"min": min_vec, "max": max_vec}
        
    def setup(self, stage: Optional[str] = None):
        """准备数据集"""
        if stage == "fit" or stage is None:
            # 构建训练变换
            train_transforms = {}
            val_transforms = {}
            
            if self.transforms_cfg is not None:
                if "train" in self.transforms_cfg:
                    train_transforms = _build_transforms(self.transforms_cfg.train)
                if "val" in self.transforms_cfg:
                    val_transforms = _build_transforms(self.transforms_cfg.val)
            
            # 构建动作归一化变换 (逐维度)
            # 动作维度 14: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
            if self.action_mode == "relative":
                # 相对动作使用 relative 归一化范围
                if self.action_min_relative is None or self.action_max_relative is None:
                    raise ValueError("使用 relative 动作模式时需要配置 action_min_relative 和 action_max_relative")
                min_vec = list(self.action_min_relative)
                max_vec = list(self.action_max_relative)
                logger.info(f"Using relative action normalization (per-dim)")
            elif self.action_mode == "delta_first":
                # delta_first 使用独立的 delta_first 归一化范围
                if self.action_min_delta_first is None or self.action_max_delta_first is None:
                    raise ValueError("使用 delta_first 动作模式时需要配置 action_min_delta_first 和 action_max_delta_first")
                min_vec = list(self.action_min_delta_first)
                max_vec = list(self.action_max_delta_first)
                logger.info(f"Using delta_first action normalization (per-dim)")
            else:
                # 使用绝对动作的归一化范围
                if self.action_min_absolute is None or self.action_max_absolute is None:
                    raise ValueError("使用绝对动作时需要配置 action_min_absolute 和 action_max_absolute")
                min_vec = list(self.action_min_absolute)
                max_vec = list(self.action_max_absolute)
                logger.info(f"Using absolute action normalization (per-dim)")
            
            # 根据arm_mode切片归一化参数
            if self.arm_mode == "left":
                min_vec = min_vec[:7]
                max_vec = max_vec[:7]
            elif self.arm_mode == "right":
                min_vec = min_vec[7:14]
                max_vec = max_vec[7:14]
            
            action_transforms = NormalizeActions(min_vec, max_vec)
            logger.info(f"Action normalization (arm_mode={self.arm_mode}): min={min_vec}, max={max_vec}")

            # 创建完整数据集
            full_dataset = MultiEpisodeDataset(
                data_dir=self.root_data_dir,
                action_seq_len=self.action_seq_len,
                camera_keys=self.camera_keys,
                state_key=self.state_key,
                action_key=self.action_key,
                transforms=train_transforms,
                action_transforms=action_transforms,
                subsample_rate=self.subsample_rate,
                lang_text=self.lang_text,
                max_episodes=self.max_episodes,
                lazy_load=self.lazy_load,
                action_mode=self.action_mode,
                arm_mode=self.arm_mode,
                # 速度监督相关
                vel_key=self.vel_key,
                return_joint_vel=self.return_joint_vel,
            )
            
            # 划分训练集和验证集
            total_len = len(full_dataset)
            val_len = int(total_len * self.val_split)
            train_len = total_len - val_len
            
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, 
                [train_len, val_len],
                generator=generator
            )
            
            logger.info(f"Dataset split: train={train_len}, val={val_len}")
            
    def train_dataloader(self):
        """返回训练数据加载器"""
        return {
            "lang": DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
                prefetch_factor=2 if self.num_workers > 0 else None,
            )
        }
    
    def val_dataloader(self):
        """返回验证数据加载器"""
        return [
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                persistent_workers=self.num_workers > 0,
                prefetch_factor=2 if self.num_workers > 0 else None,
            )
        ]
