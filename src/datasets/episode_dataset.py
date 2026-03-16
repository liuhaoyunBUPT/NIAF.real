# src/datasets/episode_dataset.py
"""
HDF5格式机器人数据集
支持直接读取HDF5格式的数据文件
"""
from __future__ import annotations

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EpisodeDataset(Dataset):
    """
    单个Episode的序列数据集
    从一个episode中采样固定长度的动作序列用于训练
    """
    
    def __init__(
        self,
        hdf5_path: str,
        action_seq_len: int = 20,
        camera_keys: Dict[str, str] = None,
        state_key: str = "observations/qpos",
        action_key: str = "action",
        transforms: Optional[Dict[str, Any]] = None,
        subsample_rate: int = 1,  # 每隔多少帧采样一次
    ):
        self.hdf5_path = hdf5_path
        self.action_seq_len = action_seq_len
        self.subsample_rate = subsample_rate
        self.transforms = transforms or {}
        
        # 默认相机映射
        if camera_keys is None:
            camera_keys = {
                "rgb_static": "observations/images/cam_high",
                "rgb_left_wrist": "observations/images/cam_left_wrist",
                "rgb_right_wrist": "observations/images/cam_right_wrist",
            }
        self.camera_keys = camera_keys
        self.state_key = state_key
        self.action_key = action_key
        
        # 加载数据到内存
        self._load_data()
        
    def _load_data(self):
        """加载HDF5数据到内存"""
        with h5py.File(self.hdf5_path, 'r') as f:
            # 加载动作
            self.actions = f[self.action_key][:].astype(np.float32)
            
            # 加载状态
            self.states = f[self.state_key][:].astype(np.float32)
            
            # 加载图像
            self.images = {}
            for out_key, src_key in self.camera_keys.items():
                self.images[out_key] = f[src_key][:]  # (T, H, W, C) uint8
                
        self.episode_length = len(self.actions)
        
        # 计算有效的起始索引（考虑动作序列长度和subsample）
        self.valid_start_indices = list(range(
            0, 
            self.episode_length - self.action_seq_len * self.subsample_rate + 1,
            self.subsample_rate
        ))
        
    def __len__(self):
        return len(self.valid_start_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start_idx = self.valid_start_indices[idx]
        
        # 获取动作序列
        action_indices = [start_idx + i * self.subsample_rate 
                         for i in range(self.action_seq_len)]
        action_indices = [min(i, self.episode_length - 1) for i in action_indices]
        actions = self.actions[action_indices]  # (action_seq_len, 14)
        
        # 获取当前帧观测
        obs_idx = start_idx
        
        # 处理图像
        rgb_obs = {}
        for out_key in self.camera_keys.keys():
            img = self.images[out_key][obs_idx]  # (H, W, C)
            img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)  # (1, C, H, W)
            
            if out_key in self.transforms:
                img = self.transforms[out_key](img)
            
            rgb_obs[out_key] = img
            
        # 获取状态
        robot_obs = torch.from_numpy(self.states[obs_idx:obs_idx+1]).float()  # (1, 14)
        
        # 动作
        actions = torch.from_numpy(actions).float()  # (action_seq_len, 14)
        
        return {
            "rgb_obs": rgb_obs,
            "robot_obs": robot_obs,
            "actions": actions,
        }


class MultiEpisodeDataset(Dataset):
    """
    多Episode数据集
    从多个HDF5文件中加载数据
    支持懒加载模式以节省内存
    """
    
    def __init__(
        self,
        data_dir: str,
        action_seq_len: int = 20,
        camera_keys: Dict[str, str] = None,
        state_key: str = "observations/qpos",
        action_key: str = "action",
        transforms: Optional[Dict[str, Any]] = None,
        action_transforms: Optional[Any] = None,
        subsample_rate: int = 1,
        lang_text: str = "pick up the soft object",  # 任务描述
        max_episodes: Optional[int] = None,  # 限制加载的episode数量
        lazy_load: bool = True,  # 懒加载模式（节省内存）
        action_mode: str = "delta_first",  # "absolute", "relative", "delta_first"
        arm_mode: str = "dual",  # "dual", "left", "right"
        # 速度监督相关
        vel_key: str = None,  # HDF5中速度数据的键，如 "observations/qvel"
        return_joint_vel: bool = False,  # 是否返回关节速度
    ):
        self.data_dir = data_dir
        self.action_seq_len = action_seq_len
        self.subsample_rate = subsample_rate
        self.transforms = transforms or {}
        self.action_transforms = action_transforms
        self.lang_text = lang_text
        self.lazy_load = lazy_load
        
        # 速度监督相关
        self.vel_key = vel_key
        self.return_joint_vel = return_joint_vel
        
        # 动作模式
        valid_modes = ["absolute", "relative", "delta_first"]
        if action_mode not in valid_modes:
            raise ValueError(f"Invalid action_mode: {action_mode}. Must be one of {valid_modes}")
        self.action_mode = action_mode
        
        self.arm_mode = arm_mode
        
        logger.info(f"Action mode: {self.action_mode}")
        
        # 根据arm_mode确定动作/状态切片
        if self.arm_mode == "left":
            self.arm_slice = slice(0, 7)
            self.gripper_indices = [6]  # 相对于切片后的索引
        elif self.arm_mode == "right":
            self.arm_slice = slice(7, 14)
            self.gripper_indices = [6]  # 切片后索引6对应原始索引13
        else:  # dual
            self.arm_slice = slice(0, 14)
            self.gripper_indices = [6, 13]
        
        logger.info(f"Arm mode: {self.arm_mode}, slice: {self.arm_slice}")
        
        # 默认相机映射
        if camera_keys is None:
            camera_keys = {
                "rgb_static": "observations/images/cam_high",
                "rgb_left_wrist": "observations/images/cam_left_wrist",
                "rgb_right_wrist": "observations/images/cam_right_wrist",
            }
        self.camera_keys = camera_keys
        self.state_key = state_key
        self.action_key = action_key
        
        # 查找所有HDF5文件
        self.hdf5_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
        if max_episodes is not None:
            self.hdf5_files = self.hdf5_files[:max_episodes]
            
        if len(self.hdf5_files) == 0:
            raise ValueError(f"No episode files found in {data_dir}")
            
        logger.info(f"Found {len(self.hdf5_files)} episodes in {data_dir}")
        
        if self.lazy_load:
            self._build_index()
        else:
            self._load_all_data()
    
    def _build_index(self):
        """构建索引（懒加载模式）"""
        self.episode_lengths = []
        self.episode_boundaries = [0]
        
        for hdf5_path in self.hdf5_files:
            with h5py.File(hdf5_path, 'r') as f:
                ep_length = f[self.action_key].shape[0]
                self.episode_lengths.append(ep_length)
                self.episode_boundaries.append(self.episode_boundaries[-1] + ep_length)
        
        # 计算有效的采样索引
        self.valid_indices = []
        for ep_idx in range(len(self.hdf5_files)):
            ep_length = self.episode_lengths[ep_idx]
            
            for i in range(0, ep_length - self.action_seq_len * self.subsample_rate + 1, 
                          self.subsample_rate):
                self.valid_indices.append((ep_idx, i))
                
        logger.info(f"Total valid samples: {len(self.valid_indices)} (lazy load mode)")
        
    def _load_all_data(self):
        """加载所有episode数据到内存"""
        self.all_actions = []
        self.all_states = []
        self.all_images = {k: [] for k in self.camera_keys.keys()}
        self.all_velocities = [] if self.return_joint_vel and self.vel_key else None
        self.episode_boundaries = [0]  # 每个episode的起始索引
        
        for hdf5_path in self.hdf5_files:
            with h5py.File(hdf5_path, 'r') as f:
                actions = f[self.action_key][:].astype(np.float32)
                states = f[self.state_key][:].astype(np.float32)
                
                self.all_actions.append(actions)
                self.all_states.append(states)
                
                # 加载速度数据 (如果需要)
                if self.all_velocities is not None and self.vel_key in f:
                    velocities = f[self.vel_key][:].astype(np.float32)
                    self.all_velocities.append(velocities)
                
                for out_key, src_key in self.camera_keys.items():
                    self.all_images[out_key].append(f[src_key][:])
                    
                self.episode_boundaries.append(
                    self.episode_boundaries[-1] + len(actions)
                )
                
        # 合并所有数据
        self.all_actions = np.concatenate(self.all_actions, axis=0)
        self.all_states = np.concatenate(self.all_states, axis=0)
        if self.all_velocities is not None and len(self.all_velocities) > 0:
            self.all_velocities = np.concatenate(self.all_velocities, axis=0)
        else:
            self.all_velocities = None
        for k in self.all_images:
            self.all_images[k] = np.concatenate(self.all_images[k], axis=0)
            
        logger.info(f"Loaded {len(self.all_actions)} frames from {len(self.hdf5_files)} episodes")
        if self.all_velocities is not None:
            logger.info(f"Loaded velocity data with shape {self.all_velocities.shape}")
        
        # 计算有效的采样索引（不跨越episode边界）
        self.valid_indices = []
        for ep_idx in range(len(self.hdf5_files)):
            ep_start = self.episode_boundaries[ep_idx]
            ep_end = self.episode_boundaries[ep_idx + 1]
            ep_length = ep_end - ep_start
            
            for i in range(0, ep_length - self.action_seq_len * self.subsample_rate + 1, 
                          self.subsample_rate):
                self.valid_indices.append(ep_start + i)
                
        logger.info(f"Total valid samples: {len(self.valid_indices)}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.lazy_load:
            return self._getitem_lazy(idx)
        else:
            return self._getitem_memory(idx)
    
    def _getitem_lazy(self, idx: int) -> Dict[str, Any]:
        """懒加载模式的getitem"""
        ep_idx, start_in_ep = self.valid_indices[idx]
        hdf5_path = self.hdf5_files[ep_idx]
        
        with h5py.File(hdf5_path, 'r') as f:
            ep_length = self.episode_lengths[ep_idx]
            
            # 获取动作序列的索引
            action_indices = [min(start_in_ep + i * self.subsample_rate, ep_length - 1)
                             for i in range(self.action_seq_len)]
            
            # 读取动作序列
            actions_full = f[self.action_key][action_indices].astype(np.float32)  # (action_seq_len, 14)
            
            # 根据arm_mode切片动作
            actions = actions_full[:, self.arm_slice]
            
            if self.action_mode == "relative":
                # 相对动作: delta_t = action_t - action_{t-1}
                prev_actions = np.empty_like(actions)
                prev_actions[1:] = actions[:-1]
                
                # 单独读取第一个prev_action
                first_prev_idx = max(action_indices[0] - self.subsample_rate, 0)
                prev_action_full = f[self.action_key][first_prev_idx].astype(np.float32)
                prev_actions[0] = prev_action_full[self.arm_slice]
                
                # 计算相对动作
                delta_actions = actions - prev_actions
                
                # 首帧相对动作: action_0 - state_0
                for i, idx in enumerate(action_indices):
                    if idx == 0:
                        state_0 = f[self.state_key][0].astype(np.float32)[self.arm_slice]
                        delta_actions[i] = actions[i] - state_0
                
                # 夹爪维度保持绝对值
                for g_idx in self.gripper_indices:
                    delta_actions[:, g_idx] = actions[:, g_idx]
                
                actions = delta_actions
                
            elif self.action_mode == "delta_first":
                # delta_first模式: 整个chunk内各时间步的动作都减去chunk起始时刻的state_0
                # action_t = action_t - state_0，把起点平移到原点
                state_0 = f[self.state_key][action_indices[0]].astype(np.float32)[self.arm_slice]
                delta_actions = actions - state_0[np.newaxis, :]
                
                # 夹爪维度保持绝对值（不做差值）
                for g_idx in self.gripper_indices:
                    delta_actions[:, g_idx] = actions[:, g_idx]
                
                actions = delta_actions
            # else: absolute 模式，actions 保持不变
            
            # 获取当前帧观测
            obs_idx = start_in_ep
            
            # 处理图像
            rgb_obs = {}
            for out_key, src_key in self.camera_keys.items():
                img = f[src_key][obs_idx]  # (H, W, C)
                img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)  # (1, C, H, W)
                
                if out_key in self.transforms:
                    img = self.transforms[out_key](img)
                
                rgb_obs[out_key] = img
                
            # 获取状态 (根据arm_mode切片)
            robot_obs_full = f[self.state_key][obs_idx].astype(np.float32)
            robot_obs = torch.from_numpy(robot_obs_full[self.arm_slice]).unsqueeze(0)  # (1, D)
            
            # 获取关节速度 (如果需要)
            joint_vel = None
            if self.return_joint_vel and self.vel_key is not None:
                vel_full = f[self.vel_key][action_indices].astype(np.float32)  # (action_seq_len, 14)
                joint_vel = torch.from_numpy(vel_full[:, self.arm_slice])  # (action_seq_len, 7 or 14)
        
        actions = torch.from_numpy(actions).float()
        
        if self.action_transforms is not None:
            actions = self.action_transforms(actions)
        
        result = {
            "rgb_obs": rgb_obs,
            "robot_obs": robot_obs,
            "actions": actions,
            "lang_text": self.lang_text,
        }
        
        if joint_vel is not None:
            result["joint_vel"] = joint_vel
        
        return result
    
    def _getitem_memory(self, idx: int) -> Dict[str, Any]:
        """内存加载模式的getitem"""
        start_idx = self.valid_indices[idx]
        
        # 获取动作序列的索引
        action_indices = [start_idx + i * self.subsample_rate 
                         for i in range(self.action_seq_len)]
        
        # 读取动作序列 (根据arm_mode切片)
        actions_full = self.all_actions[action_indices]  # (action_seq_len, 14)
        actions = actions_full[:, self.arm_slice]
        
        if self.action_mode == "relative":
            # 相对动作: delta_t = action_t - action_{t-1}
            prev_action_indices = [max(idx - self.subsample_rate, 0) for idx in action_indices]
            prev_actions_full = self.all_actions[prev_action_indices]
            prev_actions = prev_actions_full[:, self.arm_slice]
            
            # 计算相对动作
            delta_actions = actions - prev_actions
            
            # 首帧相对动作: action_0 - state_0
            for i, idx in enumerate(action_indices):
                if idx in self.episode_boundaries[:-1]:
                    state_sliced = self.all_states[idx][self.arm_slice]
                    delta_actions[i] = actions[i] - state_sliced
            
            # 夹爪维度保持绝对值
            for g_idx in self.gripper_indices:
                delta_actions[:, g_idx] = actions[:, g_idx]
            
            actions = delta_actions
            
        elif self.action_mode == "delta_first":
            # delta_first模式: 整个chunk内各时间步的动作都减去chunk起始时刻的state_0
            # action_t = action_t - state_0，把起点平移到原点
            state_0 = self.all_states[action_indices[0]][self.arm_slice]
            delta_actions = actions - state_0[np.newaxis, :]
            
            # 夹爪维度保持绝对值（不做差值）
            for g_idx in self.gripper_indices:
                delta_actions[:, g_idx] = actions[:, g_idx]
            
            actions = delta_actions
        # else: absolute 模式，actions 保持不变
        
        # 获取当前帧观测
        obs_idx = start_idx
        
        # 处理图像
        rgb_obs = {}
        for out_key in self.camera_keys.keys():
            img = self.all_images[out_key][obs_idx]  # (H, W, C)
            img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)  # (1, C, H, W)
            
            if out_key in self.transforms:
                img = self.transforms[out_key](img)
            
            rgb_obs[out_key] = img
            
        # 获取状态 (根据arm_mode切片)
        robot_obs = torch.from_numpy(self.all_states[obs_idx][self.arm_slice]).float().unsqueeze(0)  # (1, D)
        
        # 获取关节速度 (如果需要)
        joint_vel = None
        if self.return_joint_vel and hasattr(self, 'all_velocities') and self.all_velocities is not None:
            vel_full = self.all_velocities[action_indices]  # (action_seq_len, 14)
            joint_vel = torch.from_numpy(vel_full[:, self.arm_slice]).float()
        
        # 动作
        actions = torch.from_numpy(actions).float()
        
        if self.action_transforms is not None:
            actions = self.action_transforms(actions)
        
        result = {
            "rgb_obs": rgb_obs,
            "robot_obs": robot_obs,
            "actions": actions,
            "lang_text": self.lang_text,
        }
        
        if joint_vel is not None:
            result["joint_vel"] = joint_vel
        
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """自定义collate函数"""
    out: Dict[str, Any] = {"rgb_obs": {}}
    
    # 合并图像
    rgb_keys = batch[0]["rgb_obs"].keys()
    for k in rgb_keys:
        out["rgb_obs"][k] = torch.stack([b["rgb_obs"][k] for b in batch], dim=0)
    
    # 合并状态和动作
    out["robot_obs"] = torch.stack([b["robot_obs"] for b in batch], dim=0)
    out["actions"] = torch.stack([b["actions"] for b in batch], dim=0)
    
    # 合并关节速度 (如果存在)
    if "joint_vel" in batch[0]:
        out["joint_vel"] = torch.stack([b["joint_vel"] for b in batch], dim=0)
    
    # 语言指令
    out["lang_text"] = [b["lang_text"] for b in batch]
    
    return out
