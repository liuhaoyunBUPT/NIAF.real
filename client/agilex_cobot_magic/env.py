from typing import List, Optional  # noqa: UP035
import os
import logging
from datetime import datetime

import einops
import numpy as np
from PIL import Image
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

try:
    from agilex_cobot_magic import real_env as _real_env
except ModuleNotFoundError as exc:
    if exc.name != "agilex_cobot_magic":
        raise
    import real_env as _real_env

# =============================================================================
# 图像调试配置
# =============================================================================
DEBUG_SAVE_IMAGES = False  # 设为 True 开启图像保存验证
DEBUG_IMAGE_DIR = "/tmp/agilex_cobot_magic_debug_images/client"  # 客户端图像保存目录
DEBUG_SAVE_INTERVAL = 10  # 每隔多少步保存一次图像
_debug_step_counter = 0


def _save_debug_image(img: np.ndarray, name: str, step: int, stage: str) -> None:
    """保存调试图像并打印统计信息"""
    os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    
    # 打印统计信息
    print(f"[Client][{stage}] {name}: shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min()}, max={img.max()}, mean={img.mean():.2f}")
    
    # 转换为 (H, W, C) 格式保存
    if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W) 格式
        img_hwc = np.transpose(img, (1, 2, 0))
    else:
        img_hwc = img
    
    # 确保是 uint8
    if img_hwc.dtype != np.uint8:
        if img_hwc.max() <= 1.0:
            img_hwc = (img_hwc * 255).astype(np.uint8)
        else:
            img_hwc = img_hwc.astype(np.uint8)
    
    # 保存图像
    save_path = os.path.join(DEBUG_IMAGE_DIR, f"step{step:06d}_{timestamp}_{stage}_{name}.png")
    Image.fromarray(img_hwc).save(save_path)
    print(f"[Client] Saved: {save_path}")


class AgileXCobotMagicEnvironment(_environment.Environment):
    """
    An environment for AgileX Cobot Magic real hardware.
    
    支持两种控制模式:
    1. position: 通过ROS话题的位置控制 (默认)
    2. mit: 直接SDK的MIT阻抗控制 (支持位置+速度)
    """

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        model_type: str = "pi",  # "siren" 或 "pi", 用于区分图像处理分支
        control_mode: str = "position",  # "position" 或 "mit"
        mit_config: Optional[dict] = None,  # MIT控制配置
        record_config: Optional[dict] = None,
    ) -> None:
        """
        初始化 AgileX Cobot Magic 真实环境
        
        Args:
            reset_position: 复位位置
            render_height: 图像渲染高度
            render_width: 图像渲染宽度
            model_type: 模型类型 ("pi" 或 "siren")
            control_mode: 控制模式 ("position" 或 "mit")
            mit_config: MIT控制配置, 包含:
                - can_port_left: 左臂CAN端口 (默认 "can0")
                - can_port_right: 右臂CAN端口 (默认 "can2")
                - kp: 位置增益 (默认 30.0)
                - kd: 速度增益 (默认 1.0)
        """
        self._env = _real_env.make_real_env(
            init_node=True, 
            reset_position=reset_position,
            control_mode=control_mode,
            mit_config=mit_config,
            record_config=record_config,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._model_type = model_type
        self._control_mode = control_mode
        
        if model_type == "siren":
            logging.info("[Client] Model type: siren - 发送原始图像，服务器端处理")
        else:
            logging.info(f"[Client] Model type: {model_type} - 客户端 resize_with_pad 处理")
        
        logging.info(f"[Client] Control mode: {control_mode}")
        if control_mode == "mit":
            logging.info(f"[Client] MIT config: {mit_config}")

        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        global _debug_step_counter
        
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        for cam_name in obs["images"]:
            # ========== 调试: 保存原始图像 (来自ROS) ==========
            if DEBUG_SAVE_IMAGES and _debug_step_counter % DEBUG_SAVE_INTERVAL == 0:
                raw_img = obs["images"][cam_name]
                if raw_img is not None:
                    _save_debug_image(raw_img, cam_name, _debug_step_counter, "1_raw_from_ros")
            # ================================================
            
            if self._model_type == "siren":
                # ========== Siren 模型分支: 发送原始 uint8 图像 ==========
                # 服务器端会使用 torchvision.transforms.Resize 处理 (与训练一致)
                # 只做格式转换: (H, W, C) -> (C, H, W)
                img = image_tools.convert_to_uint8(obs["images"][cam_name])
                obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
                
                if DEBUG_SAVE_IMAGES and _debug_step_counter % DEBUG_SAVE_INTERVAL == 0:
                    _save_debug_image(obs["images"][cam_name], cam_name, _debug_step_counter, "2_siren_raw_chw_to_send")
            else:
                # ========== Pi 模型分支: 保持原有处理逻辑 ==========
                # 使用 resize_with_pad (PIL, 保持比例, 填充黑边)
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
                )
                
                # ========== 调试: 保存 resize 后的图像 (H,W,C) ==========
                if DEBUG_SAVE_IMAGES and _debug_step_counter % DEBUG_SAVE_INTERVAL == 0:
                    _save_debug_image(img, cam_name, _debug_step_counter, "2_after_resize_hwc")
                # ====================================================
                
                obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
                
                # ========== 调试: 保存 rearrange 后的图像 (C,H,W) - 即将发送 ==========
                if DEBUG_SAVE_IMAGES and _debug_step_counter % DEBUG_SAVE_INTERVAL == 0:
                    _save_debug_image(obs["images"][cam_name], cam_name, _debug_step_counter, "3_final_chw_to_send")
        
        _debug_step_counter += 1

        return {
            "state": obs["qpos"],
            "images": obs["images"],
        }

    @override
    def apply_action(self, action: dict) -> None:
        """
        应用动作到机器人
        
        Args:
            action: 动作字典，包含:
                - actions: 14维位置向量 [left(7), right(7)]
                - velocities: (可选) 14维速度向量，仅在MIT模式下使用
        """
        positions = action["actions"]
        velocities = action.get("velocities", None)
        chunk_id = int(action.get("_chunk_id", -1))
        in_overlap = bool(action.get("_in_blend", False))
        chunk_meta = {
            "chunk_step": int(action.get("_chunk_step", -1)),
            "chunk_actions_full": action.get("_chunk_actions_full"),
            "chunk_velocities_full": action.get("_chunk_velocities_full"),
            "chunk_skip_steps": int(action.get("_chunk_skip_steps", 0)),
            "chunk_blend_window": int(action.get("_chunk_blend_window", 0)),
        }
        
        # ========== 临时调试: 验证 velocities 传递 ==========
        # print(f"[DEBUG apply_action] action keys: {action.keys()}")
        # print(f"[DEBUG apply_action] positions shape: {positions.shape if hasattr(positions, 'shape') else len(positions)}")
        # if velocities is not None:
        #     import numpy as np
        #     vel_arr = np.array(velocities)
        #     print(f"[DEBUG apply_action] velocities received: shape={vel_arr.shape}, "
        #           f"range=[{vel_arr.min():.4f}, {vel_arr.max():.4f}], mean={vel_arr.mean():.4f}")
        # else:
        #     print(f"[DEBUG apply_action] velocities is None")
        # ====================================================
        
        # 在MIT模式下使用速度信息
        if self._control_mode == "mit" and velocities is not None:
            self._ts = self._env.step(
                positions,
                velocities=velocities,
                chunk_id=chunk_id,
                in_overlap=in_overlap,
                chunk_meta=chunk_meta,
            )
        else:
            self._ts = self._env.step(
                positions,
                chunk_id=chunk_id,
                in_overlap=in_overlap,
                chunk_meta=chunk_meta,
            )
    
    def close(self) -> None:
        """关闭环境"""
        if hasattr(self._env, 'close'):
            self._env.close()

