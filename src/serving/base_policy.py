"""Base policy class for all NIAF model deployments.

Handles image preprocessing, action postprocessing (mode conversion,
14-dim expansion), and CSV logging. Subclasses only need to implement
``_model_inference()``.
"""

import abc
import csv
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T
from openpi_client import base_policy as _base_policy
from src.debug import Debugger

from src.serving.utils import (
    CLIP_MEAN,
    CLIP_STD,
    JOINT_INDICES_7D,
    JOINT_INDICES_14D,
    LEFT_RESET_POSE,
    RIGHT_RESET_POSE,
    make_prompt_formatter,
)

logger = logging.getLogger(__name__)


class BaseServePolicy(_base_policy.BasePolicy):
    """Base policy with shared preprocessing / postprocessing for ALOHA.

    Subclasses must implement :meth:`_model_inference`.

    Action mode summary
    -------------------
    * ``absolute``:    Model outputs absolute joint positions directly.
    * ``relative``:    Model outputs per-timestep deltas; use cumsum + state.
    * ``delta_first``: Model outputs deltas relative to the first frame;
                       add state without cumsum.

    In all modes, gripper dimensions (index 6 / 13) are treated as
    absolute values — they are never cumulated or offset by state.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str = "cuda",
        default_prompt: str = "pick up the object",
        image_size: int = 224,
        robot_name: str = "AgileX Cobot Magic",
        num_arms: str = "2",
        action_space: str = "Joint Position",
        prompt_style: str = "minimal",
        action_mode: str = "delta_first",
        arm_mode: str = "dual",
        use_cam_high: bool = True,
        use_cam_left_wrist: bool = True,
        use_cam_right_wrist: bool = True,
        enable_csv_logging: bool = False,
        target_chunk_size: Optional[int] = None,
        return_joint_vel: bool = False,
        execution_hz: float = 30.0,
        debug: bool = False,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.default_prompt = default_prompt
        self.image_size = image_size
        self.action_mode = action_mode
        self.arm_mode = arm_mode
        self.enable_csv_logging = enable_csv_logging
        self.target_chunk_size = target_chunk_size
        self.return_joint_vel = return_joint_vel
        self.execution_hz = float(execution_hz)
        if self.execution_hz <= 0:
            raise ValueError(f"execution_hz must be > 0, got {self.execution_hz}")
        self.debug = debug
        self.debugger = Debugger(enabled=debug)
        self.use_native_target_chunk_sampling = False

        # ---- arm mode → dimension & camera config ----
        if arm_mode == "left":
            self.action_dim = 7
            use_cam_left_wrist, use_cam_right_wrist = True, False
        elif arm_mode == "right":
            self.action_dim = 7
            use_cam_left_wrist, use_cam_right_wrist = False, True
        else:  # dual
            self.action_dim = 14

        # ---- active cameras ----
        self.active_cameras: list[str] = []
        self.use_cam_high = use_cam_high
        self.use_cam_left_wrist = use_cam_left_wrist
        self.use_cam_right_wrist = use_cam_right_wrist
        if use_cam_high:
            self.active_cameras.append("rgb_static")
        if use_cam_left_wrist:
            self.active_cameras.append("rgb_left_wrist")
        if use_cam_right_wrist:
            self.active_cameras.append("rgb_right_wrist")

        # ---- prompt formatter ----
        self.format_instruction = make_prompt_formatter(
            robot_name=robot_name,
            num_arms=num_arms,
            action_space=action_space,
            prompt_style=prompt_style,
        )

        # ---- CLIP normalization ----
        self.clip_mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self.clip_std = torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)

        # ---- chunk size ----
        self.chunk_size = getattr(model, "chunk_size", getattr(model, "act_window_size", 20))

        # ---- CSV logging ----
        self.csv_step = 0
        if self.enable_csv_logging:
            self._init_csv_logger()

        if self.debug:
            logger.info(f"Debug enabled: output={self.debugger.output_dir}")

        logger.info(
            f"{self.__class__.__name__} initialized: device={device}, arm_mode={arm_mode}, "
            f"action_dim={self.action_dim}, chunk_size={self.chunk_size}, "
            f"action_mode={action_mode}, cameras={self.active_cameras}, "
            f"target_chunk_size={target_chunk_size}, csv_logging={enable_csv_logging}, "
            f"return_joint_vel={self.return_joint_vel}, execution_hz={self.execution_hz}, "
            f"debug={debug}"
        )

    def should_return_velocity(self, obs: Dict[str, Any]) -> bool:
        """Return True if velocity output is requested by flag or control mode."""
        if self.return_joint_vel:
            return True
        control_mode = str(obs.get("control_mode", "")).strip().lower()
        return control_mode == "mit"

    # ------------------------------------------------------------------
    #  Abstract method — subclasses implement model-specific inference
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _model_inference(
        self, model_obs: Dict[str, Any], goal: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Run model-specific inference.

        Args:
            model_obs: Preprocessed observation dict with ``rgb_obs`` etc.
            goal: ``{"lang_text": formatted_prompt}``.

        Returns:
            A dict with at least ``"actions"`` (chunk_size, action_dim) as
            **denormalized** values in the model's training action space
            (relative / delta / absolute depending on ``action_mode``).
            Optionally includes ``"velocities"`` (chunk_size, action_dim).
        """

    # ------------------------------------------------------------------
    #  Main entry point (called by WebSocket server)
    # ------------------------------------------------------------------
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess → model inference → postprocess → return actions."""
        do_debug = self.debugger.enabled
        if do_debug:
            self.debugger.begin_frame(
                obs=obs,
                model_name=self.__class__.__name__,
                action_mode=self.action_mode,
                arm_mode=self.arm_mode,
                chunk_size=int(self.chunk_size),
                target_chunk_size=self.target_chunk_size or int(self.chunk_size),
                active_cameras=list(self.active_cameras),
                default_prompt=self.default_prompt,
            )

        # 1. Preprocess
        model_obs, goal = self._preprocess(obs)
        if do_debug:
            self.debugger.record_text(
                raw_prompt=obs.get("prompt", self.default_prompt),
                formatted_prompt=goal.get("lang_text", ""),
            )
            self.debugger.record_preprocessed_images(model_obs)

        # 2. Model-specific inference (subclass)
        result = self._model_inference(model_obs, goal)
        actions_np = result["actions"]  # (chunk_size, action_dim), denormalized
        vel_np = result.get("velocities")  # optional
        if do_debug:
            self.debugger.record_actions(
                stage="raw_model_output",
                actions=actions_np,
                arm_mode=self.arm_mode,
            )
            if vel_np is not None:
                self.debugger.record_velocity(
                    stage="raw_model_output",
                    vel=vel_np,
                    arm_mode=self.arm_mode,
                )

        # 3. Convert to absolute actions according to action_mode
        current_state = self._get_current_state(obs, actions_np.shape[-1])
        actions_np = self._to_absolute_actions(actions_np, current_state)

        # 4. Expand to 14-dim for the real robot
        actions_np = self._expand_to_14d(actions_np)

        # 5. Optional interpolation
        actions_np = self._maybe_interpolate(actions_np)
        if vel_np is not None:
            vel_np = self._maybe_interpolate(vel_np)
        if do_debug:
            self.debugger.record_actions(
                stage="postprocessed_absolute",
                actions=actions_np,
                arm_mode=self.arm_mode,
            )

        # 6. CSV logging
        if self.enable_csv_logging:
            state_for_csv = np.array(
                obs.get("state", np.zeros(14, dtype=np.float32)), dtype=np.float32
            )
            self._log_actions_to_csv(state_for_csv, actions_np)

        # 7. Build response
        output: Dict[str, Any] = {"actions": actions_np.astype(np.float32)}

        if vel_np is not None:
            output["velocities"] = self._expand_vel_to_14d(vel_np).astype(np.float32)

        if do_debug:
            if vel_np is not None:
                post_vel = self._expand_vel_to_14d(vel_np)
                self.debugger.record_velocity(
                    stage="postprocessed",
                    vel=post_vel,
                    arm_mode=self.arm_mode,
                )

        if do_debug:
            self.debugger.finalize_frame(output, arm_mode=self.arm_mode)

        return output

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()

    # ------------------------------------------------------------------
    #  Image preprocessing
    # ------------------------------------------------------------------
    def _process_image(self, img: np.ndarray) -> torch.Tensor:
        """Process a single image: resize → scale → CLIP normalize → add dims.

        Args:
            img: (C, H, W) or (H, W, C) uint8 / float.

        Returns:
            Tensor of shape (1, 1, 3, image_size, image_size) on ``self.device``.
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        # Ensure (C, H, W)
        if img.ndim == 3 and img.shape[0] not in (1, 3):
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 2:
            img = np.repeat(img[None], 3, axis=0)

        img_t = torch.from_numpy(img.copy()).float()

        # Resize (stretch, no padding — matches training)
        if img_t.shape[1] != self.image_size or img_t.shape[2] != self.image_size:
            img_t = T.Resize([self.image_size, self.image_size], antialias=True)(img_t)

        # Scale [0, 255] → [0, 1]
        img_t = img_t / 255.0

        # CLIP normalize
        img_t = img_t.unsqueeze(0)  # (1, C, H, W)
        img_t = (img_t - self.clip_mean) / self.clip_std

        # Add time dim → (1, 1, C, H, W)
        return img_t.unsqueeze(1).to(self.device)

    def _preprocess(self, obs: Dict[str, Any]) -> tuple:
        """Convert client observation to model input format.

        Camera mapping (client → model):
            cam_high        → rgb_static
            cam_left_wrist  → rgb_left_wrist
            cam_right_wrist → rgb_right_wrist
        """
        images = obs.get("images", {})
        cam_map = {
            "rgb_static": ("cam_high", self.use_cam_high),
            "rgb_left_wrist": ("cam_left_wrist", self.use_cam_left_wrist),
            "rgb_right_wrist": ("cam_right_wrist", self.use_cam_right_wrist),
        }

        model_obs: Dict[str, Any] = {"rgb_obs": {}}
        model_obs["control_mode"] = obs.get("control_mode")
        for model_key, (client_key, enabled) in cam_map.items():
            if not enabled:
                continue
            raw = images.get(client_key, np.zeros((3, self.image_size, self.image_size), dtype=np.uint8))
            processed = self._process_image(raw)
            model_obs[model_key] = processed
            model_obs["rgb_obs"][model_key] = processed

        # Language instruction
        raw_prompt = obs.get("prompt", self.default_prompt)
        goal = {"lang_text": self.format_instruction(raw_prompt)}

        return model_obs, goal

    # ------------------------------------------------------------------
    #  Action postprocessing
    # ------------------------------------------------------------------
    def _get_current_state(self, obs: Dict[str, Any], action_dim: int) -> Optional[np.ndarray]:
        """Extract and slice current state from obs based on arm_mode.

        Returns None if action_mode is 'absolute' (state not needed).
        """
        if self.action_mode == "absolute":
            return None

        state = obs.get("state")
        if state is None:
            raise ValueError(
                f"action_mode='{self.action_mode}' requires 'state' in obs."
            )
        state = np.array(state, dtype=np.float32)

        # Match state dim to action_dim
        if action_dim == 7:
            if state.shape[0] == 14:
                state = state[:7] if self.arm_mode != "right" else state[7:]
        elif action_dim == 14:
            if state.shape[0] == 7:
                if self.arm_mode == "right":
                    state = np.concatenate([LEFT_RESET_POSE, state])
                else:
                    state = np.concatenate([state, RIGHT_RESET_POSE])

        return state

    def _to_absolute_actions(
        self, actions: np.ndarray, current_state: Optional[np.ndarray]
    ) -> np.ndarray:
        """Convert model output to absolute joint positions.

        Gripper dimensions are kept as-is (model outputs absolute gripper
        values regardless of action mode).

        Args:
            actions: (T, D) denormalized model output.
            current_state: (D,) current joint state, or None for absolute mode.
        """
        if self.action_mode == "absolute" or current_state is None:
            return actions

        D = actions.shape[-1]
        jidx = JOINT_INDICES_7D if D == 7 else JOINT_INDICES_14D

        if self.action_mode == "relative":
            # Cumulative sum of deltas + initial state
            actions[:, jidx] = np.cumsum(actions[:, jidx], axis=0) + current_state[jidx]
        elif self.action_mode == "delta_first":
            # Each frame is a delta from the initial state (no cumsum)
            actions[:, jidx] = actions[:, jidx] + current_state[jidx]
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        return actions

    def _expand_to_14d(self, actions: np.ndarray) -> np.ndarray:
        """Expand 7-dim single-arm actions to 14-dim dual-arm format."""
        if actions.shape[-1] == 14:
            return actions

        T = actions.shape[0]
        full = np.zeros((T, 14), dtype=np.float32)

        if self.arm_mode == "right":
            full[:, :7] = LEFT_RESET_POSE
            full[:, 7:] = actions
        else:  # left or default
            full[:, :7] = actions
            full[:, 7:] = RIGHT_RESET_POSE

        return full

    def _expand_vel_to_14d(self, vel: np.ndarray) -> np.ndarray:
        """Expand velocity to 14-dim, padding the other arm with zeros."""
        if vel.shape[-1] == 14:
            return vel

        T = vel.shape[0]
        full = np.zeros((T, 14), dtype=np.float32)

        if self.arm_mode == "right":
            full[:, 7:13] = vel[:, :6]  # 6 joint velocities
            # gripper vel (index 13) and left arm stay 0
        elif self.arm_mode == "left":
            full[:, :6] = vel[:, :6]
            # gripper vel (index 6) and right arm stay 0
        else:  # dual
            full[:, :6] = vel[:, :6]
            full[:, 7:13] = vel[:, 7:13] if vel.shape[-1] > 7 else 0

        return full

    def _maybe_interpolate(self, actions: np.ndarray) -> np.ndarray:
        """Linearly interpolate action chunk to target_chunk_size if set."""
        if self.use_native_target_chunk_sampling:
            return actions

        if self.target_chunk_size is None or self.target_chunk_size == actions.shape[0]:
            return actions

        T_src, D = actions.shape
        t_src = np.linspace(0, 1, T_src)
        t_dst = np.linspace(0, 1, self.target_chunk_size)
        out = np.zeros((self.target_chunk_size, D), dtype=np.float32)
        for d in range(D):
            out[:, d] = np.interp(t_dst, t_src, actions[:, d])
        logger.info(f"Interpolated actions: {T_src} → {self.target_chunk_size}")
        return out

    # ------------------------------------------------------------------
    #  CSV logging
    # ------------------------------------------------------------------
    def _init_csv_logger(self):
        csv_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
        os.makedirs(csv_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(csv_dir, f"action_log_{timestamp}.csv")

        with open(self.csv_path, "w", newline="") as f:
            header = ["step", "type", "arm_mode"] + [f"dim_{i}" for i in range(14)]
            csv.writer(f).writerow(header)
        logger.info(f"CSV log created: {self.csv_path}")

    def _log_actions_to_csv(
        self,
        current_state: np.ndarray,
        actions_absolute: np.ndarray,
        joint_vel: Optional[np.ndarray] = None,
    ):
        """Append action data to CSV."""
        def _pad14(arr: np.ndarray) -> list:
            if arr.shape[-1] >= 14:
                return arr[:14].tolist()
            out = np.zeros(14, dtype=np.float32)
            if self.arm_mode == "right":
                out[7 : 7 + arr.shape[-1]] = arr
            else:
                out[: arr.shape[-1]] = arr
            return out.tolist()

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.csv_step, "current_state", self.arm_mode] + _pad14(current_state))
            for t in range(actions_absolute.shape[0]):
                w.writerow([self.csv_step, f"absolute_t{t}", self.arm_mode] + actions_absolute[t].tolist())
            if joint_vel is not None:
                for t in range(joint_vel.shape[0]):
                    w.writerow([self.csv_step, f"vel_t{t}", self.arm_mode] + joint_vel[t].tolist())

        self.csv_step += 1

    # ------------------------------------------------------------------
    #  Metadata helper (used by serve.py)
    # ------------------------------------------------------------------
    def build_metadata(self, model_name: str, model_type: str) -> dict:
        """Build server metadata dict for client handshake."""
        left_reset = LEFT_RESET_POSE.tolist()
        right_reset = RIGHT_RESET_POSE.tolist()
        return {
            "model_name": model_name,
            "model_type": model_type,
            "expects_raw_images": True,
            "action_dim": self.action_dim,
            "chunk_size": self.target_chunk_size or self.chunk_size,
            "reset_pose": left_reset + right_reset,
            "left_reset_pose": left_reset,
            "right_reset_pose": right_reset,
        }
