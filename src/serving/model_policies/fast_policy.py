"""FAST model policy — autoregressive token generation + FAST tokenizer decode."""

import logging
from typing import Any, Dict

import numpy as np
import torch

from src.serving.base_policy import BaseServePolicy
from src.serving.utils import JOINT_INDICES_7D, JOINT_INDICES_14D

logger = logging.getLogger(__name__)


class FastPolicy(BaseServePolicy):
    """FAST inference: autoregressive token generation → FAST decode → denormalize."""

    def _model_inference(
        self, model_obs: Dict[str, Any], goal: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            actions = self.model.forward(model_obs, goal)  # (1, T, D) denormalized

        actions_np = actions.detach().cpu().numpy().squeeze(0).astype(np.float32)
        result: Dict[str, np.ndarray] = {"actions": actions_np}

        if self.should_return_velocity(model_obs):
            vel = np.zeros_like(actions_np, dtype=np.float32)
            if actions_np.shape[0] >= 2:
                diff = np.diff(actions_np, axis=0) * float(self.execution_hz)
                vel[:-1] = diff
                vel[-1] = diff[-1]

            # Keep gripper velocity at zero; only joint dimensions use finite difference.
            action_dim = actions_np.shape[-1]
            if action_dim == 7:
                joint_idx = JOINT_INDICES_7D
            elif action_dim == 14:
                joint_idx = JOINT_INDICES_14D
            else:
                joint_idx = list(range(action_dim))

            non_joint_idx = [i for i in range(action_dim) if i not in joint_idx]
            if non_joint_idx:
                vel[:, non_joint_idx] = 0.0

            result["velocities"] = vel

        return result
