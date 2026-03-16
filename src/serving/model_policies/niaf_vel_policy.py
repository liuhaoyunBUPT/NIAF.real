"""NIAFVel model policy — SIREN implicit function with velocity output.

This is the only model that can output joint velocities (rad/s) alongside
positions. It uses ``torch.enable_grad()`` to compute analytic derivatives
of the SIREN network w.r.t. time coordinates.

The action_mode is forced to ``delta_first`` (the only mode supported by
NIAFVel during training).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.serving.base_policy import BaseServePolicy

logger = logging.getLogger(__name__)


class NiafVelPolicy(BaseServePolicy):
    """NIAFVel inference: SIREN + analytic velocity via autograd."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        return_joint_vel: bool = False,
        **kwargs,
    ):
        # Force delta_first — the only mode NIAFVel supports
        kwargs["action_mode"] = "delta_first"
        super().__init__(model, **kwargs)
        self.return_joint_vel = return_joint_vel

    def _model_inference(
        self, model_obs: Dict[str, Any], goal: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        # Build batch in the format the model expects
        lang_text = goal.get("lang_text", "")
        batch: Dict[str, Any] = {"lang_text": [lang_text], "rgb_obs": {}}
        for cam_key in self.model.rgb_obs_keys:
            if cam_key in model_obs:
                batch["rgb_obs"][cam_key] = model_obs[cam_key]
            elif "rgb_obs" in model_obs and cam_key in model_obs["rgb_obs"]:
                batch["rgb_obs"][cam_key] = model_obs["rgb_obs"][cam_key]

        result: Dict[str, np.ndarray] = {}

        if self.return_joint_vel:
            # Gradient required for analytic velocity (d_action / d_tau)
            with torch.enable_grad():
                actions_norm, vel_pred = self.model.predict_actions_and_joint_vel(batch)
            actions = self.model.denormalize_actions(actions_norm)
            result["velocities"] = vel_pred.detach().cpu().numpy().squeeze(0)
        else:
            with torch.no_grad():
                actions = self.model.forward(model_obs, goal)  # already denormalized

        result["actions"] = actions.detach().cpu().numpy().squeeze(0)
        return result
