"""BEAST model policy — Motion Primitive tokenizer inference."""

import logging
from typing import Any, Dict

import numpy as np
import torch

from src.serving.base_policy import BaseServePolicy

logger = logging.getLogger(__name__)


class BeastPolicy(BaseServePolicy):
    """BEAST inference: forward → MP token decode → denormalized actions."""

    def _model_inference(
        self, model_obs: Dict[str, Any], goal: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            actions = self.model.forward(model_obs, goal)  # (1, T, D) denormalized
        return {"actions": actions.cpu().numpy().squeeze(0)}
