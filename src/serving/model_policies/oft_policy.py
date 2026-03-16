"""OFT model policy — direct regression via MLP head."""

import logging
from typing import Any, Dict

import numpy as np
import torch

from src.serving.base_policy import BaseServePolicy

logger = logging.getLogger(__name__)


class OftPolicy(BaseServePolicy):
    """OFT inference: empty query → VLM decoder → MLP → denormalize."""

    def _model_inference(
        self, model_obs: Dict[str, Any], goal: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            actions = self.model.forward(model_obs, goal)  # (1, T, D) denormalized
        return {"actions": actions.cpu().numpy().squeeze(0)}
