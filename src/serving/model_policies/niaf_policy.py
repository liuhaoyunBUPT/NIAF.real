"""NIAF model policy — SIREN implicit function inference."""

import logging
from typing import Any, Dict

import numpy as np
import torch

from src.serving.base_policy import BaseServePolicy

logger = logging.getLogger(__name__)


class NiafPolicy(BaseServePolicy):
    """NIAF inference: VLM → SIREN modulation → time sampling → denormalize."""

    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.use_native_target_chunk_sampling = True

    def _model_inference(
        self, model_obs: Dict[str, Any], goal: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        original_chunk_size = getattr(self.model, "chunk_size", None)
        try:
            if self.target_chunk_size is not None and original_chunk_size is not None:
                self.model.chunk_size = int(self.target_chunk_size)
            with torch.no_grad():
                actions = self.model.forward(model_obs, goal)  # (1, T, D) denormalized
        finally:
            if original_chunk_size is not None:
                self.model.chunk_size = original_chunk_size

        return {"actions": actions.cpu().numpy().squeeze(0)}
