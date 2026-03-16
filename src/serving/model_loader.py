"""Unified model loader for all NIAF model variants.

Handles checkpoint loading, torch.load monkey-patching for PyTorch 2.6+,
and optional EMA weight loading.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Type

import torch

logger = logging.getLogger(__name__)

# Registry: model_type → (module_path, class_name)
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "beast": ("src.models.beast", "BEAST"),
    "fast": ("src.models.fast", "FAST"),
    "oft": ("src.models.oft", "OFT"),
    "niaf": ("src.models.niaf", "NIAF"),
    "niaf_vel": ("src.models.niaf_vel", "NIAFVel"),
}


def _get_model_class(model_type: str) -> Type:
    """Dynamically import and return the model class."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    module_path, class_name = MODEL_REGISTRY[model_type]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _load_ema_weights(
    checkpoint_path: str, model: torch.nn.Module
) -> Tuple[int, int]:
    """Load EMA weights from a Lightning checkpoint.

    EMA weights are stored as a flat list in
    ``checkpoint["callbacks"]["EMA"]["ema_weights"]``, ordered to match
    ``checkpoint["state_dict"].keys()``.

    Returns:
        (matched_count, total_count)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    callbacks = ckpt.get("callbacks", {})
    ema_state = callbacks.get("EMA", {})
    ema_list = ema_state.get("ema_weights")

    if ema_list is None:
        logger.info("No EMA weights found in checkpoint — using standard weights.")
        return 0, 0

    state_dict_keys = list(ckpt.get("state_dict", {}).keys())
    model_sd = model.state_dict()
    total = len(state_dict_keys)

    ema_dict = {}
    for name, ema_w in zip(state_dict_keys, ema_list):
        if name in model_sd and ema_w.shape == model_sd[name].shape:
            ema_dict[name] = ema_w

    matched = len(ema_dict)
    if ema_dict:
        model.load_state_dict(ema_dict, strict=False)
        logger.info(f"Loaded {matched}/{total} EMA weights from checkpoint.")
    else:
        logger.warning("EMA weights present but none matched current model.")

    return matched, total


def _extract_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Extract hyper_parameters and other metadata from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    info: Dict[str, Any] = {}

    # Common fields
    for key in (
        "action_mode", "arm_mode", "fps",
        "delta_first_action_stats",
        "act_loss_weight", "vel_loss_weight", "jerk_loss_weight",
    ):
        if key in hp:
            info[key] = hp[key]

    # rgb_obs_keys
    if "rgb_obs_keys" in hp:
        info["rgb_obs_keys"] = hp["rgb_obs_keys"]

    return info


def load_model(
    model_type: str,
    checkpoint_path: str,
    device: str = "cuda",
    vlm_path: str = "/data1/lhy/models/Florence-2/large",
    load_ema: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a NIAF model from checkpoint.

    Args:
        model_type: One of ``beast``, ``fast``, ``oft``, ``niaf``, ``niaf_vel``.
        checkpoint_path: Path to the ``.ckpt`` file.
        device: Target device.
        vlm_path: Path to the Florence-2 backbone weights.
        load_ema: Whether to load EMA weights (if available).

    Returns:
        (model, info_dict) where info_dict contains checkpoint metadata.
    """
    model_cls = _get_model_class(model_type)
    logger.info(f"Loading {model_cls.__name__} from: {checkpoint_path}")

    # Monkey-patch torch.load for PyTorch 2.6+ compatibility
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        model = model_cls.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            strict=False,
            vlm_path=vlm_path,
        )
    finally:
        torch.load = _orig_load

    # EMA weights
    if load_ema:
        _load_ema_weights(checkpoint_path, model)

    model.eval()

    # Extract metadata
    info = _extract_checkpoint_info(checkpoint_path)
    logger.info(f"Model loaded. Checkpoint info: {info}")

    return model, info
