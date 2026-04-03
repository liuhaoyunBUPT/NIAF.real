"""Unified model loader for all NIAF model variants.

Handles checkpoint loading, torch.load monkey-patching for PyTorch 2.6+,
and optional EMA weight loading.
"""

import logging
from copy import deepcopy
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


def _to_plain_list(value: Any) -> Optional[list]:
    """Convert list-like values (ListConfig / tuple / tensor) to a plain Python list."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    try:
        return list(value)
    except TypeError:
        return None


def _resolve_action_stats_for_init(checkpoint_path: str) -> Dict[str, Any]:
    """Resolve action stats kwargs for model initialization from checkpoint.

    Older checkpoints may not contain direct ``action_min`` / ``action_max`` in
    ``hyper_parameters``. This helper falls back to mode-specific keys
    (e.g. ``action_min_relative``) and then to saved buffers in ``state_dict``.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    state_dict = ckpt.get("state_dict", {})

    action_min = _to_plain_list(hp.get("action_min"))
    action_max = _to_plain_list(hp.get("action_max"))
    action_mode = hp.get("action_mode")
    source = "hyper_parameters.action_min/action_max"

    if action_min is None or action_max is None:
        # Backward-compatible mode inference: explicit action_mode first,
        # then legacy use_relative_action, then available stats keys.
        if action_mode not in {"absolute", "relative", "delta_first"}:
            if hp.get("use_relative_action", None) is True:
                action_mode = "relative"
            elif "action_min_delta_first" in hp and "action_max_delta_first" in hp:
                action_mode = "delta_first"
            elif "action_min_relative" in hp and "action_max_relative" in hp:
                action_mode = "relative"
            elif "action_min_absolute" in hp and "action_max_absolute" in hp:
                action_mode = "absolute"

        if action_mode in {"absolute", "relative", "delta_first"}:
            mode_min = _to_plain_list(hp.get(f"action_min_{action_mode}"))
            mode_max = _to_plain_list(hp.get(f"action_max_{action_mode}"))
            if mode_min is not None and mode_max is not None:
                action_min, action_max = mode_min, mode_max
                source = f"hyper_parameters.action_min_{action_mode}/action_max_{action_mode}"

    if action_min is None or action_max is None:
        # Final fallback: saved buffers from state_dict
        buf_min = state_dict.get("action_min")
        buf_max = state_dict.get("action_max")
        if isinstance(buf_min, torch.Tensor) and isinstance(buf_max, torch.Tensor):
            action_min = buf_min.detach().cpu().tolist()
            action_max = buf_max.detach().cpu().tolist()
            source = "state_dict.action_min/action_max"

    out: Dict[str, Any] = {
        "action_min": action_min,
        "action_max": action_max,
        "action_mode": action_mode,
        "source": source,
    }
    return out


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
        "use_relative_action",
        "delta_first_action_stats",
        "act_loss_weight", "vel_loss_weight", "jerk_loss_weight",
    ):
        if key in hp:
            info[key] = hp[key]

    # rgb_obs_keys
    if "rgb_obs_keys" in hp:
        info["rgb_obs_keys"] = hp["rgb_obs_keys"]

    return info


def _resolve_beast_mp_tokenizer_for_init(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Resolve and normalize BEAST mp_tokenizer config from checkpoint.

    Some legacy checkpoints store ``mp_tokenizer._target_`` under external
    package paths (e.g. beast_calvin). Serving should always instantiate the
    local NIAF tokenizer implementation to ensure API compatibility.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    mp_tokenizer = hp.get("mp_tokenizer")

    if mp_tokenizer is None:
        return None

    mp_cfg = deepcopy(mp_tokenizer)
    target = None
    if isinstance(mp_cfg, dict):
        target = mp_cfg.get("_target_")
    else:
        target = getattr(mp_cfg, "_target_", None)

    if not isinstance(target, str):
        return mp_cfg

    if target.endswith("bspline_tokenizer.BSpline_Tokenizer") and target != "src.models.tokenizers.bspline_tokenizer.BSpline_Tokenizer":
        if isinstance(mp_cfg, dict):
            mp_cfg["_target_"] = "src.models.tokenizers.bspline_tokenizer.BSpline_Tokenizer"
        else:
            setattr(mp_cfg, "_target_", "src.models.tokenizers.bspline_tokenizer.BSpline_Tokenizer")
        logger.info(
            "Rewrote legacy mp_tokenizer target: %s -> %s",
            target,
            "src.models.tokenizers.bspline_tokenizer.BSpline_Tokenizer",
        )

    return mp_cfg


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

    # Resolve optional action stats kwargs for backward compatibility
    resolved_stats = _resolve_action_stats_for_init(checkpoint_path)
    init_kwargs: Dict[str, Any] = {"vlm_path": vlm_path}

    if model_type == "beast":
        mp_tokenizer_cfg = _resolve_beast_mp_tokenizer_for_init(checkpoint_path)
        if mp_tokenizer_cfg is not None:
            init_kwargs["mp_tokenizer"] = mp_tokenizer_cfg

    if (
        resolved_stats.get("action_min") is not None
        and resolved_stats.get("action_max") is not None
    ):
        init_kwargs["action_min"] = resolved_stats["action_min"]
        init_kwargs["action_max"] = resolved_stats["action_max"]
        logger.info(
            "Init action stats source: %s (mode=%s, dims=%d)",
            resolved_stats.get("source"),
            resolved_stats.get("action_mode"),
            len(resolved_stats["action_min"]),
        )
    else:
        logger.info("Init action stats not resolved from checkpoint; using checkpoint defaults.")

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
            **init_kwargs,
        )
    finally:
        torch.load = _orig_load

    # EMA weights
    if load_ema:
        _load_ema_weights(checkpoint_path, model)

    model.eval()

    # Extract metadata
    info = _extract_checkpoint_info(checkpoint_path)

    # Backfill action_mode when old checkpoints only contain legacy fields
    if "action_mode" not in info and resolved_stats.get("action_mode") in {
        "absolute",
        "relative",
        "delta_first",
    }:
        info["action_mode"] = resolved_stats["action_mode"]
        logger.info(
            "Checkpoint action_mode missing; inferred action_mode=%s from %s",
            resolved_stats["action_mode"],
            resolved_stats.get("source"),
        )
    
    # 将真正使用的归一化数值注册到 buf 张量上
    action_mode = info.get("action_mode", getattr(model, "action_mode", None))
    if action_mode == "delta_first" and "delta_first_action_stats" in info:
        stats = info["delta_first_action_stats"]
        arm_mode = getattr(model, "arm_mode", info.get("arm_mode", "dual"))
        if arm_mode == "left":
            stat_min = list(stats.get("min", []))[:7]
            stat_max = list(stats.get("max", []))[:7]
        elif arm_mode == "right":
            stat_min = list(stats.get("min", []))[7:14]
            stat_max = list(stats.get("max", []))[7:14]
        else:
            stat_min = list(stats.get("min", []))
            stat_max = list(stats.get("max", []))
            
        if hasattr(model, "action_min_buf") and hasattr(model, "action_max_buf"):
            model.action_min_buf.data = torch.tensor(stat_min, dtype=torch.float32, device=model.action_min_buf.device)
            model.action_max_buf.data = torch.tensor(stat_max, dtype=torch.float32, device=model.action_max_buf.device)

    logger.info(f"Model loaded. Checkpoint info: {info}")

    return model, info
