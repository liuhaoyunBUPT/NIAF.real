#!/usr/bin/env python3
"""Patch serving-critical parameters into a Lightning checkpoint.

This script writes missing fields such as ``action_mode`` and ``arm_mode``
to ``hyper_parameters``, and updates normalization buffers used at serve time.

It updates the following locations:
1) checkpoint["hyper_parameters"]
2) checkpoint["state_dict"]["action_min"/"action_max"]
3) checkpoint["callbacks"]["EMA"]["ema_weights"] (if present)

Example:
    python tools/patch_serving_ckpt_params.py \
      --checkpoint /path/to/model.ckpt \
      --config /path/to/config_aloha_fast.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a dict: {path}")
    return data


def _to_float_list(name: str, value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple, got: {type(value)}")
    return [float(x) for x in value]


def _resolve_stats_from_cfg(cfg: Dict[str, Any], action_mode: str) -> Tuple[List[float], List[float]]:
    mode_min_key = f"action_min_{action_mode}"
    mode_max_key = f"action_max_{action_mode}"

    if mode_min_key in cfg and mode_max_key in cfg:
        return (
            _to_float_list(mode_min_key, cfg[mode_min_key]),
            _to_float_list(mode_max_key, cfg[mode_max_key]),
        )

    if "action_min" in cfg and "action_max" in cfg:
        return (
            _to_float_list("action_min", cfg["action_min"]),
            _to_float_list("action_max", cfg["action_max"]),
        )

    raise ValueError(
        "Cannot resolve normalization stats from config. "
        f"Expected {mode_min_key}/{mode_max_key} or action_min/action_max."
    )


def _update_ema_if_exists(ckpt: Dict[str, Any], new_min: torch.Tensor, new_max: torch.Tensor) -> None:
    callbacks = ckpt.get("callbacks")
    if not isinstance(callbacks, dict):
        return

    ema = callbacks.get("EMA")
    if not isinstance(ema, dict):
        return

    ema_weights = ema.get("ema_weights")
    state_dict = ckpt.get("state_dict")
    if not isinstance(ema_weights, list) or not isinstance(state_dict, dict):
        return

    keys = list(state_dict.keys())
    key_to_idx = {k: i for i, k in enumerate(keys)}

    for key, tensor in (("action_min", new_min), ("action_max", new_max)):
        idx = key_to_idx.get(key)
        if idx is None:
            continue
        if idx >= len(ema_weights):
            continue
        old = ema_weights[idx]
        if isinstance(old, torch.Tensor) and tuple(old.shape) == tuple(tensor.shape):
            ema_weights[idx] = tensor.detach().cpu().clone()


def patch_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path | None,
    action_mode_override: str | None,
    arm_mode_override: str | None,
) -> Path:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = _load_yaml(config_path)

    hp = ckpt.get("hyper_parameters")
    if not isinstance(hp, dict):
        hp = {}
        ckpt["hyper_parameters"] = hp

    action_mode = action_mode_override or cfg.get("action_mode")
    arm_mode = arm_mode_override or cfg.get("arm_mode")

    if action_mode not in {"absolute", "relative", "delta_first"}:
        raise ValueError(
            "action_mode must be one of absolute/relative/delta_first; "
            f"got {action_mode!r}."
        )
    if arm_mode not in {"dual", "left", "right"}:
        raise ValueError("arm_mode must be one of dual/left/right; got {arm_mode!r}.")

    stats_min, stats_max = _resolve_stats_from_cfg(cfg, action_mode)
    if len(stats_min) != len(stats_max):
        raise ValueError("action_min/action_max length mismatch.")

    # Keep mode-specific stats in hyper_parameters
    mode_min_key = f"action_min_{action_mode}"
    mode_max_key = f"action_max_{action_mode}"

    hp["action_mode"] = action_mode
    hp["arm_mode"] = arm_mode
    hp["action_min"] = copy.deepcopy(stats_min)
    hp["action_max"] = copy.deepcopy(stats_max)
    hp[mode_min_key] = copy.deepcopy(stats_min)
    hp[mode_max_key] = copy.deepcopy(stats_max)

    # Backward-compatible legacy switch
    if action_mode == "relative":
        hp["use_relative_action"] = True
    elif "use_relative_action" in hp:
        hp["use_relative_action"] = False

    state_dict = ckpt.get("state_dict")
    if isinstance(state_dict, dict):
        new_min = torch.tensor(stats_min, dtype=torch.float32)
        new_max = torch.tensor(stats_max, dtype=torch.float32)

        if "action_min" in state_dict:
            state_dict["action_min"] = new_min.clone()
        if "action_max" in state_dict:
            state_dict["action_max"] = new_max.clone()
        if "action_min_buf" in state_dict:
            state_dict["action_min_buf"] = new_min.clone()
        if "action_max_buf" in state_dict:
            state_dict["action_max_buf"] = new_max.clone()

        _update_ema_if_exists(ckpt, new_min, new_max)

    if output_path is None:
        output_path = checkpoint_path.with_name(checkpoint_path.stem + ".serve_patched.ckpt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch serving parameters into checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Input .ckpt path")
    parser.add_argument("--config", required=True, help="Training config YAML path")
    parser.add_argument("--output", default=None, help="Output .ckpt path")
    parser.add_argument(
        "--action-mode",
        default=None,
        choices=["absolute", "relative", "delta_first"],
        help="Override action_mode from config",
    )
    parser.add_argument(
        "--arm-mode",
        default=None,
        choices=["dual", "left", "right"],
        help="Override arm_mode from config",
    )
    args = parser.parse_args()

    output = patch_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        output_path=Path(args.output) if args.output else None,
        action_mode_override=args.action_mode,
        arm_mode_override=args.arm_mode,
    )
    print(f"Patched checkpoint saved to: {output}")


if __name__ == "__main__":
    main()
