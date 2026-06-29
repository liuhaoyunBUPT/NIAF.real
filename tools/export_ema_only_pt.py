#!/usr/bin/env python3
"""Export EMA weights from a Lightning checkpoint into an EMA-only .pt bundle.

The output .pt is intentionally checkpoint-like so that serving can start with
"--ema-only-pt" without requiring the original .ckpt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def _build_ema_state_dict(ckpt: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], int, int]:
    callbacks = ckpt.get("callbacks", {})
    if not isinstance(callbacks, dict):
        raise ValueError("checkpoint callbacks is missing or invalid")

    ema_state = callbacks.get("EMA", {})
    if not isinstance(ema_state, dict):
        raise ValueError("checkpoint callbacks.EMA is missing")

    ema_weights = ema_state.get("ema_weights")
    if not isinstance(ema_weights, list):
        raise ValueError("checkpoint callbacks.EMA.ema_weights is missing")

    state_dict = ckpt.get("state_dict", {})
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint state_dict is missing or empty")

    ema_state_dict: Dict[str, torch.Tensor] = {}
    total = len(state_dict)

    for key, ema_w in zip(state_dict.keys(), ema_weights):
        ref = state_dict[key]
        if isinstance(ema_w, torch.Tensor) and isinstance(ref, torch.Tensor) and ema_w.shape == ref.shape:
            ema_state_dict[key] = ema_w.detach().cpu().clone()

    matched = len(ema_state_dict)
    if matched == 0:
        raise ValueError("no EMA weights matched checkpoint state_dict keys")

    return ema_state_dict, matched, total


def export_ema_only_pt(checkpoint_path: Path, output_path: Path) -> Path:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint payload must be a dict")

    ema_state_dict, matched, total = _build_ema_state_dict(ckpt)

    payload: Dict[str, Any] = {
        "state_dict": ema_state_dict,
        "hyper_parameters": ckpt.get("hyper_parameters", {}),
        "meta": {
            "source_checkpoint": str(checkpoint_path),
            "matched_ema_tensors": matched,
            "total_state_dict_tensors": total,
            "export_type": "ema_only_bundle",
        },
    }

    # Keep a few common Lightning metadata fields for compatibility/debugging.
    for key in ("epoch", "global_step", "pytorch-lightning_version"):
        if key in ckpt:
            payload[key] = ckpt[key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EMA-only .pt bundle from a .ckpt")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint (.ckpt)")
    parser.add_argument("--output", default=None, help="Output .pt path")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.with_name(f"ema_only_{checkpoint_path.stem}.pt")

    out = export_ema_only_pt(checkpoint_path, output_path)
    print(f"EMA-only bundle saved to: {out}")


if __name__ == "__main__":
    main()
