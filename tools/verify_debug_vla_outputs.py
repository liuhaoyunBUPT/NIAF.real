#!/usr/bin/env python3
"""Verify unified VLA debug outputs exist and are readable.

Usage:
    python tools/verify_debug_vla_outputs.py
"""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_KEYS = [
    "step",
    "timestamp",
    "model_name",
    "action_mode",
    "arm_mode",
    "chunk_size",
    "active_cameras",
    "cameras",
    "left_arm_plot_path",
    "right_arm_plot_path",
]


def _check_exists(base: Path, rel_path: str, label: str) -> None:
    if not rel_path:
        print(f"[WARN] {label}: empty path")
        return
    p = base / rel_path
    if p.exists():
        print(f"[OK] {label}: {p}")
    else:
        print(f"[FAIL] {label}: missing {p}")


def main() -> None:
    debug_dir = (Path(__file__).resolve().parents[1] / "debug").resolve()
    latest_json = debug_dir / "latest.json"
    if not latest_json.exists():
        raise SystemExit(f"latest.json not found: {latest_json}")

    payload = json.loads(latest_json.read_text(encoding="utf-8"))

    print(f"[INFO] debug_dir: {debug_dir}")
    print(f"[INFO] latest step: {payload.get('step')}")

    for key in REQUIRED_KEYS:
        if key in payload:
            print(f"[OK] key: {key}")
        else:
            print(f"[FAIL] key missing: {key}")

    _check_exists(debug_dir, payload.get("left_arm_plot_path", ""), "left_arm_plot")
    _check_exists(debug_dir, payload.get("right_arm_plot_path", ""), "right_arm_plot")

    cameras = payload.get("cameras", [])
    if not cameras:
        print("[WARN] cameras: empty")
    for cam in cameras:
        cam_name = cam.get("camera_name", "unknown")
        _check_exists(debug_dir, cam.get("raw_image_path", ""), f"{cam_name}.raw")
        _check_exists(debug_dir, cam.get("preprocessed_image_path", ""), f"{cam_name}.preprocessed")


if __name__ == "__main__":
    main()
