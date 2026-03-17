#!/usr/bin/env python3
"""Send one simulated observation to a running NIAF websocket server.

Usage examples:
    # 1) Send uint8 image data in [0,255]
    python tools/simulate_server_request.py --host 127.0.0.1 --port 8000 \
        --prompt "pick up the object" --image-range uint8 --image-mode gradient

    # 2) Send float image data in [0,1]
    python tools/simulate_server_request.py --host 127.0.0.1 --port 8000 \
        --image-range float01 --image-mode random
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np

from openpi_client.websocket_client_policy import WebsocketClientPolicy


def _make_image(height: int, width: int, image_mode: str, image_range: str) -> np.ndarray:
    if image_mode == "zeros":
        arr = np.zeros((3, height, width), dtype=np.float32)
    elif image_mode == "gradient":
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        c0 = xx
        c1 = yy
        c2 = 0.5 * (xx + yy)
        arr = np.stack([c0, c1, c2], axis=0).astype(np.float32)
    else:  # random
        arr = np.random.rand(3, height, width).astype(np.float32)

    if image_range == "uint8":
        return (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr.astype(np.float32)


def _print_image_stats(name: str, img: np.ndarray) -> None:
    print(
        f"[REQ][{name}] shape={tuple(img.shape)} dtype={img.dtype} "
        f"min={float(img.min()):.4f} max={float(img.max()):.4f} mean={float(img.mean()):.4f}"
    )


def build_obs(args: argparse.Namespace, state_dim: int) -> Dict:
    obs: Dict = {
        "prompt": args.prompt,
        "state": np.zeros((state_dim,), dtype=np.float32),
        "images": {},
    }

    if args.state_mode == "random":
        obs["state"] = np.random.uniform(-0.05, 0.05, size=(state_dim,)).astype(np.float32)

    cam_flags = {
        "cam_high": args.use_cam_high,
        "cam_left_wrist": args.use_cam_left_wrist,
        "cam_right_wrist": args.use_cam_right_wrist,
    }

    for cam_name, enabled in cam_flags.items():
        if not enabled:
            continue
        img = _make_image(args.height, args.width, args.image_mode, args.image_range)
        obs["images"][cam_name] = img
        _print_image_stats(cam_name, img)

    print(
        f"[REQ] prompt={obs['prompt']!r} state_shape={obs['state'].shape} "
        f"state_min={float(obs['state'].min()):.4f} state_max={float(obs['state'].max()):.4f}"
    )
    return obs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send one simulated request to NIAF server")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prompt", type=str, default="pick up the object")

    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--image-mode", choices=["random", "gradient", "zeros"], default="gradient")
    p.add_argument("--image-range", choices=["uint8", "float01"], default="uint8")

    p.add_argument("--state-mode", choices=["zeros", "random"], default="zeros")

    p.add_argument("--use-cam-high", action="store_true", default=True)
    p.add_argument("--use-cam-left-wrist", action="store_true", default=True)
    p.add_argument("--use-cam-right-wrist", action="store_true", default=True)
    p.add_argument("--no-cam-high", action="store_true")
    p.add_argument("--no-cam-left-wrist", action="store_true")
    p.add_argument("--no-cam-right-wrist", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.use_cam_high = args.use_cam_high and not args.no_cam_high
    args.use_cam_left_wrist = args.use_cam_left_wrist and not args.no_cam_left_wrist
    args.use_cam_right_wrist = args.use_cam_right_wrist and not args.no_cam_right_wrist

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    meta = policy.get_server_metadata()
    print(f"[META] {meta}")

    state_dim = len(meta.get("reset_pose", [])) or 14
    obs = build_obs(args, state_dim=state_dim)

    t0 = time.time()
    result = policy.infer(obs)
    dt_ms = (time.time() - t0) * 1000.0

    actions = np.asarray(result.get("actions"))
    print(
        f"[RESP] infer_time_ms={dt_ms:.2f} keys={list(result.keys())} "
        f"actions_shape={actions.shape} actions_min={float(actions.min()):.4f} "
        f"actions_max={float(actions.max()):.4f}"
    )

    if "velocities" in result:
        vel = np.asarray(result["velocities"])
        print(
            f"[RESP] velocities_shape={vel.shape} velocities_min={float(vel.min()):.4f} "
            f"velocities_max={float(vel.max()):.4f}"
        )

    print("[DONE] Sent 1 request. If server was started with --debug, check niaf/debug/index.html")


if __name__ == "__main__":
    main()
