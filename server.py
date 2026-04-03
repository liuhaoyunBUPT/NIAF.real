"""Unified model serving entry point.

Required parameters
-------------------
- --model: one of [niaf, niaf_vel, beast, fast, oft]
- --checkpoint: path to .ckpt

Common optional parameters
--------------------------
- --port / --host: websocket server address (default: 8000 / 0.0.0.0)
- --device: cuda or cpu
- --action-mode / --arm-mode: override checkpoint settings
- --target-chunk-size: interpolate output chunk length
- --return-joint-vel: only for niaf_vel

Debug mode
----------
- Add --debug when launching server.
- Debug artifacts are written to niaf/debug/ (fixed path):
  - dashboard page: niaf/debug/index.html
  - latest payload: niaf/debug/latest.json
  - figures: niaf/debug/assets/

Quick examples
--------------
    # NIAF
    python server.py --model niaf \\
        --checkpoint /path/to/niaf.ckpt --port 8000

    # NIAFVel + velocity output
    python server.py --model niaf_vel \\
        --checkpoint /path/to/niaf_vel.ckpt --return-joint-vel --port 8000

    # Any model with debug dashboard
    python server.py --model niaf \\
        --checkpoint /path/to/niaf_vel.ckpt --debug

    # 若你在远程 SSH，在 niaf 根目录执行：
    python -m http.server 18080，然后浏览器打开 http://<IP>:18080/debug/index.html
"""

import argparse
import logging
import socket

from src.serving.model_loader import load_model
from src.serving.base_policy import BaseServePolicy
from src.serving.model_policies.beast_policy import BeastPolicy
from src.serving.model_policies.fast_policy import FastPolicy
from src.serving.model_policies.oft_policy import OftPolicy
from src.serving.model_policies.niaf_policy import NiafPolicy
from src.serving.model_policies.niaf_vel_policy import NiafVelPolicy

logger = logging.getLogger(__name__)

# Model type → (Policy class, display name, metadata model_type tag)
POLICY_MAP = {
    "niaf": (NiafPolicy, "NIAF", "niaf"),
    "niaf_vel": (NiafVelPolicy, "NIAFVel", "niaf_vel"),    
    "beast": (BeastPolicy, "BEAST", "beast"),
    "fast": (FastPolicy, "FAST", "fast"),
    "oft": (OftPolicy, "OFT", "oft"),
}


def _build_policy(
    model_type: str,
    model,
    ckpt_info: dict,
    args,
) -> BaseServePolicy:
    """Construct the appropriate Policy instance."""
    policy_cls, _, _ = POLICY_MAP[model_type]

    # Determine action_mode: CLI override > checkpoint > default
    action_mode = args.action_mode
    if action_mode is None:
        action_mode = ckpt_info.get("action_mode", "delta_first")

    # Determine arm_mode: CLI override > checkpoint > default
    arm_mode = args.arm_mode
    if arm_mode is None:
        arm_mode = ckpt_info.get("arm_mode", "dual")

    # Camera toggles (--no-cam-xxx has priority)
    use_cam_high = args.use_cam_high and not args.no_cam_high
    use_cam_left_wrist = args.use_cam_left_wrist and not args.no_cam_left_wrist
    use_cam_right_wrist = args.use_cam_right_wrist and not args.no_cam_right_wrist

    common_kwargs = dict(
        device=args.device,
        default_prompt=args.default_prompt,
        image_size=224,
        robot_name=args.robot_name,
        num_arms=args.num_arms,
        action_space=args.action_space,
        prompt_style=args.prompt_style,
        action_mode=action_mode,
        arm_mode=arm_mode,
        use_cam_high=use_cam_high,
        use_cam_left_wrist=use_cam_left_wrist,
        use_cam_right_wrist=use_cam_right_wrist,
        enable_csv_logging=args.enable_csv_logging,
        target_chunk_size=args.target_chunk_size,
        return_joint_vel=args.return_joint_vel,
        execution_hz=args.execution_hz,
        debug=args.debug,
    )

    if model_type == "niaf_vel":
        return policy_cls(model, **common_kwargs)

    return policy_cls(model, **common_kwargs)


def parse_args():
    p = argparse.ArgumentParser(
        description="NIAF Unified Model Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required ----
    p.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=list(POLICY_MAP.keys()),
        help="Model type to serve",
    )
    p.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )

    # ---- Server ----
    p.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Server host")

    # ---- Device ----
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # ---- VLM backbone ----
    p.add_argument(
        "--vlm-path", type=str,
        default="/data1/lhy/models/Florence-2/large",
        help="Path to Florence-2 backbone weights",
    )

    # ---- EMA ----
    p.add_argument("--no-ema", action="store_true", help="Skip loading EMA weights")

    # ---- Action ----
    p.add_argument(
        "--action-mode", type=str, default=None,
        choices=["absolute", "relative", "delta_first"],
        help="Action mode (default: read from checkpoint, fallback delta_first)",
    )
    p.add_argument(
        "--arm-mode", type=str, default=None,
        choices=["dual", "left", "right"],
        help="Arm mode (default: read from checkpoint, fallback dual)",
    )

    # ---- Prompt ----
    p.add_argument("--default-prompt", type=str, default="pick up the object")
    p.add_argument("--robot-name", type=str, default="ALOHA Bimanual")
    p.add_argument("--num-arms", type=str, default="2")
    p.add_argument("--action-space", type=str, default="Joint Position")
    p.add_argument(
        "--prompt-style", type=str, default="minimal",
        choices=["combined", "structured", "visual", "minimal"],
    )

    # ---- Cameras ----
    p.add_argument("--use-cam-high", action="store_true", default=True)
    p.add_argument("--no-cam-high", action="store_true")
    p.add_argument("--use-cam-left-wrist", action="store_true", default=True)
    p.add_argument("--no-cam-left-wrist", action="store_true")
    p.add_argument("--use-cam-right-wrist", action="store_true", default=True)
    p.add_argument("--no-cam-right-wrist", action="store_true")

    # ---- Interpolation ----
    p.add_argument(
        "--target-chunk-size", type=int, default=None,
        help="Interpolate action chunk to this length",
    )

    # ---- Logging ----
    p.add_argument("--enable-csv-logging", action="store_true")

    # ---- Unified debug visualization ----
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable unified VLA debug capture and dashboard output",
    )

    # ---- NIAFVel specific ----
    p.add_argument(
        "--return-joint-vel", action="store_true",
        help="Return joint velocities alongside positions for supported models",
    )
    p.add_argument(
        "--execution-hz", type=float, default=30.0,
        help="Execution frequency (Hz) used for velocity conversion",
    )

    return p.parse_args()


def main():
    args = parse_args()
    model_type = args.model

    logger.info("=" * 60)
    logger.info(f"NIAF Model Server — {POLICY_MAP[model_type][1]}")
    logger.info("=" * 60)

    # 1. Load model
    model, ckpt_info = load_model(
        model_type=model_type,
        checkpoint_path=args.checkpoint,
        device=args.device,
        vlm_path=args.vlm_path,
        load_ema=not args.no_ema,
    )

    # 2. Build policy
    policy = _build_policy(model_type, model, ckpt_info, args)

    # 3. Metadata
    _, display_name, meta_type = POLICY_MAP[model_type]
    metadata = policy.build_metadata(display_name, meta_type)
    if model_type in {"beast", "fast"}:
        metadata["returns_velocity"] = True
    if args.return_joint_vel and model_type == "niaf_vel":
        metadata["returns_velocity"] = True

    # 4. Print server info
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"

    logger.info(f"Metadata: {metadata}")
    logger.info(f"Host: {hostname} ({local_ip}), Port: {args.port}")
    logger.info(
        f"Client: python -m examples.mobile_aloha_AgileX.main "
        f"--host {local_ip} --port {args.port}"
    )
    logger.info("=" * 60)

    # 5. Start WebSocket server
    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logger.info("Server running. Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main()
