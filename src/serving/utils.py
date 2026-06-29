"""Serving utilities: prompt generation and shared constants."""

from functools import partial
from typing import List

import numpy as np

# =============================================================================
# CLIP image normalization (aligned with training transforms)
# =============================================================================
CLIP_MEAN: List[float] = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD: List[float] = [0.26862954, 0.26130258, 0.27577711]

# =============================================================================
# Default reset poses (14-dim: left_arm(7) + right_arm(7))
# =============================================================================
LEFT_RESET_POSE = np.array([
    -0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
    -0.032616615295410156, -0.00286102294921875, 0.00095367431640625,
    3.557830810546875,
], dtype=np.float32)

RIGHT_RESET_POSE = np.array([
    -0.00133514404296875, 0.00438690185546875, 0.034523963928222656,
    -0.053597450256347656, -0.00476837158203125, -0.00209808349609375,
    3.557830810546875,
], dtype=np.float32)

# Joint indices (excluding grippers) for each arm configuration
JOINT_INDICES_7D = [0, 1, 2, 3, 4, 5]       # 6 joints, gripper at index 6
JOINT_INDICES_14D = [0, 1, 2, 3, 4, 5,      # left arm joints
                     7, 8, 9, 10, 11, 12]    # right arm joints
# Grippers: index 6 (left), index 13 (right) — always treated as absolute


def generate_policy_prompt(
    instruction: str,
    robot_name: str = "Franka Panda",
    num_arms: str = "1",
    action_space: str = "Delta End-Effector",
    prompt_style: str = "minimal",
    include_meta: bool = True,
) -> str:
    """Generate a structured prompt for VLA policy (aligned with training)."""
    meta_info = f"Agent Type: {num_arms}-arm {robot_name}, Action Space: {action_space}, "

    prompts = {
        "combined": (
            f"{meta_info if include_meta else ''}"
            f"</od>Task Instruction: {instruction}</od>"
            f"<grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
        ),
        "visual": (
            f"<od>Task Instruction: {instruction}, </od>"
            f"<grounding>identify key objects and their spatial relationships</grounding>"
            f"<region_cap>analyze motion paths and collision-free trajectories</region_cap>"
            f"<dense_region_caption>determine optimal grasp points and manipulation targets</dense_region_caption>"
            f"{f'<cap>{meta_info}</cap>' if include_meta else ''}"
        ),
        "structured": (
            f"<od>ROBOT CONFIGURATION:\n{meta_info if include_meta else ''}\n\n"
            f"TASK OBJECTIVE:\n{instruction}\n\n"
            f"ANALYSIS REQUIREMENTS:\n"
            f"- Identify target objects and obstacles\n"
            f"- Determine spatial relationships\n"
            f"- Plan manipulation sequence</od>"
        ),
        "minimal": (
            f"{meta_info if include_meta else ''} Task Instruction: {instruction}"
        ),
    }

    if prompt_style not in prompts:
        raise ValueError(f"Invalid prompt style: {prompt_style}. Choose from: {list(prompts.keys())}")

    prompt = prompts[prompt_style].strip()
    prompt = " ".join(line.strip() for line in prompt.split("\n"))
    return prompt


def make_prompt_formatter(
    robot_name: str = "AgileX Cobot Magic",
    num_arms: str = "2",
    action_space: str = "Joint Position",
    prompt_style: str = "minimal",
):
    """Return a callable that formats a raw instruction into a policy prompt."""
    return partial(
        generate_policy_prompt,
        robot_name=robot_name,
        action_space=action_space,
        num_arms=num_arms,
        prompt_style=prompt_style,
    )
