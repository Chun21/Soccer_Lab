# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

DEFAULT_G1_AGENT_ORDER: tuple[str, ...] = ("a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4")


def compute_g1_team_poses(
    field_size: tuple[float, float],
    team_spacing: tuple[float, float],
    base_height: float,
) -> dict[str, tuple[float, float, float, float]]:
    """Compute a symmetric 4v4 spawn layout for Unitree G1 soccer players."""

    half_field_length = field_size[0] * 0.5
    team_center_offset_x = half_field_length * 0.5
    half_dx = team_spacing[0] * 0.5
    half_dy = team_spacing[1] * 0.5

    local_offsets = [
        (-half_dx, -half_dy),
        (-half_dx, +half_dy),
        (+half_dx, -half_dy),
        (+half_dx, +half_dy),
    ]

    poses: dict[str, tuple[float, float, float, float]] = {}
    for agent_name, (offset_x, offset_y) in zip(DEFAULT_G1_AGENT_ORDER[:4], local_offsets, strict=True):
        poses[agent_name] = (-team_center_offset_x + offset_x, offset_y, base_height, 0.0)
    for agent_name, (offset_x, offset_y) in zip(DEFAULT_G1_AGENT_ORDER[4:], local_offsets, strict=True):
        poses[agent_name] = (team_center_offset_x + offset_x, offset_y, base_height, math.pi)

    # Role-based 4v4 shape: 1 goalkeeper + 2 center players + 1 forward per team.
    keeper_x_offset = min(half_field_length - 0.8, half_field_length * 0.85)
    center_x_offset = half_field_length * 0.45
    forward_x_offset = half_field_length * 0.25
    center_y_offset = half_dy

    # Team A (attacks +X)
    poses["a1"] = (-keeper_x_offset, 0.0, base_height, 0.0)  # goalkeeper
    poses["a2"] = (-center_x_offset, -center_y_offset, base_height, 0.0)  # center
    poses["a3"] = (-center_x_offset, center_y_offset, base_height, 0.0)  # center
    poses["a4"] = (-forward_x_offset, 0.0, base_height, 0.0)  # forward

    # Team B (attacks -X)
    poses["b1"] = (keeper_x_offset, 0.0, base_height, math.pi)  # goalkeeper
    poses["b2"] = (center_x_offset, -center_y_offset, base_height, math.pi)  # center
    poses["b3"] = (center_x_offset, center_y_offset, base_height, math.pi)  # center
    poses["b4"] = (forward_x_offset, 0.0, base_height, math.pi)  # forward

    return poses


def compute_single_g1_spawn_pose(
    field_size: tuple[float, float],
    team_spacing: tuple[float, float],
    base_height: float,
    spawn_reference_agent: str = "a4",
    single_agent_name: str = "a1",
) -> dict[str, tuple[float, float, float, float]]:
    """Map a single-agent task spawn pose to one of the standard 4v4 reference poses."""

    team_poses = compute_g1_team_poses(
        field_size=field_size,
        team_spacing=team_spacing,
        base_height=base_height,
    )
    if spawn_reference_agent not in team_poses:
        raise ValueError(
            f"Unknown single-agent spawn reference '{spawn_reference_agent}'. "
            f"Available references: {sorted(team_poses.keys())}"
        )
    return {single_agent_name: team_poses[spawn_reference_agent]}
