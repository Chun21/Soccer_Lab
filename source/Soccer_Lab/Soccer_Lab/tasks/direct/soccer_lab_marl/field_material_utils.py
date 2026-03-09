# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


def remap_absolute_world_looks_target(target_path: str, field_prim_path: str) -> str:
    """Remap absolute `/World/Looks/*` material target into field-local `Looks` scope."""

    world_looks_prefix = "/World/Looks/"
    if not target_path.startswith(world_looks_prefix):
        return target_path

    material_suffix = target_path[len(world_looks_prefix) :]
    if not material_suffix:
        return target_path
    return f"{field_prim_path}/Looks/{material_suffix}"
