# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from ..soccer_lab_marl.soccer_lab_marl_env_cfg import SoccerLabMarlEnvCfg


@configclass
class SoccerSingleG1EnvCfg(SoccerLabMarlEnvCfg):

    possible_agents = ["a1"]
    action_spaces = {"a1": 12}
    observation_spaces = {"a1": 24}
    single_spawn_reference = "a4"
