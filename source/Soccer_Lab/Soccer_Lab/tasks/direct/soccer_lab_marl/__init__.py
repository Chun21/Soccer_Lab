# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

for env_id in ("Template-Soccer-Lab-Marl-Direct-v0", "SoccerLab-G1-Soccer-4v4-Direct-v0"):
    gym.register(
        id=env_id,
        entry_point=f"{__name__}.soccer_lab_marl_env:SoccerLabMarlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.soccer_lab_marl_env_cfg:SoccerLabMarlEnvCfg",
            "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )
