# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ..soccer_lab_marl.layout import compute_single_g1_spawn_pose
from ..soccer_lab_marl.soccer_lab_marl_env import SoccerLabMarlEnv
from .soccer_single_g1_env_cfg import SoccerSingleG1EnvCfg


class SoccerSingleG1Env(SoccerLabMarlEnv):
    """单机器人 G1 足球环境，支持内嵌 unitree_rl_gym 站立策略。"""

    cfg: SoccerSingleG1EnvCfg

    def __init__(self, cfg: SoccerSingleG1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        controllers = self._get_motion_controllers()
        self._motion_controller = controllers.get(self.agent_names[0]) if self.agent_names else None

    def _build_spawn_poses(self) -> dict[str, tuple[float, float, float, float]]:
        if len(self.cfg.possible_agents) != 1:
            raise RuntimeError(
                "SoccerSingleG1Env expects exactly one agent in possible_agents, "
                f"got {self.cfg.possible_agents}."
            )

        single_agent_name = self.cfg.possible_agents[0]
        return compute_single_g1_spawn_pose(
            field_size=self.cfg.field_size,
            team_spacing=self.cfg.team_spacing,
            base_height=self.cfg.spawn_height,
            spawn_reference_agent=self.cfg.single_spawn_reference,
            single_agent_name=single_agent_name,
        )
