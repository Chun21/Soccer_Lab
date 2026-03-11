# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from Soccer_Lab.g1_asset import G1_COMP_CFG

from .field_specs import get_field_preset
from .layout import DEFAULT_G1_AGENT_ORDER


@configclass
class SoccerLabMarlEnvCfg(DirectMARLEnvCfg):
    """4v4 Unitree G1 足球多智能体环境配置。"""

    field_preset_name = "M"
    _field_cfg = get_field_preset("M")

    # viewer
    # Keep the whole pitch and all 8 robots visible at startup.
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, -0.5, 22.0), lookat=(0.0, 0.0, 0.0), origin_type="world")

    # env
    decimation = 4
    episode_length_s = 30.0

    # multi-agent specification and spaces definition
    possible_agents = list(DEFAULT_G1_AGENT_ORDER)
    action_spaces = {agent: 12 for agent in possible_agents}
    observation_spaces = {agent: 24 for agent in possible_agents}
    state_space = -1

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=30.0, replicate_physics=True)

    # robot
    g1_robot_cfg: ArticulationCfg = G1_COMP_CFG.replace(prim_path="/World/envs/env_.*/G1")
    # Control all articulated joints by default.
    # You can also provide a custom subset of joint names if needed.
    controlled_joint_names = ["__all__"]
    action_scale = 0.35
    fall_height_threshold = 0.42

    # onboard D455 cameras (mounted per robot on the head-top camera link)
    enable_robot_cameras = True
    robot_camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/RobotCamera",
        update_period=0.0,
        height=120,
        width=160,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Mounted directly on d455_link using the original URDF pose.
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # soccer pitch (mimic soccerLab procedural field instead of relying on external USD field assets)
    field_size = (_field_cfg.field_length, _field_cfg.field_width)
    team_spacing = (2.0, 3.0)
    spawn_height = 0.78

    # procedural pitch base
    spawn_pitch_base = True
    pitch_base_size = (
        _field_cfg.field_length + 2.0 * _field_cfg.border_strip_width + 4.0,
        _field_cfg.field_width + 2.0 * _field_cfg.border_strip_width + 4.0,
        0.04,
    )
    pitch_base_z = -0.02
    pitch_base_color = (0.12, 0.48, 0.24)
    pitch_base_friction = (1.0, 1.0)
    field_line_height = 0.01
    field_line_z = 0.005
    spawn_goal_posts = False

    # ball
    spawn_ball = True
    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                max_linear_velocity=80.0,
                max_angular_velocity=200.0,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.11), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # placeholder rewards
    rew_scale_alive = 0.05
    rew_scale_distance_to_ball = 0.01

    def __post_init__(self):
        field_cfg = get_field_preset(self.field_preset_name)
        self.field_size = (field_cfg.field_length, field_cfg.field_width)
        self.pitch_base_size = (
            field_cfg.field_length + 2.0 * field_cfg.border_strip_width + 4.0,
            field_cfg.field_width + 2.0 * field_cfg.border_strip_width + 4.0,
            0.04,
        )

        camera_height = max(16.0, field_cfg.field_length * 1.5)
        self.viewer.eye = (0.0, -0.5, camera_height)
        self.viewer.lookat = (0.0, 0.0, 0.0)
