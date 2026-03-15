# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from Soccer_Lab.g1_asset import G1_COMP_CFG

from .field_specs import get_field_preset
from .layout import DEFAULT_G1_AGENT_ORDER
from ..soccer_single_g1.g1_motion_policy import (
    G1_POLICY_DEFAULT_ANGLES,
    G1_POLICY_JOINT_NAMES,
    G1_POLICY_KD,
    G1_POLICY_KP,
)


_REPO_ROOT = Path(__file__).resolve().parents[6]
_UNITREE_RL_POLICY_PATH = str((_REPO_ROOT / "assets" / "g1_comp" / "policies" / "g1_motion.pt").resolve())
_GOAL_ASSET_PATH = str((_REPO_ROOT / "assets" / "goalpost.usd").resolve())
_BALL_ASSET_PATH = str((_REPO_ROOT / "assets" / "ball_asset" / "ball.usd").resolve())

_UPPER_BODY_HOLD_JOINT_POS: dict[str, float] = {
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.87,
    "right_wrist_roll_joint": 0.0,
    "xl330_joint": 0.0,
    "d455_joint": 0.0,
}

_POLICY_INIT_JOINT_POS: dict[str, float] = {
    joint_name: joint_pos for joint_name, joint_pos in zip(G1_POLICY_JOINT_NAMES, G1_POLICY_DEFAULT_ANGLES, strict=True)
}
_POLICY_INIT_JOINT_POS.update(_UPPER_BODY_HOLD_JOINT_POS)

_LEG_STIFFNESS = {joint_name: stiffness for joint_name, stiffness in zip(G1_POLICY_JOINT_NAMES, G1_POLICY_KP, strict=True)}
_LEG_DAMPING = {joint_name: damping for joint_name, damping in zip(G1_POLICY_JOINT_NAMES, G1_POLICY_KD, strict=True)}


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
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=30.0, replicate_physics=True)

    # robot
    g1_robot_cfg: ArticulationCfg = G1_COMP_CFG.replace(
        prim_path="/World/envs/env_.*/G1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.78),
            joint_pos=dict(_POLICY_INIT_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "policy_legs": ImplicitActuatorCfg(
                joint_names_expr=list(G1_POLICY_JOINT_NAMES),
                effort_limit_sim=300.0,
                stiffness=dict(_LEG_STIFFNESS),
                damping=dict(_LEG_DAMPING),
                armature={joint_name: 0.01 for joint_name in G1_POLICY_JOINT_NAMES},
            ),
            "upper_body": ImplicitActuatorCfg(
                joint_names_expr=[
                    "waist_yaw_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                ],
                effort_limit_sim=300.0,
                stiffness={
                    "waist_yaw_joint": 200.0,
                    ".*_shoulder_pitch_joint": 60.0,
                    ".*_shoulder_roll_joint": 60.0,
                    ".*_shoulder_yaw_joint": 40.0,
                    ".*_elbow_joint": 40.0,
                    ".*_wrist_roll_joint": 20.0,
                },
                damping={
                    "waist_yaw_joint": 5.0,
                    ".*_shoulder_pitch_joint": 6.0,
                    ".*_shoulder_roll_joint": 6.0,
                    ".*_shoulder_yaw_joint": 4.0,
                    ".*_elbow_joint": 4.0,
                    ".*_wrist_roll_joint": 2.0,
                },
                armature={
                    "waist_yaw_joint": 0.01,
                    ".*_shoulder_.*": 0.01,
                    ".*_elbow_joint": 0.01,
                    ".*_wrist_roll_joint": 0.01,
                },
            ),
            "sensors": ImplicitActuatorCfg(
                joint_names_expr=["xl330_joint", "d455_joint"],
                effort_limit_sim=5.0,
                stiffness=5.0,
                damping=0.5,
                armature=0.001,
            ),
        },
    )
    controlled_joint_names = list(G1_POLICY_JOINT_NAMES)
    action_scale = 0.25
    fall_height_threshold = 0.42

    use_unitree_rl_policy = True
    unitree_rl_policy_path = _UNITREE_RL_POLICY_PATH
    unitree_rl_control_dt = 0.02
    unitree_rl_gait_period = 0.8
    unitree_rl_action_scale = 0.25
    unitree_rl_ang_vel_scale = 0.25
    unitree_rl_dof_pos_scale = 1.0
    unitree_rl_dof_vel_scale = 0.05
    unitree_rl_cmd = (0.0, 0.0, 0.0)
    unitree_rl_cmd_scale = (2.0, 2.0, 0.25)
    unitree_rl_default_angles = G1_POLICY_DEFAULT_ANGLES
    unitree_rl_kp = G1_POLICY_KP
    unitree_rl_kd = G1_POLICY_KD
    upper_body_hold_joint_pos = dict(_UPPER_BODY_HOLD_JOINT_POS)

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
    spawn_goal_posts = True
    spawn_goal_asset = True
    goal_asset_path = _GOAL_ASSET_PATH
    goal_asset_scale = (1.0, 1.0, 1.0)
    goal_asset_z_offset = 0.0
    goal_asset_collision_enabled = False

    # ball
    spawn_ball = True
    ball_asset_path = _BALL_ASSET_PATH
    ball_asset_scale = (1.0, 1.0, 1.0)
    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ball_asset_path,
            scale=ball_asset_scale,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                max_linear_velocity=80.0,
                max_angular_velocity=200.0,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
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
