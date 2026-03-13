from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from Soccer_Lab.g1_asset import G1_COMP_CFG

from ..soccer_lab_marl.soccer_lab_marl_env_cfg import SoccerLabMarlEnvCfg
from .g1_motion_policy import (
    G1_POLICY_DEFAULT_ANGLES,
    G1_POLICY_JOINT_NAMES,
    G1_POLICY_KD,
    G1_POLICY_KP,
)


_REPO_ROOT = Path(__file__).resolve().parents[6]
_UNITREE_RL_POLICY_PATH = str((_REPO_ROOT / "assets" / "g1_comp" / "policies" / "g1_motion.pt").resolve())

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
class SoccerSingleG1EnvCfg(SoccerLabMarlEnvCfg):
    """单机器人 G1 足球环境，并内嵌 unitree_rl_gym 底层运控策略。"""

    possible_agents = ["a1"]
    action_spaces = {"a1": len(G1_POLICY_JOINT_NAMES)}
    observation_spaces = {"a1": len(G1_POLICY_JOINT_NAMES) * 2}
    single_spawn_reference = "a4"

    decimation = 4
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    controlled_joint_names = list(G1_POLICY_JOINT_NAMES)
    action_scale = 0.25

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
