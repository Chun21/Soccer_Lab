# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import Camera

from ..soccer_single_g1.g1_motion_policy import G1MotionPolicyController
from .field_specs import build_field_line_specs, build_goal_post_specs, get_field_preset
from .layout import compute_g1_team_poses
from .soccer_lab_marl_env_cfg import SoccerLabMarlEnvCfg


class SoccerLabMarlEnv(DirectMARLEnv):
    """4v4 Unitree G1 足球 MARL 环境（先搭建场景与基础动作接口）。"""

    cfg: SoccerLabMarlEnvCfg

    def __init__(self, cfg: SoccerLabMarlEnvCfg, render_mode: str | None = None, **kwargs):
        # _setup_scene is called inside DirectMARLEnv.__init__, so field config must exist before super().
        self.field_cfg = get_field_preset(cfg.field_preset_name)
        super().__init__(cfg, render_mode, **kwargs)

        self.agent_names = list(self.cfg.possible_agents)
        self.control_joint_ids = self._resolve_control_joint_ids(
            robot=self.robots[self.agent_names[0]],
            expected_joint_names=self.cfg.controlled_joint_names,
        )
        self._refresh_agent_spaces()

        team_poses = self._build_spawn_poses()
        if set(team_poses.keys()) != set(self.agent_names):
            raise RuntimeError("Team pose layout does not match configured agent names.")

        self._spawn_root_pose = {
            name: torch.tensor(
                [x, y, z, *yaw_to_quat_wxyz(yaw)],
                dtype=torch.float,
                device=self.device,
            )
            for name, (x, y, z, yaw) in team_poses.items()
        }

        self.actions = {
            name: torch.zeros((self.num_envs, self.cfg.action_spaces[name]), dtype=torch.float, device=self.device)
            for name in self.agent_names
        }

        self._default_joint_pos = {
            name: robot.data.default_joint_pos.clone() for name, robot in self.robots.items()
        }
        self._joint_pos_target = {
            name: robot.data.default_joint_pos.clone() for name, robot in self.robots.items()
        }
        self._motion_controllers: dict[str, G1MotionPolicyController] = {}
        self._policy_leg_joint_targets = {
            agent_name: robot.data.default_joint_pos[:, self.control_joint_ids].clone()
            for agent_name, robot in self.robots.items()
        }
        self._upper_body_hold_joint_ids, self._upper_body_hold_joint_pos = self._resolve_upper_body_hold_targets()
        if getattr(self.cfg, "use_unitree_rl_policy", False):
            if self.num_envs != 1:
                raise RuntimeError(
                    "SoccerLabMarlEnv currently supports unitree_rl_gym policy only with num_envs == 1, "
                    f"got {self.num_envs}."
                )
            self._motion_controllers = {
                agent_name: self._create_motion_policy_controller() for agent_name in self.agent_names
            }
        self._reset_motion_policy()

        ball_height = self.cfg.ball_cfg.init_state.pos[2]
        self._ball_spawn_offset = torch.tensor([0.0, 0.0, ball_height], dtype=torch.float, device=self.device)
        self._ball_spawn_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device)
        self._logged_spawn_info = False

    def _build_spawn_poses(self) -> dict[str, tuple[float, float, float, float]]:
        """Build robot spawn poses keyed by agent name."""

        return compute_g1_team_poses(
            field_size=self.cfg.field_size,
            team_spacing=self.cfg.team_spacing,
            base_height=self.cfg.spawn_height,
        )

    def _setup_scene(self):
        if self.cfg.spawn_pitch_base:
            pitch_base_cfg = sim_utils.CuboidCfg(
                size=self.cfg.pitch_base_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.pitch_base_color, roughness=0.9),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=self.cfg.pitch_base_friction[0],
                    dynamic_friction=self.cfg.pitch_base_friction[1],
                    restitution=0.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            )
            pitch_base_cfg.func(
                "/World/envs/env_.*/PitchBase",
                pitch_base_cfg,
                translation=(0.0, 0.0, self.cfg.pitch_base_z),
            )

        line_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.6)
        for line in build_field_line_specs(
            self.field_cfg, line_height=self.cfg.field_line_height, z_offset=self.cfg.field_line_z
        ):
            line_cfg = sim_utils.CuboidCfg(
                size=line.size,
                visual_material=line_material,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            )
            line_cfg.func(
                f"/World/envs/env_.*/FieldLine_{line.name}",
                line_cfg,
                translation=line.position,
                orientation=line.orientation,
            )

        if self.cfg.spawn_goal_posts:
            post_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.92), roughness=0.4)
            for post in build_goal_post_specs(self.field_cfg):
                post_cfg = sim_utils.CuboidCfg(
                    size=post.size,
                    visual_material=post_material,
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                )
                post_cfg.func(f"/World/envs/env_.*/Goal_{post.name}", post_cfg, translation=post.position)

        self.robots: dict[str, Articulation] = {}
        for agent_name in self.cfg.possible_agents:
            robot_cfg = self.cfg.g1_robot_cfg.replace(prim_path=f"/World/envs/env_.*/{agent_name.upper()}")
            self.robots[agent_name] = Articulation(robot_cfg)

        self.robot_cameras: dict[str, Camera] = {}
        if self.cfg.enable_robot_cameras:
            for agent_name in self.cfg.possible_agents:
                camera_prim_name = f"{agent_name.upper()}Camera"
                camera_cfg = self.cfg.robot_camera_cfg.replace(
                    prim_path=f"/World/envs/env_.*/{agent_name.upper()}/d455_link/{camera_prim_name}"
                )
                self.robot_cameras[agent_name] = Camera(camera_cfg)

        if self.cfg.spawn_ball:
            self.ball = RigidObject(self.cfg.ball_cfg)
        else:
            self.ball = None

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        for agent_name, robot in self.robots.items():
            self.scene.articulations[agent_name] = robot

        for agent_name, camera in self.robot_cameras.items():
            self.scene.sensors[f"{agent_name}_camera"] = camera

        if self.ball is not None:
            self.scene.rigid_objects["ball"] = self.ball

        light_cfg = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.62, 0.62, 0.60))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        if self._should_use_motion_policy():
            for agent_name in self.agent_names:
                if agent_name in actions:
                    self.actions[agent_name] = actions[agent_name]
                self._policy_leg_joint_targets[agent_name][:] = self._infer_motion_policy_targets(agent_name)
            return

        for agent_name in self.agent_names:
            if agent_name in actions:
                self.actions[agent_name] = actions[agent_name]

    def _apply_action(self) -> None:
        if self._should_use_motion_policy():
            for agent_name, robot in self.robots.items():
                full_body_target = self._compose_full_body_target(
                    base_target=self._default_joint_pos[agent_name],
                    leg_joint_ids=self.control_joint_ids,
                    leg_joint_targets=self._policy_leg_joint_targets[agent_name],
                    hold_joint_ids=self._upper_body_hold_joint_ids,
                    hold_joint_targets=self._upper_body_hold_joint_pos,
                )
                self._joint_pos_target[agent_name][:] = full_body_target
                robot.set_joint_position_target(self._joint_pos_target[agent_name])
            return

        for agent_name, robot in self.robots.items():
            clipped_actions = torch.clamp(self.actions[agent_name], -1.0, 1.0)
            target = self._joint_pos_target[agent_name]
            default = self._default_joint_pos[agent_name]
            target[:] = default
            target[:, self.control_joint_ids] = default[:, self.control_joint_ids] + clipped_actions * self.cfg.action_scale
            robot.set_joint_position_target(target)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations: dict[str, torch.Tensor] = {}
        for agent_name, robot in self.robots.items():
            joint_pos = robot.data.joint_pos[:, self.control_joint_ids]
            joint_vel = robot.data.joint_vel[:, self.control_joint_ids]
            observations[agent_name] = torch.cat((joint_pos, joint_vel), dim=-1)
        return observations

    def get_camera_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return RGB/depth tensors for each robot camera."""

        camera_obs: dict[str, dict[str, torch.Tensor]] = {}
        for agent_name, camera in self.robot_cameras.items():
            camera_obs[agent_name] = {
                "rgb": camera.data.output["rgb"],
                "depth": camera.data.output["depth"],
            }
        return camera_obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards: dict[str, torch.Tensor] = {}
        ball_xy = self.ball.data.root_pos_w[:, :2] if self.ball is not None else None

        for agent_name, robot in self.robots.items():
            rew = torch.full((self.num_envs,), self.cfg.rew_scale_alive, device=self.device)
            if ball_xy is not None:
                dist_to_ball = torch.linalg.norm(robot.data.root_pos_w[:, :2] - ball_xy, dim=1)
                rew -= self.cfg.rew_scale_distance_to_ball * dist_to_ball
            rewards[agent_name] = rew
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {
            agent_name: robot.data.root_pos_w[:, 2] < self.cfg.fall_height_threshold
            for agent_name, robot in self.robots.items()
        }
        time_outs = {agent_name: time_out for agent_name in self.agent_names}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robots[self.agent_names[0]]._ALL_INDICES
        super()._reset_idx(env_ids)

        env_origins = self.scene.env_origins[env_ids]

        for agent_name, robot in self.robots.items():
            default_root_state = robot.data.default_root_state[env_ids].clone()
            spawn_pose = self._spawn_root_pose[agent_name].unsqueeze(0).repeat(len(env_ids), 1)
            default_root_state[:, :3] = spawn_pose[:, :3] + env_origins
            default_root_state[:, 3:7] = spawn_pose[:, 3:7]
            default_root_state[:, 7:] = 0.0

            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()

            self._default_joint_pos[agent_name][env_ids] = joint_pos
            self._joint_pos_target[agent_name][env_ids] = joint_pos

            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if self.ball is not None:
            ball_state = self.ball.data.default_root_state[env_ids].clone()
            ball_state[:, :3] = env_origins + self._ball_spawn_offset
            ball_state[:, 3:7] = self._ball_spawn_quat
            ball_state[:, 7:] = 0.0
            self.ball.write_root_state_to_sim(ball_state, env_ids)

        if not self._logged_spawn_info and len(env_ids) > 0:
            spawn_preview = {}
            for agent_name in self.agent_names:
                pos = self._spawn_root_pose[agent_name][:3].tolist()
                spawn_preview[agent_name.upper()] = tuple(round(float(v), 3) for v in pos)
            print(f"[INFO]: G1 spawn positions (env_0): {spawn_preview}")
            self._logged_spawn_info = True

        self._reset_motion_policy()

    @staticmethod
    def _resolve_control_joint_ids(robot: Articulation, expected_joint_names: list[str] | None) -> list[int]:
        if expected_joint_names is None or len(expected_joint_names) == 0:
            return list(range(len(robot.joint_names)))
        if len(expected_joint_names) == 1 and expected_joint_names[0].lower() in {"__all__", "all", "*"}:
            return list(range(len(robot.joint_names)))

        missing = [joint_name for joint_name in expected_joint_names if joint_name not in robot.joint_names]
        if missing:
            raise RuntimeError(f"Cannot find configured control joints on G1 asset: {missing}")
        return [robot.joint_names.index(joint_name) for joint_name in expected_joint_names]

    def _refresh_agent_spaces(self) -> None:
        """Refresh action/observation spaces after resolving controlled joints."""

        action_dim = len(self.control_joint_ids)
        obs_dim = action_dim * 2
        self.cfg.action_spaces = {agent: action_dim for agent in self.agent_names}
        self.cfg.observation_spaces = {agent: obs_dim for agent in self.agent_names}
        self._configure_env_spaces()

    def _create_motion_policy_controller(self) -> G1MotionPolicyController:
        return G1MotionPolicyController(
            policy_path=self.cfg.unitree_rl_policy_path,
            device=self.device,
            control_dt=self.cfg.unitree_rl_control_dt,
            gait_period=self.cfg.unitree_rl_gait_period,
            action_scale=self.cfg.unitree_rl_action_scale,
            ang_vel_scale=self.cfg.unitree_rl_ang_vel_scale,
            dof_pos_scale=self.cfg.unitree_rl_dof_pos_scale,
            dof_vel_scale=self.cfg.unitree_rl_dof_vel_scale,
            cmd=self.cfg.unitree_rl_cmd,
            cmd_scale=self.cfg.unitree_rl_cmd_scale,
            default_angles=self.cfg.unitree_rl_default_angles,
        )

    def _get_motion_controllers(self) -> dict[str, G1MotionPolicyController]:
        controllers = getattr(self, "_motion_controllers", None)
        if controllers:
            return controllers

        legacy_controller = getattr(self, "_motion_controller", None)
        if legacy_controller is not None and len(getattr(self, "agent_names", [])) == 1:
            return {self.agent_names[0]: legacy_controller}
        return {}

    def _should_use_motion_policy(self) -> bool:
        return bool(getattr(self.cfg, "use_unitree_rl_policy", False) and self._get_motion_controllers())

    def _resolve_upper_body_hold_targets(self) -> tuple[list[int], torch.Tensor]:
        if not self.agent_names:
            return [], torch.empty(0, dtype=torch.float32, device=self.device)

        robot = self.robots[self.agent_names[0]]
        hold_joint_pos = getattr(self.cfg, "upper_body_hold_joint_pos", {})
        hold_joint_names = [joint_name for joint_name in hold_joint_pos if joint_name in robot.joint_names]
        hold_joint_ids = [robot.joint_names.index(joint_name) for joint_name in hold_joint_names]
        hold_joint_targets = torch.tensor(
            [hold_joint_pos[joint_name] for joint_name in hold_joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        return hold_joint_ids, hold_joint_targets

    def _reset_motion_policy(self) -> None:
        if not self._should_use_motion_policy():
            return

        for agent_name, controller in self._get_motion_controllers().items():
            controller.reset()
            default_targets = controller.default_angles.unsqueeze(0)
            if agent_name in self._policy_leg_joint_targets:
                self._policy_leg_joint_targets[agent_name][:] = default_targets

    def _infer_motion_policy_targets(self, agent_name: str) -> torch.Tensor:
        controller = self._get_motion_controllers()[agent_name]
        robot = self.robots[agent_name]
        leg_joint_targets = controller.infer(
            base_ang_vel_b=robot.data.root_ang_vel_b[0],
            projected_gravity_b=robot.data.projected_gravity_b[0],
            joint_pos=robot.data.joint_pos[0, self.control_joint_ids],
            joint_vel=robot.data.joint_vel[0, self.control_joint_ids],
        )
        if leg_joint_targets.ndim == 1:
            leg_joint_targets = leg_joint_targets.unsqueeze(0)
        return leg_joint_targets

    @staticmethod
    def _compose_full_body_target(
        *,
        base_target: torch.Tensor,
        leg_joint_ids: list[int],
        leg_joint_targets: torch.Tensor,
        hold_joint_ids: list[int],
        hold_joint_targets: torch.Tensor,
    ) -> torch.Tensor:
        target = base_target.clone()

        if leg_joint_targets.ndim == 1:
            leg_joint_targets = leg_joint_targets.unsqueeze(0)
        target[:, leg_joint_ids] = leg_joint_targets

        if len(hold_joint_ids) > 0:
            if hold_joint_targets.ndim == 1:
                hold_joint_targets = hold_joint_targets.unsqueeze(0).expand(target.shape[0], -1)
            target[:, hold_joint_ids] = hold_joint_targets

        return target


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
