from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch


G1_POLICY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

G1_POLICY_DEFAULT_ANGLES: tuple[float, ...] = (
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
)

G1_POLICY_KP: tuple[float, ...] = (100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 100.0, 100.0, 100.0, 150.0, 40.0, 40.0)
G1_POLICY_KD: tuple[float, ...] = (2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0)
G1_POLICY_CMD_SCALE: tuple[float, float, float] = (2.0, 2.0, 0.25)


class G1MotionPolicyController:
    """Lightweight wrapper around the unitree_rl_gym G1 TorchScript policy."""

    def __init__(
        self,
        policy_path: str | Path,
        *,
        device: str = "cpu",
        control_dt: float = 0.02,
        gait_period: float = 0.8,
        action_scale: float = 0.25,
        ang_vel_scale: float = 0.25,
        dof_pos_scale: float = 1.0,
        dof_vel_scale: float = 0.05,
        cmd: Sequence[float] = (0.0, 0.0, 0.0),
        cmd_scale: Sequence[float] = G1_POLICY_CMD_SCALE,
        default_angles: Sequence[float] = G1_POLICY_DEFAULT_ANGLES,
    ) -> None:
        self.policy_path = Path(policy_path)
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Cannot find motion policy at {self.policy_path}")

        self.device = torch.device(device)
        self.policy = torch.jit.load(str(self.policy_path), map_location=self.device)
        self.control_dt = float(control_dt)
        self.gait_period = float(gait_period)
        self.action_scale = float(action_scale)
        self.ang_vel_scale = float(ang_vel_scale)
        self.dof_pos_scale = float(dof_pos_scale)
        self.dof_vel_scale = float(dof_vel_scale)
        self.default_angles = self._to_1d_tensor(default_angles, expected_dim=12)
        self.cmd = self._to_1d_tensor(cmd, expected_dim=3)
        self.cmd_scale = self._to_1d_tensor(cmd_scale, expected_dim=3)
        self.previous_action = torch.zeros(12, dtype=torch.float32, device=self.device)
        self.step_count = 0
        self.reset()

    def reset(self) -> None:
        self.step_count = 0
        self.previous_action.zero_()
        if hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()

    def build_observation(
        self,
        *,
        base_ang_vel_b: torch.Tensor | Sequence[float],
        projected_gravity_b: torch.Tensor | Sequence[float],
        joint_pos: torch.Tensor | Sequence[float],
        joint_vel: torch.Tensor | Sequence[float],
    ) -> torch.Tensor:
        base_ang_vel_b = self._to_1d_tensor(base_ang_vel_b, expected_dim=3)
        projected_gravity_b = self._to_1d_tensor(projected_gravity_b, expected_dim=3)
        joint_pos = self._to_1d_tensor(joint_pos, expected_dim=12)
        joint_vel = self._to_1d_tensor(joint_vel, expected_dim=12)

        phase = (self.step_count * self.control_dt) % self.gait_period / self.gait_period
        sin_phase = math.sin(2.0 * math.pi * phase)
        cos_phase = math.cos(2.0 * math.pi * phase)

        return torch.cat(
            (
                base_ang_vel_b * self.ang_vel_scale,
                projected_gravity_b,
                self.cmd * self.cmd_scale,
                (joint_pos - self.default_angles) * self.dof_pos_scale,
                joint_vel * self.dof_vel_scale,
                self.previous_action,
                torch.tensor([sin_phase, cos_phase], dtype=torch.float32, device=self.device),
            ),
            dim=0,
        )

    def infer(
        self,
        *,
        base_ang_vel_b: torch.Tensor | Sequence[float],
        projected_gravity_b: torch.Tensor | Sequence[float],
        joint_pos: torch.Tensor | Sequence[float],
        joint_vel: torch.Tensor | Sequence[float],
    ) -> torch.Tensor:
        obs = self.build_observation(
            base_ang_vel_b=base_ang_vel_b,
            projected_gravity_b=projected_gravity_b,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )
        with torch.inference_mode():
            action = self.policy(obs.unsqueeze(0)).squeeze(0).to(self.device)
        self.previous_action.copy_(action)
        self.step_count += 1
        return self.default_angles + self.action_scale * action

    def _to_1d_tensor(self, value: torch.Tensor | Sequence[float], *, expected_dim: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).flatten()
        if tensor.numel() != expected_dim:
            raise ValueError(f"Expected tensor with {expected_dim} elements, got {tensor.numel()}")
        return tensor


__all__ = [
    "G1MotionPolicyController",
    "G1_POLICY_JOINT_NAMES",
    "G1_POLICY_DEFAULT_ANGLES",
    "G1_POLICY_KP",
    "G1_POLICY_KD",
    "G1_POLICY_CMD_SCALE",
]
