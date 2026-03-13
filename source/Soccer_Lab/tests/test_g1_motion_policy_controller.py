from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


TEST_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = TEST_ROOT / "Soccer_Lab" / "tasks" / "direct" / "soccer_single_g1" / "g1_motion_policy.py"
POLICY_PATH = TEST_ROOT.parents[1] / "assets" / "g1_comp" / "policies" / "g1_motion.pt"


def _load_module():
    spec = importlib.util.spec_from_file_location("g1_motion_policy", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["g1_motion_policy"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_motion_policy_controller_builds_47d_obs_and_12d_targets():
    module = _load_module()
    controller = module.G1MotionPolicyController(policy_path=POLICY_PATH, device="cpu")

    base_ang_vel_b = torch.zeros(3)
    projected_gravity_b = torch.tensor([0.0, 0.0, -1.0])
    joint_pos = controller.default_angles.clone()
    joint_vel = torch.zeros(12)

    obs = controller.build_observation(
        base_ang_vel_b=base_ang_vel_b,
        projected_gravity_b=projected_gravity_b,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )
    target = controller.infer(
        base_ang_vel_b=base_ang_vel_b,
        projected_gravity_b=projected_gravity_b,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )

    assert obs.shape == (47,)
    assert target.shape == (12,)
    assert controller.previous_action.shape == (12,)
    torch.testing.assert_close(target, controller.default_angles + controller.action_scale * controller.previous_action)


def test_motion_policy_controller_reset_clears_internal_tracking_state():
    module = _load_module()
    controller = module.G1MotionPolicyController(policy_path=POLICY_PATH, device="cpu")
    controller.previous_action[:] = 1.0
    controller.step_count = 7

    controller.reset()

    assert controller.step_count == 0
    torch.testing.assert_close(controller.previous_action, torch.zeros(12))


def test_motion_policy_joint_order_matches_unitree_mujoco_actuator_order():
    module = _load_module()
    assert module.G1_POLICY_JOINT_NAMES[:6] == (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    assert module.G1_POLICY_DEFAULT_ANGLES[:6] == (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0)
