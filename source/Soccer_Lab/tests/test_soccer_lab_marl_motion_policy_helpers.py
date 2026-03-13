from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import torch


TEST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TEST_ROOT / "Soccer_Lab"


class _CfgBase:
    def __init__(self, *args, **kwargs):
        self._args = args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def replace(self, **kwargs):
        cloned = copy.copy(self)
        cloned.__dict__ = dict(self.__dict__)
        cloned.__dict__.update(kwargs)
        return cloned


class _ArticulationCfg(_CfgBase):
    class InitialStateCfg(_CfgBase):
        pass


class _RigidObjectCfg(_CfgBase):
    class InitialStateCfg(_CfgBase):
        pass


class _CameraCfg(_CfgBase):
    class OffsetCfg(_CfgBase):
        pass


class _UrdfConverterCfg:
    class JointDriveCfg(_CfgBase):
        class PDGainsCfg(_CfgBase):
            pass


class _DirectMARLEnvCfg(_CfgBase):
    pass


class _ViewerCfg(_CfgBase):
    pass


class _InteractiveSceneCfg(_CfgBase):
    pass


class _SimulationCfg(_CfgBase):
    pass


class _DirectMARLEnv:
    def __init__(self, *args, **kwargs):
        pass


class _Articulation:
    pass


class _RigidObject:
    pass


class _Camera:
    pass


def _configclass(cls):
    return cls


def _ensure_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_module(name: str, path: Path, *, is_package: bool = False):
    kwargs = {"submodule_search_locations": [str(path.parent)]} if is_package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_isaaclab_stubs() -> None:
    isaaclab_pkg = types.ModuleType("isaaclab")
    isaaclab_pkg.__path__ = []
    sys.modules["isaaclab"] = isaaclab_pkg

    sim_mod = types.ModuleType("isaaclab.sim")
    for name in [
        "UrdfFileCfg",
        "RigidBodyPropertiesCfg",
        "ArticulationRootPropertiesCfg",
        "PreviewSurfaceCfg",
        "RigidBodyMaterialCfg",
        "CollisionPropertiesCfg",
        "SphereCfg",
        "CuboidCfg",
        "DomeLightCfg",
        "PinholeCameraCfg",
        "MassPropertiesCfg",
    ]:
        setattr(sim_mod, name, type(name, (_CfgBase,), {}))
    sim_mod.UrdfConverterCfg = _UrdfConverterCfg
    sim_mod.SimulationCfg = _SimulationCfg
    sys.modules["isaaclab.sim"] = sim_mod
    isaaclab_pkg.sim = sim_mod

    actuators_mod = types.ModuleType("isaaclab.actuators")
    actuators_mod.ImplicitActuatorCfg = type("ImplicitActuatorCfg", (_CfgBase,), {})
    sys.modules["isaaclab.actuators"] = actuators_mod

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.ArticulationCfg = _ArticulationCfg
    assets_mod.RigidObjectCfg = _RigidObjectCfg
    assets_mod.Articulation = _Articulation
    assets_mod.RigidObject = _RigidObject
    sys.modules["isaaclab.assets"] = assets_mod

    envs_mod = types.ModuleType("isaaclab.envs")
    envs_mod.DirectMARLEnvCfg = _DirectMARLEnvCfg
    envs_mod.ViewerCfg = _ViewerCfg
    envs_mod.DirectMARLEnv = _DirectMARLEnv
    sys.modules["isaaclab.envs"] = envs_mod

    scene_mod = types.ModuleType("isaaclab.scene")
    scene_mod.InteractiveSceneCfg = _InteractiveSceneCfg
    sys.modules["isaaclab.scene"] = scene_mod

    sensors_mod = types.ModuleType("isaaclab.sensors")
    sensors_mod.CameraCfg = _CameraCfg
    sensors_mod.Camera = _Camera
    sys.modules["isaaclab.sensors"] = sensors_mod

    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.configclass = _configclass
    sys.modules["isaaclab.utils"] = utils_mod


def _install_project_packages() -> None:
    _ensure_package("Soccer_Lab", SRC_ROOT)
    _ensure_package("Soccer_Lab.tasks", SRC_ROOT / "tasks")
    _ensure_package("Soccer_Lab.tasks.direct", SRC_ROOT / "tasks" / "direct")
    _ensure_package("Soccer_Lab.tasks.direct.soccer_lab_marl", SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl")
    _ensure_package("Soccer_Lab.tasks.direct.soccer_single_g1", SRC_ROOT / "tasks" / "direct" / "soccer_single_g1")


def _load_marl_env_module():
    _install_isaaclab_stubs()
    _install_project_packages()
    _load_module("Soccer_Lab.g1_asset", SRC_ROOT / "g1_asset.py")
    _load_module("Soccer_Lab.tasks.direct.soccer_lab_marl.layout", SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "layout.py")
    _load_module("Soccer_Lab.tasks.direct.soccer_lab_marl.field_specs", SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "field_specs.py")
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_single_g1.g1_motion_policy",
        SRC_ROOT / "tasks" / "direct" / "soccer_single_g1" / "g1_motion_policy.py",
    )
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_lab_marl.soccer_lab_marl_env_cfg",
        SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "soccer_lab_marl_env_cfg.py",
    )
    return _load_module(
        "Soccer_Lab.tasks.direct.soccer_lab_marl.soccer_lab_marl_env",
        SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "soccer_lab_marl_env.py",
    )


class _DummyController:
    def __init__(self, default_angles, infer_target):
        self.reset_called = 0
        self.infer_called = 0
        self.default_angles = torch.tensor(default_angles, dtype=torch.float32)
        self._infer_target = torch.tensor(infer_target, dtype=torch.float32)

    def reset(self):
        self.reset_called += 1

    def infer(self, **kwargs):
        self.infer_called += 1
        return self._infer_target.clone()


class _DummyRobot:
    def __init__(self):
        self.data = types.SimpleNamespace(
            root_ang_vel_b=torch.zeros((1, 3), dtype=torch.float32),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
            joint_pos=torch.zeros((1, 5), dtype=torch.float32),
            joint_vel=torch.zeros((1, 5), dtype=torch.float32),
        )
        self.last_target = None

    def set_joint_position_target(self, target):
        self.last_target = target.clone()


def test_motion_policy_is_updated_once_per_agent_and_reused_in_apply_action():
    env_mod = _load_marl_env_module()
    env = env_mod.SoccerLabMarlEnv.__new__(env_mod.SoccerLabMarlEnv)
    env.cfg = types.SimpleNamespace(use_unitree_rl_policy=True)
    env.agent_names = ["a1", "b1"]
    env.control_joint_ids = [1, 3]
    env._upper_body_hold_joint_ids = [4]
    env._upper_body_hold_joint_pos = torch.tensor([9.0], dtype=torch.float32)
    env._default_joint_pos = {
        "a1": torch.zeros((1, 5), dtype=torch.float32),
        "b1": torch.zeros((1, 5), dtype=torch.float32),
    }
    env._joint_pos_target = {
        "a1": torch.full((1, 5), -100.0, dtype=torch.float32),
        "b1": torch.full((1, 5), -100.0, dtype=torch.float32),
    }
    env.robots = {"a1": _DummyRobot(), "b1": _DummyRobot()}
    env.actions = {
        "a1": torch.tensor([[123.0, 456.0]], dtype=torch.float32),
        "b1": torch.tensor([[789.0, 987.0]], dtype=torch.float32),
    }
    env._policy_leg_joint_targets = {
        "a1": torch.zeros((1, 2), dtype=torch.float32),
        "b1": torch.zeros((1, 2), dtype=torch.float32),
    }
    env._motion_controllers = {
        "a1": _DummyController(default_angles=[0.1, -0.1], infer_target=[1.5, -0.5]),
        "b1": _DummyController(default_angles=[0.2, -0.2], infer_target=[2.5, -1.5]),
    }

    env._pre_physics_step(
        {
            "a1": torch.tensor([[7.0, 8.0]], dtype=torch.float32),
            "b1": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        }
    )

    assert env._motion_controllers["a1"].infer_called == 1
    assert env._motion_controllers["b1"].infer_called == 1
    torch.testing.assert_close(env._policy_leg_joint_targets["a1"], torch.tensor([[1.5, -0.5]]))
    torch.testing.assert_close(env._policy_leg_joint_targets["b1"], torch.tensor([[2.5, -1.5]]))

    env._apply_action()

    assert env._motion_controllers["a1"].infer_called == 1
    assert env._motion_controllers["b1"].infer_called == 1
    torch.testing.assert_close(env.robots["a1"].last_target[0, env.control_joint_ids], torch.tensor([1.5, -0.5]))
    torch.testing.assert_close(env.robots["b1"].last_target[0, env.control_joint_ids], torch.tensor([2.5, -1.5]))
    torch.testing.assert_close(env.robots["a1"].last_target[0, env._upper_body_hold_joint_ids], torch.tensor([9.0]))
    torch.testing.assert_close(env.robots["b1"].last_target[0, env._upper_body_hold_joint_ids], torch.tensor([9.0]))


def test_reset_motion_policy_resets_all_agent_controllers_to_defaults():
    env_mod = _load_marl_env_module()
    env = env_mod.SoccerLabMarlEnv.__new__(env_mod.SoccerLabMarlEnv)
    env.cfg = types.SimpleNamespace(use_unitree_rl_policy=True)
    env.agent_names = ["a1", "b1"]
    env._policy_leg_joint_targets = {
        "a1": torch.zeros((1, 2), dtype=torch.float32),
        "b1": torch.zeros((1, 2), dtype=torch.float32),
    }
    env._motion_controllers = {
        "a1": _DummyController(default_angles=[1.5, -0.5], infer_target=[0.0, 0.0]),
        "b1": _DummyController(default_angles=[2.5, -1.5], infer_target=[0.0, 0.0]),
    }

    env._reset_motion_policy()

    assert env._motion_controllers["a1"].reset_called == 1
    assert env._motion_controllers["b1"].reset_called == 1
    torch.testing.assert_close(env._policy_leg_joint_targets["a1"], torch.tensor([[1.5, -0.5]]))
    torch.testing.assert_close(env._policy_leg_joint_targets["b1"], torch.tensor([[2.5, -1.5]]))
