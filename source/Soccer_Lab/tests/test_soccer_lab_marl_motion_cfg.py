from __future__ import annotations

import copy
import importlib.util
import math
import sys
import types
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TEST_ROOT / "Soccer_Lab"
POLICY_MODULE_PATH = SRC_ROOT / "tasks" / "direct" / "soccer_single_g1" / "g1_motion_policy.py"
CFG_MODULE_PATH = SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "soccer_lab_marl_env_cfg.py"
POLICY_PATH = str(TEST_ROOT.parents[1] / "assets" / "g1_comp" / "policies" / "g1_motion.pt")


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
    sys.modules["isaaclab.assets"] = assets_mod

    envs_mod = types.ModuleType("isaaclab.envs")
    envs_mod.DirectMARLEnvCfg = _DirectMARLEnvCfg
    envs_mod.ViewerCfg = _ViewerCfg
    sys.modules["isaaclab.envs"] = envs_mod

    scene_mod = types.ModuleType("isaaclab.scene")
    scene_mod.InteractiveSceneCfg = _InteractiveSceneCfg
    sys.modules["isaaclab.scene"] = scene_mod

    sensors_mod = types.ModuleType("isaaclab.sensors")
    sensors_mod.CameraCfg = _CameraCfg
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


def test_soccer_lab_marl_cfg_enables_shared_motion_policy_defaults():
    _install_isaaclab_stubs()
    _install_project_packages()

    _load_module("Soccer_Lab.g1_asset", SRC_ROOT / "g1_asset.py")
    _load_module("Soccer_Lab.tasks.direct.soccer_lab_marl.field_specs", SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "field_specs.py")
    _load_module("Soccer_Lab.tasks.direct.soccer_lab_marl.layout", SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "layout.py")
    policy_mod = _load_module("Soccer_Lab.tasks.direct.soccer_single_g1.g1_motion_policy", POLICY_MODULE_PATH)
    cfg_mod = _load_module("Soccer_Lab.tasks.direct.soccer_lab_marl.soccer_lab_marl_env_cfg", CFG_MODULE_PATH)

    cfg = cfg_mod.SoccerLabMarlEnvCfg

    assert cfg.controlled_joint_names == list(policy_mod.G1_POLICY_JOINT_NAMES)
    assert cfg.use_unitree_rl_policy is True
    assert cfg.unitree_rl_policy_path == POLICY_PATH
    assert math.isclose(cfg.unitree_rl_control_dt, 0.02)
    assert cfg.decimation == 4
    assert math.isclose(cfg.sim.dt, 1.0 / 200.0)
    assert tuple(cfg.unitree_rl_default_angles) == policy_mod.G1_POLICY_DEFAULT_ANGLES
    assert tuple(cfg.unitree_rl_kp) == policy_mod.G1_POLICY_KP
    assert tuple(cfg.unitree_rl_kd) == policy_mod.G1_POLICY_KD
    assert cfg.unitree_rl_cmd == (0.0, 0.0, 0.0)
    assert cfg.upper_body_hold_joint_pos["left_shoulder_pitch_joint"] == 0.35

    leg_actuator = cfg.g1_robot_cfg.actuators["policy_legs"]
    assert leg_actuator.joint_names_expr == list(policy_mod.G1_POLICY_JOINT_NAMES)
    assert cfg.g1_robot_cfg.init_state.joint_pos["left_knee_joint"] == policy_mod.G1_POLICY_DEFAULT_ANGLES[3]
