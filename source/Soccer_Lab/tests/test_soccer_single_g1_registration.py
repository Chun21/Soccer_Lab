from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import registry


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

    isaaclab_tasks_pkg = types.ModuleType("isaaclab_tasks")
    isaaclab_tasks_pkg.__path__ = []
    sys.modules["isaaclab_tasks"] = isaaclab_tasks_pkg

    isaaclab_tasks_utils_mod = types.ModuleType("isaaclab_tasks.utils")
    isaaclab_tasks_utils_mod.import_packages = lambda *args, **kwargs: None
    sys.modules["isaaclab_tasks.utils"] = isaaclab_tasks_utils_mod


def _install_project_packages() -> None:
    _ensure_package("Soccer_Lab", SRC_ROOT)
    _ensure_package("Soccer_Lab.tasks", SRC_ROOT / "tasks")
    _ensure_package("Soccer_Lab.tasks.direct", SRC_ROOT / "tasks" / "direct")
    _ensure_package("Soccer_Lab.tasks.direct.soccer_lab_marl", SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl")
    _ensure_package("Soccer_Lab.tasks.direct.soccer_single_g1", SRC_ROOT / "tasks" / "direct" / "soccer_single_g1")


def test_soccer_single_g1_task_registration_and_single_agent_cfg():
    env_id = "SoccerLab-G1-Soccer-Single-Direct-v0"
    registry.pop(env_id, None)

    _install_isaaclab_stubs()
    _install_project_packages()

    _load_module("Soccer_Lab.g1_asset", SRC_ROOT / "g1_asset.py")
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_lab_marl.layout",
        SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "layout.py",
    )
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_lab_marl.field_specs",
        SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "field_specs.py",
    )
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_lab_marl.soccer_lab_marl_env_cfg",
        SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "soccer_lab_marl_env_cfg.py",
    )
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_lab_marl.soccer_lab_marl_env",
        SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "soccer_lab_marl_env.py",
    )
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_lab_marl.agents",
        SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "agents" / "__init__.py",
        is_package=True,
    )
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_lab_marl",
        SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "__init__.py",
        is_package=True,
    )

    cfg_mod = _load_module(
        "Soccer_Lab.tasks.direct.soccer_single_g1.soccer_single_g1_env_cfg",
        SRC_ROOT / "tasks" / "direct" / "soccer_single_g1" / "soccer_single_g1_env_cfg.py",
    )
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_single_g1.soccer_single_g1_env",
        SRC_ROOT / "tasks" / "direct" / "soccer_single_g1" / "soccer_single_g1_env.py",
    )
    _load_module(
        "Soccer_Lab.tasks.direct.soccer_single_g1",
        SRC_ROOT / "tasks" / "direct" / "soccer_single_g1" / "__init__.py",
        is_package=True,
    )

    spec = gym.spec(env_id)
    assert spec.entry_point == "Soccer_Lab.tasks.direct.soccer_single_g1.soccer_single_g1_env:SoccerSingleG1Env"
    assert spec.kwargs["env_cfg_entry_point"] == (
        "Soccer_Lab.tasks.direct.soccer_single_g1.soccer_single_g1_env_cfg:SoccerSingleG1EnvCfg"
    )
    assert cfg_mod.SoccerSingleG1EnvCfg.possible_agents == ["a1"]
    assert cfg_mod.SoccerSingleG1EnvCfg.single_spawn_reference == "a4"
