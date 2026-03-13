from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TEST_ROOT / "Soccer_Lab"
LAYOUT_PATH = SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl" / "layout.py"


def _ensure_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_single_g1_spawn_pose_reuses_a4_reference_position():
    _ensure_package("Soccer_Lab", SRC_ROOT)
    _ensure_package("Soccer_Lab.tasks", SRC_ROOT / "tasks")
    _ensure_package("Soccer_Lab.tasks.direct", SRC_ROOT / "tasks" / "direct")
    _ensure_package(
        "Soccer_Lab.tasks.direct.soccer_lab_marl",
        SRC_ROOT / "tasks" / "direct" / "soccer_lab_marl",
    )

    layout = _load_module("Soccer_Lab.tasks.direct.soccer_lab_marl.layout", LAYOUT_PATH)

    team_poses = layout.compute_g1_team_poses(field_size=(9.0, 6.0), team_spacing=(2.0, 3.0), base_height=0.78)
    single_pose = layout.compute_single_g1_spawn_pose(
        field_size=(9.0, 6.0),
        team_spacing=(2.0, 3.0),
        base_height=0.78,
        spawn_reference_agent="a4",
        single_agent_name="a1",
    )

    assert single_pose == {"a1": team_poses["a4"]}
