# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Soccer_Lab.tasks  # noqa: F401


def _sample_from_space(space: gym.Space, num_envs: int, device: str) -> torch.Tensor:
    if not hasattr(space, "shape") or space.shape is None:
        raise TypeError(f"Unsupported action space type for random agent: {type(space)}")

    tensor_shape = (num_envs, *tuple(space.shape))
    if getattr(space, "dtype", None) is not None and np.issubdtype(space.dtype, np.integer):
        return torch.zeros(tensor_shape, device=device, dtype=torch.long)
    return 2 * torch.rand(tensor_shape, device=device) - 1


def _build_random_actions(env) -> torch.Tensor | dict[str, torch.Tensor]:
    unwrapped_env = env.unwrapped
    num_envs = getattr(unwrapped_env, "num_envs", 1)
    device = unwrapped_env.device

    if hasattr(unwrapped_env, "possible_agents") and hasattr(unwrapped_env, "action_spaces"):
        return {
            agent: _sample_from_space(unwrapped_env.action_spaces[agent], num_envs=num_envs, device=device)
            for agent in unwrapped_env.possible_agents
        }
    return _sample_from_space(env.action_space, num_envs=num_envs, device=device)


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    if hasattr(env.unwrapped, "possible_agents") and hasattr(env.unwrapped, "action_spaces"):
        print(f"[INFO]: MARL observation spaces: {env.unwrapped.observation_spaces}")
        print(f"[INFO]: MARL action spaces: {env.unwrapped.action_spaces}")
    else:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = _build_random_actions(env)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close sim app even if stepping fails
        simulation_app.close()
