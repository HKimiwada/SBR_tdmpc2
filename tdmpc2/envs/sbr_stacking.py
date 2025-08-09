# envs/sbr_stacking.py - Custom stacking environment loader for TD-MPC2
import sys
import os
from pathlib import Path

# Add the sbr_env directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'sbr_env'))

import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict

from sbr_stacking_env import make_simple_env
from envs.wrappers.timeout import Timeout


class StackingTensorWrapper(gym.Wrapper):
    """
    Wrapper for converting stacking environment to TD-MPC2 format
    """

    def __init__(self, env):
        super().__init__(env)
    
    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _try_f32_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.dtype == torch.float64:
                x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, task_idx=None):
        obs, info = self.env.reset()
        return self._obs_to_tensor(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action.numpy())
        done = terminated or truncated
        
        # Convert info to TD-MPC2 format
        info_dict = defaultdict(float, info)
        info_dict['success'] = float(info.get('success', False))
        info_dict['terminated'] = torch.tensor(float(terminated))
        
        return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info_dict


def make_env(cfg):
    """
    Make custom stacking environment for TD-MPC2.
    """
    if not cfg.task.startswith('stack-'):
        raise ValueError('Unknown task:', cfg.task)
    
    assert cfg.obs == 'state', 'Stacking environment only supports state observations.'
    
    # Extract task variant from cfg.task
    # e.g., 'stack-3-bricks' -> 'stack_3_bricks'
    task_variant = cfg.task.replace('-', '_')
    
    # Create the base environment
    env = make_simple_env(task_variant=task_variant, max_episode_steps=1500)
    
    # Wrap with TD-MPC2 compatible tensor wrapper
    env = StackingTensorWrapper(env)
    
    # Set max episode steps (TD-MPC2 expects this attribute)
    env.max_episode_steps = 1500
    
    return env