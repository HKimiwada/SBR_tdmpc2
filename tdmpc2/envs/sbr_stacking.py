# envs/sbr_stacking.py - Fixed custom stacking environment loader for TD-MPC2
import sys
import os
from pathlib import Path
import warnings

# Suppress warnings early
warnings.filterwarnings('ignore')

# Set environment variables before any imports
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict

# Add the sbr_env directory to the path
current_dir = Path(__file__).parent.parent
sbr_env_path = current_dir / 'sbr_env'

# Insert at the beginning to avoid conflicts
if str(sbr_env_path) not in sys.path:
    sys.path.insert(0, str(sbr_env_path))

print(f"🔍 Looking for sbr_stacking_env in: {sbr_env_path}")

try:
    from sbr_stacking_env import make_simple_env
    print("✅ Successfully imported sbr_stacking_env")
except ImportError as e:
    print(f"❌ Could not import sbr_stacking_env: {e}")
    print(f"   Searched in: {sbr_env_path}")
    print(f"   Files in directory: {list(sbr_env_path.glob('*.py')) if sbr_env_path.exists() else 'Directory not found'}")
    raise
except Exception as e:
    print(f"❌ Unexpected error importing sbr_stacking_env: {e}")
    raise

# Import timeout wrapper with error handling
try:
    from envs.wrappers.timeout import Timeout
    print("✅ Successfully imported Timeout wrapper")
except ImportError:
    # If we can't import from envs.wrappers, try relative import
    try:
        from .wrappers.timeout import Timeout
        print("✅ Successfully imported Timeout wrapper (relative)")
    except ImportError:
        print("⚠️  Could not import Timeout wrapper - will create a simple version")
        
        class Timeout(gym.Wrapper):
            """Simple timeout wrapper"""
            def __init__(self, env, max_episode_steps):
                super().__init__(env)
                self._max_episode_steps = max_episode_steps
            
            @property
            def max_episode_steps(self):
                return self._max_episode_steps
            
            def reset(self, **kwargs):
                self._t = 0
                return self.env.reset(**kwargs)
            
            def step(self, action):
                obs, reward, done, info = self.env.step(action)
                self._t += 1
                done = done or self._t >= self.max_episode_steps
                return obs, reward, done, info


class StackingTensorWrapper(gym.Wrapper):
    """
    Wrapper for converting stacking environment to TD-MPC2 format
    """

    def __init__(self, env):
        super().__init__(env)
        print(f"🔧 Wrapping environment with StackingTensorWrapper")
    
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
        result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        return self._obs_to_tensor(obs)

    def step(self, action):
        # Convert tensor to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        
        obs, reward, terminated, truncated, info = self.env.step(action)
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
        raise ValueError(f'Unknown task: {cfg.task}. Expected task starting with "stack-"')
    
    assert cfg.obs == 'state', f'Stacking environment only supports state observations, got: {cfg.obs}'
    
    # Extract task variant from cfg.task
    # e.g., 'stack-3-bricks' -> 'stack_3_bricks'
    task_variant = cfg.task.replace('-', '_')
    
    print(f"🧱 Creating stacking environment: {task_variant}")
    
    try:
        # Create the base environment with proper episode length
        max_episode_steps = getattr(cfg, 'episode_length', 1500)
        env = make_simple_env(task_variant=task_variant, max_episode_steps=max_episode_steps)
        
        # Wrap with TD-MPC2 compatible tensor wrapper
        env = StackingTensorWrapper(env)
        
        # Set max episode steps (TD-MPC2 expects this attribute)
        env.max_episode_steps = max_episode_steps
        
        print(f"✅ Stacking environment created successfully")
        print(f"   Task variant: {task_variant}")
        print(f"   Max episode steps: {max_episode_steps}")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        return env
        
    except Exception as e:
        print(f"❌ Failed to create stacking environment: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_make_env():
    """Test the make_env function"""
    print("🧪 Testing make_env function")
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self):
            self.task = 'stack-3-bricks'
            self.obs = 'state'
            self.episode_length = 100
    
    try:
        cfg = SimpleConfig()
        env = make_env(cfg)
        
        print("✅ Environment created via make_env")
        
        # Test basic functionality
        obs = env.reset()
        print(f"✅ Reset: obs shape = {obs.shape}")
        
        action = env.rand_act()
        obs, reward, done, info = env.step(action)
        print(f"✅ Step: reward = {reward:.4f}, done = {done}")
        
        if hasattr(env, 'close'):
            env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ make_env test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Only run test if this file is executed directly
    print("🚀 Testing SBR Stacking Environment Loader")
    print("=" * 60)
    
    success = test_make_env()
    
    if success:
        print("\n✅ All tests passed! Environment loader is working correctly.")
    else:
        print("\n❌ Tests failed! Please check the error messages above.")
    
    sys.exit(0 if success else 1)