# Code to create environment for SBR_Stacking task. 
# dm_control: stack_3_blocks -> Environment that inherits from gym.Env for StableBaselines3
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['DISPLAY'] = ':0'

import gymnasium as gym
from gymnasium import spaces
from dm_control import manipulation
import numpy as np
from typing import Dict, Any, Tuple, Optional

class SimpleStackingEnv(gym.Env):
    """
    Simple, clean StableBaselines3 compatible stacking environment
    WITH FIXED EPISODE LENGTH CONTROL
    """
    
    def __init__(self, task_variant='stack_3_bricks', max_episode_steps=1500):
        super().__init__()
        
        self.task_name = task_variant
        self.max_episode_steps = max_episode_steps
        self._episode_step = 0
        
        # Load dm_control environment
        try:
            env_name = f"{task_variant}_features"
            self.env = manipulation.load(env_name)
            print(f"✅ Loaded: {env_name}")
        except:
            self.env = manipulation.load(task_variant)
            print(f"✅ Loaded: {task_variant}")
        
        # IMPORTANT: Override dm_control's episode length
        # Set a very high internal limit so we control episode termination
        if hasattr(self.env, '_time_limit'):
            self.env._time_limit = max_episode_steps * 2  # Set much higher than our limit
            print(f"🔧 Override dm_control time limit to {self.env._time_limit}")
        
        # Try to access and modify the task's time limit if possible
        try:
            if hasattr(self.env, '_task') and hasattr(self.env._task, '_time_limit'):
                self.env._task._time_limit = max_episode_steps * 2
                print(f"🔧 Override task time limit to {self.env._task._time_limit}")
        except:
            print("⚠️  Could not access task time limit directly")
        
        # Set up spaces
        self._setup_spaces()
        
        self._current_time_step = None
        print(f"✅ SimpleStackingEnv ready with {max_episode_steps} max steps")
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Action space
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32
        )
        
        # Get sample observation to calculate flattened size
        temp_time_step = self.env.reset()
        flat_obs = self._flatten_observation(temp_time_step.observation)
        
        # Create flattened observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32
        )
        
        print(f"🔧 Action space: {self.action_space.shape}")
        print(f"🔧 Observation space: {self.observation_space.shape}")
    
    def _flatten_observation(self, dm_obs: Dict) -> np.ndarray:
        """Flatten dm_control observation dict into single array"""
        obs_parts = []
        
        # Sort keys for consistent ordering
        for key in sorted(dm_obs.keys()):
            value = dm_obs[key]
            if isinstance(value, np.ndarray):
                obs_parts.append(value.flatten())
            else:
                obs_parts.append(np.array([value], dtype=np.float32))
        
        # Concatenate all parts
        flattened = np.concatenate(obs_parts).astype(np.float32)
        return flattened
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self._episode_step = 0
        self._current_time_step = self.env.reset()
        
        # Get flattened observation
        obs = self._flatten_observation(self._current_time_step.observation)
        
        info = {
            'episode_step': self._episode_step,
            'success': False,
            'max_steps': self.max_episode_steps
        }
        
        return obs, info
    
    def step(self, action):
        """Step environment - FIXED to ignore dm_control termination"""
        self._current_time_step = self.env.step(action)
        self._episode_step += 1
        
        # Get flattened observation
        obs = self._flatten_observation(self._current_time_step.observation)
        
        reward = float(self._current_time_step.reward)
        
        # CRITICAL FIX: Ignore dm_control's termination, use only our logic
        # Only terminate early if we detect task success (high reward)
        task_success = reward > 0.8  # High reward indicates successful stacking
        
        # Use our own termination logic
        terminated = task_success  # Only terminate on clear success
        truncated = self._episode_step >= self.max_episode_steps  # Truncate at our limit
        
        # Calculate success based on reward threshold
        success = reward > 0.5
        
        info = {
            'episode_step': self._episode_step,
            'success': success,
            'task_success': task_success,
            'discount': float(self._current_time_step.discount),
            'dm_control_last': self._current_time_step.last(),  # For debugging
            'max_steps': self.max_episode_steps
        }
        
        # Debug info every 100 steps
        if self._episode_step % 100 == 0:
            print(f"Step {self._episode_step}/{self.max_episode_steps}: reward={reward:.3f}, success={success}")
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        """Render environment"""
        try:
            return self.env.physics.render(height=480, width=640, camera_id=0)
        except:
            return None
    
    def close(self):
        """Close environment"""
        pass


# Simple factory function
def make_simple_env(task_variant='stack_3_bricks', max_episode_steps=1500):
    """Create simple stacking environment with proper episode length"""
    return SimpleStackingEnv(task_variant=task_variant, max_episode_steps=max_episode_steps)


# Test the fixed environment
if __name__ == "__main__":
    print("🧪 Testing Fixed Long Episode Environment")
    print("=" * 50)
    
    # Create environment with long episodes
    env = make_simple_env(max_episode_steps=1500)
    
    # Test full episode
    obs, info = env.reset()
    print(f"Reset: obs shape = {obs.shape}, max_steps = {info['max_steps']}")
    
    # Run for many steps to test episode length
    total_reward = 0
    for step in range(100):  # Test first 100 steps
        action = env.action_space.sample() * 0.1  # Small actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 25 == 0:
            print(f"Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            print(f"Episode ended at step {info['episode_step']}")
            break
    
    print(f"Total reward after {step+1} steps: {total_reward:.3f}")
    print(f"Episode should continue until step {env.max_episode_steps}")
    
    env.close()
    print("✅ Fixed environment test completed!")