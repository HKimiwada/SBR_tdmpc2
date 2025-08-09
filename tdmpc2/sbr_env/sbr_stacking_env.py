# Code to create environment for SBR_Stacking task. 
# dm_control: stack_3_blocks -> Environment that inherits from gym.Env for StableBaselines3
# May have to change code to create environment with scaled rewards (current rewards are too small for proper gradient updates)
# sbr_env/sbr_stacking_env.py - Fixed version with proper imports
import os
import sys
import warnings

# Set environment variables before any mujoco imports
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

# Suppress warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Import dm_control with error handling
try:
    from dm_control import manipulation
    print("✅ dm_control.manipulation imported successfully")
except ImportError as e:
    print(f"❌ Could not import dm_control.manipulation: {e}")
    print("   Please install with: pip install dm_control[manipulation]")
    raise
except Exception as e:
    print(f"❌ Error importing dm_control: {e}")
    print("   This might be due to missing dependencies or display issues")
    raise


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
        
        print(f"🧱 Creating {task_variant} environment...")
        
        # Load dm_control environment with error handling
        try:
            # Try different task name formats
            possible_names = [
                f"{task_variant}_features",
                task_variant,
                task_variant.replace('_', '-'),
                f"{task_variant.replace('_', '-')}_features"
            ]
            
            self.env = None
            for name in possible_names:
                try:
                    self.env = manipulation.load(name)
                    print(f"✅ Loaded: {name}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load {name}: {str(e)[:50]}...")
                    continue
            
            if self.env is None:
                raise ValueError(f"Could not load any variant of {task_variant}")
                
        except Exception as e:
            print(f"❌ Failed to create dm_control environment: {e}")
            raise
        
        # IMPORTANT: Override dm_control's episode length
        # Set a very high internal limit so we control episode termination
        if hasattr(self.env, '_time_limit'):
            self.env._time_limit = max_episode_steps * 10  # Set much higher than our limit
            print(f"🔧 Override dm_control time limit to {self.env._time_limit}")
        
        # Try to access and modify the task's time limit if possible
        try:
            if hasattr(self.env, '_task') and hasattr(self.env._task, '_time_limit'):
                self.env._task._time_limit = max_episode_steps * 10
                print(f"🔧 Override task time limit to {self.env._task._time_limit}")
        except Exception as e:
            print(f"⚠️  Could not access task time limit: {e}")
        
        # Set up spaces
        self._setup_spaces()
        
        self._current_time_step = None
        print(f"✅ SimpleStackingEnv ready with {max_episode_steps} max steps")
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        try:
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
            
        except Exception as e:
            print(f"❌ Error setting up spaces: {e}")
            raise
    
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
        
        # Debug info every 200 steps
        if self._episode_step % 200 == 0:
            print(f"Step {self._episode_step}/{self.max_episode_steps}: reward={reward:.3f}, success={success}")
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        """Render environment"""
        try:
            return self.env.physics.render(height=480, width=640, camera_id=0)
        except Exception as e:
            print(f"Warning: Could not render: {e}")
            return None
    
    def close(self):
        """Close environment"""
        if hasattr(self.env, 'close'):
            self.env.close()


def make_simple_env(task_variant='stack_3_bricks', max_episode_steps=1500):
    """Create simple stacking environment with proper episode length"""
    print(f"🏭 Creating simple stacking environment: {task_variant}")
    return SimpleStackingEnv(task_variant=task_variant, max_episode_steps=max_episode_steps)


def test_environment():
    """Test the environment creation and basic functionality"""
    print("🧪 Testing Stacking Environment")
    print("=" * 50)
    
    try:
        # Test environment creation
        env = make_simple_env('stack_3_bricks', max_episode_steps=100)
        print("✅ Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful: obs shape = {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample() * 0.1  # Small random actions
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: reward = {reward:.4f}, terminated = {terminated}, truncated = {truncated}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("✅ Environment test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test the environment if run directly
if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)