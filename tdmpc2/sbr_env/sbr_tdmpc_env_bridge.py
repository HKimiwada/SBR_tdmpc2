# custom_stacking_bridge.py - Bridge your environment to TD-MPC2
from sbr_stacking_env import make_simple_env
import numpy as np

class TD_MPC2_StackingBridge:
    """Bridge your stacking environment to TD-MPC2 format"""
    
    def __init__(self, max_episode_steps=1500):
        self.env = make_simple_env('stack_3_bricks', max_episode_steps=max_episode_steps)
        self._max_episode_steps = max_episode_steps
        
    def reset(self):
        """Reset environment - TD-MPC2 style"""
        obs, info = self.env.reset()
        # TD-MPC2 expects just observation, not (obs, info)
        return obs
    
    def step(self, action):
        """Step environment - TD-MPC2 style"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # TD-MPC2 expects (obs, reward, done, info) format
        done = terminated or truncated
        
        # Convert info to TD-MPC2 format
        td_info = {
            'success': info.get('success', False),
            'episode_step': info.get('episode_step', 0),
            'discount': info.get('discount', 0.99)
        }
        
        return obs, reward, done, td_info
    
    def observation_spec(self):
        """Get observation specification"""
        class ObsSpec:
            def __init__(self, shape, dtype=np.float32):
                self.shape = shape
                self.dtype = dtype
                
        return ObsSpec(self.env.observation_space.shape)
    
    def action_spec(self):
        """Get action specification"""
        class ActionSpec:
            def __init__(self, space):
                self.shape = space.shape
                self.minimum = space.low
                self.maximum = space.high
                self.dtype = space.dtype
                
        return ActionSpec(self.env.action_space)
    
    def render(self, mode='rgb_array'):
        """Render environment"""
        return self.env.render(mode=mode)
    
    def close(self):
        """Close environment"""
        self.env.close()

def make_custom_stacking_env():
    """Factory function for TD-MPC2"""
    return TD_MPC2_StackingBridge(max_episode_steps=1500)

# Test the bridge
if __name__ == "__main__":
    print("🧪 Testing SBR-TDMPC2 Environment Bridge")
    
    env = make_custom_stacking_env()
    
    # Test basic functionality
    obs = env.reset()
    print(f"✅ Reset: obs shape = {obs.shape}")
    
    action = np.random.uniform(
        env.action_spec().minimum,
        env.action_spec().maximum,
        env.action_spec().shape
    ) * 0.1
    
    obs, reward, done, info = env.step(action)
    print(f"✅ Step: obs shape = {obs.shape}, reward = {reward:.3f}")
    print(f"   Action spec: {env.action_spec().shape}")
    print(f"   Obs spec: {env.observation_spec().shape}")
    
    env.close()
    print("🎉 Bridge working perfectly!")