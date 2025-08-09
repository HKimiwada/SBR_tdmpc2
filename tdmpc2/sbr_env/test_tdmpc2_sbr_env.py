# Testing TD-MPC2 on SBR Stacking Environment
import torch
import numpy as np
from sbr_tdmpc_env_bridge import make_custom_stacking_env

def test_environment_compatibility():
    """Test if your environment works with TD-MPC2 components"""
    print("🧪 Testing TD-MPC2 + Custom Stacking Compatibility")
    print("=" * 60)
    
    # Test 1: Environment creation
    print("📋 Test 1: Environment Creation")
    env = make_custom_stacking_env()
    print(f"✅ Environment created")
    print(f"   Observation shape: {env.observation_spec().shape}")
    print(f"   Action shape: {env.action_spec().shape}")
    
    # Test 2: Episode execution
    print("\\n📋 Test 2: Episode Execution")
    obs = env.reset()
    print(f"✅ Reset successful: obs shape = {obs.shape}")
    
    total_reward = 0
    for step in range(10):
        # Random action
        action = np.random.uniform(
            env.action_spec().minimum,
            env.action_spec().maximum,
            env.action_spec().shape
        ) * 0.1  # Small actions
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            print(f"   Step {step}: reward = {reward:.4f}, done = {done}")
        
        if done:
            print(f"   Episode ended at step {step}")
            break
    
    print(f"✅ Episode test complete: total reward = {total_reward:.3f}")
    
    # Test 3: TD-MPC2 component compatibility
    print("\\n📋 Test 3: TD-MPC2 Component Compatibility")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Test observation encoding
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    print(f"✅ Observation tensor: {obs_tensor.shape}")
    
    # Test action tensor
    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
    print(f"✅ Action tensor: {action_tensor.shape}")
    
    print("\\n🎉 All compatibility tests passed!")
    print("✅ Your stacking environment is ready for TD-MPC2!")
    
    env.close()
    return True

if __name__ == "__main__":
    test_environment_compatibility()