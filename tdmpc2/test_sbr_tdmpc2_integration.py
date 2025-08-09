# Testing integration of SBR Stacking Environment with TD-MPC2 
import os
import sys
import torch
import hydra
from omegaconf import OmegaConf

# Add current directory to path
sys.path.append('.')

def test_config_loading():
    """Test if sbr_config.yaml loads correctly"""
    print("🔧 Testing SBR Config Loading")
    print("=" * 50)
    
    try:
        # Test loading the custom config
        if os.path.exists('sbr_config.yaml'):
            cfg = OmegaConf.load('sbr_config.yaml')
            print(f"✅ sbr_config.yaml loaded successfully")
            print(f"   Task: {cfg.task}")
            print(f"   Model size: {cfg.model_size}")
            print(f"   Steps: {cfg.steps:,}")
            print(f"   Horizon: {cfg.horizon}")
            print(f"   WandB project: {cfg.wandb_project}")
            return True
        else:
            print("❌ sbr_config.yaml not found!")
            print("   Please create sbr_config.yaml in the tdmpc2 directory")
            return False
            
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_environment_loading():
    """Test if the custom stacking environment loads correctly"""
    print("\n🧪 Testing Environment Loading")
    print("=" * 50)
    
    try:
        from envs import make_env
        from common.parser import parse_cfg
        
        # Load your custom config
        if not os.path.exists('sbr_config.yaml'):
            print("❌ sbr_config.yaml not found - skipping environment test")
            return False
            
        cfg = OmegaConf.load('sbr_config.yaml')
        
        # Parse config
        cfg = parse_cfg(cfg)
        
        # Create environment
        env = make_env(cfg)
        print(f"✅ Environment created successfully")
        print(f"   Task: {cfg.task}")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Episode length: {cfg.episode_length}")
        
        # Test reset and step
        obs = env.reset()
        print(f"✅ Environment reset: obs shape = {obs.shape}")
        
        action = env.rand_act()
        obs, reward, done, info = env.step(action)
        print(f"✅ Environment step: reward = {reward:.4f}, done = {done}")
        
        env.close() if hasattr(env, 'close') else None
        return True
        
    except Exception as e:
        print(f"❌ Environment loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tdmpc2_agent():
    """Test if TD-MPC2 agent can be created with custom environment"""
    print("\n🤖 Testing TD-MPC2 Agent Creation")
    print("=" * 50)
    
    try:
        from envs import make_env
        from tdmpc2 import TDMPC2
        from common.parser import parse_cfg
        
        # Load your custom config
        if not os.path.exists('sbr_config.yaml'):
            print("❌ sbr_config.yaml not found - skipping agent test")
            return False
            
        cfg = OmegaConf.load('sbr_config.yaml')
        
        # Disable compile for testing
        cfg.compile = False
        
        # Parse config
        cfg = parse_cfg(cfg)
        
        # Create environment and agent
        env = make_env(cfg)
        agent = TDMPC2(cfg)
        
        print(f"✅ TD-MPC2 agent created successfully")
        print(f"   Model parameters: {agent.model.total_params:,}")
        print(f"   Device: {agent.device}")
        print(f"   Planning horizon: {cfg.horizon}")
        
        # Test agent action
        obs = env.reset()
        action = agent.act(obs, t0=True, eval_mode=True)
        print(f"✅ Agent action: shape = {action.shape}")
        
        # Test environment step with agent action
        obs, reward, done, info = env.step(action)
        print(f"✅ Environment + Agent: reward = {reward:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ TD-MPC2 agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """Test training components (buffer, logger, etc.)"""
    print("\n📦 Testing Training Components")
    print("=" * 50)
    
    try:
        from envs import make_env
        from tdmpc2 import TDMPC2
        from common.buffer import Buffer
        from common.parser import parse_cfg
        
        # Load your custom config
        if not os.path.exists('sbr_config.yaml'):
            print("❌ sbr_config.yaml not found - skipping training components test")
            return False
            
        cfg = OmegaConf.load('sbr_config.yaml')
        
        # Modify config for testing
        cfg.compile = False
        cfg.steps = 10000
        cfg.buffer_size = 10000
        cfg.batch_size = 64
        cfg.num_samples = 256
        cfg.horizon = 3
        
        cfg = parse_cfg(cfg)
        
        # Create components
        env = make_env(cfg)
        agent = TDMPC2(cfg)
        buffer = Buffer(cfg)
        
        print(f"✅ Buffer created: capacity = {buffer.capacity:,}")
        print(f"✅ All training components ready")
        print(f"   Experiment name: {cfg.exp_name}")
        print(f"   Work directory: {cfg.work_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("🚀 SBR TD-MPC2 Stacking Integration Test")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Config Loading", test_config_loading),
        ("Environment Loading", test_environment_loading),
        ("TD-MPC2 Agent Creation", test_tdmpc2_agent),
        ("Training Components", test_training_components),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 SBR INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your SBR stacking environment is fully integrated with TD-MPC2")
        print("\n🚀 Ready to train! Run:")
        print("   python train.py --config-name=sbr_config steps=50000 enable_wandb=false")
        print("   python train.py --config-name=sbr_config steps=500000  # Full training")
    else:
        print("\n💥 SOME TESTS FAILED!")
        print("❌ Please fix the issues above before training")
        
        if not results.get("Config Loading", False):
            print("\n📝 Next step: Create sbr_config.yaml file in the tdmpc2 directory")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)