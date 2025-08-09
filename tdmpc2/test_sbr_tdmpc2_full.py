# test_sbr_tdmpc2_integration_fixed.py - Complete TD-MPC2 + SBR Integration Test
import os
import sys
import torch
from omegaconf import OmegaConf
from pathlib import Path
import warnings

# Set environment variables for headless mode
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
os.environ['DISPLAY'] = os.getenv('DISPLAY', ':99')

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the test-friendly config parser
try:
    from test_config_parser import parse_cfg_for_testing
    print("✅ Test config parser imported")
except ImportError as e:
    print(f"❌ Could not import test_config_parser: {e}")
    print("   Please ensure test_config_parser.py is in the current directory")
    sys.exit(1)


def test_config_loading():
    """Test if sbr_config.yaml loads correctly"""
    print("🔧 Testing SBR Config Loading")
    print("=" * 50)
    
    try:
        # Test loading the custom config
        config_path = 'sbr_config.yaml'
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            print(f"✅ {config_path} loaded successfully")
            print(f"   Task: {cfg.task}")
            print(f"   Model size: {cfg.get('model_size', 'default')}")
            print(f"   Steps: {cfg.steps:,}")
            print(f"   Horizon: {cfg.get('horizon', 'default')}")
            print(f"   WandB project: {cfg.get('wandb_project', 'not set')}")
            
            # Test parsing
            parsed_cfg = parse_cfg_for_testing(cfg)
            print(f"✅ Config parsed successfully")
            print(f"   Work dir: {parsed_cfg.work_dir}")
            print(f"   Task title: {parsed_cfg.task_title}")
            
            return True
        else:
            print(f"❌ {config_path} not found!")
            print("   Please create sbr_config.yaml in the tdmpc2 directory")
            return False
            
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Test if all required dependencies are available"""
    print("\n📦 Testing Dependencies")
    print("=" * 50)
    
    dependencies_ok = True
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("   ⚠️  CUDA not available - will use CPU")
    except ImportError:
        print("❌ PyTorch not available")
        dependencies_ok = False
    
    # Test dm_control
    try:
        from dm_control import manipulation
        print("✅ dm_control.manipulation")
    except ImportError:
        print("❌ dm_control.manipulation not available")
        print("   Install with: pip install 'dm_control[manipulation]'")
        dependencies_ok = False
    
    # Test gymnasium
    try:
        import gymnasium as gym
        print("✅ gymnasium")
    except ImportError:
        print("❌ gymnasium not available")
        dependencies_ok = False
    
    # Test hydra
    try:
        import hydra
        print("✅ hydra-core")
    except ImportError:
        print("❌ hydra-core not available")
        dependencies_ok = False
    
    # Test omegaconf
    try:
        from omegaconf import OmegaConf
        print("✅ omegaconf")
    except ImportError:
        print("❌ omegaconf not available")
        dependencies_ok = False
    
    return dependencies_ok


def test_file_structure():
    """Test if all required files are in the right place"""
    print("\n📁 Testing File Structure")
    print("=" * 50)
    
    required_files = [
        'sbr_env/sbr_stacking_env.py',
        'envs/sbr_stacking.py',
        'sbr_config.yaml',
        'test_config_parser.py'
    ]
    
    optional_files = [
        'train_stacking.py',
        'test_stacking_simple.py'
    ]
    
    all_required_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING (REQUIRED)")
            all_required_exist = False
    
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} (optional)")
        else:
            print(f"⚠️  {file_path} - missing (optional)")
    
    return all_required_exist


def test_stacking_environment():
    """Test the stacking environment directly"""
    print("\n🧱 Testing Stacking Environment")
    print("=" * 50)
    
    try:
        # Test direct import
        sys.path.insert(0, './sbr_env')
        from sbr_stacking_env import make_simple_env
        print("✅ Successfully imported make_simple_env")
        
        # Create environment
        env = make_simple_env('stack_3_bricks', max_episode_steps=100)
        print("✅ Environment created")
        
        # Test basic functionality
        obs, info = env.reset()
        print(f"✅ Reset: obs shape = {obs.shape}")
        
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step: reward = {reward:.4f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Stacking environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_loading():
    """Test if the custom stacking environment loads correctly via TD-MPC2"""
    print("\n🧪 Testing Environment Loading via TD-MPC2")
    print("=" * 50)
    
    try:
        from envs import make_env
        
        # Load your custom config
        if not os.path.exists('sbr_config.yaml'):
            print("❌ sbr_config.yaml not found - skipping environment test")
            return False
            
        cfg = OmegaConf.load('sbr_config.yaml')
        
        # Parse config with test-friendly parser
        cfg = parse_cfg_for_testing(cfg)
        
        # Create environment
        env = make_env(cfg)
        print(f"✅ Environment created successfully via make_env")
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
        
        # Close environment if it has a close method
        if hasattr(env, 'close'):
            env.close()
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
        
        # Load your custom config
        if not os.path.exists('sbr_config.yaml'):
            print("❌ sbr_config.yaml not found - skipping agent test")
            return False
            
        cfg = OmegaConf.load('sbr_config.yaml')
        
        # Disable compile for testing and reduce complexity
        cfg.compile = False
        cfg.steps = 1000  # Reduce for testing
        cfg.num_samples = 128  # Reduce for testing
        cfg.enable_wandb = False  # Disable wandb for testing
        
        # Parse config with test-friendly parser
        cfg = parse_cfg_for_testing(cfg)
        
        # Create environment and agent
        print("Creating environment...")
        env = make_env(cfg)
        
        print("Creating TD-MPC2 agent...")
        agent = TDMPC2(cfg)
        
        print(f"✅ TD-MPC2 agent created successfully")
        print(f"   Model parameters: {agent.model.total_params:,}")
        print(f"   Device: {agent.device}")
        print(f"   Planning horizon: {cfg.horizon}")
        
        # Test agent action
        obs = env.reset()
        print("Testing agent action...")
        action = agent.act(obs, t0=True, eval_mode=True)
        print(f"✅ Agent action: shape = {action.shape}")
        
        # Test environment step with agent action
        obs, reward, done, info = env.step(action)
        print(f"✅ Environment + Agent: reward = {reward:.4f}")
        
        # Clean up
        if hasattr(env, 'close'):
            env.close()
        del agent  # Free up GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
        cfg.enable_wandb = False  # Disable wandb for testing
        
        cfg = parse_cfg_for_testing(cfg)
        
        # Create components
        print("Creating environment...")
        env = make_env(cfg)
        
        print("Creating agent...")
        agent = TDMPC2(cfg)
        
        print("Creating buffer...")
        buffer = Buffer(cfg)
        
        print(f"✅ Buffer created: capacity = {buffer.capacity:,}")
        print(f"✅ All training components ready")
        print(f"   Experiment name: {cfg.exp_name}")
        print(f"   Work directory: {cfg.work_dir}")
        
        # Test buffer functionality
        print("Testing buffer functionality...")
        obs = env.reset()
        action = env.rand_act()
        obs, reward, done, info = env.step(action)
        
        # Create a simple episode for buffer
        from tensordict import TensorDict
        td = TensorDict({
            'obs': obs.unsqueeze(0).unsqueeze(0),
            'action': action.unsqueeze(0).unsqueeze(0),
            'reward': torch.tensor([[reward]]),
            'terminated': torch.tensor([[float(info['terminated'])]]),
        }, batch_size=(1, 1))
        
        buffer.add(td)
        print(f"✅ Buffer test: added episode, now has {buffer.num_eps} episodes")
        
        # Close environment if it has a close method
        if hasattr(env, 'close'):
            env.close()
        
        # Clean up
        del agent
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_compatibility():
    """Test config compatibility with TD-MPC2"""
    print("\n⚙️  Testing Config Compatibility")
    print("=" * 50)
    
    try:
        if not os.path.exists('sbr_config.yaml'):
            print("❌ sbr_config.yaml not found")
            return False
        
        cfg = OmegaConf.load('sbr_config.yaml')
        parsed_cfg = parse_cfg_for_testing(cfg)
        
        # Check required fields
        required_fields = [
            'task', 'obs', 'steps', 'batch_size', 'horizon',
            'model_size', 'latent_dim', 'action_dim', 'episode_length'
        ]
        
        missing_fields = []
        for field in required_fields:
            if not hasattr(parsed_cfg, field):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required config fields: {missing_fields}")
            return False
        
        print("✅ All required config fields present")
        print(f"   Task: {parsed_cfg.task}")
        print(f"   Model size: {parsed_cfg.model_size}")
        print(f"   Episode length: {parsed_cfg.episode_length}")
        print(f"   Latent dim: {parsed_cfg.latent_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config compatibility test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("🚀 SBR TD-MPC2 Stacking Integration Test (Complete)")
    print("=" * 70)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Run tests in order of dependency
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Config Loading", test_config_loading),
        ("Config Compatibility", test_config_compatibility),
        ("Stacking Environment", test_stacking_environment),
        ("Environment Loading", test_environment_loading),
        ("TD-MPC2 Agent Creation", test_tdmpc2_agent),
        ("Training Components", test_training_components),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        results[test_name] = test_func()
        print()
        
        # Stop early if critical tests fail
        if test_name in ["File Structure", "Dependencies"] and not results[test_name]:
            print(f"💥 Critical test '{test_name}' failed - stopping early")
            break
    
    # Summary
    print("=" * 70)
    print("🎯 SBR INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    critical_failed = False
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:<30} {status}")
        if not passed:
            all_passed = False
            if test_name in ["File Structure", "Dependencies"]:
                critical_failed = True
    
    # Final recommendations
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your SBR stacking environment is fully integrated with TD-MPC2")
        print("\n🚀 Ready to train! Next steps:")
        print("   1. Quick test:     python train_stacking.py test steps=100 compile=false")
        print("   2. Short training: python train_stacking.py steps=5000 compile=false enable_wandb=false")
        print("   3. Full training:  python train_stacking.py steps=500000")
        print("   4. Alternative:    python train.py --config-name=sbr_config steps=50000")
    elif critical_failed:
        print("💥 CRITICAL TESTS FAILED!")
        print("❌ Please fix file structure and dependencies before proceeding")
        
        if not results.get("Dependencies", False):
            print("\n📦 Install missing dependencies:")
            print("   pip install 'dm_control[manipulation]'")
            print("   pip install torch gymnasium hydra-core omegaconf")
        
        if not results.get("File Structure", False):
            print("\n📁 Missing required files - please ensure all files are in place")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("❌ Please address the issues above before training")
        print("\n🔧 Common issues:")
        print("   - Missing dm_control[manipulation]: pip install 'dm_control[manipulation]'")
        print("   - CUDA issues: Check GPU drivers and PyTorch CUDA compatibility")
        print("   - Import errors: Ensure all files are in correct locations")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)