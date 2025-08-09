# train_stacking.py - Complete training script for SBR stacking environment
import os
import sys
import warnings
from pathlib import Path

# Set environment variables before any imports
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"

# Set display for headless operation
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

# Suppress warnings
warnings.filterwarnings('ignore')

import torch
import hydra
import numpy as np
from termcolor import colored
from omegaconf import OmegaConf

# Ensure proper imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def print_banner(text, color='yellow', char='='):
    """Print a colored banner"""
    line = char * len(text)
    print(colored(line, color, attrs=['bold']))
    print(colored(text, color, attrs=['bold']))
    print(colored(line, color, attrs=['bold']))


def validate_config(cfg):
    """Validate configuration before training"""
    print(colored("🔍 Validating configuration...", 'blue'))
    
    # Check critical parameters
    assert cfg.steps > 0, f"Steps must be > 0, got {cfg.steps}"
    assert cfg.task.startswith('stack-'), f"Task must start with 'stack-', got {cfg.task}"
    assert cfg.obs == 'state', f"Only state observations supported, got {cfg.obs}"
    assert cfg.horizon > 0, f"Horizon must be > 0, got {cfg.horizon}"
    assert cfg.batch_size > 0, f"Batch size must be > 0, got {cfg.batch_size}"
    
    print(colored("✅ Configuration validated", 'green'))


def test_environment_compatibility(cfg):
    """Test environment compatibility before training"""
    print(colored("🧪 Testing environment compatibility...", 'blue'))
    
    try:
        # Test environment creation
        env = make_env(cfg)
        print(colored(f"✅ Environment created: {cfg.task}", 'green'))
        
        # Test reset
        obs = env.reset()
        print(colored(f"✅ Environment reset: obs shape {obs.shape}", 'green'))
        
        # Test step
        action = env.rand_act() * 0.1  # Small action for safety
        obs, reward, done, info = env.step(action)
        print(colored(f"✅ Environment step: reward {reward:.4f}", 'green'))
        
        # Clean up
        if hasattr(env, 'close'):
            env.close()
        
        return True
        
    except Exception as e:
        print(colored(f"❌ Environment test failed: {e}", 'red'))
        return False


def test_agent_compatibility(cfg):
    """Test agent compatibility before training"""
    print(colored("🤖 Testing agent compatibility...", 'blue'))
    
    try:
        # Create environment and agent
        env = make_env(cfg)
        agent = TDMPC2(cfg)
        
        print(colored(f"✅ Agent created: {agent.model.total_params:,} parameters", 'green'))
        
        # Test agent action
        obs = env.reset()
        action = agent.act(obs, t0=True, eval_mode=True)
        print(colored(f"✅ Agent action: shape {action.shape}", 'green'))
        
        # Test environment step with agent
        obs, reward, done, info = env.step(action)
        print(colored(f"✅ Agent + Environment: reward {reward:.4f}", 'green'))
        
        # Clean up
        if hasattr(env, 'close'):
            env.close()
        del agent
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(colored(f"❌ Agent test failed: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return False


def run_quick_test(cfg):
    """Run a quick test episode without full training"""
    print_banner("🧪 QUICK TEST MODE", 'cyan')
    
    # Validate config
    validate_config(cfg)
    
    # Test compatibility
    if not test_environment_compatibility(cfg):
        return False
    
    if not test_agent_compatibility(cfg):
        return False
    
    print(colored("🚀 Running test episode...", 'yellow'))
    
    try:
        # Create components
        env = make_env(cfg)
        agent = TDMPC2(cfg)
        
        # Run test episode
        obs = env.reset()
        total_reward = 0
        max_steps = min(getattr(cfg, 'test_steps', 100), cfg.episode_length)
        
        print(f"Running {max_steps} steps...")
        
        for step in range(max_steps):
            action = agent.act(obs, t0=(step==0), eval_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if step % 25 == 0:
                print(f'Step {step:3d}: reward = {reward:6.3f}, total = {total_reward:8.3f}, success = {info.get("success", False)}')
            
            if done:
                print(f'Episode ended at step {step+1}')
                break
        
        print(colored(f'✅ Test completed!', 'green', attrs=['bold']))
        print(colored(f'   Total reward: {total_reward:.3f}', 'green'))
        print(colored(f'   Steps taken: {step+1}', 'green'))
        print(colored(f'   Success: {info.get("success", False)}', 'green'))
        
        # Clean up
        if hasattr(env, 'close'):
            env.close()
        
        return True
        
    except Exception as e:
        print(colored(f"❌ Quick test failed: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return False


@hydra.main(config_name='sbr_config', config_path='.')
def train_stacking(cfg: dict):
    """
    Complete training script for the SBR stacking environment.
    
    Modes:
        - test: Run quick compatibility test
        - train: Full training (default)
    
    Example usage:
    ```
        # Quick test
        $ python train_stacking.py mode=test steps=100 compile=false
        
        # Short training
        $ python train_stacking.py steps=5000 compile=false enable_wandb=false
        
        # Full training with different parameters
        $ python train_stacking.py steps=500000 horizon=5 model_size=5
        
        # Training with custom parameters
        $ python train_stacking.py steps=1000000 batch_size=512 num_samples=1024
    ```
    """
    
    # Check CUDA availability (warn but don't fail)
    if not torch.cuda.is_available():
        print(colored("⚠️  CUDA not available - training will be slow on CPU", 'yellow'))
    
    # Validate basic requirements
    assert cfg.steps > 0, 'Must train for at least 1 step.'
    
    # Parse config
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    
    # Print header
    print_banner('🧱 SBR STACKING ENVIRONMENT TRAINING', 'cyan')
    
    # Print configuration
    print(colored('📋 Configuration:', 'blue', attrs=['bold']))
    print(colored(f'   Task: {cfg.task}', 'blue'))
    print(colored(f'   Steps: {cfg.steps:,}', 'blue'))
    print(colored(f'   Model size: {cfg.get("model_size", "default")}', 'blue'))
    print(colored(f'   Horizon: {cfg.horizon}', 'blue'))
    print(colored(f'   Batch size: {cfg.batch_size}', 'blue'))
    print(colored(f'   Episode length: {cfg.episode_length}', 'blue'))
    print(colored(f'   Work dir: {cfg.work_dir}', 'yellow'))
    print(colored(f'   Compile: {cfg.compile}', 'blue'))
    print(colored(f'   WandB: {cfg.enable_wandb}', 'blue'))
    print()
    
    # Check if this is test mode
    test_mode = getattr(cfg, 'mode', 'train') == 'test' or cfg.steps < 1000
    
    if test_mode:
        print(colored("🧪 Running in TEST MODE", 'yellow', attrs=['bold']))
        success = run_quick_test(cfg)
        if success:
            print(colored("✅ Test completed successfully!", 'green', attrs=['bold']))
            print(colored("🚀 Ready for full training! Remove 'mode=test' to start training.", 'cyan'))
        else:
            print(colored("❌ Test failed! Please fix issues before training.", 'red', attrs=['bold']))
        return
    
    # Full training mode
    print(colored("🚀 Running FULL TRAINING MODE", 'green', attrs=['bold']))
    
    # Validate configuration
    validate_config(cfg)
    
    # Test compatibility
    print(colored("🔍 Pre-training compatibility check...", 'yellow'))
    if not test_environment_compatibility(cfg):
        print(colored("❌ Environment compatibility test failed!", 'red'))
        return
    
    # Create trainer components
    print(colored("🏗️  Creating training components...", 'yellow'))
    
    try:
        env = make_env(cfg)
        print(colored("✅ Environment created", 'green'))
        
        agent = TDMPC2(cfg)
        print(colored(f"✅ Agent created ({agent.model.total_params:,} parameters)", 'green'))
        
        buffer = Buffer(cfg)
        print(colored(f"✅ Buffer created (capacity: {buffer.capacity:,})", 'green'))
        
        logger = Logger(cfg)
        print(colored("✅ Logger created", 'green'))
        
    except Exception as e:
        print(colored(f"❌ Failed to create training components: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return
    
    # Create trainer
    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
    print(colored(f"📚 Using {trainer_cls.__name__}", 'blue'))
    
    trainer = trainer_cls(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=buffer,
        logger=logger,
    )
    
    # Start training
    print_banner('🚀 STARTING TRAINING', 'green')
    print(colored(f"Training for {cfg.steps:,} steps...", 'green', attrs=['bold']))
    print()
    
    try:
        trainer.train()
        print_banner('✅ TRAINING COMPLETED SUCCESSFULLY!', 'green')
        
        # Print final stats
        print(colored("📊 Training Summary:", 'cyan', attrs=['bold']))
        print(colored(f"   Total steps: {cfg.steps:,}", 'cyan'))
        print(colored(f"   Task: {cfg.task}", 'cyan'))
        print(colored(f"   Model parameters: {agent.model.total_params:,}", 'cyan'))
        print(colored(f"   Work directory: {cfg.work_dir}", 'cyan'))
        
    except KeyboardInterrupt:
        print(colored("\n⚠️  Training interrupted by user", 'yellow'))
        print(colored("💾 Saving current progress...", 'yellow'))
        logger.finish(agent)
        print(colored("✅ Progress saved", 'green'))
        
    except Exception as e:
        print(colored(f"\n❌ Training failed: {e}", 'red'))
        import traceback
        traceback.print_exc()
        print(colored("💾 Attempting to save current progress...", 'yellow'))
        try:
            logger.finish(agent)
            print(colored("✅ Progress saved", 'green'))
        except:
            print(colored("❌ Could not save progress", 'red'))


@hydra.main(config_name='sbr_config', config_path='.')  
def test_only(cfg: dict):
    """Test mode entry point"""
    cfg.mode = 'test'
    train_stacking(cfg)


def main():
    """Main entry point with argument handling"""
    if len(sys.argv) > 1:
        # Check for special commands
        if sys.argv[1] == 'test':
            # Remove 'test' from sys.argv and add mode=test
            sys.argv.pop(1)
            sys.argv.append('mode=test')
            train_stacking()
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print_banner("🧱 SBR STACKING TRAINING HELP", 'cyan')
            print("""
Usage:
    python train_stacking.py [mode] [options...]

Modes:
    test                    Run quick compatibility test
    (default)              Full training mode

Common Options:
    steps=N                Number of training steps (default: from config)
    compile=true/false     Enable/disable torch.compile (default: true)
    enable_wandb=true/false Enable/disable Weights & Biases logging
    horizon=N              Planning horizon (default: 5)
    model_size=N           Model size: 1, 5, 19, 48, 317 (default: 5)
    batch_size=N           Batch size (default: 256)
    
Examples:
    # Quick test
    python train_stacking.py test
    
    # Quick test with custom steps
    python train_stacking.py test steps=50
    
    # Short training run
    python train_stacking.py steps=5000 compile=false enable_wandb=false
    
    # Full training
    python train_stacking.py steps=500000
    
    # Training with larger model
    python train_stacking.py steps=1000000 model_size=19
    
    # Training with custom hyperparameters
    python train_stacking.py steps=500000 horizon=7 batch_size=512
            """)
            return
        else:
            # Regular training with parameters
            train_stacking()
    else:
        # No arguments - regular training
        train_stacking()


if __name__ == '__main__':
    main()