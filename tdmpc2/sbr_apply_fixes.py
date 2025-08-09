# apply_fixes.py - Apply all fixes for the string/int type error
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

def fix_env_init():
    """Fix the envs/__init__.py file to handle string episode_length"""
    print("🔧 Fixing envs/__init__.py...")
    
    env_init_path = Path("envs/__init__.py")
    
    if not env_init_path.exists():
        print("❌ envs/__init__.py not found")
        return False
    
    # Read the current file
    with open(env_init_path, 'r') as f:
        content = f.read()
    
    # The problematic line
    old_line = "\tcfg.seed_steps = max(1000, 5*cfg.episode_length)"
    
    # The fixed line
    new_line = """\t# Handle episode_length as string or int
\tepisode_length = cfg.episode_length
\tif isinstance(episode_length, str):
\t\tepisode_length = int(episode_length.replace('_', ''))
\tcfg.seed_steps = max(1000, 5*episode_length)"""
    
    # Check if already fixed
    if "episode_length = cfg.episode_length" in content:
        print("✅ envs/__init__.py already fixed")
        return True
    
    # Apply the fix
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the fixed file
        with open(env_init_path, 'w') as f:
            f.write(content)
        
        print("✅ envs/__init__.py fixed")
        return True
    else:
        print("⚠️  Could not find the exact line to fix in envs/__init__.py")
        print("   You may need to manually fix line 91")
        return False

def test_config_fix():
    """Test if the config fix works"""
    print("\n🧪 Testing config fix...")
    
    try:
        # Test the import
        sys.path.insert(0, '.')
        from test_config_parser import parse_cfg_for_testing
        
        # Load config
        if not os.path.exists('sbr_config.yaml'):
            print("❌ sbr_config.yaml not found")
            return False
        
        cfg = OmegaConf.load('sbr_config.yaml')
        parsed_cfg = parse_cfg_for_testing(cfg)
        
        # Check episode_length type
        episode_length = parsed_cfg.episode_length
        if not isinstance(episode_length, int):
            print(f"❌ episode_length is still {type(episode_length).__name__}: {episode_length}")
            return False
        
        print(f"✅ episode_length is int: {episode_length}")
        
        # Test the problematic calculation
        seed_steps = max(1000, 5 * episode_length)
        print(f"✅ seed_steps calculation works: {seed_steps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_make_env():
    """Test if make_env works now"""
    print("\n🏭 Testing make_env...")
    
    try:
        from envs import make_env
        from test_config_parser import parse_cfg_for_testing
        
        # Load and parse config
        cfg = OmegaConf.load('sbr_config.yaml')
        cfg = parse_cfg_for_testing(cfg)
        
        # Try to create environment
        env = make_env(cfg)
        print("✅ make_env works!")
        
        # Test basic functionality
        obs = env.reset()
        print(f"✅ Environment reset: obs shape {obs.shape}")
        
        if hasattr(env, 'close'):
            env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ make_env test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Apply all fixes and test them"""
    print("🔧 APPLYING ALL FIXES")
    print("=" * 50)
    
    # Apply fixes
    fix1 = fix_env_init()
    fix2 = test_config_fix()
    fix3 = test_make_env()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 FIX SUMMARY")
    print("=" * 50)
    print(f"   envs/__init__.py fix:  {'✅ SUCCESS' if fix1 else '❌ FAILED'}")
    print(f"   Config parsing test:   {'✅ SUCCESS' if fix2 else '❌ FAILED'}")
    print(f"   make_env test:         {'✅ SUCCESS' if fix3 else '❌ FAILED'}")
    
    if fix1 and fix2 and fix3:
        print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("✅ The string/int type error should be resolved")
        print("\n🚀 Next steps:")
        print("   1. Run: python test_sbr_tdmpc2_integration_fixed.py")
        print("   2. Run: python train_stacking.py test")
    else:
        print("\n💥 SOME FIXES FAILED!")
        if not fix1:
            print("❌ envs/__init__.py fix failed - you may need to edit it manually")
            print("   Change line ~91 to handle string episode_length")
        if not fix2:
            print("❌ Config parsing issue - check test_config_parser.py")
        if not fix3:
            print("❌ make_env still failing - check error messages above")

if __name__ == "__main__":
    main()