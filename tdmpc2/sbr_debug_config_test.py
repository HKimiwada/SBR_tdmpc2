# debug_config_test.py - Debug the config parsing issue
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

# Add current directory to path
sys.path.insert(0, '.')

from test_config_parser import parse_cfg_for_testing

def debug_config():
    """Debug the config parsing to find the type issue"""
    print("🔍 Debug Config Parsing")
    print("=" * 50)
    
    # Load raw config
    if not os.path.exists('sbr_config.yaml'):
        print("❌ sbr_config.yaml not found")
        return False
    
    print("📖 Loading raw config...")
    cfg = OmegaConf.load('sbr_config.yaml')
    
    # Print raw values
    print("\n📋 Raw config values:")
    critical_fields = ['episode_length', 'seed_steps', 'action_dim', 'steps', 'batch_size', 'horizon']
    for field in critical_fields:
        if hasattr(cfg, field):
            value = getattr(cfg, field)
            print(f"   {field}: {value} (type: {type(value).__name__})")
        else:
            print(f"   {field}: NOT SET")
    
    # Parse config
    print("\n🔧 Parsing config...")
    try:
        parsed_cfg = parse_cfg_for_testing(cfg)
        print("✅ Config parsed successfully")
        
        print("\n📋 Parsed config values:")
        for field in critical_fields:
            if hasattr(parsed_cfg, field):
                value = getattr(parsed_cfg, field)
                print(f"   {field}: {value} (type: {type(value).__name__})")
            else:
                print(f"   {field}: NOT SET")
        
        # Test the problematic line
        print("\n🧪 Testing problematic line...")
        episode_length = parsed_cfg.episode_length
        print(f"   episode_length = {episode_length} (type: {type(episode_length).__name__})")
        
        if isinstance(episode_length, str):
            print(f"   ❌ episode_length is string: '{episode_length}'")
            print(f"   🔧 Converting to int...")
            episode_length = int(episode_length.replace('_', ''))
            print(f"   ✅ Converted: {episode_length} (type: {type(episode_length).__name__})")
        
        # Test the math operation
        seed_steps = max(1000, 5 * episode_length)
        print(f"   ✅ seed_steps calculation works: {seed_steps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_make_env():
    """Test the make_env function specifically"""
    print("\n🏭 Testing make_env function")
    print("=" * 50)
    
    try:
        from envs import make_env
        
        # Load and parse config
        cfg = OmegaConf.load('sbr_config.yaml')
        cfg = parse_cfg_for_testing(cfg)
        
        print(f"Before make_env:")
        print(f"   episode_length: {cfg.episode_length} (type: {type(cfg.episode_length).__name__})")
        print(f"   seed_steps: {getattr(cfg, 'seed_steps', 'NOT SET')}")
        
        # Try to create environment
        env = make_env(cfg)
        print("✅ make_env succeeded!")
        
        if hasattr(env, 'close'):
            env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ make_env failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debug tests"""
    print("🐛 CONFIG DEBUG TEST")
    print("=" * 60)
    
    success1 = debug_config()
    success2 = test_make_env()
    
    print("\n" + "=" * 60)
    print("🎯 DEBUG SUMMARY")
    print("=" * 60)
    print(f"   Config Parsing: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"   Make Env Test:  {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All debug tests passed!")
        print("✅ The config issue should be fixed now")
    else:
        print("\n💥 Debug tests failed!")
        print("❌ Need to investigate further")

if __name__ == "__main__":
    main()