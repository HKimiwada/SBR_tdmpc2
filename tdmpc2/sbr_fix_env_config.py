# fix_env_config.py - Patch to fix environment config issues
import os
from pathlib import Path

def patch_env_init():
    """Patch the envs/__init__.py file to handle string episode_length"""
    
    env_init_path = Path("envs/__init__.py")
    
    if not env_init_path.exists():
        print("❌ envs/__init__.py not found")
        return False
    
    print("🔧 Patching envs/__init__.py to handle string episode_length...")
    
    # Read the current file
    with open(env_init_path, 'r') as f:
        content = f.read()
    
    # The problematic line
    old_line = "cfg.seed_steps = max(1000, 5*cfg.episode_length)"
    
    # The fixed line
    new_line = """# Handle episode_length as string or int
    episode_length = cfg.episode_length
    if isinstance(episode_length, str):
        episode_length = int(episode_length.replace('_', ''))
    cfg.seed_steps = max(1000, 5*episode_length)"""
    
    # Check if patch is needed
    if old_line not in content:
        if "episode_length = cfg.episode_length" in content:
            print("✅ Patch already applied")
            return True
        else:
            print("⚠️  Could not find the problematic line - file may have been modified")
            return False
    
    # Apply the patch
    content = content.replace(old_line, new_line)
    
    # Backup the original
    backup_path = env_init_path.with_suffix('.py.backup')
    if not backup_path.exists():
        with open(backup_path, 'w') as f:
            with open(env_init_path, 'r') as orig:
                f.write(orig.read())
        print(f"📁 Backup created: {backup_path}")
    
    # Write the patched file
    with open(env_init_path, 'w') as f:
        f.write(content)
    
    print("✅ Patch applied successfully")
    return True

def main():
    """Apply the environment config fix"""
    print("🔧 Environment Config Fix")
    print("=" * 40)
    
    success = patch_env_init()
    
    if success:
        print("\n🎉 Fix applied successfully!")
        print("✅ Now run: python debug_config_test.py")
        print("✅ Then run: python test_sbr_tdmpc2_integration_fixed.py")
    else:
        print("\n❌ Fix failed!")
        print("💡 You may need to manually edit envs/__init__.py")
        print("   Change line 91 from:")
        print("     cfg.seed_steps = max(1000, 5*cfg.episode_length)")
        print("   To:")
        print("     episode_length = int(cfg.episode_length) if isinstance(cfg.episode_length, str) else cfg.episode_length")
        print("     cfg.seed_steps = max(1000, 5*episode_length)")

if __name__ == "__main__":
    main()