# test_stacking_simple.py - Simple test script to verify stacking environment
import os
import sys
import warnings

# Set environment variables for headless operation
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
os.environ['DISPLAY'] = os.getenv('DISPLAY', ':99')

# Suppress warnings
warnings.filterwarnings('ignore')

def test_dm_control_import():
    """Test if dm_control can be imported"""
    print("🔧 Testing dm_control import...")
    
    try:
        from dm_control import manipulation
        print("✅ dm_control.manipulation imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import dm_control.manipulation: {e}")
        print("   Install with: pip install 'dm_control[manipulation]'")
        return False
    except Exception as e:
        print(f"❌ Error importing dm_control: {e}")
        return False

def test_stacking_env():
    """Test the stacking environment directly"""
    print("\n🧱 Testing stacking environment...")
    
    try:
        # Import from sbr_env
        sys.path.insert(0, './sbr_env')
        from sbr_stacking_env import make_simple_env, test_environment
        
        print("✅ Successfully imported stacking environment")
        
        # Run the built-in test
        success = test_environment()
        return success
        
    except Exception as e:
        print(f"❌ Stacking environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_integration():
    """Test the environment integration with TD-MPC2 format"""
    print("\n🔗 Testing environment integration...")
    
    try:
        from envs.sbr_stacking import test_make_env
        
        print("✅ Successfully imported environment loader")
        
        # Run the integration test
        success = test_make_env()
        return success
        
    except Exception as e:
        print(f"❌ Environment integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pytorch():
    """Test PyTorch availability"""
    print("\n🔥 Testing PyTorch...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available - will use CPU")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available")
        return False

def main():
    """Run all simple tests"""
    print("🚀 Simple Stacking Environment Test")
    print("=" * 50)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    print()
    
    # Run tests
    tests = [
        ("PyTorch", test_pytorch),
        ("DM Control", test_dm_control_import),
        ("Stacking Environment", test_stacking_env),
        ("Environment Integration", test_env_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        results[test_name] = test_func()
        print()
    
    # Summary
    print("=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:<25} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your stacking environment is working correctly")
        print("\n🚀 Next steps:")
        print("   1. Run: python test_sbr_tdmpc2_integration_fixed.py")
        print("   2. Run: python train_stacking.py test")
    else:
        print("\n💥 SOME TESTS FAILED!")
        print("❌ Please fix the issues above")
        
        if not results.get("DM Control", False):
            print("\n📦 To install dm_control:")
            print("   pip install 'dm_control[manipulation]'")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)