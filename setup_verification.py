#!/usr/bin/env python3
"""
Setup verification script for Hugging Face integration.
Run this after completing the Python 3.11 setup.
"""

import sys
import os

def check_python_version():
    """Check if we're using Python 3.11."""
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro} ✓")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro}")
        print("   Please use Python 3.11 for full compatibility")
        return False

def check_virtual_env():
    """Check if we're in a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Running in virtual environment ✓")
        return True
    else:
        print("❌ Not running in virtual environment")
        print("   Activate your venv311 environment first")
        return False

def check_packages():
    """Check if required packages are installed."""
    packages = ['transformers', 'torch', 'accelerate']
    all_installed = True

    for package in packages:
        try:
            __import__(package)
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package} installed (v{version}) ✓")
        except ImportError:
            print(f"❌ {package} not installed")
            all_installed = False

    return all_installed

def test_huggingface_import():
    """Test importing our Hugging Face helper."""
    try:
        from huggingface_helper import HuggingFaceSQLAssistant
        print("✅ huggingface_helper.py imported successfully ✓")
        return True
    except ImportError as e:
        print(f"❌ Failed to import huggingface_helper.py: {e}")
        return False

def main():
    print("=" * 60)
    print("🤖 Hugging Face Integration Setup Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Required Packages", check_packages),
        ("Hugging Face Helper", test_huggingface_import),
    ]

    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Ready to test Hugging Face integration!")
        print("\nNext steps:")
        print("1. Run: python huggingface_helper.py")
        print("2. Run: python langchain_helper.py --huggingface")
        print("3. Run: streamlit run main.py (select 'huggingface' in sidebar)")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before proceeding.")
    print("=" * 60)

if __name__ == "__main__":
    main()
