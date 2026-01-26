"""Quick test to verify app.py can be imported and basic structure is correct."""

import sys
import os

# Add current directory to path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("  ✓ streamlit")
    except ImportError as e:
        print(f"  ✗ streamlit: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        import openpyxl
        print("  ✓ openpyxl")
    except ImportError as e:
        print(f"  ✗ openpyxl: {e}")
        return False
    
    # Test local modules
    modules = [
        'ingest', 'normalize', 'parse_quotes', 'parse_sentiment',
        'score', 'digest', 'render', 'export', 'edge_cases'
    ]
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            return False
    
    return True


def test_app_structure():
    """Test that app.py has required structure."""
    print("\nTesting app.py structure...")
    
    try:
        # Try to import app (this will execute it, so we need to be careful)
        # Instead, just check it exists and is readable
        with open('app.py', 'r') as f:
            content = f.read()
            
        required_imports = [
            'import streamlit',
            'import ingest',
            'import normalize',
            'import score',
            'import digest'
        ]
        
        for imp in required_imports:
            if imp in content:
                print(f"  ✓ {imp}")
            else:
                print(f"  ✗ Missing: {imp}")
                return False
        
        # Check for main function
        if 'def main()' in content:
            print("  ✓ main() function exists")
        else:
            print("  ✗ main() function not found")
            return False
        
        return True
    except FileNotFoundError:
        print("  ✗ app.py not found")
        return False
    except Exception as e:
        print(f"  ✗ Error reading app.py: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("APP STARTUP TEST")
    print("=" * 60)
    
    imports_ok = test_imports()
    structure_ok = test_app_structure()
    
    print("\n" + "=" * 60)
    if imports_ok and structure_ok:
        print("✓ All checks passed - app should start successfully")
        print("\nTo start the app, run:")
        print("  streamlit run app.py")
        sys.exit(0)
    else:
        print("✗ Some checks failed - fix issues before deploying")
        sys.exit(1)