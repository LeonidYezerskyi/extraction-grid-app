"""Script to verify repository is deployable."""

import sys
import os
import subprocess
from pathlib import Path


def check_file_exists(filepath: str) -> tuple[bool, str]:
    """Check if file exists."""
    exists = os.path.exists(filepath)
    return exists, f"{'✓' if exists else '✗'} {filepath}"


def check_requirements_txt() -> tuple[bool, list]:
    """Check requirements.txt exists and has required packages."""
    issues = []
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        return False, ["requirements.txt not found"]
    
    with open(req_file, 'r') as f:
        content = f.read()
        required_packages = ['streamlit', 'pandas', 'openpyxl']
        
        for pkg in required_packages:
            if pkg not in content:
                issues.append(f"Missing package in requirements.txt: {pkg}")
    
    return len(issues) == 0, issues


def check_module_files() -> tuple[bool, list]:
    """Check all required module files exist."""
    required_modules = [
        'app.py',
        'ingest.py',
        'normalize.py',
        'parse_quotes.py',
        'parse_sentiment.py',
        'score.py',
        'digest.py',
        'render.py',
        'export.py',
        'edge_cases.py'
    ]
    
    missing = []
    for module in required_modules:
        if not os.path.exists(module):
            missing.append(module)
    
    return len(missing) == 0, missing


def check_imports() -> tuple[bool, list]:
    """Check that main modules can be imported."""
    issues = []
    
    try:
        import ingest
    except ImportError as e:
        issues.append(f"Cannot import ingest: {e}")
    
    try:
        import normalize
    except ImportError as e:
        issues.append(f"Cannot import normalize: {e}")
    
    try:
        import parse_quotes
    except ImportError as e:
        issues.append(f"Cannot import parse_quotes: {e}")
    
    try:
        import parse_sentiment
    except ImportError as e:
        issues.append(f"Cannot import parse_sentiment: {e}")
    
    try:
        import score
    except ImportError as e:
        issues.append(f"Cannot import score: {e}")
    
    try:
        import digest
    except ImportError as e:
        issues.append(f"Cannot import digest: {e}")
    
    try:
        import render
    except ImportError as e:
        issues.append(f"Cannot import render: {e}")
    
    try:
        import export
    except ImportError as e:
        issues.append(f"Cannot import export: {e}")
    
    try:
        import edge_cases
    except ImportError as e:
        issues.append(f"Cannot import edge_cases: {e}")
    
    return len(issues) == 0, issues


def check_streamlit_import() -> tuple[bool, str]:
    """Check that streamlit can be imported."""
    try:
        import streamlit as st
        return True, "Streamlit can be imported"
    except ImportError as e:
        return False, f"Cannot import streamlit: {e}"


def check_pandas_import() -> tuple[bool, str]:
    """Check that pandas can be imported."""
    try:
        import pandas as pd
        return True, "Pandas can be imported"
    except ImportError as e:
        return False, f"Cannot import pandas: {e}"


def check_openpyxl_import() -> tuple[bool, str]:
    """Check that openpyxl can be imported."""
    try:
        import openpyxl
        return True, "Openpyxl can be imported"
    except ImportError as e:
        return False, f"Cannot import openpyxl: {e}"


def verify_deployment() -> dict:
    """
    Verify repository is deployable.
    
    Returns:
        Dictionary with verification results
    """
    print("=" * 60)
    print("DEPLOYMENT VERIFICATION")
    print("=" * 60)
    
    results = {
        'files_exist': {},
        'requirements_txt': {},
        'module_files': {},
        'imports': {},
        'dependencies': {}
    }
    
    # Check required files
    print("\n[1/5] Checking required files...")
    required_files = ['app.py', 'requirements.txt']
    for file in required_files:
        exists, msg = check_file_exists(file)
        results['files_exist'][file] = exists
        print(f"  {msg}")
    
    # Check requirements.txt
    print("\n[2/5] Checking requirements.txt...")
    req_ok, req_issues = check_requirements_txt()
    results['requirements_txt'] = {'ok': req_ok, 'issues': req_issues}
    if req_ok:
        print("  ✓ requirements.txt is valid")
    else:
        print("  ✗ requirements.txt issues:")
        for issue in req_issues:
            print(f"    - {issue}")
    
    # Check module files
    print("\n[3/5] Checking module files...")
    modules_ok, missing_modules = check_module_files()
    results['module_files'] = {'ok': modules_ok, 'missing': missing_modules}
    if modules_ok:
        print("  ✓ All required modules exist")
    else:
        print("  ✗ Missing modules:")
        for module in missing_modules:
            print(f"    - {module}")
    
    # Check imports
    print("\n[4/5] Checking module imports...")
    imports_ok, import_issues = check_imports()
    results['imports'] = {'ok': imports_ok, 'issues': import_issues}
    if imports_ok:
        print("  ✓ All modules can be imported")
    else:
        print("  ✗ Import issues:")
        for issue in import_issues:
            print(f"    - {issue}")
    
    # Check dependencies
    print("\n[5/5] Checking dependencies...")
    streamlit_ok, streamlit_msg = check_streamlit_import()
    pandas_ok, pandas_msg = check_pandas_import()
    openpyxl_ok, openpyxl_msg = check_openpyxl_import()
    
    results['dependencies'] = {
        'streamlit': {'ok': streamlit_ok, 'message': streamlit_msg},
        'pandas': {'ok': pandas_ok, 'message': pandas_msg},
        'openpyxl': {'ok': openpyxl_ok, 'message': openpyxl_msg}
    }
    
    print(f"  {streamlit_msg}")
    print(f"  {pandas_msg}")
    print(f"  {openpyxl_msg}")
    
    # Summary
    print("\n" + "=" * 60)
    all_checks = [
        req_ok,
        modules_ok,
        imports_ok,
        streamlit_ok,
        pandas_ok,
        openpyxl_ok
    ]
    passed = sum(all_checks)
    total = len(all_checks)
    
    print(f"Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ Repository is deployable!")
        results['deployable'] = True
    else:
        print("✗ Repository has issues that need to be fixed")
        results['deployable'] = False
    
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    results = verify_deployment()
    sys.exit(0 if results.get('deployable', False) else 1)