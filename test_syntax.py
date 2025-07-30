#!/usr/bin/env python3
"""
Test script to validate Python syntax of the plugin without KiCad dependencies
"""
import ast
import sys

def test_python_syntax(file_path):
    """Test if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(content)
        print(f"✅ {file_path}: Valid Python syntax")
        return True
        
    except SyntaxError as e:
        print(f"❌ {file_path}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"⚠️ {file_path}: Error reading file: {e}")
        return False

def main():
    """Test all plugin files"""
    files_to_test = [
        "addon_package/plugins/__init__.py",
        "addon_package/plugins/orthoroute_engine.py"
    ]
    
    all_valid = True
    
    for file_path in files_to_test:
        if not test_python_syntax(file_path):
            all_valid = False
    
    if all_valid:
        print("\n🎉 All files have valid Python syntax!")
        return 0
    else:
        print("\n❌ Some files have syntax errors!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
