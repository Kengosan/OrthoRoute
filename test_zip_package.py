#!/usr/bin/env python3
"""
Test the built zip package for structure and syntax validation
"""

import zipfile
import json
import ast
import sys
import os
from pathlib import Path

def test_zip_package():
    """Test the built addon package"""
    
    print("🧪 Testing OrthoRoute Addon Package")
    print("=" * 60)
    
    zip_path = Path("orthoroute-kicad-addon.zip")
    
    if not zip_path.exists():
        print("❌ Package not found: orthoroute-kicad-addon.zip")
        return False
    
    print(f"📦 Package found: {zip_path.name} ({zip_path.stat().st_size / 1024:.1f} KB)")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            
            # Test 1: Package structure
            print("\n📋 Testing package structure...")
            files = zf.namelist()
            
            required_files = [
                'metadata.json',
                'plugins/__init__.py',
                'plugins/orthoroute_engine.py',
                'plugins/icon.png',
                'resources/icon.png'
            ]
            
            missing = []
            for req_file in required_files:
                if req_file not in files:
                    missing.append(req_file)
            
            if missing:
                print(f"❌ Missing required files: {missing}")
                return False
            else:
                print("✅ All required files present")
            
            # Test 2: Metadata validation
            print("\n📋 Testing metadata...")
            metadata_content = zf.read('metadata.json').decode('utf-8')
            try:
                metadata = json.loads(metadata_content)
                required_keys = ['name', 'version', 'identifier', 'type']
                
                for key in required_keys:
                    if key not in metadata:
                        print(f"❌ Missing metadata key: {key}")
                        return False
                
                print(f"✅ Metadata valid: {metadata['name']} v{metadata['version']}")
                
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in metadata: {e}")
                return False
            
            # Test 3: Python syntax validation
            print("\n📋 Testing Python syntax...")
            python_files = [f for f in files if f.endswith('.py')]
            
            syntax_errors = []
            for py_file in python_files:
                try:
                    content = zf.read(py_file).decode('utf-8')
                    ast.parse(content)
                    print(f"✅ {py_file}: syntax OK")
                except SyntaxError as e:
                    syntax_errors.append((py_file, str(e)))
                    print(f"❌ {py_file}: syntax error - {e}")
                except UnicodeDecodeError as e:
                    print(f"⚠️ {py_file}: encoding issue - {e}")
            
            if syntax_errors:
                print(f"\n❌ Syntax errors found in {len(syntax_errors)} files")
                return False
            
            # Test 4: Main plugin validation
            print("\n📋 Testing main plugin structure...")
            main_plugin = zf.read('plugins/__init__.py').decode('utf-8')
            
            required_patterns = [
                'class OrthoRouteKiCadPlugin',
                'def Run(',
                'def defaults(',
                'ActionPlugin'
            ]
            
            missing_patterns = []
            for pattern in required_patterns:
                if pattern not in main_plugin:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                print(f"❌ Missing plugin patterns: {missing_patterns}")
                return False
            else:
                print("✅ Main plugin structure valid")
                
            # Additional checks for plugin functionality
            functionality_checks = [
                ('GPU routing', '_route_board_simple'),
                ('Config dialog', 'OrthoRouteConfigDialog'),
                ('Debug output', 'debug_print'),
                ('GPU acceleration', 'cupy'),
                ('Error handling', 'except Exception')
            ]
            
            for check_name, pattern in functionality_checks:
                if pattern in main_plugin:
                    print(f"✅ {check_name} functionality present")
                else:
                    print(f"⚠️ {check_name} functionality missing")
            
            # Check for enhanced debugging we just added
            debug_patterns = [
                ('Enhanced path extraction', '🎯 Extracting path to target'),
                ('Track creation debugging', '🛤 Creating tracks from'),
                ('Error tracebacks', 'traceback.format_exc'),
                ('Conservative processing', 'Conservative cell processing')
            ]
            
            for check_name, pattern in debug_patterns:
                if pattern in main_plugin:
                    print(f"✅ {check_name} debugging present")
                else:
                    print(f"⚠️ {check_name} debugging missing")
            
            # Test 5: File sizes check
            print("\n📋 Testing file sizes...")
            for file_info in zf.infolist():
                size_kb = file_info.file_size / 1024
                print(f"  {file_info.filename}: {size_kb:.1f} KB")
                
                # Check for suspicious file sizes
                if file_info.filename.endswith('.py') and size_kb > 200:
                    print(f"⚠️ Large Python file: {file_info.filename} ({size_kb:.1f} KB)")
                elif file_info.filename.endswith('.py') and size_kb < 0.1:
                    print(f"⚠️ Very small Python file: {file_info.filename} ({size_kb:.1f} KB)")
            
            print("\n✅ Package validation complete!")
            print(f"📊 Package contains {len(files)} files")
            print(f"📊 Total uncompressed size: {sum(f.file_size for f in zf.infolist()) / 1024:.1f} KB")
            
            return True
            
    except zipfile.BadZipFile as e:
        print(f"❌ Invalid zip file: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_import_simulation():
    """Simulate plugin import without KiCad APIs"""
    
    print("\n🔧 Testing plugin import simulation...")
    print("=" * 50)
    
    # Add the plugin directory to path
    plugin_dir = Path("addon_package/plugins")
    if not plugin_dir.exists():
        print("❌ Plugin directory not found")
        return False
    
    sys.path.insert(0, str(plugin_dir))
    
    try:
        # Test engine import (should work without KiCad APIs)
        print("📦 Testing orthoroute_engine import...")
        import orthoroute_engine
        print("✅ orthoroute_engine imported successfully")
        
        # Check for key classes/functions
        expected_attrs = ['WaveRouter', 'GPUEngine']
        for attr in expected_attrs:
            if hasattr(orthoroute_engine, attr):
                print(f"✅ Found {attr}")
            else:
                print(f"⚠️ Missing {attr}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        # Clean up
        if str(plugin_dir) in sys.path:
            sys.path.remove(str(plugin_dir))

def main():
    """Run all package tests"""
    
    print("🚀 OrthoRoute Package Validation Suite")
    print("=" * 60)
    
    tests = [
        ("Package Structure", test_zip_package),
        ("Import Simulation", test_import_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Package is ready for installation.")
        return True
    else:
        print("❌ Some tests failed. Review issues before installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
