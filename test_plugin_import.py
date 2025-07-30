#!/usr/bin/env python3
"""
Test script to verify the plugin can be imported and basic functionality works
"""
import sys
import os
import unittest.mock as mock

# Add the addon package to the path
addon_path = os.path.join(os.path.dirname(__file__), 'addon_package', 'plugins')
sys.path.insert(0, addon_path)

def test_plugin_import():
    """Test if the plugin can be imported"""
    try:
        # Mock wx and pcbnew since we don't have KiCad environment
        wx_mock = mock.MagicMock()
        wx_mock.ID_OK = 1
        wx_mock.OK = 1
        wx_mock.ICON_INFORMATION = 2
        wx_mock.ICON_ERROR = 4
        wx_mock.Dialog = mock.MagicMock
        wx_mock.Frame = mock.MagicMock
        
        pcbnew_mock = mock.MagicMock()
        pcbnew_mock.ActionPlugin = mock.MagicMock
        pcbnew_mock.GetBoard = mock.MagicMock(return_value=mock.MagicMock())
        
        sys.modules['wx'] = wx_mock
        sys.modules['pcbnew'] = pcbnew_mock
        
        # Now try to import the plugin
        import __init__ as plugin
        
        print("✅ Plugin import successful")
        
        # Test that the main classes exist
        if hasattr(plugin, 'OrthoRouteKiCadPlugin'):
            print("✅ OrthoRouteKiCadPlugin class found")
        
        if hasattr(plugin, 'OrthoRouteConfigDialog'):
            print("✅ OrthoRouteConfigDialog class found")
            
        return True
        
    except Exception as e:
        print(f"❌ Plugin import failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def test_plugin_registration():
    """Test if the plugin registers correctly"""
    try:
        # Mock pcbnew
        pcbnew_mock = mock.MagicMock()
        sys.modules['pcbnew'] = pcbnew_mock
        
        # Import and test registration
        import __init__ as plugin
        
        # Create an instance
        plugin_instance = plugin.OrthoRouteKiCadPlugin()
        print("✅ Plugin instance created successfully")
        
        # Test the description
        desc = plugin_instance.GetDescription()
        if "OrthoRoute" in desc:
            print("✅ Plugin description contains 'OrthoRoute'")
        
        return True
        
    except Exception as e:
        print(f"❌ Plugin registration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing OrthoRoute plugin import...")
    
    success = True
    
    # Test import
    if not test_plugin_import():
        success = False
    
    # Test registration  
    if not test_plugin_registration():
        success = False
    
    if success:
        print("\n🎉 All plugin tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some plugin tests failed!")
        sys.exit(1)
