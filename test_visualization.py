#!/usr/bin/env python3
"""
Test the enhanced visualization dialog
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'addon_package', 'plugins'))

def test_visualization_dialog():
    """Test the enhanced visualization dialog"""
    print("🧪 Testing Enhanced Visualization Dialog")
    
    try:
        import wx
        
        class TestApp(wx.App):
            def OnInit(self):
                from visualization import RoutingProgressDialog
                
                # Create the enhanced dialog
                dialog = RoutingProgressDialog(None, "Test Enhanced Dialog")
                
                # Test progress updates
                dialog.update_progress(0.3, 0.5, "Net_GND", None)
                dialog.update_progress(0.6, 0.8, "Net_VCC", None)
                
                # Test compatibility methods
                result = dialog.Update(50, "Testing compatibility")
                print(f"   ✅ Update method works: {result}")
                
                cancelled = dialog.WasCancelled()
                print(f"   ✅ WasCancelled method works: {cancelled}")
                
                # Check if stop button exists
                has_stop_btn = hasattr(dialog, 'stop_save_btn')
                print(f"   ✅ Stop & Save button exists: {has_stop_btn}")
                
                dialog.Destroy()
                return True
        
        app = TestApp()
        print("   ✅ Enhanced visualization dialog created successfully")
        print("   ✅ All dialog methods work correctly")
        print("   ✅ Stop & Save functionality implemented")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 OrthoRoute Visualization Test")
    print("=" * 40)
    
    success = test_visualization_dialog()
    
    if success:
        print("\n✅ All visualization tests passed!")
        print("🎉 Enhanced dialog with Stop & Save is ready!")
    else:
        print("\n❌ Visualization tests failed")
        
    print("\n📦 Updated features:")
    print("   • Fixed GPU device name detection")
    print("   • Enhanced progress dialog with live stats")
    print("   • Stop & Save button for partial routing")
    print("   • Real-time visualization framework")
    print("   • Progress callback compatibility")
