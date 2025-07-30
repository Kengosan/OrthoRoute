"""
OrthoRoute GPU Autorouter - MINIMAL SAFE VERSION
This version strips out all potentially crash-causing components for debugging.
"""

import pcbnew
import wx
import sys
import traceback

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Minimal safe version of OrthoRoute plugin for crash debugging"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Safe Mode)"
        self.category = "Routing"
        self.description = "Safe debug version - minimal functionality"
        self.show_toolbar_button = True
        # Don't load icon to avoid file access issues
    
    def Run(self):
        """Minimal safe run method"""
        try:
            print("🔍 OrthoRoute Safe Mode - Starting...")
            
            # Test 1: Basic message box
            wx.MessageBox("Test 1: Basic message box works", "Safe Mode Test", wx.OK)
            print("✅ Test 1 passed: Basic message box")
            
            # Test 2: Get board
            try:
                board = pcbnew.GetBoard()
                if not board:
                    wx.MessageBox("No board found", "Safe Mode", wx.OK | wx.ICON_WARNING)
                    return
                print("✅ Test 2 passed: Board access")
            except Exception as e:
                print(f"❌ Test 2 failed: Board access - {e}")
                wx.MessageBox(f"Board access failed: {e}", "Safe Mode Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Test 3: Basic dialog
            try:
                dlg = wx.Dialog(None, title="Safe Mode Test Dialog")
                dlg.SetSize((300, 200))
                
                panel = wx.Panel(dlg)
                sizer = wx.BoxSizer(wx.VERTICAL)
                
                text = wx.StaticText(panel, label="This is a minimal safe dialog.\nNo crashes should occur.")
                sizer.Add(text, 0, wx.ALL | wx.CENTER, 10)
                
                ok_btn = wx.Button(panel, wx.ID_OK, "OK")
                sizer.Add(ok_btn, 0, wx.ALL | wx.CENTER, 10)
                
                panel.SetSizer(sizer)
                
                if dlg.ShowModal() == wx.ID_OK:
                    print("✅ Test 3 passed: Basic dialog")
                dlg.Destroy()
            except Exception as e:
                print(f"❌ Test 3 failed: Basic dialog - {e}")
                wx.MessageBox(f"Dialog test failed: {e}", "Safe Mode Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Test 4: Import testing (one by one)
            print("🔍 Testing imports individually...")
            
            # Test numpy first
            try:
                import numpy as np
                print("✅ NumPy import successful")
            except Exception as e:
                print(f"❌ NumPy import failed: {e}")
                wx.MessageBox(f"NumPy not available: {e}", "Import Warning", wx.OK | wx.ICON_WARNING)
            
            # Test CuPy (this is likely the culprit)
            try:
                print("🔍 Testing CuPy import...")
                import cupy as cp
                print("✅ CuPy import successful")
                
                # Test basic CuPy operation
                try:
                    test_array = cp.array([1, 2, 3])
                    print("✅ CuPy basic operation successful")
                except Exception as e:
                    print(f"❌ CuPy operation failed: {e}")
                    wx.MessageBox(f"CuPy operation failed: {e}\n\nThis may be the cause of KiCad crashes.", 
                                "CuPy Error", wx.OK | wx.ICON_ERROR)
                    
            except ImportError as e:
                print(f"⚠️ CuPy not available: {e}")
                wx.MessageBox("CuPy not installed - this is expected in safe mode", "Safe Mode", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                print(f"❌ CuPy import caused error: {e}")
                traceback.print_exc()
                wx.MessageBox(f"CuPy import error: {e}\n\nThis is likely causing KiCad to crash!", 
                            "Critical - CuPy Issue", wx.OK | wx.ICON_ERROR)
                return
            
            # Test our local modules (without importing them)
            print("🔍 Testing local module file access...")
            import os
            plugin_dir = os.path.dirname(__file__)
            
            for module_name in ['visualization.py', 'orthoroute_engine.py', 'board_exporter.py']:
                module_path = os.path.join(plugin_dir, module_name)
                if os.path.exists(module_path):
                    print(f"✅ {module_name} file exists")
                else:
                    print(f"❌ {module_name} file missing")
            
            # Success message
            wx.MessageBox(
                "Safe Mode Tests Completed!\n\n" + 
                "✅ Basic functionality works\n" +
                "✅ Board access works\n" + 
                "✅ Dialog creation works\n\n" +
                "Check console for import test results.\n" +
                "If CuPy caused errors, that's likely the crash source.",
                "Safe Mode - Tests Complete", 
                wx.OK | wx.ICON_INFORMATION
            )
            
        except Exception as e:
            print(f"💥 CRITICAL ERROR in safe mode: {e}")
            traceback.print_exc()
            try:
                wx.MessageBox(f"Critical error in safe mode: {e}", "Critical Error", wx.OK | wx.ICON_ERROR)
            except:
                print("💥 Can't even show error dialog!")

# Register the plugin
OrthoRouteKiCadPlugin().register()
