"""
OrthoRoute GPU Autorouter - PROGRESSIVE DEBUG VERSION
This version tests each component step by step to isolate crashes.
"""

import pcbnew
import wx
import sys
import traceback

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Progressive debug version to isolate crash causes"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Progressive Debug)"
        self.category = "Routing"
        self.description = "Step-by-step debug version"
        self.show_toolbar_button = True
    
    def Run(self):
        """Progressive debug run method"""
        try:
            print("🔍 Progressive Debug - Starting...")
            
            # Step 1: Basic functionality test
            result = wx.MessageBox(
                "Step 1: Basic Test\n\n" +
                "Click YES to continue with more tests\n" +
                "Click NO to stop here",
                "Progressive Debug - Step 1", 
                wx.YES_NO | wx.ICON_QUESTION
            )
            
            if result != wx.YES:
                print("🛑 User stopped at Step 1")
                return
            
            # Step 2: Test board access
            try:
                board = pcbnew.GetBoard()
                if not board:
                    wx.MessageBox("No board found", "Step 2 Error", wx.OK | wx.ICON_ERROR)
                    return
                print("✅ Step 2 passed: Board access")
                
                result = wx.MessageBox(
                    "Step 2: Board Access ✅\n\n" +
                    "Click YES to test imports\n" +
                    "Click NO to stop here",
                    "Progressive Debug - Step 2", 
                    wx.YES_NO | wx.ICON_QUESTION
                )
                
                if result != wx.YES:
                    print("🛑 User stopped at Step 2")
                    return
                    
            except Exception as e:
                wx.MessageBox(f"Step 2 failed: {e}", "Step 2 Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Step 3: Test individual imports (this might be the crash point)
            try:
                print("🔍 Step 3: Testing imports individually...")
                
                # Test basic imports first
                import json
                import tempfile
                import os
                print("✅ Basic Python modules imported")
                
                # Test numpy (common crash point)
                try:
                    import numpy as np
                    print("✅ NumPy imported successfully")
                except Exception as e:
                    print(f"❌ NumPy import failed: {e}")
                    wx.MessageBox(f"NumPy import failed: {e}", "Import Error", wx.OK | wx.ICON_ERROR)
                    return
                
                result = wx.MessageBox(
                    "Step 3: Basic Imports ✅\n\n" +
                    "Click YES to test CuPy (potential crash point)\n" +
                    "Click NO to stop here",
                    "Progressive Debug - Step 3", 
                    wx.YES_NO | wx.ICON_QUESTION
                )
                
                if result != wx.YES:
                    print("🛑 User stopped at Step 3")
                    return
                    
            except Exception as e:
                print(f"❌ Step 3 failed: {e}")
                wx.MessageBox(f"Step 3 import test failed: {e}", "Step 3 Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Step 4: Test CuPy import (MAJOR crash risk)
            try:
                print("🔍 Step 4: Testing CuPy import - HIGH CRASH RISK!")
                
                # This is likely where KiCad crashes
                import cupy as cp
                print("✅ CuPy imported without crash!")
                
                # Test basic CuPy operation
                test_array = cp.array([1, 2, 3])
                print("✅ CuPy basic operation successful")
                
                result = wx.MessageBox(
                    "Step 4: CuPy Import ✅\n\n" +
                    "Amazing! CuPy didn't crash KiCad!\n" +
                    "Click YES to test local module imports\n" +
                    "Click NO to stop here",
                    "Progressive Debug - Step 4", 
                    wx.YES_NO | wx.ICON_QUESTION
                )
                
                if result != wx.YES:
                    print("🛑 User stopped at Step 4")
                    return
                    
            except ImportError as e:
                print(f"⚠️ CuPy not available: {e}")
                wx.MessageBox(f"CuPy not installed: {e}\n\nThis is expected.", "Step 4 Info", wx.OK | wx.ICON_INFORMATION)
                
                result = wx.MessageBox(
                    "Step 4: CuPy Not Available ⚠️\n\n" +
                    "Click YES to test local module imports\n" +
                    "Click NO to stop here",
                    "Progressive Debug - Step 4", 
                    wx.YES_NO | wx.ICON_QUESTION
                )
                
                if result != wx.YES:
                    print("🛑 User stopped at Step 4")
                    return
                    
            except Exception as e:
                print(f"💥 CRASH LIKELY HERE: CuPy import error: {e}")
                traceback.print_exc()
                wx.MessageBox(
                    f"Step 4 CRASH POINT FOUND!\n\n" +
                    f"CuPy import error: {e}\n\n" +
                    "This is likely why KiCad quits!\n" +
                    "CuPy may be incompatible with your system.",
                    "CRASH POINT IDENTIFIED", 
                    wx.OK | wx.ICON_ERROR
                )
                return
            
            # Step 5: Test local module imports
            try:
                print("🔍 Step 5: Testing local module imports...")
                
                # Test each local module individually
                try:
                    from . import visualization
                    print("✅ visualization module imported")
                except Exception as e:
                    print(f"❌ visualization import failed: {e}")
                    wx.MessageBox(f"Visualization module error: {e}", "Step 5 Error", wx.OK | wx.ICON_ERROR)
                    return
                
                try:
                    from . import orthoroute_engine
                    print("✅ orthoroute_engine module imported")
                except Exception as e:
                    print(f"❌ orthoroute_engine import failed: {e}")
                    wx.MessageBox(f"OrthoRoute engine error: {e}", "Step 5 Error", wx.OK | wx.ICON_ERROR)
                    return
                
                wx.MessageBox(
                    "Step 5: Local Modules ✅\n\n" +
                    "All imports successful!\n" +
                    "The crash must be in the complex logic.",
                    "Progressive Debug - Step 5 Complete", 
                    wx.OK | wx.ICON_INFORMATION
                )
                
            except Exception as e:
                print(f"❌ Step 5 failed: {e}")
                traceback.print_exc()
                wx.MessageBox(f"Step 5 local import failed: {e}", "Step 5 Error", wx.OK | wx.ICON_ERROR)
                return
            
            print("✅ All progressive debug steps completed successfully!")
            wx.MessageBox(
                "Progressive Debug Complete! ✅\n\n" +
                "All basic functionality works.\n" +
                "The crash must be in complex operations.\n\n" +
                "Check console for detailed logs.",
                "Debug Complete", 
                wx.OK | wx.ICON_INFORMATION
            )
            
        except Exception as e:
            print(f"💥 CRITICAL ERROR in progressive debug: {e}")
            traceback.print_exc()
            try:
                wx.MessageBox(f"Critical error: {e}", "Critical Error", wx.OK | wx.ICON_ERROR)
            except:
                print("💥 Can't even show error dialog!")

# Register the plugin
OrthoRouteKiCadPlugin().register()
