"""
OrthoRoute GPU Autorouter - MODULE IMPORT TEST
This version tests importing the complex modules to isolate the crash.
"""

import pcbnew
import wx

class OrthoRouteConfigDialog(wx.Dialog):
    """Simple configuration dialog for testing"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Configuration (Import Test)", size=(400, 300))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Simple text
        label = wx.StaticText(panel, label="Module Import Test\n\nThis tests importing routing modules without execution.")
        sizer.Add(label, 0, wx.ALL | wx.CENTER, 20)
        
        # Simple controls
        algorithm_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Algorithm")
        self.algorithm_choice = wx.Choice(panel, choices=["Lee's Algorithm (Test)", "A* Algorithm (Test)"])
        self.algorithm_choice.SetSelection(0)
        algorithm_box.Add(self.algorithm_choice, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(algorithm_box, 0, wx.EXPAND | wx.ALL, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        test_btn = wx.Button(panel, wx.ID_OK, "Test Module Imports")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(test_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        self.CenterOnParent()
    
    def get_config(self):
        """Get configuration settings"""
        return {
            'algorithm': self.algorithm_choice.GetStringSelection(),
            'test_mode': True
        }

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Test version with module imports"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Import Test)"
        self.category = "Routing"
        self.description = "Test version with module imports"
        self.show_toolbar_button = True
    
    def Run(self):
        """Run with module import test"""
        try:
            # Check for board
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board found. Please open a PCB first.", 
                            "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Show configuration dialog
            dlg = OrthoRouteConfigDialog(None)
            if dlg.ShowModal() == wx.ID_OK:
                config = dlg.get_config()
                dlg.Destroy()
                
                # Test module imports
                self._test_module_imports()
                
            else:
                dlg.Destroy()
                
        except Exception as e:
            print(f"❌ Error in import test: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Import test error: {str(e)}", 
                        "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
    
    def _test_module_imports(self):
        """Test importing each module individually"""
        import_results = []
        
        try:
            print("🔍 Testing module imports...")
            
            # Test CuPy import
            try:
                print("🔍 Testing CuPy import...")
                import cupy as cp
                print("✅ CuPy imported successfully")
                import_results.append("✅ CuPy: Success")
            except Exception as e:
                print(f"⚠️ CuPy import failed: {e}")
                import_results.append(f"⚠️ CuPy: {e}")
            
            # Test visualization module
            try:
                print("🔍 Testing visualization import...")
                from .visualization import RoutingProgressDialog
                print("✅ Visualization imported successfully")
                import_results.append("✅ Visualization: Success")
            except Exception as e:
                print(f"❌ Visualization import failed: {e}")
                import_results.append(f"❌ Visualization: {e}")
                
            # Test orthoroute_engine module
            try:
                print("🔍 Testing orthoroute_engine import...")
                from .orthoroute_engine import OrthoRouteEngine
                print("✅ OrthoRouteEngine imported successfully")
                import_results.append("✅ OrthoRouteEngine: Success")
            except Exception as e:
                print(f"❌ OrthoRouteEngine import failed: {e}")
                import_results.append(f"❌ OrthoRouteEngine: {e}")
                
            # Test board_exporter module
            try:
                print("🔍 Testing board_exporter import...")
                from .board_exporter import BoardDataExporter
                print("✅ BoardDataExporter imported successfully")
                import_results.append("✅ BoardDataExporter: Success")
            except Exception as e:
                print(f"❌ BoardDataExporter import failed: {e}")
                import_results.append(f"❌ BoardDataExporter: {e}")
                
            # Test route_importer module
            try:
                print("🔍 Testing route_importer import...")
                from .route_importer import RouteImporter
                print("✅ RouteImporter imported successfully")
                import_results.append("✅ RouteImporter: Success")
            except Exception as e:
                print(f"❌ RouteImporter import failed: {e}")
                import_results.append(f"❌ RouteImporter: {e}")
                
            # Test grid_router module
            try:
                print("🔍 Testing grid_router import...")
                from .grid_router import GridBasedRouter
                print("✅ GridBasedRouter imported successfully")
                import_results.append("✅ GridBasedRouter: Success")
            except Exception as e:
                print(f"❌ GridBasedRouter import failed: {e}")
                import_results.append(f"❌ GridBasedRouter: {e}")
            
            # Show results
            results_text = "Module Import Test Results:\n\n" + "\n".join(import_results)
            results_text += "\n\nNo routing execution - imports only."
            
            wx.MessageBox(results_text, "Import Test Results", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            print(f"❌ Module import test failed: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Module import test failed: {str(e)}", 
                        "Import Test Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRouteKiCadPlugin().register()
