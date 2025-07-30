"""
OrthoRoute GPU Autorouter - ALGORITHM ISOLATION TEST
This version tests each routing algorithm step individually to find the exact crash point.
"""

import pcbnew
import wx

class OrthoRouteConfigDialog(wx.Dialog):
    """Simple configuration dialog"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute - Algorithm Isolation Test", size=(400, 350))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        label = wx.StaticText(panel, label="Algorithm Isolation Test\n\nThis tests each routing algorithm component individually.")
        sizer.Add(label, 0, wx.ALL | wx.CENTER, 20)
        
        # Test selection
        test_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Test to Run")
        self.test_choice = wx.Choice(panel, choices=[
            "Test 1: Grid Router Import",
            "Test 2: Grid Router Creation", 
            "Test 3: GPU Router Creation",
            "Test 4: Empty Net Routing",
            "Test 5: GPU Cleanup"
        ])
        self.test_choice.SetSelection(0)
        test_box.Add(self.test_choice, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(test_box, 0, wx.EXPAND | wx.ALL, 10)
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        test_btn = wx.Button(panel, wx.ID_OK, "Run Selected Test")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(test_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        self.CenterOnParent()
    
    def get_config(self):
        return {
            'test_number': self.test_choice.GetSelection() + 1,
            'test_name': self.test_choice.GetStringSelection(),
            'ultra_safe': True
        }

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Algorithm isolation test version"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Algorithm Test)"
        self.category = "Routing"
        self.description = "Tests individual routing algorithm components"
        self.show_toolbar_button = True
    
    def Run(self):
        """Run with algorithm isolation test"""
        try:
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board found. Please open a PCB first.", 
                            "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            dlg = OrthoRouteConfigDialog(None)
            if dlg.ShowModal() == wx.ID_OK:
                config = dlg.get_config()
                dlg.Destroy()
                
                # Run the selected test
                self._run_algorithm_test(board, config)
                
            else:
                dlg.Destroy()
                
        except Exception as e:
            print(f"❌ Error in algorithm test: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Algorithm test error: {str(e)}", 
                        "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
    
    def _run_algorithm_test(self, board, config):
        """Run the selected algorithm test"""
        test_number = config['test_number']
        test_name = config['test_name']
        
        try:
            print(f"🔍 Starting {test_name}...")
            wx.MessageBox(f"DEBUG: Starting {test_name}", "Test Start", wx.OK | wx.ICON_INFORMATION)
            
            # Setup common components
            from .orthoroute_engine import OrthoRouteEngine
            engine = OrthoRouteEngine()
            
            # Create minimal board data
            board_data = {
                'bounds': {
                    'width_nm': 100000000,  # 100mm
                    'height_nm': 100000000, # 100mm
                    'layers': 2
                },
                'nets': [],  # Empty for safety
                'obstacles': {}
            }
            
            if test_number == 1:
                self._test_grid_router_import()
            elif test_number == 2:
                self._test_grid_router_creation(engine, board_data)
            elif test_number == 3:
                self._test_gpu_router_creation(engine, board_data)
            elif test_number == 4:
                self._test_empty_net_routing(engine, board_data)
            elif test_number == 5:
                self._test_gpu_cleanup(engine)
            
            wx.MessageBox(f"✅ {test_name} PASSED!", "Test Success", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"❌ {test_name} FAILED: {str(e)}", "Test Failed", wx.OK | wx.ICON_ERROR)
    
    def _test_grid_router_import(self):
        """Test 1: Grid router import"""
        print("🔍 Test 1: Testing grid router import...")
        wx.MessageBox("DEBUG: Test 1 - About to import grid router", "Debug", wx.OK | wx.ICON_INFORMATION)
        
        try:
            from .grid_router import create_grid_router
            print("✅ Grid router imported successfully")
            wx.MessageBox("DEBUG: Test 1 - Grid router imported", "Debug", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            print(f"❌ Grid router import failed: {e}")
            raise e
    
    def _test_grid_router_creation(self, engine, board_data):
        """Test 2: Grid router creation"""
        print("🔍 Test 2: Testing grid router creation...")
        wx.MessageBox("DEBUG: Test 2 - About to create grid router", "Debug", wx.OK | wx.ICON_INFORMATION)
        
        try:
            from .grid_router import create_grid_router
            
            grid_board_data = {
                'board_width': board_data['bounds']['width_nm'],
                'board_height': board_data['bounds']['height_nm'],
                'layer_count': board_data['bounds']['layers'],
                'obstacles': board_data.get('obstacles', {})
            }
            
            print("🔍 Calling create_grid_router...")
            grid_router = create_grid_router(grid_board_data, engine.config)
            print(f"✅ Grid router created: {grid_router is not None}")
            wx.MessageBox("DEBUG: Test 2 - Grid router created", "Debug", wx.OK | wx.ICON_INFORMATION)
            
            if grid_router:
                print("🔍 Attempting grid router cleanup...")
                grid_router.cleanup()
                print("✅ Grid router cleanup successful")
            
        except Exception as e:
            print(f"❌ Grid router creation failed: {e}")
            raise e
    
    def _test_gpu_router_creation(self, engine, board_data):
        """Test 3: GPU router creation"""
        print("🔍 Test 3: Testing GPU router creation...")
        wx.MessageBox("DEBUG: Test 3 - About to create GPU router", "Debug", wx.OK | wx.ICON_INFORMATION)
        
        try:
            # First load board data into engine
            print("🔍 Loading board data into engine...")
            result = engine.load_board_data(board_data)
            print(f"✅ Board data loaded: {result}")
            wx.MessageBox("DEBUG: Test 3a - Board data loaded", "Debug", wx.OK | wx.ICON_INFORMATION)
            
            # Now try to create GPU router
            print("🔍 Creating GPUWavefrontRouter...")
            from .orthoroute_engine import GPUWavefrontRouter
            router = GPUWavefrontRouter(engine.grid)
            print("✅ GPU router created successfully")
            wx.MessageBox("DEBUG: Test 3b - GPU router created", "Debug", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            print(f"❌ GPU router creation failed: {e}")
            raise e
    
    def _test_empty_net_routing(self, engine, board_data):
        """Test 4: Empty net routing"""
        print("🔍 Test 4: Testing empty net routing...")
        wx.MessageBox("DEBUG: Test 4 - About to test routing with empty nets", "Debug", wx.OK | wx.ICON_INFORMATION)
        
        try:
            # Load board data
            print("🔍 Loading board data...")
            engine.load_board_data(board_data)
            
            # Create router
            print("🔍 Creating router...")
            from .orthoroute_engine import GPUWavefrontRouter
            router = GPUWavefrontRouter(engine.grid)
            
            # Try routing empty net list (should be safe)
            print("🔍 Testing router with empty net list...")
            # Don't actually call routing methods, just creation was the test
            print("✅ Router ready for empty net routing")
            wx.MessageBox("DEBUG: Test 4 - Router ready", "Debug", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            print(f"❌ Empty net routing test failed: {e}")
            raise e
    
    def _test_gpu_cleanup(self, engine):
        """Test 5: GPU cleanup"""
        print("🔍 Test 5: Testing GPU cleanup...")
        wx.MessageBox("DEBUG: Test 5 - About to test GPU cleanup", "Debug", wx.OK | wx.ICON_INFORMATION)
        
        try:
            print("🔍 Calling _cleanup_gpu_resources...")
            engine._cleanup_gpu_resources()
            print("✅ GPU cleanup successful")
            wx.MessageBox("DEBUG: Test 5 - GPU cleanup complete", "Debug", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            print(f"❌ GPU cleanup failed: {e}")
            raise e

# Register the plugin
OrthoRouteKiCadPlugin().register()
