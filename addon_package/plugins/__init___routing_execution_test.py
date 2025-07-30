"""
OrthoRoute GPU Autorouter - ROUTING EXECUTION TEST
This version tests actual routing execution with ultra-conservative steps.
"""

import pcbnew
import wx

class OrthoRouteConfigDialog(wx.Dialog):
    """Simple configuration dialog"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute - Routing Execution Test", size=(400, 300))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        label = wx.StaticText(panel, label="Routing Execution Test\n\nThis tests the actual routing execution with safety bailouts.")
        sizer.Add(label, 0, wx.ALL | wx.CENTER, 20)
        
        algorithm_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Algorithm")
        self.algorithm_choice = wx.Choice(panel, choices=["Lee's Algorithm (Test)", "Ultra-Safe Mode"])
        self.algorithm_choice.SetSelection(0)
        algorithm_box.Add(self.algorithm_choice, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(algorithm_box, 0, wx.EXPAND | wx.ALL, 10)
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        test_btn = wx.Button(panel, wx.ID_OK, "Test Routing Execution")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(test_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        self.CenterOnParent()
    
    def get_config(self):
        return {
            'algorithm': self.algorithm_choice.GetStringSelection(),
            'test_mode': True,
            'grid_pitch': 0.2,      # Large grid for safety
            'max_iterations': 1,    # Single iteration only
            'batch_size': 1,        # One net at a time
            'enable_visualization': False,  # Disable viz for safety
            'ultra_safe': True
        }

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Routing execution test version"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Execution Test)"
        self.category = "Routing"
        self.description = "Tests actual routing execution with conservative settings"
        self.show_toolbar_button = True
    
    def Run(self):
        """Run with routing execution test"""
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
                
                # Test routing execution
                self._test_routing_execution(board, config)
                
            else:
                dlg.Destroy()
                
        except Exception as e:
            print(f"❌ Error in routing execution test: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Routing execution test error: {str(e)}", 
                        "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
    
    def _test_routing_execution(self, board, config):
        """Test routing execution with ultra-conservative steps"""
        try:
            print("🔍 Starting routing execution test...")
            wx.MessageBox("DEBUG: Starting routing execution test", "Debug", wx.OK | wx.ICON_INFORMATION)
            
            # Step 1: Setup (we know this works)
            print("🔍 Step 1: Setup...")
            from .orthoroute_engine import OrthoRouteEngine
            engine = OrthoRouteEngine()
            print("✅ Engine created")
            wx.MessageBox("DEBUG: Step 1 complete - Engine ready", "Debug", wx.OK | wx.ICON_INFORMATION)
            
            # Step 2: Board data export (minimal)
            print("🔍 Step 2: Minimal board data export...")
            try:
                from .board_exporter import BoardDataExporter
                exporter = BoardDataExporter()
                
                # Get just basic board info
                board_bounds = board.GetBoardEdgesBoundingBox()
                nets = board.GetNetInfo()
                net_count = nets.GetNetCount()
                
                # Create minimal board data structure (don't export everything)
                board_data = {
                    'bounds': {
                        'min_x': board_bounds.GetLeft(),
                        'min_y': board_bounds.GetTop(),
                        'max_x': board_bounds.GetRight(),
                        'max_y': board_bounds.GetBottom()
                    },
                    'layers': 2,  # Assume 2-layer for safety
                    'nets': []   # Start with empty nets list
                }
                
                print(f"✅ Minimal board data: {net_count} nets total")
                wx.MessageBox(f"DEBUG: Step 2 complete - {net_count} nets found", "Debug", wx.OK | wx.ICON_INFORMATION)
                
            except Exception as e:
                print(f"❌ Board data export failed: {e}")
                wx.MessageBox(f"DEBUG: Step 2 FAILED - Export: {e}", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Step 3: Test engine initialization (this might crash)
            print("🔍 Step 3: Engine initialization...")
            try:
                # Try to initialize engine with minimal data
                print("🔍 Step 3a: Calling engine.initialize_routing...")
                
                # Create ultra-minimal config
                minimal_config = {
                    'grid_pitch': 0.5,  # Very large grid
                    'max_iterations': 1,
                    'batch_size': 1,
                    'enable_gpu': False,  # Force CPU mode for safety
                    'enable_visualization': False
                }
                
                print("✅ About to call engine initialize...")
                wx.MessageBox("DEBUG: Step 3a - About to initialize engine", "Debug", wx.OK | wx.ICON_INFORMATION)
                
                # THIS MIGHT BE WHERE IT CRASHES
                result = engine.initialize_routing(board_data, minimal_config)
                
                print(f"✅ Engine initialized: {result}")
                wx.MessageBox("DEBUG: Step 3a complete - Engine initialized", "Debug", wx.OK | wx.ICON_INFORMATION)
                
            except Exception as e:
                print(f"❌ Engine initialization failed: {e}")
                import traceback
                print(f"❌ Initialization traceback: {traceback.format_exc()}")
                wx.MessageBox(f"DEBUG: Step 3 FAILED - Initialization: {e}", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Step 4: Test actual routing (single net, minimal)
            print("🔍 Step 4: Test minimal routing...")
            try:
                print("🔍 Step 4a: About to call route_nets...")
                wx.MessageBox("DEBUG: Step 4a - About to start routing", "Debug", wx.OK | wx.ICON_INFORMATION)
                
                # Try routing with empty nets list (should be safe)
                routing_result = engine.route_nets([], minimal_config)
                
                print(f"✅ Routing completed: {routing_result}")
                wx.MessageBox("DEBUG: Step 4 complete - Routing finished", "Debug", wx.OK | wx.ICON_INFORMATION)
                
            except Exception as e:
                print(f"❌ Routing execution failed: {e}")
                import traceback
                print(f"❌ Routing traceback: {traceback.format_exc()}")
                wx.MessageBox(f"DEBUG: Step 4 FAILED - Routing: {e}", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # If we get here, show success
            wx.MessageBox(
                "Routing Execution Test PASSED!\n\n" +
                "All routing steps completed:\n" +
                "✅ Engine creation\n" +
                "✅ Board data export\n" +
                "✅ Engine initialization\n" +
                "✅ Routing execution\n\n" +
                "The crash must be in specific routing algorithm or GPU operations.",
                "Execution Test Success", 
                wx.OK | wx.ICON_INFORMATION
            )
            
        except Exception as e:
            print(f"❌ Routing execution test failed: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Routing execution test failed: {str(e)}", 
                        "Execution Test Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRouteKiCadPlugin().register()
