"""
OrthoRoute GPU Autorouter - SAFE ROUTING TEST
This version attempts actual routing with extensive safety checks.
"""

import pcbnew
import wx

class OrthoRouteConfigDialog(wx.Dialog):
    """Simple configuration dialog"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Configuration (Safe Routing)", size=(400, 300))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        label = wx.StaticText(panel, label="Safe Routing Test\n\nThis tests routing execution with safety checks.")
        sizer.Add(label, 0, wx.ALL | wx.CENTER, 20)
        
        algorithm_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Algorithm")
        self.algorithm_choice = wx.Choice(panel, choices=["Lee's Algorithm (Safe)", "A* Algorithm (Safe)"])
        self.algorithm_choice.SetSelection(0)
        algorithm_box.Add(self.algorithm_choice, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(algorithm_box, 0, wx.EXPAND | wx.ALL, 10)
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        test_btn = wx.Button(panel, wx.ID_OK, "Start Safe Routing")
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
            'grid_pitch': 0.1,
            'max_iterations': 1,  # Keep very low for safety
            'batch_size': 1       # Process one net at a time
        }

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Safe routing test version"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Safe Test)"
        self.category = "Routing"
        self.description = "Safe routing test version with extensive checks"
        self.show_toolbar_button = True
    
    def Run(self):
        """Run with safe routing test"""
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
                
                # Safe routing test
                self._safe_routing_test(board, config)
                
            else:
                dlg.Destroy()
                
        except Exception as e:
            print(f"❌ Error in safe routing test: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Safe routing test error: {str(e)}", 
                        "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
    
    def _safe_routing_test(self, board, config):
        """Test routing with extensive safety checks and early exits"""
        try:
            print("🔍 Starting safe routing test...")
            wx.MessageBox("DEBUG: Starting safe routing test", "Debug", wx.OK | wx.ICON_INFORMATION)
            
            # Step 1: Test basic board data extraction
            print("🔍 Step 1: Basic board analysis...")
            board_bounds = board.GetBoardEdgesBoundingBox()
            nets = board.GetNetInfo()
            net_count = nets.GetNetCount()
            print(f"✅ Board: {board_bounds.GetWidth()/1e6:.1f}x{board_bounds.GetHeight()/1e6:.1f}mm, {net_count} nets")
            wx.MessageBox(f"DEBUG: Step 1 complete - {net_count} nets found", "Debug", wx.OK | wx.ICON_INFORMATION)
            
            # Step 2: Test module imports (we know these work)
            print("🔍 Step 2: Testing critical imports...")
            try:
                from .orthoroute_engine import OrthoRouteEngine
                print("✅ OrthoRouteEngine imported")
                wx.MessageBox("DEBUG: Step 2 complete - OrthoRouteEngine imported", "Debug", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                print(f"❌ OrthoRouteEngine import failed: {e}")
                wx.MessageBox(f"DEBUG: Step 2 FAILED - Import error: {e}", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Step 3: Test engine creation
            print("🔍 Step 3: Creating routing engine...")
            try:
                engine = OrthoRouteEngine()
                print("✅ OrthoRouteEngine created")
                wx.MessageBox("DEBUG: Step 3 complete - Engine created", "Debug", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                print(f"❌ Engine creation failed: {e}")
                wx.MessageBox(f"DEBUG: Step 3 FAILED - Engine creation: {e}", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Step 4: Test board data export (this might be the problem)
            print("🔍 Step 4: Testing board data export...")
            try:
                from .board_exporter import BoardDataExporter
                exporter = BoardDataExporter()
                print("✅ BoardDataExporter created")
                wx.MessageBox("DEBUG: Step 4a complete - Exporter created", "Debug", wx.OK | wx.ICON_INFORMATION)
                
                # Try actual export - THIS MIGHT CRASH
                print("🔍 Step 4b: Attempting board export...")
                board_data = exporter.export_board_data(board, {})
                print(f"✅ Board data exported: {len(board_data.get('nets', []))} nets")
                wx.MessageBox(f"DEBUG: Step 4b complete - {len(board_data.get('nets', []))} nets exported", "Debug", wx.OK | wx.ICON_INFORMATION)
                
            except Exception as e:
                print(f"❌ Board export failed: {e}")
                import traceback
                print(f"❌ Board export traceback: {traceback.format_exc()}")
                wx.MessageBox(f"DEBUG: Step 4 FAILED - Board export: {e}", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # If we get here, show success
            wx.MessageBox(
                "Safe Routing Test PASSED!\n\n" +
                "All critical steps completed:\n" +
                "✅ Board analysis\n" +
                "✅ Module imports\n" +
                "✅ Engine creation\n" +
                "✅ Board data export\n\n" +
                "The crash must be in actual routing execution.",
                "Safe Test Success", 
                wx.OK | wx.ICON_INFORMATION
            )
            
        except Exception as e:
            print(f"❌ Safe routing test failed: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Safe routing test failed: {str(e)}", 
                        "Safe Test Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRouteKiCadPlugin().register()
