"""
OrthoRoute GPU Autorouter - BOARD ANALYSIS TEST
This version adds board data extraction to test KiCad API calls.
"""

import pcbnew
import wx

class OrthoRouteConfigDialog(wx.Dialog):
    """Simple configuration dialog for testing"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Configuration (Board Test)", size=(400, 300))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Simple text
        label = wx.StaticText(panel, label="Board Analysis Test\n\nThis tests board data extraction without routing.")
        sizer.Add(label, 0, wx.ALL | wx.CENTER, 20)
        
        # Simple controls
        algorithm_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Algorithm")
        self.algorithm_choice = wx.Choice(panel, choices=["Lee's Algorithm (Test)", "A* Algorithm (Test)"])
        self.algorithm_choice.SetSelection(0)
        algorithm_box.Add(self.algorithm_choice, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(algorithm_box, 0, wx.EXPAND | wx.ALL, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        test_btn = wx.Button(panel, wx.ID_OK, "Test Board Analysis")
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
    """Test version with board analysis"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Board Test)"
        self.category = "Routing"
        self.description = "Test version with board analysis"
        self.show_toolbar_button = True
    
    def Run(self):
        """Run with board analysis test"""
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
                
                # Test board analysis
                self._test_board_analysis(board)
                
            else:
                dlg.Destroy()
                
        except Exception as e:
            print(f"❌ Error in board test: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Board test error: {str(e)}", 
                        "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
    
    def _test_board_analysis(self, board):
        """Test basic board data extraction"""
        try:
            print("🔍 Starting board analysis test...")
            
            # Test basic board properties
            board_bounds = board.GetBoardEdgesBoundingBox()
            print(f"✅ Board bounds: {board_bounds.GetWidth()/1e6:.2f} x {board_bounds.GetHeight()/1e6:.2f} mm")
            
            # Test layer count
            layer_count = board.GetCopperLayerCount()
            print(f"✅ Copper layers: {layer_count}")
            
            # Test footprint enumeration
            footprints = board.GetFootprints()
            footprint_count = len(footprints)
            print(f"✅ Footprints found: {footprint_count}")
            
            # Test net enumeration
            nets = board.GetNetInfo()
            net_count = nets.GetNetCount()
            print(f"✅ Nets found: {net_count}")
            
            # Test zones (this is where the crash might be)
            print("🔍 Testing zone enumeration...")
            zones = []
            for zone in board.Zones():
                zones.append(zone)
            print(f"✅ Zones found: {len(zones)}")
            
            # Test tracks
            tracks = []
            for track in board.GetTracks():
                tracks.append(track)
            print(f"✅ Tracks found: {len(tracks)}")
            
            # Show success message
            results_text = (
                f"Board Analysis Test Successful!\n\n"
                f"Board Size: {board_bounds.GetWidth()/1e6:.2f} × {board_bounds.GetHeight()/1e6:.2f} mm\n"
                f"Copper Layers: {layer_count}\n"
                f"Footprints: {footprint_count}\n"
                f"Nets: {net_count}\n"
                f"Zones: {len(zones)}\n"
                f"Tracks: {len(tracks)}\n\n"
                f"No routing performed - analysis only."
            )
            
            wx.MessageBox(results_text, "Board Analysis Success", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            print(f"❌ Board analysis failed: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Board analysis failed: {str(e)}", 
                        "Analysis Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRouteKiCadPlugin().register()
