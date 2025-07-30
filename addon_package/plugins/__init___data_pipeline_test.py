"""
OrthoRoute GPU Autorouter - DATA PIPELINE TEST
This version tests the data flow to debug why no routing appears on the PCB.
"""

import pcbnew
import wx

class OrthoRouteConfigDialog(wx.Dialog):
    """Simple configuration dialog"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute - Data Pipeline Test", size=(400, 300))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        label = wx.StaticText(panel, label="Data Pipeline Test\n\nThis tests the data flow to see why no tracks appear.")
        sizer.Add(label, 0, wx.ALL | wx.CENTER, 20)
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        test_btn = wx.Button(panel, wx.ID_OK, "Test Data Pipeline")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(test_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        self.CenterOnParent()

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Data pipeline test version"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Data Pipeline Test)"
        self.category = "Routing"
        self.description = "Tests data pipeline to debug why no tracks appear"
        self.show_toolbar_button = True
    
    def Run(self):
        """Run data pipeline test"""
        try:
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board found. Please open a PCB first.", 
                            "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            dlg = OrthoRouteConfigDialog(None)
            if dlg.ShowModal() == wx.ID_OK:
                dlg.Destroy()
                self._test_data_pipeline(board)
            else:
                dlg.Destroy()
                
        except Exception as e:
            print(f"❌ Error in data pipeline test: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Data pipeline test error: {str(e)}", 
                        "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
    
    def _test_data_pipeline(self, board):
        """Test the complete data pipeline"""
        try:
            print("🔍 Starting data pipeline test...")
            
            # Step 1: Basic board info
            print("\n=== STEP 1: BASIC BOARD INFO ===")
            board_bounds = board.GetBoardEdgesBoundingBox()
            print(f"Board bounds: {board_bounds.GetWidth()/1e6:.1f}x{board_bounds.GetHeight()/1e6:.1f}mm")
            print(f"Board position: ({board_bounds.GetLeft()/1e6:.1f}, {board_bounds.GetTop()/1e6:.1f})mm")
            
            # Step 2: Net analysis
            print("\n=== STEP 2: NET ANALYSIS ===")
            nets = board.GetNetInfo()
            net_count = nets.GetNetCount()
            print(f"Total nets found: {net_count}")
            
            # Show first 10 nets
            print("Net details:")
            for i in range(min(10, net_count)):
                net = nets.GetNet(i)
                if net:
                    print(f"  Net {i}: '{net.GetNetname()}' (code: {net.GetNetCode()})")
            
            # Step 3: Pad analysis  
            print("\n=== STEP 3: PAD ANALYSIS ===")
            footprints = board.GetFootprints()
            total_pads = 0
            net_to_pads = {}
            
            for footprint in footprints:
                pads = footprint.GetPads()
                for pad in pads:
                    total_pads += 1
                    net = pad.GetNet()
                    if net:
                        net_name = net.GetNetname()
                        if net_name not in net_to_pads:
                            net_to_pads[net_name] = []
                        net_to_pads[net_name].append({
                            'footprint': footprint.GetReference(),
                            'pad': pad.GetNumber(),
                            'pos': pad.GetPosition(),
                            'layer': pad.GetLayerSet().Seq()
                        })
            
            print(f"Total pads found: {total_pads}")
            print(f"Nets with pads: {len(net_to_pads)}")
            
            # Show nets with multiple pads (these need routing)
            routeable_nets = []
            for net_name, pads in net_to_pads.items():
                if len(pads) >= 2 and net_name not in ['', 'No Net']:  # Need at least 2 pads to route
                    routeable_nets.append((net_name, pads))
            
            print(f"Nets that need routing: {len(routeable_nets)}")
            
            # Show first few routeable nets
            print("\nRouteable nets (first 5):")
            for i, (net_name, pads) in enumerate(routeable_nets[:5]):
                print(f"  {net_name}: {len(pads)} pads")
                for j, pad in enumerate(pads[:3]):  # Show first 3 pads
                    pos = pad['pos']
                    print(f"    Pad {j+1}: {pad['footprint']}.{pad['pad']} at ({pos.x/1e6:.2f}, {pos.y/1e6:.2f})mm")
                if len(pads) > 3:
                    print(f"    ... and {len(pads)-3} more pads")
            
            # Step 4: Existing track analysis
            print("\n=== STEP 4: EXISTING TRACKS ===")
            tracks = board.GetTracks()
            track_count = len(tracks)
            print(f"Existing tracks: {track_count}")
            
            if track_count > 0:
                print("Track types:")
                track_types = {}
                for track in tracks:
                    track_type = type(track).__name__
                    track_types[track_type] = track_types.get(track_type, 0) + 1
                for track_type, count in track_types.items():
                    print(f"  {track_type}: {count}")
            
            # Summary and recommendations
            print("\n=== SUMMARY AND DIAGNOSIS ===")
            if len(routeable_nets) == 0:
                print("❌ PROBLEM FOUND: No nets need routing!")
                print("   Possible causes:")
                print("   - All nets are already routed")
                print("   - Components aren't assigned to nets")
                print("   - Netlist needs to be updated from schematic")
                print("   - Only single-pad nets exist")
            else:
                print(f"✅ Found {len(routeable_nets)} nets that need routing")
                print("   The routing engine should process these nets")
                print("   If no tracks appear after routing, the problem is likely:")
                print("   - Route results aren't being imported back to KiCad")
                print("   - Routing algorithm is failing silently")
                print("   - Results are being generated but not applied to board")
            
            # Create detailed report
            report = f"""
DATA PIPELINE TEST REPORT
========================

Board Info:
- Size: {board_bounds.GetWidth()/1e6:.1f}x{board_bounds.GetHeight()/1e6:.1f}mm
- Position: ({board_bounds.GetLeft()/1e6:.1f}, {board_bounds.GetTop()/1e6:.1f})mm

Network Analysis:
- Total nets: {net_count}
- Nets with pads: {len(net_to_pads)}
- Routeable nets: {len(routeable_nets)}
- Total pads: {total_pads}
- Existing tracks: {track_count}

Conclusion:
{("✅ Board has routeable nets - routing should work" if len(routeable_nets) > 0 else "❌ No nets need routing - check netlist")}

Next Steps:
{("Test actual routing execution to see why tracks don't appear" if len(routeable_nets) > 0 else "Update netlist from schematic or check component net assignments")}
"""
            
            # Show report in dialog
            report_dlg = wx.Dialog(None, title="Data Pipeline Test Report", 
                                 style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
            report_dlg.SetSize((800, 600))
            
            report_text = wx.TextCtrl(report_dlg, style=wx.TE_MULTILINE | wx.TE_READONLY)
            report_text.SetValue(report)
            
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(report_text, 1, wx.EXPAND | wx.ALL, 10)
            
            close_btn = wx.Button(report_dlg, wx.ID_OK, "Close")
            sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
            
            report_dlg.SetSizer(sizer)
            report_dlg.ShowModal()
            report_dlg.Destroy()
            
        except Exception as e:
            print(f"❌ Data pipeline test failed: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Data pipeline test failed: {str(e)}", 
                        "Test Failed", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRouteKiCadPlugin().register()
