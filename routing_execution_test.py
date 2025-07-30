"""
OrthoRoute GPU Autorouter - ROUTING EXECUTION TEST
This version tests the actual routing execution to see where tracks disappear.
"""

import pcbnew
import wx
import traceback

class OrthoRouteConfigDialog(wx.Dialog):
    """Simple configuration dialog for routing test"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute - Routing Execution Test", size=(450, 350))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="Routing Execution Test")
        title_font = title.GetFont()
        title_font.SetPointSize(14)
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Info
        info = wx.StaticText(panel, label="This test will:\n"
                                         "• Select a routeable net\n"
                                         "• Create a track manually\n"
                                         "• Verify track creation\n"
                                         "• Check track visibility\n"
                                         "• Determine why tracks don't appear")
        sizer.Add(info, 0, wx.ALL | wx.EXPAND, 10)
        
        # Test options
        test_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Test Options")
        
        self.verbose_cb = wx.CheckBox(panel, label="Verbose logging (console output)")
        self.verbose_cb.SetValue(True)
        test_box.Add(self.verbose_cb, 0, wx.ALL, 5)
        
        self.step_by_step_cb = wx.CheckBox(panel, label="Step-by-step execution with prompts")
        self.step_by_step_cb.SetValue(True)
        test_box.Add(self.step_by_step_cb, 0, wx.ALL, 5)
        
        sizer.Add(test_box, 0, wx.ALL | wx.EXPAND, 10)
        
        # Warning
        warning = wx.StaticText(panel, label="⚠️ This will create a test track on your board.\nMake sure you have a backup!")
        warning.SetForegroundColour(wx.Colour(200, 100, 0))
        sizer.Add(warning, 0, wx.ALL | wx.CENTER, 5)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        test_btn = wx.Button(panel, wx.ID_OK, "Start Routing Test")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        
        btn_sizer.Add(test_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        
        sizer.Add(btn_sizer, 0, wx.ALL | wx.CENTER, 10)
        
        panel.SetSizer(sizer)
        self.Centre()

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Routing execution test version"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Routing Test)"
        self.category = "Routing"
        self.description = "Tests routing execution to debug why no tracks appear"
        self.show_toolbar_button = True
    
    def Run(self):
        """Run routing execution test"""
        try:
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board found. Please open a PCB first.", 
                            "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            dlg = OrthoRouteConfigDialog(None)
            if dlg.ShowModal() == wx.ID_OK:
                verbose = dlg.verbose_cb.GetValue()
                step_by_step = dlg.step_by_step_cb.GetValue()
                dlg.Destroy()
                self._test_routing_execution(board, verbose, step_by_step)
            else:
                dlg.Destroy()
                
        except Exception as e:
            print(f"❌ Error in routing test: {e}")
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Routing test error: {str(e)}", 
                        "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
    
    def _test_routing_execution(self, board, verbose, step_by_step):
        """Test routing execution step by step"""
        try:
            print("\n🔬 ROUTING EXECUTION TEST")
            print("=" * 50)
            
            # Step 1: Pre-routing analysis
            if verbose:
                print("\n=== STEP 1: PRE-ROUTING ANALYSIS ===")
            
            initial_tracks = len(list(board.GetTracks()))
            if verbose:
                print(f"Initial tracks on board: {initial_tracks}")
            
            # Find a net to route
            nets = board.GetNetsByNetcode()
            target_net = None
            target_pads = []
            
            for netcode, net in nets.items():
                if netcode == 0:  # Skip unconnected
                    continue
                
                # Find pads for this net
                net_pads = []
                for footprint in board.GetFootprints():
                    try:
                        pads_collection = footprint.Pads()
                        for pad in pads_collection:
                            if pad.GetNet() and pad.GetNet().GetNetCode() == netcode:
                                net_pads.append(pad)
                    except Exception as e:
                        if verbose:
                            print(f"Error accessing pads for {footprint.GetReference()}: {e}")
                        continue
                
                if len(net_pads) >= 2:  # Need at least 2 pads to route
                    target_net = net
                    target_pads = net_pads
                    if verbose:
                        print(f"Selected net for routing: {net.GetNetname()} (pads: {len(net_pads)})")
                        for i, pad in enumerate(net_pads[:2]):
                            print(f"  Pad {i+1}: {pad.GetParentFootprint().GetReference()}.{pad.GetNumber()} at {pad.GetPosition()}")
                    break
            
            if not target_net:
                wx.MessageBox("No suitable net found for routing test", 
                            "Test Result", wx.OK | wx.ICON_WARNING)
                return
            
            if step_by_step:
                result = wx.MessageBox(f"Found net '{target_net.GetNetname()}' with {len(target_pads)} pads.\n\nProceed with manual track creation test?", 
                                     "Step 1 Complete", wx.YES_NO | wx.ICON_QUESTION)
                if result != wx.YES:
                    return
            
            # Step 2: Create a simple track manually
            if verbose:
                print("\n=== STEP 2: MANUAL TRACK CREATION TEST ===")
            
            pad1, pad2 = target_pads[0], target_pads[1]
            
            if verbose:
                print(f"Creating track between:")
                print(f"  Pad 1: {pad1.GetParentFootprint().GetReference()}.{pad1.GetNumber()}")
                print(f"         Position: {pad1.GetPosition()}")
                print(f"  Pad 2: {pad2.GetParentFootprint().GetReference()}.{pad2.GetNumber()}")
                print(f"         Position: {pad2.GetPosition()}")
            
            # Create a track between the pads
            try:
                track = pcbnew.PCB_TRACK(board)
                track.SetStart(pad1.GetPosition())
                track.SetEnd(pad2.GetPosition())
                track.SetNet(target_net)
                track.SetLayer(pcbnew.F_Cu)  # Front copper
                track.SetWidth(pcbnew.FromMM(0.2))  # 0.2mm width
                
                # Add track to board
                board.Add(track)
                
                if verbose:
                    print(f"✅ Created track:")
                    print(f"   Start: {track.GetStart()}")
                    print(f"   End: {track.GetEnd()}")
                    print(f"   Layer: {track.GetLayerName()}")
                    print(f"   Width: {pcbnew.ToMM(track.GetWidth()):.2f}mm")
                    print(f"   Net: {track.GetNet().GetNetname()}")
                
            except Exception as e:
                error_msg = f"Error creating track: {e}"
                print(f"❌ {error_msg}")
                wx.MessageBox(error_msg, "Test Error", wx.OK | wx.ICON_ERROR)
                return
            
            if step_by_step:
                result = wx.MessageBox("Track created and added to board.\n\nLook at your PCB now - do you see a new track?\n\nClick YES if you see it, NO if you don't see it.", 
                                     "Step 2 Complete - Check Your PCB!", wx.YES_NO | wx.ICON_QUESTION)
                track_visible = (result == wx.YES)
            else:
                track_visible = None
            
            # Step 3: Verify track creation
            if verbose:
                print("\n=== STEP 3: TRACK VERIFICATION ===")
            
            current_tracks = list(board.GetTracks())
            current_count = len(current_tracks)
            
            if verbose:
                print(f"Tracks after creation: {current_count}")
                print(f"Track count increase: {current_count - initial_tracks}")
            
            # Find our track
            our_track = None
            for track in current_tracks:
                if (track.GetNet() and 
                    track.GetNet().GetNetCode() == target_net.GetNetCode() and
                    track.GetStart() == pad1.GetPosition() and
                    track.GetEnd() == pad2.GetPosition()):
                    our_track = track
                    break
            
            if our_track:
                if verbose:
                    print("✅ Track found in board track list")
                    print(f"   UUID: {our_track.m_Uuid}")
                    print(f"   Layer: {our_track.GetLayerName()}")
                    print(f"   Net code: {our_track.GetNet().GetNetCode()}")
            else:
                print("❌ Track not found in board track list!")
            
            # Step 4: Force board refresh
            if verbose:
                print("\n=== STEP 4: BOARD REFRESH ATTEMPT ===")
            
            try:
                # Try to refresh the board display
                if hasattr(pcbnew, 'Refresh'):
                    pcbnew.Refresh()
                    if verbose:
                        print("✅ Called pcbnew.Refresh()")
                else:
                    if verbose:
                        print("⚠️ pcbnew.Refresh() not available")
                
                # Try other refresh methods
                if hasattr(board, 'GetConnectivity'):
                    board.GetConnectivity().RecalculateRatsnest()
                    if verbose:
                        print("✅ Called RecalculateRatsnest()")
                        
            except Exception as e:
                if verbose:
                    print(f"⚠️ Refresh error: {e}")
            
            # Final results summary
            success = our_track is not None
            tracks_added = current_count - initial_tracks
            
            if step_by_step and track_visible is not None:
                visibility_text = "✅ VISIBLE" if track_visible else "❌ NOT VISIBLE"
            else:
                visibility_text = "❓ UNKNOWN (check manually)"
            
            result_msg = f"""ROUTING EXECUTION TEST RESULTS
===============================

Target Net: {target_net.GetNetname()}
Initial Tracks: {initial_tracks}
Final Tracks: {current_count}
Tracks Added: {tracks_added}

Track Creation: {'✅ SUCCESS' if success else '❌ FAILED'}
Track in Board: {'✅ YES' if our_track else '❌ NO'}
Track Visible: {visibility_text}

DIAGNOSIS:
"""
            
            if not success:
                result_msg += "❌ PROBLEM: Track creation failed\n"
                result_msg += "   → Issue with KiCad API usage\n"
                result_msg += "   → Check console for errors"
            elif track_visible == False:
                result_msg += "⚠️ PROBLEM: Track created but not visible\n"
                result_msg += "   → Display/refresh issue\n"
                result_msg += "   → Check layer visibility\n"
                result_msg += "   → Try zooming/panning"
            elif track_visible == True:
                result_msg += "✅ GOOD: Track creation and display work\n"
                result_msg += "   → Problem is in routing algorithm\n"
                result_msg += "   → Algorithm not creating tracks\n"
                result_msg += "   → Check routing logic"
            else:
                result_msg += "❓ CHECK MANUALLY:\n"
                result_msg += "   → Look for new track on board\n"
                result_msg += "   → Check if layers are visible\n"
                result_msg += "   → Try refreshing display"
            
            result_msg += f"\nNEXT STEPS:\n"
            if success and track_visible != False:
                result_msg += "→ Test actual routing algorithm\n"
                result_msg += "→ Check why algorithm doesn't create tracks"
            else:
                result_msg += "→ Fix track creation/display issues first\n"
                result_msg += "→ Check KiCad documentation"
            
            print(f"\n{result_msg}")
            wx.MessageBox(result_msg, "Test Complete", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            error_msg = f"Routing execution test failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            wx.MessageBox(error_msg, "Test Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRouteKiCadPlugin().register()
