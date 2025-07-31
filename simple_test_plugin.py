"""
Simple OrthoRoute Test - Minimal Version
Tests basic functionality without CuPy requirements
"""

import pcbnew
import wx
import traceback
import os

class SimpleDebugDialog(wx.Dialog):
    """Simple debug output window"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Debug", size=(600, 400))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.text_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        
        close_btn = wx.Button(panel, wx.ID_CLOSE, "Close")
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_CLOSE))
        sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        self.Centre()
        
    def append_text(self, text):
        self.text_ctrl.AppendText(text + "\n")

class SimpleConfigDialog(wx.Dialog):
    """Simple configuration dialog"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Test", size=(400, 200))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Just a debug mode checkbox
        self.debug_cb = wx.CheckBox(panel, label="Enable Debug Output")
        sizer.Add(self.debug_cb, 0, wx.ALL, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_btn = wx.Button(panel, wx.ID_OK, "Test Plugin")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(ok_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        self.Centre()
    
    def get_debug_mode(self):
        return self.debug_cb.GetValue()

class OrthoRouteSimplePlugin(pcbnew.ActionPlugin):
    """Simplified test plugin"""
    
    def defaults(self):
        self.name = "OrthoRoute Simple Test"
        self.category = "Test"
        self.description = "Simple test of OrthoRoute functionality"
        self.show_toolbar_button = False
    
    def Run(self):
        """Main plugin entry point"""
        try:
            # Get current board
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board found. Please open a PCB first.", 
                            "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Show config dialog
            config_dlg = SimpleConfigDialog(None)
            if config_dlg.ShowModal() == wx.ID_OK:
                debug_mode = config_dlg.get_debug_mode()
                config_dlg.Destroy()
                
                # Create debug window if requested
                debug_dialog = None
                if debug_mode:
                    debug_dialog = SimpleDebugDialog(None)
                    debug_dialog.Show()
                    
                    def debug_print(text):
                        print(text)
                        debug_dialog.append_text(text)
                else:
                    debug_print = print
                
                # Test board analysis
                self.test_board_analysis(board, debug_print)
                
                if debug_dialog:
                    debug_dialog.append_text("Test complete!")
                    
            else:
                config_dlg.Destroy()
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            wx.MessageBox(error_msg, "Plugin Error", wx.OK | wx.ICON_ERROR)
    
    def test_board_analysis(self, board, debug_print):
        """Test basic board analysis"""
        debug_print("=== ORTHOROUTE BOARD ANALYSIS TEST ===")
        
        # Board dimensions
        bbox = board.GetBoardEdgesBoundingBox()
        width_mm = bbox.GetWidth() / 1e6
        height_mm = bbox.GetHeight() / 1e6
        debug_print(f"Board size: {width_mm:.2f} x {height_mm:.2f} mm")
        debug_print(f"Layers: {board.GetCopperLayerCount()}")
        
        # Footprints
        footprints = list(board.GetFootprints())
        debug_print(f"Footprints: {len(footprints)}")
        
        # Count pads
        total_pads = 0
        for fp in footprints:
            pads = list(fp.Pads())
            total_pads += len(pads)
            if len(footprints) <= 5:  # Show details for first few
                debug_print(f"  {fp.GetReference()}: {len(pads)} pads")
        
        debug_print(f"Total pads: {total_pads}")
        
        # Nets analysis
        net_count = board.GetNetCount()
        debug_print(f"Total nets: {net_count}")
        
        nets_with_pads = 0
        routeable_nets = 0
        
        for netcode in range(1, net_count):  # Skip net 0
            net_info = board.GetNetInfo().GetNetItem(netcode)
            if not net_info:
                continue
                
            net_name = net_info.GetNetname()
            
            # Count pads for this net
            pad_count = 0
            for fp in footprints:
                for pad in fp.Pads():
                    pad_net = pad.GetNet()
                    if pad_net and pad_net.GetNetCode() == netcode:
                        pad_count += 1
            
            if pad_count > 0:
                nets_with_pads += 1
                if nets_with_pads <= 5:  # Show first few
                    debug_print(f"  Net {netcode} '{net_name}': {pad_count} pads")
                
                if pad_count >= 2:
                    routeable_nets += 1
        
        debug_print(f"Nets with pads: {nets_with_pads}")
        debug_print(f"Routeable nets (2+ pads): {routeable_nets}")
        
        if routeable_nets == 0:
            debug_print("WARNING: No routeable nets found!")
            debug_print("Check that your board has components with connected nets")
            debug_print("Try 'Update PCB from Schematic' in KiCad")

# Register the plugin
OrthoRouteSimplePlugin().register()
