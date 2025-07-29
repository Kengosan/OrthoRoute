import os
import pcbnew
import wx
import wx.lib.newevent
import threading
import time
import json
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Custom events for threading communication
UpdateProgressEvent, EVT_UPDATE_PROGRESS = wx.lib.newevent.NewEvent()

class OrthoRouteConfigDialog(wx.Dialog):
    """Main configuration dialog for OrthoRoute settings"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute GPU Autorouter Configuration", 
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.config = self._get_default_config()
        self._create_ui()
        
        # Size and center
        self.SetSize((400, 300))
        self.CenterOnParent()
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            "via_spacing": 1.0,
            "track_spacing": 0.25,
            "max_vias": 10,
        }
    
    def _create_ui(self):
        """Create the dialog UI"""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="OrthoRoute Configuration")
        font = title.GetFont()
        font.PointSize += 2
        font = font.Bold()
        title.SetFont(font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Settings
        settings = wx.BoxSizer(wx.VERTICAL)
        
        # Via spacing
        via_box = wx.BoxSizer(wx.HORIZONTAL)
        via_label = wx.StaticText(panel, label="Via Spacing (mm):")
        self.via_spacing = wx.SpinCtrlDouble(panel, value="1.0", min=0.1, max=10.0, inc=0.1)
        via_box.Add(via_label, 0, wx.ALL | wx.CENTER, 5)
        via_box.Add(self.via_spacing, 1, wx.ALL | wx.EXPAND, 5)
        settings.Add(via_box, 0, wx.ALL | wx.EXPAND, 5)
        
        # Track spacing
        track_box = wx.BoxSizer(wx.HORIZONTAL)
        track_label = wx.StaticText(panel, label="Track Spacing (mm):")
        self.track_spacing = wx.SpinCtrlDouble(panel, value="0.25", min=0.1, max=5.0, inc=0.05)
        track_box.Add(track_label, 0, wx.ALL | wx.CENTER, 5)
        track_box.Add(self.track_spacing, 1, wx.ALL | wx.EXPAND, 5)
        settings.Add(track_box, 0, wx.ALL | wx.EXPAND, 5)
        
        # Max vias
        vias_box = wx.BoxSizer(wx.HORIZONTAL)
        vias_label = wx.StaticText(panel, label="Maximum Vias per Net:")
        self.max_vias = wx.SpinCtrl(panel, value="10", min=1, max=100)
        vias_box.Add(vias_label, 0, wx.ALL | wx.CENTER, 5)
        vias_box.Add(self.max_vias, 1, wx.ALL | wx.EXPAND, 5)
        settings.Add(vias_box, 0, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(settings, 1, wx.ALL | wx.EXPAND, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_btn = wx.Button(panel, wx.ID_OK, "Start Routing")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(ok_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.CENTER, 10)
        
        panel.SetSizer(sizer)
        self.Fit()
    
    def get_config(self):
        """Get the current configuration"""
        return {
            "via_spacing": self.via_spacing.GetValue(),
            "track_spacing": self.track_spacing.GetValue(),
            "max_vias": self.max_vias.GetValue(),
        }

class OrthoRoutePlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter"
        self.category = "Autorouter"
        self.description = "GPU-accelerated PCB autorouting using CuPy"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")
    
    def Run(self):
        """Main plugin entry point"""
        try:
            # Get current board
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board loaded!", "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Show configuration dialog
            config_dialog = OrthoRouteConfigDialog(None)
            result = config_dialog.ShowModal()
            
            if result == wx.ID_OK:
                config = config_dialog.get_config()
                progress = wx.ProgressDialog(
                    "OrthoRoute GPU Autorouter",
                    "Starting...",
                    100,
                    style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_ELAPSED_TIME
                )
                progress.Update(10, "Basic routing functionality coming soon!")
                time.sleep(1)
                progress.Destroy()
                wx.MessageBox("Basic config dialog test successful!\n\nConfig values:\n" + 
                            "\n".join(f"{k}: {v}" for k, v in config.items()),
                            "OrthoRoute", wx.OK | wx.ICON_INFORMATION)
            
            config_dialog.Destroy()
            
        except Exception as e:
            wx.MessageBox(f"OrthoRoute Error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRoutePlugin().register()
