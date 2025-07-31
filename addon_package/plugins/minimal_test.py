"""
Minimal Test Plugin
==================
Simplest possible KiCad plugin to test if basic loading works
"""

import pcbnew
import wx
import os
import datetime

class MinimalTestPlugin(pcbnew.ActionPlugin):
    """Minimal test plugin"""
    
    def defaults(self):
        self.name = "Minimal Test Plugin"
        self.category = "Test"
        self.description = "Minimal test plugin to verify loading"
        self.show_toolbar_button = True
        
    def Run(self):
        """Minimal test"""
        # Create log file
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(desktop_path, f"Minimal_Plugin_Test_{timestamp}.txt")
        
        with open(log_path, 'w') as f:
            f.write(f"Minimal plugin ran successfully at {datetime.datetime.now()}\n")
            f.write(f"Board: {pcbnew.GetBoard().GetFileName() if pcbnew.GetBoard() else 'No board'}\n")
        
        wx.MessageBox(
            f"Minimal Test Plugin Works!\n\nLog saved to:\n{log_path}",
            "Success",
            wx.OK | wx.ICON_INFORMATION
        )
