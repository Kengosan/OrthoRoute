"""
OrthoRoute GPU Autorouter - ISOLATED TEST VERSION
This version is completely isolated from all other modules.
"""

import pcbnew
import wx

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Completely isolated test version"""
    
    def defaults(self):
        self.name = "OrthoRoute Test (Isolated)"
        self.category = "Routing"
        self.description = "Isolated test - no module dependencies"
        self.show_toolbar_button = True
    
    def Run(self):
        """Completely isolated test"""
        try:
            wx.MessageBox(
                "ISOLATED TEST VERSION\n\n" +
                "This version has ZERO dependencies on our modules.\n" +
                "If this works, the issue is in our module imports.\n" +
                "If this crashes, the issue is with KiCad/Python itself.",
                "Isolated Test", 
                wx.OK | wx.ICON_INFORMATION
            )
            
        except Exception as e:
            wx.MessageBox(f"Error in isolated test: {e}", "Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRouteKiCadPlugin().register()
