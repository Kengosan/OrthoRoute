"""
OrthoRoute GPU Autorouter - ULTRA MINIMAL VERSION
This version has absolutely minimal functionality to test basic plugin loading.
"""

import pcbnew
import wx

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Ultra minimal version - just shows a message"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Ultra Safe)"
        self.category = "Routing"
        self.description = "Ultra minimal test version"
        self.show_toolbar_button = True
    
    def Run(self):
        """Ultra minimal run method"""
        try:
            wx.MessageBox(
                "Ultra Safe Mode Test\n\n" +
                "If you see this message, basic plugin functionality works.\n" + 
                "KiCad should NOT crash when you click OK.",
                "Ultra Safe Mode", 
                wx.OK | wx.ICON_INFORMATION
            )
            print("✅ Ultra safe mode completed successfully")
            
        except Exception as e:
            print(f"💥 Error even in ultra safe mode: {e}")

# Register the plugin
OrthoRouteKiCadPlugin().register()
