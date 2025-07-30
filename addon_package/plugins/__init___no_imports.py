"""
OrthoRoute GPU Autorouter - NO IMPORTS VERSION
This version has ZERO imports to test if imports are causing crashes.
"""

import pcbnew
import wx

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """Absolutely minimal - no imports version"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (No Imports)"
        self.category = "Routing"
        self.description = "Zero imports test version"
        self.show_toolbar_button = True
    
    def Run(self):
        """Ultra minimal - no imports"""
        try:
            # Test if the crash happens even without any imports
            result = wx.MessageBox(
                "NO IMPORTS TEST\n\n" +
                "This version imports NOTHING.\n" +
                "If KiCad crashes, the issue is NOT imports.\n\n" +
                "Click OK to test basic dialog.",
                "No Imports Test", 
                wx.OK | wx.ICON_INFORMATION
            )
            
            # Test if creating a simple dialog crashes
            dlg = wx.Dialog(None, title="Simple Test Dialog")
            dlg.SetSize((200, 100))
            
            panel = wx.Panel(dlg)
            sizer = wx.BoxSizer(wx.VERTICAL)
            
            text = wx.StaticText(panel, label="Simple test")
            sizer.Add(text, 0, wx.ALL, 10)
            
            ok_btn = wx.Button(panel, wx.ID_OK, "OK")
            sizer.Add(ok_btn, 0, wx.ALL, 10)
            
            panel.SetSizer(sizer)
            
            if dlg.ShowModal() == wx.ID_OK:
                wx.MessageBox("Dialog test successful!", "Success", wx.OK)
            
            dlg.Destroy()
            
        except Exception as e:
            wx.MessageBox(f"Error even with no imports: {e}", "Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRouteKiCadPlugin().register()
