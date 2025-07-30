"""
OrthoRoute GPU Autorouter - CONFIG DIALOG TEST
This version adds just the configuration dialog to test UI components.
"""

import pcbnew
import wx

class OrthoRouteConfigDialog(wx.Dialog):
    """Simple configuration dialog for testing"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Configuration (Test)", size=(400, 300))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Simple text
        label = wx.StaticText(panel, label="Configuration Dialog Test\n\nThis tests if the dialog works without crashing.")
        sizer.Add(label, 0, wx.ALL | wx.CENTER, 20)
        
        # Simple controls
        algorithm_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Algorithm")
        self.algorithm_choice = wx.Choice(panel, choices=["Lee's Algorithm (Test)", "A* Algorithm (Test)"])
        self.algorithm_choice.SetSelection(0)
        algorithm_box.Add(self.algorithm_choice, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(algorithm_box, 0, wx.EXPAND | wx.ALL, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        test_btn = wx.Button(panel, wx.ID_OK, "Test Mode - No Routing")
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
    """Test version with configuration dialog"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (Config Test)"
        self.category = "Routing"
        self.description = "Test version with configuration dialog"
        self.show_toolbar_button = True
    
    def Run(self):
        """Run with configuration dialog test"""
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
                
                # Show success message instead of routing
                wx.MessageBox(
                    f"Configuration Test Successful!\n\n" +
                    f"Selected Algorithm: {config['algorithm']}\n" +
                    f"Test Mode: {config['test_mode']}\n\n" +
                    f"No actual routing performed - this is just a dialog test.",
                    "Config Test Success", 
                    wx.OK | wx.ICON_INFORMATION
                )
            else:
                dlg.Destroy()
                
        except Exception as e:
            print(f"❌ Error in config test: {e}")
            wx.MessageBox(f"Config test error: {str(e)}", 
                        "OrthoRoute Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
OrthoRouteKiCadPlugin().register()
