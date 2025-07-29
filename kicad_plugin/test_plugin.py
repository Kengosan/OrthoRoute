import os
import pcbnew
import wx

class TestPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "Test Plugin"
        self.category = "Test"
        self.description = "A test plugin"
        self.show_toolbar_button = True
    
    def Run(self):
        wx.MessageBox("Test Plugin Works!", "Test", wx.OK | wx.ICON_INFORMATION)

TestPlugin().register()
