import os
import sys
import pcbnew
import wx

# Add the plugin directory to the Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

try:
    from orthoroute_kicad import OrthoRouteKiCadPlugin
    
    class OrthoRoutePluginLoader(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "OrthoRoute GPU Autorouter"
            self.category = "Autorouter"
            self.description = "GPU-accelerated PCB autorouting using CuPy"
            self.show_toolbar_button = True
            self.icon_file_name = "icon.png"
        
        def Run(self):
            try:
                plugin = OrthoRouteKiCadPlugin()
                plugin.Run()
            except Exception as e:
                wx.MessageBox(f"Error running OrthoRoute: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

    # Register the plugin
    OrthoRoutePluginLoader().register()
    
except Exception as e:
    import traceback
    wx.MessageBox(f"Failed to load OrthoRoute plugin: {str(e)}\n\n{traceback.format_exc()}", "Error", wx.OK | wx.ICON_ERROR)
