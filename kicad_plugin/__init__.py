"""
OrthoRoute KiCad Plugin
========================

GPU-accelerated PCB autorouter integration for KiCad
"""

# Register the plugin to KiCad's plugin system
try:
    # Register the main plugin from orthoroute_kicad.py
    from .orthoroute_kicad import OrthoRouteKiCadPlugin
    OrthoRouteKiCadPlugin().register()
    
    # Uncomment the line below if you want to register the other plugin as well
    # from .orthoroute_plugin import OrthoRoute
    # OrthoRoute().register()
except Exception as e:
    import wx
    import traceback
    wx.MessageBox(f"Error loading OrthoRoute plugin: {str(e)}\n{traceback.format_exc()}", 
                "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
