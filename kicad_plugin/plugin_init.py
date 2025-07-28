"""
OrthoRoute Plugin Initialization
"""
import os
import sys
import wx

def check_dependencies() -> bool:
    """Check if all required dependencies are available"""
    try:
        import cupy
        import orthoroute
        return True
    except ImportError as e:
        return False

def show_install_instructions():
    """Show installation instructions dialog"""
    message = """
OrthoRoute Dependencies Missing

Please install the required packages:

1. Install OrthoRoute from source:
   git clone https://github.com/bbenchoff/OrthoRoute.git
   cd OrthoRoute
   pip install .

2. Install GPU dependencies:
   pip install cupy-cuda12x  (for CUDA 12.x)
   pip install cupy-cuda11x  (for CUDA 11.x)

3. Restart KiCad

Visit https://github.com/bbenchoff/OrthoRoute for more information.
"""
    wx.MessageBox(message, "OrthoRoute Installation Required", wx.OK | wx.ICON_INFORMATION)

def register():
    """Register the plugin with KiCad"""
    try:
        # Check dependencies first
        if not check_dependencies():
            show_install_instructions()
            return
            
        # Import plugin class only if dependencies are available
        from .orthoroute_kicad import OrthoRouteKiCadPlugin
        OrthoRouteKiCadPlugin().register()
        
    except Exception as e:
        wx.MessageBox(f"Error registering OrthoRoute plugin: {str(e)}",
                     "Plugin Error", wx.OK | wx.ICON_ERROR)
