"""
OrthoRoute KiCad Plugin
"""

# Import and register the plugin
try:
    from .orthoroute_kicad import OrthoRouteKiCadPlugin
    
    # Register the plugin
    OrthoRouteKiCadPlugin().register()
    
except Exception as e:
    print(f"Failed to register OrthoRoute plugin: {e}")
    import traceback
    traceback.print_exc()