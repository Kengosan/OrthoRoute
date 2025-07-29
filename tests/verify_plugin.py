import os
import sys

# Add the orthoroute directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

print("OrthoRoute KiCad Plugin Verification")
print("=================================")

# Just import the module to check path resolution
kicad_plugin_path = os.path.join(parent_dir, "kicad_plugin")
print(f"Plugin directory: {kicad_plugin_path}")

# Check for icon.png
icon_path = os.path.join(kicad_plugin_path, "icon.png") 
print(f"Icon path: {icon_path}")
print(f"Icon exists: {os.path.exists(icon_path)}")
if os.path.exists(icon_path):
    from PIL import Image
    try:
        img = Image.open(icon_path)
        print(f"Icon dimensions: {img.size}")
    except Exception as e:
        print(f"Error opening icon: {str(e)}")

# Check for OrthoRouteEngine and route method
print("\nChecking OrthoRouteEngine class...")
try:
    from orthoroute.gpu_engine import OrthoRouteEngine
    
    engine = OrthoRouteEngine()
    methods = [method for method in dir(engine) if not method.startswith('_') and callable(getattr(engine, method))]
    
    print("Available methods:")
    for method in methods:
        print(f"  - {method}")
        
    if hasattr(engine, 'route'):
        print("\nroute() method is available - plugin should work correctly!")
    else:
        print("\nWARNING: route() method not found!")
        
except Exception as e:
    print(f"Error checking OrthoRouteEngine: {str(e)}")
