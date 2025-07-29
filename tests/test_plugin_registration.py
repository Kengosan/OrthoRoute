import os
import sys
import importlib

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

print("Testing OrthoRoute KiCad Plugin Registration")
print("-----------------------------------------")

# Create mock environment
print("Creating mock environment...")

class MockBoard:
    def __init__(self):
        self.tracks = []
        self.vias = []
    
    def RefreshRatsnest(self):
        print("Board RefreshRatsnest called")

# Mock pcbnew module
class MockPcbnew:
    @staticmethod
    def GetBoard():
        print("GetBoard called")
        return MockBoard()
        
    @staticmethod
    def Refresh():
        print("Refresh called")
    
    class ActionPlugin:
        def __init__(self):
            self.registered = False
            
        def defaults(self):
            pass
            
        def register(self):
            print(f"Registering plugin: {self.name}")
            print(f"  Description: {self.description}")
            print(f"  Icon: {getattr(self, 'icon_file_name', 'None')}")
            print(f"  Show toolbar button: {getattr(self, 'show_toolbar_button', False)}")
            self.registered = True

# Mock wx module
class MockWx:
    @staticmethod
    def MessageBox(message, title, style=None):
        print(f"MessageBox: {title} - {message}")
    
    OK = 1
    ICON_ERROR = 2
    ICON_INFORMATION = 4
    
    class Dialog:
        def __init__(self, *args, **kwargs):
            pass
            
        def ShowModal(self):
            return 1  # Return OK value
            
        def Destroy(self):
            pass

# Install mocks
sys.modules['pcbnew'] = MockPcbnew
sys.modules['wx'] = MockWx

# Test plugin registration
print("\nTesting plugin registration...")

# First try loading the plugin module directly
try:
    # Import as if KiCad would
    import kicad_plugin
    print("Plugin module imported successfully")
    
    # Try accessing the registered plugin
    if hasattr(kicad_plugin, 'OrthoRouteKiCadPlugin'):
        plugin = getattr(kicad_plugin, 'OrthoRouteKiCadPlugin')
        print(f"Plugin class found: {plugin.__name__}")
    else:
        print("Plugin class not exposed at module level")
    
except Exception as e:
    print(f"Error importing plugin module: {str(e)}")
    import traceback
    traceback.print_exc()

# Try simulating a Run call
print("\nTesting plugin execution...")
try:
    # Import the plugin class directly
    from kicad_plugin.orthoroute_kicad import OrthoRouteKiCadPlugin
    
    # Create an instance and call Run
    plugin = OrthoRouteKiCadPlugin()
    plugin.defaults()
    print(f"\nPlugin icon path: {plugin.icon_file_name}")
    print(f"Icon exists: {os.path.exists(plugin.icon_file_name)}")
    
    print("\nSimulating plugin Run() method call...")
    plugin.Run()
    
except Exception as e:
    print(f"Error executing plugin: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nTest complete")
