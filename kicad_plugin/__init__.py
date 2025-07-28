"""
OrthoRoute KiCad Plugin
========================

GPU-accelerated PCB autorouter integration for KiCad

This plugin provides a seamless interface between KiCad and the OrthoRoute
GPU routing engine, allowing users to route complex PCBs using CUDA acceleration
directly from the KiCad PCB editor.

Installation:
    Copy this folder to your KiCad plugins directory:
    - Linux: ~/.local/share/kicad/7.0/3rdparty/plugins/OrthoRoute/
    - Windows: %APPDATA%\\kicad\\7.0\\3rdparty\\plugins\\OrthoRoute\\
    - macOS: ~/Library/Application Support/kicad/7.0/3rdparty/plugins/OrthoRoute/

Requirements:
    - NVIDIA GPU with CUDA support
    - CuPy: pip install cupy-cuda12x
    - OrthoRoute package: pip install orthoroute

Usage:
    1. Open your PCB in KiCad PCB Editor
    2. Click the OrthoRoute button in the toolbar
    3. Configure grid and routing settings
    4. Click "Start GPU Routing"
    5. Routes are automatically applied to your board
"""

import os
import sys
import warnings

# Plugin version and metadata
__version__ = "0.1.0"
__author__ = "Brian Benchoff"
__description__ = "GPU-accelerated PCB autorouter for KiCad"
__license__ = "MIT"

def check_orthoroute_installed():
    """Check if OrthoRoute package is installed"""
    try:
        import orthoroute
        return True
    except ImportError:
        return False

def show_installation_instructions():
    """Show installation instructions for OrthoRoute"""
    message = """
OrthoRoute Package Not Found!

Please install the OrthoRoute package:

1. Download OrthoRoute from GitHub:
   git clone https://github.com/bbenchoff/OrthoRoute.git

2. Install the package:
   cd OrthoRoute
   pip install .

3. Ensure CUDA and CuPy are installed:
   pip install cupy-cuda12x  (for CUDA 12.x)
   pip install cupy-cuda11x  (for CUDA 11.x)

4. Restart KiCad

Visit https://github.com/bbenchoff/OrthoRoute for more information.
"""
    import wx
    wx.MessageBox(message, "OrthoRoute Installation Required", wx.OK | wx.ICON_INFORMATION)

# Check for OrthoRoute package
if not check_orthoroute_installed():
    show_installation_instructions()

# Check KiCad availability
try:
    import pcbnew
    KICAD_AVAILABLE = True
    KICAD_VERSION = getattr(pcbnew, 'Version', lambda: 'Unknown')()
except ImportError:
    KICAD_AVAILABLE = False
    KICAD_VERSION = 'Not Available'

# Check dependencies
def check_dependencies():
    """Check if all required dependencies are available"""
    deps_status = {
        'kicad': KICAD_AVAILABLE,
        'cupy': False,
        'orthoroute': False,
        'numpy': False
    }
    
    try:
        import cupy as cp
        # Test GPU access
        test_array = cp.array([1, 2, 3])
        deps_status['cupy'] = True
    except (ImportError, Exception):
        pass
    
    try:
        import orthoroute
        deps_status['orthoroute'] = True
    except ImportError:
        pass
    
    try:
        import numpy as np
        deps_status['numpy'] = True
    except ImportError:
        pass
    
    return deps_status

# Initialize plugin
def initialize_plugin():
    """Initialize the OrthoRoute KiCad plugin"""
    deps = check_dependencies()
    
    if not deps['kicad']:
        print("❌ KiCad not available - plugin cannot be loaded")
        return False
    
    if not all([deps['cupy'], deps['orthoroute'], deps['numpy']]):
        missing = [dep for dep, available in deps.items() if not available and dep != 'kicad']
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print("Plugin will load with limited functionality")
    
    try:
        # Import and register the main plugin
        from .orthoroute_kicad import OrthoRouteKiCadPlugin
        
        # Register the plugin with KiCad
        plugin_instance = OrthoRouteKiCadPlugin()
        plugin_instance.register()
        
        print(f"✅ OrthoRoute plugin v{__version__} loaded successfully")
        print(f"   KiCad version: {KICAD_VERSION}")
        
        if deps['cupy']:
            try:
                import cupy as cp
                device = cp.cuda.Device()
                print(f"   GPU: {device.name}")
            except:
                print("   GPU: Detection failed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import OrthoRoute plugin: {e}")
        
        # Create a fallback error plugin
        try:
            from .error_plugin import OrthoRouteErrorPlugin
            error_plugin = OrthoRouteErrorPlugin(str(e))
            error_plugin.register()
            print("   Error plugin registered - shows installation instructions")
        except ImportError:
            # Create minimal error plugin inline
            _create_minimal_error_plugin(str(e))
        
        return False
    
    except Exception as e:
        print(f"💥 Unexpected error initializing OrthoRoute plugin: {e}")
        return False

def _create_minimal_error_plugin(error_message):
    """Create a minimal error plugin when imports fail"""
    if not KICAD_AVAILABLE:
        return
    
    import pcbnew
    import wx
    
    class MinimalErrorPlugin(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "OrthoRoute (Installation Error)"
            self.category = "Autorouter"
            self.description = f"OrthoRoute failed to load: {error_message}"
            self.show_toolbar_button = True
            self.icon_file_name = os.path.join(plugin_dir, "icon_error.png")
        
        def Run(self):
            error_msg = (
                f"OrthoRoute Plugin Installation Error\n\n"
                f"Error: {error_message}\n\n"
                f"Installation Requirements:\n\n"
                f"1. NVIDIA GPU with CUDA support\n"
                f"2. Install CUDA Toolkit 11.8+ or 12.x\n"
                f"3. Install Python dependencies:\n"
                f"   pip install cupy-cuda12x numpy\n"
                f"   pip install orthoroute\n\n"
                f"4. Restart KiCad\n\n"
                f"For detailed instructions, visit:\n"
                f"https://github.com/username/OrthoRoute"
            )
            
            wx.MessageBox(error_msg, "OrthoRoute Installation Required", 
                         wx.OK | wx.ICON_INFORMATION)
    
    try:
        minimal_plugin = MinimalErrorPlugin()
        minimal_plugin.register()
    except Exception as e:
        print(f"Failed to create minimal error plugin: {e}")

def get_plugin_info():
    """Get plugin information for debugging"""
    deps = check_dependencies()
    
    info = {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'kicad_version': KICAD_VERSION,
        'plugin_dir': plugin_dir,
        'dependencies': deps,
        'python_version': sys.version,
        'python_path': sys.path[:3]  # First few entries
    }
    
    return info

def print_debug_info():
    """Print detailed debug information"""
    info = get_plugin_info()
    
    print("\n" + "="*50)
    print("OrthoRoute Plugin Debug Information")
    print("="*50)
    print(f"Plugin Version: {info['version']}")
    print(f"Plugin Directory: {info['plugin_dir']}")
    print(f"KiCad Version: {info['kicad_version']}")
    print(f"Python Version: {info['python_version']}")
    
    print("\nDependency Status:")
    for dep, status in info['dependencies'].items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {dep}")
    
    print("\nPython Path (first 3 entries):")
    for i, path in enumerate(info['python_path']):
        print(f"  {i}: {path}")
    
    # GPU Information if available
    if info['dependencies']['cupy']:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            attrs = device.attributes
            mem_info = device.mem_info
            
            print(f"\nGPU Information:")
            print(f"  Device: {device.name}")
            print(f"  Compute Capability: {attrs['major']}.{attrs['minor']}")
            print(f"  Memory: {mem_info[1] / (1024**3):.1f} GB total")
            print(f"  Available: {mem_info[0] / (1024**3):.1f} GB")
        except Exception as e:
            print(f"\nGPU Information: Failed to get details - {e}")
    
    print("="*50)

# Utility functions for plugin development
def reload_plugin():
    """Reload the plugin (useful for development)"""
    if 'orthoroute_kicad' in sys.modules:
        import importlib
        importlib.reload(sys.modules['orthoroute_kicad'])
        print("🔄 Plugin reloaded")
    else:
        print("Plugin not loaded, initializing...")
        initialize_plugin()

def get_installation_instructions():
    """Get detailed installation instructions"""
    return """
OrthoRoute Installation Guide
============================

Prerequisites:
1. NVIDIA GPU with CUDA Compute Capability 7.5+
2. CUDA Toolkit 11.8+ or 12.x installed
3. KiCad 7.0+

Installation Steps:

1. Install CUDA Toolkit:
   Download from: https://developer.nvidia.com/cuda-downloads
   Follow installation instructions for your platform

2. Install Python dependencies:
   # For CUDA 12.x:
   pip install cupy-cuda12x numpy
   
   # For CUDA 11.x:
   pip install cupy-cuda11x numpy

3. Install OrthoRoute:
   # From PyPI (when released):
   pip install orthoroute
   
   # From source:
   git clone https://github.com/username/OrthoRoute.git
   cd OrthoRoute
   pip install -e .

4. Install KiCad Plugin:
   Copy the kicad_plugin folder to:
   
   Linux:   ~/.local/share/kicad/7.0/3rdparty/plugins/OrthoRoute/
   Windows: %APPDATA%\\kicad\\7.0\\3rdparty\\plugins\\OrthoRoute\\
   macOS:   ~/Library/Application Support/kicad/7.0/3rdparty/plugins/OrthoRoute/

5. Restart KiCad and refresh plugins:
   Tools → External Plugins → Refresh Plugins

6. Look for "OrthoRoute GPU Autorouter" in the toolbar

Troubleshooting:
- Check CUDA installation: nvcc --version
- Test CuPy: python -c "import cupy; print('CuPy OK')"
- Check KiCad console for error messages
- Ensure plugin files have correct permissions

For detailed help, visit:
https://github.com/bbenchoff/OrthoRoute
"""

# Register cleanup function
import atexit

def cleanup_plugin():
    """Cleanup function called on exit"""
    try:
        # Stop any running visualizations or GPU processes
        if 'orthoroute_kicad' in sys.modules:
            module = sys.modules['orthoroute_kicad']
            if hasattr(module, 'cleanup'):
                module.cleanup()
    except Exception as e:
        print(f"Error during plugin cleanup: {e}")

atexit.register(cleanup_plugin)

# Auto-initialize if running in KiCad
if KICAD_AVAILABLE:
    # Initialize the plugin when imported
    success = initialize_plugin()
    
    # Export key functions and info
    __all__ = [
        '__version__',
        '__author__', 
        '__description__',
        'check_dependencies',
        'get_plugin_info',
        'print_debug_info',
        'reload_plugin',
        'get_installation_instructions'
    ]
    
    # Development helpers (only in debug mode)
    if os.environ.get('ORTHOROUTE_DEBUG'):
        print_debug_info()
        __all__.extend(['initialize_plugin', 'cleanup_plugin'])

else:
    print("⚠️  KiCad not available - OrthoRoute plugin not loaded")
    __all__ = ['get_installation_instructions', 'check_dependencies']