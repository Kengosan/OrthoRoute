"""
OrthoRoute Error Logging Plugin
==============================
A version of the plugin that logs any errors to help debug silent failures
"""

import os
import sys
import datetime
import traceback

# Create error log file immediately
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
error_log_path = os.path.join(desktop_path, f"OrthoRoute_Plugin_Errors_{timestamp}.txt")

def log_error(message):
    """Log any error to desktop file"""
    try:
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now()}: {message}\n")
            f.flush()
    except:
        pass  # Can't log the logging error

# Log plugin startup
log_error("=== ORTHOROUTE PLUGIN STARTUP ===")
log_error(f"Python executable: {sys.executable}")
log_error(f"Python version: {sys.version}")
log_error(f"Working directory: {os.getcwd()}")
log_error(f"Python path: {sys.path}")

try:
    log_error("Attempting to import pcbnew...")
    import pcbnew
    log_error("✅ pcbnew imported successfully")
except Exception as e:
    log_error(f"❌ pcbnew import failed: {e}")
    log_error(traceback.format_exc())
    raise

try:
    log_error("Attempting to import wx...")
    import wx
    log_error("✅ wx imported successfully")
except Exception as e:
    log_error(f"❌ wx import failed: {e}")
    log_error(traceback.format_exc())
    raise

try:
    log_error("Importing other modules...")
    import json
    import tempfile
    from typing import Dict, List, Optional, Tuple, Any
    log_error("✅ Standard imports successful")
except Exception as e:
    log_error(f"❌ Standard imports failed: {e}")
    log_error(traceback.format_exc())
    raise

log_error("=== DEFINING PLUGIN CLASS ===")

class OrthoRouteErrorLoggedPlugin(pcbnew.ActionPlugin):
    """KiCad plugin with comprehensive error logging"""
    
    def defaults(self):
        try:
            log_error("Plugin defaults() called")
            self.name = "OrthoRoute Debug Logger"
            self.category = "Debug"
            self.description = "Debug version with error logging"
            self.show_toolbar_button = True
            self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")
            log_error(f"✅ Plugin defaults set: {self.name}")
        except Exception as e:
            log_error(f"❌ defaults() failed: {e}")
            log_error(traceback.format_exc())
            raise
    
    def Run(self):
        """Main plugin entry point with full error logging"""
        try:
            log_error("=== PLUGIN RUN() CALLED ===")
            
            # Test basic KiCad access
            log_error("Testing board access...")
            board = pcbnew.GetBoard()
            if board:
                log_error(f"✅ Board found: {board.GetFileName()}")
            else:
                log_error("❌ No board found")
                return
            
            # Test basic UI
            log_error("Testing wx MessageBox...")
            wx.MessageBox(
                f"OrthoRoute Debug Plugin Running!\n\n"
                f"Board: {board.GetFileName()}\n"
                f"Check desktop for error log:\n"
                f"{error_log_path}",
                "Debug Success",
                wx.OK | wx.ICON_INFORMATION
            )
            log_error("✅ MessageBox displayed successfully")
            
            # Test path injection
            log_error("Testing path injection...")
            original_path_count = len(sys.path)
            
            system_site_packages = [
                r"C:\Users\Benchoff\AppData\Roaming\Python\Python312\site-packages",
                r"C:\Python312\Lib\site-packages"
            ]
            
            for path in system_site_packages:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)
                    log_error(f"✅ Added path: {path}")
            
            log_error(f"Path count: {original_path_count} → {len(sys.path)}")
            
            # Test CuPy import
            log_error("Testing CuPy import...")
            try:
                import cupy as cp
                device = cp.cuda.Device()
                props = cp.cuda.runtime.getDeviceProperties(device.id)
                gpu_name = props["name"].decode("utf-8")
                log_error(f"✅ CuPy available: {gpu_name}")
                
                wx.MessageBox(
                    f"GPU Test Successful!\n\n"
                    f"GPU: {gpu_name}\n"
                    f"CuPy Version: {cp.__version__}",
                    "GPU Debug",
                    wx.OK | wx.ICON_INFORMATION
                )
                
            except Exception as e:
                log_error(f"❌ CuPy failed: {e}")
                log_error(traceback.format_exc())
                
                wx.MessageBox(
                    f"GPU Test Failed!\n\n"
                    f"Error: {str(e)}\n"
                    f"Check log for details: {error_log_path}",
                    "GPU Debug Failed",
                    wx.OK | wx.ICON_ERROR
                )
            
            log_error("=== PLUGIN RUN() COMPLETED SUCCESSFULLY ===")
            
        except Exception as e:
            log_error(f"❌ CRITICAL RUN() ERROR: {e}")
            log_error(traceback.format_exc())
            
            try:
                wx.MessageBox(
                    f"Plugin Run() Failed!\n\n"
                    f"Error: {str(e)}\n"
                    f"Check log: {error_log_path}",
                    "Critical Plugin Error",
                    wx.OK | wx.ICON_ERROR
                )
            except:
                pass  # Even the error dialog failed

log_error("=== PLUGIN CLASS DEFINITION COMPLETE ===")
log_error(f"Error log location: {error_log_path}")

# Register the plugin
try:
    log_error("Registering plugin with KiCad...")
    # KiCad will automatically find this class
    log_error("✅ Plugin registration complete")
except Exception as e:
    log_error(f"❌ Plugin registration failed: {e}")
    log_error(traceback.format_exc())
