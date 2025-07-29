import os
import pcbnew
import wx
import sys
from pathlib import Path

class OrthoRoutePlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter"
        self.category = "Autorouter"
        self.description = "GPU-accelerated PCB autorouting using CuPy"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")
    
    def __init__(self):
        """Initialize plugin and prepare imports"""
        super().__init__()
        self.initialized = False
        
    def _initialize(self):
        """Initialize imports and classes"""
        if self.initialized:
            return
            
        try:
            # Make sure our plugin directory is in the path
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(plugin_dir)
            
            # Add both the plugin directory and its parent to the path
            for path in [plugin_dir, parent_dir]:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            # Debug path info
            debug_info = f"Python Path:\n{chr(10).join(sys.path)}\n\nPlugin Dir: {plugin_dir}"
            wx.MessageBox(debug_info, "Initialization Debug", wx.OK)
            
            # Import our modules
            try:
                import ui_dialogs
                wx.MessageBox("ui_dialogs imported successfully", "Debug", wx.OK)
            except Exception as e:
                wx.MessageBox(f"Failed to import ui_dialogs: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
                raise
                
            try:
                import board_export
                import route_import
                from orthoroute.gpu_engine import OrthoRouteEngine
            except Exception as e:
                wx.MessageBox(f"Failed to import other modules: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
                raise
            
            # Store classes as instance attributes
            self.OrthoRouteConfigDialog = ui_dialogs.OrthoRouteConfigDialog
            self.BoardExporter = board_export.BoardExporter
            self.RouteImporter = route_import.RouteImporter
            self.OrthoRouteEngine = OrthoRouteEngine
            
            self.initialized = True
            wx.MessageBox("Initialization complete", "Debug", wx.OK)
            
        except Exception as e:
            wx.MessageBox(f"Initialization failed: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
            raise
    
    def Run(self):
        """Main plugin entry point"""
        try:
            # Debug info
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            debug_info = f"Python Path: {sys.path}\nPlugin Dir: {plugin_dir}\nCWD: {os.getcwd()}"
            wx.MessageBox(debug_info, "Debug Info", wx.OK | wx.ICON_INFORMATION)
            
            # Initialize imports if needed
            self._initialize()
            wx.MessageBox("Initialization complete", "Debug", wx.OK)
            
            # Check CuPy availability
            if not self._check_cupy_available():
                self._show_cupy_install_dialog()
                return
            
            # Get current board
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board loaded!", "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Show configuration dialog
            config_dialog = self.OrthoRouteConfigDialog(None)
            if config_dialog.ShowModal() == wx.ID_OK:
                config = config_dialog.get_config()
                self._route_board_gpu(board, config)
            
            config_dialog.Destroy()
            
        except Exception as e:
            wx.MessageBox(f"OrthoRoute Error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
    
    def _check_cupy_available(self) -> bool:
        """Check if CuPy is available"""
        try:
            import cupy as cp
            # Test GPU access
            test_array = cp.array([1, 2, 3])
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def _show_cupy_install_dialog(self):
        """Show CuPy installation instructions"""
        message = (
            "CuPy is required for GPU acceleration!\n\n"
            "Installation steps:\n\n"
            "1. Ensure NVIDIA GPU with CUDA support\n"
            "2. Install CUDA Toolkit 11.8+ or 12.x\n"
            "3. Install CuPy:\n"
            "   pip install cupy-cuda12x  (for CUDA 12.x)\n"
            "   pip install cupy-cuda11x  (for CUDA 11.x)\n\n"
            "4. Restart KiCad\n\n"
            "Visit https://docs.cupy.dev/en/stable/install.html for details."
        )
        wx.MessageBox(message, "CuPy Installation Required", wx.OK | wx.ICON_INFORMATION)
    
    def _route_board_gpu(self, board: pcbnew.BOARD, config: dict):
        """Route board using GPU engine"""
        progress = wx.ProgressDialog(
            "OrthoRoute GPU Routing",
            "Initializing...",
            maximum=100,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
        )
        
        try:
            # Create instances
            engine = self.OrthoRouteEngine()
            exporter = self.BoardExporter(board)
            importer = self.RouteImporter(board)
            
            # Step 1: Export board data
            progress.Update(10, "Exporting board data...")
            board_data = exporter.export_board(config)
            
            if not board_data['nets']:
                wx.MessageBox("No nets found to route!", "OrthoRoute", wx.OK | wx.ICON_WARNING)
                return
            
            # Step 2: Initialize GPU engine
            progress.Update(30, "Initializing GPU engine...")
            
            # Step 3: Route on GPU
            progress.Update(50, f"Routing {len(board_data['nets'])} nets on GPU...")
            results = engine.route_board(board_data)
            
            if not results['success']:
                error_msg = results.get('error', 'Unknown GPU routing error')
                wx.MessageBox(f"GPU routing failed: {error_msg}", "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Step 4: Import results
            progress.Update(80, "Importing routes to KiCad...")
            applied_count = importer.apply_routes(results['routed_nets'])
            
            progress.Update(95, "Refreshing display...")
            pcbnew.Refresh()
            
            # Show success dialog
            stats = results['stats']
            message = (
                f"OrthoRoute GPU Routing Complete!\n\n"
                f"Nets processed: {stats['total_nets']}\n"
                f"Successfully routed: {stats['successful_nets']}\n"
                f"Success rate: {stats['success_rate']:.1f}%\n"
                f"Routing time: {stats['routing_time_seconds']:.2f} seconds\n"
                f"Performance: {stats['nets_per_second']:.1f} nets/second\n\n"
                f"Applied {applied_count} routes to board."
            )
            wx.MessageBox(message, "OrthoRoute Success", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            wx.MessageBox(f"Routing error: {str(e)}", "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
        finally:
            progress.Destroy()

# Handle registration
try:
    plugin_instance = OrthoRoutePlugin()
    # Initialize before registration to catch any import errors early
    plugin_instance._initialize()
    plugin_instance.register()
    print("OrthoRoute plugin registered successfully")
except Exception as e:
    import sys
    sys.stderr.write(f"Failed to register OrthoRoute plugin: {str(e)}\n")
    raise
