
import pcbnew
import wx
import json
import tempfile
import os
from typing import Dict, List
import ui_dialogs
import board_export
import route_import
# Import moved inside _route_board_gpu to avoid import errors if CuPy not available

OrthoRouteConfigDialog = ui_dialogs.OrthoRouteConfigDialog
BoardExporter = board_export.BoardExporter
RouteImporter = route_import.RouteImporter

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """KiCad plugin for OrthoRoute GPU autorouter"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter"
        self.category = "Autorouter"
        self.description = "GPU-accelerated PCB autorouter using CuPy"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")
    
    def Run(self):
        """Main plugin entry point"""
        try:
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
            config_dialog = OrthoRouteConfigDialog(None)
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
    
    def _route_board_gpu(self, board: pcbnew.BOARD, config: Dict):
        """Route board using GPU engine"""
        progress = wx.ProgressDialog(
            "OrthoRoute GPU Routing",
            "Initializing...",
            maximum=100,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
        )
        
        try:
            # Import here to avoid import errors if CuPy not available
            import orthoroute.gpu_engine as gpu_engine
            from . import board_export
            from . import route_import
            
            # Step 1: Export board data
            progress.Update(10, "Exporting board data...")
            exporter = BoardExporter(board)
            board_data = exporter.export_board(config)
            
            if not board_data['nets']:
                wx.MessageBox("No nets found to route!", "OrthoRoute", wx.OK | wx.ICON_WARNING)
                return
            
            # Step 2: Initialize GPU engine
            progress.Update(30, "Initializing GPU engine...")
            # For debugging - make sure we only create one engine instance
            print("DEBUG KICAD: Creating OrthoRouteEngine instance")
            
            # Import entire module to avoid namespace issues
            import orthoroute.gpu_engine
            # Use the module's class (should have all attributes)
            engine = orthoroute.gpu_engine.OrthoRouteEngine()
            
            # Use a try-except to avoid attribute errors in debug messages
            try:
                print(f"DEBUG KICAD: Engine created with ID: {engine.engine_id}")
            except AttributeError:
                print("DEBUG KICAD: Engine created (no ID attribute available)")
            
            # Setup visualization if enabled
            if config.get('show_visualization', False):
                print("DEBUG KICAD: Visualization enabled in config")
                try:
                    print("DEBUG KICAD: Importing visualization modules")
                    from orthoroute.visualization import RoutingVisualizer, VisualizationConfig
                    
                    print("DEBUG KICAD: Creating VisualizationConfig")
                    viz_config = VisualizationConfig(
                        backend="matplotlib",
                        update_interval=0.25,
                        show_grid=True,
                        show_obstacles=True,
                        show_nets=True
                    )
                    
                    # Enable visualization on engine
                    print("DEBUG KICAD: Calling engine.enable_visualization()")
                    
                    # Make sure the visualizer attribute exists
                    if not hasattr(engine, 'visualizer'):
                        print("DEBUG KICAD: Adding visualizer attribute to engine")
                        engine.visualizer = None
                    
                    # Make sure viz_config attribute exists 
                    if hasattr(engine, 'enable_visualization'):
                        try:
                            engine.enable_visualization(viz_config)
                            print("DEBUG KICAD: Visualization successfully enabled")
                        except Exception as e:
                            print(f"DEBUG KICAD: Error enabling visualization: {e}")
                            # Fall back solution - add attribute directly
                            engine.viz_config = viz_config
                            print("DEBUG KICAD: viz_config attribute manually added")
                    else:
                        print("DEBUG KICAD: ERROR - engine has no enable_visualization method!")
                        # Fall back solution - add attribute directly
                        engine.viz_config = viz_config
                        print("DEBUG KICAD: viz_config attribute manually added")
                except ImportError as e:
                    print(f"DEBUG KICAD: ImportError: {e}")
                    wx.MessageBox(
                        f"Visualization dependencies not available: {e}\nInstall matplotlib for visualization.",
                        "Visualization Warning", wx.OK | wx.ICON_WARNING
                    )
                except Exception as e:
                    print(f"DEBUG KICAD: Visualization setup error: {e}")
                    wx.MessageBox(
                        f"Error setting up visualization: {e}",
                        "Visualization Error", wx.OK | wx.ICON_WARNING
                    )
            
            # Step 3: Route on GPU
            progress.Update(50, f"Routing {len(board_data['nets'])} nets on GPU...")
            results = engine.route_board(board_data)
            
            if not results['success']:
                error_msg = results.get('error', 'Unknown GPU routing error')
                wx.MessageBox(f"GPU routing failed: {error_msg}", "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Step 4: Import results
            progress.Update(80, "Importing routes to KiCad...")
            importer = RouteImporter(board)
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
            
        except ImportError as e:
            wx.MessageBox(f"Import error: {e}\nEnsure OrthoRoute package is installed", 
                         "Import Error", wx.OK | wx.ICON_ERROR)
        except Exception as e:
            wx.MessageBox(f"Routing error: {str(e)}", "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
        finally:
            progress.Destroy()

class OrthoRouteConfigDialog(wx.Dialog):
    """Configuration dialog for OrthoRoute"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute GPU Configuration")
        self._create_ui()
        self.Center()
    
    def _create_ui(self):
        """Create configuration UI"""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Grid settings
        grid_box = wx.StaticBox(panel, label="Grid Settings")
        grid_sizer = wx.StaticBoxSizer(grid_box, wx.VERTICAL)
        
        # Grid pitch
        pitch_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pitch_sizer.Add(wx.StaticText(panel, label="Grid Pitch (mm):"), 0, wx.CENTER | wx.RIGHT, 5)
        self.pitch_ctrl = wx.SpinCtrlDouble(panel)
        self.pitch_ctrl.SetValue(0.1)
        self.pitch_ctrl.SetRange(0.05, 0.5)
        self.pitch_ctrl.SetIncrement(0.05)
        self.pitch_ctrl.SetDigits(2)
        pitch_sizer.Add(self.pitch_ctrl, 1, wx.EXPAND)
        grid_sizer.Add(pitch_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Via size
        via_sizer = wx.BoxSizer(wx.HORIZONTAL)
        via_sizer.Add(wx.StaticText(panel, label="Via Size (mm):"), 0, wx.CENTER | wx.RIGHT, 5)
        self.via_ctrl = wx.SpinCtrlDouble(panel)
        self.via_ctrl.SetValue(0.2)
        self.via_ctrl.SetRange(0.1, 1.0)
        self.via_ctrl.SetIncrement(0.1)
        self.via_ctrl.SetDigits(2)
        via_sizer.Add(self.via_ctrl, 1, wx.EXPAND)
        grid_sizer.Add(via_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(grid_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Routing settings
        routing_box = wx.StaticBox(panel, label="Routing Settings")
        routing_sizer = wx.StaticBoxSizer(routing_box, wx.VERTICAL)
        
        # Trace width
        width_sizer = wx.BoxSizer(wx.HORIZONTAL)
        width_sizer.Add(wx.StaticText(panel, label="Trace Width (mm):"), 0, wx.CENTER | wx.RIGHT, 5)
        self.width_ctrl = wx.SpinCtrlDouble(panel)
        self.width_ctrl.SetValue(0.2)
        self.width_ctrl.SetRange(0.1, 1.0)
        self.width_ctrl.SetIncrement(0.1)
        self.width_ctrl.SetDigits(2)
        width_sizer.Add(self.width_ctrl, 1, wx.EXPAND)
        routing_sizer.Add(width_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Congestion factor
        cong_sizer = wx.BoxSizer(wx.HORIZONTAL)
        cong_sizer.Add(wx.StaticText(panel, label="Congestion Factor:"), 0, wx.CENTER | wx.RIGHT, 5)
        self.cong_ctrl = wx.SpinCtrlDouble(panel)
        self.cong_ctrl.SetValue(0.5)
        self.cong_ctrl.SetRange(0.1, 2.0)
        self.cong_ctrl.SetIncrement(0.1)
        self.cong_ctrl.SetDigits(2)
        cong_sizer.Add(self.cong_ctrl, 1, wx.EXPAND)
        routing_sizer.Add(cong_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Max iterations
        iter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        iter_sizer.Add(wx.StaticText(panel, label="Max Iterations:"), 0, wx.CENTER | wx.RIGHT, 5)
        self.iter_ctrl = wx.SpinCtrl(panel)
        self.iter_ctrl.SetValue(1000)
        self.iter_ctrl.SetRange(100, 10000)
        iter_sizer.Add(self.iter_ctrl, 1, wx.EXPAND)
        routing_sizer.Add(iter_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(routing_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(wx.Button(panel, wx.ID_CANCEL), 0, wx.RIGHT, 5)
        ok_btn = wx.Button(panel, wx.ID_OK, "Start GPU Routing")
        ok_btn.SetDefault()
        btn_sizer.Add(ok_btn, 0)
        
        sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        sizer.Fit(self)
    
    def get_config(self) -> Dict:
        """Get configuration from dialog"""
        return {
            'grid': {
                'pitch_nm': int(self.pitch_ctrl.GetValue() * 1000000),
                'via_size_nm': int(self.via_ctrl.GetValue() * 1000000)
            },
            'routing': {
                'width_nm': int(self.width_ctrl.GetValue() * 1000000),
                'congestion_factor': self.cong_ctrl.GetValue(),
                'max_iterations': self.iter_ctrl.GetValue()
            },
            'options': {
                'skip_power_nets': True,
                'skip_routed_nets': True,
                'include_existing_tracks': True,
                'include_component_keepouts': True,
                'include_edge_keepouts': True
            }
        }

# Register the plugin with KiCad
OrthoRoutePlugin = OrthoRouteKiCadPlugin()