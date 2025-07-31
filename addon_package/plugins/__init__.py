"""
OrthoRoute GPU Autorouter - KiCad Plugin
A GPU-accelerated PCB autorouter using CuPy/CUDA
"""

import pcbnew
import wx
import json
import tempfile
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

class OrthoRouteDebugDialog(wx.Dialog):
    """Simple debug output window"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Debug Output", 
                        size=(700, 500), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Text control for debug output
        self.text_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.text_ctrl.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        
        # Close button
        close_btn = wx.Button(panel, wx.ID_CLOSE, "Close")
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_CLOSE))
        sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        self.Centre()
        
    def append_text(self, text):
        """Add text to debug output"""
        self.text_ctrl.AppendText(text + "\n")

class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
    """KiCad plugin for OrthoRoute GPU autorouter"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter"
        self.category = "Routing"
        self.description = "GPU-accelerated PCB autorouter using CuPy/CUDA"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")
    
    def Run(self):
        """Main plugin entry point"""
        try:
            # Set up debug file logging
            import os
            import datetime
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file_path = os.path.join(desktop_path, f"OrthoRoute_Debug_{timestamp}.txt")
            
            # Create debug file writer
            debug_file = open(debug_file_path, 'w', encoding='utf-8')
            debug_file.write(f"OrthoRoute Debug Log - {datetime.datetime.now()}\n")
            debug_file.write("=" * 60 + "\n\n")
            debug_file.flush()
            
            def write_debug(message):
                """Write to both console and file"""
                print(message)
                debug_file.write(message + "\n")
                debug_file.flush()
            
            write_debug("🚀 OrthoRoute Plugin Started")
            write_debug(f"📝 Debug log: {debug_file_path}")
            
            # Get current board
            board = pcbnew.GetBoard()
            if not board:
                error_msg = "No board found. Please open a PCB first."
                write_debug(f"❌ {error_msg}")
                debug_file.close()
                wx.MessageBox(error_msg, "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            write_debug("✅ Board found, showing configuration dialog...")
            
            # Show configuration dialog
            dlg = OrthoRouteConfigDialog(None)
            if dlg.ShowModal() == wx.ID_OK:
                config = dlg.get_config()
                debug_mode = config.get('debug_mode', False)
                dlg.Destroy()
                
                write_debug(f"⚙️ Configuration: {config}")
                
                # Create debug window if debug mode enabled
                debug_dialog = None
                if debug_mode:
                    debug_dialog = OrthoRouteDebugDialog(None)
                    debug_dialog.Show()
                    write_debug("🔍 Debug window opened")
                
                # Route the board (without CuPy requirement)
                write_debug("🛤 Starting board routing...")
                try:
                    result = self._route_board_simple(board, config, debug_dialog, write_debug)
                    write_debug(f"🏁 Routing completed with result: {result}")
                except Exception as e:
                    write_debug(f"❌ ROUTING FAILED: {str(e)}")
                    import traceback
                    write_debug(f"📋 Full traceback:\n{traceback.format_exc()}")
                    raise
            else:
                write_debug("❌ User cancelled configuration dialog")
                dlg.Destroy()
            
            write_debug("🏁 Plugin execution completed")
            debug_file.close()
                
        except Exception as e:
            error_msg = f"Critical error in plugin Run method: {e}"
            print(error_msg)
            import traceback
            full_trace = traceback.format_exc()
            print(f"Full traceback: {full_trace}")
            
            # Also write to debug file if it exists
            try:
                debug_file.write(f"\n❌ CRITICAL ERROR: {error_msg}\n")
                debug_file.write(f"📋 Traceback:\n{full_trace}\n")
                debug_file.close()
            except:
                pass
                
            wx.MessageBox(f"Plugin error: {str(e)}\n\nCheck console for detailed traceback.\nDebug log saved to desktop.", 
                        "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
    
    def _check_cupy_available(self) -> bool:
        """Check if CuPy is available"""
        try:
            import cupy as cp
            # Test basic functionality
            test_array = cp.array([1, 2, 3])
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def _show_cupy_install_dialog(self):
        """Show dialog with CuPy installation instructions"""
        message = """CuPy is required for GPU acceleration but was not found.

Installation instructions:
1. Ensure you have an NVIDIA GPU with CUDA support
2. Install CUDA Toolkit (11.8+ or 12.x)
3. Install CuPy using one of these commands:

For CUDA 12.x:
pip install cupy-cuda12x

For CUDA 11.x:
pip install cupy-cuda11x

For more details, visit: https://docs.cupy.dev/en/stable/install.html"""

        wx.MessageBox(message, "CuPy Installation Required", 
                     wx.OK | wx.ICON_INFORMATION)
    
    def _route_board_gpu(self, board, config, debug_dialog=None):
        """Route the board using GPU acceleration"""
        
        def debug_print(text):
            """Print to console and debug window"""
            print(text)
            if debug_dialog:
                debug_dialog.append_text(text)
        
        try:
            debug_print("Starting route_board_gpu method...")
            
            # Test imports first to identify any import issues
            try:
                debug_print("Testing CuPy import...")
                import cupy as cp
                debug_print("CuPy imported successfully")
            except Exception as e:
                print(f"❌ CuPy import failed: {e}")
                wx.MessageBox(f"CuPy import failed: {e}", "Import Error", wx.OK | wx.ICON_ERROR)
                return
            
            try:
                print("🔍 Testing visualization import...")
                from .visualization import RoutingProgressDialog
                print("✅ Visualization imported successfully")
            except Exception as e:
                print(f"❌ Visualization import failed: {e}")
                wx.MessageBox(f"Visualization import failed: {e}", "Import Error", wx.OK | wx.ICON_ERROR)
                return
            
            try:
                print("🔍 Testing orthoroute_engine import...")
                from .orthoroute_engine import OrthoRouteEngine
                print("✅ OrthoRouteEngine imported successfully")
            except Exception as e:
                print(f"❌ OrthoRouteEngine import failed: {e}")
                wx.MessageBox(f"OrthoRouteEngine import failed: {e}", "Import Error", wx.OK | wx.ICON_ERROR)
                return
            
            print("🔍 All imports successful, proceeding with routing setup...")
            
            # Create debug output dialog first
            debug_dialog = wx.Dialog(None, title="OrthoRoute Debug Output", 
                                   style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
            debug_dialog.SetSize((600, 400))
            
            debug_text = wx.TextCtrl(debug_dialog, style=wx.TE_MULTILINE | wx.TE_READONLY)
            debug_sizer = wx.BoxSizer(wx.VERTICAL)
            debug_sizer.Add(debug_text, 1, wx.EXPAND | wx.ALL, 5)
            
            close_btn = wx.Button(debug_dialog, wx.ID_OK, "Close")
            debug_sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            debug_dialog.SetSizer(debug_sizer)
            debug_dialog.Show()
            
            def debug_print(msg):
                """Print to both console and debug dialog with buffering"""
                print(msg)
                debug_text.AppendText(msg + "\n")
                # Only update UI every 10 messages or on important messages
                if hasattr(debug_print, '_msg_count'):
                    debug_print._msg_count += 1
                else:
                    debug_print._msg_count = 1
                    
                if (debug_print._msg_count % 10 == 0 or 
                    any(keyword in msg for keyword in ['✅', '❌', '📊', '🚀', '🔍'])):
                    wx.SafeYield()  # Update UI only occasionally
            
            debug_print("🔍 OrthoRoute Debug Output")
            debug_print("=" * 40)
            
            # Show enhanced progress dialog with visualization
            try:
                print("🔍 Creating RoutingProgressDialog...")
                progress_dlg = RoutingProgressDialog(
                    parent=None,
                    title="OrthoRoute - GPU Routing Progress"
                )
                progress_dlg.Show()
                print("✅ RoutingProgressDialog created and shown")
            except Exception as e:
                print(f"❌ Failed to create RoutingProgressDialog: {e}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
                # Fall back to basic progress dialog
                progress_dlg = wx.ProgressDialog(
                    "OrthoRoute - GPU Routing",
                    "Routing in progress...",
                    maximum=100,
                    parent=None,
                    style=wx.PD_APP_MODAL | wx.PD_CAN_ABORT | wx.PD_AUTO_HIDE
                )
                progress_dlg.Show()
            
            try:
                # Export board data with detailed logging
                progress_dlg.Update(10, "Exporting board data...")
                debug_print("🔍 Exporting board data...")
                
                try:
                    board_data = self._export_board_data(board, debug_print, config)
                    debug_print(f"📊 Board export results:")
                    debug_print(f"   - Board bounds: {board_data.get('bounds', {})}")
                    debug_print(f"   - Nets found: {len(board_data.get('nets', []))}")
                    
                    # Print first few nets for debugging
                    nets = board_data.get('nets', [])
                    for i, net in enumerate(nets[:3]):  # Show first 3 nets
                        pins = net.get('pins', [])
                        debug_print(f"   - Net {i+1}: {net.get('name', 'Unknown')} ({len(pins)} pins)")
                        for j, pin in enumerate(pins[:2]):  # Show first 2 pins
                            debug_print(f"     Pin {j+1}: ({pin.get('x', 0)/1e6:.2f}, {pin.get('y', 0)/1e6:.2f}) mm, layer {pin.get('layer', 0)}")
                    
                    if len(nets) > 3:
                        debug_print(f"   - ... and {len(nets) - 3} more nets")
                        
                except Exception as export_error:
                    debug_print(f"❌ Board export failed: {export_error}")
                    import traceback
                    debug_print(traceback.format_exc())
                    wx.MessageBox(f"Failed to export board data: {export_error}", 
                                "Export Error", wx.OK | wx.ICON_ERROR)
                    return
                
                if not board_data.get('nets'):
                    debug_print("⚠️ No nets found to route!")
                    debug_print("🔍 Troubleshooting suggestions:")
                    debug_print("   1. Ensure your PCB has components with pads")
                    debug_print("   2. Ensure components are assigned to nets (connected)")
                    debug_print("   3. Check that nets aren't already fully routed")
                    debug_print("   4. Try updating netlist from schematic")
                    
                    wx.MessageBox("No nets found to route.\n\nPlease ensure your PCB has:\n" +
                                "• Components with pads\n" +
                                "• Nets assigned to pads\n" +
                                "• Unrouted connections\n\n" +
                                "Check the debug dialog for detailed diagnostics.", 
                                "No Nets Found", wx.OK | wx.ICON_WARNING)
                    return
                
                # Initialize GPU engine
                progress_dlg.Update(20, "Initializing routing engine...")
                debug_print("Initializing routing engine...")
                try:
                    print("🔍 Creating OrthoRouteEngine...")
                    engine = OrthoRouteEngine()
                    print("✅ OrthoRouteEngine created successfully")
                except Exception as e:
                    print(f"❌ Failed to create OrthoRouteEngine: {e}")
                    import traceback
                    print(f"❌ Traceback: {traceback.format_exc()}")
                    debug_print(f"❌ Failed to create OrthoRouteEngine: {e}")
                    wx.MessageBox(f"Failed to initialize routing engine: {e}", "Engine Error", wx.OK | wx.ICON_ERROR)
                    return
                
                # Pass debug_print to engine for logging
                engine.debug_print = debug_print
                
                # Enable visualization if requested (force enabled for debugging)
                enable_viz = config.get('enable_visualization', True)  # Default to True
                debug_print(f"🎨 Visualization enabled: {enable_viz}")
                if enable_viz:
                    try:
                        debug_print("🎨 Starting visualization setup...")
                        progress_dlg.update_progress(0.3, 0.0, "Setting up visualization...")
                        debug_print("🎨 Setting up visualization...")
                        
                        # Set up board data for visualization
                        debug_print("🎨 Getting board bounds...")
                        debug_print(f"🎨 Self object type: {type(self)}")
                        debug_print(f"🎨 Self object methods: {[m for m in dir(self) if 'board' in m.lower()]}")
                        
                        # Get board bounds with fallback
                        try:
                            board_bounds = self._get_board_bounds(board)
                        except AttributeError:
                            debug_print("🎨 Using fallback board bounds...")
                            try:
                                bbox = board.GetBoardEdgesBoundingBox()
                                board_bounds = [
                                    float(bbox.GetX()) / 1e6,  # Convert to mm
                                    float(bbox.GetY()) / 1e6,
                                    float(bbox.GetWidth()) / 1e6,
                                    float(bbox.GetHeight()) / 1e6
                                ]
                            except:
                                board_bounds = [0, 0, 100, 80]  # Default
                        debug_print(f"🎨 Board bounds: {board_bounds}")
                        
                        debug_print("🎨 Getting board pads...")
                        # Get board pads with fallback
                        try:
                            pads = self._get_board_pads(board)
                        except AttributeError:
                            debug_print("🎨 Using fallback board pads...")
                            pads = []
                            try:
                                footprint_count = 0
                                pad_count = 0
                                
                                for footprint in board.GetFootprints():
                                    footprint_count += 1
                                    footprint_pads = 0
                                    
                                    for pad in footprint.Pads():
                                        try:
                                            bbox = pad.GetBoundingBox()
                                            pos = pad.GetPosition()
                                            
                                            pad_data = {
                                                'bounds': [
                                                    float(bbox.GetX()) / 1e6,  # Convert to mm
                                                    float(bbox.GetY()) / 1e6,
                                                    float(bbox.GetWidth()) / 1e6,
                                                    float(bbox.GetHeight()) / 1e6
                                                ],
                                                'center': [
                                                    float(pos.x) / 1e6,  # Pad center in mm
                                                    float(pos.y) / 1e6
                                                ],
                                                'net': pad.GetNetname(),
                                                'ref': footprint.GetReference()
                                            }
                                            pads.append(pad_data)
                                            pad_count += 1
                                            footprint_pads += 1
                                            
                                        except Exception as e:
                                            debug_print(f"Error processing pad in {footprint.GetReference()}: {e}")
                                    
                                    # Debug for first few footprints
                                    if footprint_count <= 3:
                                        debug_print(f"   Footprint {footprint.GetReference()}: {footprint_pads} pads")
                                            
                                debug_print(f"📍 Pad extraction: {footprint_count} footprints, {pad_count} total pads")
                                
                            except Exception as e:
                                debug_print(f"Error getting pads: {e}")
                        
                        debug_print(f"🎨 Pads found: {len(pads)}")
                        
                        debug_print("🎨 Getting board obstacles...")
                        # Get board obstacles with fallback  
                        try:
                            obstacles = self._get_board_obstacles(board)
                        except AttributeError:
                            debug_print("🎨 Using fallback board obstacles...")
                            obstacles = []
                            try:
                                # Get existing tracks as obstacles
                                for track in board.GetTracks():
                                    if hasattr(track, 'GetBoundingBox'):
                                        bbox = track.GetBoundingBox()
                                        obstacles.append({
                                            'bounds': [
                                                float(bbox.GetX()) / 1e6,
                                                float(bbox.GetY()) / 1e6,
                                                float(bbox.GetWidth()) / 1e6,
                                                float(bbox.GetHeight()) / 1e6
                                            ],
                                            'type': 'track'
                                        })
                            except Exception as e:
                                debug_print(f"Error getting obstacles: {e}")
                        debug_print(f"🎨 Obstacles found: {len(obstacles)}")
                        
                        if pads:
                            debug_print(f"🎨 Sample pad: {pads[0]}")
                        
                        debug_print("🎨 Calling progress_dlg.set_board_data...")
                        # Temporarily redirect prints to debug dialog
                        import sys
                        import builtins
                        original_print = builtins.print
                        
                        def debug_print_wrapper(*args, **kwargs):
                            message = ' '.join(str(arg) for arg in args)
                            # Send to debug dialog directly to avoid recursion
                            if hasattr(self, 'debug_dialog') and self.debug_dialog:
                                try:
                                    wx.CallAfter(self.debug_dialog.append_text, f"🎨 VIZ: {message}")
                                except:
                                    pass
                            # Also call original print
                            original_print(*args, **kwargs)
                        
                        # Temporarily replace print in visualization module
                        builtins.print = debug_print_wrapper
                        
                        try:
                            progress_dlg.set_board_data(board_bounds, pads, obstacles)
                            debug_print("🎨 Board data set successfully!")
                            
                            # Also set the routing pin data for more accurate visualization
                            debug_print("🎨 Setting routing pin data for accurate visualization...")
                            routing_pins = []
                            for net_data in board_data.get('nets', []):
                                for pin_data in net_data.get('pins', []):
                                    # Convert pin data to pad-like format for visualization
                                    routing_pins.append({
                                        'bounds': [
                                            pin_data['x'] / 1e6 - 0.5,  # Convert nm to mm with 1mm pad size
                                            pin_data['y'] / 1e6 - 0.5,
                                            1.0,  # 1mm width
                                            1.0   # 1mm height
                                        ],
                                        'center': [
                                            pin_data['x'] / 1e6,
                                            pin_data['y'] / 1e6
                                        ],
                                        'net': net_data.get('name', 'Unknown'),
                                        'ref': 'PIN',
                                        'is_routing_pin': True
                                    })
                            
                            debug_print(f"🎨 Created {len(routing_pins)} routing pin visualizations")
                            
                            # Update visualization with routing pins (these are the actual pins being routed)
                            progress_dlg.viz_panel.pads = routing_pins
                            progress_dlg.viz_panel.UpdateDrawing()
                            
                        finally:
                            # Restore original print
                            builtins.print = original_print
                        
                        debug_print("🎨 Enabling engine visualization...")
                        
                        # Create proper progress callback wrapper
                        def routing_progress_callback(progress_data):
                            """Convert routing engine progress to visualization format"""
                            try:
                                debug_print(f"📊 Progress callback received: {progress_data}")
                                
                                current_net = progress_data.get('current_net', 'Unknown')
                                progress = progress_data.get('progress', 0)
                                stage = progress_data.get('stage', 'unknown')
                                success = progress_data.get('success', None)
                                
                                # Create minimal stats object
                                from .visualization import RoutingStats
                                stats = RoutingStats()
                                stats.current_net = current_net
                                stats.stage = stage
                                if success is not None:
                                    stats.success_rate = 100.0 if success else 0.0
                                
                                # Update the progress dialog
                                progress_dlg.update_progress(
                                    overall_progress=progress / 100.0,
                                    net_progress=progress / 100.0,
                                    current_net=current_net,
                                    stats=stats
                                )
                                
                                debug_print(f"📊 Progress updated: {current_net} ({progress:.1f}%)")
                                
                            except Exception as e:
                                debug_print(f"❌ Progress callback error: {e}")
                                import traceback
                                debug_print(f"❌ Traceback: {traceback.format_exc()}")
                        
                        engine.enable_visualization({
                            'real_time': True,
                            'show_progress': True,
                            'progress_callback': routing_progress_callback
                        })
                        debug_print("🎨 Visualization setup complete!")
                    except Exception as e:
                        debug_print(f"❌ Visualization setup failed: {e}")
                        import traceback
                        debug_print(f"❌ Full traceback: {traceback.format_exc()}")
                        # Fall back to basic progress dialog
                        progress_dlg.Update(30, "Visualization setup failed, continuing...")
                else:
                    progress_dlg.Update(30, "Skipping visualization...")
                
                # Route the board with threading for UI responsiveness
                progress_dlg.Update(40, "Starting routing...")
                print("Starting routing...")
                
                # Use threading to keep UI responsive
                import threading
                routing_complete = False
                routing_results = {}
                routing_error = None
                self.routing_cancelled = False  # Add cancellation flag
                
                def routing_worker():
                    """Run routing in separate thread"""
                    nonlocal routing_complete, routing_results, routing_error
                    try:
                        # Add cancellation callback to config
                        config_with_cancel = config.copy()
                        config_with_cancel['should_cancel'] = lambda: self.routing_cancelled
                        
                        routing_results = engine.route(board_data, config_with_cancel)
                        print(f"Routing completed with success={routing_results.get('success', False)}")
                    except Exception as e:
                        routing_error = e
                        print(f"Routing error: {e}")
                    finally:
                        routing_complete = True
                        # Always cleanup on exit
                        try:
                            engine._cleanup_gpu_resources()
                        except Exception as cleanup_error:
                            print(f"Cleanup error: {cleanup_error}")
                
                # Start routing thread
                routing_thread = threading.Thread(target=routing_worker, daemon=True)
                routing_thread.start()
                
                # Update UI while routing is running
                routing_progress = 40
                while not routing_complete and not progress_dlg.WasCancelled():
                    routing_progress = min(75, routing_progress + 1)
                    progress_dlg.Update(routing_progress, "Routing in progress...")
                    
                    # Check for stop and save request
                    if hasattr(progress_dlg, 'should_stop_and_save') and progress_dlg.should_stop_and_save:
                        print("Stop and save requested...")
                        # Signal the routing thread to stop
                        if hasattr(self, 'routing_cancelled'):
                            self.routing_cancelled = True
                        break
                    
                    wx.MilliSleep(200)  # Update every 200ms
                    wx.GetApp().Yield()  # Keep UI responsive
                
                # Check if user cancelled
                if progress_dlg.WasCancelled():
                    print("🛑 User cancelled - setting cancellation flag")
                    self.routing_cancelled = True
                    # Wait a bit for routing thread to respond to cancellation
                    wx.MilliSleep(2000)  # Wait 2 seconds
                    wx.MessageBox("Routing cancelled by user.", "Cancelled", wx.OK | wx.ICON_INFORMATION)
                    return
                
                # Check if user requested stop and save
                if hasattr(progress_dlg, 'should_stop_and_save') and progress_dlg.should_stop_and_save:
                    # Wait a bit for routing thread to finish current nets
                    wx.MilliSleep(1000)
                    progress_dlg.Update(80, "Stopping and saving current progress...")
                    print("Stopping and saving current progress...")
                
                # Check for routing errors
                if routing_error:
                    raise routing_error
                
                results = routing_results
                
                progress_dlg.Update(80, "Importing routes...")
                if results['success']:
                    print("Importing routes...")
                    self._import_routes(board, results)
                    progress_dlg.Update(100, "Routing complete!")
                    
                    # Show results
                    stats = results.get('stats', {})
                    success_rate = stats.get('success_rate', 0)
                    total_nets = stats.get('total_nets', 0)
                    successful_nets = stats.get('successful_nets', 0)
                    
                    message = f"""Routing completed successfully!

Statistics:
• Total nets: {total_nets}
• Successfully routed: {successful_nets}
• Success rate: {success_rate:.1f}%
• Time: {stats.get('total_time_seconds', 0):.1f} seconds

Note: Check the PCB editor to see the routed tracks."""
                    
                    wx.MessageBox(message, "Routing Complete", 
                                wx.OK | wx.ICON_INFORMATION)
                else:
                    error = results.get('error', 'Unknown error')
                    print(f"Routing failed: {error}")
                    wx.MessageBox(f"Routing failed: {error}", 
                                "Routing Error", wx.OK | wx.ICON_ERROR)
                    
            finally:
                progress_dlg.Destroy()
                
        except ImportError as e:
            print(f"❌ Import error in _route_board_gpu: {e}")
            import traceback
            print(f"❌ Import traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Import error: {str(e)}\n\nPlease ensure CuPy is installed for GPU acceleration.\n\nCheck console for detailed traceback.", 
                        "Import Error", wx.OK | wx.ICON_ERROR)
        except Exception as e:
            print(f"❌ Critical error in _route_board_gpu: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Routing error: {str(e)}\n\nCheck console for detailed traceback.", 
                        "Routing Error", wx.OK | wx.ICON_ERROR)
    
    def _route_board_simple(self, board, config, debug_dialog=None, debug_writer=None):
        """GPU routing with dynamic CuPy path injection"""
        
        def debug_print(text):
            """Print to console, debug window, and file"""
            print(text)
            if debug_dialog:
                debug_dialog.append_text(text)
            if debug_writer:
                debug_writer(text)
        
        try:
            debug_print("=== ORTHOROUTE GPU ROUTING ===")
            debug_print("🔧 INJECTING SYSTEM PYTHON PATHS FOR GPU ACCESS...")
            
            # CRITICAL: Inject system Python site-packages into KiCad's Python path
            import sys
            
            # Add system Python site-packages where CuPy is installed
            system_site_packages = [
                r"C:\Users\Benchoff\AppData\Roaming\Python\Python312\site-packages",
                r"C:\Python312\Lib\site-packages"
            ]
            
            paths_added = 0
            for path in system_site_packages:
                if path not in sys.path:
                    sys.path.insert(0, path)  # Insert at beginning for priority
                    paths_added += 1
                    debug_print(f"  ✓ Added: {path}")
                else:
                    debug_print(f"  ⚠ Already present: {path}")
            
            debug_print(f"🎯 Injected {paths_added} new paths, total paths: {len(sys.path)}")
            
            # Now test CuPy import after path injection
            debug_print("\n🚀 TESTING GPU ACCESS AFTER PATH INJECTION...")
            try:
                import cupy as cp
                device = cp.cuda.Device()
                props = cp.cuda.runtime.getDeviceProperties(device.id)
                gpu_name = props["name"].decode("utf-8")
                debug_print(f"✅ CuPy SUCCESSFULLY IMPORTED!")
                debug_print(f"✅ GPU DETECTED: {gpu_name}")
                debug_print(f"✅ CuPy version: {cp.__version__}")
                gpu_available = True
            except Exception as e:
                debug_print(f"❌ CuPy import failed: {str(e)}")
                gpu_available = False
                
            # Basic board info
            bbox = board.GetBoardEdgesBoundingBox()
            width_mm = bbox.GetWidth() / 1e6
            height_mm = bbox.GetHeight() / 1e6
            layers = board.GetCopperLayerCount()
            
            debug_print(f"\n📊 Board: {width_mm:.2f} x {height_mm:.2f} mm, {layers} layers")
            
            # Footprints and pads
            footprints = list(board.GetFootprints())
            total_pads = sum(len(list(fp.Pads())) for fp in footprints)
            debug_print(f"📊 Footprints: {len(footprints)}, Total pads: {total_pads}")
            
            # Net analysis
            net_count = board.GetNetCount()
            debug_print(f"📊 Total nets: {net_count}")
            
            routeable_nets = []
            nets_with_pads = 0
            
            for netcode in range(1, net_count):  # Skip net 0
                net_info = board.GetNetInfo().GetNetItem(netcode)
                if not net_info:
                    continue
                    
                net_name = net_info.GetNetname()
                
                # Skip power/ground nets (common names)
                power_net_names = ['GND', 'VCC', 'VDD', '+5V', '+3V3', '+12V', '-12V', 'VEE', 'VSS', 'AGND', 'DGND']
                is_power_net = any(power_name in net_name.upper() for power_name in power_net_names)
                
                # Check if net has copper fills/zones
                has_copper_fill = self._net_has_copper_fill(board, netcode)
                
                # Count pads for this net
                pads_for_net = []
                for fp in footprints:
                    for pad in fp.Pads():
                        pad_net = pad.GetNet()
                        if pad_net and pad_net.GetNetCode() == netcode:
                            pads_for_net.append(pad)
                
                if len(pads_for_net) > 0:
                    nets_with_pads += 1
                    
                    status = ""
                    if is_power_net:
                        status += " [POWER/GND]"
                    if has_copper_fill:
                        status += " [HAS_FILL]"
                    
                    if nets_with_pads <= 10:  # Show details for first 10
                        debug_print(f"  Net {netcode} '{net_name}': {len(pads_for_net)} pads{status}")
                    
                    # Only count as routeable if it's not a power net and doesn't have fills
                    if len(pads_for_net) >= 2 and not is_power_net and not has_copper_fill:
                        routeable_nets.append((net_name, net_info, pads_for_net))
            
            debug_print(f"📊 Nets with pads: {nets_with_pads}")
            debug_print(f"🎯 Routeable nets (excluding power/fills): {len(routeable_nets)}")
            
            if not gpu_available:
                debug_print("\n❌ GPU ROUTING UNAVAILABLE")
                debug_print("CuPy could not be imported even after path injection.")
                debug_print("GPU routing requires CUDA/CuPy access.")
                
                wx.MessageBox(
                    f"❌ GPU ROUTING UNAVAILABLE\n\n"
                    f"CuPy could not be imported even after path injection.\n"
                    f"GPU routing requires CUDA/CuPy access.\n\n"
                    f"Found {len(routeable_nets)} nets that need routing.",
                    "GPU Required",
                    wx.OK | wx.ICON_ERROR
                )
                return False
            
            if len(routeable_nets) == 0:
                debug_print("\n📋 No signal nets need routing!")
                debug_print("This is normal if your board only has:")
                debug_print("- Power/ground nets (handled by copper pours)")
                debug_print("- Single-pin nets (test points, etc.)")
                debug_print("- Nets already routed with copper fills")
                
                wx.MessageBox("No signal nets need routing!\n\nPower/ground nets are handled by copper pours.\nAll other nets appear to be single-pin or already filled.",
                            "No Routing Needed", wx.OK | wx.ICON_INFORMATION)
                return True
            
            # GPU IS AVAILABLE AND WE HAVE NETS - PROCEED WITH ACTUAL ROUTING
            debug_print(f"\n🚀 STARTING GPU ROUTING ON {gpu_name}...")
            debug_print(f"Processing {len(routeable_nets)} signal nets...")
            
            routed_count = 0
            failed_count = 0
            
            for net_name, net_info, pads in routeable_nets:
                debug_print(f"\n⚡ Routing net: {net_name} ({len(pads)} pads)")
                
                try:
                    # ACTUAL GPU ROUTING IMPLEMENTATION
                    success = self._route_net_gpu(board, net_info, pads, debug_print, cp)
                    if success:
                        routed_count += 1
                        debug_print(f"  ✅ {net_name} routed successfully")
                    else:
                        failed_count += 1
                        debug_print(f"  ❌ {net_name} routing failed")
                        
                except Exception as e:
                    failed_count += 1
                    debug_print(f"  ❌ {net_name} error: {str(e)}")
                    import traceback
                    debug_print(traceback.format_exc())
            
            # Final results
            debug_print(f"\n🏁 ROUTING COMPLETE:")
            debug_print(f"  ✅ Successfully routed: {routed_count} nets")
            debug_print(f"  ❌ Failed to route: {failed_count} nets")
            debug_print(f"  🎯 Total processed: {len(routeable_nets)} nets")
            
            # Safer display refresh - avoid pcbnew.Refresh() which can crash
            try:
                import pcbnew
                # Use board-specific refresh instead of global refresh
                board_editor = pcbnew.GetBoard()
                if board_editor:
                    # Force board update without full refresh
                    board_editor.BuildListOfNets()
                    debug_print("  🔄 Board display updated safely")
                else:
                    debug_print("  ⚠ Could not get board editor for refresh")
            except Exception as e:
                debug_print(f"  ⚠ Display refresh failed (not critical): {str(e)}")
            
            if routed_count > 0:
                # Use CallAfter to ensure dialog appears on main UI thread
                def show_success_dialog():
                    try:
                        wx.MessageBox(
                            f"🚀 GPU ROUTING COMPLETE!\n\n"
                            f"✅ Successfully routed: {routed_count} nets\n"
                            f"❌ Failed to route: {failed_count} nets\n"
                            f"🎯 Total processed: {len(routeable_nets)} nets\n\n"
                            f"GPU: {gpu_name}",
                            "Routing Complete",
                            wx.OK | wx.ICON_INFORMATION
                        )
                        debug_print("🎉 Success dialog displayed")
                    except Exception as e:
                        debug_print(f"❌ Error showing success dialog: {str(e)}")
                
                # Schedule dialog on main thread
                wx.CallAfter(show_success_dialog)
                debug_print("📅 Success dialog scheduled")
            else:
                # Use CallAfter for failure dialog too
                def show_failure_dialog():
                    try:
                        wx.MessageBox(
                            f"⚠ ROUTING ATTEMPTED BUT NO SUCCESS\n\n"
                            f"❌ Failed to route: {failed_count} nets\n"
                            f"🎯 Total attempted: {len(routeable_nets)} nets\n\n"
                            f"Check debug output for details.",
                            "Routing Issues",
                            wx.OK | wx.ICON_WARNING
                        )
                        debug_print("⚠ Failure dialog displayed")
                    except Exception as e:
                        debug_print(f"❌ Error showing failure dialog: {str(e)}")
                
                # Schedule dialog on main thread  
                wx.CallAfter(show_failure_dialog)
                debug_print("📅 Failure dialog scheduled")
            
            debug_print(f"🏁 Routing completed with result: {routed_count > 0}")
            
            # Add a small delay to ensure all operations complete
            import time
            time.sleep(0.1)
            
            # Also try immediate dialog as fallback
            if routed_count > 0:
                try:
                    # Try immediate dialog first
                    wx.MessageBox(
                        f"🚀 GPU ROUTING COMPLETE!\n\n"
                        f"✅ Successfully routed: {routed_count} nets\n"
                        f"❌ Failed to route: {failed_count} nets\n"
                        f"🎯 Total processed: {len(routeable_nets)} nets\n\n"
                        f"GPU: {gpu_name}",
                        "Routing Complete",
                        wx.OK | wx.ICON_INFORMATION
                    )
                    debug_print("🎉 Immediate success dialog displayed")
                except Exception as e:
                    debug_print(f"❌ Immediate dialog failed: {str(e)}")
            
            debug_print("🏁 Plugin execution completed")
            return routed_count > 0
            
        except Exception as e:
            debug_print(f"❌ ERROR in GPU routing: {str(e)}")
            import traceback
            debug_print(traceback.format_exc())
            wx.MessageBox(f"GPU routing failed: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
            return False
    
    def _route_net_gpu(self, board, net_info, pads, debug_print, cp):
        """
        Route a single net using GPU acceleration with comprehensive error handling
        
        Args:
            board: KiCad board object
            net_info: Net information object
            pads: List of pads for this net
            debug_print: Debug output function
            cp: CuPy module for GPU computation
            
        Returns:
            bool: True if routing successful, False otherwise
        """
        try:
            net_name = net_info.GetNetname()
            netcode = net_info.GetNetCode()
            
            debug_print(f"    📍 Analyzing {len(pads)} pads for net {net_name}")
            
            # Get pad positions and layers with validation
            pad_positions = []
            for i, pad in enumerate(pads):
                try:
                    pos = pad.GetPosition()
                    layer = pad.GetLayer()
                    pad_positions.append((pos.x, pos.y, layer))
                    debug_print(f"      Pad {i+1}: ({pos.x/1e6:.3f}, {pos.y/1e6:.3f}) layer {layer}")
                except Exception as e:
                    debug_print(f"      ❌ Error reading pad {i}: {str(e)}")
                    return False
            
            if len(pad_positions) < 2:
                debug_print(f"    ⚠ Net {net_name} has less than 2 pads, skipping")
                return False
            
            # Get board bounds for routing grid with validation
            try:
                bbox = board.GetBoardEdgesBoundingBox()
                board_min_x = bbox.GetLeft()
                board_min_y = bbox.GetTop()
                board_max_x = bbox.GetRight()
                board_max_y = bbox.GetBottom()
                
                if board_max_x <= board_min_x or board_max_y <= board_min_y:
                    debug_print(f"    ❌ Invalid board bounds: ({board_min_x}, {board_min_y}) to ({board_max_x}, {board_max_y})")
                    return False
                    
                debug_print(f"    📏 Board bounds: ({board_min_x/1e6:.2f}, {board_min_y/1e6:.2f}) to ({board_max_x/1e6:.2f}, {board_max_y/1e6:.2f}) mm")
                
            except Exception as e:
                debug_print(f"    ❌ Error getting board bounds: {str(e)}")
                return False
            
            # Set up routing grid with memory safety checks
            try:
                grid_size = int(0.25 * 1e6)  # Increased to 0.25mm for memory safety
                grid_width = (board_max_x - board_min_x) // grid_size + 1
                grid_height = (board_max_y - board_min_y) // grid_size + 1
                layers = board.GetCopperLayerCount()
                
                # Memory safety check
                total_cells = grid_width * grid_height * layers
                memory_estimate_mb = total_cells * 8 / (1024 * 1024)  # 8 bytes per cell
                
                debug_print(f"    📊 Grid: {grid_width}x{grid_height}x{layers}, resolution: 0.25mm")
                debug_print(f"    💾 Estimated memory: {memory_estimate_mb:.1f} MB ({total_cells:,} cells)")
                
                if memory_estimate_mb > 1000:  # Limit to 1GB
                    debug_print(f"    ❌ Grid too large ({memory_estimate_mb:.1f} MB), skipping net")
                    return False
                    
            except Exception as e:
                debug_print(f"    ❌ Error calculating grid size: {str(e)}")
                return False
            
            # Create routing grid on GPU with error handling
            try:
                debug_print(f"    🚀 Allocating GPU memory...")
                grid_shape = (grid_width, grid_height, layers)
                
                # Test small allocation first
                test_array = cp.zeros((10, 10, 2), dtype=cp.uint8)
                del test_array
                
                obstacles = cp.zeros(grid_shape, dtype=cp.uint8)
                distances = cp.full(grid_shape, 999999, dtype=cp.int32)
                
                debug_print(f"    ✅ GPU memory allocated successfully")
                
            except Exception as e:
                debug_print(f"    ❌ GPU memory allocation failed: {str(e)}")
                debug_print(f"    💡 Try reducing board size or using coarser grid resolution")
                return False
            
            # Mark existing tracks as obstacles with error handling
            try:
                debug_print(f"    🚧 Marking existing tracks as obstacles...")
                self._mark_existing_tracks_gpu(board, obstacles, board_min_x, board_min_y, grid_size, cp, debug_print)
            except Exception as e:
                debug_print(f"    ⚠ Warning: Error marking obstacles: {str(e)}")
                # Continue without obstacle marking
            
            # Convert pad positions to grid coordinates with validation
            try:
                start_pads = []
                target_pads = []
                
                for i, (x, y, layer) in enumerate(pad_positions):
                    grid_x = (x - board_min_x) // grid_size
                    grid_y = (y - board_min_y) // grid_size
                    grid_layer = min(layer, layers - 1) if layer < layers else 0
                    
                    # Clamp to grid bounds
                    grid_x = max(0, min(grid_width - 1, grid_x))
                    grid_y = max(0, min(grid_height - 1, grid_y))
                    
                    if i == 0:
                        start_pads.append((grid_x, grid_y, grid_layer))
                    else:
                        target_pads.append((grid_x, grid_y, grid_layer))
                
                debug_print(f"    🎯 Start: {start_pads[0]}, Targets: {target_pads}")
                
            except Exception as e:
                debug_print(f"    ❌ Error converting pad positions: {str(e)}")
                return False
            
            # Run GPU wavefront routing algorithm with comprehensive error handling
            try:
                debug_print(f"    ⚡ Starting GPU wavefront algorithm...")
                path_found = self._wavefront_gpu(obstacles, distances, start_pads[0], target_pads, cp, debug_print)
                
                if not path_found:
                    debug_print(f"    ❌ Wavefront routing failed for {net_name}")
                    return False
                    
            except Exception as e:
                debug_print(f"    ❌ GPU wavefront error: {str(e)}")
                import traceback
                debug_print(f"    📋 Traceback: {traceback.format_exc()}")
                return False
            
            # Extract path and create tracks with comprehensive error handling
            try:
                debug_print(f"    📤 Starting path extraction phase...")
                debug_print(f"    🎯 Extracting path from start {start_pads[0]} to target {target_pads[0]}")
                
                # Add safety check before path extraction
                start_x, start_y, start_layer = start_pads[0]
                target_x, target_y, target_layer = target_pads[0]
                
                # Verify target was actually reached
                try:
                    target_distance = distances[target_x, target_y, target_layer]
                    debug_print(f"    📊 Target distance: {target_distance}")
                    
                    if target_distance >= 999999:
                        debug_print(f"    ❌ Target was never reached (distance={target_distance})")
                        return False
                        
                except Exception as e:
                    debug_print(f"    ❌ Error checking target distance: {str(e)}")
                    return False
                
                # Extract path with detailed error handling
                path_points = self._extract_path_gpu(distances, start_pads[0], target_pads[0], cp, debug_print)
                
                if not path_points:
                    debug_print(f"    ❌ Path extraction failed - no valid path found")
                    return False
                    
                if len(path_points) < 2:
                    debug_print(f"    ❌ Invalid path - only {len(path_points)} points")
                    return False
                    
                debug_print(f"    ✅ Path extracted successfully: {len(path_points)} points")
                debug_print(f"    🛤 Creating tracks from path...")
                
                # Create tracks with detailed error handling
                success = self._create_tracks_from_path_gpu(board, path_points, netcode, board_min_x, board_min_y, grid_size, debug_print)
                
                if success:
                    debug_print(f"    ✅ Successfully routed {net_name}")
                    return True
                else:
                    debug_print(f"    ❌ Track creation failed for {net_name}")
                    return False
                    
            except Exception as e:
                debug_print(f"    ❌ PATH/TRACK CREATION ERROR: {str(e)}")
                import traceback
                debug_print(f"    📋 Full traceback: {traceback.format_exc()}")
                return False
            
        except Exception as e:
            debug_print(f"    ❌ GPU routing error for net {net_info.GetNetname()}: {str(e)}")
            import traceback
            debug_print(f"    📋 Full traceback: {traceback.format_exc()}")
            return False
        finally:
            # Clean up GPU memory
            try:
                if 'obstacles' in locals():
                    del obstacles
                if 'distances' in locals():
                    del distances
                cp.get_default_memory_pool().free_all_blocks()
                debug_print(f"    🧹 GPU memory cleaned up")
            except Exception as e:
                debug_print(f"    ⚠ Warning: GPU cleanup error: {str(e)}")
    
    def _mark_existing_tracks_gpu(self, board, obstacles, board_min_x, board_min_y, grid_size, cp, debug_print):
        """Mark existing tracks and vias as obstacles in the GPU grid"""
        try:
            track_count = 0
            via_count = 0
            
            # Mark all existing tracks
            for track in board.GetTracks():
                if track.GetClass() == "PCB_TRACK":
                    start = track.GetStart()
                    end = track.GetEnd()
                    layer = track.GetLayer()
                    
                    # Convert to grid coordinates
                    start_x = (start.x - board_min_x) // grid_size
                    start_y = (start.y - board_min_y) // grid_size
                    end_x = (end.x - board_min_x) // grid_size
                    end_y = (end.y - board_min_y) // grid_size
                    
                    # Mark line between start and end as obstacle
                    self._mark_line_gpu(obstacles, start_x, start_y, end_x, end_y, layer, cp)
                    track_count += 1
                    
                elif track.GetClass() == "PCB_VIA":
                    pos = track.GetPosition()
                    pos_x = (pos.x - board_min_x) // grid_size
                    pos_y = (pos.y - board_min_y) // grid_size
                    
                    # Mark via on all layers
                    for layer in range(obstacles.shape[2]):
                        if 0 <= pos_x < obstacles.shape[0] and 0 <= pos_y < obstacles.shape[1]:
                            obstacles[pos_x, pos_y, layer] = 1
                    via_count += 1
            
            debug_print(f"      Marked {track_count} tracks and {via_count} vias as obstacles")
            
        except Exception as e:
            debug_print(f"      Warning: Error marking obstacles: {str(e)}")
    
    def _mark_line_gpu(self, obstacles, x0, y0, x1, y1, layer, cp):
        """Mark a line as obstacle using Bresenham's algorithm on GPU"""
        try:
            # Simple line marking - could be optimized with proper GPU kernels
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            steps = max(dx, dy)
            
            if steps == 0:
                if 0 <= x0 < obstacles.shape[0] and 0 <= y0 < obstacles.shape[1] and 0 <= layer < obstacles.shape[2]:
                    obstacles[x0, y0, layer] = 1
                return
            
            x_step = (x1 - x0) / steps
            y_step = (y1 - y0) / steps
            
            for i in range(steps + 1):
                x = int(x0 + i * x_step)
                y = int(y0 + i * y_step)
                if 0 <= x < obstacles.shape[0] and 0 <= y < obstacles.shape[1] and 0 <= layer < obstacles.shape[2]:
                    obstacles[x, y, layer] = 1
                    
        except Exception as e:
            pass  # Ignore individual point errors
    
    def _wavefront_gpu(self, obstacles, distances, start, targets, cp, debug_print):
        """GPU-accelerated wavefront pathfinding algorithm with robust error handling"""
        try:
            start_x, start_y, start_layer = start
            debug_print(f"      🚀 Starting wavefront from ({start_x}, {start_y}, {start_layer})")
            
            # Validate start position
            if not (0 <= start_x < obstacles.shape[0] and 
                   0 <= start_y < obstacles.shape[1] and 
                   0 <= start_layer < obstacles.shape[2]):
                debug_print(f"      ❌ Invalid start position: {start}")
                return False
            
            # Validate targets
            for i, (tx, ty, tl) in enumerate(targets):
                if not (0 <= tx < obstacles.shape[0] and 
                       0 <= ty < obstacles.shape[1] and 
                       0 <= tl < obstacles.shape[2]):
                    debug_print(f"      ❌ Invalid target {i}: ({tx}, {ty}, {tl})")
                    return False
            
            # Initialize wavefront with safety checks
            try:
                distances[:] = 999999
                distances[start_x, start_y, start_layer] = 0
                debug_print(f"      ✓ Wavefront initialized, grid shape: {distances.shape}")
            except Exception as e:
                debug_print(f"      ❌ Failed to initialize distances: {str(e)}")
                return False
            
            # Simple BFS implementation with memory protection
            current_distance = 0
            found_target = False
            max_iterations = 200  # Increased for better routing success
            no_expansion_count = 0  # Count iterations with no expansion
            max_no_expansion = 15   # More lenient - allow more attempts before giving up
            
            # Calculate maximum reasonable distance (Manhattan distance + buffer)
            max_manhattan_distance = 0
            for target_x, target_y, target_layer in targets:
                manhattan_dist = abs(start_x - target_x) + abs(start_y - target_y)
                max_manhattan_distance = max(max_manhattan_distance, manhattan_dist)
            
            max_reasonable_distance = max_manhattan_distance * 3  # Allow for detours
            debug_print(f"      📏 Max reasonable routing distance: {max_reasonable_distance} (Manhattan: {max_manhattan_distance})")
            
            debug_print(f"      🔄 Starting BFS expansion (max {max_iterations} iterations)...")
            
            for iteration in range(max_iterations):
                try:
                    # Early termination if no expansion for several iterations
                    if no_expansion_count >= max_no_expansion:
                        debug_print(f"      ⏹ No expansion for {max_no_expansion} iterations, stopping search")
                        break
                    
                    # Distance-based timeout to prevent infinite routing
                    if current_distance > max_reasonable_distance:
                        debug_print(f"      ⏹ Distance limit reached ({current_distance} > {max_reasonable_distance}), target likely unreachable")
                        break
                    
                    # Memory and bounds safety check at start of each iteration
                    if iteration > 0 and iteration % 25 == 0:  # Check every 25 iterations (less frequent)
                        try:
                            # Test GPU memory is still accessible
                            test_val = distances[start_x, start_y, start_layer]
                            if test_val != 0:
                                debug_print(f"      ⚠ Start position distance changed unexpectedly: {test_val}")
                        except Exception as e:
                            debug_print(f"      ❌ GPU memory corrupted at iteration {iteration}: {str(e)}")
                            return False
                    
                    # Find all cells at current distance - use CPU arrays for safety
                    debug_print(f"      🔍 Finding cells at distance {current_distance}...")
                    
                    # Use more conservative memory approach
                    current_mask = (distances == current_distance)
                    current_indices = cp.where(current_mask)
                    
                    if len(current_indices) != 3:  # Should be (x, y, z) indices
                        debug_print(f"      ❌ Invalid indices structure: {len(current_indices)} dimensions")
                        break
                    
                    if len(current_indices[0]) == 0:
                        debug_print(f"      ⏹ No more cells to expand at iteration {iteration}")
                        break
                    
                    num_cells = len(current_indices[0])
                    if num_cells > 1000:  # More reasonable limit for normal routing
                        debug_print(f"      ⚠ Too many cells ({num_cells}), limiting to 1000 for stability")
                        # Truncate arrays to prevent memory explosion
                        current_indices = (
                            current_indices[0][:1000],
                            current_indices[1][:1000], 
                            current_indices[2][:1000]
                        )
                        num_cells = 1000
                    
                    # Emergency stop if expansion is getting out of control
                    if num_cells > 500 and iteration > 50:
                        debug_print(f"      🛑 EMERGENCY STOP: {num_cells} cells at iteration {iteration} - preventing crash")
                        debug_print(f"      💡 Try reducing grid resolution or using simpler routing")
                        return False
                    
                    debug_print(f"      📊 Iteration {iteration}: expanding {num_cells} cells")
                    
                    # Check if we reached any target BEFORE expanding neighbors (GPU computation)
                    target_found = False
                    found_target = None
                    try:
                        for target_x, target_y, target_layer in targets:
                            # Create target mask on GPU
                            target_mask = (
                                (current_indices[0] == target_x) & 
                                (current_indices[1] == target_y) & 
                                (current_indices[2] == target_layer)
                            )
                            if cp.any(target_mask):
                                target_found = True
                                found_target = (target_x, target_y, target_layer)
                                debug_print(f"      🎯 TARGET FOUND at iteration {iteration}, distance {current_distance}")
                                debug_print(f"      ✅ Target ({target_x}, {target_y}, {target_layer}) reached")
                                break
                        
                        if target_found:
                            return True
                            
                    except Exception as e:
                        debug_print(f"      ⚠ GPU target check failed, falling back to CPU: {str(e)}")
                        # Fallback to CPU conversion only if GPU fails
                        try:
                            x_coords = cp.asnumpy(current_indices[0])
                            y_coords = cp.asnumpy(current_indices[1]) 
                            z_coords = cp.asnumpy(current_indices[2])
                            for target_x, target_y, target_layer in targets:
                                for i in range(len(x_coords)):
                                    if (x_coords[i] == target_x and 
                                        y_coords[i] == target_y and 
                                        z_coords[i] == target_layer):
                                        debug_print(f"      🎯 TARGET FOUND at iteration {iteration}, distance {current_distance}")
                                        debug_print(f"      ✅ Target ({target_x}, {target_y}, {target_layer}) reached")
                                        return True
                        except Exception as e2:
                            debug_print(f"      ❌ Both GPU and CPU target check failed: {str(e2)}")
                            return False
                    
                    # GPU-based neighbor expansion for much better performance
                    updated = False
                    try:
                        # Get current cell coordinates
                        curr_x = current_indices[0]
                        curr_y = current_indices[1] 
                        curr_z = current_indices[2]
                        
                        # Generate all possible neighbors on GPU
                        # 4-connected neighbors + layer changes
                        neighbor_offsets = [
                            (1, 0, 0), (-1, 0, 0),  # x neighbors
                            (0, 1, 0), (0, -1, 0),  # y neighbors
                        ]
                        
                        # Add layer changes if multiple layers
                        if obstacles.shape[2] > 1:
                            neighbor_offsets.extend([(0, 0, 1), (0, 0, -1)])
                        
                        # Vectorized neighbor expansion
                        new_distance = current_distance + 1
                        update_count = 0
                        
                        for dx, dy, dz in neighbor_offsets:
                            # Calculate neighbor coordinates
                            nx = curr_x + dx
                            ny = curr_y + dy  
                            nz = curr_z + dz
                            
                            # Bounds checking mask
                            valid_mask = (
                                (nx >= 0) & (nx < obstacles.shape[0]) &
                                (ny >= 0) & (ny < obstacles.shape[1]) & 
                                (nz >= 0) & (nz < obstacles.shape[2])
                            )
                            
                            if cp.any(valid_mask):
                                # Filter to valid coordinates only
                                valid_nx = nx[valid_mask]
                                valid_ny = ny[valid_mask] 
                                valid_nz = nz[valid_mask]
                                
                                # Check obstacles and distances for valid neighbors
                                neighbor_obstacles = obstacles[valid_nx, valid_ny, valid_nz]
                                neighbor_distances = distances[valid_nx, valid_ny, valid_nz]
                                
                                # Update cells that are free and have longer distance
                                update_mask = (neighbor_obstacles == 0) & (neighbor_distances > new_distance)
                                
                                if cp.any(update_mask):
                                    # Update distances for valid neighbors
                                    final_nx = valid_nx[update_mask]
                                    final_ny = valid_ny[update_mask]
                                    final_nz = valid_nz[update_mask]
                                    
                                    distances[final_nx, final_ny, final_nz] = new_distance
                                    update_count += len(final_nx)
                                    updated = True
                        
                        if update_count > 0:
                            debug_print(f"      ⚡ GPU updated {update_count} cells in parallel")
                            
                    except Exception as e:
                        debug_print(f"      ⚠ GPU neighbor expansion failed, trying CPU fallback: {str(e)}")
                        # CPU fallback only if GPU fails
                        try:
                            x_coords = cp.asnumpy(current_indices[0])
                            y_coords = cp.asnumpy(current_indices[1])
                            z_coords = cp.asnumpy(current_indices[2])
                            
                            max_cells_per_iteration = min(len(x_coords), 100)  # Reduced for stability
                            cells_to_process = min(len(x_coords), max_cells_per_iteration)
                            
                            debug_print(f"      🔄 CPU fallback: processing {cells_to_process} cells")
                            
                            for i in range(cells_to_process):
                                x, y, layer = int(x_coords[i]), int(y_coords[i]), int(z_coords[i])
                                
                                # 4-connected neighbors (no diagonal movement)
                                neighbors = [
                                    (x+1, y, layer), (x-1, y, layer),
                                    (x, y+1, layer), (x, y-1, layer)
                                ]
                                
                                # Add layer changes if multiple layers exist
                                if obstacles.shape[2] > 1:
                                    if layer + 1 < obstacles.shape[2]:
                                        neighbors.append((x, y, layer+1))
                                    if layer - 1 >= 0:
                                        neighbors.append((x, y, layer-1))
                                
                                for nx, ny, nl in neighbors:
                                    if (0 <= nx < obstacles.shape[0] and 
                                        0 <= ny < obstacles.shape[1] and 
                                        0 <= nl < obstacles.shape[2]):
                                        
                                        if obstacles[nx, ny, nl] == 0 and distances[nx, ny, nl] > current_distance + 1:
                                            distances[nx, ny, nl] = current_distance + 1
                                            updated = True
                                            
                        except Exception as e2:
                            debug_print(f"      ❌ Both GPU and CPU neighbor expansion failed: {str(e2)}")
                            break
                    
                    if not updated:
                        no_expansion_count += 1
                        debug_print(f"      ⏹ No updates at iteration {iteration} (no expansion: {no_expansion_count}/{max_no_expansion})")
                        if no_expansion_count >= max_no_expansion:
                            debug_print(f"      ⏹ Search complete - no expansion for {max_no_expansion} iterations")
                            break
                    else:
                        no_expansion_count = 0  # Reset counter if we had expansion
                        
                        # Check if we're getting closer to any target (GPU computation when possible)
                        min_distance_to_target = float('inf')
                        try:
                            # GPU-based distance calculation
                            for target_x, target_y, target_layer in targets:
                                # Calculate Manhattan distance on GPU
                                distances_to_target = (
                                    cp.abs(current_indices[0] - target_x) + 
                                    cp.abs(current_indices[1] - target_y)
                                )
                                if len(distances_to_target) > 0:
                                    min_dist = float(cp.min(distances_to_target))
                                    min_distance_to_target = min(min_distance_to_target, min_dist)
                                    
                        except Exception as e:
                            # Fallback to CPU calculation if needed
                            debug_print(f"      ⚠ GPU distance calc failed, using CPU: {str(e)}")
                            try:
                                x_coords = cp.asnumpy(current_indices[0])
                                y_coords = cp.asnumpy(current_indices[1])
                                for target_x, target_y, target_layer in targets:
                                    for i in range(min(len(x_coords), 100)):  # Limit to first 100 for speed
                                        cell_x, cell_y = int(x_coords[i]), int(y_coords[i])
                                        distance_to_target = abs(cell_x - target_x) + abs(cell_y - target_y)
                                        min_distance_to_target = min(min_distance_to_target, distance_to_target)
                            except:
                                min_distance_to_target = current_distance  # Fallback estimate
                        
                        debug_print(f"      📍 Closest distance to target: {min_distance_to_target} cells")
                        
                    current_distance += 1
                    
                    # Progress update every 10 iterations (reduced frequency)
                    if iteration % 10 == 0 and iteration > 0:
                        debug_print(f"      🔄 Progress: {iteration} iterations, distance {current_distance}")
                    
                    iteration += 1  # Increment iteration counter
                    
                except Exception as e:
                    debug_print(f"      ❌ Error in iteration {iteration}: {str(e)}")
                    debug_print(f"      🛑 Stopping wavefront expansion due to error")
                    return False
            
            debug_print(f"      ❌ No path found after {iteration+1} iterations")
            return False
            
        except Exception as e:
            debug_print(f"      ❌ Wavefront algorithm error: {str(e)}")
            import traceback
            debug_print(f"      📋 Traceback: {traceback.format_exc()}")
            return False
    
    def _extract_path_gpu(self, distances, start, target, cp, debug_print):
        """Extract the routing path from the distance grid with comprehensive error handling"""
        try:
            target_x, target_y, target_layer = target
            debug_print(f"      🎯 Extracting path to target ({target_x}, {target_y}, {target_layer})")
            
            # Validate target position
            if not (0 <= target_x < distances.shape[0] and 
                   0 <= target_y < distances.shape[1] and 
                   0 <= target_layer < distances.shape[2]):
                debug_print(f"      ❌ Invalid target position: {target}")
                return None
            
            # Check target distance
            try:
                current_distance = distances[target_x, target_y, target_layer]
                debug_print(f"      📊 Target distance: {current_distance}")
                
                if current_distance >= 999999:
                    debug_print(f"      ❌ Target unreachable (distance: {current_distance})")
                    return None
                    
            except Exception as e:
                debug_print(f"      ❌ Error reading target distance: {str(e)}")
                return None
            
            # Initialize path
            path = [(target_x, target_y, target_layer)]
            current_x, current_y, current_layer = target_x, target_y, target_layer
            
            debug_print(f"      🔄 Tracing path backward from distance {current_distance}...")
            max_path_length = 1000  # Safety limit
            
            # Trace back to start
            while current_distance > 0 and len(path) < max_path_length:
                found_next = False
                
                # Check 4 neighbors (no diagonal or layer changes for now)
                neighbors = [
                    (current_x+1, current_y, current_layer), 
                    (current_x-1, current_y, current_layer),
                    (current_x, current_y+1, current_layer), 
                    (current_x, current_y-1, current_layer)
                ]
                
                # Add layer changes if multiple layers
                if distances.shape[2] > 1:
                    if current_layer + 1 < distances.shape[2]:
                        neighbors.append((current_x, current_y, current_layer+1))
                    if current_layer - 1 >= 0:
                        neighbors.append((current_x, current_y, current_layer-1))
                
                for nx, ny, nl in neighbors:
                    try:
                        # Bounds check
                        if not (0 <= nx < distances.shape[0] and 
                               0 <= ny < distances.shape[1] and 
                               0 <= nl < distances.shape[2]):
                            continue
                        
                        # Check if this neighbor has the previous distance
                        neighbor_distance = distances[nx, ny, nl]
                        
                        if neighbor_distance == current_distance - 1:
                            path.append((nx, ny, nl))
                            current_x, current_y, current_layer = nx, ny, nl
                            current_distance -= 1
                            found_next = True
                            break
                            
                    except Exception as e:
                        debug_print(f"      ⚠ Error checking neighbor ({nx}, {ny}, {nl}): {str(e)}")
                        continue
                
                if not found_next:
                    debug_print(f"      ❌ Path trace failed at distance {current_distance}")
                    debug_print(f"      📍 Stuck at position ({current_x}, {current_y}, {current_layer})")
                    debug_print(f"      🔍 Checked {len(neighbors)} neighbors")
                    return None
                
                # Progress update every 50 steps
                if len(path) % 50 == 0:
                    debug_print(f"      📈 Path length: {len(path)}, current distance: {current_distance}")
            
            if len(path) >= max_path_length:
                debug_print(f"      ❌ Path too long ({len(path)} points), aborting")
                return None
            
            if current_distance > 0:
                debug_print(f"      ❌ Failed to reach start (stopped at distance {current_distance})")
                return None
            
            path.reverse()  # Start to target order
            debug_print(f"      ✅ Path extracted successfully: {len(path)} points")
            debug_print(f"      📍 Start: {path[0]}, End: {path[-1]}")
            
            return path
            
        except Exception as e:
            debug_print(f"      ❌ Path extraction error: {str(e)}")
            import traceback
            debug_print(f"      📋 Traceback: {traceback.format_exc()}")
            return None
    
    def _optimize_path_segments(self, board_points, debug_print):
        """Consolidate straight-line segments to reduce track count"""
        if len(board_points) < 3:
            return board_points
            
        optimized = [board_points[0]]  # Always include start point
        
        for i in range(1, len(board_points) - 1):
            prev_x, prev_y, prev_layer = board_points[i-1]
            curr_x, curr_y, curr_layer = board_points[i]
            next_x, next_y, next_layer = board_points[i+1]
            
            # Check if this point is on a straight line
            # If direction changes or layer changes, keep the point
            prev_dx = curr_x - prev_x
            prev_dy = curr_y - prev_y
            next_dx = next_x - curr_x
            next_dy = next_y - curr_y
            
            # Keep point if direction changes or layer changes
            direction_changed = (prev_dx != next_dx) or (prev_dy != next_dy)
            layer_changed = (prev_layer != curr_layer) or (curr_layer != next_layer)
            
            if direction_changed or layer_changed:
                optimized.append(board_points[i])
        
        optimized.append(board_points[-1])  # Always include end point
        return optimized
    
    def _create_tracks_from_path_gpu(self, board, path_points, netcode, board_min_x, board_min_y, grid_size, debug_print):
        """Create KiCad tracks from the GPU-calculated path with comprehensive error handling"""
        try:
            debug_print(f"      🛤 Creating tracks from {len(path_points)} path points")
            
            if len(path_points) < 2:
                debug_print(f"      ❌ Path too short: {len(path_points)} points")
                return False
            
            tracks_created = 0
            vias_created = 0
            
            # Get the net info
            try:
                net_info = board.GetNetInfo().GetNetItem(netcode)
                if not net_info:
                    debug_print(f"      ❌ Could not get net info for netcode {netcode}")
                    return False
                debug_print(f"      ✅ Got net info for netcode {netcode}")
            except Exception as e:
                debug_print(f"      ❌ Error getting net info: {str(e)}")
                return False
            
            # Convert path points back to board coordinates
            try:
                board_points = []
                for i, (grid_x, grid_y, layer) in enumerate(path_points):
                    try:
                        board_x = board_min_x + grid_x * grid_size
                        board_y = board_min_y + grid_y * grid_size
                        board_points.append((board_x, board_y, layer))
                        
                        if i % 50 == 0:  # Progress update
                            debug_print(f"      📍 Converted {i+1}/{len(path_points)} points")
                            
                    except Exception as e:
                        debug_print(f"      ❌ Error converting point {i}: {str(e)}")
                        continue
                
                debug_print(f"      ✅ Converted {len(board_points)} board points")
                
            except Exception as e:
                debug_print(f"      ❌ Error in coordinate conversion: {str(e)}")
                return False
            
            # Optimize path by consolidating straight-line segments
            optimized_points = self._optimize_path_segments(board_points, debug_print)
            debug_print(f"      🔧 Optimized {len(board_points)} points to {len(optimized_points)} segments")
            
            # Create tracks between consecutive points on the same layer
            for i in range(len(optimized_points) - 1):
                try:
                    x1, y1, layer1 = optimized_points[i]
                    x2, y2, layer2 = optimized_points[i + 1]
                    
                    debug_print(f"        📍 Segment {i}: ({x1:.0f},{y1:.0f},L{layer1}) -> ({x2:.0f},{y2:.0f},L{layer2})")
                    
                    if layer1 == layer2:
                        # Same layer - create track with safer error handling
                        try:
                            import pcbnew
                            track = pcbnew.PCB_TRACK(board)
                            track.SetStart(pcbnew.VECTOR2I(int(x1), int(y1)))
                            track.SetEnd(pcbnew.VECTOR2I(int(x2), int(y2)))
                            track.SetWidth(int(0.2 * 1e6))  # 0.2mm width
                            track.SetLayer(layer1)
                            track.SetNet(net_info)
                            
                            # Add track safely
                            board.Add(track)
                            tracks_created += 1
                            
                            # Reduced logging frequency to prevent console spam
                            if tracks_created % 25 == 0:
                                debug_print(f"        📈 Created {tracks_created} tracks so far...")
                                
                        except Exception as e:
                            debug_print(f"      ❌ Track creation error at segment {i}: {str(e)}")
                            # Don't spam traceback for every failure - just continue
                            continue
                    else:
                        # Layer change - create via with safer error handling
                        try:
                            import pcbnew
                            via = pcbnew.PCB_VIA(board)
                            via.SetPosition(pcbnew.VECTOR2I(int(x1), int(y1)))
                            via.SetWidth(int(0.4 * 1e6))  # 0.4mm via size
                            via.SetDrill(int(0.2 * 1e6))  # 0.2mm drill
                            via.SetNet(net_info)
                            via.SetViaType(pcbnew.VIATYPE_THROUGH)
                            
                            # Add via safely
                            board.Add(via)
                            vias_created += 1
                            
                            debug_print(f"        🔗 Via created at ({x1:.0f},{y1:.0f})")
                            
                            # Create track on second layer if not last point - simplified
                            if i + 1 < len(optimized_points) - 1:
                                track = pcbnew.PCB_TRACK(board)
                                track.SetStart(pcbnew.VECTOR2I(int(x1), int(y1)))
                                track.SetEnd(pcbnew.VECTOR2I(int(x2), int(y2)))
                                track.SetWidth(int(0.2 * 1e6))  # 0.2mm width
                                track.SetLayer(layer2)
                                track.SetNet(net_info)
                                board.Add(track)
                                tracks_created += 1
                                
                        except Exception as e:
                            debug_print(f"      ❌ Via creation error at segment {i}: {str(e)}")
                            # Continue without spamming traceback
                            continue
                
                except Exception as e:
                    debug_print(f"      ❌ Segment processing error at {i}: {str(e)}")
                    import traceback
                    debug_print(f"      📋 Segment traceback: {traceback.format_exc()}")
                    continue
            
            debug_print(f"      ✅ Track creation complete: {tracks_created} tracks, {vias_created} vias")
            
            if tracks_created == 0 and vias_created == 0:
                debug_print(f"      ❌ No tracks or vias created from {len(board_points)} board points")
                return False
                
            return tracks_created > 0
            
        except Exception as e:
            debug_print(f"      ❌ Track creation fatal error: {str(e)}")
            import traceback
            debug_print(f"      📋 Fatal traceback: {traceback.format_exc()}")
            return False
    
    def _net_has_copper_fill(self, board, netcode):
        """Check if a net has copper fills/zones"""
        try:
            # Get all zones (copper fills) on the board
            zones = []
            for i in range(board.GetAreaCount()):
                zone = board.GetArea(i)
                if zone:
                    zones.append(zone)
            
            # Check if any zone is assigned to this net
            for zone in zones:
                try:
                    zone_net = zone.GetNet()
                    if zone_net and zone_net.GetNetCode() == netcode:
                        return True
                except:
                    continue
            
            return False
            
        except Exception:
            # If we can't determine, assume no fill
            return False
    
    def _export_board_data(self, board, debug_print=None, config=None) -> Dict:
        """Export board data for routing"""
        if debug_print is None:
            debug_print = print
            
        debug_print("🔍 Starting board data export...")
        
        try:
            # Get layer count first
            layer_count = board.GetCopperLayerCount()
            debug_print(f"📚 Copper layers: {layer_count}")
            
            # Get net count using the most reliable method
            try:
                net_count = board.GetNetCount()
                debug_print(f"🔗 Total nets in board: {net_count}")
            except Exception as e:
                debug_print(f"❌ Failed to get net count: {e}")
                net_count = 0
            
            if net_count == 0:
                debug_print("⚠️ No nets found in board!")
                return {
                    'bounds': {'width_nm': 100000000, 'height_nm': 100000000, 'layers': layer_count},
                    'nets': [],
                    'design_rules': {'min_track_width_nm': 200000, 'min_clearance_nm': 200000, 'min_via_size_nm': 400000}
                }
            
            # STEP 1: Extract ALL nets and collect ALL pins
            nets = []
            all_pin_coordinates = []  # Collect coordinates of ALL pins for bounds calculation
            nets_processed = 0
            nets_with_pins = 0
            
            debug_print("🔍 Collecting all net pins for bounds calculation...")
            
            for net_code in range(1, net_count):  # Skip net 0 (no net)
                try:
                    net_info = board.GetNetInfo().GetNetItem(net_code)
                    if not net_info:
                        continue
                        
                    net_name = net_info.GetNetname()
                    if not net_name or net_name.startswith("unconnected-"):
                        continue  # Skip unconnected/unnamed nets
                    
                    pins = self._extract_pins_for_net(board, net_code, debug_print)
                    nets_processed += 1
                    
                    if len(pins) >= 2:  # Only include nets with 2+ pins
                        nets_with_pins += 1
                        
                        # Check if we should skip this net due to fills/pours
                        should_skip = False
                        
                        if config and config.get('skip_filled_nets', True):
                            # Skip common power/ground nets that usually have fills
                            common_filled_nets = ['GND', 'GNDA', 'GNDD', 'VCC', 'VDD', 'VSS', '+5V', '+3V3', '+3.3V', '-5V', '+12V', '-12V']
                            
                            for common_net in common_filled_nets:
                                if common_net.lower() in net_name.lower():
                                    debug_print(f"⚠️ Skipping net '{net_name}' - likely has copper pour/fill")
                                    should_skip = True
                                    break
                        
                        if not should_skip:
                            # Convert pin dictionaries to the format expected by routing engine
                            formatted_pins = []
                            for pin in pins:
                                formatted_pins.append({
                                    'x': pin['x'],  # Keep in nanometers
                                    'y': pin['y'],  # Keep in nanometers
                                    'layer': pin['layer']
                                })
                                # Add to coordinate collection for bounds calculation
                                all_pin_coordinates.append((pin['x'], pin['y']))
                            
                            nets.append({
                                'id': net_code,
                                'name': net_name,
                                'pins': formatted_pins,
                                'width_nm': 200000  # Default 0.2mm trace width
                            })
                            debug_print(f"✅ Net {net_code}: '{net_name}' ({len(pins)} pins)")
                        else:
                            debug_print(f"⏭️ Net {net_code}: '{net_name}' skipped (likely filled)")
                        
                        # Only show pin coordinates for first 2 nets for debugging
                        if net_code <= 2:
                            for i, pin in enumerate(formatted_pins[:2]):
                                debug_print(f"   Pin {i+1}: ({pin['x']/1e6:.2f}, {pin['y']/1e6:.2f}) mm, layer {pin['layer']}")
                            
                    else:
                        debug_print(f"⏭️ Net {net_code}: '{net_name}' skipped ({len(pins)} pins)")
                        
                except Exception as e:
                    debug_print(f"❌ Error processing net {net_code}: {e}")
                    continue
            
            # STEP 2: Calculate bounds from actual routing pins
            if not all_pin_coordinates:
                debug_print("⚠️ No valid pins found for routing!")
                return {
                    'bounds': {'width_nm': 100000000, 'height_nm': 100000000, 'layers': layer_count},
                    'nets': [],
                    'design_rules': {'min_track_width_nm': 200000, 'min_clearance_nm': 200000, 'min_via_size_nm': 400000}
                }
            
            # Calculate bounds from actual routing pins
            min_x = min(coord[0] for coord in all_pin_coordinates)
            min_y = min(coord[1] for coord in all_pin_coordinates)
            max_x = max(coord[0] for coord in all_pin_coordinates)
            max_y = max(coord[1] for coord in all_pin_coordinates)
            
            width_nm = int(max_x - min_x)
            height_nm = int(max_y - min_y)
            
            debug_print(f"📏 Board size from routing pins: {width_nm/1e6:.1f}mm x {height_nm/1e6:.1f}mm")
            debug_print(f"📍 Pin coordinate range: X({min_x/1e6:.1f} to {max_x/1e6:.1f}mm), Y({min_y/1e6:.1f} to {max_y/1e6:.1f}mm)")
            debug_print(f"📊 Found {len(all_pin_coordinates)} total pins in {len(nets)} valid nets")
            
            debug_print(f"📊 Export summary:")
            debug_print(f"   - Nets processed: {nets_processed}")
            debug_print(f"   - Nets with 2+ pins: {nets_with_pins}")
            debug_print(f"   - Nets ready for routing: {len(nets)}")
            
        except Exception as e:
            debug_print(f"❌ Critical error in board export: {e}")
            import traceback
            debug_print(traceback.format_exc())
            raise
        
        return {
            'bounds': {
                'width_nm': width_nm,
                'height_nm': height_nm,
                'layers': layer_count,
                'min_x_nm': min_x,
                'min_y_nm': min_y,
                'max_x_nm': max_x,
                'max_y_nm': max_y
            },
            'nets': nets,
            'design_rules': {
                'min_track_width_nm': 200000,
                'min_clearance_nm': 200000,
                'min_via_size_nm': 400000
            }
        }
    
    def _extract_pins_for_net(self, board, net_code, debug_print=None):
        """Extract pins for a specific net, excluding those already connected by filled zones"""
        if debug_print is None:
            debug_print = print
            
        pins = []
        
        try:
            # Find pads connected to this net
            footprint_count = 0
            pad_count = 0
            
            for module in board.GetFootprints():
                footprint_count += 1
                
                try:
                    # Get pads using the most compatible method
                    if hasattr(module, 'Pads'):
                        pads = module.Pads()
                    else:
                        debug_print(f"❌ Footprint {module.GetReference()} has no Pads() method")
                        continue
                        
                    for pad in pads:
                        pad_count += 1
                        
                        try:
                            if pad.GetNetCode() == net_code:
                                pos = pad.GetPosition()
                                layer = pad.GetLayer()
                                pins.append({
                                    'x': int(pos.x),
                                    'y': int(pos.y),
                                    'layer': 0 if layer == pcbnew.F_Cu else 1,  # Simplified layer mapping
                                    'pad': pad  # Keep reference for zone checking
                                })
                                
                        except Exception as e:
                            debug_print(f"❌ Error processing pad in {module.GetReference()}: {e}")
                            continue
                            
                except Exception as e:
                    debug_print(f"❌ Error processing footprint {module.GetReference()}: {e}")
                    continue
            
            if net_code <= 5 or len(pins) > 0:  # Debug first few nets or any with pins
                if len(pins) > 0:
                    debug_print(f"   Net {net_code}: Found {len(pins)} pins (scanned {footprint_count} footprints)")
                elif net_code <= 3:  # Only show details for first 3 nets
                    debug_print(f"   Net {net_code}: Found {len(pins)} pins (scanned {footprint_count} footprints, {pad_count} pads)")
                
        except Exception as e:
            debug_print(f"❌ Critical error in pin extraction for net {net_code}: {e}")
            import traceback
            debug_print(traceback.format_exc())
            
        # Check if pins are already connected by filled zones
        if len(pins) >= 2:
            try:
                connected_groups = self._find_zone_connected_groups(board, pins, net_code)
                
                # If all pins are in one connected group, no routing needed
                if len(connected_groups) <= 1:
                    debug_print(f"Net {net_code}: All pins connected by filled zones, skipping routing")
                    return []  # Return empty list to skip this net
                
                # Return representative pins from each group
                result_pins = []
                for group in connected_groups:
                    result_pins.append(group[0])  # Take first pin from each group
                
                return result_pins
                
            except Exception as e:
                debug_print(f"❌ Error checking zone connections for net {net_code}: {e}")
                # Fall back to returning all pins if zone checking fails
                return pins
        
        return pins
    
    def _find_zone_connected_groups(self, board, pins, net_code):
        """Find groups of pins connected by filled zones"""
        # Get all filled zones for this net
        zones = []
        for zone in board.Zones():
            if zone.GetNetCode() == net_code and zone.IsFilled():
                zones.append(zone)
        
        if not zones:
            # No zones, each pin is its own group
            return [[pin] for pin in pins]
        
        # Group pins by zone connectivity
        groups = []
        ungrouped_pins = pins.copy()
        
        for zone in zones:
            zone_pins = []
            remaining_pins = []
            
            for pin in ungrouped_pins:
                # Check if pin is inside this zone's filled area
                pad_pos = pcbnew.VECTOR2I(pin['x'], pin['y'])
                # Use the appropriate layer (F_Cu or B_Cu based on pin layer)
                layer = pcbnew.F_Cu if pin['layer'] == 0 else pcbnew.B_Cu
                
                # Try different KiCad API methods for zone hit testing
                pin_in_zone = False
                try:
                    # Method 1: HitTestFilledArea (newer KiCad)
                    if hasattr(zone, 'HitTestFilledArea'):
                        pin_in_zone = zone.HitTestFilledArea(layer, pad_pos, 0)
                    # Method 2: HitTestInsideZone (older KiCad)  
                    elif hasattr(zone, 'HitTestInsideZone'):
                        pin_in_zone = zone.HitTestInsideZone(pad_pos)
                    # Method 3: GetBoundingBox fallback
                    else:
                        bbox = zone.GetBoundingBox()
                        pin_in_zone = bbox.Contains(pad_pos)
                        print(f"⚠️ Using fallback zone detection for pin at ({pin['x']}, {pin['y']})")
                        
                except Exception as e:
                    print(f"❌ Zone hit test failed for pin at ({pin['x']}, {pin['y']}): {e}")
                    # Fallback to bounding box
                    try:
                        bbox = zone.GetBoundingBox()
                        pin_in_zone = bbox.Contains(pad_pos)
                    except:
                        pin_in_zone = False
                
                if pin_in_zone:
                    zone_pins.append(pin)
                else:
                    remaining_pins.append(pin)
            
            if zone_pins:
                groups.append(zone_pins)
            ungrouped_pins = remaining_pins
        
        # Add any remaining ungrouped pins as individual groups
        for pin in ungrouped_pins:
            groups.append([pin])
        
        return groups
    
    def _import_routes(self, board, results):
        """Import routing results back to the board"""
        if not results.get('nets'):
            return
        
        # Create tracks for each routed net
        for net_result in results['nets']:
            net_id = net_result['id']
            path = net_result.get('path', [])
            
            if len(path) < 2:
                continue
            
            # Get net info
            try:
                # Try different methods to get net info
                if hasattr(board, 'GetNetlist'):
                    netlist = board.GetNetlist()
                    net_info = netlist.GetNetItem(net_id)
                else:
                    net_info = board.GetNetInfo().GetNetItem(net_id)
            except Exception as e:
                print(f"Error getting net info for net {net_id}: {e}")
                continue
                
            if not net_info:
                continue
            
            # Create track segments
            for i in range(len(path) - 1):
                start_point = path[i]
                end_point = path[i + 1]
                
                # Create track segment
                track = pcbnew.PCB_TRACK(board)
                track.SetStart(pcbnew.VECTOR2I(start_point['x'], start_point['y']))
                track.SetEnd(pcbnew.VECTOR2I(end_point['x'], end_point['y']))
                track.SetWidth(200000)  # 0.2mm
                track.SetLayer(pcbnew.F_Cu if start_point['layer'] == 0 else pcbnew.B_Cu)
                track.SetNetCode(net_id)
                
                board.Add(track)
                
                # Add via if layer changes
                if start_point['layer'] != end_point['layer']:
                    via = pcbnew.PCB_VIA(board)
                    via.SetPosition(pcbnew.VECTOR2I(start_point['x'], start_point['y']))
                    via.SetWidth(400000)  # 0.4mm via
                    via.SetDrill(200000)  # 0.2mm drill
                    via.SetNetCode(net_id)
                    board.Add(via)
        
        # Note: Removed pcbnew.Refresh() here as it can cause KiCad crashes
        # Display will be updated automatically when the dialog closes


class OrthoRouteConfigDialog(wx.Dialog):
    """Configuration dialog for OrthoRoute settings"""
    
    def __init__(self, parent, cpu_mode=False):
        super().__init__(parent, title="OrthoRoute GPU Autorouter Configuration",
                        size=(500, 650),  # Increased height from default to 650px
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.cpu_mode = cpu_mode
        self.config = {
            'grid_pitch_mm': 0.1,
            'max_iterations': 20,
            'enable_visualization': True,  # Enable by default for debugging
            'routing_algorithm': 'gpu_wavefront',  # Default to GPU wavefront (Lee's algorithm)
            'batch_size': 256,
            'via_cost': 10,
            'conflict_penalty': 50,
            'skip_filled_nets': True  # Skip nets that already have fills/pours
        }
        
        self._create_ui()
        self.SetSize((500, 650))  # Made taller to show all controls including GPU info
        self.CenterOnParent()
    
    def _create_ui(self):
        """Create the user interface"""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="OrthoRoute GPU Autorouter Settings")
        title_font = title.GetFont()
        title_font.SetPointSize(title_font.GetPointSize() + 2)
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Grid settings
        grid_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Grid Settings")
        
        # Grid pitch
        grid_pitch_sizer = wx.BoxSizer(wx.HORIZONTAL)
        grid_pitch_sizer.Add(wx.StaticText(panel, label="Grid Pitch (mm):"), 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.grid_pitch_spin = wx.SpinCtrlDouble(panel, value="0.1", min=0.05, max=1.0, inc=0.05)
        self.grid_pitch_spin.SetDigits(2)
        grid_pitch_sizer.Add(self.grid_pitch_spin, 0, wx.ALL, 5)
        grid_box.Add(grid_pitch_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(grid_box, 0, wx.EXPAND | wx.ALL, 10)
        
        # Routing settings
        routing_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Routing Settings")
        
        # Max iterations
        iter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        iter_sizer.Add(wx.StaticText(panel, label="Max Iterations:"), 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.max_iter_spin = wx.SpinCtrl(panel, value="20", min=1, max=100)
        iter_sizer.Add(self.max_iter_spin, 0, wx.ALL, 5)
        routing_box.Add(iter_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Batch size
        batch_sizer = wx.BoxSizer(wx.HORIZONTAL)
        batch_sizer.Add(wx.StaticText(panel, label="Batch Size:"), 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.batch_size_spin = wx.SpinCtrl(panel, value="256", min=64, max=2048)
        batch_sizer.Add(self.batch_size_spin, 0, wx.ALL, 5)
        routing_box.Add(batch_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Via cost
        via_sizer = wx.BoxSizer(wx.HORIZONTAL)
        via_sizer.Add(wx.StaticText(panel, label="Via Cost:"), 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.via_cost_spin = wx.SpinCtrl(panel, value="10", min=1, max=100)
        via_sizer.Add(self.via_cost_spin, 0, wx.ALL, 5)
        routing_box.Add(via_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Routing algorithm choice
        algo_sizer = wx.BoxSizer(wx.HORIZONTAL)
        algo_sizer.Add(wx.StaticText(panel, label="Routing Algorithm:"), 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.algo_choice = wx.Choice(panel, choices=[
            "GPU Wavefront (Lee's Algorithm)",
            "Grid-Based Routing", 
            "CPU Fallback"
        ])
        self.algo_choice.SetSelection(0)  # Default to GPU Wavefront
        algo_sizer.Add(self.algo_choice, 0, wx.ALL, 5)
        routing_box.Add(algo_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Net filtering options
        filter_sizer = wx.BoxSizer(wx.VERTICAL)
        self.skip_filled_cb = wx.CheckBox(panel, label="Skip nets with existing fills/pours (recommended)")
        self.skip_filled_cb.SetValue(True)  # Enabled by default
        filter_sizer.Add(self.skip_filled_cb, 0, wx.ALL, 5)
        routing_box.Add(filter_sizer, 0, wx.EXPAND | wx.ALL, 5)
        routing_box.Add(via_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(routing_box, 0, wx.EXPAND | wx.ALL, 10)
        
        # Visualization settings
        viz_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Visualization")
        
        self.enable_viz_cb = wx.CheckBox(panel, label="Enable real-time visualization")
        viz_box.Add(self.enable_viz_cb, 0, wx.ALL, 5)
        
        self.debug_mode_cb = wx.CheckBox(panel, label="Debug Mode (show debug output)")
        viz_box.Add(self.debug_mode_cb, 0, wx.ALL, 5)
        
        sizer.Add(viz_box, 0, wx.EXPAND | wx.ALL, 10)
        
        # GPU info
        gpu_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "GPU Information")
        gpu_info = self._get_gpu_info()
        gpu_text = wx.StaticText(panel, label=gpu_info)
        gpu_box.Add(gpu_text, 0, wx.ALL, 5)
        sizer.Add(gpu_box, 1, wx.EXPAND | wx.ALL, 10)
        
        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(panel, wx.ID_OK, "Start Routing")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.AddButton(ok_btn)
        btn_sizer.AddButton(cancel_btn)
        btn_sizer.Realize()
        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        panel.SetSizer(sizer)
    
    def _get_gpu_info(self) -> str:
        """Get GPU information for display"""
        if self.cpu_mode:
            return "ℹ Running in CPU-only mode\n  CuPy not available for GPU acceleration"
        
        try:
            import cupy as cp
            
            # Get device
            try:
                device = cp.cuda.Device()
            except Exception as e:
                return f"✗ GPU Device Error: {str(e)}"
            
            # Get memory info
            try:
                mem_info = device.mem_info
                if callable(mem_info):
                    mem_info = mem_info()
                    
                if isinstance(mem_info, (list, tuple)) and len(mem_info) >= 2:
                    total_mem = float(mem_info[1]) / (1024**3)
                else:
                    total_mem = 0.0
            except Exception as e:
                print(f"Memory info error: {e}")
                total_mem = 0.0
            
            # Get device name with multiple fallbacks
            device_name = "Unknown GPU"
            try:
                # Method 1: Use getDeviceProperties (most reliable)
                device_props = cp.cuda.runtime.getDeviceProperties(device.id)
                if 'name' in device_props:
                    name_bytes = device_props['name']
                    if isinstance(name_bytes, bytes):
                        device_name = name_bytes.decode('utf-8')
                    else:
                        device_name = str(name_bytes)
                else:
                    device_name = f"CUDA Device {device.id}"
                    
            except Exception as e:
                print(f"Device properties error: {e}")
                try:
                    # Method 2: Fallback to device ID
                    device_name = f"CUDA Device {device.id}"
                except Exception as e2:
                    print(f"Device ID error: {e2}")
                    device_name = "Unknown GPU Device"
            
            # Format the result
            if total_mem > 0:
                return f"✓ GPU Ready: {device_name}\n  Memory: {total_mem:.1f} GB"
            else:
                return f"✓ GPU Ready: {device_name}\n  Memory: Available"
                
        except ImportError:
            return "✗ CuPy not available\n  Install CuPy for GPU acceleration"
        except Exception as e:
            import traceback
            print(f"GPU info error: {e}")
            traceback.print_exc()
            return f"✗ GPU Error: {str(e)}"
    
    def get_config(self) -> Dict:
        """Get the current configuration"""
        # Map algorithm choice to internal value
        algo_map = {
            0: 'gpu_wavefront',    # GPU Wavefront (Lee's Algorithm)
            1: 'grid_based',       # Grid-Based Routing
            2: 'cpu_fallback'      # CPU Fallback
        }
        
        return {
            'grid_pitch_mm': self.grid_pitch_spin.GetValue(),
            'max_iterations': self.max_iter_spin.GetValue(),
            'batch_size': self.batch_size_spin.GetValue(),
            'via_cost': self.via_cost_spin.GetValue(),
            'enable_visualization': self.enable_viz_cb.GetValue(),
            'routing_algorithm': algo_map.get(self.algo_choice.GetSelection(), 'gpu_wavefront'),
            'skip_filled_nets': self.skip_filled_cb.GetValue(),
            'debug_mode': self.debug_mode_cb.GetValue()
        }
    
    def _get_board_bounds(self, board):
        """Get board bounding box for visualization"""
        try:
            bbox = board.GetBoardEdgesBoundingBox()
            return [
                float(bbox.GetX()) / 1e6,  # Convert to mm
                float(bbox.GetY()) / 1e6,
                float(bbox.GetWidth()) / 1e6,
                float(bbox.GetHeight()) / 1e6
            ]
        except:
            return [0, 0, 100, 80]  # Default board size
    
    def _get_board_pads(self, board):
        """Get pad information for visualization"""
        pads = []
        try:
            footprint_count = 0
            pad_count = 0
            
            for footprint in board.GetFootprints():
                footprint_count += 1
                footprint_pads = 0
                
                for pad in footprint.Pads():
                    try:
                        bbox = pad.GetBoundingBox()
                        pos = pad.GetPosition()
                        
                        # Extract actual pad shape and size information
                        pad_shape = "unknown"
                        try:
                            # Get pad shape - KiCad PAD_SHAPE constants
                            shape_id = pad.GetShape()
                            shape_map = {
                                0: "circle",      # PAD_SHAPE::CIRCLE  
                                1: "rect",        # PAD_SHAPE::RECT
                                2: "oval",        # PAD_SHAPE::OVAL
                                3: "trapezoid",   # PAD_SHAPE::TRAPEZOID
                                4: "roundrect",   # PAD_SHAPE::ROUNDRECT
                                5: "chamfered_rect", # PAD_SHAPE::CHAMFERED_RECT
                                6: "custom"       # PAD_SHAPE::CUSTOM
                            }
                            pad_shape = shape_map.get(shape_id, f"shape_{shape_id}")
                        except:
                            pad_shape = "rect"  # Default fallback
                        
                        # Get actual pad size (not bounding box)
                        try:
                            pad_size = pad.GetSize()
                            actual_width = float(pad_size.x) / 1e6  # Convert nm to mm
                            actual_height = float(pad_size.y) / 1e6
                        except:
                            # Fallback to bounding box
                            actual_width = float(bbox.GetWidth()) / 1e6
                            actual_height = float(bbox.GetHeight()) / 1e6
                        
                        # Get drill size if it's a through-hole pad
                        drill_size = 0.0
                        try:
                            drill = pad.GetDrillSize()
                            if drill.x > 0:
                                drill_size = float(drill.x) / 1e6  # Convert to mm
                        except:
                            pass
                        
                        pad_data = {
                            'bounds': [
                                float(bbox.GetX()) / 1e6,  # Convert to mm
                                float(bbox.GetY()) / 1e6,
                                float(bbox.GetWidth()) / 1e6,
                                float(bbox.GetHeight()) / 1e6
                            ],
                            'center': [
                                float(pos.x) / 1e6,  # Pad center in mm
                                float(pos.y) / 1e6
                            ],
                            'actual_size': [actual_width, actual_height],  # Real pad dimensions
                            'shape': pad_shape,  # Actual pad shape
                            'drill_size': drill_size,  # Drill diameter for TH pads
                            'net': pad.GetNetname(),
                            'ref': footprint.GetReference(),
                            'pad_name': pad.GetName() if hasattr(pad, 'GetName') else str(pad_count)
                        }
                        pads.append(pad_data)
                        pad_count += 1
                        footprint_pads += 1
                        
                    except Exception as e:
                        print(f"Error processing pad in {footprint.GetReference()}: {e}")
                
                # Debug for first few footprints
                if footprint_count <= 3:
                    print(f"   Footprint {footprint.GetReference()}: {footprint_pads} pads")
                        
            print(f"📍 Pad extraction: {footprint_count} footprints, {pad_count} total pads")
            
        except Exception as e:
            print(f"Error getting pads: {e}")
            
        return pads
    
    def _get_board_obstacles(self, board):
        """Get obstacle information for visualization"""
        obstacles = []
        try:
            # Get existing tracks as obstacles
            for track in board.GetTracks():
                if hasattr(track, 'GetBoundingBox'):
                    bbox = track.GetBoundingBox()
                    obstacles.append({
                        'bounds': [
                            float(bbox.GetX()) / 1e6,
                            float(bbox.GetY()) / 1e6,
                            float(bbox.GetWidth()) / 1e6,
                            float(bbox.GetHeight()) / 1e6
                        ],
                        'type': 'track'
                    })
        except Exception as e:
            print(f"Error getting obstacles: {e}")
        return obstacles


# Register the plugin
OrthoRouteKiCadPlugin().register()
