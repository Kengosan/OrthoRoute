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
            # Check CuPy availability
            if not self._check_cupy_available():
                self._show_cupy_install_dialog()
                return
            
            # Get current board
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board found. Please open a PCB first.", 
                            "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Show configuration dialog
            dlg = OrthoRouteConfigDialog(None)
            if dlg.ShowModal() == wx.ID_OK:
                config = dlg.get_config()
                dlg.Destroy()
                
                # Route the board
                self._route_board_gpu(board, config)
            else:
                dlg.Destroy()
                
        except Exception as e:
            print(f"❌ Critical error in plugin Run method: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            wx.MessageBox(f"Plugin error: {str(e)}\n\nCheck console for detailed traceback.", 
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
    
    def _route_board_gpu(self, board, config):
        """Route the board using GPU acceleration"""
        try:
            print("🔍 Starting _route_board_gpu method...")
            
            # Test imports first to identify any import issues
            try:
                print("🔍 Testing CuPy import...")
                import cupy as cp
                print("✅ CuPy imported successfully")
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
        
        # Refresh display
        pcbnew.Refresh()


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
            'skip_filled_nets': self.skip_filled_cb.GetValue()
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
