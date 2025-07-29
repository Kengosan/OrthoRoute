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
            wx.MessageBox(f"Plugin error: {str(e)}", 
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
            # Import here to avoid early import errors
            import cupy as cp
            
            # Show progress dialog
            progress_dlg = wx.ProgressDialog(
                "OrthoRoute GPU Autorouter",
                "Initializing GPU routing engine...",
                maximum=100,
                style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT
            )
            
            try:
                # Export board data
                progress_dlg.Update(10, "Exporting board data...")
                print("Exporting board data...")
                board_data = self._export_board_data(board)
                print(f"Exported {len(board_data.get('nets', []))} nets")
                
                if not board_data.get('nets'):
                    wx.MessageBox("No nets found to route. Please ensure your PCB has nets with pads.", 
                                "No Nets Found", wx.OK | wx.ICON_WARNING)
                    return
                
                # Initialize GPU engine
                progress_dlg.Update(20, "Initializing routing engine...")
                print("Initializing routing engine...")
                from .orthoroute_engine import OrthoRouteEngine
                engine = OrthoRouteEngine()
                
                # Enable visualization if requested
                if config.get('enable_visualization', False):
                    progress_dlg.Update(30, "Setting up visualization...")
                    print("Setting up visualization...")
                    engine.enable_visualization({
                        'real_time': True,
                        'show_progress': True
                    })
                
                # Route the board
                progress_dlg.Update(40, "Starting routing...")
                print("Starting routing...")
                results = engine.route(board_data, config)
                print(f"Routing completed with success={results.get('success', False)}")
                
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
            wx.MessageBox(f"Import error: {str(e)}\n\nPlease ensure CuPy is installed for GPU acceleration.", 
                        "Import Error", wx.OK | wx.ICON_ERROR)
        except Exception as e:
            wx.MessageBox(f"Routing error: {str(e)}", 
                        "Routing Error", wx.OK | wx.ICON_ERROR)
    
    def _export_board_data(self, board) -> Dict:
        """Export board data for routing"""
        # Get board bounds
        bbox = board.GetBoundingBox()
        width_nm = int(bbox.GetWidth())
        height_nm = int(bbox.GetHeight())
        
        # Get layer count
        layer_count = board.GetCopperLayerCount()
        
        # Extract nets
        nets = []
        
        # Try different methods to get nets based on KiCad version
        try:
            # Method 1: Try GetNetlist() (older KiCad)
            if hasattr(board, 'GetNetlist'):
                netlist = board.GetNetlist()
                net_count = netlist.GetNetCount()
            else:
                # Method 2: Use GetNetInfo() (newer KiCad)
                netlist = None
                net_count = board.GetNetCount()
        except Exception as e:
            print(f"Error accessing netlist: {e}")
            # Method 3: Direct iteration through nets
            netlist = None
            net_count = board.GetNetCount()
        
        print(f"Found {net_count} nets in board")
        
        # Extract nets using appropriate method
        if netlist:
            # Use netlist object
            for net_code in range(1, net_count):  # Skip net 0 (no net)
                net_info = netlist.GetNetItem(net_code)
                if not net_info:
                    continue
                    
                net_name = net_info.GetNetname()
                pins = self._extract_pins_for_net(board, net_code)
                
                if len(pins) >= 2:  # Only include nets with 2+ pins
                    nets.append({
                        'id': net_code,
                        'name': net_name,
                        'pins': pins,
                        'width_nm': 200000  # Default 0.2mm trace width
                    })
        else:
            # Direct method using board nets
            for net_code in range(1, net_count):
                net_info = board.GetNetInfo().GetNetItem(net_code)
                if not net_info:
                    continue
                    
                net_name = net_info.GetNetname()
                pins = self._extract_pins_for_net(board, net_code)
                
                if len(pins) >= 2:  # Only include nets with 2+ pins
                    nets.append({
                        'id': net_code,
                        'name': net_name,
                        'pins': pins,
                        'width_nm': 200000  # Default 0.2mm trace width
                    })
        
        return {
            'bounds': {
                'width_nm': width_nm,
                'height_nm': height_nm,
                'layers': layer_count
            },
            'nets': nets,
            'design_rules': {
                'min_track_width_nm': 200000,
                'min_clearance_nm': 200000,
                'min_via_size_nm': 400000
            }
        }
    
    def _extract_pins_for_net(self, board, net_code):
        """Extract pins for a specific net, excluding those already connected by filled zones"""
        pins = []
        
        # Find pads connected to this net
        for module in board.GetFootprints():
            for pad in module.Pads():
                if pad.GetNetCode() == net_code:
                    pos = pad.GetPosition()
                    layer = pad.GetLayer()
                    pins.append({
                        'x': int(pos.x),
                        'y': int(pos.y),
                        'layer': 0 if layer == pcbnew.F_Cu else 1,  # Simplified layer mapping
                        'pad': pad  # Keep reference for zone checking
                    })
        
        # Check if pins are already connected by filled zones
        if len(pins) >= 2:
            connected_groups = self._find_zone_connected_groups(board, pins, net_code)
            
            # If all pins are in one connected group, no routing needed
            if len(connected_groups) <= 1:
                print(f"Net {net_code}: All pins connected by filled zones, skipping routing")
                return []  # Return empty list to skip this net
            
            # Return representative pins from each group
            representative_pins = []
            for group in connected_groups:
                # Take first pin from each group as representative
                pin = group[0]
                representative_pins.append({
                    'x': pin['x'],
                    'y': pin['y'], 
                    'layer': pin['layer']
                })
            pins = representative_pins
        
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
                if zone.HitTestFilledArea(layer, pad_pos, 0):
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
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.cpu_mode = cpu_mode
        self.config = {
            'grid_pitch_mm': 0.1,
            'max_iterations': 20,
            'enable_visualization': False,
            'batch_size': 256,
            'via_cost': 10,
            'conflict_penalty': 50
        }
        
        self._create_ui()
        self.SetSize((450, 500))  # Made taller to prevent button clipping
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
            device = cp.cuda.Device()
            mem_info = device.mem_info
            total_mem = mem_info[1] / (1024**3)
            return f"✓ GPU Ready: {device.name}\n  Memory: {total_mem:.1f} GB"
        except ImportError:
            return "✗ CuPy not available\n  Install CuPy for GPU acceleration"
        except Exception as e:
            return f"✗ GPU Error: {str(e)}"
    
    def get_config(self) -> Dict:
        """Get the current configuration"""
        return {
            'grid_pitch_mm': self.grid_pitch_spin.GetValue(),
            'max_iterations': self.max_iter_spin.GetValue(),
            'batch_size': self.batch_size_spin.GetValue(),
            'via_cost': self.via_cost_spin.GetValue(),
            'enable_visualization': self.enable_viz_cb.GetValue()
        }


# Register the plugin
OrthoRouteKiCadPlugin().register()
