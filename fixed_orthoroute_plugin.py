"""
OrthoRoute GPU Autorouter - FIXED VERSION
This version includes the missing track creation functionality.
"""

import pcbnew
import wx
import traceback
from .orthoroute_engine import OrthoRouteEngine

class OrthoRouteConfigDialog(wx.Dialog):
    """Configuration dialog for OrthoRoute"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute GPU Autorouter", size=(500, 400))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="OrthoRoute GPU Autorouter")
        title_font = title.GetFont()
        title_font.SetPointSize(14)
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Parameters
        params_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Routing Parameters")
        
        # Grid pitch
        grid_pitch_sizer = wx.BoxSizer(wx.HORIZONTAL)
        grid_pitch_sizer.Add(wx.StaticText(panel, label="Grid Pitch (mm):"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.grid_pitch_ctrl = wx.SpinCtrlDouble(panel, value=0.1, min=0.05, max=1.0, inc=0.05)
        grid_pitch_sizer.Add(self.grid_pitch_ctrl, 1, wx.ALL, 5)
        params_box.Add(grid_pitch_sizer, 0, wx.ALL | wx.EXPAND, 5)
        
        # Max iterations
        iter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        iter_sizer.Add(wx.StaticText(panel, label="Max Iterations:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.max_iter_ctrl = wx.SpinCtrl(panel, value=3, min=1, max=10)
        iter_sizer.Add(self.max_iter_ctrl, 1, wx.ALL, 5)
        params_box.Add(iter_sizer, 0, wx.ALL | wx.EXPAND, 5)
        
        # Via cost
        via_sizer = wx.BoxSizer(wx.HORIZONTAL)
        via_sizer.Add(wx.StaticText(panel, label="Via Cost:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.via_cost_ctrl = wx.SpinCtrl(panel, value=10, min=1, max=100)
        via_sizer.Add(self.via_cost_ctrl, 1, wx.ALL, 5)
        params_box.Add(via_sizer, 0, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(params_box, 0, wx.ALL | wx.EXPAND, 10)
        
        # Options
        options_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Options")
        
        self.progress_cb = wx.CheckBox(panel, label="Show progress dialog")
        self.progress_cb.SetValue(True)
        options_box.Add(self.progress_cb, 0, wx.ALL, 5)
        
        self.debug_cb = wx.CheckBox(panel, label="Debug output to console")
        self.debug_cb.SetValue(False)
        options_box.Add(self.debug_cb, 0, wx.ALL, 5)
        
        sizer.Add(options_box, 0, wx.ALL | wx.EXPAND, 10)
        
        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.start_btn = wx.Button(panel, wx.ID_OK, "Start Routing")
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        
        button_sizer.Add(self.start_btn, 0, wx.ALL, 5)
        button_sizer.Add(self.cancel_btn, 0, wx.ALL, 5)
        
        sizer.Add(button_sizer, 0, wx.ALL | wx.CENTER, 10)
        
        panel.SetSizer(sizer)
        self.Centre()


class OrthoRouteProgressDialog(wx.Dialog):
    """Progress dialog for routing operations"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Progress", size=(400, 200))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.status_text = wx.StaticText(panel, label="Initializing...")
        sizer.Add(self.status_text, 0, wx.ALL | wx.EXPAND, 10)
        
        self.progress_gauge = wx.Gauge(panel, range=100)
        sizer.Add(self.progress_gauge, 0, wx.ALL | wx.EXPAND, 10)
        
        self.net_text = wx.StaticText(panel, label="")
        sizer.Add(self.net_text, 0, wx.ALL | wx.EXPAND, 10)
        
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        sizer.Add(self.cancel_btn, 0, wx.ALL | wx.CENTER, 10)
        
        panel.SetSizer(sizer)
        self.Centre()
        
        self.should_cancel = False
        self.Bind(wx.EVT_BUTTON, self.on_cancel, self.cancel_btn)
    
    def on_cancel(self, event):
        self.should_cancel = True
        self.status_text.SetLabel("Cancelling...")
        
    def update_progress(self, data):
        if not self.should_cancel:
            progress = data.get('progress', 0)
            current_net = data.get('current_net', '')
            stage = data.get('stage', '')
            
            self.progress_gauge.SetValue(int(progress))
            self.net_text.SetLabel(f"Net: {current_net}")
            self.status_text.SetLabel(f"Stage: {stage}")
            
            wx.GetApp().Yield()


class OrthoRoutePlugin(pcbnew.ActionPlugin):
    """OrthoRoute GPU Autorouter Plugin"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter"
        self.category = "Routing"
        self.description = "GPU-accelerated PCB autorouter using Lee's algorithm"
        self.show_toolbar_button = True
        self.icon_file_name = "icon.png"
        
    def Run(self):
        """Main plugin entry point"""
        try:
            print("🚀 OrthoRoute GPU Autorouter starting...")
            
            # Get the current board
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board loaded!", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Show configuration dialog
            config_dialog = OrthoRouteConfigDialog(None)
            if config_dialog.ShowModal() != wx.ID_OK:
                config_dialog.Destroy()
                return
            
            # Get configuration
            config = {
                'grid_pitch_mm': config_dialog.grid_pitch_ctrl.GetValue(),
                'max_iterations': config_dialog.max_iter_ctrl.GetValue(),
                'via_cost': config_dialog.via_cost_ctrl.GetValue(),
                'show_progress': config_dialog.progress_cb.GetValue(),
                'debug_output': config_dialog.debug_cb.GetValue()
            }
            
            config_dialog.Destroy()
            
            # Create progress dialog if requested
            progress_dialog = None
            if config['show_progress']:
                progress_dialog = OrthoRouteProgressDialog(None)
                progress_dialog.Show()
            
            # Extract board data
            print("📊 Extracting board data...")
            board_data = self.extract_board_data(board)
            
            if not board_data['nets']:
                message = "No nets found to route!"
                print(f"❌ {message}")
                if progress_dialog:
                    progress_dialog.Destroy()
                wx.MessageBox(message, "No Work", wx.OK | wx.ICON_WARNING)
                return
            
            print(f"✅ Found {len(board_data['nets'])} nets to route")
            
            # Create routing engine
            engine = OrthoRouteEngine()
            if config['debug_output']:
                engine.debug_print = print
            
            # Set up progress callback
            cancel_callback = None
            progress_callback = None
            
            if progress_dialog:
                cancel_callback = lambda: progress_dialog.should_cancel
                progress_callback = progress_dialog.update_progress
            
            routing_config = {
                **config,
                'progress_callback': progress_callback,
                'should_cancel': cancel_callback or (lambda: False)
            }
            
            # Start routing WITH BOARD REFERENCE (THIS IS THE FIX!)
            print("🔄 Starting routing process...")
            results = engine.route(board_data, routing_config, board=board)
            
            # Close progress dialog
            if progress_dialog:
                progress_dialog.Destroy()
            
            # Show results
            if results['success']:
                stats = results['stats']
                tracks_created = len(results.get('tracks', []))
                
                message = (f"Routing completed!\n\n"
                          f"Nets processed: {stats['total_nets']}\n"
                          f"Successfully routed: {stats['successful_nets']}\n"
                          f"Success rate: {stats['success_rate']:.1f}%\n"
                          f"Tracks created: {tracks_created}\n"
                          f"Time: {stats['total_time_seconds']:.2f} seconds")
                
                print(f"✅ {message}")
                wx.MessageBox(message, "Routing Complete", wx.OK | wx.ICON_INFORMATION)
                
                # Refresh the display
                pcbnew.Refresh()
                
            else:
                error_msg = results.get('error', 'Unknown error')
                print(f"❌ Routing failed: {error_msg}")
                wx.MessageBox(f"Routing failed: {error_msg}", "Error", wx.OK | wx.ICON_ERROR)
                
        except Exception as e:
            error_msg = f"Plugin error: {str(e)}\n\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            wx.MessageBox(error_msg, "Plugin Error", wx.OK | wx.ICON_ERROR)
    
    def extract_board_data(self, board):
        """Extract board data for routing"""
        
        # Get board bounds
        bounds = board.GetBoardEdgesBoundingBox()
        width_nm = bounds.GetWidth()
        height_nm = bounds.GetHeight()
        
        min_x_nm = bounds.GetX()
        min_y_nm = bounds.GetY()
        max_x_nm = min_x_nm + width_nm
        max_y_nm = min_y_nm + height_nm
        
        # Get layer count
        layer_count = board.GetCopperLayerCount()
        
        print(f"📐 Board: {width_nm/1e6:.1f}mm x {height_nm/1e6:.1f}mm, {layer_count} layers")
        
        # Extract nets
        nets_data = []
        netcodes = board.GetNetsByNetcode()
        
        for netcode, kicad_net in netcodes.items():
            if netcode == 0:  # Skip unconnected net
                continue
                
            net_name = kicad_net.GetNetname()
            if not net_name or net_name.startswith('/'):  # Skip power nets for now
                continue
            
            # Get all pads for this net
            pins = []
            for footprint in board.GetFootprints():
                for pad in footprint.Pads():
                    if pad.GetNet() == kicad_net:
                        pos = pad.GetPosition()
                        layer = pad.GetLayer()
                        
                        # Convert layer to internal layer number
                        if layer == pcbnew.F_Cu:
                            internal_layer = 0
                        elif layer == pcbnew.B_Cu:
                            internal_layer = 1
                        else:
                            internal_layer = min(layer - pcbnew.In1_Cu + 2, layer_count - 1)
                        
                        pins.append({
                            'x': int(pos.x),
                            'y': int(pos.y), 
                            'layer': internal_layer
                        })
            
            # Only include nets with 2+ pins
            if len(pins) >= 2:
                nets_data.append({
                    'id': netcode,
                    'name': net_name,
                    'pins': pins,
                    'kicad_net': kicad_net,  # Store KiCad net reference
                    'width_nm': 200000  # Default 0.2mm track width
                })
        
        board_data = {
            'bounds': {
                'width_nm': width_nm,
                'height_nm': height_nm,
                'min_x_nm': min_x_nm,
                'min_y_nm': min_y_nm,
                'max_x_nm': max_x_nm,
                'max_y_nm': max_y_nm,
                'layers': layer_count
            },
            'nets': nets_data,
            'obstacles': {}  # TODO: Extract existing tracks/vias
        }
        
        return board_data


# Register the plugin
OrthoRoutePlugin().register()
