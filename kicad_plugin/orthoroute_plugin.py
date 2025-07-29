import os
import sys
import pcbnew
import wx

# Add the parent directory to the Python path to find the orthoroute package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ConfigDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute GPU Autorouter", 
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self._create_ui()
        self.SetSize((400, 300))
        self.CenterOnParent()
    
    def _get_default_config(self):
        return {
            "via_spacing": 1.0,
            "track_spacing": 0.25,
            "max_vias": 10,
        }
    
    def _create_ui(self):
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="OrthoRoute GPU Autorouter")
        title_font = title.GetFont()
        title_font.SetPointSize(12)
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Settings
        settings = wx.BoxSizer(wx.VERTICAL)
        
        # Via spacing
        via_box = wx.BoxSizer(wx.HORIZONTAL)
        via_label = wx.StaticText(panel, label="Via Spacing (mm):")
        self.via_spacing = wx.SpinCtrlDouble(panel, value="1.0", min=0.1, max=10.0, inc=0.1)
        via_box.Add(via_label, 0, wx.ALL | wx.CENTER, 5)
        via_box.Add(self.via_spacing, 1, wx.ALL | wx.EXPAND, 5)
        settings.Add(via_box, 0, wx.ALL | wx.EXPAND, 5)
        
        # Track spacing
        track_box = wx.BoxSizer(wx.HORIZONTAL)
        track_label = wx.StaticText(panel, label="Track Spacing (mm):")
        self.track_spacing = wx.SpinCtrlDouble(panel, value="0.25", min=0.1, max=5.0, inc=0.05)
        track_box.Add(track_label, 0, wx.ALL | wx.CENTER, 5)
        track_box.Add(self.track_spacing, 1, wx.ALL | wx.EXPAND, 5)
        settings.Add(track_box, 0, wx.ALL | wx.EXPAND, 5)
        
        # Max vias
        vias_box = wx.BoxSizer(wx.HORIZONTAL)
        vias_label = wx.StaticText(panel, label="Maximum Vias:")
        self.max_vias = wx.SpinCtrl(panel, value="10", min=0, max=1000)
        vias_box.Add(vias_label, 0, wx.ALL | wx.CENTER, 5)
        vias_box.Add(self.max_vias, 1, wx.ALL | wx.EXPAND, 5)
        settings.Add(vias_box, 0, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(settings, 1, wx.ALL | wx.EXPAND, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_btn = wx.Button(panel, wx.ID_OK, "OK")
        btn_sizer.Add(ok_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.CENTER, 10)
        
        panel.SetSizer(sizer)

class OrthoRoute(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter"
        self.category = "PCB"
        self.description = "GPU-accelerated PCB autorouter"
        self.show_toolbar_button = True
        self.engine = None
    
    def init_engine(self):
        """Initialize the GPU routing engine if not already initialized."""
        if self.engine is None:
            try:
                from orthoroute.gpu_engine import OrthoRouteEngine
                self.engine = OrthoRouteEngine()
            except ImportError as e:
                import wx
                wx.MessageBox(f"Error loading OrthoRoute engine: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
                return False
            return True
    
    def get_board_data(self, board):
        """Extract relevant data from the KiCad board."""
        netlist = {}
        tracks = []
        vias = []
        
        # Get board bounds
        bbox = board.GetBoardEdgesBoundingBox()
        board_width = bbox.GetWidth()
        board_height = bbox.GetHeight()
        
        # Process nets and pads
        for pad in board.GetPads():
            net_name = pad.GetNetname()
            if net_name:
                if net_name not in netlist:
                    netlist[net_name] = []
                netlist[net_name].append({
                    'x': pad.GetPosition().x,
                    'y': pad.GetPosition().y,
                    'layer': pad.GetLayer()
                })
        
        return {
            'netlist': netlist,
            'width': board_width,
            'height': board_height,
            'tracks': tracks,
            'vias': vias
        }
    
    def apply_routes(self, board, routes):
        """Apply the computed routes to the board."""
        for route in routes:
            # Create tracks
            for track in route['tracks']:
                new_track = pcbnew.PCB_TRACK(board)
                new_track.SetStart(pcbnew.wxPoint(track['start'][0], track['start'][1]))
                new_track.SetEnd(pcbnew.wxPoint(track['end'][0], track['end'][1]))
                new_track.SetLayer(track['layer'])
                new_track.SetWidth(track['width'])
                board.Add(new_track)
            
            # Create vias
            for via in route['vias']:
                new_via = pcbnew.PCB_VIA(board)
                new_via.SetPosition(pcbnew.wxPoint(via['x'], via['y']))
                new_via.SetDrill(via['drill'])
                new_via.SetWidth(via['diameter'])
                board.Add(new_via)
    
    def Run(self):
        try:
            # Get current board
            board = pcbnew.GetBoard()
            
            # Show configuration dialog
            dlg = ConfigDialog(None)
            if dlg.ShowModal() != wx.ID_OK:
                dlg.Destroy()
                return
            
            # Get configuration
            config = {
                'via_spacing': dlg.via_spacing.GetValue(),
                'track_spacing': dlg.track_spacing.GetValue(),
                'max_vias': dlg.max_vias.GetValue()
            }
            dlg.Destroy()
            
            # Initialize GPU engine
            if not self.init_engine():
                return
            
            # Extract board data
            board_data = self.get_board_data(board)
            
            # Run autorouting
            progress_dlg = wx.ProgressDialog("OrthoRoute GPU Autorouter",
                                          "Routing in progress...",
                                          maximum=100,
                                          parent=None,
                                          style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE)
            
            try:
                routes = self.engine.route(board_data, config)
                self.apply_routes(board, routes)
                board.RefreshRatsnest()
                pcbnew.Refresh()
                wx.MessageBox("Routing completed successfully!", "OrthoRoute GPU Autorouter", wx.OK)
            except Exception as e:
                wx.MessageBox(f"Routing error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
            finally:
                progress_dlg.Destroy()
                
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

# The plugin is registered in __init__.py
