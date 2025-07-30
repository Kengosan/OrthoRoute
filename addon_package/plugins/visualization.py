"""
OrthoRoute Visualization Module
Real-time visualization and progress tracking for routing operations.
"""

import wx
import time
import math
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass

@dataclass
class RoutingStats:
    """Real-time routing statistics"""
    nets_total: int = 0
    nets_completed: int = 0
    nets_failed: int = 0
    current_net: str = ""
    current_net_progress: float = 0.0
    total_segments: int = 0
    total_vias: int = 0
    total_length_mm: float = 0.0
    routing_time_seconds: float = 0.0
    grid_cells_processed: int = 0
    gpu_memory_used_mb: float = 0.0

class RoutingCanvas(wx.Panel):
    """Interactive PCB routing visualization canvas with zoom and pan"""
    
    def __init__(self, parent):
        super().__init__(parent, style=wx.BORDER_SUNKEN)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.SetMinSize((400, 300))
        
        # Visualization state
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.board_bounds = None
        self.routing_data = []
        self.current_nets = []
        self.pads = []
        self.obstacles = []
        
        # Mouse interaction
        self.last_mouse_pos = None
        self.dragging = False
        
        # Initialize buffer for double buffering
        self._buffer = None
        
        # Colors
        self.colors = {
            'background': wx.Colour(20, 20, 20),
            'board': wx.Colour(40, 60, 40),
            'pad': wx.Colour(200, 200, 100),
            'keepout': wx.Colour(80, 80, 60),  # Lighter color for keepout areas
            'trace': wx.Colour(0, 255, 0),
            'current_trace': wx.Colour(255, 100, 100),
            'via': wx.Colour(150, 150, 255),
            'grid': wx.Colour(60, 60, 60),
            'text': wx.Colour(255, 255, 255)
        }
        
        # Bind events
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, self._on_size)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_left_up)
        self.Bind(wx.EVT_MOTION, self._on_motion)
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_mouse_wheel)
        
    def set_board_data(self, board_bounds, pads, obstacles):
        """Set the PCB board data for visualization"""
        print(f"🎨 VIZ: Receiving board data:")
        print(f"🎨 VIZ:    Board bounds: {board_bounds}")
        print(f"🎨 VIZ:    Pads: {len(pads) if pads else 0}")
        print(f"🎨 VIZ:    Obstacles: {len(obstacles) if obstacles else 0}")
        
        self.board_bounds = board_bounds
        self.pads = pads
        self.obstacles = obstacles
        self._fit_to_board()
        
        print(f"🎨 VIZ: Data set, calling UpdateDrawing to trigger rendering...")
        self.UpdateDrawing()  # This is the missing call that triggers Draw()!
        
        print(f"🎨 VIZ: UpdateDrawing complete, canvas should now be visible!")
        
    def add_routing_segment(self, net_name, start_pos, end_pos, layer):
        """Add a routing segment to the visualization"""
        print(f"🎨 VIZ: Adding routing segment for net {net_name}")
        self.routing_data.append({
            'net': net_name,
            'start': start_pos,
            'end': end_pos,
            'layer': layer,
            'timestamp': time.time()
        })
        self.UpdateDrawing()  # Trigger re-render
        
    def set_current_nets(self, nets):
        """Set the currently active nets being routed"""
        print(f"🎨 VIZ: Setting current nets: {len(nets) if nets else 0}")
        self.current_nets = nets
        self.UpdateDrawing()  # Trigger re-render
        
    def Draw(self, dc):
        """Main drawing method - called by UpdateDrawing()"""
        print(f"🎨 VIZ: Draw() method called! Canvas size: {self.GetSize()}")
        
        # Clear background
        dc.SetBackground(wx.Brush(self.colors['background']))
        dc.Clear()
        
        # Setup coordinate transformation
        self._setup_transform(dc)
        
        # Draw PCB elements
        if self.board_bounds:
            print(f"🎨 VIZ: Drawing board with bounds: {self.board_bounds}")
            self._draw_board(dc)
            self._draw_grid(dc)
            self._draw_pads(dc)
            self._draw_obstacles(dc)
            self._draw_routing(dc)
            self._draw_current_nets(dc)
        else:
            print(f"🎨 VIZ: No board bounds available for drawing")
            
        # Draw info overlay
        self._draw_info_overlay(dc)
        
        print(f"🎨 VIZ: Draw() method complete!")
        
    def UpdateDrawing(self):
        """Update the canvas by drawing to buffer and refreshing"""
        print(f"🎨 VIZ: UpdateDrawing() called")
        
        # Get current size
        size = self.GetSize()
        if size.width <= 0 or size.height <= 0:
            print(f"🎨 VIZ: Canvas size invalid: {size}")
            return
            
        # Create or recreate buffer if size changed
        if not self._buffer or self._buffer.GetSize() != size:
            print(f"🎨 VIZ: Creating new buffer: {size}")
            self._buffer = wx.Bitmap(size.width, size.height)
            
        # Draw to buffer
        dc = wx.MemoryDC()
        dc.SelectObject(self._buffer)
        
        # Call our Draw method
        self.Draw(dc)
        
        # Clean up and refresh
        del dc
        self.Refresh()
        self.Update()
        
        print(f"🎨 VIZ: UpdateDrawing() complete!")
        
    def _on_paint(self, event):
        """Handle paint events - just copy buffer to screen"""
        print(f"🎨 VIZ: _on_paint() called")
        
        # If we have a buffer, just copy it to screen
        if self._buffer:
            dc = wx.PaintDC(self)
            dc.DrawBitmap(self._buffer, 0, 0)
            print(f"🎨 VIZ: Buffer copied to screen")
        else:
            # No buffer yet, force UpdateDrawing
            print(f"🎨 VIZ: No buffer, calling UpdateDrawing")
            self.UpdateDrawing()
        
    def _setup_transform(self, dc):
        """Set up coordinate transformation for zoom and pan"""
        size = self.GetSize()
        dc.SetDeviceOrigin(size.width // 2 + int(self.pan_offset[0]), 
                          size.height // 2 + int(self.pan_offset[1]))
        dc.SetUserScale(self.zoom_factor, self.zoom_factor)
        
    def _draw_board(self, dc):
        """Draw the PCB board outline"""
        if not self.board_bounds:
            print("🎨 No board bounds to draw")
            return
            
        print(f"🎨 Drawing board outline: {self.board_bounds}")
        dc.SetPen(wx.Pen(self.colors['board'], 2))
        dc.SetBrush(wx.Brush(self.colors['board'], wx.BRUSHSTYLE_TRANSPARENT))
        
        x, y, w, h = self.board_bounds
        dc.DrawRectangle(int(x), int(y), int(w), int(h))
        print(f"🎨 Board rectangle drawn at ({x}, {y}) size ({w}, {h})")
        
    def _draw_grid(self, dc):
        """Draw a helpful grid"""
        if not self.board_bounds or self.zoom_factor < 0.5:
            return
            
        dc.SetPen(wx.Pen(self.colors['grid'], 1))
        x, y, w, h = self.board_bounds
        
        # Draw grid lines every 5mm
        grid_spacing = 5.0
        for i in range(int(x), int(x + w), int(grid_spacing)):
            dc.DrawLine(i, int(y), i, int(y + h))
        for i in range(int(y), int(y + h), int(grid_spacing)):
            dc.DrawLine(int(x), i, int(x + w), i)
            
    def _draw_pads(self, dc):
        """Draw PCB pads with accurate shapes and keepout areas"""
        print(f"🎨 Drawing {len(self.pads)} pads...")
        
        drawn_count = 0
        for i, pad in enumerate(self.pads):
            bounds = pad.get('bounds', [0, 0, 1, 1])
            if len(bounds) >= 4:
                x, y, w, h = bounds
                
                # Get actual pad properties
                center = pad.get('center', [x + w/2, y + h/2])
                actual_size = pad.get('actual_size', [w, h])
                pad_shape = pad.get('shape', 'rect')
                drill_size = pad.get('drill_size', 0.0)
                
                center_x, center_y = center
                actual_w, actual_h = actual_size
                
                # Debug output for first few pads
                if i < 3:
                    print(f"   Pad {i}: {pad_shape} ({actual_w:.2f}x{actual_h:.2f}mm) at ({center_x:.2f}, {center_y:.2f}mm)")
                    if drill_size > 0:
                        print(f"     Drill: {drill_size:.2f}mm")
                
                # Draw keepout area first (larger, transparent)
                keepout_margin = 0.3  # 0.3mm keepout margin
                keepout_w = actual_w + keepout_margin * 2
                keepout_h = actual_h + keepout_margin * 2
                
                dc.SetPen(wx.Pen(self.colors['keepout'], 1))
                dc.SetBrush(wx.Brush(self.colors['keepout'], wx.BRUSHSTYLE_TRANSPARENT))
                
                # Draw keepout based on pad shape
                if pad_shape == 'circle':
                    radius = max(keepout_w, keepout_h) / 2
                    dc.DrawCircle(int(center_x), int(center_y), int(radius))
                elif pad_shape == 'oval':
                    # Draw oval keepout as circle with max dimension
                    radius = max(keepout_w, keepout_h) / 2
                    dc.DrawCircle(int(center_x), int(center_y), int(radius))
                else:  # Rectangle, roundrect, trapezoid, etc.
                    dc.DrawRectangle(int(center_x - keepout_w/2), int(center_y - keepout_h/2), 
                                   max(1, int(keepout_w)), max(1, int(keepout_h)))
                
                # Draw actual pad shape (solid)
                dc.SetPen(wx.Pen(self.colors['pad'], 1))
                dc.SetBrush(wx.Brush(self.colors['pad']))
                
                # Draw pad based on actual shape
                if pad_shape == 'circle':
                    radius = min(actual_w, actual_h) / 2
                    dc.DrawCircle(int(center_x), int(center_y), max(1, int(radius)))
                elif pad_shape == 'oval':
                    # Draw oval as ellipse or circle
                    if abs(actual_w - actual_h) < 0.1:
                        radius = min(actual_w, actual_h) / 2
                        dc.DrawCircle(int(center_x), int(center_y), max(1, int(radius)))
                    else:
                        # For oval, draw as rounded rectangle approximation
                        dc.DrawRectangle(int(center_x - actual_w/2), int(center_y - actual_h/2), 
                                       max(1, int(actual_w)), max(1, int(actual_h)))
                elif pad_shape == 'roundrect':
                    # Draw rounded rectangle as regular rectangle for now
                    dc.DrawRectangle(int(center_x - actual_w/2), int(center_y - actual_h/2), 
                                   max(1, int(actual_w)), max(1, int(actual_h)))
                else:  # rect, trapezoid, chamfered_rect, custom
                    # Draw as rectangle
                    dc.DrawRectangle(int(center_x - actual_w/2), int(center_y - actual_h/2), 
                                   max(1, int(actual_w)), max(1, int(actual_h)))
                
                # Draw drill hole if it's a through-hole pad
                if drill_size > 0.2:  # Only draw if drill is significant (>0.2mm)
                    dc.SetPen(wx.Pen(self.colors['background'], 1))
                    dc.SetBrush(wx.Brush(self.colors['background']))
                    drill_radius = drill_size / 2
                    dc.DrawCircle(int(center_x), int(center_y), max(1, int(drill_radius)))
                
                drawn_count += 1
        
        print(f"🎨 Drew {drawn_count} pads with accurate shapes and keepouts")
            
    def _draw_obstacles(self, dc):
        """Draw obstacles and existing traces"""
        dc.SetPen(wx.Pen(wx.Colour(100, 100, 100), 1))
        dc.SetBrush(wx.Brush(wx.Colour(80, 80, 80)))
        
        for obstacle in self.obstacles:
            x, y, w, h = obstacle.get('bounds', [0, 0, 1, 1])
            dc.DrawRectangle(int(x), int(y), max(1, int(w)), max(1, int(h)))
            
    def _draw_routing(self, dc):
        """Draw completed routing segments"""
        dc.SetPen(wx.Pen(self.colors['trace'], 2))
        
        for segment in self.routing_data:
            start = segment['start']
            end = segment['end']
            dc.DrawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
            
    def _draw_current_nets(self, dc):
        """Draw currently active nets being routed"""
        dc.SetPen(wx.Pen(self.colors['current_trace'], 3))
        
        for net in self.current_nets:
            # Draw animated current routing
            if 'path' in net:
                for i in range(len(net['path']) - 1):
                    start = net['path'][i]
                    end = net['path'][i + 1]
                    dc.DrawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
                    
    def _draw_info_overlay(self, dc):
        """Draw information overlay"""
        size = self.GetSize()
        dc.SetTextForeground(self.colors['text'])
        dc.SetFont(wx.Font(10, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # Reset transformation for overlay
        dc.SetDeviceOrigin(0, 0)
        dc.SetUserScale(1.0, 1.0)
        
        # Draw zoom level
        zoom_text = f"Zoom: {self.zoom_factor * 100:.0f}%"
        dc.DrawText(zoom_text, 10, 10)
        
        # Draw routing count
        route_text = f"Routes: {len(self.routing_data)}"
        dc.DrawText(route_text, 10, 30)
        
        # Draw instructions
        if self.zoom_factor == 1.0 and not self.routing_data:
            help_text = "Mouse wheel: Zoom | Left click + drag: Pan"
            text_size = dc.GetTextExtent(help_text)
            dc.DrawText(help_text, (size.width - text_size.width) // 2, size.height - 30)
            
    def _fit_to_board(self):
        """Fit the view to show the entire board"""
        if not self.board_bounds:
            return
            
        size = self.GetSize()
        x, y, w, h = self.board_bounds
        
        # Calculate zoom to fit board with some padding
        padding = 0.1  # 10% padding
        zoom_x = (size.width * (1 - padding)) / w if w > 0 else 1.0
        zoom_y = (size.height * (1 - padding)) / h if h > 0 else 1.0
        
        self.zoom_factor = min(zoom_x, zoom_y, 10.0)  # Max zoom 10x
        self.pan_offset = [-(x + w/2) * self.zoom_factor, -(y + h/2) * self.zoom_factor]
        
        # Update parent zoom label
        if hasattr(self.GetParent(), 'zoom_label'):
            self.GetParent().zoom_label.SetLabel(f"Zoom: {self.zoom_factor * 100:.0f}%")
            
    def _on_size(self, event):
        """Handle window resize"""
        print(f"🎨 VIZ: Canvas resized to {self.GetSize()}")
        self.UpdateDrawing()  # Recreate buffer and redraw
        event.Skip()
        
    def _on_left_down(self, event):
        """Handle left mouse button down"""
        self.last_mouse_pos = event.GetPosition()
        self.dragging = True
        self.CaptureMouse()
        
    def _on_left_up(self, event):
        """Handle left mouse button up"""
        if self.HasCapture():
            self.ReleaseMouse()
        self.dragging = False
        
    def _on_motion(self, event):
        """Handle mouse motion for panning"""
        if self.dragging and self.last_mouse_pos:
            current_pos = event.GetPosition()
            dx = current_pos.x - self.last_mouse_pos.x
            dy = current_pos.y - self.last_mouse_pos.y
            
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            
            self.last_mouse_pos = current_pos
            self.Refresh()
            
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        zoom_delta = 1.2 if event.GetWheelRotation() > 0 else 1.0 / 1.2
        
        # Zoom towards mouse position
        mouse_pos = event.GetPosition()
        size = self.GetSize()
        
        # Convert mouse position to world coordinates
        world_x = (mouse_pos.x - size.width // 2 - self.pan_offset[0]) / self.zoom_factor
        world_y = (mouse_pos.y - size.height // 2 - self.pan_offset[1]) / self.zoom_factor
        
        # Apply zoom
        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(20.0, self.zoom_factor * zoom_delta))
        
        # Adjust pan to keep mouse position fixed
        self.pan_offset[0] = mouse_pos.x - size.width // 2 - world_x * self.zoom_factor
        self.pan_offset[1] = mouse_pos.y - size.height // 2 - world_y * self.zoom_factor
        
        # Update zoom label
        if hasattr(self.GetParent(), 'zoom_label'):
            self.GetParent().zoom_label.SetLabel(f"Zoom: {self.zoom_factor * 100:.0f}%")
            
        self.Refresh()

class RoutingProgressDialog(wx.Dialog):
    """Enhanced progress dialog with real-time visualization"""
    
    def __init__(self, parent, title="OrthoRoute Progress"):
        super().__init__(parent, title=title, size=(900, 700), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        self.SetIcon(wx.Icon())
        self.SetMinSize((600, 400))  # Minimum size
        
        # Statistics
        self.stats = RoutingStats()
        self.start_time = time.time()
        self.update_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer_update)
        
        # Control flags
        self.should_stop_and_save = False
        self._cancelled = False
        
        # Visualization data
        self.board_bounds = None
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.routing_data = []
        self.current_nets = []
        
        # Create UI
        self._create_ui()
        self.CenterOnParent()
        
        # Start timer for regular updates
        self.update_timer.Start(100)  # Update every 100ms for smoother visualization
        
    def _create_ui(self):
        """Create the progress dialog UI"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="🚀 OrthoRoute GPU Autorouter - Live Visualization")
        title_font = title.GetFont()
        title_font.SetPointSize(14)
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        main_sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Progress bars
        progress_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Overall Progress
        overall_box = wx.BoxSizer(wx.VERTICAL)
        overall_box.Add(wx.StaticText(self, label="Overall Progress:"), 0, wx.ALL, 2)
        self.overall_gauge = wx.Gauge(self, range=1000, size=(200, 20))
        overall_box.Add(self.overall_gauge, 0, wx.EXPAND | wx.ALL, 2)
        progress_sizer.Add(overall_box, 1, wx.EXPAND | wx.ALL, 5)
        
        # Current Net Progress  
        net_box = wx.BoxSizer(wx.VERTICAL)
        net_box.Add(wx.StaticText(self, label="Current Net:"), 0, wx.ALL, 2)
        self.current_net_label = wx.StaticText(self, label="Initializing...")
        net_box.Add(self.current_net_label, 0, wx.ALL, 2)
        self.net_gauge = wx.Gauge(self, range=100, size=(200, 20))
        net_box.Add(self.net_gauge, 0, wx.EXPAND | wx.ALL, 2)
        progress_sizer.Add(net_box, 1, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(progress_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Main content: Statistics and Visualization side by side
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Statistics Panel (left side)
        stats_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Real-time Statistics")
        self.stats_text = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(280, 300))
        self.stats_text.SetFont(wx.Font(9, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        stats_box.Add(self.stats_text, 1, wx.EXPAND | wx.ALL, 5)
        content_sizer.Add(stats_box, 0, wx.EXPAND | wx.ALL, 5)
        
        # Visualization Panel (right side)
        viz_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Live PCB Routing Visualization")
        
        # Create visualization canvas
        self.viz_panel = RoutingCanvas(self)
        viz_box.Add(self.viz_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        # Zoom and pan controls
        control_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        zoom_in_btn = wx.Button(self, label="🔍+ Zoom In")
        zoom_out_btn = wx.Button(self, label="🔍- Zoom Out")
        zoom_fit_btn = wx.Button(self, label="🎯 Fit All")
        pan_reset_btn = wx.Button(self, label="🏠 Center")
        
        control_sizer.Add(zoom_in_btn, 0, wx.ALL, 2)
        control_sizer.Add(zoom_out_btn, 0, wx.ALL, 2)
        control_sizer.Add(zoom_fit_btn, 0, wx.ALL, 2)
        control_sizer.Add(pan_reset_btn, 0, wx.ALL, 2)
        control_sizer.AddStretchSpacer()
        
        # Zoom level indicator
        self.zoom_label = wx.StaticText(self, label="Zoom: 100%")
        control_sizer.Add(self.zoom_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        viz_box.Add(control_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Bind zoom controls
        zoom_in_btn.Bind(wx.EVT_BUTTON, lambda e: self._zoom_in())
        zoom_out_btn.Bind(wx.EVT_BUTTON, lambda e: self._zoom_out()) 
        zoom_fit_btn.Bind(wx.EVT_BUTTON, lambda e: self._zoom_fit())
        pan_reset_btn.Bind(wx.EVT_BUTTON, lambda e: self._pan_reset())
        
        content_sizer.Add(viz_box, 1, wx.EXPAND | wx.ALL, 5)  # Visualization gets more space
        
        main_sizer.Add(content_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        # Control buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.pause_btn = wx.Button(self, label="⏸ Pause")
        self.stop_save_btn = wx.Button(self, label="🛑 Stop & Save")
        self.cancel_btn = wx.Button(self, label="❌ Cancel")
        
        btn_sizer.Add(self.pause_btn, 0, wx.ALL, 5)
        btn_sizer.Add(self.stop_save_btn, 0, wx.ALL, 5)
        btn_sizer.Add(self.cancel_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(btn_sizer, 0, wx.CENTER | wx.ALL, 10)
        
        self.SetSizer(main_sizer)
        
        # Bind events
        self.pause_btn.Bind(wx.EVT_BUTTON, self._on_pause)
        self.stop_save_btn.Bind(wx.EVT_BUTTON, self._on_stop_save)
        self.cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel)
        
    def update_progress(self, overall_progress: float, net_progress: float = 0.0, 
                       current_net: str = "", stats: RoutingStats = None):
        """Update progress display"""
        try:
            # Update progress bars
            self.overall_gauge.SetValue(int(overall_progress * 10))  # 0-1000 range
            self.net_gauge.SetValue(int(net_progress * 100))  # 0-100 range
            
            # Update current net
            if current_net:
                self.current_net_label.SetLabel(f"🔗 {current_net}")
            
            # Update statistics
            if stats:
                self.stats = stats
                self._update_stats_display()
            
            # Force refresh
            wx.GetApp().Yield()
            
        except Exception as e:
            print(f"⚠️ Progress update error: {e}")
    
    def update_stats(self, **kwargs):
        """Update individual statistics"""
        for key, value in kwargs.items():
            if hasattr(self.stats, key):
                setattr(self.stats, key, value)
        self._update_stats_display()
    
    def _update_stats_display(self):
        """Update the statistics text display"""
        elapsed_time = time.time() - self.start_time
        completion_rate = (self.stats.nets_completed / max(self.stats.nets_total, 1)) * 100
        
        stats_text = (
            f"📊 Routing Statistics:\n"
            f"├─ Nets: {self.stats.nets_completed}/{self.stats.nets_total} completed ({completion_rate:.1f}%)\n"
            f"├─ Failed: {self.stats.nets_failed}\n"
            f"├─ Segments: {self.stats.total_segments}\n"
            f"├─ Vias: {self.stats.total_vias}\n"
            f"├─ Total length: {self.stats.total_length_mm:.2f}mm\n"
            f"├─ Elapsed time: {elapsed_time:.1f}s\n"
            f"├─ Grid cells processed: {self.stats.grid_cells_processed:,}\n"
            f"└─ GPU memory: {self.stats.gpu_memory_used_mb:.1f}MB\n\n"
            f"🔗 Current Net:\n"
            f"├─ Name: {self.stats.current_net}\n"
            f"└─ Progress: {self.stats.current_net_progress:.1f}%\n\n"
            f"⚡ Performance:\n"
            f"├─ Nets/second: {self.stats.nets_completed/max(elapsed_time, 0.1):.2f}\n"
            f"└─ Est. remaining: {self._estimate_remaining_time():.1f}s"
        )
        
        self.stats_text.SetValue(stats_text)
    
    def _estimate_remaining_time(self) -> float:
        """Estimate remaining routing time"""
        elapsed = time.time() - self.start_time
        if self.stats.nets_completed == 0:
            return 0.0
        
        nets_per_second = self.stats.nets_completed / elapsed
        remaining_nets = self.stats.nets_total - self.stats.nets_completed
        return remaining_nets / max(nets_per_second, 0.001)
    
    def _on_timer_update(self, event):
        """Regular timer update"""
        self._update_stats_display()
    
    def _on_pause(self, event):
        """Handle pause button"""
        # TODO: Implement pause functionality
        wx.MessageBox("Pause functionality will be implemented in future version", 
                     "Feature Coming Soon", wx.OK | wx.ICON_INFORMATION)
    
    def _on_stop_save(self, event):
        """Handle stop and save button"""
        result = wx.MessageBox("Stop routing and save current progress?\n\nAny completed routes will be saved to the PCB.", 
                             "Stop and Save", wx.YES_NO | wx.ICON_QUESTION)
        if result == wx.YES:
            # Set a flag that the routing worker can check
            self.should_stop_and_save = True
            self.EndModal(wx.ID_RETRY)  # Use RETRY as stop-and-save signal
    
    def _on_cancel(self, event):
        """Handle cancel button"""
        result = wx.MessageBox("Are you sure you want to cancel routing?", 
                             "Cancel Routing", wx.YES_NO | wx.ICON_QUESTION)
        if result == wx.YES:
            self._cancelled = True
            self.EndModal(wx.ID_CANCEL)
    
    def close(self):
        """Close the dialog"""
        self.update_timer.Stop()
        self.EndModal(wx.ID_OK)
    
    def _zoom_in(self):
        """Zoom in on the visualization"""
        self.viz_panel.zoom_factor = min(20.0, self.viz_panel.zoom_factor * 1.5)
        self.zoom_label.SetLabel(f"Zoom: {self.viz_panel.zoom_factor * 100:.0f}%")
        self.viz_panel.UpdateDrawing()  # Force redraw with new zoom
        
    def _zoom_out(self):
        """Zoom out on the visualization"""
        self.viz_panel.zoom_factor = max(0.1, self.viz_panel.zoom_factor / 1.5)
        self.zoom_label.SetLabel(f"Zoom: {self.viz_panel.zoom_factor * 100:.0f}%")
        self.viz_panel.UpdateDrawing()  # Force redraw with new zoom
        
    def _zoom_fit(self):
        """Fit the view to show the entire board"""
        self.viz_panel._fit_to_board()
        self.zoom_label.SetLabel(f"Zoom: {self.viz_panel.zoom_factor * 100:.0f}%")
        self.viz_panel.UpdateDrawing()  # Force redraw with new zoom
        
    def _pan_reset(self):
        """Reset pan to center"""
        self.viz_panel.pan_offset = [0, 0]
        self.viz_panel.UpdateDrawing()  # Force redraw with reset pan
    
    def set_board_data(self, board_bounds, pads=None, obstacles=None):
        """Set board data for visualization"""
        self.viz_panel.set_board_data(board_bounds, pads or [], obstacles or [])
        
    def add_routing_segment(self, net_name, start_pos, end_pos, layer=0):
        """Add a routing segment to the visualization"""
        self.viz_panel.add_routing_segment(net_name, start_pos, end_pos, layer)
        
    def set_current_nets(self, nets):
        """Set currently active nets"""
        self.viz_panel.set_current_nets(nets)
    
    def WasCancelled(self):
        """Check if dialog was cancelled (compatibility with wx.ProgressDialog)"""
        return hasattr(self, '_cancelled') and self._cancelled
    
    def Update(self, progress, message=""):
        """Update progress (compatibility with wx.ProgressDialog)"""
        self.update_progress(progress / 100.0, current_net=message)
        return not self.WasCancelled()

class RoutingVisualizer:
    """Real-time routing visualization and callback handler"""
    
    def __init__(self, progress_dialog: RoutingProgressDialog = None):
        self.progress_dialog = progress_dialog
        self.stats = RoutingStats()
        self.callbacks = []
        self.is_active = True
        
    def add_callback(self, callback: Callable):
        """Add a progress callback function"""
        self.callbacks.append(callback)
    
    def update_routing_progress(self, net_name: str, net_progress: float, 
                              overall_progress: float, **kwargs):
        """Update routing progress with detailed information"""
        if not self.is_active:
            return
        
        try:
            # Update internal stats
            self.stats.current_net = net_name
            self.stats.current_net_progress = net_progress * 100
            
            # Update any additional stats
            for key, value in kwargs.items():
                if hasattr(self.stats, key):
                    setattr(self.stats, key, value)
            
            # Update progress dialog
            if self.progress_dialog:
                self.progress_dialog.update_progress(
                    overall_progress, net_progress, net_name, self.stats
                )
            
            # Call external callbacks
            for callback in self.callbacks:
                try:
                    callback(net_name, net_progress, overall_progress, self.stats)
                except Exception as e:
                    print(f"⚠️ Callback error: {e}")
                    
        except Exception as e:
            print(f"⚠️ Visualization update error: {e}")
    
    def update_net_completed(self, net_name: str, success: bool, segments: int = 0, 
                           vias: int = 0, length_mm: float = 0.0):
        """Update when a net routing is completed"""
        if success:
            self.stats.nets_completed += 1
            self.stats.total_segments += segments
            self.stats.total_vias += vias
            self.stats.total_length_mm += length_mm
            print(f"✅ Net '{net_name}' routed: {segments} segments, {vias} vias, {length_mm:.2f}mm")
        else:
            self.stats.nets_failed += 1
            print(f"❌ Net '{net_name}' failed to route")
        
        # Update display
        if self.progress_dialog:
            self.progress_dialog.update_stats(
                nets_completed=self.stats.nets_completed,
                nets_failed=self.stats.nets_failed,
                total_segments=self.stats.total_segments,
                total_vias=self.stats.total_vias,
                total_length_mm=self.stats.total_length_mm
            )
    
    def update_gpu_stats(self, memory_used_mb: float, cells_processed: int = 0):
        """Update GPU usage statistics"""
        self.stats.gpu_memory_used_mb = memory_used_mb
        if cells_processed > 0:
            self.stats.grid_cells_processed = cells_processed
        
        if self.progress_dialog:
            self.progress_dialog.update_stats(
                gpu_memory_used_mb=memory_used_mb,
                grid_cells_processed=self.stats.grid_cells_processed
            )
    
    def initialize_routing(self, total_nets: int):
        """Initialize routing session"""
        self.stats.nets_total = total_nets
        self.stats.nets_completed = 0
        self.stats.nets_failed = 0
        
        if self.progress_dialog:
            self.progress_dialog.update_stats(nets_total=total_nets)
        
        print(f"🚀 Routing visualization initialized: {total_nets} nets to route")
    
    def show_final_results(self, results: Dict):
        """Display final routing results"""
        if not self.is_active:
            return
        
        try:
            stats = results.get('statistics', {})
            completion_rate = stats.get('completion_rate', 0)
            
            # Update final stats
            self.stats.nets_completed = stats.get('nets_completed', self.stats.nets_completed)
            self.stats.nets_failed = stats.get('nets_failed', self.stats.nets_failed)
            
            # Create results summary
            summary = (
                f"🎉 Routing Complete!\n\n"
                f"📊 Final Statistics:\n"
                f"• Completion rate: {completion_rate:.1f}%\n"
                f"• Nets routed: {self.stats.nets_completed}/{self.stats.nets_total}\n"
                f"• Total segments: {self.stats.total_segments}\n"
                f"• Total vias: {self.stats.total_vias}\n"
                f"• Total length: {self.stats.total_length_mm:.2f}mm\n"
                f"• Routing time: {stats.get('routing_time_seconds', 0):.1f}s"
            )
            
            print(summary)
            
        except Exception as e:
            print(f"⚠️ Results display error: {e}")
    
    def close(self):
        """Close visualization"""
        self.is_active = False
        if self.progress_dialog:
            try:
                self.progress_dialog.close()
            except:
                pass
        print("🎨 Visualization closed")

def create_progress_dialog(parent=None, title="OrthoRoute Progress") -> RoutingProgressDialog:
    """Factory function to create a progress dialog"""
    return RoutingProgressDialog(parent, title)

def create_visualizer(progress_dialog=None) -> RoutingVisualizer:
    """Factory function to create a routing visualizer"""
    return RoutingVisualizer(progress_dialog)
