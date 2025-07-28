"""
Real-Time Visualization for OrthoRoute
Interactive display of GPU routing progress and results

This module provides real-time visualization of the routing process,
including grid state, net routing, congestion analysis, and performance metrics.
"""

import cupy as cp
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import time
import threading
from dataclasses import dataclass
from .grid_manager import Point3D, Net, GPUGrid
from .design_rules import DesignRuleViolation

# Optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Rectangle, Circle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import tkinter.font as tkFont
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

@dataclass
class VisualizationConfig:
    """Configuration for visualization system"""
    backend: str = "matplotlib"  # "matplotlib", "tkinter", "plotly", "headless"
    update_interval: float = 0.1  # seconds
    show_grid: bool = True
    show_obstacles: bool = True
    show_nets: bool = True
    show_congestion: bool = True
    show_vias: bool = True
    color_by_net: bool = True
    animation_speed: float = 1.0
    window_size: Tuple[int, int] = (1200, 800)
    dpi: int = 100

class ColorScheme:
    """Color schemes for different visualization modes"""
    
    DARK_THEME = {
        'background': '#1e1e1e',
        'grid': '#404040',
        'obstacles': '#8b0000',
        'available': '#2d2d2d',
        'routing': '#00ff00',
        'completed': '#0080ff',
        'congested': '#ff4444',
        'via': '#ffff00',
        'text': '#ffffff'
    }
    
    LIGHT_THEME = {
        'background': '#ffffff',
        'grid': '#e0e0e0',
        'obstacles': '#ff6b6b',
        'available': '#f8f8f8',
        'routing': '#4ecdc4',
        'completed': '#45b7d1',
        'congested': '#ff6b6b',
        'via': '#f39c12',
        'text': '#2c3e50'
    }
    
    HIGH_CONTRAST = {
        'background': '#000000',
        'grid': '#333333',
        'obstacles': '#ff0000',
        'available': '#111111',
        'routing': '#00ff00',
        'completed': '#0000ff',
        'congested': '#ffff00',
        'via': '#ff00ff',
        'text': '#ffffff'
    }

class RoutingVisualizer:
    """Real-time visualization of GPU routing process"""
    
    def __init__(self, grid: GPUGrid, config: VisualizationConfig = None):
        self.grid = grid
        self.config = config or VisualizationConfig()
        self.color_scheme = ColorScheme.DARK_THEME
        
        # Visualization state
        self.active = False
        self.paused = False
        self.current_layer = 0
        self.zoom_level = 1.0
        self.pan_offset = (0, 0)
        
        # Data arrays (CPU copies for visualization)
        self.grid_state = np.ones((grid.layers, grid.height, grid.width), dtype=np.uint8)
        self.net_colors = {}
        self.routing_progress = {}
        self.congestion_heatmap = np.zeros((grid.height, grid.width), dtype=np.float32)
        
        # Performance tracking
        self.frame_times = []
        self.routing_stats = {
            'nets_completed': 0,
            'nets_total': 0,
            'current_iteration': 0,
            'routing_time': 0.0,
            'nets_per_second': 0.0
        }
        
        # Threading for non-blocking updates
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Initialize backend
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected visualization backend"""
        if self.config.backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            self._init_matplotlib()
        elif self.config.backend == "tkinter" and TKINTER_AVAILABLE:
            self._init_tkinter()
        elif self.config.backend == "plotly" and PLOTLY_AVAILABLE:
            self._init_plotly()
        else:
            self.config.backend = "headless"
            print("⚠️  Visualization backend not available, running headless")
    
    def _init_matplotlib(self):
        """Initialize matplotlib-based visualization"""
        plt.style.use('dark_background')
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('OrthoRoute Real-Time Visualization', fontsize=16, color='white')
        
        # Main routing view
        self.main_ax = self.axes[0, 0]
        self.main_ax.set_title('Routing Grid (Layer 0)')
        self.main_ax.set_aspect('equal')
        
        # Congestion heatmap
        self.congestion_ax = self.axes[0, 1]
        self.congestion_ax.set_title('Congestion Analysis')
        
        # Statistics plot
        self.stats_ax = self.axes[1, 0]
        self.stats_ax.set_title('Routing Progress')
        
        # Performance plot
        self.perf_ax = self.axes[1, 1]
        self.perf_ax.set_title('Performance Metrics')
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show(block=False)
    
    def _init_tkinter(self):
        """Initialize tkinter-based visualization"""
        self.root = tk.Tk()
        self.root.title("OrthoRoute Visualization")
        self.root.geometry(f"{self.config.window_size[0]}x{self.config.window_size[1]}")
        self.root.configure(bg=self.color_scheme['background'])
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for grid visualization
        self.canvas = tk.Canvas(
            self.main_frame,
            bg=self.color_scheme['background'],
            width=800,
            height=600
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create control panel
        self._create_control_panel()
        
        # Bind events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<MouseWheel>", self._on_canvas_scroll)
        self.root.bind("<Key>", self._on_key_press)
    
    def _init_plotly(self):
        """Initialize plotly-based visualization"""
        # Create subplot layout
        self.plotly_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Routing Grid', 'Congestion Heatmap', 
                          'Progress', 'Performance'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        self.plotly_fig.update_layout(
            title="OrthoRoute Real-Time Visualization",
            template="plotly_dark",
            height=800
        )
    
    def start_visualization(self, update_callback: Optional[Callable] = None):
        """Start real-time visualization"""
        if self.config.backend == "headless":
            return
        
        self.active = True
        self.stop_event.clear()
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            args=(update_callback,)
        )
        self.update_thread.daemon = True
        self.update_thread.start()
        
        print("🎬 Visualization started")
    
    def stop_visualization(self):
        """Stop visualization and cleanup"""
        self.active = False
        if self.update_thread:
            self.stop_event.set()
            self.update_thread.join(timeout=1.0)
        
        if self.config.backend == "matplotlib":
            plt.close('all')
        elif self.config.backend == "tkinter":
            self.root.quit()
        
        print("⏹️  Visualization stopped")
    
    def update_grid_state(self, grid: GPUGrid):
        """Update visualization with current grid state"""
        # Copy GPU arrays to CPU for visualization
        self.grid_state = cp.asnumpy(grid.availability)
        congestion = cp.asnumpy(grid.congestion_cost)
        usage = cp.asnumpy(grid.usage_count)
        
        # Create congestion heatmap
        self.congestion_heatmap = np.mean(congestion, axis=0)  # Average across layers
    
    def update_routing_progress(self, nets: List[Net], iteration: int, 
                              routing_time: float):
        """Update routing progress information"""
        completed_nets = sum(1 for net in nets if net.routed)
        
        self.routing_stats.update({
            'nets_completed': completed_nets,
            'nets_total': len(nets),
            'current_iteration': iteration,
            'routing_time': routing_time,
            'nets_per_second': completed_nets / routing_time if routing_time > 0 else 0
        })
        
        # Update net colors and paths
        for net in nets:
            if net.net_id not in self.net_colors:
                self.net_colors[net.net_id] = self._generate_net_color(net.net_id)
            
            if net.routed and net.route_path:
                self.routing_progress[net.net_id] = net.route_path
    
    def add_violations(self, violations: List[DesignRuleViolation]):
        """Add design rule violations to visualization"""
        self.violations = violations
        # Violations will be shown as overlays on the main grid
    
    def _update_loop(self, update_callback: Optional[Callable]):
        """Main update loop for visualization"""
        last_update = time.time()
        
        while self.active and not self.stop_event.is_set():
            current_time = time.time()
            
            if current_time - last_update >= self.config.update_interval:
                start_frame = time.time()
                
                # Update visualization
                if not self.paused:
                    self._update_display()
                
                # Call external update callback if provided
                if update_callback:
                    update_callback(self)
                
                # Track frame time
                frame_time = time.time() - start_frame
                self.frame_times.append(frame_time)
                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)
                
                last_update = current_time
            
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
    
    def _update_display(self):
        """Update the display based on current backend"""
        if self.config.backend == "matplotlib":
            self._update_matplotlib()
        elif self.config.backend == "tkinter":
            self._update_tkinter()
        elif self.config.backend == "plotly":
            self._update_plotly()
    
    def _update_matplotlib(self):
        """Update matplotlib visualization"""
        # Clear axes
        self.main_ax.clear()
        self.congestion_ax.clear()
        self.stats_ax.clear()
        self.perf_ax.clear()
        
        # Update main grid view
        layer_data = self.grid_state[self.current_layer]
        
        # Show grid
        if self.config.show_grid:
            self.main_ax.imshow(layer_data, cmap='gray', alpha=0.3)
        
        # Show routed nets
        if self.config.show_nets:
            for net_id, path in self.routing_progress.items():
                if path:
                    # Filter points on current layer
                    layer_points = [(p.x, p.y) for p in path if p.layer == self.current_layer]
                    if layer_points:
                        xs, ys = zip(*layer_points)
                        color = self.net_colors.get(net_id, '#00ff00')
                        self.main_ax.plot(xs, ys, color=color, linewidth=2, alpha=0.8)
                        
                        # Show vias
                        if self.config.show_vias:
                            via_points = [(p.x, p.y) for i, p in enumerate(path) 
                                        if i > 0 and p.layer != path[i-1].layer]
                            if via_points:
                                vx, vy = zip(*via_points)
                                self.main_ax.scatter(vx, vy, c='yellow', s=20, marker='o')
        
        # Update congestion heatmap
        if self.config.show_congestion:
            im = self.congestion_ax.imshow(self.congestion_heatmap, cmap='hot', alpha=0.8)
            self.congestion_ax.set_title(f'Congestion (Max: {np.max(self.congestion_heatmap):.1f})')
        
        # Update statistics
        stats = self.routing_stats
        self.stats_ax.bar(['Completed', 'Remaining'], 
                         [stats['nets_completed'], stats['nets_total'] - stats['nets_completed']],
                         color=['green', 'red'])
        self.stats_ax.set_title(f"Progress: {stats['nets_completed']}/{stats['nets_total']} "
                               f"(Iter {stats['current_iteration']})")
        
        # Update performance metrics
        if self.frame_times:
            self.perf_ax.plot(self.frame_times[-50:], color='cyan')
            self.perf_ax.set_title(f"Frame Time: {np.mean(self.frame_times[-10:]):.3f}s "
                                  f"({stats['nets_per_second']:.1f} nets/s)")
        
        # Set titles and labels
        self.main_ax.set_title(f'Routing Grid (Layer {self.current_layer})')
        
        plt.draw()
        plt.pause(0.001)
    
    def _update_tkinter(self):
        """Update tkinter visualization"""
        self.canvas.delete("all")  # Clear canvas
        
        # Calculate scaling
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return  # Canvas not ready
        
        scale_x = canvas_width / self.grid.width
        scale_y = canvas_height / self.grid.height
        scale = min(scale_x, scale_y) * self.zoom_level
        
        # Draw grid
        if self.config.show_grid:
            for x in range(0, self.grid.width, 10):
                x_pos = x * scale + self.pan_offset[0]
                self.canvas.create_line(x_pos, 0, x_pos, canvas_height, 
                                      fill=self.color_scheme['grid'], width=1)
            
            for y in range(0, self.grid.height, 10):
                y_pos = y * scale + self.pan_offset[1]
                self.canvas.create_line(0, y_pos, canvas_width, y_pos,
                                      fill=self.color_scheme['grid'], width=1)
        
        # Draw obstacles
        if self.config.show_obstacles:
            layer_data = self.grid_state[self.current_layer]
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    if layer_data[y, x] == 0:  # Obstacle
                        x1 = x * scale + self.pan_offset[0]
                        y1 = y * scale + self.pan_offset[1]
                        x2 = x1 + scale
                        y2 = y1 + scale
                        self.canvas.create_rectangle(x1, y1, x2, y2,
                                                   fill=self.color_scheme['obstacles'],
                                                   outline="")
        
        # Draw routed nets
        if self.config.show_nets:
            for net_id, path in self.routing_progress.items():
                if path:
                    color = self.net_colors.get(net_id, self.color_scheme['routing'])
                    
                    # Draw path segments
                    layer_points = [(p.x * scale + self.pan_offset[0], 
                                   p.y * scale + self.pan_offset[1]) 
                                  for p in path if p.layer == self.current_layer]
                    
                    for i in range(len(layer_points) - 1):
                        self.canvas.create_line(layer_points[i][0], layer_points[i][1],
                                              layer_points[i+1][0], layer_points[i+1][1],
                                              fill=color, width=2)
        
        # Update status
        self._update_status_text()
        
        self.root.update_idletasks()
    
    def _update_plotly(self):
        """Update plotly visualization"""
        # This would update the plotly figure
        # Implementation would depend on specific requirements
        pass
    
    def _create_control_panel(self):
        """Create control panel for tkinter interface"""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Layer controls
        ttk.Label(control_frame, text="Layer:").pack(pady=2)
        self.layer_var = tk.IntVar(value=0)
        layer_scale = ttk.Scale(control_frame, from_=0, to=self.grid.layers-1,
                               variable=self.layer_var, orient=tk.HORIZONTAL,
                               command=self._on_layer_change)
        layer_scale.pack(fill=tk.X, pady=2)
        
        # View controls
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="View Controls:").pack(pady=2)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Grid", 
                       variable=self.show_grid_var).pack(anchor=tk.W)
        
        self.show_nets_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Nets",
                       variable=self.show_nets_var).pack(anchor=tk.W)
        
        self.show_congestion_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Congestion",
                       variable=self.show_congestion_var).pack(anchor=tk.W)
        
        # Control buttons
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="Pause/Resume",
                  command=self._toggle_pause).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset View",
                  command=self._reset_view).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Export Image",
                  command=self._export_image).pack(fill=tk.X, pady=2)
        
        # Statistics display
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.stats_text = tk.Text(control_frame, height=10, width=25)
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=2)
    
    def _update_status_text(self):
        """Update status text in tkinter interface"""
        if hasattr(self, 'stats_text'):
            stats = self.routing_stats
            status = f"""Routing Statistics:

Nets: {stats['nets_completed']}/{stats['nets_total']}
Success Rate: {stats['nets_completed']/stats['nets_total']*100:.1f}%
Iteration: {stats['current_iteration']}
Time: {stats['routing_time']:.1f}s
Rate: {stats['nets_per_second']:.1f} nets/s

Layer: {self.current_layer}
Zoom: {self.zoom_level:.1f}x

Violations: {len(getattr(self, 'violations', []))}
"""
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, status)
    
    def _generate_net_color(self, net_id: int) -> str:
        """Generate unique color for net"""
        # Use deterministic color generation
        np.random.seed(net_id)
        hue = np.random.random()
        saturation = 0.7 + 0.3 * np.random.random()
        value = 0.8 + 0.2 * np.random.random()
        
        # Convert HSV to RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
    
    def _on_layer_change(self, value):
        """Handle layer change"""
        self.current_layer = int(float(value))
    
    def _on_canvas_click(self, event):
        """Handle canvas click events"""
        # Could implement net selection, zoom to point, etc.
        pass
    
    def _on_canvas_scroll(self, event):
        """Handle canvas scroll for zooming"""
        if event.delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1
        self.zoom_level = max(0.1, min(10.0, self.zoom_level))
    
    def _on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.char == ' ':
            self._toggle_pause()
        elif event.char == 'r':
            self._reset_view()
        elif event.keysym in ['Up', 'Down']:
            direction = 1 if event.keysym == 'Up' else -1
            self.current_layer = max(0, min(self.grid.layers-1, 
                                          self.current_layer + direction))
    
    def _toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
    
    def _reset_view(self):
        """Reset view to default"""
        self.zoom_level = 1.0
        self.pan_offset = (0, 0)
        self.current_layer = 0
    
    def _export_image(self):
        """Export current view as image"""
        if self.config.backend == "matplotlib":
            plt.savefig(f"orthoroute_view_{int(time.time())}.png", dpi=300, 
                       bbox_inches='tight', facecolor='black')
        # Other backends would implement their own export methods

def create_visualization_demo():
    """Create a demo visualization for testing"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for demo")
        return
    
    # Create dummy grid and data
    from .grid_manager import GPUGrid
    
    grid = GPUGrid(50, 50, 4, 0.1)
    config = VisualizationConfig(backend="matplotlib")
    
    viz = RoutingVisualizer(grid, config)
    viz.start_visualization()
    
    # Simulate some routing activity
    import time
    for i in range(10):
        # Simulate routing progress
        time.sleep(1)
        viz.routing_stats['nets_completed'] = i
        viz.routing_stats['nets_total'] = 10
        viz.routing_stats['current_iteration'] = i
    
    viz.stop_visualization()

# Utility functions
def check_visualization_dependencies() -> Dict[str, bool]:
    """Check which visualization backends are available"""
    return {
        'matplotlib': MATPLOTLIB_AVAILABLE,
        'tkinter': TKINTER_AVAILABLE,
        'plotly': PLOTLY_AVAILABLE
    }

def get_recommended_backend() -> str:
    """Get recommended visualization backend based on available dependencies"""
    if TKINTER_AVAILABLE:
        return "tkinter"  # Most interactive
    elif MATPLOTLIB_AVAILABLE:
        return "matplotlib"  # Good for static analysis
    elif PLOTLY_AVAILABLE:
        return "plotly"  # Web-based, good for sharing
    else:
        return "headless"  # No visualization