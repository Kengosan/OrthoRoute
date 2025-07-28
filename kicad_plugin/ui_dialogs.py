"""
UI Dialogs for OrthoRoute KiCad Plugin
User interface components for GPU autorouter configuration and control

This module provides all the dialog windows and UI components needed
for the OrthoRoute KiCad plugin, including configuration, progress tracking,
results display, and error handling.
"""

import wx
import wx.lib.newevent
import threading
import time
import json
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Custom events for threading communication
UpdateProgressEvent, EVT_UPDATE_PROGRESS = wx.lib.newevent.NewEvent()
RoutingCompleteEvent, EVT_ROUTING_COMPLETE = wx.lib.newevent.NewEvent()
ErrorEvent, EVT_ERROR = wx.lib.newevent.NewEvent()

class RoutingStatus(Enum):
    """Routing operation status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    EXPORTING = "exporting"
    ROUTING = "routing"
    IMPORTING = "importing"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class ProgressUpdate:
    """Progress update data structure"""
    status: RoutingStatus
    progress: int  # 0-100
    message: str
    details: Optional[Dict] = None

class OrthoRouteConfigDialog(wx.Dialog):
    """Main configuration dialog for OrthoRoute settings"""
    
    def __init__(self, parent, board_stats: Dict = None):
        super().__init__(parent, title="OrthoRoute GPU Autorouter Configuration", 
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.board_stats = board_stats or {}
        self.config = self._get_default_config()
        
        # Dialog state
        self.gpu_info = None
        self.performance_estimate = None
        
        self._create_ui()
        self._load_gpu_info()
        self._update_performance_estimate()
        self._bind_events()
        
        # Size and center
        self.SetSize((600, 700))
        self.CenterOnParent()
    
    def _create_ui(self):
        """Create the main UI layout"""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title and description
        title = wx.StaticText(panel, label="OrthoRoute GPU Autorouter")
        title_font = title.GetFont()
        title_font.PointSize += 4
        title_font = title_font.Bold()
        title.SetFont(title_font)
        main_sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        description = wx.StaticText(panel, 
            label="Configure GPU-accelerated PCB routing settings")
        main_sizer.Add(description, 0, wx.ALL | wx.CENTER, 5)
        
        # Create notebook for tabbed interface
        notebook = wx.Notebook(panel)
        
        # Tab 1: Basic Settings
        basic_panel = self._create_basic_settings_panel(notebook)
        notebook.AddPage(basic_panel, "Basic Settings")
        
        # Tab 2: Advanced Settings
        advanced_panel = self._create_advanced_settings_panel(notebook)
        notebook.AddPage(advanced_panel, "Advanced")
        
        # Tab 3: Net Filtering
        filtering_panel = self._create_filtering_panel(notebook)
        notebook.AddPage(filtering_panel, "Net Filtering")
        
        # Tab 4: GPU Info
        gpu_panel = self._create_gpu_info_panel(notebook)
        notebook.AddPage(gpu_panel, "GPU Status")
        
        main_sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 10)
        
        # Performance estimate
        self.perf_text = wx.StaticText(panel, label="")
        main_sizer.Add(self.perf_text, 0, wx.ALL | wx.CENTER, 5)
        
        # Buttons
        button_sizer = self._create_button_panel(panel)
        main_sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        panel.SetSizer(main_sizer)
    
    def _create_basic_settings_panel(self, parent) -> wx.Panel:
        """Create basic settings tab"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Grid Settings
        grid_box = wx.StaticBox(panel, label="Grid Configuration")
        grid_sizer = wx.StaticBoxSizer(grid_box, wx.VERTICAL)
        
        # Grid pitch
        pitch_sizer = wx.FlexGridSizer(2, 2, 5, 10)
        pitch_sizer.Add(wx.StaticText(panel, label="Grid Pitch (mm):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.pitch_ctrl = wx.SpinCtrlDouble(panel, value=0.1, min=0.025, max=0.5, inc=0.025)
        self.pitch_ctrl.SetDigits(3)
        pitch_sizer.Add(self.pitch_ctrl, 1, wx.EXPAND)
        
        pitch_sizer.Add(wx.StaticText(panel, label="Routing Layers:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.layers_ctrl = wx.SpinCtrl(panel, value=4, min=2, max=32)
        pitch_sizer.Add(self.layers_ctrl, 1, wx.EXPAND)
        
        grid_sizer.Add(pitch_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Grid info
        self.grid_info = wx.StaticText(panel, label="Grid size will be calculated from board bounds")
        grid_sizer.Add(self.grid_info, 0, wx.ALL, 5)
        
        sizer.Add(grid_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Routing Settings
        routing_box = wx.StaticBox(panel, label="Routing Configuration")
        routing_sizer = wx.StaticBoxSizer(routing_box, wx.VERTICAL)
        
        routing_grid = wx.FlexGridSizer(3, 2, 5, 10)
        
        routing_grid.Add(wx.StaticText(panel, label="Max Iterations:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.iterations_ctrl = wx.SpinCtrl(panel, value=20, min=5, max=100)
        routing_grid.Add(self.iterations_ctrl, 1, wx.EXPAND)
        
        routing_grid.Add(wx.StaticText(panel, label="Batch Size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.batch_ctrl = wx.SpinCtrl(panel, value=256, min=64, max=2048)
        routing_grid.Add(self.batch_ctrl, 1, wx.EXPAND)
        
        routing_grid.Add(wx.StaticText(panel, label="Tile Size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tile_ctrl = wx.SpinCtrl(panel, value=64, min=32, max=256)
        routing_grid.Add(self.tile_ctrl, 1, wx.EXPAND)
        
        routing_sizer.Add(routing_grid, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(routing_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Quality vs Speed
        quality_box = wx.StaticBox(panel, label="Quality vs Speed")
        quality_sizer = wx.StaticBoxSizer(quality_box, wx.VERTICAL)
        
        self.quality_slider = wx.Slider(panel, value=3, minValue=1, maxValue=5,
                                       style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        quality_sizer.Add(wx.StaticText(panel, label="1 = Fast, 5 = High Quality"), 0, wx.ALL, 2)
        quality_sizer.Add(self.quality_slider, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(quality_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_advanced_settings_panel(self, parent) -> wx.Panel:
        """Create advanced settings tab"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Algorithm Settings
        algo_box = wx.StaticBox(panel, label="Algorithm Parameters")
        algo_sizer = wx.StaticBoxSizer(algo_box, wx.VERTICAL)
        
        algo_grid = wx.FlexGridSizer(4, 2, 5, 10)
        
        algo_grid.Add(wx.StaticText(panel, label="Congestion Factor:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.congestion_ctrl = wx.SpinCtrlDouble(panel, value=1.5, min=1.1, max=3.0, inc=0.1)
        self.congestion_ctrl.SetDigits(1)
        algo_grid.Add(self.congestion_ctrl, 1, wx.EXPAND)
        
        algo_grid.Add(wx.StaticText(panel, label="Via Cost:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.via_cost_ctrl = wx.SpinCtrl(panel, value=10, min=1, max=100)
        algo_grid.Add(self.via_cost_ctrl, 1, wx.EXPAND)
        
        algo_grid.Add(wx.StaticText(panel, label="Direction Change Cost:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.direction_cost_ctrl = wx.SpinCtrl(panel, value=2, min=0, max=10)
        algo_grid.Add(self.direction_cost_ctrl, 1, wx.EXPAND)
        
        algo_grid.Add(wx.StaticText(panel, label="Trace Cost:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.trace_cost_ctrl = wx.SpinCtrl(panel, value=1, min=1, max=10)
        algo_grid.Add(self.trace_cost_ctrl, 1, wx.EXPAND)
        
        algo_sizer.Add(algo_grid, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(algo_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # GPU Settings
        gpu_box = wx.StaticBox(panel, label="GPU Configuration")
        gpu_sizer = wx.StaticBoxSizer(gpu_box, wx.VERTICAL)
        
        gpu_grid = wx.FlexGridSizer(2, 2, 5, 10)
        
        gpu_grid.Add(wx.StaticText(panel, label="GPU Device ID:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.gpu_id_ctrl = wx.SpinCtrl(panel, value=0, min=0, max=7)
        gpu_grid.Add(self.gpu_id_ctrl, 1, wx.EXPAND)
        
        gpu_grid.Add(wx.StaticText(panel, label="Memory Limit (GB):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.memory_limit_ctrl = wx.SpinCtrl(panel, value=0, min=0, max=64)  # 0 = auto
        gpu_grid.Add(self.memory_limit_ctrl, 1, wx.EXPAND)
        
        gpu_sizer.Add(gpu_grid, 0, wx.EXPAND | wx.ALL, 5)
        
        # Memory limit help
        memory_help = wx.StaticText(panel, label="0 = Automatic memory management")
        gpu_sizer.Add(memory_help, 0, wx.ALL, 5)
        
        sizer.Add(gpu_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Output Settings
        output_box = wx.StaticBox(panel, label="Output Options")
        output_sizer = wx.StaticBoxSizer(output_box, wx.VERTICAL)
        
        self.save_results_cb = wx.CheckBox(panel, label="Save routing results to file")
        self.save_results_cb.SetValue(True)
        output_sizer.Add(self.save_results_cb, 0, wx.ALL, 5)
        
        self.show_visualization_cb = wx.CheckBox(panel, label="Show real-time visualization")
        self.show_visualization_cb.SetValue(False)
        output_sizer.Add(self.show_visualization_cb, 0, wx.ALL, 5)
        
        self.verbose_cb = wx.CheckBox(panel, label="Verbose logging")
        self.verbose_cb.SetValue(False)
        output_sizer.Add(self.verbose_cb, 0, wx.ALL, 5)
        
        sizer.Add(output_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_filtering_panel(self, parent) -> wx.Panel:
        """Create net filtering tab"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Net Selection
        selection_box = wx.StaticBox(panel, label="Net Selection")
        selection_sizer = wx.StaticBoxSizer(selection_box, wx.VERTICAL)
        
        # Net pattern
        pattern_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pattern_sizer.Add(wx.StaticText(panel, label="Net Name Pattern:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.net_pattern_ctrl = wx.TextCtrl(panel, value="")
        pattern_sizer.Add(self.net_pattern_ctrl, 1, wx.EXPAND)
        selection_sizer.Add(pattern_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        pattern_help = wx.StaticText(panel, label="Leave empty to route all nets, or enter text that must be in net name")
        selection_sizer.Add(pattern_help, 0, wx.ALL, 5)
        
        # Pin count limits
        pin_sizer = wx.FlexGridSizer(2, 2, 5, 10)
        pin_sizer.Add(wx.StaticText(panel, label="Minimum pins per net:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.min_pins_ctrl = wx.SpinCtrl(panel, value=2, min=2, max=100)
        pin_sizer.Add(self.min_pins_ctrl, 1, wx.EXPAND)
        
        pin_sizer.Add(wx.StaticText(panel, label="Maximum pins per net:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.max_pins_ctrl = wx.SpinCtrl(panel, value=100, min=2, max=1000)
        pin_sizer.Add(self.max_pins_ctrl, 1, wx.EXPAND)
        
        selection_sizer.Add(pin_sizer, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(selection_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Exclusion Options
        exclusion_box = wx.StaticBox(panel, label="Exclusion Options")
        exclusion_sizer = wx.StaticBoxSizer(exclusion_box, wx.VERTICAL)
        
        self.skip_power_cb = wx.CheckBox(panel, label="Skip power/ground nets (VCC, GND, etc.)")
        self.skip_power_cb.SetValue(True)
        exclusion_sizer.Add(self.skip_power_cb, 0, wx.ALL, 5)
        
        self.skip_routed_cb = wx.CheckBox(panel, label="Skip already routed nets")
        self.skip_routed_cb.SetValue(True)
        exclusion_sizer.Add(self.skip_routed_cb, 0, wx.ALL, 5)
        
        self.skip_locked_cb = wx.CheckBox(panel, label="Skip locked tracks")
        self.skip_locked_cb.SetValue(True)
        exclusion_sizer.Add(self.skip_locked_cb, 0, wx.ALL, 5)
        
        sizer.Add(exclusion_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Net Priority
        priority_box = wx.StaticBox(panel, label="Net Priority")
        priority_sizer = wx.StaticBoxSizer(priority_box, wx.VERTICAL)
        
        priority_choices = [
            "Automatic (by net class)",
            "All nets equal priority",
            "Power nets first",
            "Clock nets first",
            "Short nets first",
            "Long nets first"
        ]
        
        self.priority_choice = wx.Choice(panel, choices=priority_choices)
        self.priority_choice.SetSelection(0)
        priority_sizer.Add(self.priority_choice, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(priority_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Board Statistics
        if self.board_stats:
            stats_box = wx.StaticBox(panel, label="Current Board Statistics")
            stats_sizer = wx.StaticBoxSizer(stats_box, wx.VERTICAL)
            
            stats_text = (
                f"Total nets: {self.board_stats.get('total_nets', 0)}\n"
                f"Routed nets: {self.board_stats.get('routed_nets', 0)}\n"
                f"Power nets: {self.board_stats.get('power_nets', 0)}\n"
                f"Signal nets: {self.board_stats.get('signal_nets', 0)}\n"
                f"Copper layers: {self.board_stats.get('copper_layers', 0)}"
            )
            
            stats_label = wx.StaticText(panel, label=stats_text)
            stats_sizer.Add(stats_label, 0, wx.ALL, 5)
            
            sizer.Add(stats_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_gpu_info_panel(self, parent) -> wx.Panel:
        """Create GPU information tab"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # GPU Status
        status_box = wx.StaticBox(panel, label="GPU Status")
        status_sizer = wx.StaticBoxSizer(status_box, wx.VERTICAL)
        
        self.gpu_status_text = wx.StaticText(panel, label="Checking GPU availability...")
        status_sizer.Add(self.gpu_status_text, 1, wx.EXPAND | wx.ALL, 5)
        
        # Refresh button
        refresh_btn = wx.Button(panel, label="Refresh GPU Info")
        refresh_btn.Bind(wx.EVT_BUTTON, self._on_refresh_gpu)
        status_sizer.Add(refresh_btn, 0, wx.ALL, 5)
        
        sizer.Add(status_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        # Performance Recommendations
        perf_box = wx.StaticBox(panel, label="Performance Recommendations")
        perf_sizer = wx.StaticBoxSizer(perf_box, wx.VERTICAL)
        
        self.perf_recommendations = wx.StaticText(panel, label="")
        perf_sizer.Add(self.perf_recommendations, 1, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(perf_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_button_panel(self, parent) -> wx.BoxSizer:
        """Create button panel"""
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Help button
        help_btn = wx.Button(parent, label="Help")
        help_btn.Bind(wx.EVT_BUTTON, self._on_help)
        sizer.Add(help_btn, 0, wx.RIGHT, 5)
        
        # Test GPU button
        test_btn = wx.Button(parent, label="Test GPU")
        test_btn.Bind(wx.EVT_BUTTON, self._on_test_gpu)
        sizer.Add(test_btn, 0, wx.RIGHT, 10)
        
        sizer.AddStretchSpacer()
        
        # Cancel button
        cancel_btn = wx.Button(parent, wx.ID_CANCEL, label="Cancel")
        sizer.Add(cancel_btn, 0, wx.RIGHT, 5)
        
        # Start routing button
        self.start_btn = wx.Button(parent, wx.ID_OK, label="Start GPU Routing")
        self.start_btn.SetDefault()
        sizer.Add(self.start_btn, 0)
        
        return sizer
    
    def _bind_events(self):
        """Bind control events"""
        # Update estimates when settings change
        self.pitch_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_setting_changed)
        self.layers_ctrl.Bind(wx.EVT_SPINCTRL, self._on_setting_changed)
        self.iterations_ctrl.Bind(wx.EVT_SPINCTRL, self._on_setting_changed)
        self.batch_ctrl.Bind(wx.EVT_SPINCTRL, self._on_setting_changed)
        self.quality_slider.Bind(wx.EVT_SLIDER, self._on_setting_changed)
    
    def _load_gpu_info(self):
        """Load GPU information in background thread"""
        def load_info():
            try:
                import cupy as cp
                
                device = cp.cuda.Device()
                attrs = device.attributes
                mem_info = device.mem_info
                
                gpu_info = {
                    'name': device.name,
                    'compute_capability': f"{attrs['major']}.{attrs['minor']}",
                    'multiprocessors': attrs['multiProcessorCount'],
                    'cuda_cores': attrs['multiProcessorCount'] * 128,  # Estimate
                    'memory_total_gb': mem_info[1] / (1024**3),
                    'memory_free_gb': mem_info[0] / (1024**3),
                    'cuda_version': cp.cuda.runtime.runtimeGetVersion()
                }
                
                wx.CallAfter(self._update_gpu_info, gpu_info, None)
                
            except Exception as e:
                wx.CallAfter(self._update_gpu_info, None, str(e))
        
        threading.Thread(target=load_info, daemon=True).start()
    
    def _update_gpu_info(self, gpu_info: Optional[Dict], error: Optional[str]):
        """Update GPU information display"""
        if error:
            status_text = (
                f"❌ GPU Error: {error}\n\n"
                f"Installation required:\n"
                f"1. NVIDIA GPU with CUDA support\n"
                f"2. CUDA Toolkit 11.8+ or 12.x\n"
                f"3. Install CuPy:\n"
                f"   pip install cupy-cuda12x\n\n"
                f"For help, visit:\n"
                f"https://docs.cupy.dev/en/stable/install.html"
            )
            self.start_btn.Enable(False)
        else:
            self.gpu_info = gpu_info
            status_text = (
                f"✅ GPU Ready\n\n"
                f"Device: {gpu_info['name']}\n"
                f"Compute Capability: {gpu_info['compute_capability']}\n"
                f"CUDA Cores: ~{gpu_info['cuda_cores']:,}\n"
                f"Memory: {gpu_info['memory_total_gb']:.1f} GB total\n"
                f"Available: {gpu_info['memory_free_gb']:.1f} GB\n"
                f"CUDA Version: {gpu_info['cuda_version']}"
            )
            self.start_btn.Enable(True)
        
        self.gpu_status_text.SetLabel(status_text)
        
        # Update performance recommendations
        self._update_performance_recommendations()
    
    def _update_performance_recommendations(self):
        """Update performance recommendations"""
        if not self.gpu_info:
            return
        
        net_count = self.board_stats.get('signal_nets', 1000)  # Estimate if unknown
        memory_gb = self.gpu_info['memory_free_gb']
        cuda_cores = self.gpu_info['cuda_cores']
        
        # Generate recommendations
        recommendations = []
        
        if net_count < 500:
            recommendations.append("✅ Small board - excellent performance expected")
        elif net_count < 2000:
            recommendations.append("✅ Medium board - good performance expected")
        elif net_count < 8000:
            recommendations.append("⚠️  Large board - may take several minutes")
        else:
            recommendations.append("🔥 Extreme board - high-end GPU recommended")
        
        if memory_gb < 4:
            recommendations.append("⚠️  Low GPU memory - reduce grid pitch or batch size")
        elif memory_gb > 8:
            recommendations.append("✅ Plenty of GPU memory - can use fine grid pitch")
        
        if cuda_cores < 2000:
            recommendations.append("⚠️  Older GPU - consider reducing batch size")
        elif cuda_cores > 5000:
            recommendations.append("🚀 High-performance GPU - use large batch sizes")
        
        # Grid recommendations
        if net_count > 2000:
            recommendations.append("💡 For large boards, try 0.15mm grid pitch")
        else:
            recommendations.append("💡 For best quality, try 0.05-0.1mm grid pitch")
        
        rec_text = "\n".join(recommendations)
        self.perf_recommendations.SetLabel(rec_text)
    
    def _update_performance_estimate(self):
        """Update performance estimate"""
        if not self.gpu_info:
            self.perf_text.SetLabel("GPU information needed for performance estimate")
            return
        
        net_count = self.board_stats.get('signal_nets', 1000)
        iterations = self.iterations_ctrl.GetValue()
        batch_size = self.batch_ctrl.GetValue()
        
        # Rough performance estimate
        cuda_cores = self.gpu_info['cuda_cores']
        base_rate = min(100, cuda_cores / 30)  # Nets per second estimate
        
        # Adjust for settings
        quality_factor = self.quality_slider.GetValue() / 3.0
        rate = base_rate / quality_factor
        
        estimated_time = (net_count * iterations) / (rate * batch_size)
        estimated_time = max(1, estimated_time)  # Minimum 1 second
        
        if estimated_time < 60:
            time_str = f"{estimated_time:.0f} seconds"
        elif estimated_time < 3600:
            time_str = f"{estimated_time/60:.1f} minutes"
        else:
            time_str = f"{estimated_time/3600:.1f} hours"
        
        self.perf_text.SetLabel(f"Estimated routing time: {time_str}")
    
    def _on_setting_changed(self, event):
        """Handle setting change"""
        self._update_performance_estimate()
        event.Skip()
    
    def _on_refresh_gpu(self, event):
        """Handle refresh GPU button"""
        self.gpu_status_text.SetLabel("Refreshing GPU information...")
        self._load_gpu_info()
    
    def _on_test_gpu(self, event):
        """Handle test GPU button"""
        def test_gpu():
            try:
                import cupy as cp
                import time
                
                # Simple GPU performance test
                start_time = time.time()
                
                # Create test arrays
                a = cp.random.random((1000, 1000), dtype=cp.float32)
                b = cp.random.random((1000, 1000), dtype=cp.float32)
                
                # Matrix multiplication benchmark
                for _ in range(10):
                    c = cp.dot(a, b)
                    cp.cuda.Stream.null.synchronize()
                
                elapsed = time.time() - start_time
                gflops = (10 * 2 * 1000**3) / (elapsed * 1e9)
                
                wx.CallAfter(self._show_test_results, gflops, None)
                
            except Exception as e:
                wx.CallAfter(self._show_test_results, None, str(e))
        
        threading.Thread(target=test_gpu, daemon=True).start()
        
        # Show progress
        self.gpu_status_text.SetLabel("Running GPU performance test...")
    
    def _show_test_results(self, gflops: Optional[float], error: Optional[str]):
        """Show GPU test results"""
        if error:
            message = f"GPU test failed: {error}"
            wx.MessageBox(message, "GPU Test Failed", wx.OK | wx.ICON_ERROR)
        else:
            message = (
                f"GPU Performance Test Results\n\n"
                f"Performance: {gflops:.1f} GFLOPS\n\n"
                f"Rating: {self._get_performance_rating(gflops)}\n\n"
                f"This gives a rough indication of GPU compute performance."
            )
            wx.MessageBox(message, "GPU Test Results", wx.OK | wx.ICON_INFORMATION)
        
        # Restore GPU info
        self._load_gpu_info()
    
    def _get_performance_rating(self, gflops: float) -> str:
        """Get performance rating from GFLOPS"""
        if gflops > 500:
            return "🔥 Excellent (High-end GPU)"
        elif gflops > 200:
            return "✅ Very Good (Gaming GPU)"
        elif gflops > 100:
            return "👍 Good (Mid-range GPU)"
        elif gflops > 50:
            return "⚠️ Acceptable (Entry-level GPU)"
        else:
            return "🐌 Poor (Upgrade recommended)"
    
    def _on_help(self, event):
        """Show help dialog"""
        help_dialog = OrthoRouteHelpDialog(self)
        help_dialog.ShowModal()
        help_dialog.Destroy()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'grid_pitch_mm': 0.1,
            'max_layers': 4,
            'max_iterations': 20,
            'batch_size': 256,
            'tile_size': 64,
            'congestion_factor': 1.5,
            'via_cost': 10,
            'direction_cost': 2,
            'trace_cost': 1,
            'gpu_id': 0,
            'memory_limit_gb': 0,
            'net_pattern': '',
            'min_pins': 2,
            'max_pins': 100,
            'skip_power_nets': True,
            'skip_routed_nets': True,
            'skip_locked_nets': True,
            'priority_mode': 0,
            'save_results': True,
            'show_visualization': False,
            'verbose': False
        }
    
    def get_config(self) -> Dict:
        """Get current configuration from dialog"""
        config = {}
        
        # Basic settings
        config['grid_pitch_mm'] = self.pitch_ctrl.GetValue()
        config['max_layers'] = self.layers_ctrl.GetValue()
        config['max_iterations'] = self.iterations_ctrl.GetValue()
        config['batch_size'] = self.batch_ctrl.GetValue()
        config['tile_size'] = self.tile_ctrl.GetValue()
        
        # Quality adjustment
        quality = self.quality_slider.GetValue()
        if quality == 1:  # Fast
            config['max_iterations'] = max(5, config['max_iterations'] // 2)
            config['batch_size'] = min(1024, config['batch_size'] * 2)
        elif quality == 5:  # High quality
            config['max_iterations'] = config['max_iterations'] * 2
            config['grid_pitch_mm'] = max(0.025, config['grid_pitch_mm'] / 2)
        
        # Advanced settings
        config['congestion_factor'] = self.congestion_ctrl.GetValue()
        config['via_cost'] = self.via_cost_ctrl.GetValue()
        config['direction_cost'] = self.direction_cost_ctrl.GetValue()
        config['trace_cost'] = self.trace_cost_ctrl.GetValue()
        config['gpu_id'] = self.gpu_id_ctrl.GetValue()
        config['memory_limit_gb'] = self.memory_limit_ctrl.GetValue()
        
        # Filtering settings
        config['net_pattern'] = self.net_pattern_ctrl.GetValue()
        config['min_pins'] = self.min_pins_ctrl.GetValue()
        config['max_pins'] = self.max_pins_ctrl.GetValue()
        config['skip_power_nets'] = self.skip_power_cb.GetValue()
        config['skip_routed_nets'] = self.skip_routed_cb.GetValue()
        config['skip_locked_nets'] = self.skip_locked_cb.GetValue()
        config['priority_mode'] = self.priority_choice.GetSelection()
        
        # Output settings
        config['save_results'] = self.save_results_cb.GetValue()
        config['show_visualization'] = self.show_visualization_cb.GetValue()
        config['verbose'] = self.verbose_cb.GetValue()
        
        return config


class OrthoRouteProgressDialog(wx.Dialog):
    """Progress dialog for GPU routing operation"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute GPU Routing Progress",
                         style=wx.DEFAULT_DIALOG_STYLE)
        
        self.status = RoutingStatus.IDLE
        self.cancelled = False
        self.start_time = None
        
        self._create_ui()
        self._bind_events()
        
        self.SetSize((500, 400))
        self.CenterOnParent()
    
    def _create_ui(self):
        """Create progress dialog UI"""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="GPU Routing in Progress")
        title_font = title.GetFont()
        title_font.PointSize += 2
        title_font = title_font.Bold()
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Status
        self.status_text = wx.StaticText(panel, label="Initializing...")
        sizer.Add(self.status_text, 0, wx.ALL | wx.CENTER, 5)
        
        # Progress bar
        self.progress_bar = wx.Gauge(panel, range=100)
        sizer.Add(self.progress_bar, 0, wx.EXPAND | wx.ALL, 10)
        
        # Progress details
        self.details_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.details_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sizer.Add(self.details_text, 1, wx.EXPAND | wx.ALL, 10)
        
        # Statistics
        stats_box = wx.StaticBox(panel, label="Current Statistics")
        stats_sizer = wx.StaticBoxSizer(stats_box, wx.VERTICAL)
        
        self.stats_text = wx.StaticText(panel, label="")
        stats_sizer.Add(self.stats_text, 0, wx.ALL, 5)
        
        sizer.Add(stats_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, label="Cancel")
        button_sizer.Add(self.cancel_btn, 0)
        
        sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        panel.SetSizer(sizer)
    
    def _bind_events(self):
        """Bind events"""
        self.Bind(EVT_UPDATE_PROGRESS, self._on_update_progress)
        self.Bind(EVT_ROUTING_COMPLETE, self._on_routing_complete)
        self.Bind(EVT_ERROR, self._on_error)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel)
    
    def start_routing(self, routing_func: Callable, *args, **kwargs):
        """Start routing operation in background thread"""
        self.start_time = time.time()
        
        def routing_thread():
            try:
                result = routing_func(*args, **kwargs)
                wx.PostEvent(self, RoutingCompleteEvent(result=result))
            except Exception as e:
                wx.PostEvent(self, ErrorEvent(error=str(e)))
        
        self.routing_thread = threading.Thread(target=routing_thread, daemon=True)
        self.routing_thread.start()
    
    def update_progress(self, update: ProgressUpdate):
        """Update progress from external thread"""
        wx.PostEvent(self, UpdateProgressEvent(update=update))
    
    def _on_update_progress(self, event):
        """Handle progress update event"""
        update = event.update
        self.status = update.status
        
        # Update status text
        self.status_text.SetLabel(update.message)
        
        # Update progress bar
        self.progress_bar.SetValue(update.progress)
        
        # Add to details log
        timestamp = time.strftime("%H:%M:%S")
        detail_line = f"[{timestamp}] {update.message}\n"
        self.details_text.AppendText(detail_line)
        
        # Update statistics if provided
        if update.details:
            self._update_statistics(update.details)
    
    def _update_statistics(self, details: Dict):
        """Update statistics display"""
        stats_lines = []
        
        if 'nets_completed' in details:
            stats_lines.append(f"Nets completed: {details['nets_completed']}")
        
        if 'nets_total' in details:
            stats_lines.append(f"Total nets: {details['nets_total']}")
        
        if 'current_iteration' in details:
            stats_lines.append(f"Iteration: {details['current_iteration']}")
        
        if 'success_rate' in details:
            stats_lines.append(f"Success rate: {details['success_rate']:.1f}%")
        
        if 'routing_time' in details:
            stats_lines.append(f"Elapsed: {details['routing_time']:.1f}s")
        
        if 'nets_per_second' in details:
            stats_lines.append(f"Rate: {details['nets_per_second']:.1f} nets/s")
        
        # Estimate remaining time
        if (self.start_time and 'nets_completed' in details and 
            'nets_total' in details and details['nets_completed'] > 0):
            
            elapsed = time.time() - self.start_time
            progress_ratio = details['nets_completed'] / details['nets_total']
            if progress_ratio > 0:
                estimated_total = elapsed / progress_ratio
                remaining = estimated_total - elapsed
                if remaining > 0:
                    stats_lines.append(f"ETA: {remaining:.0f}s")
        
        self.stats_text.SetLabel("\n".join(stats_lines))
    
    def _on_routing_complete(self, event):
        """Handle routing completion"""
        result = event.result
        
        # Update UI to show completion
        self.progress_bar.SetValue(100)
        self.status_text.SetLabel("Routing completed successfully!")
        
        # Change cancel button to close
        self.cancel_btn.SetLabel("Close")
        
        # Add completion message
        timestamp = time.strftime("%H:%M:%S")
        self.details_text.AppendText(f"[{timestamp}] Routing completed!\n")
        
        # Store result for parent dialog
        self.routing_result = result
        
        # Auto-close after delay (optional)
        if hasattr(self, 'auto_close') and self.auto_close:
            wx.CallLater(2000, self.EndModal, wx.ID_OK)
    
    def _on_error(self, event):
        """Handle routing error"""
        error = event.error
        
        self.status_text.SetLabel(f"Error: {error}")
        self.cancel_btn.SetLabel("Close")
        
        timestamp = time.strftime("%H:%M:%S")
        self.details_text.AppendText(f"[{timestamp}] ERROR: {error}\n")
        
        wx.MessageBox(f"Routing failed: {error}", "OrthoRoute Error", 
                     wx.OK | wx.ICON_ERROR)
    
    def _on_cancel(self, event):
        """Handle cancel button"""
        if self.status in [RoutingStatus.COMPLETE, RoutingStatus.ERROR]:
            self.EndModal(wx.ID_CANCEL)
        else:
            # Confirm cancellation
            result = wx.MessageBox(
                "Are you sure you want to cancel the routing operation?",
                "Cancel Routing",
                wx.YES_NO | wx.ICON_QUESTION
            )
            
            if result == wx.YES:
                self.cancelled = True
                self.status_text.SetLabel("Cancelling...")
                self.cancel_btn.Enable(False)
                # Note: Actual cancellation would need to be implemented in the routing engine
    
    def _on_close(self, event):
        """Handle window close"""
        if self.status not in [RoutingStatus.COMPLETE, RoutingStatus.ERROR]:
            self._on_cancel(event)
        else:
            event.Skip()


class OrthoRouteResultsDialog(wx.Dialog):
    """Results dialog showing routing statistics and options"""
    
    def __init__(self, parent, results: Dict):
        super().__init__(parent, title="OrthoRoute Results",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.results = results
        self._create_ui()
        
        self.SetSize((600, 500))
        self.CenterOnParent()
    
    def _create_ui(self):
        """Create results dialog UI"""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title with success/failure icon
        success = self.results.get('success', False)
        icon = "✅" if success else "❌"
        title = wx.StaticText(panel, label=f"{icon} Routing Results")
        title_font = title.GetFont()
        title_font.PointSize += 4
        title_font = title_font.Bold()
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Create notebook for detailed results
        notebook = wx.Notebook(panel)
        
        # Summary tab
        summary_panel = self._create_summary_panel(notebook)
        notebook.AddPage(summary_panel, "Summary")
        
        # Detailed statistics tab
        stats_panel = self._create_statistics_panel(notebook)
        notebook.AddPage(stats_panel, "Detailed Stats")
        
        # Quality analysis tab
        quality_panel = self._create_quality_panel(notebook)
        notebook.AddPage(quality_panel, "Quality Analysis")
        
        sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 10)
        
        # Action buttons
        button_sizer = self._create_action_buttons(panel)
        sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        panel.SetSizer(sizer)
    
    def _create_summary_panel(self, parent) -> wx.Panel:
        """Create summary results panel"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        if not self.results.get('success', False):
            # Error summary
            error_text = self.results.get('error', 'Unknown error')
            error_label = wx.StaticText(panel, label=f"Routing failed: {error_text}")
            error_label.SetForegroundColour(wx.Colour(255, 0, 0))
            sizer.Add(error_label, 0, wx.ALL, 10)
            
            panel.SetSizer(sizer)
            return panel
        
        stats = self.results.get('stats', {})
        
        # Success metrics
        success_box = wx.StaticBox(panel, label="Success Metrics")
        success_sizer = wx.StaticBoxSizer(success_box, wx.VERTICAL)
        
        success_text = (
            f"Total nets: {stats.get('total_nets', 0)}\n"
            f"Successfully routed: {stats.get('successful_nets', 0)}\n"
            f"Failed to route: {stats.get('failed_nets', 0)}\n"
            f"Success rate: {stats.get('success_rate', 0):.1f}%"
        )
        
        success_label = wx.StaticText(panel, label=success_text)
        success_sizer.Add(success_label, 0, wx.ALL, 5)
        sizer.Add(success_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Performance metrics
        perf_box = wx.StaticBox(panel, label="Performance Metrics")
        perf_sizer = wx.StaticBoxSizer(perf_box, wx.VERTICAL)
        
        perf_text = (
            f"Routing time: {stats.get('routing_time_seconds', 0):.2f} seconds\n"
            f"Processing rate: {stats.get('nets_per_second', 0):.1f} nets/second\n"
            f"Total execution time: {stats.get('total_execution_time', 0):.2f} seconds"
        )
        
        perf_label = wx.StaticText(panel, label=perf_text)
        perf_sizer.Add(perf_label, 0, wx.ALL, 5)
        sizer.Add(perf_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Quality metrics
        quality_box = wx.StaticBox(panel, label="Quality Metrics")
        quality_sizer = wx.StaticBoxSizer(quality_box, wx.VERTICAL)
        
        total_length = stats.get('total_length_mm', 0)
        total_vias = stats.get('total_vias', 0)
        successful_nets = stats.get('successful_nets', 1)
        
        quality_text = (
            f"Total wire length: {total_length:.1f} mm\n"
            f"Total vias: {total_vias}\n"
            f"Average length per net: {total_length / successful_nets:.2f} mm\n"
            f"Average vias per net: {total_vias / successful_nets:.1f}"
        )
        
        quality_label = wx.StaticText(panel, label=quality_text)
        quality_sizer.Add(quality_label, 0, wx.ALL, 5)
        sizer.Add(quality_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_statistics_panel(self, parent) -> wx.Panel:
        """Create detailed statistics panel"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Statistics text control
        stats_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        stats_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # Format detailed statistics
        stats_content = self._format_detailed_statistics()
        stats_text.SetValue(stats_content)
        
        sizer.Add(stats_text, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_quality_panel(self, parent) -> wx.Panel:
        """Create quality analysis panel"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Quality rating
        rating = self._calculate_quality_rating()
        rating_text = wx.StaticText(panel, label=f"Overall Quality Rating: {rating}")
        rating_font = rating_text.GetFont()
        rating_font.PointSize += 2
        rating_font = rating_font.Bold()
        rating_text.SetFont(rating_font)
        sizer.Add(rating_text, 0, wx.ALL | wx.CENTER, 10)
        
        # Quality analysis
        analysis = self._generate_quality_analysis()
        analysis_text = wx.StaticText(panel, label=analysis)
        sizer.Add(analysis_text, 1, wx.EXPAND | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_action_buttons(self, parent) -> wx.BoxSizer:
        """Create action buttons"""
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Save results button
        save_btn = wx.Button(parent, label="Save Results")
        save_btn.Bind(wx.EVT_BUTTON, self._on_save_results)
        sizer.Add(save_btn, 0, wx.RIGHT, 5)
        
        # View DRC button (if applicable)
        if self._has_violations():
            drc_btn = wx.Button(parent, label="View DRC Report")
            drc_btn.Bind(wx.EVT_BUTTON, self._on_view_drc)
            sizer.Add(drc_btn, 0, wx.RIGHT, 10)
        
        sizer.AddStretchSpacer()
        
        # Close button
        close_btn = wx.Button(parent, wx.ID_OK, label="Close")
        close_btn.SetDefault()
        sizer.Add(close_btn, 0)
        
        return sizer
    
    def _format_detailed_statistics(self) -> str:
        """Format detailed statistics for display"""
        if not self.results.get('success', False):
            return f"Routing failed: {self.results.get('error', 'Unknown error')}"
        
        stats = self.results.get('stats', {})
        
        output = []
        output.append("ORTHOROUTE DETAILED STATISTICS")
        output.append("=" * 50)
        output.append("")
        
        # Success metrics
        output.append("SUCCESS METRICS:")
        output.append(f"  Total nets processed: {stats.get('total_nets', 0)}")
        output.append(f"  Successfully routed:  {stats.get('successful_nets', 0)}")
        output.append(f"  Failed to route:      {stats.get('failed_nets', 0)}")
        output.append(f"  Success rate:         {stats.get('success_rate', 0):.1f}%")
        output.append("")
        
        # Performance metrics
        output.append("PERFORMANCE METRICS:")
        output.append(f"  Routing time:         {stats.get('routing_time_seconds', 0):.2f} seconds")
        output.append(f"  Total execution:      {stats.get('total_execution_time', 0):.2f} seconds")
        output.append(f"  Processing rate:      {stats.get('nets_per_second', 0):.1f} nets/second")
        output.append("")
        
        # Quality metrics
        output.append("QUALITY METRICS:")
        output.append(f"  Total wire length:    {stats.get('total_length_mm', 0):.1f} mm")
        output.append(f"  Total vias:           {stats.get('total_vias', 0)}")
        
        successful_nets = stats.get('successful_nets', 1)
        if successful_nets > 0:
            avg_length = stats.get('total_length_mm', 0) / successful_nets
            avg_vias = stats.get('total_vias', 0) / successful_nets
            output.append(f"  Avg length per net:   {avg_length:.2f} mm")
            output.append(f"  Avg vias per net:     {avg_vias:.1f}")
        
        output.append("")
        
        # GPU metrics (if available)
        if 'gpu_memory_mb' in stats:
            output.append("GPU METRICS:")
            output.append(f"  GPU memory used:      {stats['gpu_memory_mb']:.1f} MB")
            output.append("")
        
        return "\n".join(output)
    
    def _calculate_quality_rating(self) -> str:
        """Calculate overall quality rating"""
        if not self.results.get('success', False):
            return "❌ FAILED"
        
        stats = self.results.get('stats', {})
        success_rate = stats.get('success_rate', 0)
        
        if success_rate >= 98:
            return "🏆 EXCELLENT"
        elif success_rate >= 90:
            return "✅ VERY GOOD"
        elif success_rate >= 80:
            return "👍 GOOD"
        elif success_rate >= 60:
            return "⚠️ ACCEPTABLE"
        else:
            return "❌ POOR"
    
    def _generate_quality_analysis(self) -> str:
        """Generate quality analysis text"""
        if not self.results.get('success', False):
            return "Routing failed - no quality analysis available."
        
        stats = self.results.get('stats', {})
        success_rate = stats.get('success_rate', 0)
        
        analysis = []
        
        # Success rate analysis
        if success_rate >= 95:
            analysis.append("🎯 Excellent success rate - routing quality is very high")
        elif success_rate >= 85:
            analysis.append("✅ Good success rate - most nets routed successfully")
        elif success_rate >= 70:
            analysis.append("⚠️ Moderate success rate - some nets may need manual routing")
        else:
            analysis.append("❌ Low success rate - consider adjusting routing parameters")
        
        # Performance analysis
        nets_per_sec = stats.get('nets_per_second', 0)
        if nets_per_sec > 100:
            analysis.append("🚀 Excellent performance - GPU acceleration working well")
        elif nets_per_sec > 50:
            analysis.append("✅ Good performance - routing completed efficiently")
        elif nets_per_sec > 20:
            analysis.append("👍 Acceptable performance")
        else:
            analysis.append("🐌 Slow performance - consider optimizing settings")
        
        # Via usage analysis
        total_vias = stats.get('total_vias', 0)
        successful_nets = stats.get('successful_nets', 1)
        avg_vias = total_vias / successful_nets if successful_nets > 0 else 0
        
        if avg_vias < 1.5:
            analysis.append("✅ Low via count - good for manufacturing")
        elif avg_vias < 3.0:
            analysis.append("👍 Moderate via count - acceptable")
        else:
            analysis.append("⚠️ High via count - consider layer optimization")
        
        # Recommendations
        analysis.append("")
        analysis.append("RECOMMENDATIONS:")
        
        if success_rate < 90:
            analysis.append("• Consider reducing grid pitch for better routing")
            analysis.append("• Increase max iterations for complex boards")
            analysis.append("• Check for design rule violations")
        
        if avg_vias > 2.5:
            analysis.append("• Consider optimizing layer assignment")
            analysis.append("• Review via costs in algorithm settings")
        
        if nets_per_sec < 30:
            analysis.append("• Consider upgrading GPU for better performance")
            analysis.append("• Reduce batch size if memory limited")
        
        return "\n".join(analysis)
    
    def _has_violations(self) -> bool:
        """Check if there are DRC violations to display"""
        return 'violations' in self.results and len(self.results['violations']) > 0
    
    def _on_save_results(self, event):
        """Handle save results button"""
        with wx.FileDialog(self, "Save routing results",
                          wildcard="JSON files (*.json)|*.json",
                          style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dialog:
            
            if dialog.ShowModal() == wx.ID_OK:
                try:
                    with open(dialog.GetPath(), 'w') as f:
                        json.dump(self.results, f, indent=2)
                    
                    wx.MessageBox("Results saved successfully!", "Save Results", 
                                 wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Failed to save results: {e}", "Save Error", 
                                 wx.OK | wx.ICON_ERROR)
    
    def _on_view_drc(self, event):
        """Handle view DRC button"""
        violations = self.results.get('violations', [])
        drc_dialog = DRCReportDialog(self, violations)
        drc_dialog.ShowModal()
        drc_dialog.Destroy()


class DRCReportDialog(wx.Dialog):
    """Dialog for displaying design rule check violations"""
    
    def __init__(self, parent, violations: List[Dict]):
        super().__init__(parent, title="Design Rule Check Report",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.violations = violations
        self._create_ui()
        
        self.SetSize((700, 500))
        self.CenterOnParent()
    
    def _create_ui(self):
        """Create DRC report UI"""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label=f"Design Rule Violations ({len(self.violations)})")
        title_font = title.GetFont()
        title_font.PointSize += 2
        title_font = title_font.Bold()
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Violations list
        self.violations_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.violations_list.AppendColumn("Type", width=120)
        self.violations_list.AppendColumn("Severity", width=80)
        self.violations_list.AppendColumn("Net", width=100)
        self.violations_list.AppendColumn("Location", width=120)
        self.violations_list.AppendColumn("Description", width=250)
        
        # Populate violations
        for i, violation in enumerate(self.violations):
            index = self.violations_list.InsertItem(i, violation.get('type', ''))
            self.violations_list.SetItem(index, 1, violation.get('severity', ''))
            self.violations_list.SetItem(index, 2, violation.get('net_name', ''))
            location = violation.get('location', {})
            loc_str = f"({location.get('x', 0)}, {location.get('y', 0)}, L{location.get('layer', 0)})"
            self.violations_list.SetItem(index, 3, loc_str)
            self.violations_list.SetItem(index, 4, violation.get('description', ''))
        
        sizer.Add(self.violations_list, 1, wx.EXPAND | wx.ALL, 10)
        
        # Details panel
        details_box = wx.StaticBox(panel, label="Violation Details")
        details_sizer = wx.StaticBoxSizer(details_box, wx.VERTICAL)
        
        self.details_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        details_sizer.Add(self.details_text, 1, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(details_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        export_btn = wx.Button(panel, label="Export Report")
        export_btn.Bind(wx.EVT_BUTTON, self._on_export_report)
        button_sizer.Add(export_btn, 0, wx.RIGHT, 10)
        
        button_sizer.AddStretchSpacer()
        
        close_btn = wx.Button(panel, wx.ID_OK, label="Close")
        button_sizer.Add(close_btn, 0)
        
        sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Bind events
        self.violations_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_violation_selected)
        
        panel.SetSizer(sizer)
    
    def _on_violation_selected(self, event):
        """Handle violation selection"""
        index = event.GetIndex()
        if 0 <= index < len(self.violations):
            violation = self.violations[index]
            
            details = (
                f"Type: {violation.get('type', 'Unknown')}\n"
                f"Severity: {violation.get('severity', 'Unknown')}\n"
                f"Net: {violation.get('net_name', 'Unknown')}\n"
                f"Location: {violation.get('location', {})}\n\n"
                f"Description:\n{violation.get('description', '')}\n\n"
                f"Measured Value: {violation.get('measured_value', 'N/A')}\n"
                f"Required Value: {violation.get('required_value', 'N/A')}\n\n"
                f"Suggestion:\n{violation.get('suggestion', 'No suggestion available')}"
            )
            
            self.details_text.SetValue(details)
    
    def _on_export_report(self, event):
        """Export DRC report to file"""
        with wx.FileDialog(self, "Export DRC Report",
                          wildcard="Text files (*.txt)|*.txt|CSV files (*.csv)|*.csv",
                          style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dialog:
            
            if dialog.ShowModal() == wx.ID_OK:
                try:
                    filepath = dialog.GetPath()
                    if filepath.endswith('.csv'):
                        self._export_csv(filepath)
                    else:
                        self._export_text(filepath)
                    
                    wx.MessageBox("DRC report exported successfully!", "Export Complete", 
                                 wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Failed to export report: {e}", "Export Error", 
                                 wx.OK | wx.ICON_ERROR)
    
    def _export_text(self, filepath: str):
        """Export DRC report as text file"""
        with open(filepath, 'w') as f:
            f.write("ORTHOROUTE DESIGN RULE CHECK REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Violations: {len(self.violations)}\n\n")
            
            # Group by severity
            severity_counts = {}
            for violation in self.violations:
                severity = violation.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            f.write("Violations by Severity:\n")
            for severity, count in severity_counts.items():
                f.write(f"  {severity.upper()}: {count}\n")
            f.write("\n")
            
            # List all violations
            f.write("DETAILED VIOLATIONS:\n")
            f.write("-" * 30 + "\n\n")
            
            for i, violation in enumerate(self.violations):
                f.write(f"Violation {i+1}:\n")
                f.write(f"  Type: {violation.get('type', 'Unknown')}\n")
                f.write(f"  Severity: {violation.get('severity', 'Unknown')}\n")
                f.write(f"  Net: {violation.get('net_name', 'Unknown')}\n")
                location = violation.get('location', {})
                f.write(f"  Location: ({location.get('x', 0)}, {location.get('y', 0)}, L{location.get('layer', 0)})\n")
                f.write(f"  Description: {violation.get('description', '')}\n")
                f.write(f"  Measured: {violation.get('measured_value', 'N/A')}\n")
                f.write(f"  Required: {violation.get('required_value', 'N/A')}\n")
                if violation.get('suggestion'):
                    f.write(f"  Suggestion: {violation.get('suggestion')}\n")
                f.write("\n")
    
    def _export_csv(self, filepath: str):
        """Export DRC report as CSV file"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Type', 'Severity', 'Net', 'X', 'Y', 'Layer', 
                           'Description', 'Measured', 'Required', 'Suggestion'])
            
            # Data
            for violation in self.violations:
                location = violation.get('location', {})
                writer.writerow([
                    violation.get('type', ''),
                    violation.get('severity', ''),
                    violation.get('net_name', ''),
                    location.get('x', 0),
                    location.get('y', 0),
                    location.get('layer', 0),
                    violation.get('description', ''),
                    violation.get('measured_value', ''),
                    violation.get('required_value', ''),
                    violation.get('suggestion', '')
                ])


class OrthoRouteHelpDialog(wx.Dialog):
    """Help dialog with usage instructions and tips"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Help",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self._create_ui()
        
        self.SetSize((800, 600))
        self.CenterOnParent()
    
    def _create_ui(self):
        """Create help dialog UI"""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="OrthoRoute GPU Autorouter Help")
        title_font = title.GetFont()
        title_font.PointSize += 4
        title_font = title_font.Bold()
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Help notebook
        notebook = wx.Notebook(panel)
        
        # Getting Started tab
        getting_started_panel = self._create_getting_started_panel(notebook)
        notebook.AddPage(getting_started_panel, "Getting Started")
        
        # Settings Guide tab
        settings_panel = self._create_settings_panel(notebook)
        notebook.AddPage(settings_panel, "Settings Guide")
        
        # Troubleshooting tab
        troubleshooting_panel = self._create_troubleshooting_panel(notebook)
        notebook.AddPage(troubleshooting_panel, "Troubleshooting")
        
        # Tips & Tricks tab
        tips_panel = self._create_tips_panel(notebook)
        notebook.AddPage(tips_panel, "Tips & Tricks")
        
        sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 10)
        
        # Close button
        close_btn = wx.Button(panel, wx.ID_OK, label="Close")
        sizer.Add(close_btn, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        panel.SetSizer(sizer)
    
    def _create_getting_started_panel(self, parent) -> wx.Panel:
        """Create getting started help panel"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        help_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_WORDWRAP)
        
        content = """
GETTING STARTED WITH ORTHOROUTE

OrthoRoute is a GPU-accelerated PCB autorouter that uses NVIDIA CUDA to route 
complex circuit boards much faster than traditional CPU-based routers.

BASIC WORKFLOW:
1. Open your PCB design in KiCad PCB Editor
2. Click the OrthoRoute button in the toolbar
3. Configure routing settings in the dialog
4. Click "Start GPU Routing"
5. Review results and apply routes to your board

BEFORE YOU START:
• Ensure your board design is complete with all components placed
• Define net classes for different signal types (power, high-speed, etc.)
• Set appropriate design rules in KiCad
• Save your project before routing (recommended)

SYSTEM REQUIREMENTS:
• NVIDIA GPU with CUDA Compute Capability 7.5+
• CUDA Toolkit 11.8+ or 12.x installed
• At least 4GB GPU memory for medium boards
• CuPy Python package installed

FIRST TIME SETUP:
1. Install CUDA Toolkit from NVIDIA website
2. Install CuPy: pip install cupy-cuda12x
3. Restart KiCad and verify GPU detection in the plugin

The plugin will automatically detect your GPU and provide performance 
recommendations based on your board complexity.
        """
        
        help_text.SetValue(content.strip())
        sizer.Add(help_text, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_settings_panel(self, parent) -> wx.Panel:
        """Create settings guide panel"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        help_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_WORDWRAP)
        
        content = """
SETTINGS GUIDE

GRID SETTINGS:
• Grid Pitch: Controls routing resolution
  - 0.05mm: High quality, slower routing
  - 0.1mm: Good balance (recommended)
  - 0.2mm: Fast routing, lower quality
• Routing Layers: Number of layers available for routing

ROUTING SETTINGS:
• Max Iterations: More iterations = better results but slower
  - 10-15: Fast routing
  - 20-30: Balanced (recommended)
  - 50+: High quality, complex boards
• Batch Size: Number of nets processed simultaneously
  - 64-128: Memory limited GPUs
  - 256-512: Modern GPUs (recommended)
  - 1024+: High-end GPUs with lots of memory

QUALITY VS SPEED SLIDER:
1 - Fast: Optimized for speed, may miss some routes
2 - Balanced Fast: Good speed with decent quality
3 - Balanced: Good balance of speed and quality (default)
4 - High Quality: Slower but better results
5 - Maximum Quality: Best results, longest time

ADVANCED SETTINGS:
• Congestion Factor: How much contested areas are penalized
• Via Cost: Penalty for using vias (higher = fewer vias)
• Direction Change Cost: Penalty for changing trace direction
• GPU Device ID: Which GPU to use (0 for first GPU)

NET FILTERING:
• Use pattern matching to route specific nets only
• Skip power/ground nets if they need special routing
• Set pin count limits to focus on specific net types
        """
        
        help_text.SetValue(content.strip())
        sizer.Add(help_text, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_troubleshooting_panel(self, parent) -> wx.Panel:
        """Create troubleshooting help panel"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        help_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_WORDWRAP)
        
        content = """
TROUBLESHOOTING

COMMON ISSUES:

"CuPy not found" Error:
• Install CuPy: pip install cupy-cuda12x (for CUDA 12.x)
• Or: pip install cupy-cuda11x (for CUDA 11.x)
• Verify CUDA installation: nvcc --version
• Restart KiCad after installation

"No GPU detected" Error:
• Check if NVIDIA drivers are installed and up to date
• Verify GPU supports CUDA (GeForce GTX 1050 or newer)
• Check CUDA installation with: nvidia-smi
• Try different GPU device ID if you have multiple GPUs

"Out of memory" Error:
• Reduce batch size (try 128 or 64)
• Increase grid pitch (0.15mm or 0.2mm)
• Close other GPU applications
• Use memory limit setting

Poor Routing Results:
• Decrease grid pitch for better resolution
• Increase max iterations
• Check board for overlapping components
• Verify net classes are properly defined
• Review design rules for conflicts

Routing Takes Too Long:
• Increase grid pitch (0.15mm or 0.2mm)
• Reduce max iterations
• Use quality slider on "Fast" setting
• Increase batch size if you have GPU memory
• Filter nets to route only specific ones

Plugin Won't Load:
• Check KiCad console for error messages
• Verify plugin is in correct directory
• Check Python path and imports
• Try reinstalling the plugin
• Restart KiCad

No Routes Applied:
• Check import settings in advanced options
• Verify nets exist in the board
• Check for conflicting existing routes
• Review DRC violations that might block import
        """
        
        help_text.SetValue(content.strip())
        sizer.Add(help_text, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_tips_panel(self, parent) -> wx.Panel:
        """Create tips and tricks panel"""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        help_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_WORDWRAP)
        
        content = """
TIPS & TRICKS

PERFORMANCE OPTIMIZATION:
• Use the largest batch size your GPU memory allows
• Start with 0.1mm grid pitch, adjust based on results
• Route power nets manually, then use OrthoRoute for signals
• Use multiple passes: fast first, then high quality

QUALITY IMPROVEMENTS:
• Define proper net classes with appropriate widths
• Set up design rules before routing
• Use finer grid pitch (0.05-0.08mm) for dense boards
• Increase iterations for complex boards (30-50)

BOARD PREPARATION:
• Place all components before routing
• Lock critical traces that shouldn't be changed
• Define keepout areas for sensitive circuits
• Group related components to minimize trace lengths

GPU RECOMMENDATIONS BY BOARD SIZE:
• Small boards (<500 nets): GTX 1660 or better
• Medium boards (500-2000 nets): RTX 3060 or better  
• Large boards (2000-8000 nets): RTX 4070 or better
• Extreme boards (8000+ nets): RTX 4080/4090

WORKFLOW OPTIMIZATION:
1. Route power and ground manually with thick traces
2. Route critical signals (clocks, resets) manually
3. Use OrthoRoute for remaining signal nets
4. Run DRC check and fix violations
5. Optimize via placement and trace lengths

MEMORY MANAGEMENT:
• Monitor GPU memory usage during routing
• Close other applications using GPU
• Use tiled processing for very large boards
• Consider cloud GPU instances for extreme boards

DEBUGGING FAILED ROUTES:
• Check board statistics to understand complexity
• Look for isolated pins or unreachable areas
• Verify component courtyard clearances
• Review keepout zones that might block routing
• Use visualization to see routing progress

ADVANCED TECHNIQUES:
• Use different grid pitches for different net classes
• Adjust algorithm parameters for specific board types
• Save and reuse successful configurations
• Combine manual and automatic routing strategically
        """
        
        help_text.SetValue(content.strip())
        sizer.Add(help_text, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel


class OrthoRouteAboutDialog(wx.Dialog):
    """About dialog with version and license information"""
    
    def __init__(self, parent):
        super().__init__(parent, title="About OrthoRoute",
                         style=wx.DEFAULT_DIALOG_STYLE)
        
        self._create_ui()
        
        self.SetSize((500, 400))
        self.CenterOnParent()
    
    def _create_ui(self):
        """Create about dialog UI"""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Logo/Icon (if available)
        # logo = wx.StaticBitmap(panel, bitmap=wx.Bitmap("icon.png"))
        # sizer.Add(logo, 0, wx.ALL | wx.CENTER, 10)
        
        # Title
        title = wx.StaticText(panel, label="OrthoRoute")
        title_font = title.GetFont()
        title_font.PointSize += 8
        title_font = title_font.Bold()
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Subtitle
        subtitle = wx.StaticText(panel, label="GPU-Accelerated PCB Autorouter")
        subtitle_font = subtitle.GetFont()
        subtitle_font.PointSize += 2
        subtitle.SetFont(subtitle_font)
        sizer.Add(subtitle, 0, wx.ALL | wx.CENTER, 5)
        
        # Version info
        version_text = (
            "Version 0.1.0\n"
            "Build Date: 2024\n"
            "KiCad Plugin API: 7.0+"
        )
        version_label = wx.StaticText(panel, label=version_text)
        sizer.Add(version_label, 0, wx.ALL | wx.CENTER, 10)
        
        # Description
        description = (
            "The world's first GPU-accelerated PCB autorouter using NVIDIA CUDA.\n"
            "Route thousands of nets in seconds using massively parallel algorithms."
        )
        desc_label = wx.StaticText(panel, label=description)
        desc_label.Wrap(400)
        sizer.Add(desc_label, 0, wx.ALL | wx.CENTER, 10)
        
        # Credits
        credits_text = (
            "Created by: Brian Benchoff\n"
            "Built with: Python, CuPy, CUDA, KiCad API\n"
            "License: MIT License"
        )
        credits_label = wx.StaticText(panel, label=credits_text)
        sizer.Add(credits_label, 0, wx.ALL | wx.CENTER, 10)
        
        # Links
        links_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        github_btn = wx.Button(panel, label="GitHub")
        github_btn.Bind(wx.EVT_BUTTON, self._on_github)
        links_sizer.Add(github_btn, 0, wx.RIGHT, 5)
        
        docs_btn = wx.Button(panel, label="Documentation")
        docs_btn.Bind(wx.EVT_BUTTON, self._on_docs)
        links_sizer.Add(docs_btn, 0, wx.RIGHT, 5)
        
        support_btn = wx.Button(panel, label="Support")
        support_btn.Bind(wx.EVT_BUTTON, self._on_support)
        links_sizer.Add(support_btn, 0)
        
        sizer.Add(links_sizer, 0, wx.ALL | wx.CENTER, 10)
        
        # Close button
        close_btn = wx.Button(panel, wx.ID_OK, label="Close")
        close_btn.SetDefault()
        sizer.Add(close_btn, 0, wx.ALL | wx.CENTER, 10)
        
        panel.SetSizer(sizer)
    
    def _on_github(self, event):
        """Open GitHub repository"""
        import webbrowser
        webbrowser.open("https://github.com/bbenchoff/OrthoRoute")
    
    def _on_docs(self, event):
        """Open documentation"""
        import webbrowser
        webbrowser.open("https://github.com/bbenchoff/OrthoRoute/docs")
    
    def _on_support(self, event):
        """Open support page"""
        import webbrowser
        webbrowser.open("https://github.com/bbenchoff/OrthoRoute/issues")


# Utility functions for dialog management
def show_configuration_dialog(parent, board_stats: Dict = None) -> Optional[Dict]:
    """Show configuration dialog and return settings"""
    dialog = OrthoRouteConfigDialog(parent, board_stats)
    
    if dialog.ShowModal() == wx.ID_OK:
        config = dialog.get_config()
        dialog.Destroy()
        return config
    else:
        dialog.Destroy()
        return None

def show_progress_dialog(parent, routing_func: Callable, *args, **kwargs) -> Optional[Dict]:
    """Show progress dialog and execute routing function"""
    dialog = OrthoRouteProgressDialog(parent)
    
    # Start routing in background
    dialog.start_routing(routing_func, *args, **kwargs)
    
    result = None
    if dialog.ShowModal() == wx.ID_OK:
        result = getattr(dialog, 'routing_result', None)
    
    dialog.Destroy()
    return result

def show_results_dialog(parent, results: Dict):
    """Show routing results dialog"""
    dialog = OrthoRouteResultsDialog(parent, results)
    dialog.ShowModal()
    dialog.Destroy()

def show_help_dialog(parent):
    """Show help dialog"""
    dialog = OrthoRouteHelpDialog(parent)
    dialog.ShowModal()
    dialog.Destroy()

def show_about_dialog(parent):
    """Show about dialog"""
    dialog = OrthoRouteAboutDialog(parent)
    dialog.ShowModal()
    dialog.Destroy()

def create_progress_callback(dialog: OrthoRouteProgressDialog) -> Callable:
    """Create progress callback function for routing engine"""
    def progress_callback(status: str, progress: int, message: str, details: Dict = None):
        try:
            status_enum = RoutingStatus(status)
        except ValueError:
            status_enum = RoutingStatus.ROUTING
        
        update = ProgressUpdate(
            status=status_enum,
            progress=progress,
            message=message,
            details=details
        )
        
        dialog.update_progress(update)
    
    return progress_callback