#!/usr/bin/env python3
"""
Manhattan Router Visualization System

Provides real-time visualization of the Manhattan routing process with:
- Bright white highlighting of current routing traces
- Standard KiCad colors for completed routes
- Via visualization with proper layer colors
- Progress updates every 10 nets
- Grid overlay showing routing constraints
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# KiCad layer colors from theme
LAYER_COLORS = {
    'F.Cu': 'rgb(200, 52, 52)',      # Front copper - red
    'In1.Cu': 'rgb(127, 200, 127)',  # Inner 1 - green
    'In2.Cu': 'rgb(206, 125, 44)',   # Inner 2 - orange
    'In3.Cu': 'rgb(79, 203, 203)',   # Inner 3 - cyan
    'In4.Cu': 'rgb(219, 98, 139)',   # Inner 4 - pink
    'In5.Cu': 'rgb(167, 165, 198)',  # Inner 5 - purple
    'In6.Cu': 'rgb(40, 204, 217)',   # Inner 6 - light blue
    'In7.Cu': 'rgb(232, 178, 167)',  # Inner 7 - beige
    'In8.Cu': 'rgb(242, 237, 161)',  # Inner 8 - yellow
    'In9.Cu': 'rgb(141, 203, 129)',  # Inner 9 - light green
    'In10.Cu': 'rgb(237, 124, 51)',  # Inner 10 - orange
    'B.Cu': 'rgb(77, 127, 196)',     # Back copper - blue
}

# Colors for routing states
ROUTING_COLORS = {
    'active': 'rgb(255, 255, 255)',      # Bright white for active routing
    'completed': 'layer_default',        # Use standard layer color
    'failed': 'rgb(255, 0, 0)',          # Red for failed routes
    'ripped': 'rgb(255, 165, 0)',        # Orange for ripped routes
}

@dataclass
class VisualizationState:
    """Current state of the visualization"""
    current_net: Optional[str] = None
    nets_completed: int = 0
    nets_failed: int = 0
    total_nets: int = 0
    routing_active: bool = False
    last_update: float = 0.0

class ManhattanVisualizer:
    """Handles visualization of Manhattan routing progress"""
    
    def __init__(self, kicad_interface=None):
        """Initialize the visualizer"""
        self.kicad_interface = kicad_interface
        self.state = VisualizationState()
        self.active_tracks = []  # Tracks being routed (bright white)
        self.completed_tracks = []  # Completed tracks (standard colors)
        self.active_vias = []  # Vias being routed
        self.completed_vias = []  # Completed vias
        
        # Callbacks from router
        self.track_callback = self._handle_track_update
        self.via_callback = self._handle_via_update
        self.progress_callback = self._handle_progress_update
        
        logger.info("Manhattan visualizer initialized")
    
    def start_routing_visualization(self, total_nets: int):
        """Start visualization session"""
        self.state = VisualizationState(
            total_nets=total_nets,
            routing_active=True,
            last_update=time.time()
        )
        
        # Clear previous visualization
        self.active_tracks.clear()
        self.completed_tracks.clear()
        self.active_vias.clear()
        self.completed_vias.clear()
        
        logger.info(f"Started routing visualization for {total_nets} nets")
        self._update_display()
    
    def _handle_track_update(self, track_data: Dict):
        """Handle track update from router"""
        # Add to active tracks (bright white)
        track_viz = {
            'start': track_data['start'],
            'end': track_data['end'],
            'layer': track_data['layer'],
            'width': track_data['width'],
            'net': track_data['net'],
            'color': ROUTING_COLORS['active'],
            'timestamp': time.time()
        }
        self.active_tracks.append(track_viz)
        
        # Update display if needed
        self._maybe_update_display()
    
    def _handle_via_update(self, via_data: Dict):
        """Handle via update from router"""
        # Add to active vias
        via_viz = {
            'x': via_data['x'],
            'y': via_data['y'],
            'size': via_data['size'],
            'drill': via_data['drill'],
            'layers': via_data['layers'],
            'net': via_data['net'],
            'type': via_data['type'],
            'color': ROUTING_COLORS['active'],
            'timestamp': time.time()
        }
        self.active_vias.append(via_viz)
        
        # Update display if needed
        self._maybe_update_display()
    
    def _handle_progress_update(self, progress_data: Dict):
        """Handle progress update from router"""
        self.state.current_net = progress_data.get('current_net')
        self.state.nets_completed = progress_data.get('nets_completed', 0)
        self.state.nets_failed = progress_data.get('nets_failed', 0)
        
        # Move active tracks/vias to completed every 10 nets
        if self.state.nets_completed % 10 == 0 and self.state.nets_completed > 0:
            self._finalize_batch()
            logger.info(f"Progress: {self.state.nets_completed}/{self.state.total_nets} nets routed "
                       f"({self.state.nets_failed} failed)")
    
    def _finalize_batch(self):
        """Move active tracks/vias to completed with standard colors"""
        # Move active tracks to completed
        for track in self.active_tracks:
            track['color'] = self._get_layer_color(track['layer'])
            self.completed_tracks.append(track)
        self.active_tracks.clear()
        
        # Move active vias to completed  
        for via in self.active_vias:
            via['color'] = self._get_via_color(via['layers'])
            self.completed_vias.append(via)
        self.active_vias.clear()
        
        self._update_display()
    
    def _get_layer_color(self, layer_name: str) -> str:
        """Get standard color for a layer"""
        return LAYER_COLORS.get(layer_name, 'rgb(128, 128, 128)')
    
    def _get_via_color(self, layers: List[str]) -> str:
        """Get color for via based on layers it connects"""
        if 'F.Cu' in layers:
            return LAYER_COLORS['F.Cu']
        elif 'B.Cu' in layers:
            return LAYER_COLORS['B.Cu']
        else:
            # Use color of first inner layer
            for layer in layers:
                if layer in LAYER_COLORS:
                    return LAYER_COLORS[layer]
        return 'rgb(128, 128, 128)'  # Default gray
    
    def _maybe_update_display(self):
        """Update display if enough time has passed"""
        now = time.time()
        if now - self.state.last_update > 0.1:  # Update at most 10 times per second
            self._update_display()
            self.state.last_update = now
    
    def _update_display(self):
        """Update the visual display"""
        if not self.kicad_interface:
            return
        
        try:
            # Clear previous visualization
            self._clear_visualization()
            
            # Draw completed tracks (standard colors)
            for track in self.completed_tracks:
                self._draw_track(track)
            
            # Draw active tracks (bright white)
            for track in self.active_tracks:
                self._draw_track(track)
            
            # Draw completed vias (standard colors)
            for via in self.completed_vias:
                self._draw_via(via)
            
            # Draw active vias (bright white)
            for via in self.active_vias:
                self._draw_via(via)
            
            # Update progress display
            self._update_progress_display()
            
        except Exception as e:
            logger.warning(f"Failed to update display: {e}")
    
    def _clear_visualization(self):
        """Clear previous visualization elements"""
        if hasattr(self.kicad_interface, 'clear_visualization'):
            self.kicad_interface.clear_visualization()
    
    def _draw_track(self, track_data: Dict):
        """Draw a track on the display"""
        try:
            if hasattr(self.kicad_interface, 'draw_track'):
                self.kicad_interface.draw_track(
                    start=track_data['start'],
                    end=track_data['end'],
                    layer=track_data['layer'],
                    width=track_data['width'],
                    color=track_data['color']
                )
            elif hasattr(self.kicad_interface, 'create_track'):
                # Fallback to creating actual tracks
                self.kicad_interface.create_track(
                    track_data['start'][0], track_data['start'][1],
                    track_data['end'][0], track_data['end'][1],
                    track_data['layer'], track_data['width'],
                    track_data['net']
                )
        except Exception as e:
            logger.debug(f"Failed to draw track: {e}")
    
    def _draw_via(self, via_data: Dict):
        """Draw a via on the display"""
        try:
            if hasattr(self.kicad_interface, 'draw_via'):
                self.kicad_interface.draw_via(
                    x=via_data['x'],
                    y=via_data['y'],
                    size=via_data['size'],
                    drill=via_data['drill'],
                    layers=via_data['layers'],
                    color=via_data['color']
                )
            elif hasattr(self.kicad_interface, 'create_via'):
                # Fallback to creating actual vias
                layer_names = via_data['layers']
                from_layer = layer_names[0] if layer_names else 'F.Cu'
                to_layer = layer_names[1] if len(layer_names) > 1 else 'B.Cu'
                
                self.kicad_interface.create_via(
                    via_data['x'], via_data['y'],
                    via_data['size'], via_data['drill'],
                    from_layer, to_layer, via_data['net']
                )
        except Exception as e:
            logger.debug(f"Failed to draw via: {e}")
    
    def _update_progress_display(self):
        """Update progress indicators"""
        if hasattr(self.kicad_interface, 'update_progress'):
            progress_info = {
                'current_net': self.state.current_net,
                'completed': self.state.nets_completed,
                'failed': self.state.nets_failed,
                'total': self.state.total_nets,
                'progress': self.state.nets_completed / max(1, self.state.total_nets)
            }
            self.kicad_interface.update_progress(progress_info)
    
    def complete_net_routing(self, net_id: str, success: bool):
        """Mark a net as completed (success or failure)"""
        if success:
            self.state.nets_completed += 1
        else:
            self.state.nets_failed += 1
            # Mark failed tracks/vias in red
            self._mark_net_failed(net_id)
        
        # Update current net
        self.state.current_net = None
    
    def _mark_net_failed(self, net_id: str):
        """Mark tracks/vias for a failed net in red"""
        for track in self.active_tracks:
            if track['net'] == net_id:
                track['color'] = ROUTING_COLORS['failed']
        
        for via in self.active_vias:
            if via['net'] == net_id:
                via['color'] = ROUTING_COLORS['failed']
    
    def handle_ripup(self, net_id: str):
        """Handle visualization of net ripup"""
        logger.info(f"Visualizing ripup of net {net_id}")
        
        # Mark ripped tracks/vias in orange
        for track in self.completed_tracks + self.active_tracks:
            if track['net'] == net_id:
                track['color'] = ROUTING_COLORS['ripped']
        
        for via in self.completed_vias + self.active_vias:
            if via['net'] == net_id:
                via['color'] = ROUTING_COLORS['ripped']
        
        self._update_display()
        
        # Remove after brief display
        time.sleep(0.5)
        self._remove_net_from_display(net_id)
    
    def _remove_net_from_display(self, net_id: str):
        """Remove a net's visualization elements"""
        self.completed_tracks = [t for t in self.completed_tracks if t['net'] != net_id]
        self.active_tracks = [t for t in self.active_tracks if t['net'] != net_id]
        self.completed_vias = [v for v in self.completed_vias if v['net'] != net_id]
        self.active_vias = [v for v in self.active_vias if v['net'] != net_id]
    
    def finish_routing_visualization(self):
        """Complete the visualization session"""
        self.state.routing_active = False
        
        # Finalize all remaining tracks/vias
        self._finalize_batch()
        
        # Final statistics
        success_rate = self.state.nets_completed / max(1, self.state.total_nets)
        logger.info(f"Routing visualization completed:")
        logger.info(f"  Total nets: {self.state.total_nets}")
        logger.info(f"  Successfully routed: {self.state.nets_completed}")
        logger.info(f"  Failed: {self.state.nets_failed}")
        logger.info(f"  Success rate: {success_rate:.1%}")
        
        # Show final results
        self._update_display()
    
    def get_visualization_callbacks(self):
        """Get callbacks for router integration"""
        return {
            'progress_callback': self.progress_callback,
            'track_callback': self.track_callback,
            'via_callback': self.via_callback
        }

def create_manhattan_visualizer(kicad_interface=None) -> ManhattanVisualizer:
    """Factory function to create Manhattan visualizer"""
    return ManhattanVisualizer(kicad_interface)