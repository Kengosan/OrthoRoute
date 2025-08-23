"""Application service for managing visualization."""
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import asyncio

from ...domain.events.routing_events import (
    NetRouted, VisualizationUpdate, RoutingProgress, DomainEvent
)
from ..interfaces.event_publisher import EventPublisher

logger = logging.getLogger(__name__)


class VisualizationService:
    """Application service for managing real-time visualization updates."""
    
    def __init__(self, event_publisher: EventPublisher):
        """Initialize visualization service."""
        self.event_publisher = event_publisher
        
        # Visualization state
        self.active_tracks = []
        self.completed_tracks = []
        self.active_vias = []
        self.completed_vias = []
        
        # Rendering settings
        self.batch_size = 10  # Update UI every N items
        self.update_interval = 0.1  # seconds
        self.last_update = datetime.now()
        
        # Colors following KiCad theme
        self.layer_colors = {
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
        
        # UI callbacks
        self.ui_update_callbacks: List[Callable] = []
        
        # Subscribe to routing events
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self):
        """Set up event subscriptions for visualization updates."""
        self.event_publisher.subscribe(NetRouted, self._handle_net_routed)
        self.event_publisher.subscribe(RoutingProgress, self._handle_routing_progress)
        self.event_publisher.subscribe(VisualizationUpdate, self._handle_visualization_update)
    
    def register_ui_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a UI update callback."""
        self.ui_update_callbacks.append(callback)
    
    def unregister_ui_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Unregister a UI update callback."""
        if callback in self.ui_update_callbacks:
            self.ui_update_callbacks.remove(callback)
    
    def _handle_net_routed(self, event: NetRouted):
        """Handle net routed event."""
        try:
            # Move active items to completed with proper colors
            self._finalize_active_items(event.net_id)
            
            # Add new route visualization
            if event.route:
                self._add_route_visualization(event.route)
            
            # Trigger UI update
            self._maybe_update_ui()
            
        except Exception as e:
            logger.error(f"Error handling NetRouted event: {e}")
    
    def _handle_routing_progress(self, event: RoutingProgress):
        """Handle routing progress event."""
        try:
            # Update progress display
            progress_data = {
                'current_net': event.current_net,
                'completed': event.nets_completed,
                'failed': event.nets_failed,
                'total': event.total_nets,
                'percentage': event.completion_percentage
            }
            
            self._update_progress_display(progress_data)
            
            # Batch finalize items every 10 nets
            if event.nets_completed % 10 == 0 and event.nets_completed > 0:
                self._batch_finalize_items()
                
        except Exception as e:
            logger.error(f"Error handling RoutingProgress event: {e}")
    
    def _handle_visualization_update(self, event: VisualizationUpdate):
        """Handle visualization update event."""
        try:
            if event.update_type == 'track':
                self._add_active_track(event.data, event.net_id)
            elif event.update_type == 'via':
                self._add_active_via(event.data, event.net_id)
            elif event.update_type == 'clear':
                self._clear_net_visualization(event.net_id)
            
            self._maybe_update_ui()
            
        except Exception as e:
            logger.error(f"Error handling VisualizationUpdate event: {e}")
    
    def _add_active_track(self, track_data: Dict[str, Any], net_id: str):
        """Add a track to active visualization."""
        track_viz = {
            'start': track_data.get('start'),
            'end': track_data.get('end'),
            'layer': track_data.get('layer'),
            'width': track_data.get('width'),
            'net': net_id,
            'color': 'rgb(255, 255, 255)',  # Bright white for active
            'timestamp': datetime.now()
        }
        self.active_tracks.append(track_viz)
    
    def _add_active_via(self, via_data: Dict[str, Any], net_id: str):
        """Add a via to active visualization."""
        via_viz = {
            'x': via_data.get('x'),
            'y': via_data.get('y'),
            'size': via_data.get('size'),
            'drill': via_data.get('drill'),
            'layers': via_data.get('layers', []),
            'net': net_id,
            'type': via_data.get('type'),
            'color': 'rgb(255, 255, 255)',  # Bright white for active
            'timestamp': datetime.now()
        }
        self.active_vias.append(via_viz)
    
    def _add_route_visualization(self, route):
        """Add route visualization from completed route."""
        # This would extract tracks and vias from the route object
        # and add them to completed visualizations
        for segment in route.segments:
            track_viz = {
                'start': (segment.start.x, segment.start.y),
                'end': (segment.end.x, segment.end.y),
                'layer': segment.layer,
                'width': segment.width,
                'net': route.net_id,
                'color': self._get_layer_color(segment.layer),
                'timestamp': datetime.now()
            }
            self.completed_tracks.append(track_viz)
        
        for via in route.vias:
            via_viz = {
                'x': via.position.x,
                'y': via.position.y,
                'size': via.diameter,
                'drill': via.drill_size,
                'layers': [via.from_layer, via.to_layer],
                'net': route.net_id,
                'type': via.via_type.value,
                'color': self._get_via_color([via.from_layer, via.to_layer]),
                'timestamp': datetime.now()
            }
            self.completed_vias.append(via_viz)
    
    def _finalize_active_items(self, net_id: str):
        """Move active items for a net to completed with proper colors."""
        # Finalize tracks
        tracks_to_move = [t for t in self.active_tracks if t['net'] == net_id]
        for track in tracks_to_move:
            track['color'] = self._get_layer_color(track['layer'])
            self.completed_tracks.append(track)
        
        self.active_tracks = [t for t in self.active_tracks if t['net'] != net_id]
        
        # Finalize vias
        vias_to_move = [v for v in self.active_vias if v['net'] == net_id]
        for via in vias_to_move:
            via['color'] = self._get_via_color(via['layers'])
            self.completed_vias.append(via)
        
        self.active_vias = [v for v in self.active_vias if v['net'] != net_id]
    
    def _batch_finalize_items(self):
        """Finalize all active items in batch."""
        # Move all active tracks to completed
        for track in self.active_tracks:
            track['color'] = self._get_layer_color(track['layer'])
            self.completed_tracks.append(track)
        self.active_tracks.clear()
        
        # Move all active vias to completed
        for via in self.active_vias:
            via['color'] = self._get_via_color(via['layers'])
            self.completed_vias.append(via)
        self.active_vias.clear()
        
        self._force_ui_update()
    
    def _clear_net_visualization(self, net_id: str):
        """Clear visualization for a specific net."""
        self.active_tracks = [t for t in self.active_tracks if t['net'] != net_id]
        self.completed_tracks = [t for t in self.completed_tracks if t['net'] != net_id]
        self.active_vias = [v for v in self.active_vias if v['net'] != net_id]
        self.completed_vias = [v for v in self.completed_vias if v['net'] != net_id]
    
    def clear_all_visualization(self):
        """Clear all visualization data."""
        self.active_tracks.clear()
        self.completed_tracks.clear()
        self.active_vias.clear()
        self.completed_vias.clear()
        self._force_ui_update()
    
    def _get_layer_color(self, layer_name: str) -> str:
        """Get standard color for a layer."""
        return self.layer_colors.get(layer_name, 'rgb(128, 128, 128)')
    
    def _get_via_color(self, layers: List[str]) -> str:
        """Get color for via based on layers it connects."""
        if 'F.Cu' in layers:
            return self.layer_colors['F.Cu']
        elif 'B.Cu' in layers:
            return self.layer_colors['B.Cu']
        else:
            # Use color of first inner layer
            for layer in layers:
                if layer in self.layer_colors:
                    return self.layer_colors[layer]
        return 'rgb(128, 128, 128)'  # Default gray
    
    def _maybe_update_ui(self):
        """Update UI if enough time has passed."""
        now = datetime.now()
        if (now - self.last_update).total_seconds() > self.update_interval:
            self._force_ui_update()
            self.last_update = now
    
    def _force_ui_update(self):
        """Force UI update immediately."""
        visualization_data = self.get_visualization_data()
        
        for callback in self.ui_update_callbacks:
            try:
                callback(visualization_data)
            except Exception as e:
                logger.error(f"Error in UI callback: {e}")
    
    def _update_progress_display(self, progress_data: Dict[str, Any]):
        """Update progress display."""
        for callback in self.ui_update_callbacks:
            try:
                callback({'type': 'progress', 'data': progress_data})
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get current visualization data."""
        return {
            'type': 'visualization',
            'active_tracks': self.active_tracks.copy(),
            'completed_tracks': self.completed_tracks.copy(),
            'active_vias': self.active_vias.copy(),
            'completed_vias': self.completed_vias.copy(),
            'track_count': len(self.active_tracks) + len(self.completed_tracks),
            'via_count': len(self.active_vias) + len(self.completed_vias)
        }
    
    def set_visualization_settings(self, settings: Dict[str, Any]):
        """Update visualization settings."""
        if 'batch_size' in settings:
            self.batch_size = settings['batch_size']
        
        if 'update_interval' in settings:
            self.update_interval = settings['update_interval']
        
        if 'layer_colors' in settings:
            self.layer_colors.update(settings['layer_colors'])
        
        logger.info("Visualization settings updated")