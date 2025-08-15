#!/usr/bin/env python3
"""
Base Router Interface

Defines the common interface and shared functionality for all routing algorithms.
Provides the foundation for Lee's algorithm, Manhattan routing, and future algorithms.
"""
import logging
import time
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from core.drc_rules import DRCRules
from core.gpu_manager import GPUManager
from core.board_interface import BoardInterface
from data_structures.grid_config import GridConfig

logger = logging.getLogger(__name__)


class RoutingResult(Enum):
    """Routing operation results"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


@dataclass
class RouteSegment:
    """Represents a single route segment (track or via)"""
    type: str  # 'track' or 'via'
    start_x: float
    start_y: float
    end_x: float = None
    end_y: float = None
    width: float = 0.25
    layer: str = 'F.Cu'
    net_name: str = ''
    
    def to_kicad_dict(self) -> Dict:
        """Convert to KiCad-compatible dictionary"""
        if self.type == 'via':
            return {
                'type': 'via',
                'x': self.start_x,
                'y': self.start_y,
                'diameter': self.width,
                'drill': self.width * 0.5,
                'net': self.net_name,
                'layers': ['F.Cu', 'B.Cu']
            }
        else:
            return {
                'type': 'track',
                'start_x': self.start_x,
                'start_y': self.start_y,
                'end_x': self.end_x,
                'end_y': self.end_y,
                'width': self.width,
                'layer': 0 if self.layer == 'F.Cu' else 31,
                'net': self.net_name
            }


@dataclass 
class RoutingStats:
    """Statistics for routing operations"""
    nets_attempted: int = 0
    nets_routed: int = 0
    nets_failed: int = 0
    tracks_added: int = 0
    vias_added: int = 0
    total_length_mm: float = 0.0
    routing_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.nets_attempted == 0:
            return 0.0
        return (self.nets_routed / self.nets_attempted) * 100
    
    def update_success(self, segments: List[RouteSegment]):
        """Update stats for successful routing"""
        self.nets_routed += 1
        
        for segment in segments:
            if segment.type == 'track':
                self.tracks_added += 1
                if segment.end_x is not None and segment.end_y is not None:
                    dx = segment.end_x - segment.start_x
                    dy = segment.end_y - segment.start_y
                    self.total_length_mm += (dx * dx + dy * dy) ** 0.5
            elif segment.type == 'via':
                self.vias_added += 1
    
    def update_failure(self):
        """Update stats for failed routing"""
        self.nets_failed += 1


class BaseRouter(ABC):
    """Abstract base class for all routing algorithms"""
    
    def __init__(self, board_interface: BoardInterface, drc_rules: DRCRules, 
                 gpu_manager: GPUManager, grid_config: GridConfig):
        """
        Initialize base router
        
        Args:
            board_interface: Board data interface
            drc_rules: Design rule constraints
            gpu_manager: GPU resource manager
            grid_config: Grid configuration
        """
        self.board_interface = board_interface
        self.drc_rules = drc_rules
        self.gpu_manager = gpu_manager
        self.grid_config = grid_config
        
        # Initialize obstacle grids for pathfinding
        self.obstacle_grids = {}
        self.layers = board_interface.get_layers()
        
        # Route solution storage
        self.routed_segments = []
        self.stats = RoutingStats()
        
        # Progress and callbacks
        self.progress_callback = None
        self.track_callback = None
        
        logger.info(f"🚀 {self.__class__.__name__} initialized")
        logger.info(f"  Layers: {self.layers}")
        logger.info(f"  GPU enabled: {self.gpu_manager.is_gpu_enabled()}")
        
        self._initialize_obstacle_grids()
    
    @abstractmethod
    def route_net(self, net_name: str, timeout: float = 10.0) -> RoutingResult:
        """
        Route a specific net
        
        Args:
            net_name: Name of the net to route
            timeout: Maximum time to spend routing this net
            
        Returns:
            RoutingResult indicating success/failure
        """
        pass
    
    @abstractmethod
    def route_two_pads(self, pad_a: Dict, pad_b: Dict, net_name: str, 
                      timeout: float = 5.0) -> Optional[List[RouteSegment]]:
        """
        Route between two specific pads
        
        Args:
            pad_a: Source pad
            pad_b: Target pad
            net_name: Net name for DRC constraints
            timeout: Maximum routing time
            
        Returns:
            List of route segments or None if routing failed
        """
        pass
    
    def route_all_nets(self, timeout_per_net: float = 5.0, 
                      total_timeout: float = 300.0) -> RoutingStats:
        """
        Route all routable nets on the board
        
        Args:
            timeout_per_net: Maximum time per individual net
            total_timeout: Maximum total routing time
            
        Returns:
            Routing statistics
        """
        start_time = time.time()
        routable_nets = self.board_interface.get_routable_nets()
        
        logger.info(f"🔄 Starting routing of {len(routable_nets)} nets with {self.__class__.__name__}")
        
        # Sort nets by complexity (fewer pads first)
        nets_by_complexity = sorted(routable_nets.items(), 
                                  key=lambda x: x[1]['pad_count'])
        
        for net_name, net_data in nets_by_complexity:
            # Check total timeout
            elapsed = time.time() - start_time
            if elapsed > total_timeout:
                logger.warning(f"⏰ Total routing timeout reached after {elapsed:.1f}s")
                break
            
            # Skip already routed nets
            if net_data['routed']:
                continue
            
            # Update progress
            if self.progress_callback:
                progress = (self.stats.nets_attempted / len(routable_nets)) * 100
                self.progress_callback(progress, f"Routing {net_name}")
            
            self.stats.nets_attempted += 1
            
            # Route this net
            logger.info(f"🔗 Routing net '{net_name}' ({net_data['pad_count']} pads)")
            
            try:
                result = self.route_net(net_name, timeout_per_net)
                
                if result == RoutingResult.SUCCESS:
                    logger.info(f"✅ Successfully routed {net_name}")
                    self.board_interface.mark_net_as_routed(net_name)
                else:
                    logger.warning(f"❌ Failed to route {net_name}: {result}")
                    self.stats.update_failure()
                    
            except Exception as e:
                logger.error(f"❌ Exception routing {net_name}: {e}")
                self.stats.update_failure()
        
        self.stats.routing_time = time.time() - start_time
        
        logger.info(f"🏁 Routing completed in {self.stats.routing_time:.2f}s")
        logger.info(f"📊 Results: {self.stats.nets_routed}/{self.stats.nets_attempted} nets routed ({self.stats.success_rate:.1f}%)")
        logger.info(f"🔧 Created: {self.stats.tracks_added} tracks, {self.stats.vias_added} vias")
        logger.info(f"📏 Total length: {self.stats.total_length_mm:.1f}mm")
        
        return self.stats
    
    def get_routed_tracks(self) -> List[Dict]:
        """Get all routed tracks in KiCad format"""
        tracks = []
        for segment in self.routed_segments:
            if segment.type == 'track':
                tracks.append(segment.to_kicad_dict())
        return tracks
    
    def get_routed_vias(self) -> List[Dict]:
        """Get all routed vias in KiCad format"""
        vias = []
        for segment in self.routed_segments:
            if segment.type == 'via':
                vias.append(segment.to_kicad_dict())
        return vias
    
    def set_progress_callback(self, callback):
        """Set progress update callback"""
        self.progress_callback = callback
    
    def set_track_callback(self, callback):
        """Set real-time track update callback"""
        self.track_callback = callback
    
    def _initialize_obstacle_grids(self):
        """Initialize obstacle grids for all layers"""
        logger.info("🗺️ Initializing obstacle grids using IPC-2221A pathfinding methodology...")
        
        for layer in self.layers:
            # Create obstacle grid
            obstacle_grid = self.gpu_manager.create_array(
                (self.grid_config.height, self.grid_config.width), 
                dtype=None,  # Let GPU manager choose appropriate bool type
                fill_value=0
            )
            
            # Mark obstacles using IPC-2221A methodology
            self._mark_layer_obstacles(obstacle_grid, layer)
            
            self.obstacle_grids[layer] = obstacle_grid
        
        self._log_obstacle_statistics()
    
    def _mark_layer_obstacles(self, obstacle_grid, layer: str):
        """Mark obstacles on a specific layer"""
        # Mark pads as obstacles
        self._mark_pads_as_obstacles(obstacle_grid, layer)
        
        # Mark existing tracks as obstacles  
        self._mark_tracks_as_obstacles(obstacle_grid, layer)
        
        # Mark existing vias as obstacles
        self._mark_vias_as_obstacles(obstacle_grid, layer)
        
        # Mark copper zones as obstacles
        self._mark_zones_as_obstacles(obstacle_grid, layer)
    
    def _mark_pads_as_obstacles(self, obstacle_grid, layer: str, exclude_net: str = None):
        """Mark pads as obstacles with proper clearance"""
        pads = self.board_interface.get_all_pads()
        marked_count = 0
        
        for pad in pads:
            # Skip pads not on this layer
            if not self.board_interface.is_pad_on_layer(pad, layer):
                continue
            
            # Skip pads on excluded net
            pad_net_name = self.board_interface._extract_net_name(pad.get('net'))
            if exclude_net and pad_net_name == exclude_net:
                continue
            
            # Get pad geometry and mark obstacle area
            geometry = self.board_interface.get_pad_geometry(pad)
            marked_count += self._mark_rectangular_obstacle(
                obstacle_grid, 
                geometry['x'], geometry['y'],
                geometry['size_x'], geometry['size_y']
            )
        
        return marked_count
    
    def _mark_tracks_as_obstacles(self, obstacle_grid, layer: str):
        """Mark existing tracks as obstacles"""
        tracks = self.board_interface.get_all_tracks()
        marked_count = 0
        
        for track in tracks:
            track_geometry = self.board_interface.get_track_geometry(track)
            
            # Only mark tracks on this layer
            track_layer_id = track_geometry['layer']
            expected_layer_id = self.board_interface.get_layer_id(layer)
            
            if track_layer_id == expected_layer_id:
                marked_count += self._mark_line_obstacle(
                    obstacle_grid,
                    track_geometry['start_x'], track_geometry['start_y'],
                    track_geometry['end_x'], track_geometry['end_y'],
                    track_geometry['width']
                )
        
        return marked_count
    
    def _mark_vias_as_obstacles(self, obstacle_grid, layer: str):
        """Mark existing vias as obstacles"""
        vias = self.board_interface.get_all_vias()
        marked_count = 0
        
        for via in vias:
            # Vias affect all layers
            via_x = via.get('x', 0)
            via_y = via.get('y', 0)
            via_diameter = via.get('diameter', self.drc_rules.via_diameter)
            
            marked_count += self._mark_circular_obstacle(
                obstacle_grid, via_x, via_y, via_diameter / 2
            )
        
        return marked_count
    
    def _mark_zones_as_obstacles(self, obstacle_grid, layer: str):
        """Mark copper zones as obstacles"""
        zones = self.board_interface.get_all_zones()
        # Simplified implementation - full zone processing would be more complex
        return 0
    
    def _mark_rectangular_obstacle(self, obstacle_grid, center_x: float, center_y: float, 
                                 size_x: float, size_y: float) -> int:
        """Mark a rectangular obstacle on the grid"""
        # Convert to grid coordinates
        grid_x, grid_y = self.grid_config.world_to_grid(center_x, center_y)
        
        # Calculate grid extents
        half_size_x_cells = max(1, int(math.ceil((size_x / 2) / self.grid_config.resolution)))
        half_size_y_cells = max(1, int(math.ceil((size_y / 2) / self.grid_config.resolution)))
        
        marked_count = 0
        
        # Mark the rectangular area
        for dx in range(-half_size_x_cells, half_size_x_cells + 1):
            for dy in range(-half_size_y_cells, half_size_y_cells + 1):
                gx, gy = grid_x + dx, grid_y + dy
                
                if self.grid_config.is_valid_grid_position(gx, gy):
                    obstacle_grid[gy, gx] = True
                    marked_count += 1
        
        return marked_count
    
    def _mark_line_obstacle(self, obstacle_grid, start_x: float, start_y: float,
                           end_x: float, end_y: float, width: float) -> int:
        """Mark a line (track) obstacle on the grid"""
        # This is a simplified implementation
        # Full implementation would use proper line rasterization
        return self._mark_rectangular_obstacle(
            obstacle_grid, 
            (start_x + end_x) / 2, 
            (start_y + end_y) / 2,
            abs(end_x - start_x) + width,
            abs(end_y - start_y) + width
        )
    
    def _mark_circular_obstacle(self, obstacle_grid, center_x: float, center_y: float, 
                               radius: float) -> int:
        """Mark a circular obstacle on the grid"""
        # For simplicity, use rectangular approximation
        return self._mark_rectangular_obstacle(
            obstacle_grid, center_x, center_y, radius * 2, radius * 2
        )
    
    def _log_obstacle_statistics(self):
        """Log obstacle grid statistics"""
        import numpy as np
        
        total_obstacles = 0
        total_cells = 0
        
        for layer, grid in self.obstacle_grids.items():
            # Convert to CPU for counting if needed
            grid_cpu = self.gpu_manager.to_cpu(grid) if hasattr(grid, 'get') else grid
            
            layer_obstacles = int(np.sum(grid_cpu))
            layer_cells = grid_cpu.size
            
            total_obstacles += layer_obstacles
            total_cells += layer_cells
            
            density = (layer_obstacles / layer_cells) * 100
            logger.info(f"🚧 {layer}: {layer_obstacles} obstacles ({density:.1f}% density)")
        
        overall_density = (total_obstacles / total_cells) * 100 if total_cells > 0 else 0
        logger.info(f"🚧 Total: {total_obstacles} obstacles across {len(self.layers)} layers ({overall_density:.1f}% density)")
    
    def get_routing_statistics(self) -> RoutingStats:
        """Get current routing statistics"""
        return self.stats
    
    def clear_routes(self):
        """Clear all routed segments"""
        self.routed_segments.clear()
        self.stats = RoutingStats()
        logger.info("🗑️ Cleared all routed segments")
