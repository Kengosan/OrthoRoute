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
import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

# Add src directory to Python path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try absolute imports first, fall back to relative
try:
    from core.drc_rules import DRCRules
    from core.gpu_manager import GPUManager
    from core.board_interface import BoardInterface
    from data_structures.grid_config import GridConfig
    from routing_engines.virtual_copper_generator import VirtualCopperGenerator
except ImportError:
    try:
        from ..core.drc_rules import DRCRules
        from ..core.gpu_manager import GPUManager
        from ..core.board_interface import BoardInterface
        from ..data_structures.grid_config import GridConfig
        from .virtual_copper_generator import VirtualCopperGenerator
    except ImportError:
        # Handle the case where we're running from the src directory
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from core.drc_rules import DRCRules
        from core.gpu_manager import GPUManager
        from core.board_interface import BoardInterface
        from data_structures.grid_config import GridConfig
        from routing_engines.virtual_copper_generator import VirtualCopperGenerator

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
                 gpu_manager: GPUManager, grid_config: GridConfig, 
                 use_virtual_copper: bool = False):
        """
        Initialize base router
        
        Args:
            board_interface: Board data interface
            drc_rules: Design rule constraints
            gpu_manager: GPU resource manager
            grid_config: Grid configuration
            use_virtual_copper: Whether to use Virtual Copper Generator for obstacle detection
        """
        self.board_interface = board_interface
        self.drc_rules = drc_rules
        self.gpu_manager = gpu_manager
        self.grid_config = grid_config
        self.use_virtual_copper = use_virtual_copper
        
        # Initialize basic data structures (but not obstacle grids - let each router decide)
        self.obstacle_grids = {}
        self.layers = board_interface.get_layers()
        
        # Route solution storage
        self.routed_segments = []
        self.stats = RoutingStats()
        
        # Progress and callbacks
        self.progress_callback = None
        self.track_callback = None
        self.via_callback = None
        
        logger.info(f"🚀 {self.__class__.__name__} initialized")
        logger.info(f"  Layers: {self.layers}")
        logger.info(f"  GPU enabled: {self.gpu_manager.is_gpu_enabled()}")
        logger.info(f"  Using virtual copper: {self.use_virtual_copper}")
        
        # Initialize obstacle grids if virtual copper is enabled
        if self.use_virtual_copper:
            logger.info("🗺️ Initializing Free Routing Space architecture...")
            self._initialize_obstacle_grids()
        else:
            logger.info("🚀 Skipping virtual copper generation - using router-specific obstacle detection")
    
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
        
        logger.info(f"� BASE ROUTER route_all_nets called! routable_nets type: {type(routable_nets)}")
        logger.info(f"�🔄 Starting routing of {len(routable_nets)} nets with {self.__class__.__name__}")
        
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
            
            # Skip nets that have copper planes (GND, power planes, etc.)
            if self._is_plane_connected_net(net_name):
                logger.info(f"⚡ Skipping '{net_name}' - connected via copper plane")
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
                tracks.append(segment.to_kicad_dict(self))
        return tracks
    
    def get_routed_vias(self) -> List[Dict]:
        """Get all routed vias in KiCad format"""
        vias = []
        for segment in self.routed_segments:
            if segment.type == 'via':
                vias.append(segment.to_kicad_dict(self))
        return vias
    
    def set_progress_callback(self, callback):
        """Set progress update callback"""
        self.progress_callback = callback
    
    def set_track_callback(self, callback):
        """Set real-time track update callback"""
        self.track_callback = callback
    
    def set_via_callback(self, callback):
        """Set real-time via update callback"""
        self.via_callback = callback
    
    def _initialize_obstacle_grids(self):
        """Initialize obstacle grids using Virtual Copper Generator methodology"""
        logger.info("🗺️ Generating obstacle grids using Virtual Copper Generator...")
        
        # Initialize the virtual copper generator
        self.virtual_copper_generator = VirtualCopperGenerator(
            self.board_interface, 
            self.drc_rules, 
            self.grid_config
        )
        
        # Generate obstacle grids for all layers using virtual copper methodology
        self.obstacle_grids = {}
        for layer in self.layers:
            logger.info(f"🌟 Generating virtual copper pour for layer {layer}")
            virtual_copper_grid = self.virtual_copper_generator.generate_virtual_copper_grid(layer)
            
            # Obstacle grid is the INVERSE of virtual copper grid
            # True = obstacle (cannot route), False = free space (can route)
            import numpy as np
            obstacle_grid = ~virtual_copper_grid  # Logical NOT using NumPy first
            
            # Convert to GPU array if GPU is enabled
            if self.gpu_manager.is_gpu_enabled():
                # Convert processed NumPy array to GPU array
                obstacle_grid = self.gpu_manager.to_gpu(obstacle_grid)
            
            self.obstacle_grids[layer] = obstacle_grid
        
        self._log_obstacle_statistics()
    
    def _generate_free_routing_space(self, layer: str):
        """Generate free routing space using intelligent net-aware exclusions
        
        Unlike pure virtual copper pour, this considers net connectivity:
        - Excludes existing tracks/vias (with DRC clearances)
        - Excludes copper zones (with DRC clearances)
        - Does NOT exclude any pads - those will be handled per-net during routing
        
        The obstacle grid will be inverted from this free space.
        """
        # Start with entire board as free space
        free_space = self.gpu_manager.create_array(
            (self.grid_config.height, self.grid_config.width), 
            dtype=None,  # Let GPU manager choose appropriate bool type
            fill_value=1  # Start with all free
        )
        
        # Apply exclusions for fixed obstacles only (not pads - they're handled per-net)
        self._exclude_fixed_obstacles_from_free_space(free_space, layer)
        
        return free_space
    
    def _exclude_fixed_obstacles_from_free_space(self, free_space, layer: str):
        """Exclude only permanent obstacles from free space
        
        This excludes:
        - Existing tracks (with clearances)
        - Existing vias (with clearances)  
        - Copper zones (with clearances)
        
        But does NOT exclude any pads - those will be handled per-net during routing.
        """
        # Apply exclusions for existing tracks  
        self._exclude_tracks_from_free_space(free_space, layer)
        
        # Apply exclusions for existing vias
        self._exclude_vias_from_free_space(free_space, layer)
        
        # Apply exclusions for copper zones
        self._exclude_zones_from_free_space(free_space, layer)
        
        # Apply exclusions for existing tracks  
        self._exclude_tracks_from_free_space(free_space, layer)
        
        # Apply exclusions for existing vias
        self._exclude_vias_from_free_space(free_space, layer)
        
        # Apply exclusions for copper zones
        self._exclude_zones_from_free_space(free_space, layer)
    
    def _exclude_pads_from_free_space(self, free_space, layer: str):
        """Exclude pad areas from free space with proper DRC clearances
        
        Uses virtual copper pour methodology - removes areas around pads
        where traces cannot be placed due to DRC clearance requirements.
        This uses more conservative clearances suitable for general routing.
        """
        pads = self.board_interface.get_all_pads()
        excluded_count = 0
        
        for pad in pads:
            if not self.board_interface.is_pad_on_layer(pad, layer):
                continue
                
            geometry = self.board_interface.get_pad_geometry(pad)
            
            # Use minimal DRC clearance - just enough to prevent violations
            # The connection point clearing in LeeRouter will handle specific net access
            drc_clearance = self.drc_rules.min_trace_spacing  # 0.508mm base clearance
            
            # Conservative exclusion: pad + minimal clearance
            # This allows traces to route close to pads while maintaining DRC compliance
            exclusion_x = geometry['size_x'] + 2 * drc_clearance
            exclusion_y = geometry['size_y'] + 2 * drc_clearance
            
            self._mark_exclusion_area(
                free_space,
                geometry['x'], geometry['y'],
                exclusion_x, exclusion_y
            )
            excluded_count += 1
        
        logger.debug(f"🚫 Excluded {excluded_count} pads from free space on {layer} with minimal DRC clearances")
    
    def _exclude_tracks_from_free_space(self, free_space, layer: str):
        """Exclude existing track areas from free space"""
        tracks = self.board_interface.get_all_tracks()
        excluded_count = 0
        
        # Convert layer name to layer ID for comparison
        layer_id = self.board_interface.get_layer_id(layer)
        
        # Debug: Log first few tracks to see their layer values
        if tracks:
            logger.debug(f"🔍 Checking tracks for layer {layer} (layer_id={layer_id})")
            for i, track in enumerate(tracks[:5]):  # Show first 5 tracks
                track_layer = track.get('layer', 'unknown')
                logger.debug(f"  Track {i}: layer={track_layer}, net={track.get('net', 'unknown')}")
        
        for track in tracks:
            if track.get('layer') == layer_id:
                # Mark track area as excluded with proper clearance
                self._mark_line_exclusion(
                    free_space,
                    track['start_x'], track['start_y'],
                    track['end_x'], track['end_y'],
                    track['width'] + 2 * self.drc_rules.min_trace_spacing
                )
                excluded_count += 1
        
        logger.debug(f"🚫 Excluded {excluded_count} tracks from free space on {layer} (layer_id={layer_id})")
    
    def _exclude_vias_from_free_space(self, free_space, layer: str):
        """Exclude via areas from free space"""
        vias = self.board_interface.get_all_vias()
        excluded_count = 0
        
        for via in vias:
            # Vias affect all layers
            exclusion_diameter = via['via_diameter'] + 2 * self.drc_rules.min_trace_spacing
            self._mark_circular_exclusion(
                free_space,
                via['x'], via['y'],
                exclusion_diameter / 2
            )
            excluded_count += 1
        
        logger.debug(f"🚫 Excluded {excluded_count} vias from free space on {layer}")
    
    def _exclude_zones_from_free_space(self, free_space, layer: str):
        """Exclude copper zone areas from free space"""
        zones = self.board_interface.get_all_zones()
        excluded_count = 0
        
        for zone in zones:
            if layer in zone.get('layers', []):
                # Simplified implementation - mark zone boundary areas as excluded
                # Full implementation would process complex polygon shapes
                excluded_count += 1
        
        logger.debug(f"🚫 Excluded {excluded_count} zones from free space on {layer}")
    
    def _mark_exclusion_area(self, free_space, center_x: float, center_y: float, 
                           size_x: float, size_y: float):
        """Mark a rectangular area as excluded from free space (sets to False)"""
        # Convert to grid coordinates
        grid_x, grid_y = self.grid_config.world_to_grid(center_x, center_y)
        
        # Calculate grid extents
        half_size_x_cells = max(1, int(math.ceil((size_x / 2) / self.grid_config.resolution)))
        half_size_y_cells = max(1, int(math.ceil((size_y / 2) / self.grid_config.resolution)))
        
        # Mark the rectangular area as excluded (False = not free space)
        for dx in range(-half_size_x_cells, half_size_x_cells + 1):
            for dy in range(-half_size_y_cells, half_size_y_cells + 1):
                gx, gy = grid_x + dx, grid_y + dy
                
                if self.grid_config.is_valid_grid_position(gx, gy):
                    free_space[gy, gx] = False
    
    def _mark_line_exclusion(self, free_space, start_x: float, start_y: float,
                            end_x: float, end_y: float, width: float):
        """Mark a line area as excluded from free space"""
        # Simplified implementation using rectangular approximation
        self._mark_exclusion_area(
            free_space,
            (start_x + end_x) / 2, 
            (start_y + end_y) / 2,
            abs(end_x - start_x) + width,
            abs(end_y - start_y) + width
        )
    
    def _mark_circular_exclusion(self, free_space, center_x: float, center_y: float, 
                                radius: float):
        """Mark a circular area as excluded from free space"""
        # For simplicity, use rectangular approximation
        self._mark_exclusion_area(
            free_space, center_x, center_y, radius * 2, radius * 2
        )

    def _mark_layer_obstacles(self, obstacle_grid, layer: str):
        """Mark obstacles on a specific layer - DEPRECATED
        
        This method is replaced by the Free Routing Space approach.
        Kept for compatibility but should not be used.
        """
        logger.warning("⚠️ _mark_layer_obstacles called - this is deprecated in Free Routing Space architecture")
        pass
    
    def _mark_pads_as_obstacles(self, obstacle_grid, layer: str, exclude_net: str = None):
        """Mark pads as obstacles with minimal clearance to allow routing"""
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
            
            # Get pad geometry and mark obstacle area with MINIMAL clearance
            geometry = self.board_interface.get_pad_geometry(pad)
            
            # Use minimal clearance - just the pad plus a small safety margin
            # This prevents the grid from becoming too dense while still protecting pads
            safety_margin = 0.1  # 0.1mm safety margin around pads
            minimal_size_x = geometry['size_x'] + 2 * safety_margin
            minimal_size_y = geometry['size_y'] + 2 * safety_margin
            
            marked_count += self._mark_rectangular_obstacle(
                obstacle_grid, 
                geometry['x'], geometry['y'],
                minimal_size_x, minimal_size_y  # Use minimal size for obstacles
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
    
    def _is_plane_connected_net(self, net_name: str) -> bool:
        """Check if a net is connected via copper planes (should not be routed)"""
        # Check if the board interface has copper zone information
        if hasattr(self.board_interface, 'get_all_zones'):
            zones = self.board_interface.get_all_zones()
            for zone in zones:
                zone_net = zone.get('net', '')
                if zone_net == net_name:
                    return True
        
        # Common plane-connected net names
        plane_nets = {'GND', 'GROUND', '+5V', '+3V3', '+12V', '-12V', 'VCC', 'VDD', 'VSS'}
        return net_name.upper() in plane_nets
    
    def get_routing_statistics(self) -> RoutingStats:
        """Get current routing statistics"""
        return self.stats
    
    def clear_routes(self):
        """Clear all routed segments"""
        self.routed_segments.clear()
        self.stats = RoutingStats()
        logger.info("🗑️ Cleared all routed segments")
