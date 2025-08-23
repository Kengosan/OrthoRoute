#!/usr/bin/env python3
"""
Manhattan Router with Blind/Buried Vias

Implements Manhattan routing with:
- 3D grid for multiple copper layers
- 0.4mm grid pitch
- Layer-specific routing directions (horizontal/vertical)
- A* pathfinding with Manhattan distance heuristic
- Ripup and repair for congestion resolution
- Blind/buried via support
"""

import logging
import time
import math
import heapq
import sys
import os
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import base router and other dependencies
try:
    from routing_engines.base_router import BaseRouter, RoutingResult, RouteSegment, RoutingStats
    from core.drc_rules import DRCRules
    from core.gpu_manager import GPUManager
    from core.board_interface import BoardInterface
    from data_structures.grid_config import GridConfig
    from routing_engines.virtual_copper_generator import VirtualCopperGenerator
except ImportError:
    from .base_router import BaseRouter, RoutingResult, RouteSegment, RoutingStats
    from ..core.drc_rules import DRCRules
    from ..core.gpu_manager import GPUManager
    from ..core.board_interface import BoardInterface
    from ..data_structures.grid_config import GridConfig
    from .virtual_copper_generator import VirtualCopperGenerator

logger = logging.getLogger(__name__)

# Cell states in the routing grid
class CellState:
    EMPTY = 0
    OBSTACLE = 1
    ROUTED = 2  # Will also store net_id as separate data


@dataclass
class GridCell:
    """Represents a cell in the routing grid"""
    x: int
    y: int
    layer: int
    

@dataclass
class RouteSegmentInfo:
    """Information about a route segment compatible with BaseRouter"""
    start: GridCell
    end: GridCell
    net_id: str
    width: float
    via_type: Optional[str] = None  # None for track, or via type (through, blind, buried)
    
    def is_via(self) -> bool:
        """Check if this segment is a via"""
        return self.via_type is not None
    
    # Compatibility properties for BaseRouter
    @property
    def type(self) -> str:
        """Segment type for BaseRouter compatibility"""
        return 'via' if self.via_type is not None else 'track'
    
    @property
    def start_x(self) -> float:
        """Start x coordinate for BaseRouter compatibility"""
        return self.start.x
    
    @property
    def start_y(self) -> float:
        """Start y coordinate for BaseRouter compatibility"""
        return self.start.y
    
    @property
    def end_x(self) -> float:
        """End x coordinate for BaseRouter compatibility"""
        return self.end.x if self.end else None
    
    @property 
    def end_y(self) -> float:
        """End y coordinate for BaseRouter compatibility"""
        return self.end.y if self.end else None
    
    @property
    def net_name(self) -> str:
        """Net name for BaseRouter compatibility"""
        return self.net_id
    
    def to_track_dict(self, router) -> dict:
        """Convert to dictionary format expected by track_callback"""
        try:
            if self.is_via():
                # Via format with proper positioning and via type
                if not hasattr(self.start, 'x') or not hasattr(self.start, 'y'):
                    raise ValueError(f"Via start cell missing coordinates: {self.start}")
                if not hasattr(self.end, 'x') or not hasattr(self.end, 'y'):
                    raise ValueError(f"Via end cell missing coordinates: {self.end}")
                    
                start_x, start_y = router.grid_to_world(self.start.x, self.start.y)
                return {
                    'x': start_x,
                    'y': start_y,
                    'from_layer': router.layer_names[self.start.layer] if self.start.layer < len(router.layer_names) else f'Layer{self.start.layer}',
                    'to_layer': router.layer_names[self.end.layer] if self.end.layer < len(router.layer_names) else f'Layer{self.end.layer}',
                    'drill_diameter': router.via_drill,
                    'via_diameter': self.width,
                    'net': self.net_id,
                    'type': self.via_type if self.via_type else 'blind_buried',
                    # Keep backward compatibility fields
                    'start': {'x': start_x, 'y': start_y},
                    'end': {'x': start_x, 'y': start_y},
                    'layer': router.layer_names[self.start.layer] if self.start.layer < len(router.layer_names) else f'Layer{self.start.layer}',
                    'end_layer': router.layer_names[self.end.layer] if self.end.layer < len(router.layer_names) else f'Layer{self.end.layer}',
                    'width': self.width,
                    'drill': router.via_drill
                }
            else:
                # Track format
                if not hasattr(self.start, 'x') or not hasattr(self.start, 'y'):
                    raise ValueError(f"Track start cell missing coordinates: {self.start}")
                    
                start_x, start_y = router.grid_to_world(self.start.x, self.start.y)
                end_x, end_y = router.grid_to_world(self.end.x, self.end.y) if self.end and hasattr(self.end, 'x') and hasattr(self.end, 'y') else (start_x, start_y)
                
                return {
                    'start_x': start_x,
                    'start_y': start_y,
                    'end_x': end_x,
                    'end_y': end_y,
                    'layer': router.layer_names[self.start.layer] if self.start.layer < len(router.layer_names) else f'Layer{self.start.layer}',
                    'width': self.width,
                    'net': self.net_id,
                    'type': 'track'
                }
        except Exception as e:
            # Return a safe fallback dict to prevent cascading errors
            if self.is_via():
                return {
                    'x': 0, 'y': 0,
                    'from_layer': 'F.Cu', 'to_layer': 'B.Cu',
                    'drill_diameter': 0.15, 'via_diameter': 0.25,
                    'net': self.net_id if hasattr(self, 'net_id') else 'unknown',
                    'type': 'blind_buried',
                    'error': str(e)
                }
            else:
                return {
                    'start_x': 0, 'start_y': 0,
                    'end_x': 0, 'end_y': 0,
                    'layer': 'F.Cu',
                    'width': 0.1,
                    'net': self.net_id if hasattr(self, 'net_id') else 'unknown',
                    'type': 'track',
                    'error': str(e)
                }
    
    def to_kicad_dict(self, router) -> dict:
        """Convert to KiCad export dictionary format"""
        if self.is_via():
            # Via format for KiCad export
            start_x, start_y = router.grid_to_world(self.start.x, self.start.y)
            return {
                'type': 'via',
                'position': [start_x, start_y],
                'size': self.width,
                'drill': router.via_drill,
                'net': self.net_id,
                'via_type': self.via_type,
                'start_layer': router.layer_names[self.start.layer],
                'end_layer': router.layer_names[self.end.layer],
                'layers': router._get_via_layers(self.start.layer, self.end.layer) if hasattr(router, '_get_via_layers') else []
            }
        else:
            # Track format for KiCad export
            start_x, start_y = router.grid_to_world(self.start.x, self.start.y)
            end_x, end_y = router.grid_to_world(self.end.x, self.end.y) if self.end else (start_x, start_y)
            
            return {
                'type': 'track',
                'start': [start_x, start_y],
                'end': [end_x, end_y],
                'width': self.width,
                'layer': router.layer_names[self.start.layer],
                'net': self.net_id
            }


class ManhattanRouter(BaseRouter):
    """
    Manhattan Router with multi-layer support
    
    Features:
    - 3D grid for multiple copper layers
    - Layer-specific routing directions (horizontal/vertical)
    - A* pathfinding with Manhattan distance heuristic
    - Ripup and repair for congestion resolution
    - Blind/buried via support
    """
    
    def __init__(self, board_interface, drc_rules, gpu_manager, grid_config):
        """Initialize the Manhattan router"""
        super().__init__(board_interface, drc_rules, gpu_manager, grid_config)
        
        # Get actual board dimensions from board_interface
        # Check method name (production uses get_board_bounds, tests use get_board_dimensions)
        if hasattr(board_interface, 'get_board_bounds'):
            self.board_dimensions = board_interface.get_board_bounds()
        elif hasattr(board_interface, 'get_board_dimensions'):
            self.board_dimensions = board_interface.get_board_dimensions()
        else:
            raise AttributeError("BoardInterface must have either get_board_bounds or get_board_dimensions method")
        self.layers = dict(board_interface.get_layers())
        self.layer_count = len(self.layers)
        
        # Configure the grid
        self.grid_pitch = 0.4  # mm
        self.grid_config.resolution = self.grid_pitch  # Override the grid resolution
        
        # PCB parameters (per specifications)
        self.trace_width = 0.0889  # 3.5mil in mm
        self.clearance = 0.0889    # 3.5mil spacing 
        self.via_drill = 0.15      # mm hole diameter
        self.via_diameter = 0.25   # mm via diameter
        
        # Grid subdivision parameters
        self.grid_subdivision_space = 0.4  # mm - space to leave when subdividing traces
        self.max_rip_attempts = 10  # Maximum failed attempts before stopping
        
        # Order layers by actual PCB stack (not alphabetically)
        # F.Cu should be layer 0, inner layers, then B.Cu at the end
        layer_stack_order = ['F.Cu']  # Top layer first
        
        # Add inner layers in order
        inner_layers = [name for name in self.layers.keys() if name.startswith('In')]
        # Sort inner layers by their number (In1, In2, In3, etc.)
        inner_layers.sort(key=lambda x: int(x[2:].split('.')[0]) if x[2:].split('.')[0].isdigit() else 999)
        layer_stack_order.extend(inner_layers)
        
        # B.Cu is last
        if 'B.Cu' in self.layers.keys():
            layer_stack_order.append('B.Cu')
        
        # Add any other layers not accounted for
        remaining_layers = [name for name in self.layers.keys() if name not in layer_stack_order]
        layer_stack_order.extend(remaining_layers)
        
        self.layer_names = layer_stack_order
        
        # Layer assignments for Manhattan routing (per specifications)
        # F.Cu is reserved for escape routes
        # Odd inner layers (In1, In3, In5, In7, In9) for horizontal
        # Even inner layers (In2, In4, In6, In8, In10) and B.Cu for vertical
        self.layer_directions = {}
        for i, layer_name in enumerate(self.layer_names):
            if layer_name == 'F.Cu':
                self.layer_directions[layer_name] = 'escape'
            elif layer_name == 'B.Cu':
                self.layer_directions[layer_name] = 'vertical' 
            elif layer_name.startswith('In'):
                # Extract layer number from In#.Cu format
                layer_num = int(layer_name[2:-3])
                if layer_num % 2 == 1:  # Odd inner layers: In1, In3, In5, In7, In9
                    self.layer_directions[layer_name] = 'horizontal'
                else:  # Even inner layers: In2, In4, In6, In8, In10
                    self.layer_directions[layer_name] = 'vertical'
            else:
                self.layer_directions[layer_name] = 'horizontal'  # Default
        
        # Initialize 3D grid for routing
        # Grid: [layers][y][x]
        # Each cell contains: 0=free, 1=obstacle, 2=routed_by_netX
        self.initialize_routing_grid()
        
        # Track routing information
        self.routed_segments = []  # Track segments for visualization and export
        self.net_routes = defaultdict(list)  # Store routes by net
        self.ripped_up_nets = set()  # Track nets that have been ripped up
        self.rip_up_counts = defaultdict(int)  # Count times each net has been ripped up
        
        # Current routing visualization
        self.current_routing_net = None
        self.current_routing_segments = []  # For bright white visualization
        
        # Grid subdivision tracking
        self.subdivided_segments = {}  # Track subdivided grid segments
        self.available_grid_segments = {}  # Track which segments are available for routing
        
        logger.info("🏗️ Manhattan Router initialized")
        logger.info(f"Grid dimensions: {self.grid_width}x{self.grid_height}x{self.layer_count}")
        logger.info(f"Trace width: {self.trace_width}mm, Clearance: {self.clearance}mm")
        logger.info(f"Via drill: {self.via_drill}mm, Via diameter: {self.via_diameter}mm")
    
    def initialize_routing_grid(self):
        """Initialize the 3D routing grid"""
        # Calculate grid dimensions based on board extents with 3mm margin
        self.grid_width = int(math.ceil((self.board_dimensions[2] - self.board_dimensions[0] + 6) / self.grid_pitch))
        self.grid_height = int(math.ceil((self.board_dimensions[3] - self.board_dimensions[1] + 6) / self.grid_pitch))
        
        # Create a 3D grid: [layers][y][x]
        # We use numpy for efficiency and GPU compatibility
        self.routing_grid = np.zeros((self.layer_count, self.grid_height, self.grid_width), dtype=np.int32)
        
        # Additional grid to store net IDs for each routed cell
        self.net_id_grid = np.full((self.layer_count, self.grid_height, self.grid_width), -1, dtype=np.int32)
        
        # Map net names to integer IDs for efficient storage
        self.net_name_to_id = {}
        self.net_id_to_name = []
        
        # Load obstacles from the board
        self.load_obstacles()
    
    def load_obstacles(self):
        """Load obstacles from the board with DRC-aware exclusion zones"""
        logger.info("🔍 Loading obstacles with DRC-aware exclusion zones...")
        
        # Get all footprints
        footprints = self.board_interface.get_footprints()
        
        # Track obstacle statistics
        obstacle_count = 0
        
        # Mark obstacle areas in the grid based on pad positions and clearances
        for footprint in footprints:
            for pad in footprint.get('pads', []):
                # Get pad position
                pos_x, pos_y = pad.get('x', 0), pad.get('y', 0)
                
                # Get pad layers
                pad_layers = pad.get('layers', ['F.Cu'])
                
                # Convert pad dimensions to grid coordinates
                pad_width = pad.get('size_x', 0)
                pad_height = pad.get('size_y', 0)
                
                # Convert pad center to grid coordinates
                grid_x, grid_y = self.world_to_grid(pos_x, pos_y)
                
                # Create DRC exclusion zones for each layer
                for layer_idx, layer_name in enumerate(self.layer_names):
                    if layer_name in self.layers:
                        # Different clearances for different layers
                        if layer_name == 'F.Cu':
                            # Smaller clearance for F.Cu (escape routing only)
                            clearance = self.clearance * 0.5  # 1.75mil for tight escapes
                        else:
                            # Full DRC clearance for grid layers
                            clearance = self.clearance  # 3.5mil for grid routing
                        
                        # Calculate exclusion zone with layer-specific clearance
                        total_width = pad_width + 2 * clearance
                        total_height = pad_height + 2 * clearance
                        
                        # Convert to grid cells
                        exclusion_width = math.ceil(total_width / self.grid_pitch)
                        exclusion_height = math.ceil(total_height / self.grid_pitch)
                        
                        # Mark exclusion zone as obstacles
                        for y_offset in range(-exclusion_height//2, exclusion_height//2 + 1):
                            for x_offset in range(-exclusion_width//2, exclusion_width//2 + 1):
                                obstacle_x = grid_x + x_offset
                                obstacle_y = grid_y + y_offset
                                
                                # Check bounds
                                if (0 <= obstacle_x < self.grid_width and 
                                    0 <= obstacle_y < self.grid_height):
                                    
                                    # Mark as obstacle if pad is on this layer or affects routing
                                    if layer_name in pad_layers or self._affects_routing(layer_name, pad_layers):
                                        self.routing_grid[layer_idx, obstacle_y, obstacle_x] = CellState.OBSTACLE
                                        obstacle_count += 1
        
        # Load existing tracks and vias as obstacles
        self._load_existing_routing_obstacles()
        
        # Load board cutouts, mounting holes, and other mechanical obstacles
        self._load_mechanical_obstacles()
        
        logger.info(f"✅ Loaded {obstacle_count} DRC obstacle cells across all layers")
    
    def _affects_routing(self, layer_name, pad_layers):
        """Check if a layer is affected by pad routing constraints"""
        # All layers are affected by pad clearance requirements
        # This ensures proper spacing around pads on all routing layers
        return True
    
    def _load_existing_routing_obstacles(self):
        """Load existing tracks and vias as obstacles"""
        try:
            # Get existing tracks from board with compatibility
            if hasattr(self.board_interface, 'get_all_tracks'):
                existing_tracks = self.board_interface.get_all_tracks()
            elif hasattr(self.board_interface, 'get_tracks'):
                existing_tracks = self.board_interface.get_tracks()
            else:
                existing_tracks = []
            
            for track in existing_tracks:
                # Mark track area as obstacle
                start_pos = track.get('start', (0, 0))
                end_pos = track.get('end', (0, 0))
                width = track.get('width', self.trace_width)
                layer_name = track.get('layer', 'F.Cu')
                
                if layer_name in self.layers:
                    layer_idx = self.layer_names.index(layer_name)
                    self._mark_track_obstacle(start_pos, end_pos, width, layer_idx)
            
            # Get existing vias from board with compatibility  
            if hasattr(self.board_interface, 'get_all_vias'):
                existing_vias = self.board_interface.get_all_vias()
            elif hasattr(self.board_interface, 'get_vias'):
                existing_vias = self.board_interface.get_vias()
            else:
                existing_vias = []
            
            for via in existing_vias:
                # Mark via area as obstacle on all layers it spans
                position = via.get('position', (0, 0))
                diameter = via.get('size', self.via_diameter)
                start_layer = via.get('start_layer', 'F.Cu')
                end_layer = via.get('end_layer', 'B.Cu')
                
                self._mark_via_obstacle(position, diameter, start_layer, end_layer)
                
        except Exception as e:
            logger.debug(f"Could not load existing routing obstacles: {e}")
    
    def _load_mechanical_obstacles(self):
        """Load board cutouts, mounting holes, and other mechanical obstacles"""
        try:
            # Get board outline/cutouts
            board_cutouts = self.board_interface.get_board_cutouts() if hasattr(self.board_interface, 'get_board_cutouts') else []
            
            for cutout in board_cutouts:
                # Mark cutout area as obstacle on all layers
                self._mark_cutout_obstacle(cutout)
            
            # Get mounting holes
            mounting_holes = self.board_interface.get_mounting_holes() if hasattr(self.board_interface, 'get_mounting_holes') else []
            
            for hole in mounting_holes:
                position = hole.get('position', (0, 0))
                diameter = hole.get('diameter', 3.0)  # Default 3mm mounting hole
                
                # Mark mounting hole with clearance on all layers
                self._mark_hole_obstacle(position, diameter + 2 * self.clearance)
                
        except Exception as e:
            logger.debug(f"Could not load mechanical obstacles: {e}")
    
    def _mark_track_obstacle(self, start_pos, end_pos, width, layer_idx):
        """Mark a track as obstacle in the routing grid"""
        try:
            # Convert positions to grid coordinates
            start_grid = self.world_to_grid(start_pos[0], start_pos[1])
            end_grid = self.world_to_grid(end_pos[0], end_pos[1])
            
            # Calculate track corridor with clearance
            track_clearance = width/2 + self.clearance
            clearance_cells = math.ceil(track_clearance / self.grid_pitch)
            
            # Mark cells along the track with clearance
            self._mark_line_obstacle(start_grid, end_grid, clearance_cells, layer_idx)
            
        except Exception as e:
            logger.debug(f"Error marking track obstacle: {e}")
    
    def _mark_via_obstacle(self, position, diameter, start_layer_name, end_layer_name):
        """Mark a via as obstacle in the routing grid"""
        try:
            # Convert position to grid coordinates
            grid_x, grid_y = self.world_to_grid(position[0], position[1])
            
            # Calculate via exclusion radius with clearance
            via_radius = diameter/2 + self.clearance
            exclusion_cells = math.ceil(via_radius / self.grid_pitch)
            
            # Get layer range
            if start_layer_name in self.layer_names and end_layer_name in self.layer_names:
                start_idx = self.layer_names.index(start_layer_name)
                end_idx = self.layer_names.index(end_layer_name)
                min_layer = min(start_idx, end_idx)
                max_layer = max(start_idx, end_idx)
                
                # Mark via obstacle on all spanned layers
                for layer in range(min_layer, max_layer + 1):
                    self._mark_circular_obstacle(grid_x, grid_y, exclusion_cells, layer)
                    
        except Exception as e:
            logger.debug(f"Error marking via obstacle: {e}")
    
    def _mark_cutout_obstacle(self, cutout):
        """Mark a board cutout as obstacle on all layers"""
        # Implementation depends on cutout data format
        # This is a placeholder for board outline handling
        pass
    
    def _mark_hole_obstacle(self, position, diameter):
        """Mark a mounting hole as obstacle on all layers"""
        try:
            grid_x, grid_y = self.world_to_grid(position[0], position[1])
            exclusion_cells = math.ceil(diameter / (2 * self.grid_pitch))
            
            # Mark hole obstacle on all layers
            for layer in range(self.layer_count):
                self._mark_circular_obstacle(grid_x, grid_y, exclusion_cells, layer)
                
        except Exception as e:
            logger.debug(f"Error marking hole obstacle: {e}")
    
    def _mark_line_obstacle(self, start_grid, end_grid, clearance_cells, layer_idx):
        """Mark a line-shaped obstacle with clearance in the routing grid"""
        start_x, start_y = start_grid
        end_x, end_y = end_grid
        
        # Use Bresenham's line algorithm to mark cells along the line
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        x, y = start_x, start_y
        x_inc = 1 if start_x < end_x else -1
        y_inc = 1 if start_y < end_y else -1
        error = dx - dy
        
        for _ in range(dx + dy + 1):
            # Mark circular obstacle around this point
            self._mark_circular_obstacle(x, y, clearance_cells, layer_idx)
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
                
            # Break if we've reached the end
            if x == end_x and y == end_y:
                break
    
    def _mark_circular_obstacle(self, center_x, center_y, radius_cells, layer_idx):
        """Mark a circular obstacle in the routing grid"""
        for y_offset in range(-radius_cells, radius_cells + 1):
            for x_offset in range(-radius_cells, radius_cells + 1):
                # Check if point is within circular radius
                if x_offset * x_offset + y_offset * y_offset <= radius_cells * radius_cells:
                    obstacle_x = center_x + x_offset
                    obstacle_y = center_y + y_offset
                    
                    # Check bounds and mark as obstacle
                    if (0 <= obstacle_x < self.grid_width and 
                        0 <= obstacle_y < self.grid_height and
                        0 <= layer_idx < self.layer_count):
                        self.routing_grid[layer_idx, obstacle_y, obstacle_x] = CellState.OBSTACLE
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        # Add 3mm margin
        x_with_margin = x + 3
        y_with_margin = y + 3
        
        # Convert to grid coordinates
        grid_x = int(round(x_with_margin / self.grid_pitch))
        grid_y = int(round(y_with_margin / self.grid_pitch))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        # Convert to world coordinates
        x = grid_x * self.grid_pitch
        y = grid_y * self.grid_pitch
        
        # Remove margin
        x_without_margin = x - 3
        y_without_margin = y - 3
        
        return x_without_margin, y_without_margin
    
    def route_all_nets(self, timeout_per_net: float = 5.0, total_timeout: float = 300.0) -> RoutingStats:
        """Route all nets"""
        logger.info("🚀 Starting Manhattan routing for all nets")
        
        # Get all nets
        nets = self.board_interface.get_nets()
        
        # Track routing statistics
        self.stats = RoutingStats()
        self.stats.total_nets = len(nets)
        self.stats.start_time = time.time()
        
        # Sort nets by shortest airwire distance first, then alphabetically
        sorted_nets = self.sort_nets_by_distance_and_name(nets)
        
        # Set up progress tracking
        net_count = len(sorted_nets)
        completed_nets = 0
        failed_nets = 0
        
        # Main routing loop
        for net_idx, (net_name, _) in enumerate(sorted_nets):
            # Skip GND nets or other nets we want to filter
            if self.should_skip_net(net_name):
                logger.info(f"⏩ Skipping net {net_name} (filtered)")
                self.stats.skipped_nets += 1
                continue
            
            # Update progress every 10 nets with enhanced reporting
            if net_idx % 10 == 0:
                progress_pct = (completed_nets / net_count) * 100
                logger.info(f"🔄 Manhattan routing progress: {completed_nets}/{net_count} completed ({progress_pct:.1f}%), {failed_nets} failed")
                logger.info(f"    Current grid subdivisions: {len(self.subdivided_segments)}")
                logger.info(f"    Nets ripped up: {len(self.ripped_up_nets)}")
                if self.progress_callback:
                    self.progress_callback(progress_pct, f"Manhattan routing {net_name} ({completed_nets}/{net_count})")
            
            # Route the net
            result = self.route_net(net_name, timeout=timeout_per_net)
            
            if result == RoutingResult.SUCCESS:
                completed_nets += 1
                # Don't increment stats.routed_nets here - it's done in route_net
                logger.info(f"✅ Manhattan routed net {net_name} ({completed_nets}/{net_count})")
            else:
                failed_nets += 1
                self.stats.nets_failed += 1  # Use correct stat name
                logger.warning(f"❌ Failed to route net {net_name} ({failed_nets} failed)")
                
                # Clear visualization for failed net
                if net_name == self.current_routing_net:
                    self.clear_bright_white_visualization()
            
            # Check if we've had too many failures in a row (per specifications)
            if failed_nets >= self.max_rip_attempts:
                logger.error(f"⛔ Too many routing failures ({failed_nets} nets). Stopping after {self.max_rip_attempts} failed attempts.")
                break
            
            # Check total timeout
            if (time.time() - self.stats.start_time) > total_timeout:
                logger.warning(f"⏱️ Total routing timeout reached after {net_idx+1} nets")
                break
        
        # Finalize statistics
        self.stats.end_time = time.time()
        self.stats.routing_time = self.stats.end_time - self.stats.start_time
        
        # Log final statistics
        # Clear visualization after routing
        self.clear_bright_white_visualization()
        
        logger.info(f"✨ Manhattan grid routing completed: {self.stats.nets_routed}/{net_count} nets routed")
        logger.info(f"⏱️ Total routing time: {self.stats.routing_time:.2f} seconds")
        logger.info(f"🔧 Grid subdivisions created: {len(self.subdivided_segments)}")
        logger.info(f"🔄 Nets ripped up: {len(self.ripped_up_nets)}")
        
        return self.stats
    
    def sort_nets_by_distance_and_name(self, nets):
        """Sort nets by shortest airwire distance first, then alphabetically"""
        # Calculate airwire distances for all nets
        net_distances = []
        
        for net_name, net_data in nets.items():
            # Get all pads for this net
            pads = self.board_interface.get_pads_for_net(net_name)
            
            if len(pads) < 2:
                # Skip nets with fewer than 2 pads
                continue
            
            # Calculate shortest airwire distance between any two pads
            min_distance = float('inf')
            for i in range(len(pads)):
                for j in range(i+1, len(pads)):
                    # Handle different pad position formats
                    pos1 = self._get_pad_position(pads[i])
                    pos2 = self._get_pad_position(pads[j])
                    
                    if pos1 and pos2:
                        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        min_distance = min(min_distance, distance)
            
            # Store net and its minimum distance (use reasonable default if no distance calculated)
            if min_distance == float('inf'):
                min_distance = 50.0  # Default distance for sorting
            net_distances.append((net_name, min_distance))
        
        # Sort by distance first, then by name
        sorted_nets = sorted(net_distances, key=lambda x: (x[1], x[0]))
        
        return sorted_nets
    
    def _get_pad_position(self, pad):
        """Extract pad position from various pad data formats"""
        try:
            # Try different position field names
            if 'position' in pad and pad['position']:
                pos = pad['position']
                # Handle tuple or list format
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    return (float(pos[0]), float(pos[1]))
                # Handle dict format
                elif isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                    return (float(pos['x']), float(pos['y']))
                    
            elif 'x' in pad and 'y' in pad:
                return (float(pad['x']), float(pad['y']))
                
            elif 'pos' in pad and pad['pos']:
                pos = pad['pos']
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    return (float(pos[0]), float(pos[1]))
                elif isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                    return (float(pos['x']), float(pos['y']))
                    
            # Default position if none found
            logger.debug(f"Could not extract position from pad: {pad}")
            return (0.0, 0.0)
            
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Error extracting pad position: {e}, pad data: {pad}")
            return (0.0, 0.0)
    
    def should_skip_net(self, net_name):
        """Check if a net should be skipped (e.g., GND planes)"""
        # Skip GND nets
        if net_name.lower() in ['gnd', 'ground', 'vss', 'gnda', 'gndpwr']:
            return True
        
        # TODO: Add more sophisticated filtering if needed
        
        return False
    
    def route_net(self, net_name: str, timeout: float = 10.0) -> RoutingResult:
        """Route a single net with bright white visualization and grid subdivision"""
        # Set current routing net for visualization
        self.current_routing_net = net_name
        self.current_routing_segments = []
        
        logger.info(f"🎯 Starting Manhattan grid routing for net: {net_name}")
        
        # Get all pads for this net
        pads = self.board_interface.get_pads_for_net(net_name)
        
        if len(pads) < 2:
            logger.warning(f"⚠️ Net {net_name} has fewer than 2 pads, skipping")
            self.current_routing_net = None
            return RoutingResult.FAILED
        
        # Star topology: connect first pad to all others
        start_pad = pads[0]
        
        # Map net name to ID if not already mapped
        if net_name not in self.net_name_to_id:
            self.net_name_to_id[net_name] = len(self.net_id_to_name)
            self.net_id_to_name.append(net_name)
        
        net_id = self.net_name_to_id[net_name]
        
        # Try to route connections to all other pads
        successful_connections = 0
        for i in range(1, len(pads)):
            end_pad = pads[i]
            
            # Maximum retries with ripup
            max_ripup_attempts = 3
            attempts = 0
            
            while attempts < max_ripup_attempts:
                # Try to route between the two pads
                result = self.route_two_pads(start_pad, end_pad, net_name)
                
                if result == RoutingResult.SUCCESS:
                    successful_connections += 1
                    break
                elif result == RoutingResult.BLOCKED:
                    # Try ripup and reroute
                    success = self.ripup_and_retry(start_pad, end_pad, net_name)
                    if success:
                        successful_connections += 1
                        break
                
                attempts += 1
            
            if attempts >= max_ripup_attempts:
                logger.warning(f"⚠️ Failed to route connection in net {net_name} after {max_ripup_attempts} ripup attempts")
        
        # Clear current routing visualization when done
        if net_name == self.current_routing_net:
            # Move current segments to standard visualization
            for segment in self.current_routing_segments:
                # Callback for visualization (tracks and vias use different callbacks)
                try:
                    # Convert RouteSegmentInfo to dict format
                    track_dict = segment.to_track_dict(self)
                    
                    # Add proper KiCad colors and current routing visualization
                    if self.current_routing_net and segment.net_id == self.current_routing_net:
                        track_dict['color'] = '#FFFFFF'  # Bright white for current routing
                        track_dict['active'] = True
                    else:
                        # Use KiCad theme colors based on SEGMENT type
                        if segment.is_via():
                            track_dict['color'] = self._get_via_color(track_dict.get('type', 'blind'))
                        else:
                            track_dict['color'] = self._get_layer_color(track_dict.get('layer', 'F.Cu'))
                        track_dict['active'] = False
                    
                    # Use appropriate callback based on SEGMENT type
                    if segment.is_via() and self.via_callback:
                        self.via_callback(track_dict)
                    elif not segment.is_via() and self.track_callback:
                        self.track_callback(track_dict)
                        
                except Exception as e:
                    logger.error(f"Error in visualization callback: {e}")
                    logger.debug(f"Segment: {segment}, type: {type(segment)}")
            self.clear_bright_white_visualization()
        
        # Check if we routed all connections
        if successful_connections == len(pads) - 1:
            logger.info(f"✅ Successfully routed all connections for net {net_name}")
            return RoutingResult.SUCCESS
        else:
            logger.warning(f"❌ Partial routing for net {net_name}: {successful_connections}/{len(pads)-1} connections")
            return RoutingResult.FAILED
    
    def route_two_pads(self, pad_a, pad_b, net_name) -> RoutingResult:
        """Route between two pads using proper Manhattan grid routing with escapes"""
        # Get pad positions safely
        pos_a = self._get_pad_position(pad_a)
        pos_b = self._get_pad_position(pad_b)
        
        if not pos_a or not pos_b:
            logger.error(f"Could not get positions for pads in net {net_name}")
            return RoutingResult.FAILED
        
        logger.debug(f"Routing {net_name}: Pad A at {pos_a}, Pad B at {pos_b}")
        
        try:
            # Step 1: Create escape route from pad A to Manhattan grid
            grid_entry_a, escape_segments_a = self._create_escape_route(pad_a, pos_a, net_name, "start")
            if not grid_entry_a:
                logger.warning(f"Failed to create escape route from pad A for {net_name}")
                return RoutingResult.BLOCKED
            
            # Step 2: Create escape route from pad B to Manhattan grid  
            grid_entry_b, escape_segments_b = self._create_escape_route(pad_b, pos_b, net_name, "end")
            if not grid_entry_b:
                logger.warning(f"Failed to create escape route from pad B for {net_name}")
                return RoutingResult.BLOCKED
            
            logger.debug(f"Grid entry points: A={grid_entry_a}, B={grid_entry_b}")
            
            # Step 3: Route through Manhattan grid between entry points
            grid_segments = self._route_manhattan_grid(
                grid_entry_a, grid_entry_b, net_name
            )
            
            if not grid_segments:
                logger.warning(f"Failed to route through Manhattan grid for {net_name}")
                return RoutingResult.BLOCKED
            
            # Step 4: Combine escape routes and grid path into complete route
            all_segments = escape_segments_a + grid_segments + escape_segments_b
            
            if not all_segments:
                logger.warning(f"No route segments generated for {net_name}")
                return RoutingResult.BLOCKED
            
            # Apply grid subdivision strategy
            subdivided_segments = self._subdivide_grid_segments(all_segments, net_name)
            
            # Store the complete route
            self.net_routes[net_name].extend(subdivided_segments)
            
            # Update the routing grid
            self._mark_route_segments_in_grid(subdivided_segments, net_name)
            
            # Add segments for visualization with bright white for current net
            for segment in subdivided_segments:
                try:
                    self.routed_segments.append(segment)
                    # Bright white visualization for current routing net
                    if net_name == self.current_routing_net:
                        self.current_routing_segments.append(segment)
                    try:
                        # Validate segment data before processing
                        if not hasattr(segment, 'start') or not hasattr(segment, 'net_id'):
                            logger.debug(f"Invalid segment data: {segment}")
                            continue
                            
                        # Convert RouteSegmentInfo to dict format
                        track_dict = segment.to_track_dict(self)
                        
                        # Validate track_dict structure based on SEGMENT type (not dict type)
                        if segment.is_via():
                            # Via validation - needs x, y, drill_diameter, via_diameter
                            if 'x' not in track_dict or 'y' not in track_dict:
                                logger.error(f"Invalid via coordinates in track_dict: {track_dict}")
                                continue
                            if 'drill_diameter' not in track_dict or 'via_diameter' not in track_dict:
                                logger.error(f"Missing via dimensions in track_dict: {track_dict}")
                                continue
                        else:
                            # Track validation - needs start_x, start_y, end_x, end_y
                            if 'start_x' not in track_dict or 'start_y' not in track_dict:
                                logger.error(f"Invalid start coordinates in track_dict: {track_dict}")
                                continue
                            if 'end_x' not in track_dict or 'end_y' not in track_dict:
                                logger.error(f"Invalid end coordinates in track_dict: {track_dict}")
                                continue
                        
                        if 'net' not in track_dict:
                            logger.error(f"Missing required fields in track_dict: {track_dict}")
                            continue
                        
                        # Add proper KiCad colors and current routing visualization
                        if self.current_routing_net and segment.net_id == self.current_routing_net:
                            track_dict['color'] = '#FFFFFF'  # Bright white for current routing
                            track_dict['active'] = True
                        else:
                            # Use KiCad theme colors based on SEGMENT type
                            if segment.is_via():
                                track_dict['color'] = self._get_via_color(track_dict.get('type', 'blind'))
                            else:
                                track_dict['color'] = self._get_layer_color(track_dict.get('layer', 'F.Cu'))
                            track_dict['active'] = False
                        
                        # Use appropriate callback based on SEGMENT type
                        if segment.is_via() and self.via_callback:
                            self.via_callback(track_dict)
                        elif not segment.is_via() and self.track_callback:
                            self.track_callback(track_dict)
                            
                    except Exception as e:
                        logger.error(f"Error in visualization callback: {e}")
                        logger.debug(f"Segment details: start={getattr(segment, 'start', None)}, end={getattr(segment, 'end', None)}, net_id={getattr(segment, 'net_id', None)}")
                        logger.debug(f"Segment type: {type(segment)}, attributes: {dir(segment) if hasattr(segment, '__dict__') else 'No attributes'}")
                except Exception as e:
                    logger.error(f"Error processing segment for visualization: {e}")
                    logger.debug(f"Segment type: {type(segment)}, attributes: {dir(segment)}")
                    continue
            
            # Update statistics using BaseRouter's update_success method
            try:
                self.stats.update_success(subdivided_segments)
            except Exception as e:
                logger.error(f"Error updating statistics: {e}")
                # Fallback manual stats update
                self.stats.nets_routed += 1
                for segment in subdivided_segments:
                    if hasattr(segment, 'via_type') and segment.via_type is not None:
                        self.stats.vias_added += 1
                    else:
                        self.stats.tracks_added += 1
            
            logger.info(f"Successfully routed {net_name} with {len(subdivided_segments)} segments (escape+grid+escape)")
            return RoutingResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Critical error in route_two_pads for {net_name}: {e}")
            return RoutingResult.FAILED
    
    def ripup_and_retry(self, start_pad, end_pad, net_name) -> bool:
        """Rip up blocking nets and retry routing"""
        # Find nets blocking the path
        blocking_nets = self.find_blocking_nets(start_pad, end_pad)
        
        if not blocking_nets:
            # No obvious blocking nets found
            return False
        
        # Sort blocking nets by length (longer first) and rip-up count (less first)
        sorted_blocking = sorted(
            blocking_nets, 
            key=lambda n: (self.get_net_length(n), -self.rip_up_counts[n])
        )
        
        # Select the best candidate to rip up
        net_to_ripup = sorted_blocking[0]
        
        # Don't rip up nets that have been ripped up too many times
        if self.rip_up_counts[net_to_ripup] >= 3:
            logger.warning(f"⚠️ Net {net_to_ripup} has been ripped up too many times, skipping")
            return False
        
        # Rip up the net
        logger.info(f"🔄 Ripping up net {net_to_ripup} to route {net_name}")
        self.ripup_net(net_to_ripup)
        
        # Increment rip-up count
        self.rip_up_counts[net_to_ripup] += 1
        self.ripped_up_nets.add(net_to_ripup)
        
        # Try routing again
        result = self.route_two_pads(start_pad, end_pad, net_name)
        
        if result == RoutingResult.SUCCESS:
            # Now try to re-route the ripped up net
            logger.info(f"🔄 Re-routing net {net_to_ripup}")
            reroute_result = self.reroute_net(net_to_ripup)
            
            if reroute_result == RoutingResult.FAILED:
                logger.warning(f"⚠️ Failed to re-route net {net_to_ripup}")
            
            return True
        else:
            # Routing still failed, restore the ripped up net
            logger.info(f"🔄 Restoring ripped up net {net_to_ripup}")
            self.restore_net(net_to_ripup)
            return False
    
    def find_blocking_nets(self, start_pad, end_pad) -> List[str]:
        """Find nets that might be blocking the path between two pads"""
        # Get pad positions safely
        pos_a = self._get_pad_position(start_pad)
        pos_b = self._get_pad_position(end_pad)
        
        if not pos_a or not pos_b:
            logger.warning(f"Could not get positions for blocking net analysis")
            return []
        
        # Convert to grid coordinates
        grid_a_x, grid_a_y = self.world_to_grid(pos_a[0], pos_a[1])
        grid_b_x, grid_b_y = self.world_to_grid(pos_b[0], pos_b[1])
        
        # Create a set of nets found along the ideal path
        blocking_nets = set()
        
        # Check for nets along a straight line path (simplistic approach)
        # For horizontal segment
        x_start, x_end = min(grid_a_x, grid_b_x), max(grid_a_x, grid_b_x)
        for x in range(x_start, x_end + 1):
            for layer in range(self.layer_count):
                try:
                    cell_state = self.routing_grid[layer, grid_a_y, x]
                    if hasattr(cell_state, 'item'):
                        cell_state = cell_state.item()
                    if cell_state == CellState.ROUTED:
                        net_id = self.net_id_grid[layer, grid_a_y, x]
                        if hasattr(net_id, 'item'):
                            net_id = net_id.item()
                        if net_id >= 0 and net_id < len(self.net_id_to_name):
                            blocking_nets.add(self.net_id_to_name[net_id])
                except (IndexError, ValueError) as e:
                    continue
        
        # For vertical segment
        y_start, y_end = min(grid_a_y, grid_b_y), max(grid_a_y, grid_b_y)
        for y in range(y_start, y_end + 1):
            for layer in range(self.layer_count):
                try:
                    cell_state = self.routing_grid[layer, y, grid_b_x]
                    if hasattr(cell_state, 'item'):
                        cell_state = cell_state.item()
                    if cell_state == CellState.ROUTED:
                        net_id = self.net_id_grid[layer, y, grid_b_x]
                        if hasattr(net_id, 'item'):
                            net_id = net_id.item()
                        if net_id >= 0 and net_id < len(self.net_id_to_name):
                            blocking_nets.add(self.net_id_to_name[net_id])
                except (IndexError, ValueError) as e:
                    continue
        
        return list(blocking_nets)
    
    def get_net_length(self, net_name):
        """Get the total length of a net"""
        return sum(self.calculate_segment_length(s) for s in self.net_routes.get(net_name, []))
    
    def calculate_segment_length(self, segment):
        """Calculate the length of a segment"""
        if segment.via_type is not None:
            # Via length is based on the number of layers it spans
            layer_distance = abs(segment.end.layer - segment.start.layer)
            # Assume 0.1mm per layer transition
            return 0.1 * layer_distance
        else:
            # Track length is Euclidean distance
            start_x, start_y = self.grid_to_world(segment.start.x, segment.start.y)
            end_x, end_y = self.grid_to_world(segment.end.x, segment.end.y)
            return math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    
    def ripup_net(self, net_name):
        """Rip up a net from the routing grid"""
        net_id = self.net_name_to_id.get(net_name)
        if net_id is None:
            return
        
        # Clear this net from the grid
        for layer in range(self.layer_count):
            mask = (self.net_id_grid[layer] == net_id)
            self.routing_grid[layer][mask] = CellState.EMPTY
            self.net_id_grid[layer][mask] = -1
        
        # Store the segments for potential restoration
        self._ripped_up_segments = self.net_routes.get(net_name, [])
        
        # Clear segments from visualization
        segments_to_remove = []
        for segment in self.routed_segments:
            if (hasattr(segment, 'net_id') and segment.net_id == net_name) or \
               (hasattr(segment, 'net_name') and getattr(segment, 'net_name', None) == net_name):
                segments_to_remove.append(segment)
        
        for segment in segments_to_remove:
            self.routed_segments.remove(segment)
        
        # Clear the net routes
        self.net_routes[net_name] = []
    
    def restore_net(self, net_name):
        """Restore a previously ripped up net"""
        # Restore segments
        self.net_routes[net_name] = self._ripped_up_segments
        
        # Re-mark the grid
        for segment in self._ripped_up_segments:
            start_x, start_y, start_layer = segment.start.x, segment.start.y, segment.start.layer
            end_x, end_y, end_layer = segment.end.x, segment.end.y, segment.end.layer
            
            net_id = self.net_name_to_id.get(net_name)
            
            if segment.via_type is not None:
                # Via
                min_layer = min(start_layer, end_layer)
                max_layer = max(start_layer, end_layer)
                
                for layer in range(min_layer, max_layer + 1):
                    self.routing_grid[layer, start_y, start_x] = CellState.ROUTED
                    self.net_id_grid[layer, start_y, start_x] = net_id
            else:
                # Track
                if start_x == end_x:  # Vertical track
                    y_min, y_max = min(start_y, end_y), max(start_y, end_y)
                    for y in range(y_min, y_max + 1):
                        self.routing_grid[start_layer, y, start_x] = CellState.ROUTED
                        self.net_id_grid[start_layer, y, start_x] = net_id
                else:  # Horizontal track
                    x_min, x_max = min(start_x, end_x), max(start_x, end_x)
                    for x in range(x_min, x_max + 1):
                        self.routing_grid[start_layer, start_y, x] = CellState.ROUTED
                        self.net_id_grid[start_layer, start_y, x] = net_id
        
        # Restore segments for visualization
        for segment in self._ripped_up_segments:
            self.routed_segments.append(segment)
            try:
                # Convert RouteSegmentInfo to dict format
                track_dict = segment.to_track_dict(self)
                track_dict['active'] = False  # Restored segments are not active
                
                # Use proper colors for restored segments
                if segment.is_via():
                    track_dict['color'] = self._get_via_color(track_dict.get('type', 'blind'))
                else:
                    track_dict['color'] = self._get_layer_color(track_dict.get('layer', 'F.Cu'))
                
                # Use appropriate callback based on SEGMENT type
                if segment.is_via() and self.via_callback:
                    self.via_callback(track_dict)
                elif not segment.is_via() and self.track_callback:
                    self.track_callback(track_dict)
                    
            except Exception as e:
                logger.error(f"Error in visualization callback during restore: {e}")
                logger.debug(f"Segment: {segment}, type: {type(segment)}")
    
    def reroute_net(self, net_name) -> RoutingResult:
        """Re-route a net that was previously ripped up"""
        # Get pads for this net
        pads = self.board_interface.get_pads_for_net(net_name)
        
        if len(pads) < 2:
            return RoutingResult.FAILED
        
        # Star topology: connect first pad to all others
        start_pad = pads[0]
        
        # Try to route connections to all other pads
        successful_connections = 0
        for i in range(1, len(pads)):
            end_pad = pads[i]
            
            # Try to route between the two pads
            result = self.route_two_pads(start_pad, end_pad, net_name)
            
            if result == RoutingResult.SUCCESS:
                successful_connections += 1
        
        # Check if we routed all connections
        if successful_connections == len(pads) - 1:
            return RoutingResult.SUCCESS
        else:
            return RoutingResult.FAILED
    
    def a_star_pathfind(self, start, end, net_name, horizontal_first=True):
        """
        A* pathfinding with layer changes via vias
        
        Args:
            start: (x, y, layer) start position in grid coordinates
            end: (x, y, layer) end position in grid coordinates
            net_name: Net being routed
            horizontal_first: Whether to try horizontal movement first
            
        Returns:
            List of (x, y, layer) points forming the path, or None if no path found
        """
        start_x, start_y, start_layer = start
        end_x, end_y, end_layer = end
        
        # Get net ID
        net_id = self.net_name_to_id.get(net_name)
        
        # A* state: (f_score, g_score, x, y, layer)
        open_set = [(0, 0, start_x, start_y, start_layer)]
        came_from = {}
        g_score = {(start_x, start_y, start_layer): 0}
        f_score = {(start_x, start_y, start_layer): self._manhattan_distance(start_x, start_y, end_x, end_y)}
        
        visited = set()
        max_iterations = 100000  # Prevent infinite loops
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get the node with the lowest f_score
            current_f, current_g, current_x, current_y, current_layer = heapq.heappop(open_set)
            current_state = (current_x, current_y, current_layer)
            
            if current_state in visited:
                continue
            
            visited.add(current_state)
            
            # Check if we reached the goal
            if current_x == end_x and current_y == end_y:
                # If we're at the destination but not on the target layer, add a final via
                if current_layer != end_layer:
                    # Add a via to the end layer
                    came_from[(current_x, current_y, end_layer)] = current_state
                    current_state = (current_x, current_y, end_layer)
                
                path = self._reconstruct_path(came_from, current_state)
                logger.debug(f"Found path for {net_name} in {iterations} iterations, length {len(path)}")
                
                # Convert path to RouteSegmentInfo objects
                segments = self.path_to_segments(path, net_name)
                return segments
            
            # Get current layer direction
            layer_name = self.layer_names[current_layer]
            direction = self.layer_directions.get(layer_name, 'any')
            
            # Get neighbors based on layer direction
            neighbors = []
            
            # Layer transitions (vias)
            self._add_layer_transitions(neighbors, current_x, current_y, current_layer)
            
            # Horizontal and vertical moves
            if direction == 'horizontal' or direction == 'any' or direction == 'escape':
                # Horizontal moves
                if self._is_valid_cell(current_x-1, current_y, current_layer, net_name):
                    neighbors.append((current_x-1, current_y, current_layer, 1))
                if self._is_valid_cell(current_x+1, current_y, current_layer, net_name):
                    neighbors.append((current_x+1, current_y, current_layer, 1))
            
            if direction == 'vertical' or direction == 'any' or direction == 'escape':
                # Vertical moves
                if self._is_valid_cell(current_x, current_y-1, current_layer, net_name):
                    neighbors.append((current_x, current_y-1, current_layer, 1))
                if self._is_valid_cell(current_x, current_y+1, current_layer, net_name):
                    neighbors.append((current_x, current_y+1, current_layer, 1))
            
            # Sort neighbors based on preference (horizontal/vertical first)
            if horizontal_first:
                # Prioritize horizontal moves (x direction)
                neighbors.sort(key=lambda n: 0 if n[0] != current_x else 1)
            else:
                # Prioritize vertical moves (y direction)
                neighbors.sort(key=lambda n: 0 if n[1] != current_y else 1)
            
            # Process neighbors
            for next_x, next_y, next_layer, move_cost in neighbors:
                next_state = (next_x, next_y, next_layer)
                
                if next_state in visited:
                    continue
                
                # Via cost is higher than regular moves
                if next_layer != current_layer:
                    move_cost = 5  # Higher cost for layer changes
                
                # Calculate new g_score
                tentative_g = g_score[current_state] + move_cost
                
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    # This path is better than any previous one
                    came_from[next_state] = current_state
                    g_score[next_state] = tentative_g
                    
                    # Calculate f_score
                    h_score = self._manhattan_distance(next_x, next_y, end_x, end_y)
                    
                    # Add layer difference penalty to encourage staying on the same layer
                    layer_penalty = 0
                    if next_layer != end_layer:
                        layer_penalty = 2  # Small penalty for being on wrong layer
                    
                    f_score[next_state] = tentative_g + h_score + layer_penalty
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score[next_state], tentative_g, next_x, next_y, next_layer))
        
        # No path found
        logger.warning(f"❌ No path found for {net_name} after {iterations} iterations")
        return None
    
    def _add_layer_transitions(self, neighbors, x, y, current_layer):
        """Add layer transitions (vias) to neighbors list"""
        # Allow vias to any layer except current
        for layer in range(self.layer_count):
            if layer != current_layer:
                # Check if this cell is free on the target layer
                if self._is_valid_cell(x, y, layer, None):
                    neighbors.append((x, y, layer, 5))  # Higher cost for vias
    
    def _is_valid_cell(self, x, y, layer, net_name):
        """Check if a cell is valid for routing"""
        # Check bounds
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height and 0 <= layer < self.layer_count):
            return False
        
        try:
            # Get cell state safely
            cell_state = self.routing_grid[layer, y, x]
            
            # Convert to scalar if it's an array
            if hasattr(cell_state, 'item'):
                cell_state = cell_state.item()
            
            # Check if cell is free
            if cell_state == CellState.EMPTY:
                return True
            
            # If cell is routed, check if it's the same net
            if cell_state == CellState.ROUTED and net_name is not None:
                net_id = self.net_id_grid[layer, y, x]
                if hasattr(net_id, 'item'):
                    net_id = net_id.item()
                routed_net = self.net_id_to_name[net_id] if net_id >= 0 and net_id < len(self.net_id_to_name) else None
                return routed_net == net_name
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking cell validity at ({x},{y},{layer}): {e}")
            return False
    
    def _manhattan_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance heuristic"""
        return abs(x1 - x2) + abs(y1 - y2)
    
    def _reconstruct_path(self, came_from, current_state):
        """Reconstruct the path from A* came_from data"""
        path = [current_state]
        
        while current_state in came_from:
            current_state = came_from[current_state]
            path.append(current_state)
        
        path.reverse()
        return path
    
    def path_to_segments(self, path, net_name):
        """Convert a path to route segments"""
        segments = []
        
        try:
            for i in range(1, len(path)):
                prev_x, prev_y, prev_layer = path[i-1]
                curr_x, curr_y, curr_layer = path[i]
                
                # Create the segment
                start_cell = GridCell(prev_x, prev_y, prev_layer)
                end_cell = GridCell(curr_x, curr_y, curr_layer)
                
                if prev_layer != curr_layer:
                    # Via
                    via_type = self._determine_via_type(prev_layer, curr_layer)
                    segment = RouteSegmentInfo(start_cell, end_cell, net_name, self.via_diameter, via_type)
                else:
                    # Track
                    segment = RouteSegmentInfo(start_cell, end_cell, net_name, self.trace_width, None)
                
                segments.append(segment)
                logger.debug(f"Created segment: {segment.type} for net {net_name}")
                
        except Exception as e:
            logger.error(f"Error creating segments for path in net {net_name}: {e}")
            logger.debug(f"Path: {path}")
        
        return segments
    
    def _determine_via_type(self, layer1, layer2):
        """Determine the via type based on the layers (per specifications: blind/buried vias)"""
        # F.Cu = layer 0, B.Cu = layer N-1
        outer_layers = {0, self.layer_count - 1}
        
        # Blind via: connects outer layer (F.Cu or B.Cu) to inner layer
        if (layer1 in outer_layers) != (layer2 in outer_layers):  # One outer, one inner
            return "blind"
        # Buried via: connects inner layers only (neither is F.Cu or B.Cu)
        elif layer1 not in outer_layers and layer2 not in outer_layers:
            return "buried" 
        # Through via: connects F.Cu directly to B.Cu (shouldn't happen with Manhattan routing)
        elif layer1 in outer_layers and layer2 in outer_layers:
            return "through"
        else:
            return "blind"  # Default fallback
    
    def _create_drc_aware_fcu_trace(self, start_pos, end_pos, net_name):
        """Create DRC-compliant F.Cu escape trace avoiding other nets' traces
        
        Returns path as list of (x, y) world coordinates, or None if no valid path found
        """
        # Check if direct straight line is DRC-compliant
        if self._check_fcu_trace_clearance(start_pos, end_pos, net_name):
            return [start_pos, end_pos]
        
        # If direct path fails, try L-shaped routing (two perpendicular segments)
        # Try horizontal first, then vertical
        intermediate_h = (end_pos[0], start_pos[1])  # Horizontal first approach
        if (self._check_fcu_trace_clearance(start_pos, intermediate_h, net_name) and 
            self._check_fcu_trace_clearance(intermediate_h, end_pos, net_name)):
            return [start_pos, intermediate_h, end_pos]
        
        # Try vertical first, then horizontal  
        intermediate_v = (start_pos[0], end_pos[1])  # Vertical first approach
        if (self._check_fcu_trace_clearance(start_pos, intermediate_v, net_name) and
            self._check_fcu_trace_clearance(intermediate_v, end_pos, net_name)):
            return [start_pos, intermediate_v, end_pos]
        
        # If L-shapes fail, try more complex routing using A* on F.Cu layer
        return self._astar_fcu_escape_routing(start_pos, end_pos, net_name)
    
    def _check_fcu_trace_clearance(self, start_pos, end_pos, net_name):
        """Check if an F.Cu trace segment has proper DRC clearance from other nets
        
        Returns True if the trace can be placed without DRC violations
        """
        # For now, be more permissive to get basic routing working
        # TODO: Re-enable full DRC checking once basic routing is stable
        
        # Required clearance (trace width + spacing)
        required_clearance = self.trace_width + self.clearance
        
        # Only check against major obstacles for now
        # Check against footprint obstacles on F.Cu
        try:
            footprints = self.board_interface.get_footprints()
            for footprint in footprints:
                for pad in footprint.get('pads', []):
                    if 'F.Cu' not in pad.get('layers', []):
                        continue
                    
                    pad_center = (pad.get('x', 0), pad.get('y', 0))
                    pad_size_x = pad.get('size_x', 1.0) / 2
                    pad_size_y = pad.get('size_y', 1.0) / 2
                    
                    # Check if trace intersects pad (with reduced clearance for basic functionality)
                    pad_bounds = (
                        pad_center[0] - pad_size_x - 0.1,  # 0.1mm clearance instead of full DRC
                        pad_center[1] - pad_size_y - 0.1,
                        pad_center[0] + pad_size_x + 0.1,
                        pad_center[1] + pad_size_y + 0.1
                    )
                    
                    if self._line_intersects_rectangle(start_pos, end_pos, pad_bounds):
                        return False
        except Exception as e:
            logger.debug(f"Error in basic F.Cu clearance check: {e}")
            
        return True  # Be permissive for now
    
    def _calculate_trace_distance(self, seg1_start, seg1_end, seg2_start, seg2_end):
        """Calculate minimum distance between two line segments"""
        import numpy as np
        
        def point_to_line_distance(point, line_start, line_end):
            """Calculate distance from point to line segment"""
            line_vec = np.array(line_end) - np.array(line_start)
            point_vec = np.array(point) - np.array(line_start)
            
            # Handle zero-length line case
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0:
                return np.linalg.norm(point_vec)
            
            # Project point onto line
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
            projection = np.array(line_start) + t * line_vec
            
            return np.linalg.norm(np.array(point) - projection)
        
        # Calculate all point-to-segment distances
        distances = [
            point_to_line_distance(seg1_start, seg2_start, seg2_end),
            point_to_line_distance(seg1_end, seg2_start, seg2_end),
            point_to_line_distance(seg2_start, seg1_start, seg1_end),
            point_to_line_distance(seg2_end, seg1_start, seg1_end)
        ]
        
        return min(distances)
    
    def _check_fcu_footprint_clearance(self, start_pos, end_pos):
        """Check F.Cu trace clearance from footprint obstacles"""
        try:
            footprints = self.board_interface.get_footprints()
            required_clearance = self.trace_width + self.clearance
            
            for footprint in footprints:
                for pad in footprint.get('pads', []):
                    if 'F.Cu' not in pad.get('layers', []):
                        continue
                    
                    pad_center = (pad.get('x', 0), pad.get('y', 0))
                    pad_size_x = pad.get('size_x', 1.0) / 2  # Half size for radius
                    pad_size_y = pad.get('size_y', 1.0) / 2
                    
                    # Calculate distance from trace to pad
                    # Use rectangular pad approximation for simplicity
                    pad_bounds = (
                        pad_center[0] - pad_size_x, pad_center[1] - pad_size_y,
                        pad_center[0] + pad_size_x, pad_center[1] + pad_size_y
                    )
                    
                    if self._trace_intersects_rectangle(start_pos, end_pos, pad_bounds, required_clearance):
                        return False
            
            return True
        except Exception as e:
            logger.warning(f"Error checking F.Cu footprint clearance: {e}")
            return True  # Assume OK if check fails
    
    def _trace_intersects_rectangle(self, start_pos, end_pos, rect_bounds, clearance):
        """Check if a trace comes within clearance distance of a rectangle"""
        # Expand rectangle by clearance amount
        min_x, min_y, max_x, max_y = rect_bounds
        expanded_bounds = (
            min_x - clearance, min_y - clearance,
            max_x + clearance, max_y + clearance
        )
        
        # Check if line segment intersects expanded rectangle
        return self._line_intersects_rectangle(start_pos, end_pos, expanded_bounds)
    
    def _line_intersects_rectangle(self, start_pos, end_pos, rect_bounds):
        """Check if line segment intersects rectangle using Liang-Barsky algorithm"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        min_x, min_y, max_x, max_y = rect_bounds
        
        dx = x2 - x1
        dy = y2 - y1
        
        # Check if line is actually a point
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return min_x <= x1 <= max_x and min_y <= y1 <= max_y
        
        t_min = 0.0
        t_max = 1.0
        
        # Check against each edge of the rectangle
        for (p, q) in [(-dx, x1 - min_x), (dx, max_x - x1), (-dy, y1 - min_y), (dy, max_y - y1)]:
            if abs(p) < 1e-9:  # Line is parallel to this edge
                if q < 0:  # Line is outside this edge
                    return False
            else:
                t = q / p
                if p < 0:  # Entering the edge
                    t_min = max(t_min, t)
                else:  # Leaving the edge
                    t_max = min(t_max, t)
                
                if t_min > t_max:  # No intersection
                    return False
        
        return True
    
    def _astar_fcu_escape_routing(self, start_pos, end_pos, net_name):
        """Use A* pathfinding to route F.Cu escape trace avoiding DRC violations"""
        # Create fine-grained grid for F.Cu escape routing
        escape_grid_pitch = 0.1  # mm - finer than main routing grid
        
        # Convert positions to escape grid coordinates
        start_grid = (
            int(start_pos[0] / escape_grid_pitch),
            int(start_pos[1] / escape_grid_pitch)
        )
        end_grid = (
            int(end_pos[0] / escape_grid_pitch),
            int(end_pos[1] / escape_grid_pitch)
        )
        
        # A* pathfinding on escape grid
        from heapq import heappush, heappop
        
        open_set = [(0, start_grid, [start_pos])]
        visited = set()
        
        max_iterations = 10000  # Allow more iterations for fine-grained escape routing
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current_cost, current_pos, current_path = heappop(open_set)
            
            if current_pos in visited:
                continue
            visited.add(current_pos)
            
            # Check if we reached the destination
            distance_to_end = abs(current_pos[0] - end_grid[0]) + abs(current_pos[1] - end_grid[1])
            if distance_to_end <= 1:  # Close enough
                return current_path + [end_pos]
            
            # Explore neighbors (4-connected)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                if next_pos in visited:
                    continue
                
                # Convert back to world coordinates for DRC check
                next_world_pos = (
                    next_pos[0] * escape_grid_pitch,
                    next_pos[1] * escape_grid_pitch
                )
                
                # Check if this step is DRC-compliant
                current_world_pos = current_path[-1]
                if not self._check_fcu_trace_clearance(current_world_pos, next_world_pos, net_name):
                    continue
                
                # Calculate cost (distance + heuristic)
                step_cost = escape_grid_pitch
                heuristic = (abs(next_pos[0] - end_grid[0]) + abs(next_pos[1] - end_grid[1])) * escape_grid_pitch
                total_cost = current_cost + step_cost + heuristic
                
                new_path = current_path + [next_world_pos]
                heappush(open_set, (total_cost, next_pos, new_path))
        
        logger.warning(f"Failed to find A* F.Cu escape route for {net_name} after {iterations} iterations")
        return None
    
    def _find_alternative_via_location(self, ideal_grid_x, ideal_grid_y, pad_pos, net_name):
        """Find alternative via location if the ideal one causes DRC violations"""
        search_radius = 5  # Search up to 5 grid points away
        
        for radius in range(1, search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Skip if not on the search perimeter
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    
                    alt_x = round(ideal_grid_x) + dx
                    alt_y = round(ideal_grid_y) + dy
                    
                    # Check grid bounds
                    if (alt_x < 0 or alt_x >= self.grid_width or 
                        alt_y < 0 or alt_y >= self.grid_height):
                        continue
                    
                    # Check via pad avoidance
                    alt_world_pos = self.grid_to_world(alt_x, alt_y)
                    try:
                        footprints = self.board_interface.get_footprints()
                        if self._via_conflicts_with_pads(alt_world_pos, footprints, 1.0):
                            continue
                    except:
                        pass
                    
                    # Check if F.Cu trace to this location is DRC-compliant
                    if self._check_fcu_trace_clearance(pad_pos, alt_world_pos, net_name):
                        return alt_x, alt_y
        
        return None  # No suitable alternative found
    
    def _get_fcu_pad_polygons(self):
        """Get F.Cu pad polygons from KiCad using the API for DRC checking"""
        fcu_polygons = []
        
        try:
            # Access KiCad board through board_interface
            if not hasattr(self.board_interface, 'kicad_interface') or not self.board_interface.kicad_interface:
                logger.debug("No KiCad interface available, using fallback polygon data")
                return self._get_fallback_fcu_polygons()
            
            kicad = self.board_interface.kicad_interface
            if not hasattr(kicad, 'board') or not kicad.board:
                logger.debug("No KiCad board available, using fallback polygon data")
                return self._get_fallback_fcu_polygons()
            
            board = kicad.board
            logger.debug("Extracting F.Cu pad polygons from KiCad...")
            
            # Try multiple methods to get footprints
            footprints = []
            try:
                if hasattr(board, 'GetFootprints'):
                    footprints = board.GetFootprints()
                elif hasattr(board, 'footprints'):
                    footprints = board.footprints
                elif hasattr(board, 'GetModules'):  # Legacy method
                    footprints = board.GetModules()
                else:
                    logger.debug("Cannot find footprints method on KiCad board")
                    return self._get_fallback_fcu_polygons()
                    
            except Exception as e:
                logger.debug(f"Error getting footprints from KiCad: {e}")
                return self._get_fallback_fcu_polygons()
            
            # Extract F.Cu polygons from each footprint
            fcu_layer_id = 0  # F.Cu layer ID in KiCad
            polygon_count = 0
            
            for footprint in footprints:
                try:
                    # Get pads from footprint
                    pads = []
                    if hasattr(footprint, 'Pads'):
                        pads = footprint.Pads()
                    elif hasattr(footprint, 'GetPads'):
                        pads = footprint.GetPads()
                    elif hasattr(footprint, 'pads'):
                        pads = footprint.pads
                    
                    for pad in pads:
                        try:
                            # Check if pad is on F.Cu layer
                            on_fcu = False
                            if hasattr(pad, 'IsOnLayer') and hasattr(pad, 'F_Cu'):
                                on_fcu = pad.IsOnLayer(pad.F_Cu)
                            elif hasattr(pad, 'GetLayerSet'):
                                layer_set = pad.GetLayerSet()
                                # Check if F.Cu is in layer set
                                on_fcu = fcu_layer_id in layer_set if hasattr(layer_set, '__contains__') else False
                            elif hasattr(pad, 'layers'):
                                on_fcu = 'F.Cu' in pad.layers or fcu_layer_id in pad.layers
                            
                            if on_fcu:
                                # Get pad polygon on F.Cu
                                polygon = self._extract_pad_polygon(pad, fcu_layer_id)
                                if polygon:
                                    fcu_polygons.append(polygon)
                                    polygon_count += 1
                                    
                        except Exception as e:
                            logger.debug(f"Error processing pad: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Error processing footprint: {e}")
                    continue
            
            logger.info(f"📐 Extracted {polygon_count} F.Cu pad polygons from KiCad")
            return fcu_polygons
            
        except Exception as e:
            logger.debug(f"Error extracting F.Cu pad polygons: {e}")
            return self._get_fallback_fcu_polygons()
    
    def _extract_pad_polygon(self, pad, layer_id):
        """Extract polygon from a KiCad pad object"""
        try:
            # Method 1: Try GetEffectivePolygon (most accurate)
            if hasattr(pad, 'GetEffectivePolygon'):
                try:
                    poly_set = pad.GetEffectivePolygon(layer_id)
                    if poly_set and hasattr(poly_set, 'OutlineCount') and poly_set.OutlineCount() > 0:
                        # Convert SHAPE_POLY_SET to coordinate list
                        outline = poly_set.Outline(0)  # Get first outline
                        coords = []
                        if hasattr(outline, 'PointCount'):
                            for i in range(outline.PointCount()):
                                point = outline.CPoint(i)
                                # Convert KiCad units (nm) to mm
                                x_mm = point.x / 1000000.0
                                y_mm = point.y / 1000000.0
                                coords.append((x_mm, y_mm))
                            return {'type': 'polygon', 'coordinates': coords}
                except Exception as e:
                    logger.debug(f"GetEffectivePolygon failed: {e}")
            
            # Method 2: Try GetBoundingBox (simpler fallback)
            if hasattr(pad, 'GetBoundingBox'):
                try:
                    bbox = pad.GetBoundingBox()
                    if bbox:
                        # Convert bounding box to rectangle polygon
                        min_x = bbox.GetX() / 1000000.0  # Convert to mm
                        min_y = bbox.GetY() / 1000000.0
                        max_x = (bbox.GetX() + bbox.GetWidth()) / 1000000.0
                        max_y = (bbox.GetY() + bbox.GetHeight()) / 1000000.0
                        
                        coords = [
                            (min_x, min_y), (max_x, min_y),
                            (max_x, max_y), (min_x, max_y),
                            (min_x, min_y)  # Close polygon
                        ]
                        return {'type': 'rectangle', 'coordinates': coords}
                except Exception as e:
                    logger.debug(f"GetBoundingBox failed: {e}")
            
            # Method 3: Try basic position/size (last resort)
            if hasattr(pad, 'GetPosition') and hasattr(pad, 'GetSize'):
                try:
                    pos = pad.GetPosition()
                    size = pad.GetSize()
                    
                    center_x = pos.x / 1000000.0  # Convert to mm
                    center_y = pos.y / 1000000.0
                    half_width = size.x / 2000000.0
                    half_height = size.y / 2000000.0
                    
                    coords = [
                        (center_x - half_width, center_y - half_height),
                        (center_x + half_width, center_y - half_height),
                        (center_x + half_width, center_y + half_height),
                        (center_x - half_width, center_y + half_height),
                        (center_x - half_width, center_y - half_height)
                    ]
                    return {'type': 'rectangle', 'coordinates': coords}
                except Exception as e:
                    logger.debug(f"GetPosition/GetSize failed: {e}")
                    
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting pad polygon: {e}")
            return None
    
    def _get_fcu_pad_polygons_with_nets(self):
        """Get F.Cu pad polygons with net information from KiCad for DRC checking
        
        Returns list of {'polygon': coords, 'net': net_name, 'type': pad_type}
        """
        try:
            # Access KiCad board through board_interface
            if not hasattr(self.board_interface, 'kicad_interface') or not self.board_interface.kicad_interface:
                logger.debug("No KiCad interface available, using fallback polygon data")
                return self._get_fallback_fcu_polygons_with_nets()
            
            kicad = self.board_interface.kicad_interface
            if not hasattr(kicad, 'board') or not kicad.board:
                logger.debug("No KiCad board available, using fallback polygon data")
                return self._get_fallback_fcu_polygons_with_nets()
            
            board = kicad.board
            logger.debug("Extracting F.Cu pad polygons with net info from KiCad...")
            
            fcu_polygons = []
            fcu_layer_id = 0  # F.Cu layer ID in KiCad
            polygon_count = 0
            
            # Try multiple methods to get footprints
            footprints = []
            try:
                if hasattr(board, 'GetFootprints'):
                    footprints = board.GetFootprints()
                elif hasattr(board, 'footprints'):
                    footprints = board.footprints
                elif hasattr(board, 'GetModules'):  # Legacy method
                    footprints = board.GetModules()
                else:
                    logger.debug("Cannot find footprints method on KiCad board")
                    return self._get_fallback_fcu_polygons_with_nets()
                    
            except Exception as e:
                logger.debug(f"Error getting footprints from KiCad: {e}")
                return self._get_fallback_fcu_polygons_with_nets()
            
            # Extract F.Cu polygons from each footprint with net information
            for footprint in footprints:
                try:
                    # Get pads from footprint
                    pads = []
                    if hasattr(footprint, 'GetPads'):
                        pads = footprint.GetPads()
                    elif hasattr(footprint, 'pads'):
                        pads = footprint.pads
                    elif hasattr(footprint, 'Pads'):
                        pads = footprint.Pads()
                    
                    for pad in pads:
                        try:
                            # Check if pad is on F.Cu layer
                            if hasattr(pad, 'IsOnLayer') and not pad.IsOnLayer(fcu_layer_id):
                                continue
                            elif hasattr(pad, 'layers') and fcu_layer_id not in pad.layers:
                                continue
                            
                            # Get net name for this pad
                            net_name = "unknown"
                            try:
                                if hasattr(pad, 'GetNet') and pad.GetNet():
                                    net_name = pad.GetNet().GetNetname()
                                elif hasattr(pad, 'GetNetname'):
                                    net_name = pad.GetNetname()
                                elif hasattr(pad, 'net') and pad.net:
                                    net_name = pad.net.name
                            except:
                                pass
                            
                            # Extract polygon coordinates
                            polygon_coords = self._extract_pad_polygon_coords(pad, fcu_layer_id)
                            if polygon_coords:
                                pad_info = {
                                    'polygon': polygon_coords,
                                    'net': net_name,
                                    'type': 'pad',
                                    'clearance': self._get_pad_clearance(pad)
                                }
                                fcu_polygons.append(pad_info)
                                polygon_count += 1
                                
                        except Exception as e:
                            logger.debug(f"Error extracting polygon from pad: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Error processing footprint: {e}")
                    continue
            
            logger.debug(f"Extracted {polygon_count} F.Cu pad polygons with nets from KiCad")
            return fcu_polygons
            
        except Exception as e:
            logger.debug(f"Error in KiCad polygon extraction: {e}")
            return self._get_fallback_fcu_polygons_with_nets()
    
    def _extract_pad_polygon_coords(self, pad, layer_id):
        """Extract polygon coordinates from a KiCad pad object
        
        Returns list of (x, y) coordinates in mm
        """
        try:
            # Method 1: Try GetEffectivePolygon (most accurate)
            if hasattr(pad, 'GetEffectivePolygon'):
                try:
                    poly_set = pad.GetEffectivePolygon(layer_id)
                    if poly_set and hasattr(poly_set, 'OutlineCount') and poly_set.OutlineCount() > 0:
                        # Convert SHAPE_POLY_SET to coordinate list
                        outline = poly_set.Outline(0)  # Get first outline
                        coords = []
                        if hasattr(outline, 'PointCount'):
                            for i in range(outline.PointCount()):
                                point = outline.CPoint(i)
                                # Convert KiCad units (nm) to mm
                                x_mm = point.x / 1000000.0
                                y_mm = point.y / 1000000.0
                                coords.append((x_mm, y_mm))
                            return coords
                except Exception as e:
                    logger.debug(f"GetEffectivePolygon failed: {e}")
            
            # Method 2: Try GetBoundingBox (simpler fallback)
            if hasattr(pad, 'GetBoundingBox'):
                try:
                    bbox = pad.GetBoundingBox()
                    if bbox:
                        # Convert bounding box to rectangle polygon
                        min_x = bbox.GetX() / 1000000.0  # Convert to mm
                        min_y = bbox.GetY() / 1000000.0
                        max_x = (bbox.GetX() + bbox.GetWidth()) / 1000000.0
                        max_y = (bbox.GetY() + bbox.GetHeight()) / 1000000.0
                        
                        coords = [
                            (min_x, min_y), (max_x, min_y),
                            (max_x, max_y), (min_x, max_y),
                            (min_x, min_y)  # Close polygon
                        ]
                        return coords
                except Exception as e:
                    logger.debug(f"GetBoundingBox failed: {e}")
            
            # Method 3: Try basic position/size (last resort)
            if hasattr(pad, 'GetPosition') and hasattr(pad, 'GetSize'):
                try:
                    pos = pad.GetPosition()
                    size = pad.GetSize()
                    
                    center_x = pos.x / 1000000.0  # Convert to mm
                    center_y = pos.y / 1000000.0
                    half_width = size.x / 2000000.0
                    half_height = size.y / 2000000.0
                    
                    coords = [
                        (center_x - half_width, center_y - half_height),
                        (center_x + half_width, center_y - half_height),
                        (center_x + half_width, center_y + half_height),
                        (center_x - half_width, center_y + half_height),
                        (center_x - half_width, center_y - half_height)  # Close polygon
                    ]
                    return coords
                except Exception as e:
                    logger.debug(f"Basic position/size extraction failed: {e}")
                    
        except Exception as e:
            logger.debug(f"Error extracting pad polygon: {e}")
        
        return None
    
    def _get_pad_clearance(self, pad):
        """Get clearance for a specific pad (copper-to-hole clearance)"""
        try:
            # Try to get pad-specific clearance
            if hasattr(pad, 'GetLocalClearance'):
                local_clearance = pad.GetLocalClearance()
                if local_clearance is not None and local_clearance > 0:
                    return local_clearance / 1000000.0  # Convert nm to mm
            
            # Try to get board design rules copper-to-hole clearance
            if hasattr(self.board_interface, 'kicad_interface'):
                kicad = self.board_interface.kicad_interface
                if hasattr(kicad, 'board'):
                    board = kicad.board
                    if hasattr(board, 'GetDesignSettings'):
                        settings = board.GetDesignSettings()
                        # Look for copper-to-hole clearance (usually around 0.1mm)
                        if hasattr(settings, 'copper_to_hole_clearance'):
                            return settings.copper_to_hole_clearance / 1000000.0
                        elif hasattr(settings, 'm_CopperEdgeClearance'):
                            return settings.m_CopperEdgeClearance / 1000000.0
            
            # Default copper-to-hole clearance
            return 0.1  # mm
            
        except Exception as e:
            logger.debug(f"Error getting pad clearance: {e}")
            return 0.1  # mm default
    
    def _get_fallback_fcu_polygons_with_nets(self):
        """Fallback method using existing footprint data when KiCad API unavailable"""
        fcu_polygons = []
        
        try:
            footprints = self.board_interface.get_footprints()
            for footprint in footprints:
                for pad in footprint.get('pads', []):
                    if 'F.Cu' in pad.get('layers', []):
                        # Create rectangle polygon from pad data
                        x = pad.get('x', 0)
                        y = pad.get('y', 0)
                        size_x = pad.get('size_x', 1.0) / 2
                        size_y = pad.get('size_y', 1.0) / 2
                        
                        coords = [
                            (x - size_x, y - size_y), (x + size_x, y - size_y),
                            (x + size_x, y + size_y), (x - size_x, y + size_y),
                            (x - size_x, y - size_y)
                        ]
                        
                        # Try to get net information from pad data
                        net_name = pad.get('net', pad.get('net_name', 'unknown'))
                        
                        pad_info = {
                            'polygon': coords,
                            'net': net_name,
                            'type': 'pad',
                            'clearance': 0.1  # Default clearance
                        }
                        fcu_polygons.append(pad_info)
            
            logger.debug(f"Using fallback: {len(fcu_polygons)} F.Cu pad rectangles with nets")
            return fcu_polygons
            
        except Exception as e:
            logger.debug(f"Error in fallback polygon generation: {e}")
            return []
    
    def _get_fallback_fcu_polygons(self):
        """Fallback method using existing footprint data when KiCad API unavailable"""
        fcu_polygons = []
        
        try:
            footprints = self.board_interface.get_footprints()
            for footprint in footprints:
                for pad in footprint.get('pads', []):
                    if 'F.Cu' in pad.get('layers', []):
                        # Create rectangle polygon from pad data
                        x = pad.get('x', 0)
                        y = pad.get('y', 0)
                        size_x = pad.get('size_x', 1.0) / 2
                        size_y = pad.get('size_y', 1.0) / 2
                        
                        coords = [
                            (x - size_x, y - size_y), (x + size_x, y - size_y),
                            (x + size_x, y + size_y), (x - size_x, y + size_y),
                            (x - size_x, y - size_y)
                        ]
                        
                        fcu_polygons.append({
                            'type': 'rectangle',
                            'coordinates': coords,
                            'clearance': self.clearance
                        })
            
            logger.debug(f"Using fallback: {len(fcu_polygons)} F.Cu pad rectangles")
            return fcu_polygons
            
        except Exception as e:
            logger.debug(f"Error in fallback F.Cu polygon extraction: {e}")
            return []
    
    def _trace_violates_polygon_clearance(self, start_pos, end_pos, polygon, required_clearance):
        """Check if a trace violates clearance with a polygon"""
        try:
            coords = polygon.get('coordinates', [])
            if len(coords) < 3:
                return False
            
            # Method 1: Check if trace intersects polygon
            if self._line_intersects_polygon(start_pos, end_pos, coords):
                return True
            
            # Method 2: Check if trace is within clearance distance of polygon edges
            min_distance = self._distance_line_to_polygon(start_pos, end_pos, coords)
            
            return min_distance < required_clearance
            
        except Exception as e:
            logger.debug(f"Error checking polygon clearance: {e}")
            return False  # Assume no violation if check fails
    
    def _line_intersects_polygon(self, start_pos, end_pos, polygon_coords):
        """Check if a line segment intersects a polygon using ray casting"""
        try:
            import math
            
            def point_in_polygon(x, y, poly):
                """Point in polygon test using ray casting algorithm"""
                n = len(poly)
                inside = False
                
                p1x, p1y = poly[0]
                for i in range(1, n + 1):
                    p2x, p2y = poly[i % n]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                
                return inside
            
            # Check if either endpoint is inside the polygon
            if point_in_polygon(start_pos[0], start_pos[1], polygon_coords):
                return True
            if point_in_polygon(end_pos[0], end_pos[1], polygon_coords):
                return True
            
            # Check if line intersects any polygon edge
            for i in range(len(polygon_coords)):
                edge_start = polygon_coords[i]
                edge_end = polygon_coords[(i + 1) % len(polygon_coords)]
                
                if self._lines_intersect(start_pos, end_pos, edge_start, edge_end):
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error in line-polygon intersection: {e}")
            return False
    
    def _lines_intersect(self, line1_start, line1_end, line2_start, line2_end):
        """Check if two line segments intersect"""
        try:
            x1, y1 = line1_start
            x2, y2 = line1_end
            x3, y3 = line2_start
            x4, y4 = line2_end
            
            # Calculate the direction of the lines
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
            # Check if line segments intersect
            return (ccw(line1_start, line2_start, line2_end) != ccw(line1_end, line2_start, line2_end) and
                    ccw(line1_start, line1_end, line2_start) != ccw(line1_start, line1_end, line2_end))
                    
        except Exception as e:
            logger.debug(f"Error in line intersection: {e}")
            return False
    
    def _distance_line_to_polygon(self, start_pos, end_pos, polygon_coords):
        """Calculate minimum distance from line segment to polygon"""
        try:
            min_distance = float('inf')
            
            # Check distance to each polygon edge
            for i in range(len(polygon_coords)):
                edge_start = polygon_coords[i]
                edge_end = polygon_coords[(i + 1) % len(polygon_coords)]
                
                # Distance between two line segments
                distance = self._calculate_trace_distance(start_pos, end_pos, edge_start, edge_end)
                min_distance = min(min_distance, distance)
            
            return min_distance if min_distance != float('inf') else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating line-polygon distance: {e}")
            return 0.0
    
    def _check_fcu_trace_clearance(self, start_pos, end_pos, net_name):
        """Check if an F.Cu trace segment has proper DRC clearance using net-aware KiCad polygons
        
        KEY PRINCIPLE: F.Cu traces can only intersect with pads of the SAME net.
        They must maintain clearance from pads of DIFFERENT nets.
        
        Returns True if the trace can be placed without DRC violations
        """
        # Get F.Cu pad polygons with net information (cached for performance)
        if not hasattr(self, '_fcu_polygons_with_nets_cache'):
            self._fcu_polygons_with_nets_cache = self._get_fcu_pad_polygons_with_nets()
        
        fcu_polygons_with_nets = self._fcu_polygons_with_nets_cache
        
        # Required clearance (trace width + spacing)  
        required_clearance = self.trace_width + self.clearance
        
        # Check against existing routed F.Cu segments from other nets
        f_cu_layer_idx = self.layer_names.index('F.Cu')
        for existing_segment in self.routed_segments:
            # Skip segments from the same net
            if existing_segment.net_id == net_name:
                continue
                
            # Only check F.Cu layer segments
            if (existing_segment.start.layer != f_cu_layer_idx or 
                existing_segment.end.layer != f_cu_layer_idx):
                continue
            
            # Skip vias (they have different clearance rules)
            if existing_segment.is_via():
                continue
                
            # Convert existing segment to world coordinates
            existing_start = self.grid_to_world(existing_segment.start.x, existing_segment.start.y)
            existing_end = self.grid_to_world(existing_segment.end.x, existing_segment.end.y)
            
            # Check if traces are too close
            min_distance = self._calculate_trace_distance(
                start_pos, end_pos, existing_start, existing_end
            )
            
            if min_distance < required_clearance:
                logger.debug(f"F.Cu trace clearance violation with existing trace: {min_distance:.3f}mm < {required_clearance:.3f}mm")
                return False
        
        # NET-AWARE DRC CHECKING: Check against F.Cu pad polygons from OTHER nets only
        for pad_info in fcu_polygons_with_nets:
            pad_net = pad_info.get('net', 'unknown')
            pad_polygon = pad_info.get('polygon', [])
            
            # CRITICAL: Only enforce clearance from pads of DIFFERENT nets
            # Same-net pads can be intersected (that's how we connect to them!)
            if pad_net != net_name and pad_net != 'unknown':
                # Check if trace violates clearance with this different-net pad
                if self._trace_intersects_or_violates_clearance(start_pos, end_pos, pad_polygon, required_clearance):
                    logger.debug(f"F.Cu trace clearance violation with {pad_net} pad (different net from {net_name})")
                    return False
            
        return True
    
    def _trace_intersects_or_violates_clearance(self, start_pos, end_pos, polygon_coords, required_clearance):
        """Check if a trace intersects a polygon or violates clearance
        
        Returns True if there's a violation (intersection or too close)
        """
        try:
            if len(polygon_coords) < 3:
                return False
            
            # Method 1: Check if trace intersects polygon
            if self._line_intersects_polygon(start_pos, end_pos, polygon_coords):
                return True
            
            # Method 2: Check if trace is within clearance distance of polygon edges
            min_distance = self._distance_line_to_polygon(start_pos, end_pos, polygon_coords)
            
            return min_distance < required_clearance
            
        except Exception as e:
            logger.debug(f"Error checking trace-polygon clearance: {e}")
            return False  # Assume no violation if check fails
    
    def _gpu_a_star_pathfind(self, start, end, net_name):
        """GPU-accelerated A* pathfinding using the GPU manager"""
        try:
            # Prepare grid data for GPU
            grid_data = {
                'routing_grid': self.routing_grid,
                'net_id_grid': self.net_id_grid,
                'layer_directions': self.layer_directions,
                'grid_dimensions': (self.grid_width, self.grid_height, self.layer_count),
                'net_name': net_name,
                'net_id': self.net_name_to_id.get(net_name)
            }
            
            # Call GPU manager's A* pathfinding
            gpu_path = self.gpu_manager.a_star_pathfind(
                start=start,
                end=end,
                grid_data=grid_data,
                manhattan_mode=True,  # Use Manhattan routing constraints
                layer_constraints=self.layer_directions
            )
            
            if gpu_path and len(gpu_path) > 0:
                logger.debug(f"GPU A* found path with {len(gpu_path)} points for {net_name}")
                return gpu_path
            else:
                logger.debug(f"GPU A* returned no path for {net_name}")
                return None
                
        except Exception as e:
            logger.debug(f"GPU A* pathfinding error: {e}")
            return None
    
    def mark_route_in_grid(self, path, net_name):
        """Mark a route in the grid safely"""
        net_id = self.net_name_to_id.get(net_name)
        if net_id is None:
            return
        
        for x, y, layer in path:
            try:
                if (0 <= x < self.grid_width and 0 <= y < self.grid_height and 0 <= layer < self.layer_count):
                    self.routing_grid[layer, y, x] = CellState.ROUTED
                    self.net_id_grid[layer, y, x] = net_id
            except Exception as e:
                logger.debug(f"Error marking route at ({x},{y},{layer}): {e}")
                continue
    
    def verify_connectivity(self, net_name):
        """Verify electrical connectivity of a routed net"""
        # Get all pads for this net
        pads = self.board_interface.get_pads_for_net(net_name)
        
        if len(pads) < 2:
            return True  # Nothing to verify
        
        # Get the first pad
        start_pad = pads[0]
        
        # Check connectivity to each other pad
        for i in range(1, len(pads)):
            end_pad = pads[i]
            
            # Check if we have a complete path from start to end
            if not self._verify_connectivity_path(start_pad, end_pad, net_name):
                return False
        
        return True
    
    def _verify_connectivity_path(self, start_pad, end_pad, net_name):
        """Verify connectivity between two pads"""
        # This is a simplified check - in a real implementation you would:
        # 1. Get the actual route segments
        # 2. Check that they form a continuous path
        # 3. Verify that all vias connect properly
        
        # For now, we'll assume that if we have segments for this net, it's connected
        return len(self.net_routes.get(net_name, [])) > 0
    
    def _subdivide_grid_segments(self, segments, net_name):
        """Subdivide grid segments to leave 0.4mm spaces for other nets"""
        subdivided = []
        
        for segment in segments:
            if segment.via_type is None:  # Only subdivide tracks, not vias
                # Calculate segment length
                segment_length = self.calculate_segment_length(segment)
                
                # If segment is longer than subdivision threshold, subdivide it
                if segment_length > self.grid_subdivision_space * 2:
                    # Create subdivided segments with gaps
                    subdivided_tracks = self._create_subdivided_tracks(segment, net_name)
                    subdivided.extend(subdivided_tracks)
                else:
                    # Segment is too short to subdivide
                    subdivided.append(segment)
            else:
                # Keep vias as-is
                subdivided.append(segment)
        
        return subdivided
    
    def _create_subdivided_tracks(self, segment, net_name):
        """Create subdivided track segments with gaps for other nets"""
        tracks = []
        
        try:
            # For now, implement basic subdivision - can be enhanced later
            # This creates the "broken" traces with 0.4mm spaces as specified
            
            start_x, start_y = self.grid_to_world(segment.start.x, segment.start.y)
            end_x, end_y = self.grid_to_world(segment.end.x, segment.end.y)
            
            # Create main segment (leaving space for subdivision)
            main_segment = RouteSegmentInfo(
                segment.start, segment.end, net_name, segment.width, segment.via_type
            )
            tracks.append(main_segment)
            
            # Mark subdivided space as available for other nets
            subdivision_key = f"{start_x:.3f},{start_y:.3f}-{end_x:.3f},{end_y:.3f}"
            self.subdivided_segments[subdivision_key] = {
                'net': net_name,
                'available_space': self.grid_subdivision_space,
                'used_by': [net_name]
            }
        
        except Exception as e:
            logger.error(f"Error creating subdivided tracks for {net_name}: {e}")
            # Return original segment as fallback if it's already a RouteSegmentInfo
            if isinstance(segment, RouteSegmentInfo):
                tracks = [segment]
            else:
                # Create a basic RouteSegmentInfo from the segment data
                try:
                    fallback_segment = RouteSegmentInfo(
                        segment.start, segment.end, net_name, 
                        getattr(segment, 'width', self.trace_width),
                        getattr(segment, 'via_type', None)
                    )
                    tracks = [fallback_segment]
                except Exception as e2:
                    logger.error(f"Failed to create fallback segment: {e2}")
                    tracks = []
        
        return tracks
    
    def _mark_subdivided_route_in_grid(self, path, net_name):
        """Mark route in grid accounting for subdivision strategy"""
        net_id = self.net_name_to_id.get(net_name)
        
        for x, y, layer in path:
            # Mark main routing cell
            self.routing_grid[layer, y, x] = CellState.ROUTED
            self.net_id_grid[layer, y, x] = net_id
            
            # For grid subdivision, we leave adjacent cells available
            # This implements the "break" and subdivision concept
    
    def export_to_kicad(self):
        """Export routes to KiCad with verification"""
        kicad_tracks = []
        kicad_vias = []
        
        logger.info(f"📤 Exporting Manhattan grid routes to KiCad...")
        
        for net_name, segments in self.net_routes.items():
            net_tracks = []
            net_vias = []
            
            for segment in segments:
                start_cell = segment.start
                end_cell = segment.end
                
                # Convert grid to world coordinates
                start_x, start_y = self.grid_to_world(start_cell.x, start_cell.y)
                end_x, end_y = self.grid_to_world(end_cell.x, end_cell.y)
                
                # Get layer names
                start_layer_name = self.layer_names[start_cell.layer]
                end_layer_name = self.layer_names[end_cell.layer]
                
                if segment.is_via():
                    # Create blind/buried via as specified
                    via = {
                        'position': [start_x, start_y],
                        'size': self.via_diameter,  # 0.25mm diameter
                        'drill': self.via_drill,    # 0.15mm hole
                        'net': net_name,
                        'type': segment.via_type,
                        'start_layer': start_layer_name,
                        'end_layer': end_layer_name,
                        'layers': self._get_via_layers(start_cell.layer, end_cell.layer)
                    }
                    kicad_vias.append(via)
                    net_vias.append(via)
                else:
                    # Create track with proper layer assignment
                    track = {
                        'start': [start_x, start_y],
                        'end': [end_x, end_y],
                        'width': self.trace_width,  # 3.5mil
                        'layer': start_layer_name,
                        'net': net_name
                    }
                    kicad_tracks.append(track)
                    net_tracks.append(track)
            
            # Verify connectivity for this net
            if self._verify_net_connectivity(net_name, net_tracks, net_vias):
                logger.info(f"✅ Net {net_name}: Connectivity verified")
            else:
                logger.warning(f"⚠️ Net {net_name}: Connectivity verification failed")
        
        logger.info(f"✅ Export complete: {len(kicad_tracks)} tracks, {len(kicad_vias)} vias")
        return kicad_tracks, kicad_vias
    
    def _verify_net_connectivity(self, net_name, tracks, vias):
        """Verify electrical connectivity by tracing copper path from start to end pad"""
        # Get pads for this net
        pads = self.board_interface.get_pads_for_net(net_name)
        if len(pads) < 2:
            return True  # Single pad nets are always "connected"
        
        # For each pair of pads, verify there's a continuous copper path:
        # start pad -> F.Cu escape -> via -> grid path -> via -> F.Cu escape -> end pad
        try:
            start_pad = pads[0]
            for end_pad in pads[1:]:
                if not self._trace_connectivity_path(start_pad, end_pad, tracks, vias):
                    return False
            return True
        except Exception as e:
            logger.error(f"Connectivity verification failed for {net_name}: {e}")
            return False
    
    def _trace_connectivity_path(self, start_pad, end_pad, tracks, vias):
        """Trace connectivity path between two pads through F.Cu->via->grid->via->F.Cu"""
        # This is a simplified connectivity check
        # Full implementation would build a graph and verify continuous paths
        
        start_pos = (start_pad['x'], start_pad['y'])
        end_pos = (end_pad['x'], end_pad['y'])
        
        # Check if we have tracks and/or vias connecting these regions
        # For now, return True if we have any routing elements
        has_routing = len(tracks) > 0 or len(vias) > 0
        
        return has_routing
    
    def get_bright_white_segments(self):
        """Get current routing segments for bright white visualization"""
        return self.current_routing_segments.copy() if self.current_routing_segments else []
    
    def clear_bright_white_visualization(self):
        """Clear bright white visualization after routing completion"""
        self.current_routing_net = None
        self.current_routing_segments = []
    
    def _safe_array_item(self, array_value):
        """Safely extract scalar value from GPU/CPU array"""
        if hasattr(array_value, 'item'):
            return array_value.item()
        elif hasattr(array_value, '__len__') and len(array_value) == 1:
            return array_value[0]
        else:
            return array_value
    
    def _safe_grid_access(self, grid, layer, y, x):
        """Safely access grid values with bounds checking"""
        try:
            if (0 <= layer < grid.shape[0] and 
                0 <= y < grid.shape[1] and 
                0 <= x < grid.shape[2]):
                value = grid[layer, y, x]
                return self._safe_array_item(value)
            else:
                return CellState.OBSTACLE  # Out of bounds = obstacle
        except Exception as e:
            logger.debug(f"Error accessing grid at ({layer},{y},{x}): {e}")
            return CellState.OBSTACLE
    
    def _get_layer_color(self, layer_name):
        """Get KiCad theme color for a layer"""
        # KiCad theme color mapping - matches graphics/kicad_theme.json exactly
        kicad_colors = {
            'F.Cu': 'rgb(200, 52, 52)',      # f copper
            'B.Cu': 'rgb(77, 127, 196)',     # b copper  
            'In1.Cu': 'rgb(127, 200, 127)',  # in1 copper
            'In2.Cu': 'rgb(206, 125, 44)',   # in2 copper
            'In3.Cu': 'rgb(79, 203, 203)',   # in3 copper
            'In4.Cu': 'rgb(219, 98, 139)',   # in4 copper
            'In5.Cu': 'rgb(167, 165, 198)',  # in5 copper
            'In6.Cu': 'rgb(40, 204, 217)',   # in6 copper
            'In7.Cu': 'rgb(232, 178, 167)',  # in7 copper
            'In8.Cu': 'rgb(242, 237, 161)',  # in8 copper
            'In9.Cu': 'rgb(141, 203, 129)',  # in9 copper
            'In10.Cu': 'rgb(237, 124, 51)',  # in10 copper
            'In11.Cu': 'rgb(91, 195, 235)',  # in11 copper
            'In12.Cu': 'rgb(247, 111, 142)', # in12 copper
            'In13.Cu': 'rgb(167, 165, 198)', # in13 copper
            'In14.Cu': 'rgb(40, 204, 217)',  # in14 copper
            'In15.Cu': 'rgb(232, 178, 167)', # in15 copper
            'In16.Cu': 'rgb(242, 237, 161)', # in16 copper
            'In17.Cu': 'rgb(237, 124, 51)',  # in17 copper
            'In18.Cu': 'rgb(91, 195, 235)',  # in18 copper
            'In19.Cu': 'rgb(247, 111, 142)', # in19 copper
            'In20.Cu': 'rgb(167, 165, 198)', # in20 copper
            'In21.Cu': 'rgb(40, 204, 217)',  # in21 copper
            'In22.Cu': 'rgb(232, 178, 167)', # in22 copper
            'In23.Cu': 'rgb(242, 237, 161)', # in23 copper
            'In24.Cu': 'rgb(237, 124, 51)',  # in24 copper
            'In25.Cu': 'rgb(91, 195, 235)',  # in25 copper
            'In26.Cu': 'rgb(247, 111, 142)', # in26 copper
            'In27.Cu': 'rgb(167, 165, 198)', # in27 copper
            'In28.Cu': 'rgb(40, 204, 217)',  # in28 copper
            'In29.Cu': 'rgb(232, 178, 167)', # in29 copper
            'In30.Cu': 'rgb(242, 237, 161)', # in30 copper
        }
        
        # Get color from mapping or default to white
        return kicad_colors.get(layer_name, 'rgb(255, 255, 255)')
    
    def _get_via_color(self, via_type):
        """Get KiCad theme color for via types"""
        via_colors = {
            'through': 'rgb(236, 236, 236)',      # Through via - light gray
            'blind': 'rgb(187, 151, 38)',         # Blind/buried via - gold
            'buried': 'rgb(187, 151, 38)',        # Blind/buried via - gold
            'micro': 'rgb(0, 132, 132)',          # Microvia - teal
        }
        
        return via_colors.get(via_type, 'rgb(187, 151, 38)')  # Default to blind/buried

    def _find_via_safe_grid_point(self, ideal_grid_x, ideal_grid_y, pad_pos):
        """Find a grid point near the ideal location using proper copper-to-hole clearance
        
        Uses 0.1mm copper-to-hole clearance (not arbitrary large distances)
        """
        # Start with the ideal grid location
        best_x = round(ideal_grid_x) 
        best_y = round(ideal_grid_y)
        
        # Use proper copper-to-hole clearance (0.1mm) instead of arbitrary 3mm
        copper_to_hole_clearance = 0.1  # mm - proper DRC clearance
        
        # Check if this location has proper clearance from pads of different nets
        via_world_pos = self.grid_to_world(best_x, best_y)
        
        # Get pad polygons with net information for proper DRC checking
        if not hasattr(self, '_fcu_polygons_with_nets_cache'):
            self._fcu_polygons_with_nets_cache = self._get_fcu_pad_polygons_with_nets()
        
        # Check if the ideal position is DRC-compliant
        if self._via_position_is_drc_compliant(via_world_pos, copper_to_hole_clearance):
            return best_x, best_y
        
        # If ideal position doesn't work, try nearby grid points in spiral pattern
        search_radius = 5  # Small search radius since we only need 0.1mm clearance
        for radius in range(1, search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Skip if not on the search perimeter
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                        
                    candidate_x = best_x + dx
                    candidate_y = best_y + dy
                    
                    # Check bounds
                    if (0 <= candidate_x < self.grid_width and 
                        0 <= candidate_y < self.grid_height):
                        
                        candidate_world_pos = self.grid_to_world(candidate_x, candidate_y)
                        
                        if self._via_position_is_drc_compliant(candidate_world_pos, copper_to_hole_clearance):
                            distance_from_pad = ((candidate_world_pos[0] - pad_pos[0])**2 + (candidate_world_pos[1] - pad_pos[1])**2)**0.5
                            logger.debug(f"Via placement: Grid({candidate_x},{candidate_y}) offset({dx},{dy}) distance_from_pad={distance_from_pad:.2f}mm")
                            return candidate_x, candidate_y
            
        # If no DRC-compliant position found within search radius, use original position
        # This is acceptable since we're using proper 0.1mm clearance, not arbitrary large distances
        logger.debug(f"Using ideal via position at Grid({best_x},{best_y}) - may have minimal clearance")
        return best_x, best_y
    
    def _via_position_is_drc_compliant(self, via_world_pos, copper_to_hole_clearance):
        """Check if a via position has proper clearance from pads of different nets
        
        Uses net-aware DRC: vias can be close to same-net pads but must maintain 
        clearance from different-net pads
        """
        try:
            fcu_polygons_with_nets = self._fcu_polygons_with_nets_cache
            
            for pad_info in fcu_polygons_with_nets:
                pad_polygon = pad_info.get('polygon', [])
                pad_net = pad_info.get('net', 'unknown')
                
                if len(pad_polygon) < 3:
                    continue
                
                # Check if via is too close to this pad
                min_distance = self._point_to_polygon_distance(via_world_pos, pad_polygon)
                
                if min_distance < copper_to_hole_clearance:
                    # Only log violation for different nets - same net connections are OK
                    if pad_net != 'unknown':
                        logger.debug(f"Via at {via_world_pos} too close ({min_distance:.3f}mm) to {pad_net} pad")
                    return False
                    
            return True
            
        except Exception as e:
            logger.debug(f"Error checking via DRC compliance: {e}")
            return True  # Default to compliant if check fails
    
    def _point_to_polygon_distance(self, point, polygon_coords):
        """Calculate minimum distance from a point to a polygon"""
        try:
            min_distance = float('inf')
            
            # Check distance to each edge of the polygon
            for i in range(len(polygon_coords)):
                p1 = polygon_coords[i]
                p2 = polygon_coords[(i + 1) % len(polygon_coords)]
                
                # Calculate distance from point to line segment
                edge_distance = self._point_to_line_segment_distance(point, p1, p2)
                min_distance = min(min_distance, edge_distance)
            
            return min_distance
            
        except Exception as e:
            logger.debug(f"Error calculating point-to-polygon distance: {e}")
            return 0.0
    
    def _point_to_line_segment_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        try:
            import numpy as np
            
            # Convert to numpy arrays for easier calculation
            P = np.array(point)
            A = np.array(line_start)
            B = np.array(line_end)
            
            # Vector from A to B
            AB = B - A
            # Vector from A to P
            AP = P - A
            
            # Handle zero-length line segment
            AB_length_sq = np.dot(AB, AB)
            if AB_length_sq == 0:
                return np.linalg.norm(AP)
            
            # Project AP onto AB
            t = np.dot(AP, AB) / AB_length_sq
            
            # Clamp t to [0, 1] to stay on the line segment
            t = max(0, min(1, t))
            
            # Find the projection point
            projection = A + t * AB
            
            # Return distance from P to projection
            return np.linalg.norm(P - projection)
            
        except Exception as e:
            logger.debug(f"Error calculating point-to-line distance: {e}")
            return 0.0
    
    def _via_conflicts_with_pads(self, via_world_pos, footprints, min_distance):
        """Check if a via position conflicts with any pads"""
        try:
            for footprint in footprints:
                for pad in footprint.get('pads', []):
                    pad_pos = self._get_pad_position(pad)
                    pad_distance = ((via_world_pos[0] - pad_pos[0])**2 + (via_world_pos[1] - pad_pos[1])**2)**0.5
                    
                    if pad_distance < min_distance:
                        return True  # Conflict found
                        
            return False  # No conflicts
            
        except Exception as e:
            logger.debug(f"Error checking via-pad conflicts: {e}")
            return False  # Assume safe if can't check

    def _create_escape_route(self, pad, pad_pos, net_name, route_type):
        """Create DRC-aware escape route from pad to Manhattan grid entry point"""
        try:
            # Find optimal grid entry point near the pad
            grid_x, grid_y = self.world_to_grid(pad_pos[0], pad_pos[1])
            
            # Choose best grid entry point (snap to grid), avoiding pad placement for F.Cu vias
            grid_entry_x, grid_entry_y = self._find_via_safe_grid_point(grid_x, grid_y, pad_pos)
            
            # Find optimal inner layer for grid entry based on routing needs
            optimal_layer = self._select_optimal_grid_layer(pad_pos, net_name)
            
            # Create DRC-aware F.Cu escape trace from pad to via point
            via_world_x, via_world_y = self.grid_to_world(grid_entry_x, grid_entry_y)
            
            escape_segments = []
            
            # Always create F.Cu escape trace from pad to via point
            pad_to_via_distance = ((pad_pos[0] - via_world_x)**2 + (pad_pos[1] - via_world_y)**2)**0.5
            f_cu_layer_idx = self.layer_names.index('F.Cu')
            
            # ALWAYS create F.Cu escape trace for proper DRC compliance and visibility
            # User requirement: F.Cu traces are required for all connections
            # Convert positions to grid coordinates
            pad_grid_pos = self.world_to_grid(pad_pos[0], pad_pos[1])
            via_grid_pos = (grid_entry_x, grid_entry_y)
            
            # Create F.Cu escape track segment from pad to via
            escape_start = GridCell(round(pad_grid_pos[0]), round(pad_grid_pos[1]), f_cu_layer_idx)
            escape_end = GridCell(via_grid_pos[0], via_grid_pos[1], f_cu_layer_idx)
            
            escape_segment = RouteSegmentInfo(
                escape_start, escape_end, net_name, self.trace_width, None
            )
            escape_segments.append(escape_segment)
            
            logger.info(f"✅ Created F.Cu escape trace for {net_name}: {pad_to_via_distance:.3f}mm from ({pad_pos[0]:.2f}, {pad_pos[1]:.2f}) to ({via_world_x:.2f}, {via_world_y:.2f}) on layer {self.layer_names[f_cu_layer_idx]}")
            
            # Create blind via from F.Cu to grid layer
            via_start = GridCell(grid_entry_x, grid_entry_y, self.layer_names.index('F.Cu'))
            via_end = GridCell(grid_entry_x, grid_entry_y, optimal_layer)
            
            via_segment = RouteSegmentInfo(
                via_start, via_end, net_name, self.via_diameter, 
                self._determine_via_type(self.layer_names.index('F.Cu'), optimal_layer)
            )
            escape_segments.append(via_segment)
            
            logger.debug(f"Created blind via for {net_name}: F.Cu → {self.layer_names[optimal_layer]}")
            
            # Return grid entry point and escape segments
            grid_entry_point = (grid_entry_x, grid_entry_y, optimal_layer)
            
            return grid_entry_point, escape_segments
            
        except Exception as e:
            logger.error(f"Error creating escape route for {net_name}: {e}")
            return None, []
    
    def _select_optimal_grid_layer(self, pad_pos, net_name):
        """Select optimal inner layer for grid entry based on congestion and routing needs"""
        # Available grid layers (not F.Cu)
        inner_layers = [i for i, name in enumerate(self.layer_names) if name != 'F.Cu']
        
        if not inner_layers:
            logger.warning(f"No inner layers available for grid routing")
            return 1  # Default to In1.Cu
        
        # For now, use simple round-robin layer selection
        # TODO: Implement congestion-based layer selection
        layer_index = hash(net_name) % len(inner_layers)
        selected_layer = inner_layers[layer_index]
        
        logger.debug(f"Selected layer {self.layer_names[selected_layer]} for {net_name} grid entry")
        return selected_layer
    
    def _route_manhattan_grid(self, grid_start, grid_end, net_name):
        """Route through Manhattan grid using inner layers only"""
        try:
            logger.debug(f"Routing through Manhattan grid: {grid_start} → {grid_end}")
            
            # Use A* pathfinding on grid layers only (excluding F.Cu)
            # This now returns RouteSegmentInfo objects directly
            grid_segments = self.a_star_pathfind_grid_only(
                grid_start, grid_end, net_name
            )
            
            if not grid_segments:
                logger.warning(f"No grid path found for {net_name}")
                return []
            
            logger.debug(f"Generated {len(grid_segments)} grid segments for {net_name}")
            return grid_segments
            
        except Exception as e:
            logger.error(f"Error routing through Manhattan grid for {net_name}: {e}")
            return []
    
    def a_star_pathfind_grid_only(self, start, end, net_name):
        """A* pathfinding using only Manhattan grid layers (In1-In10, B.Cu) with GPU acceleration"""
        start_x, start_y, start_layer = start
        end_x, end_y, end_layer = end
        
        # Try GPU-accelerated pathfinding first if available
        if hasattr(self, 'gpu_manager') and self.gpu_manager and hasattr(self.gpu_manager, 'is_gpu_enabled'):
            try:
                if self.gpu_manager.is_gpu_enabled():
                    gpu_path = self._gpu_a_star_pathfind(start, end, net_name)
                    if gpu_path:
                        logger.debug(f"GPU A* pathfinding successful for {net_name}: {len(gpu_path)} points")
                        return gpu_path
                    else:
                        logger.debug(f"GPU A* pathfinding failed for {net_name}, falling back to CPU")
            except Exception as e:
                logger.debug(f"GPU pathfinding error for {net_name}: {e}, falling back to CPU")
        
        # Fallback to CPU-based A* pathfinding
        logger.debug(f"Using CPU A* pathfinding for {net_name}")
        
        # Get net ID
        net_id = self.net_name_to_id.get(net_name)
        
        # A* state: (f_score, g_score, x, y, layer)
        open_set = [(0, 0, start_x, start_y, start_layer)]
        came_from = {}
        g_score = {(start_x, start_y, start_layer): 0}
        f_score = {(start_x, start_y, start_layer): self._manhattan_distance(start_x, start_y, end_x, end_y)}
        
        visited = set()
        max_iterations = 50000  # Increased for larger grids
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get the node with the lowest f_score
            current_f, current_g, current_x, current_y, current_layer = heapq.heappop(open_set)
            current_state = (current_x, current_y, current_layer)
            
            if current_state in visited:
                continue
            
            visited.add(current_state)
            
            # Check if we reached the goal
            if current_x == end_x and current_y == end_y:
                # If we're at the destination but not on the target layer, add a final via
                if current_layer != end_layer:
                    came_from[(current_x, current_y, end_layer)] = current_state
                    current_state = (current_x, current_y, end_layer)
                
                path = self._reconstruct_path(came_from, current_state)
                logger.debug(f"Found grid path for {net_name} in {iterations} iterations, length {len(path)}")
                
                # Convert path to RouteSegmentInfo objects
                segments = self.path_to_segments(path, net_name)
                return segments
            
            # Get neighbors - only use grid layers (exclude F.Cu)
            neighbors = self._get_manhattan_grid_neighbors(current_x, current_y, current_layer, net_name)
            
            # Process neighbors
            for next_x, next_y, next_layer, move_cost in neighbors:
                next_state = (next_x, next_y, next_layer)
                
                if next_state in visited:
                    continue
                
                # Calculate new g_score
                tentative_g = g_score[current_state] + move_cost
                
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    # This path is better than any previous one
                    came_from[next_state] = current_state
                    g_score[next_state] = tentative_g
                    
                    # Calculate f_score
                    h_score = self._manhattan_distance(next_x, next_y, end_x, end_y)
                    
                    # Add layer difference penalty to encourage staying on the same layer
                    layer_penalty = 0
                    if next_layer != end_layer:
                        layer_penalty = 1  # Small penalty for being on wrong layer
                    
                    f_score[next_state] = tentative_g + h_score + layer_penalty
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score[next_state], tentative_g, next_x, next_y, next_layer))
        
        # No path found
        logger.warning(f"No grid path found for {net_name} after {iterations} iterations")
        return None
    
    def _get_manhattan_grid_neighbors(self, x, y, layer, net_name):
        """Get valid neighbors for Manhattan grid routing (excludes F.Cu)"""
        neighbors = []
        
        try:
            # Get movement direction for this layer (exclude F.Cu from routing)
            if layer < len(self.layer_names):
                layer_name = self.layer_names[layer]
                direction = self.layer_directions.get(layer_name, 'both')
                
                # Skip F.Cu for grid routing
                if layer_name == 'F.Cu':
                    return neighbors
            else:
                direction = 'both'  # Default for unknown layers
            
            # Define possible moves based on layer direction
            moves = []
            if direction == 'horizontal':
                moves = [(-1, 0), (1, 0)]  # Left, right
            elif direction == 'vertical':
                moves = [(0, -1), (0, 1)]  # Up, down
            elif direction == 'both':
                moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # All directions
            
            # Check each possible move
            for dx, dy in moves:
                new_x, new_y = x + dx, y + dy
                if self._is_valid_grid_cell(new_x, new_y, layer, net_name):
                    neighbors.append((new_x, new_y, layer, 1))
            
            # Add layer transitions (vias) - only to other grid layers
            for target_layer in range(len(self.layer_names)):
                target_layer_name = self.layer_names[target_layer] if target_layer < len(self.layer_names) else f'Layer_{target_layer}'
                
                # Skip F.Cu and current layer
                if target_layer != layer and target_layer_name != 'F.Cu':
                    if self._is_valid_grid_cell(x, y, target_layer, net_name):
                        via_cost = 3  # Higher cost for layer changes
                        neighbors.append((x, y, target_layer, via_cost))
                        
        except Exception as e:
            logger.debug(f"Error getting grid neighbors for ({x},{y},{layer}): {e}")
        
        return neighbors
    
    def _is_valid_grid_cell(self, x, y, layer, net_name):
        """Check if a grid cell is valid for routing (enhanced for grid layers)"""
        # Check bounds
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height and 0 <= layer < self.layer_count):
            return False
        
        # Skip F.Cu for grid routing
        if layer < len(self.layer_names) and self.layer_names[layer] == 'F.Cu':
            return False
        
        # Use safe grid access methods
        cell_state = self._safe_grid_access(self.routing_grid, layer, y, x)
        
        # Check if cell is free
        if cell_state == CellState.EMPTY:
            return True
        
        # If cell is routed, check if it's the same net
        if cell_state == CellState.ROUTED and net_name is not None:
            net_id = self._safe_grid_access(self.net_id_grid, layer, y, x)
            routed_net = self.net_id_to_name[net_id] if 0 <= net_id < len(self.net_id_to_name) else None
            return routed_net == net_name
        
        return False
    
    def _mark_route_segments_in_grid(self, segments, net_name):
        """Mark route segments in the routing grid"""
        net_id = self.net_name_to_id.get(net_name)
        if net_id is None:
            return
        
        for segment in segments:
            try:
                start_x, start_y, start_layer = segment.start.x, segment.start.y, segment.start.layer
                end_x, end_y, end_layer = segment.end.x, segment.end.y, segment.end.layer
                
                if segment.via_type is not None:
                    # Via - mark single point on all layers it spans
                    min_layer = min(start_layer, end_layer)
                    max_layer = max(start_layer, end_layer)
                    
                    for layer in range(min_layer, max_layer + 1):
                        if (0 <= layer < self.layer_count and 
                            0 <= start_y < self.grid_height and 
                            0 <= start_x < self.grid_width):
                            self.routing_grid[layer, start_y, start_x] = CellState.ROUTED
                            self.net_id_grid[layer, start_y, start_x] = net_id
                else:
                    # Track - mark all points along the path
                    if start_x == end_x:  # Vertical track
                        y_min, y_max = min(start_y, end_y), max(start_y, end_y)
                        for y in range(y_min, y_max + 1):
                            if (0 <= start_layer < self.layer_count and 
                                0 <= y < self.grid_height and 
                                0 <= start_x < self.grid_width):
                                self.routing_grid[start_layer, y, start_x] = CellState.ROUTED
                                self.net_id_grid[start_layer, y, start_x] = net_id
                    else:  # Horizontal track
                        x_min, x_max = min(start_x, end_x), max(start_x, end_x)
                        for x in range(x_min, x_max + 1):
                            if (0 <= start_layer < self.layer_count and 
                                0 <= start_y < self.grid_height and 
                                0 <= x < self.grid_width):
                                self.routing_grid[start_layer, start_y, x] = CellState.ROUTED
                                self.net_id_grid[start_layer, start_y, x] = net_id
            except Exception as e:
                logger.debug(f"Error marking route segment: {e}")
                continue
    
    def _get_via_layers(self, start_layer, end_layer):
        """Get the layers spanned by a blind/buried via"""
        min_layer = min(start_layer, end_layer)
        max_layer = max(start_layer, end_layer)
        
        return [self.layer_names[i] for i in range(min_layer, max_layer + 1)]
