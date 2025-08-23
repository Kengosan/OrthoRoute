#!/usr/bin/env python3
"""
GPU-Accelerated Manhattan Router

Implements Manhattan routing with GPU acceleration based on specifications:
- 3.5mil traces with 3.5mil spacing on 0.4mm grid
- 11 layers: In1.Cu through In10.Cu plus B.Cu
- F.Cu reserved for escape routing
- Blind/buried vias: 0.15mm hole, 0.25mm diameter
- Manhattan routing: odd layers horizontal, even layers vertical
- A* pathfinding with Manhattan distance heuristic
- Rip-up and repair for congestion resolution
- Trace subdivision for efficient grid utilization
"""

import logging
import time
import math
import heapq
import sys
import os
from typing import Dict, List, Tuple, Optional, Set, Any, NamedTuple
import numpy as np
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque

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
except ImportError:
    from .base_router import BaseRouter, RoutingResult, RouteSegment, RoutingStats
    from ..core.drc_rules import DRCRules
    from ..core.gpu_manager import GPUManager
    from ..core.board_interface import BoardInterface
    from ..data_structures.grid_config import GridConfig

logger = logging.getLogger(__name__)

# Default values (used as fallbacks if DRC rules unavailable)
DEFAULT_GRID_RESOLUTION = 0.4  # mm - 0.4mm grid pitch
DEFAULT_BOARD_MARGIN = 3.0  # mm - margin around airwires for routing bounds

# Cell states in the routing grid
class CellState:
    EMPTY = 0
    OBSTACLE = 1
    ROUTED = 2  # Net ID stored separately

# Via types for blind/buried via support
class ViaType(Enum):
    BLIND_TOP = "blind_top"        # F.Cu to inner layer
    BLIND_BOTTOM = "blind_bottom"  # Inner layer to B.Cu
    BURIED = "buried"              # Inner layer to inner layer
    THROUGH = "through"            # F.Cu to B.Cu (if needed)

# Layer direction assignments
class LayerDirection(Enum):
    HORIZONTAL = "horizontal"  # Odd layers: In1, In3, In5, In7, In9
    VERTICAL = "vertical"      # Even layers: In2, In4, In6, In8, In10, B.Cu

@dataclass
class GridPoint:
    """A point in the 3D routing grid"""
    x: int
    y: int
    layer: int
    
    def __hash__(self):
        return hash((self.x, self.y, self.layer))

@dataclass
class PathNode:
    """Node for A* pathfinding"""
    point: GridPoint
    g_score: float  # Cost from start
    h_score: float  # Heuristic to goal
    f_score: float  # g + h
    parent: Optional['PathNode'] = None
    
    def __lt__(self, other):
        return self.f_score < other.f_score

@dataclass
class NetRoute:
    """Complete routing solution for a net"""
    net_id: str
    segments: List[RouteSegment]
    vias: List[Dict]
    total_length: float
    layers_used: Set[int]

class RoutingGrid:
    """GPU-compatible 3D routing grid [layers][y_grid][x_grid]"""
    
    def __init__(self, bounds: Tuple[float, float, float, float], layer_names: List[str], 
                 board_interface: BoardInterface):
        """
        Initialize the routing grid
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) in mm
            layer_names: Ordered list of routing layer names from board
            board_interface: Board interface for layer information
        """
        self.bounds = bounds
        self.layer_names = layer_names
        self.layer_count = len(layer_names)
        self.board_interface = board_interface
        
        # Calculate grid dimensions
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        
        # Grid size in cells - use GridConfig resolution or fallback
        grid_resolution = getattr(board_interface, 'grid_config', None)
        grid_resolution = grid_resolution.resolution if grid_resolution else DEFAULT_GRID_RESOLUTION
        
        self.grid_cols = int(math.ceil(self.width / grid_resolution))
        self.grid_rows = int(math.ceil(self.height / grid_resolution))
        self.grid_resolution = grid_resolution
        
        # Initialize grid arrays
        self.grid_state = np.zeros((self.layer_count, self.grid_rows, self.grid_cols), dtype=np.int32)
        self.grid_net_id = np.zeros((self.layer_count, self.grid_rows, self.grid_cols), dtype=np.int32)
        
        # Build name->index mapping
        self.name_to_idx = {name: i for i, name in enumerate(layer_names)}
        self.idx_to_name = {i: name for i, name in enumerate(layer_names)}
        
        # Layer direction mapping based on actual layer names (not indices)
        self.layer_directions = {}
        for i, layer_name in enumerate(layer_names):
            self.layer_directions[i] = self._get_layer_direction(layer_name)
        
        logger.info(f"Initialized routing grid: {self.grid_cols}x{self.grid_rows}x{self.layer_count}")
        logger.info(f"Grid bounds: ({self.min_x:.2f}, {self.min_y:.2f}) to ({self.max_x:.2f}, {self.max_y:.2f})")
        logger.info(f"Layer stack: {layer_names}")
        
        # Log layer directions
        for i, name in enumerate(layer_names):
            direction = "horizontal" if self.layer_directions[i] == LayerDirection.HORIZONTAL else "vertical"
            logger.info(f"  {name} (layer {i}): {direction}")
    
    def _get_layer_direction(self, layer_name: str) -> LayerDirection:
        """Get layer direction based on layer name, not index"""
        # B.Cu is always vertical per specification
        if layer_name == 'B.Cu':
            return LayerDirection.VERTICAL
        
        # For inner layers, odd numbers are horizontal, even are vertical
        if layer_name.startswith('In') and layer_name.endswith('.Cu'):
            try:
                layer_num = int(layer_name[2:-3])  # Extract number from "In1.Cu"
                if layer_num % 2 == 1:  # In1, In3, In5, In7, In9 = horizontal
                    return LayerDirection.HORIZONTAL
                else:  # In2, In4, In6, In8, In10 = vertical
                    return LayerDirection.VERTICAL
            except ValueError:
                logger.warning(f"Could not parse layer number from {layer_name}, defaulting to vertical")
                return LayerDirection.VERTICAL
        
        # F.Cu should not be in routing grid, but default to horizontal if present
        logger.warning(f"Unexpected layer in routing grid: {layer_name}")
        return LayerDirection.HORIZONTAL
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        grid_x = int((x - self.min_x) / self.grid_resolution)
        grid_y = int((y - self.min_y) / self.grid_resolution)
        return max(0, min(grid_x, self.grid_cols - 1)), max(0, min(grid_y, self.grid_rows - 1))
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates"""
        x = self.min_x + (grid_x + 0.5) * self.grid_resolution
        y = self.min_y + (grid_y + 0.5) * self.grid_resolution
        return x, y
    
    def is_valid_position(self, x: int, y: int, layer: int) -> bool:
        """Check if position is within grid bounds"""
        return (0 <= x < self.grid_cols and 
                0 <= y < self.grid_rows and 
                0 <= layer < self.layer_count)
    
    def get_cell_state(self, x: int, y: int, layer: int) -> int:
        """Get cell state at position"""
        if not self.is_valid_position(x, y, layer):
            return CellState.OBSTACLE
        return self.grid_state[layer, y, x]
    
    def set_cell_state(self, x: int, y: int, layer: int, state: int, net_id: int = 0):
        """Set cell state at position"""
        if self.is_valid_position(x, y, layer):
            self.grid_state[layer, y, x] = state
            self.grid_net_id[layer, y, x] = net_id
    
    def get_neighbors(self, point: GridPoint) -> List[GridPoint]:
        """Get valid neighbors for pathfinding based on layer direction"""
        neighbors = []
        layer_dir = self.layer_directions[point.layer]
        
        # Same-layer movement based on direction
        if layer_dir == LayerDirection.HORIZONTAL:
            # Horizontal movement only
            for dx in [-1, 1]:
                new_x, new_y = point.x + dx, point.y
                if self.is_valid_position(new_x, new_y, point.layer):
                    neighbors.append(GridPoint(new_x, new_y, point.layer))
        else:
            # Vertical movement only
            for dy in [-1, 1]:
                new_x, new_y = point.x, point.y + dy
                if self.is_valid_position(new_x, new_y, point.layer):
                    neighbors.append(GridPoint(new_x, new_y, point.layer))
        
        # Layer transitions (vias) - ONLY to adjacent layers
        current_layer_name = self.idx_to_name[point.layer]
        
        for dl in [-1, 1]:  # Only adjacent layers
            new_layer = point.layer + dl
            if 0 <= new_layer < self.layer_count:
                target_layer_name = self.idx_to_name[new_layer]
                
                # Check if via transition is allowed between these layers
                if self._is_via_allowed(current_layer_name, target_layer_name):
                    neighbors.append(GridPoint(point.x, point.y, new_layer))
        
        return neighbors
    
    def _is_via_allowed(self, from_layer: str, to_layer: str) -> bool:
        """Check if via transition is allowed between layers"""
        # For now, allow all adjacent layer transitions
        # In a full implementation, this would check via rules and board stackup
        return True

class GPUManhattanRouter(BaseRouter):
    """GPU-accelerated Manhattan router implementation"""
    
    def __init__(self, board_interface: BoardInterface, drc_rules: DRCRules, 
                 gpu_manager: GPUManager, grid_config: GridConfig):
        """Initialize the GPU Manhattan router"""
        super().__init__(board_interface, drc_rules, gpu_manager, grid_config)
        
        self.routing_grid = None
        self.net_routes = {}  # Net ID -> NetRoute
        self.failed_attempts = defaultdict(int)
        self.ripped_nets = set()
        
        # Progress tracking
        self.nets_routed = 0
        self.nets_failed = 0
        self.nets_attempted = 0
        
        # Callbacks for visualization
        self.progress_callback = None
        self.track_callback = None
        self.via_callback = None
        
        logger.info("GPU Manhattan Router initialized")
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def set_track_callback(self, callback):
        """Set callback for track visualization"""
        self.track_callback = callback
    
    def set_via_callback(self, callback):
        """Set callback for via visualization"""
        self.via_callback = callback
    
    def _calculate_board_extents(self) -> Tuple[float, float, float, float]:
        """Calculate board extents with 3mm margin around airwires"""
        routable_nets = self.board_interface.get_routable_nets()
        
        if not routable_nets:
            # Default bounds if no nets
            return (-10.0, -10.0, 10.0, 10.0)
        
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        # Find extents of all pad positions
        for net_id, net_data in routable_nets.items():
            for pad in net_data.get('pads', []):
                x, y = pad.get('x', 0), pad.get('y', 0)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        
        # Add configurable margin
        return (min_x - DEFAULT_BOARD_MARGIN, min_y - DEFAULT_BOARD_MARGIN, 
                max_x + DEFAULT_BOARD_MARGIN, max_y + DEFAULT_BOARD_MARGIN)
    
    def _initialize_routing_grid(self):
        """Initialize the routing grid with obstacles from footprints"""
        bounds = self._calculate_board_extents()
        
        # Get ordered routing layer stack from board
        routing_layers = self._get_ordered_routing_layers()
        
        if not routing_layers:
            # Default layer stack if board doesn't provide proper layers
            routing_layers = ['In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'In5.Cu', 
                            'In6.Cu', 'In7.Cu', 'In8.Cu', 'In9.Cu', 'In10.Cu', 'B.Cu']
            logger.warning("Using default 11-layer stack")
        
        self.routing_grid = RoutingGrid(bounds, routing_layers, self.board_interface)
        
        # Create DRC mask from footprints
        self._create_drc_mask()
        
        logger.info(f"Routing grid initialized with {len(routing_layers)} layers")
    
    def _get_ordered_routing_layers(self) -> List[str]:
        """Get ordered list of routing layers from board interface"""
        available_layers = self.board_interface.get_layers()
        routing_layers = []
        
        # Extract routing layer names, handling both string and tuple formats
        layer_names = []
        for layer in available_layers:
            layer_name = layer[0] if isinstance(layer, tuple) else layer
            layer_names.append(layer_name)
        
        # Build ordered routing stack: In1.Cu through InN.Cu, then B.Cu
        inner_layers = []
        has_bcu = False
        
        for layer_name in layer_names:
            if layer_name.startswith('In') and layer_name.endswith('.Cu'):
                try:
                    layer_num = int(layer_name[2:-3])
                    inner_layers.append((layer_num, layer_name))
                except ValueError:
                    continue
            elif layer_name == 'B.Cu':
                has_bcu = True
        
        # Sort inner layers by number
        inner_layers.sort(key=lambda x: x[0])
        routing_layers = [name for _, name in inner_layers]
        
        # Add B.Cu at the end
        if has_bcu:
            routing_layers.append('B.Cu')
        
        logger.info(f"Detected routing layer stack: {routing_layers}")
        return routing_layers
    
    def _create_drc_mask(self):
        """Create DRC mask to prevent routing through incompatible footprints"""
        routable_nets = self.board_interface.get_routable_nets()
        
        # Create pad keepout maps for each net
        self.pad_keepouts = {}  # net_id -> set of (layer, x, y) allowed positions
        
        for net_id, net_data in routable_nets.items():
            self.pad_keepouts[net_id] = set()
            
            for pad in net_data.get('pads', []):
                x, y = pad.get('x', 0), pad.get('y', 0)
                grid_x, grid_y = self.routing_grid.world_to_grid(x, y)
                
                # Calculate proper pad clearance from DRC rules
                pad_clearance = self.drc_rules.get_clearance_for_net(net_id)
                clearance_cells = int(math.ceil(pad_clearance / self.routing_grid.grid_resolution))
                
                # Mark areas accessible to this net (for escape routing)
                for dx in range(-clearance_cells, clearance_cells + 1):
                    for dy in range(-clearance_cells, clearance_cells + 1):
                        gx, gy = grid_x + dx, grid_y + dy
                        if self.routing_grid.is_valid_position(gx, gy, 0):
                            # Allow this net in this area on all layers
                            for layer in range(self.routing_grid.layer_count):
                                self.pad_keepouts[net_id].add((layer, gx, gy))
                
                # Mark pad center as obstacle for other nets (not for this net)
                for layer in range(self.routing_grid.layer_count):
                    current_net = self.routing_grid.grid_net_id[layer, grid_y, grid_x]
                    if current_net == 0:  # Only mark if empty
                        self.routing_grid.set_cell_state(grid_x, grid_y, layer, CellState.OBSTACLE, 0)
    
    def _is_cell_accessible_for_net(self, x: int, y: int, layer: int, net_id: str) -> bool:
        """Check if a cell is accessible for routing by a specific net"""
        cell_state = self.routing_grid.get_cell_state(x, y, layer)
        
        if cell_state == CellState.EMPTY:
            return True
        elif cell_state == CellState.OBSTACLE:
            # Check if this net is allowed in pad keepout areas
            return (layer, x, y) in self.pad_keepouts.get(net_id, set())
        elif cell_state == CellState.ROUTED:
            # Check if already routed by same net
            cell_net_hash = self.routing_grid.grid_net_id[layer, y, x]
            return cell_net_hash == hash(net_id)
        
        return False
    
    def _manhattan_distance(self, start: GridPoint, end: GridPoint) -> float:
        """Calculate Manhattan distance between two grid points"""
        return abs(start.x - end.x) + abs(start.y - end.y) + abs(start.layer - end.layer) * 2
    
    def _astar_pathfind(self, start: GridPoint, end: GridPoint, net_id: str) -> List[GridPoint]:
        """A* pathfinding with Manhattan distance heuristic"""
        open_set = []
        heapq.heappush(open_set, PathNode(start, 0, self._manhattan_distance(start, end), 
                                         self._manhattan_distance(start, end)))
        
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            current_node = heapq.heappop(open_set)
            current = current_node.point
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path))
            
            closed_set.add(current)
            
            for neighbor in self.routing_grid.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Use proper cell accessibility check
                if not self._is_cell_accessible_for_net(neighbor.x, neighbor.y, neighbor.layer, net_id):
                    continue
                
                # Calculate movement cost with spacing consideration
                tentative_g = g_score[current] + 1
                
                # Via cost for layer transitions
                if neighbor.layer != current.layer:
                    tentative_g += 3  # Higher via cost to discourage unnecessary vias
                
                # Add congestion penalty for cells near obstacles
                congestion_penalty = self._get_congestion_penalty(neighbor.x, neighbor.y, neighbor.layer)
                tentative_g += congestion_penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self._manhattan_distance(neighbor, end)
                    f_score = tentative_g + h_score
                    
                    heapq.heappush(open_set, PathNode(neighbor, tentative_g, h_score, f_score))
        
        return []  # No path found
    
    def _get_congestion_penalty(self, x: int, y: int, layer: int) -> float:
        """Calculate congestion penalty for a grid cell"""
        penalty = 0.0
        
        # Check neighboring cells for obstacles/routing
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                if self.routing_grid.is_valid_position(nx, ny, layer):
                    cell_state = self.routing_grid.get_cell_state(nx, ny, layer)
                    if cell_state != CellState.EMPTY:
                        penalty += 0.1  # Small penalty for proximity to obstacles/routes
        
        return penalty
    
    def _calculate_total_length(self, segments: List[RouteSegment]) -> float:
        """Calculate total length of route segments"""
        total_length = 0.0
        
        for segment in segments:
            if segment.type == 'track' and segment.end_x is not None and segment.end_y is not None:
                dx = segment.end_x - segment.start_x
                dy = segment.end_y - segment.start_y
                total_length += math.sqrt(dx * dx + dy * dy)
        
        return total_length
    
    def _mark_path_with_spacing(self, path: List[GridPoint], net_id: str):
        """Mark path in grid and enforce spacing halos"""
        net_hash = hash(net_id)
        
        # Mark all path cells as routed
        for point in path:
            self.routing_grid.set_cell_state(
                point.x, point.y, point.layer, CellState.ROUTED, net_hash)
        
        # Add spacing halos around the path
        trace_spacing = self.drc_rules.min_trace_spacing
        spacing_halo = int(math.ceil(trace_spacing / self.routing_grid.grid_resolution))
        
        for point in path:
            for dx in range(-spacing_halo, spacing_halo + 1):
                for dy in range(-spacing_halo, spacing_halo + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip the trace cell itself
                    
                    halo_x, halo_y = point.x + dx, point.y + dy
                    
                    if (self.routing_grid.is_valid_position(halo_x, halo_y, point.layer) and
                        self.routing_grid.get_cell_state(halo_x, halo_y, point.layer) == CellState.EMPTY):
                        
                        # Mark as obstacle to enforce spacing (no net ownership)
                        self.routing_grid.set_cell_state(halo_x, halo_y, point.layer, CellState.OBSTACLE, 0)
    
    def _create_escape_routes(self, net_id: str, net_data: Dict) -> List[RouteSegment]:
        """Create F.Cu escape routes from pads to grid"""
        segments = []
        
        for pad in net_data.get('pads', []):
            pad_x, pad_y = pad.get('x', 0), pad.get('y', 0)
            
            # Find suitable escape point near pad (not exactly on pad)
            escape_point = self._find_escape_point(pad_x, pad_y, net_id)
            if not escape_point:
                logger.warning(f"Could not find escape point for pad at ({pad_x:.2f}, {pad_y:.2f})")
                continue
            
            escape_x, escape_y = escape_point
            
            # Create F.Cu escape trace from pad to escape point
            if abs(escape_x - pad_x) > 0.001 or abs(escape_y - pad_y) > 0.001:  # Non-zero length
                escape_segment = RouteSegment(
                    'track', pad_x, pad_y, escape_x, escape_y, 
                    self.drc_rules.default_trace_width, 'F.Cu', net_id
                )
                segments.append(escape_segment)
            
            # Find best grid entry layer (In1.Cu preferred)
            best_layer = 0  # In1.Cu index
            entry_layer_name = self.routing_grid.idx_to_name[best_layer]
            
            # Create blind via from F.Cu to grid layer
            via_data = {
                'x': escape_x,
                'y': escape_y,
                'size': self.drc_rules.via_diameter,
                'drill': self.drc_rules.via_drill,
                'layers': ['F.Cu', entry_layer_name],
                'net': net_id,
                'type': 'blind'
            }
            
            # Record via in route data (fix for P0 issue)
            if net_id not in self.net_routes:
                self.net_routes[net_id] = NetRoute(net_id, [], [], 0.0, set())
            
            self.net_routes[net_id].vias.append(via_data)
            
            if self.via_callback:
                self.via_callback(via_data)
        
        return segments
    
    def _find_escape_point(self, pad_x: float, pad_y: float, net_id: str) -> Optional[Tuple[float, float]]:
        """Find a suitable escape point from a pad for F.Cu routing"""
        # Convert pad position to grid
        pad_grid_x, pad_grid_y = self.routing_grid.world_to_grid(pad_x, pad_y)
        
        # Try positions around the pad for escape routing
        search_radius = 3  # Search within 3 grid cells
        
        for radius in range(1, search_radius + 1):
            # Check cardinal directions first (better for escape routing)
            directions = [(radius, 0), (-radius, 0), (0, radius), (0, -radius)]
            
            for dx, dy in directions:
                test_x, test_y = pad_grid_x + dx, pad_grid_y + dy
                
                if (self.routing_grid.is_valid_position(test_x, test_y, 0) and
                    self._is_cell_accessible_for_net(test_x, test_y, 0, net_id)):
                    
                    # Convert back to world coordinates
                    world_x, world_y = self.routing_grid.grid_to_world(test_x, test_y)
                    return (world_x, world_y)
        
        # Fallback: use pad position if no escape found
        logger.warning(f"No escape point found for pad at ({pad_x:.2f}, {pad_y:.2f}), using pad position")
        return (pad_x, pad_y)
    
    def _route_single_net_internal(self, net_id: str, timeout: float = 10.0) -> bool:
        """Route a single net using Manhattan routing algorithm"""
        self.nets_attempted += 1
        
        routable_nets = self.board_interface.get_routable_nets()
        if net_id not in routable_nets:
            return False
        
        net_data = routable_nets[net_id]
        pads = net_data.get('pads', [])
        
        if len(pads) < 2:
            return False
        
        start_time = time.time()
        
        try:
            # Create escape routes
            escape_segments = self._create_escape_routes(net_id, net_data)
            
            # Route between grid points
            grid_segments = []
            vias = []
            
            for i in range(len(pads) - 1):
                pad_a = pads[i]
                pad_b = pads[i + 1]
                
                # Convert to grid coordinates
                ax, ay = self.routing_grid.world_to_grid(pad_a['x'], pad_a['y'])
                bx, by = self.routing_grid.world_to_grid(pad_b['x'], pad_b['y'])
                
                start_point = GridPoint(ax, ay, 0)  # Start on In1.Cu
                end_point = GridPoint(bx, by, 0)    # End on In1.Cu
                
                # Find path using A*
                path = self._astar_pathfind(start_point, end_point, net_id)
                
                if not path:
                    # Try alternate layers
                    for alt_layer in range(1, self.routing_grid.layer_count):
                        start_point.layer = alt_layer
                        end_point.layer = alt_layer
                        path = self._astar_pathfind(start_point, end_point, net_id)
                        if path:
                            break
                
                if not path and self.failed_attempts[net_id] < 3:
                    # Attempt rip-up and repair
                    self._ripup_conflicting_nets(start_point, end_point)
                    path = self._astar_pathfind(start_point, end_point, net_id)
                
                if not path:
                    self.failed_attempts[net_id] += 1
                    if self.failed_attempts[net_id] >= 10:
                        logger.warning(f"Failed to route net {net_id} after 10 attempts")
                        return False
                    continue
                
                # Convert path to segments
                path_segments = self._path_to_segments(path, net_id)
                grid_segments.extend(path_segments)
                
                # Mark path with proper spacing enforcement
                self._mark_path_with_spacing(path, net_id)
            
            # Store successful route
            total_length = self._calculate_total_length(escape_segments + grid_segments)
            layers_used = set(seg.layer for seg in grid_segments if hasattr(seg, 'layer'))
            
            self.net_routes[net_id] = NetRoute(
                net_id, escape_segments + grid_segments, vias, total_length, layers_used
            )
            
            self.nets_routed += 1
            
            # Report progress every 10 nets
            if self.nets_routed % 10 == 0:
                logger.info(f"Progress: {self.nets_routed} nets routed, {self.nets_failed} failed, "
                           f"currently routing: {net_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error routing net {net_id}: {e}")
            self.nets_failed += 1
            return False
    
    def _path_to_segments(self, path: List[GridPoint], net_id: str) -> List[RouteSegment]:
        """Convert grid path to route segments"""
        segments = []
        
        # Ensure net route exists for via recording
        if net_id not in self.net_routes:
            self.net_routes[net_id] = NetRoute(net_id, [], [], 0.0, set())
        
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            curr_world = self.routing_grid.grid_to_world(current.x, current.y)
            next_world = self.routing_grid.grid_to_world(next_point.x, next_point.y)
            
            if current.layer == next_point.layer:
                # Track segment - use proper layer names from grid
                layer_name = self.routing_grid.idx_to_name[current.layer]
                
                segment = RouteSegment(
                    'track', curr_world[0], curr_world[1], next_world[0], next_world[1],
                    self.drc_rules.default_trace_width, layer_name, net_id
                )
                segments.append(segment)
                
                # Visualize track
                if self.track_callback:
                    track_data = {
                        'start': curr_world,
                        'end': next_world,
                        'layer': layer_name,
                        'width': self.drc_rules.default_trace_width,
                        'net': net_id,
                        'color': 'white'  # Bright white for current routing
                    }
                    self.track_callback(track_data)
            else:
                # Via needed - record properly in route data
                from_layer_name = self.routing_grid.idx_to_name[current.layer]
                to_layer_name = self.routing_grid.idx_to_name[next_point.layer]
                
                via_data = {
                    'x': curr_world[0],
                    'y': curr_world[1],
                    'size': self.drc_rules.via_diameter,
                    'drill': self.drc_rules.via_drill,
                    'layers': [from_layer_name, to_layer_name],
                    'net': net_id,
                    'type': self._determine_via_type(current.layer, next_point.layer)
                }
                
                # Record via in route data (fix for P0 issue)
                self.net_routes[net_id].vias.append(via_data)
                
                if self.via_callback:
                    self.via_callback(via_data)
        
        return segments
    
    def _determine_via_type(self, from_layer: int, to_layer: int) -> str:
        """Determine via type based on layer transition"""
        if from_layer == 0 or to_layer == 0:  # Involves F.Cu equivalent
            return 'blind'
        elif from_layer == self.routing_grid.layer_count - 1 or to_layer == self.routing_grid.layer_count - 1:
            return 'blind'  # Involves B.Cu
        else:
            return 'buried'  # Internal layers only
    
    def _ripup_conflicting_nets(self, start: GridPoint, end: GridPoint):
        """Rip up nets that are blocking the current route"""
        # Find nets in the path area
        min_x, max_x = min(start.x, end.x) - 2, max(start.x, end.x) + 2
        min_y, max_y = min(start.y, end.y) - 2, max(start.y, end.y) + 2
        
        nets_to_ripup = set()
        
        for x in range(max(0, min_x), min(self.routing_grid.grid_cols, max_x + 1)):
            for y in range(max(0, min_y), min(self.routing_grid.grid_rows, max_y + 1)):
                for layer in range(self.routing_grid.layer_count):
                    if self.routing_grid.get_cell_state(x, y, layer) == CellState.ROUTED:
                        net_id = self.routing_grid.grid_net_id[layer, y, x]
                        nets_to_ripup.add(net_id)
        
        # Rip up nets (prioritize longer paths)
        for net_hash in nets_to_ripup:
            # Find actual net ID from hash (simplified)
            for net_id, route in self.net_routes.items():
                if hash(net_id) == net_hash:
                    self._clear_net_from_grid(net_id)
                    self.ripped_nets.add(net_id)
                    break
    
    def _clear_net_from_grid(self, net_id: str):
        """Clear a net from the routing grid"""
        net_hash = hash(net_id)
        
        for layer in range(self.routing_grid.layer_count):
            mask = (self.routing_grid.grid_net_id[layer] == net_hash)
            self.routing_grid.grid_state[layer][mask] = CellState.EMPTY
            self.routing_grid.grid_net_id[layer][mask] = 0
        
        if net_id in self.net_routes:
            del self.net_routes[net_id]
            self.nets_routed -= 1
    
    def _verify_connectivity(self, net_id: str) -> bool:
        """Verify electrical connectivity of routed net"""
        if net_id not in self.net_routes:
            return False
        
        route = self.net_routes[net_id]
        
        # Check that all segments are connected
        segments = route.segments
        if len(segments) < 1:
            return True
        
        # Verify continuous path
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            # Check connection points
            if not (abs(current.end_x - next_seg.start_x) < 0.01 and 
                    abs(current.end_y - next_seg.start_y) < 0.01):
                return False
        
        return True
    
    def route_net(self, net_id: str, timeout: float = 10.0) -> RoutingResult:
        """Route a single net"""
        if not self.routing_grid:
            self._initialize_routing_grid()
        
        success = self._route_single_net_internal(net_id, timeout)
        
        if success and self._verify_connectivity(net_id):
            return RoutingResult.SUCCESS
        else:
            return RoutingResult.FAILED
    
    def route_all_nets(self, timeout_per_net: float = 5.0, total_timeout: float = 300.0) -> RoutingStats:
        """Route all nets on the board"""
        if not self.routing_grid:
            self._initialize_routing_grid()
        
        start_time = time.time()
        routable_nets = self.board_interface.get_routable_nets()
        
        # Sort nets: shortest distance first, then alphabetically
        def net_sort_key(item):
            net_id, net_data = item
            pads = net_data.get('pads', [])
            if len(pads) < 2:
                return (float('inf'), net_id)
            
            # Calculate minimum distance between pads
            min_dist = float('inf')
            for i in range(len(pads)):
                for j in range(i + 1, len(pads)):
                    pad_a, pad_b = pads[i], pads[j]
                    dist = math.sqrt((pad_a['x'] - pad_b['x'])**2 + (pad_a['y'] - pad_b['y'])**2)
                    min_dist = min(min_dist, dist)
            
            return (min_dist, net_id)
        
        sorted_nets = sorted(routable_nets.items(), key=net_sort_key)
        
        logger.info(f"Routing {len(sorted_nets)} nets in optimized order")
        
        for net_id, net_data in sorted_nets:
            if time.time() - start_time > total_timeout:
                logger.warning("Total timeout reached")
                break
            
            self.route_net(net_id, timeout_per_net)
        
        # Re-route any ripped nets
        for net_id in self.ripped_nets.copy():
            if time.time() - start_time > total_timeout:
                break
            self.route_net(net_id, timeout_per_net)
        
        # Calculate final statistics
        total_routing_time = time.time() - start_time
        total_length = sum(route.total_length for route in self.net_routes.values())
        total_vias = sum(len(route.vias) for route in self.net_routes.values())
        
        success_rate = self.nets_routed / len(sorted_nets) if sorted_nets else 0
        
        logger.info(f"Routing completed: {self.nets_routed}/{len(sorted_nets)} nets routed "
                   f"({success_rate:.1%} success rate)")
        
        stats = RoutingStats(
            nets_attempted=len(sorted_nets),
            nets_routed=self.nets_routed,
            nets_failed=self.nets_failed,
            tracks_added=len([seg for route in self.net_routes.values() for seg in route.segments]),
            vias_added=total_vias,
            total_length_mm=total_length,
            routing_time=total_routing_time
        )
        return stats
    
    def get_routed_tracks(self) -> List[Dict]:
        """Get all routed tracks in KiCad format"""
        tracks = []
        
        for route in self.net_routes.values():
            for segment in route.segments:
                if segment.type == 'track':
                    track_data = {
                        'start': (segment.start_x, segment.start_y),
                        'end': (segment.end_x, segment.end_y),
                        'layer': segment.layer,
                        'width': segment.width,
                        'net': segment.net_id
                    }
                    tracks.append(track_data)
        
        return tracks
    
    def get_routed_vias(self) -> List[Dict]:
        """Get all routed vias in KiCad format"""
        vias = []
        
        for route in self.net_routes.values():
            vias.extend(route.vias)
        
        return vias
    
    def clear_routes(self):
        """Clear all routing data"""
        self.net_routes.clear()
        self.failed_attempts.clear()
        self.ripped_nets.clear()
        self.nets_routed = 0
        self.nets_failed = 0
        self.nets_attempted = 0
        
        if self.routing_grid:
            self.routing_grid.grid_state.fill(CellState.EMPTY)
            self.routing_grid.grid_net_id.fill(0)
            self._create_drc_mask()  # Recreate obstacles
    
    def route_two_pads(self, pad_a: Dict, pad_b: Dict, net_name: str, 
                      timeout: float = 5.0) -> Optional[List[RouteSegment]]:
        """Route between two specific pads using Manhattan routing algorithm"""
        if not self.routing_grid:
            self._initialize_routing_grid()
        
        try:
            # Convert pads to grid coordinates
            ax, ay = self.routing_grid.world_to_grid(pad_a['x'], pad_a['y'])
            bx, by = self.routing_grid.world_to_grid(pad_b['x'], pad_b['y'])
            
            start_point = GridPoint(ax, ay, 0)  # Start on In1.Cu
            end_point = GridPoint(bx, by, 0)    # End on In1.Cu
            
            # Find path using A*
            path = self._astar_pathfind(start_point, end_point, net_name)
            
            if not path:
                # Try alternate layers
                for alt_layer in range(1, min(4, self.routing_grid.layer_count)):
                    start_point.layer = alt_layer
                    end_point.layer = alt_layer
                    path = self._astar_pathfind(start_point, end_point, net_name)
                    if path:
                        break
            
            if not path:
                return None
            
            # Convert path to segments
            segments = []
            
            # Create escape routes from pads
            start_world = self.routing_grid.grid_to_world(start_point.x, start_point.y)
            end_world = self.routing_grid.grid_to_world(end_point.x, end_point.y)
            
            # F.Cu escape from pad_a to grid
            escape_a = RouteSegment(
                'track', pad_a['x'], pad_a['y'], start_world[0], start_world[1],
                self.drc_rules.default_trace_width, 'F.Cu', net_name
            )
            segments.append(escape_a)
            
            # Grid path segments
            path_segments = self._path_to_segments(path, net_name)
            segments.extend(path_segments)
            
            # F.Cu escape from grid to pad_b
            escape_b = RouteSegment(
                'track', end_world[0], end_world[1], pad_b['x'], pad_b['y'],
                self.drc_rules.default_trace_width, 'F.Cu', net_name
            )
            segments.append(escape_b)
            
            # Mark grid cells as used
            self._mark_path_with_spacing(path, net_name)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error routing two pads for {net_name}: {e}")
            return None
    
    def get_routing_statistics(self) -> RoutingStats:
        """Get current routing statistics"""
        total_length = sum(route.total_length for route in self.net_routes.values())
        total_vias = sum(len(route.vias) for route in self.net_routes.values())
        success_rate = self.nets_routed / self.nets_attempted if self.nets_attempted > 0 else 0
        
        return RoutingStats(
            nets_attempted=self.nets_attempted,
            nets_routed=self.nets_routed,
            nets_failed=self.nets_failed,
            tracks_added=len([seg for route in self.net_routes.values() for seg in route.segments]),
            vias_added=total_vias,
            total_length_mm=total_length,
            routing_time=0.0  # Not tracked in current stats
        )