"""
Grid management for Manhattan routing
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from .types import Pad, RoutingConfig, FREE, PAD, TRACE, KEEPOUT

logger = logging.getLogger(__name__)

class GridManager:
    """Manages the 3D routing grid for Manhattan routing"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        
        # Grid dimensions
        self.grid_width = 0
        self.grid_height = 0
        self.num_layers = 0
        
        # Layer configuration
        self.layer_directions = {}  # layer_idx -> 'H' or 'V'
        self.layer_mapping = {}  # KiCad layer name -> grid layer index
        self.reverse_layer_mapping = {}  # grid layer index -> KiCad layer name
        
        # 3D grids [layer, y, x] - GPU compatible format
        self.occupancy = None    # uint8: 0=free, 1=obstacle, 2=routed
        self.net_id_grid = None  # int32: 0=none, >0=net_id
        self.cost_grid = None    # uint16: congestion penalty
        
        # Routing bounds
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
    
    def initialize_from_board(self, board_data: Dict[str, Any], board_pads: List[Pad]):
        """Initialize grid from board data and pad snapshot"""
        logger.info("🏗️ Initializing Manhattan routing grid")
        
        # Calculate routing bounds from airwires/pads with expansion margin
        self._calculate_routing_bounds(board_pads)
        
        # Set up layers for Manhattan routing
        self._setup_manhattan_layers(board_data)
        
        # Allocate GPU-compatible grid arrays
        self._allocate_grid_arrays()
        
        # Rasterize obstacles (footprints, keepouts)
        self._rasterize_obstacles(board_pads)
        
        logger.info(f"Grid initialized: {self.grid_width}x{self.grid_height}x{self.num_layers}")
        logger.info(f"Routing bounds: ({self.min_x:.2f},{self.min_y:.2f}) to ({self.max_x:.2f},{self.max_y:.2f})")
    
    def _calculate_routing_bounds(self, board_pads: List[Pad]):
        """Calculate routing bounds from pads with expansion margin"""
        if not board_pads:
            self.min_x = self.min_y = 0.0
            self.max_x = self.max_y = 10.0
            return
        
        # Find extents of all pads
        pad_xs = [pad.x_mm for pad in board_pads]
        pad_ys = [pad.y_mm for pad in board_pads]
        
        pad_min_x, pad_max_x = min(pad_xs), max(pad_xs)
        pad_min_y, pad_max_y = min(pad_ys), max(pad_ys)
        
        # Add expansion margin per spec
        self.min_x = pad_min_x - self.config.expansion_margin
        self.max_x = pad_max_x + self.config.expansion_margin
        self.min_y = pad_min_y - self.config.expansion_margin
        self.max_y = pad_max_y + self.config.expansion_margin
        
        # Calculate grid dimensions
        self.grid_width = int(np.ceil((self.max_x - self.min_x) / self.config.grid_pitch))
        self.grid_height = int(np.ceil((self.max_y - self.min_y) / self.config.grid_pitch))
    
    def _setup_manhattan_layers(self, board_data: Dict[str, Any]):
        """Set up Manhattan layer configuration"""
        # Per spec: In1-In10 + B.Cu (11 layers total)
        # Odd layers (In1,In3,In5,In7,In9) = horizontal
        # Even layers (In2,In4,In6,In8,In10,B.Cu) = vertical
        
        manhattan_layers = []
        
        # Add internal layers In1 through In10
        for i in range(1, 11):  # In1.Cu through In10.Cu
            layer_name = f"In{i}.Cu"
            manhattan_layers.append(layer_name)
        
        # Add bottom layer
        manhattan_layers.append("B.Cu")
        
        self.num_layers = len(manhattan_layers)
        
        # Create layer mappings
        for idx, layer_name in enumerate(manhattan_layers):
            self.layer_mapping[layer_name] = idx
            self.reverse_layer_mapping[idx] = layer_name
            
            # Set direction based on Manhattan routing pattern
            if layer_name == "B.Cu":
                self.layer_directions[idx] = 'V'  # Bottom layer is vertical
            else:
                layer_num = int(layer_name[2:-3])  # Extract number from "InX.Cu"
                self.layer_directions[idx] = 'H' if layer_num % 2 == 1 else 'V'
        
        logger.info(f"Manhattan layers configured: {len(manhattan_layers)} layers")
        for idx, layer_name in enumerate(manhattan_layers):
            direction = self.layer_directions[idx]
            logger.info(f"  Layer {idx}: {layer_name} ({direction})")
    
    def _allocate_grid_arrays(self):
        """Allocate GPU-compatible 3D grid arrays"""
        shape = (self.num_layers, self.grid_height, self.grid_width)
        
        # Use GPU-compatible data types
        self.occupancy = np.zeros(shape, dtype=np.uint8)  # 0=free, 1=obstacle, 2=routed
        self.net_id_grid = np.zeros(shape, dtype=np.int32)  # 0=none, >0=net_id
        self.cost_grid = np.ones(shape, dtype=np.uint16)  # Base cost = 1
        
        logger.info(f"Allocated grid arrays: {shape} ({self.occupancy.nbytes / 1024**2:.1f}MB total)")
    
    def _rasterize_obstacles(self, board_pads: List[Pad]):
        """Rasterize footprint obstacles for DRC-aware routing"""
        obstacle_count = 0
        
        # Create DRC mask for each pad/footprint
        for pad in board_pads:
            obstacle_count += self._rasterize_pad_footprint(pad)
        
        logger.info(f"Rasterized {obstacle_count} obstacle cells with DRC awareness")
    
    def _rasterize_pad_footprint(self, pad: Pad) -> int:
        """Rasterize a single pad footprint with proper DRC handling"""
        cells_marked = 0
        
        # Get pad grid position
        center_gx, center_gy = self.world_to_grid(pad.x_mm, pad.y_mm)
        
        # Calculate pad size in grid cells (with DRC clearance)
        clearance_cells = int(np.ceil(self.config.clearance / self.config.grid_pitch))
        width_cells = max(1, int(np.ceil(pad.width_mm / self.config.grid_pitch))) + clearance_cells
        height_cells = max(1, int(np.ceil(pad.height_mm / self.config.grid_pitch))) + clearance_cells
        
        # Get net ID for this pad
        net_id = hash(pad.net_name) % 32767 + 1  # Ensure positive net ID
        
        # Rasterize pad shape with DRC awareness
        for dy in range(-height_cells//2, height_cells//2 + 1):
            for dx in range(-width_cells//2, width_cells//2 + 1):
                gx, gy = center_gx + dx, center_gy + dy
                
                if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                    # Mark on all layers for through-hole pads, or appropriate layers for SMD
                    target_layers = self._get_pad_layers(pad)
                    
                    for layer_idx in target_layers:
                        if 0 <= layer_idx < self.num_layers:
                            # Only mark as obstacle if it's not the same net
                            current_net = self.net_id_grid[layer_idx, gy, gx]
                            if current_net == 0:  # Empty cell
                                self.occupancy[layer_idx, gy, gx] = PAD
                                self.net_id_grid[layer_idx, gy, gx] = net_id
                                cells_marked += 1
                            elif current_net != net_id:  # Different net - mark as keepout
                                self.occupancy[layer_idx, gy, gx] = KEEPOUT
        
        return cells_marked
    
    def _get_pad_layers(self, pad: Pad) -> List[int]:
        """Get layer indices that this pad affects for DRC"""
        if pad.is_through_hole:
            # Through-hole pads affect all layers
            return list(range(self.num_layers))
        else:
            # SMD pads on F.Cu don't directly affect internal routing layers
            # but we need to mark them for escape routing planning
            return []  # Internal layers not directly affected by F.Cu SMD pads
    
    def world_to_grid(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        gx = int((x_mm - self.min_x) / self.config.grid_pitch)
        gy = int((y_mm - self.min_y) / self.config.grid_pitch)
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x_mm = self.min_x + gx * self.config.grid_pitch
        y_mm = self.min_y + gy * self.config.grid_pitch
        return x_mm, y_mm
    
    def is_cell_free(self, layer: int, y: int, x: int, net_id: int) -> bool:
        """Check if a grid cell is free for routing"""
        if not (0 <= layer < self.num_layers and 0 <= y < self.grid_height and 0 <= x < self.grid_width):
            return False
        
        cell_occ = self.occupancy[layer, y, x]
        cell_net = self.net_id_grid[layer, y, x]
        
        # Cell is free if it's empty OR already used by same net
        return cell_occ == FREE or (cell_occ == TRACE and cell_net == net_id)
    
    def mark_cell_used(self, layer: int, y: int, x: int, net_id: int):
        """Mark a grid cell as used by a net"""
        if 0 <= layer < self.num_layers and 0 <= y < self.grid_height and 0 <= x < self.grid_width:
            self.occupancy[layer, y, x] = TRACE
            self.net_id_grid[layer, y, x] = net_id
    
    def clear_net_from_grid(self, net_id: int) -> int:
        """Clear all traces of a net from the grid (for ripup)"""
        mask = (self.net_id_grid == net_id)
        cells_cleared = np.sum(mask)
        
        self.occupancy[mask] = FREE
        self.net_id_grid[mask] = 0
        
        return int(cells_cleared)
    
    def get_layer_direction(self, layer_idx: int) -> str:
        """Get preferred routing direction for layer"""
        return self.layer_directions.get(layer_idx, 'H')