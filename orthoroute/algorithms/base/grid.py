"""Base routing grid implementation."""
import logging
import math
from typing import List, Tuple, Optional, Set
from enum import Enum
import numpy as np

from ...domain.models.board import Board, Bounds, Coordinate
from ...application.interfaces.gpu_provider import GPUProvider

logger = logging.getLogger(__name__)


class CellState(Enum):
    """States of grid cells."""
    EMPTY = 0
    OBSTACLE = 1
    ROUTED = 2
    BLOCKED = 3


class GridCell:
    """Individual grid cell with state and metadata."""
    
    def __init__(self, x: int, y: int, layer: int):
        """Initialize grid cell."""
        self.x = x
        self.y = y
        self.layer = layer
        self.state = CellState.EMPTY
        self.net_id: Optional[str] = None
        self.cost_penalty = 0.0
        self.accessibility_mask: Set[str] = set()  # Net IDs that can access this cell
    
    def is_accessible_by_net(self, net_id: str) -> bool:
        """Check if cell is accessible by given net."""
        if self.state == CellState.EMPTY:
            return True
        elif self.state == CellState.ROUTED:
            return self.net_id == net_id
        elif self.state == CellState.OBSTACLE:
            return net_id in self.accessibility_mask
        else:
            return False
    
    def set_routed(self, net_id: str):
        """Mark cell as routed by net."""
        self.state = CellState.ROUTED
        self.net_id = net_id
    
    def set_obstacle(self, accessible_nets: Optional[Set[str]] = None):
        """Mark cell as obstacle with optional accessibility."""
        self.state = CellState.OBSTACLE
        if accessible_nets:
            self.accessibility_mask = accessible_nets
    
    def clear(self):
        """Clear cell to empty state."""
        self.state = CellState.EMPTY
        self.net_id = None
        self.cost_penalty = 0.0
        self.accessibility_mask.clear()


class RoutingGrid:
    """3D routing grid with GPU/CPU abstraction."""
    
    def __init__(self, bounds: Bounds, layer_names: List[str], 
                 resolution: float, gpu_provider: Optional[GPUProvider] = None):
        """Initialize routing grid."""
        self.bounds = bounds
        self.layer_names = layer_names
        self.layer_count = len(layer_names)
        self.resolution = resolution
        self.gpu_provider = gpu_provider
        
        # Calculate grid dimensions
        self.width = int(math.ceil(bounds.width / resolution))
        self.height = int(math.ceil(bounds.height / resolution))
        
        # Layer mapping
        self.layer_name_to_index = {name: i for i, name in enumerate(layer_names)}
        self.layer_index_to_name = {i: name for i, name in enumerate(layer_names)}
        
        # Initialize grid data structures
        if gpu_provider and gpu_provider.is_available():
            self._init_gpu_arrays()
        else:
            self._init_cpu_arrays()
        
        # Track grid cells for detailed operations
        self._detailed_cells = {}  # (x, y, layer) -> GridCell
        
        logger.info(f"Initialized routing grid: {self.width}x{self.height}x{self.layer_count}")
        logger.info(f"Grid bounds: ({bounds.min_x:.2f}, {bounds.min_y:.2f}) to ({bounds.max_x:.2f}, {bounds.max_y:.2f})")
        logger.info(f"Resolution: {resolution}mm, Layers: {layer_names}")
    
    def _init_gpu_arrays(self):
        """Initialize GPU arrays."""
        try:
            self.state_array = self.gpu_provider.create_array(
                (self.layer_count, self.height, self.width), 
                dtype=np.int32, 
                fill_value=0
            )
            self.net_array = self.gpu_provider.create_array(
                (self.layer_count, self.height, self.width),
                dtype=np.int32,
                fill_value=0
            )
            self.cost_array = self.gpu_provider.create_array(
                (self.layer_count, self.height, self.width),
                dtype=np.float32,
                fill_value=1.0
            )
            
            self.using_gpu = True
            logger.info("Grid initialized with GPU arrays")
            
        except Exception as e:
            logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
            self._init_cpu_arrays()
    
    def _init_cpu_arrays(self):
        """Initialize CPU arrays."""
        self.state_array = np.zeros((self.layer_count, self.height, self.width), dtype=np.int32)
        self.net_array = np.zeros((self.layer_count, self.height, self.width), dtype=np.int32)
        self.cost_array = np.ones((self.layer_count, self.height, self.width), dtype=np.float32)
        
        self.using_gpu = False
        logger.info("Grid initialized with CPU arrays")
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int((x - self.bounds.min_x) / self.resolution)
        grid_y = int((y - self.bounds.min_y) / self.resolution)
        
        # Clamp to valid range
        grid_x = max(0, min(grid_x, self.width - 1))
        grid_y = max(0, min(grid_y, self.height - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Coordinate:
        """Convert grid coordinates to world coordinates (center of cell)."""
        x = self.bounds.min_x + (grid_x + 0.5) * self.resolution
        y = self.bounds.min_y + (grid_y + 0.5) * self.resolution
        return Coordinate(x, y)
    
    def is_valid_position(self, x: int, y: int, layer: int) -> bool:
        """Check if grid position is valid."""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                0 <= layer < self.layer_count)
    
    def get_layer_index(self, layer_name: str) -> int:
        """Get layer index from name."""
        if layer_name not in self.layer_name_to_index:
            raise ValueError(f"Unknown layer: {layer_name}")
        return self.layer_name_to_index[layer_name]
    
    def get_layer_name(self, layer_index: int) -> str:
        """Get layer name from index."""
        if layer_index not in self.layer_index_to_name:
            raise ValueError(f"Invalid layer index: {layer_index}")
        return self.layer_index_to_name[layer_index]
    
    def get_cell_state(self, x: int, y: int, layer: int) -> CellState:
        """Get state of grid cell."""
        if not self.is_valid_position(x, y, layer):
            return CellState.BLOCKED
        
        if self.using_gpu:
            state_array = self.gpu_provider.to_cpu(self.state_array)
            state_value = state_array[layer, y, x]
        else:
            state_value = self.state_array[layer, y, x]
        
        return CellState(state_value)
    
    def set_cell_state(self, x: int, y: int, layer: int, state: CellState, net_id: Optional[str] = None):
        """Set state of grid cell."""
        if not self.is_valid_position(x, y, layer):
            return
        
        net_hash = hash(net_id) if net_id else 0
        
        if self.using_gpu:
            # For GPU arrays, we need to update the arrays on GPU
            # This is simplified - a real implementation would use GPU kernels
            state_array = self.gpu_provider.to_cpu(self.state_array)
            net_array = self.gpu_provider.to_cpu(self.net_array)
            
            state_array[layer, y, x] = state.value
            net_array[layer, y, x] = net_hash
            
            self.state_array = self.gpu_provider.to_gpu(state_array)
            self.net_array = self.gpu_provider.to_gpu(net_array)
        else:
            self.state_array[layer, y, x] = state.value
            self.net_array[layer, y, x] = net_hash
    
    def get_cell(self, x: int, y: int, layer: int) -> GridCell:
        """Get detailed grid cell object."""
        if not self.is_valid_position(x, y, layer):
            # Return dummy blocked cell
            cell = GridCell(x, y, layer)
            cell.state = CellState.BLOCKED
            return cell
        
        cell_key = (x, y, layer)
        if cell_key not in self._detailed_cells:
            # Create cell from array data
            cell = GridCell(x, y, layer)
            cell.state = self.get_cell_state(x, y, layer)
            
            # Get net ID from array
            if self.using_gpu:
                net_array = self.gpu_provider.to_cpu(self.net_array)
                net_hash = net_array[layer, y, x]
            else:
                net_hash = self.net_array[layer, y, x]
            
            # Note: We can't reverse the hash to get net_id easily
            # In a real implementation, we'd maintain a separate mapping
            cell.net_id = str(net_hash) if net_hash != 0 else None
            
            self._detailed_cells[cell_key] = cell
        
        return self._detailed_cells[cell_key]
    
    def mark_obstacles_from_board(self, board: Board):
        """Mark obstacles based on board footprints and pads with proper DRC clearances."""
        logger.info("Marking DRC-compliant obstacles from board components")
        
        obstacle_count = 0
        layer_obstacle_counts = {i: 0 for i in range(len(self.layer_names))}
        
        # Get DRC constraints from board or use defaults
        drc_constraints = getattr(board, 'drc_constraints', None)
        if not drc_constraints:
            from ...domain.models.constraints import DRCConstraints
            drc_constraints = DRCConstraints()
            logger.warning("No DRC constraints found on board, using defaults")
        
        # Calculate DRC clearance distances
        default_clearance = drc_constraints.default_clearance  # 0.2mm default
        track_to_pad_clearance = drc_constraints.get_clearance_for_nets('Default', 'Default')
        
        logger.info(f"Using DRC clearances: default={default_clearance:.3f}mm, track-to-pad={track_to_pad_clearance:.3f}mm")
        
        for component in board.components:
            for pad in component.pads:
                # Convert pad position to grid
                grid_x, grid_y = self.world_to_grid(pad.position.x, pad.position.y)
                
                # Process pad on all relevant layers
                pad_layers = self._get_pad_layers(pad)
                
                for layer_name in pad_layers:
                    try:
                        layer_index = self.get_layer_index(layer_name)
                    except ValueError:
                        continue  # Skip unknown layers
                    
                    # Calculate DRC-compliant keepout area
                    pad_radius = max(pad.size[0], pad.size[1]) / 2.0
                    
                    # Get netclass for this pad's net
                    net_name = pad.net_id or 'Default'
                    netclass = drc_constraints.get_netclass(net_name)
                    
                    # Use the larger of: default clearance or netclass clearance
                    required_clearance = max(default_clearance, netclass.clearance)
                    
                    # Total keepout radius = pad radius + DRC clearance
                    keepout_radius = pad_radius + required_clearance
                    keepout_radius_cells = int(math.ceil(keepout_radius / self.resolution))
                    
                    # Mark circular keepout area
                    for dx in range(-keepout_radius_cells, keepout_radius_cells + 1):
                        for dy in range(-keepout_radius_cells, keepout_radius_cells + 1):
                            obs_x, obs_y = grid_x + dx, grid_y + dy
                            
                            if self.is_valid_position(obs_x, obs_y, layer_index):
                                # Check if point is within circular keepout
                                distance = math.sqrt(dx*dx + dy*dy) * self.resolution
                                
                                if distance <= keepout_radius:
                                    # Allow the pad's own net to access connection point
                                    accessible_nets = {pad.net_id} if pad.net_id else set()
                                    
                                    # For the pad center area, allow direct connection
                                    if distance <= pad_radius:
                                        # This is the pad itself - allow connection but mark as special
                                        cell = self.get_cell(obs_x, obs_y, layer_index)
                                        if cell.state == CellState.EMPTY:
                                            cell.set_obstacle(accessible_nets)
                                            # Don't change to OBSTACLE for pad center - allow routing to it
                                    else:
                                        # This is keepout area - block all routing
                                        cell = self.get_cell(obs_x, obs_y, layer_index)
                                        if cell.state == CellState.EMPTY:
                                            cell.set_obstacle(set())  # No nets can access keepout
                                            self.set_cell_state(obs_x, obs_y, layer_index, CellState.OBSTACLE)
                                            obstacle_count += 1
                                            layer_obstacle_counts[layer_index] += 1
        
        # Log per-layer obstacle counts
        for layer_idx, count in layer_obstacle_counts.items():
            layer_name = self.layer_names[layer_idx]
            logger.info(f"  {layer_name}: {count} DRC obstacle cells")
        
        logger.info(f"Marked {obstacle_count} DRC-compliant obstacle cells from board components")
    
    def _get_pad_layers(self, pad) -> List[str]:
        """Get list of layers this pad affects for routing."""
        layers = []
        
        # Add the pad's primary layer
        if hasattr(pad, 'layer') and pad.layer:
            layers.append(pad.layer)
        
        # For through-hole pads, affect all layers
        if hasattr(pad, 'drill') and pad.drill and pad.drill > 0:
            layers.extend(self.layer_names)
        
        # For pads without specific layer info, assume they affect F.Cu
        if not layers:
            layers.append('F.Cu')
        
        # Remove duplicates and filter to known layers
        valid_layers = []
        for layer in set(layers):
            if layer in self.layer_names:
                valid_layers.append(layer)
        
        return valid_layers
    
    def clear_net_routes(self, net_id: str):
        """Clear all routes for a specific net."""
        net_hash = hash(net_id)
        cleared_count = 0
        
        if self.using_gpu:
            # For GPU, we'd use a kernel to clear efficiently
            # This is a simplified CPU-based approach
            state_array = self.gpu_provider.to_cpu(self.state_array)
            net_array = self.gpu_provider.to_cpu(self.net_array)
            
            mask = (net_array == net_hash) & (state_array == CellState.ROUTED.value)
            state_array[mask] = CellState.EMPTY.value
            net_array[mask] = 0
            cleared_count = np.sum(mask)
            
            self.state_array = self.gpu_provider.to_gpu(state_array)
            self.net_array = self.gpu_provider.to_gpu(net_array)
        else:
            mask = (self.net_array == net_hash) & (self.state_array == CellState.ROUTED.value)
            self.state_array[mask] = CellState.EMPTY.value
            self.net_array[mask] = 0
            cleared_count = np.sum(mask)
        
        # Clear detailed cells
        to_remove = []
        for cell_key, cell in self._detailed_cells.items():
            if cell.net_id == net_id and cell.state == CellState.ROUTED:
                cell.clear()
                to_remove.append(cell_key)
        
        for key in to_remove:
            del self._detailed_cells[key]
        
        logger.debug(f"Cleared {cleared_count} cells for net {net_id}")
    
    def clear_all_routes(self):
        """Clear all routed cells."""
        if self.using_gpu:
            state_array = self.gpu_provider.to_cpu(self.state_array)
            net_array = self.gpu_provider.to_cpu(self.net_array)
            
            route_mask = (state_array == CellState.ROUTED.value)
            state_array[route_mask] = CellState.EMPTY.value
            net_array[route_mask] = 0
            
            self.state_array = self.gpu_provider.to_gpu(state_array)
            self.net_array = self.gpu_provider.to_gpu(net_array)
        else:
            route_mask = (self.state_array == CellState.ROUTED.value)
            self.state_array[route_mask] = CellState.EMPTY.value
            self.net_array[route_mask] = 0
        
        # Clear detailed cells
        for cell in self._detailed_cells.values():
            if cell.state == CellState.ROUTED:
                cell.clear()
        
        logger.info("Cleared all routes from grid")
    
    def get_memory_usage(self) -> dict:
        """Get memory usage information."""
        if self.using_gpu and self.gpu_provider:
            return self.gpu_provider.get_memory_info()
        else:
            # Calculate CPU memory usage
            array_memory = (self.state_array.nbytes + self.net_array.nbytes + self.cost_array.nbytes)
            cell_memory = len(self._detailed_cells) * 200  # Rough estimate
            
            return {
                'total_memory': array_memory + cell_memory,
                'array_memory': array_memory,
                'cell_memory': cell_memory,
                'using_gpu': False
            }
    
    def get_statistics(self) -> dict:
        """Get grid statistics."""
        if self.using_gpu:
            state_array = self.gpu_provider.to_cpu(self.state_array)
        else:
            state_array = self.state_array
        
        empty_cells = np.sum(state_array == CellState.EMPTY.value)
        obstacle_cells = np.sum(state_array == CellState.OBSTACLE.value)
        routed_cells = np.sum(state_array == CellState.ROUTED.value)
        total_cells = self.width * self.height * self.layer_count
        
        return {
            'dimensions': f"{self.width}x{self.height}x{self.layer_count}",
            'resolution': self.resolution,
            'total_cells': total_cells,
            'empty_cells': int(empty_cells),
            'obstacle_cells': int(obstacle_cells),
            'routed_cells': int(routed_cells),
            'utilization': routed_cells / total_cells * 100,
            'using_gpu': self.using_gpu,
            'detailed_cells': len(self._detailed_cells)
        }