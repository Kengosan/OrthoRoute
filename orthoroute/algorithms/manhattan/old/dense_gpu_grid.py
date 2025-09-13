"""
GPU-Native Dense Grid Router - Eliminates Python object overhead
Builds everything directly on GPU as arrays, no Python objects
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CuPy available - GPU routing enabled")
except ImportError:
    import numpy as cp  # Fallback
    GPU_AVAILABLE = False
    logger.warning("CuPy not available - CPU fallback only")

@dataclass
class DenseGridConfig:
    """Configuration for dense GPU grid"""
    pitch: float = 0.025  # Grid resolution in mm
    max_memory_gb: float = 16.0  # Maximum GPU memory to use
    layers: int = 6  # Number of copper layers (will be updated from board data)
    
    # Routing costs
    via_cost: float = 2.0
    track_cost: float = 1.0
    congestion_penalty: float = 5.0

class DenseGPURouter:
    """GPU-native dense grid router - no Python objects"""
    
    def __init__(self, config: DenseGridConfig):
        self.config = config
        self.use_gpu = GPU_AVAILABLE
        
        # Grid dimensions
        self.cols = 0
        self.rows = 0
        self.layers = config.layers
        
        # World coordinate mapping
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        
        # GPU arrays - the heart of the system
        self.obstacle_grid = None      # bool[layer,y,x] - blocked cells
        self.cost_grid = None          # float32[layer,y,x] - routing costs
        self.distance_grid = None      # int32[layer,y,x] - pathfinding distances
        self.parent_grid = None        # int32[layer,y,x] - backtrack pointers
        
        # Pad mapping (minimal Python structures)
        self.pad_positions = {}        # pad_name -> (layer, y, x)
        self.grid_to_pad = {}          # (layer, y, x) -> pad_name
        
        logger.info(f"Dense GPU router initialized (GPU: {self.use_gpu})")
        
    def build_grid(self, board_bounds: Tuple[float, float, float, float], 
                   pads: List[Dict], nets: List[Dict]) -> bool:
        """Build dense GPU grid from board data"""
        
        self.min_x, self.min_y, self.max_x, self.max_y = board_bounds
        board_width = self.max_x - self.min_x
        board_height = self.max_y - self.min_y
        
        logger.info(f"Building dense GPU grid for {board_width:.1f}x{board_height:.1f}mm board")
        
        # Calculate grid dimensions
        self.cols = int(board_width / self.config.pitch) + 1
        self.rows = int(board_height / self.config.pitch) + 1
        total_cells = self.cols * self.rows * self.layers
        
        # Check memory requirements
        memory_gb = (total_cells * 13) / (1024**3)  # 13 bytes per cell
        logger.info(f"Dense grid: {self.cols}x{self.rows}x{self.layers} = {total_cells:,} cells")
        logger.info(f"GPU memory required: {memory_gb:.1f}GB")
        
        if memory_gb > self.config.max_memory_gb:
            logger.error(f"Grid too large: {memory_gb:.1f}GB > {self.config.max_memory_gb}GB limit")
            return False
            
        # Create GPU arrays
        if not self._create_gpu_arrays():
            return False
            
        # Populate grid from board data
        self._populate_pads(pads)
        self._initialize_costs()
        
        logger.info("Dense GPU grid built successfully")
        return True
        
    def _create_gpu_arrays(self) -> bool:
        """Create core GPU arrays"""
        shape = (self.layers, self.rows, self.cols)
        
        try:
            if self.use_gpu:
                # Check available GPU memory
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                required_bytes = np.prod(shape) * 13
                
                if required_bytes > free_mem * 0.8:
                    logger.error(f"Insufficient GPU memory: need {required_bytes/(1024**3):.1f}GB, have {free_mem/(1024**3):.1f}GB")
                    self.use_gpu = False
                    return self._create_cpu_arrays(shape)
                
                logger.info(f"Allocating {required_bytes/(1024**3):.1f}GB on GPU...")
                
                # Create GPU arrays
                self.obstacle_grid = cp.zeros(shape, dtype=cp.bool_)
                self.cost_grid = cp.ones(shape, dtype=cp.float32)
                self.distance_grid = cp.full(shape, -1, dtype=cp.int32)
                self.parent_grid = cp.full(shape, -1, dtype=cp.int32)
                
                logger.info("GPU arrays allocated successfully")
                return True
                
            else:
                return self._create_cpu_arrays(shape)
                
        except Exception as e:
            logger.error(f"GPU array creation failed: {e}")
            logger.warning("Falling back to CPU arrays")
            self.use_gpu = False
            return self._create_cpu_arrays(shape)
    
    def _create_cpu_arrays(self, shape) -> bool:
        """Fallback CPU arrays"""
        try:
            logger.info(f"Creating CPU arrays: {np.prod(shape) * 13 / (1024**3):.1f}GB")
            
            self.obstacle_grid = np.zeros(shape, dtype=np.bool_)
            self.cost_grid = np.ones(shape, dtype=np.float32)
            self.distance_grid = np.full(shape, -1, dtype=np.int32)
            self.parent_grid = np.full(shape, -1, dtype=np.int32)
            
            logger.info("CPU arrays created successfully")
            return True
            
        except MemoryError as e:
            logger.error(f"CPU array creation failed: {e}")
            return False
    
    def _populate_pads(self, pads: List[Dict]):
        """Map pads to grid coordinates"""
        logger.info(f"Mapping {len(pads)} pads to dense grid...")
        
        mapped_count = 0
        for pad in pads:
            try:
                # Convert world coordinates to grid coordinates
                grid_x = int((pad['x'] - self.min_x) / self.config.pitch)
                grid_y = int((pad['y'] - self.min_y) / self.config.pitch)
                
                # Clamp to grid bounds
                grid_x = max(0, min(grid_x, self.cols - 1))
                grid_y = max(0, min(grid_y, self.rows - 1))
                
                # Create unique pad identifier based on position and net
                net_name = pad.get('net', f"pad_{mapped_count}")
                pad_name = pad.get('name', f"{net_name}_pad_{mapped_count}")
                
                # Create unique identifier: net_name + position
                unique_pad_id = f"{net_name}@{pad['x']:.3f},{pad['y']:.3f}"
                
                # Also support various naming conventions
                pad_identifiers = [
                    unique_pad_id,           # Unique position-based ID
                    net_name,                # Net name (for single-pad nets)
                    pad_name,                # Pad name if available
                    f"{net_name}_src",       # Source naming
                    f"{net_name}_sink",      # Sink naming
                    f"pad_{mapped_count}",   # Fallback
                ]
                
                for layer in range(self.layers):
                    grid_pos = (layer, grid_y, grid_x)
                    
                    # Store all naming variants for this position
                    for identifier in pad_identifiers:
                        self.pad_positions[identifier] = grid_pos
                    
                    # Reverse lookup uses unique ID to avoid conflicts
                    self.grid_to_pad[grid_pos] = unique_pad_id
                    
                    # Mark as routable (not an obstacle)
                    self.obstacle_grid[layer, grid_y, grid_x] = False
                    
                mapped_count += 1
                
                if mapped_count % 1000 == 0:
                    logger.debug(f"Mapped {mapped_count} pads...")
                
            except Exception as e:
                logger.warning(f"Failed to map pad {pad}: {e}")
        
        logger.info(f"Mapped {mapped_count} pads to dense grid")
        
    def _initialize_costs(self):
        """Initialize routing costs"""
        # Base cost is 1.0 everywhere (already set in array creation)
        
        # Add layer change costs for vias
        if self.use_gpu:
            # GPU parallel cost initialization
            pass  # Costs already initialized to 1.0
        else:
            # CPU cost initialization  
            pass  # Costs already initialized to 1.0
            
        logger.info("Routing costs initialized")
    
    def route_net(self, source_pad: str, sink_pad: str, net_id: str) -> Optional[List[Tuple[int, int, int]]]:
        """Route single net using GPU pathfinding"""
        
        if source_pad not in self.pad_positions:
            logger.error(f"Source pad {source_pad} not found")
            return None
            
        if sink_pad not in self.pad_positions:
            logger.error(f"Sink pad {sink_pad} not found")
            return None
            
        source_pos = self.pad_positions[source_pad]
        sink_pos = self.pad_positions[sink_pad]
        
        logger.info(f"GPU routing {net_id}: {source_pad}@{source_pos} -> {sink_pad}@{sink_pos}")
        
        # Reset distance grids
        self.distance_grid.fill(-1)
        self.parent_grid.fill(-1)
        
        # Run GPU wavefront pathfinding
        path = self._gpu_wavefront_search(source_pos, sink_pos)
        
        if path:
            logger.info(f"Route SUCCESS: {net_id} - {len(path)} cells")
            # Mark path as used (increase costs for congestion)
            self._mark_path_used(path)
            return path
        else:
            logger.warning(f"Route FAILED: {net_id}")
            return None
    
    def _gpu_wavefront_search(self, source: Tuple[int, int, int], 
                             sink: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """GPU-accelerated wavefront pathfinding"""
        
        source_layer, source_y, source_x = source
        sink_layer, sink_y, sink_x = sink
        
        # Check if source and sink are reachable
        if self.obstacle_grid[source_layer, source_y, source_x]:
            logger.error(f"Source {source} is blocked")
            return None
            
        if self.obstacle_grid[sink_layer, sink_y, sink_x]:
            logger.error(f"Sink {sink} is blocked")
            return None
        
        # Initialize wavefront
        self.distance_grid[source_layer, source_y, source_x] = 0
        current_distance = 0
        max_iterations = 10000
        
        logger.debug(f"Starting wavefront: {source} -> {sink}")
        
        while current_distance < max_iterations:
            # Find all cells at current distance
            if self.use_gpu:
                current_cells = (self.distance_grid == current_distance)
                if not cp.any(current_cells):
                    break
            else:
                current_cells = (self.distance_grid == current_distance)
                if not np.any(current_cells):
                    break
            
            # Check if we reached the sink
            if self.distance_grid[sink_layer, sink_y, sink_x] >= 0:
                logger.debug(f"Wavefront reached sink in {current_distance} iterations")
                return self._trace_path(sink)
            
            # Expand wavefront
            self._expand_wavefront(current_cells, current_distance + 1)
            current_distance += 1
            
            if current_distance % 1000 == 0:
                logger.debug(f"Wavefront iteration {current_distance}")
        
        logger.warning(f"Wavefront search failed after {current_distance} iterations")
        return None
    
    def _expand_wavefront(self, current_cells, new_distance: int):
        """Expand wavefront to neighboring cells"""
        
        if self.use_gpu:
            # Get coordinates of current cells
            coords = cp.where(current_cells)
            layers_idx = coords[0]
            rows_idx = coords[1] 
            cols_idx = coords[2]
            
            # Convert to CPU for iteration (optimization: could be done on GPU)
            layers_cpu = cp.asnumpy(layers_idx)
            rows_cpu = cp.asnumpy(rows_idx)
            cols_cpu = cp.asnumpy(cols_idx)
        else:
            coords = np.where(current_cells)
            layers_cpu = coords[0]
            rows_cpu = coords[1]
            cols_cpu = coords[2]
        
        # Expand to neighbors
        directions = [
            (0, -1, 0),   # North
            (0, 1, 0),    # South  
            (0, 0, -1),   # West
            (0, 0, 1),    # East
            (1, 0, 0),    # Layer up (via)
            (-1, 0, 0),   # Layer down (via)
        ]
        
        expanded_count = 0
        for i in range(len(layers_cpu)):
            curr_layer, curr_row, curr_col = layers_cpu[i], rows_cpu[i], cols_cpu[i]
            
            for dl, dr, dc in directions:
                new_layer = curr_layer + dl
                new_row = curr_row + dr
                new_col = curr_col + dc
                
                # Check bounds
                if (0 <= new_layer < self.layers and
                    0 <= new_row < self.rows and
                    0 <= new_col < self.cols):
                    
                    # Check if unvisited and not blocked
                    if (self.distance_grid[new_layer, new_row, new_col] == -1 and
                        not self.obstacle_grid[new_layer, new_row, new_col]):
                        
                        # Calculate cost (add via cost for layer changes)
                        cost = self.cost_grid[new_layer, new_row, new_col]
                        if dl != 0:  # Layer change = via
                            cost += self.config.via_cost
                        
                        # Mark as reachable
                        self.distance_grid[new_layer, new_row, new_col] = new_distance
                        
                        # Set parent for path reconstruction
                        parent_encoded = curr_layer * 1000000 + curr_row * 1000 + curr_col
                        self.parent_grid[new_layer, new_row, new_col] = parent_encoded
                        
                        expanded_count += 1
        
        if expanded_count > 0:
            logger.debug(f"Expanded {expanded_count} cells to distance {new_distance}")
    
    def _trace_path(self, sink: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Trace path from sink back to source"""
        path = []
        current = sink
        
        while True:
            path.append(current)
            layer, row, col = current
            
            # Get parent
            parent_encoded = int(self.parent_grid[layer, row, col])
            if parent_encoded == -1:
                break  # Reached source
                
            # Decode parent coordinates
            parent_layer = parent_encoded // 1000000
            parent_row = (parent_encoded % 1000000) // 1000
            parent_col = parent_encoded % 1000
            
            current = (parent_layer, parent_row, parent_col)
            
            if len(path) > 100000:  # Safety limit
                logger.warning("Path trace exceeded limit")
                break
        
        path.reverse()  # Source to sink order
        return path
    
    def _mark_path_used(self, path: List[Tuple[int, int, int]]):
        """Mark path as used to create congestion costs"""
        for layer, row, col in path:
            # Increase cost for future routing (congestion avoidance)
            self.cost_grid[layer, row, col] += self.config.congestion_penalty
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU/CPU memory usage"""
        if self.use_gpu:
            try:
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                used_mem = total_mem - free_mem
                return {
                    'total_gb': total_mem / (1024**3),
                    'used_gb': used_mem / (1024**3),
                    'free_gb': free_mem / (1024**3),
                    'grid_gb': (self.cols * self.rows * self.layers * 13) / (1024**3)
                }
            except:
                return {'error': 'Cannot get GPU memory info'}
        else:
            import psutil
            ram = psutil.virtual_memory()
            return {
                'total_gb': ram.total / (1024**3),
                'used_gb': ram.used / (1024**3), 
                'free_gb': ram.available / (1024**3),
                'grid_gb': (self.cols * self.rows * self.layers * 13) / (1024**3)
            }
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.use_gpu:
            try:
                del self.obstacle_grid
                del self.cost_grid
                del self.distance_grid
                del self.parent_grid
                
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("GPU memory cleaned up")
            except:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()