"""
GPU-Accelerated Wavefront Pathfinder for RRG routing
Replaces slow A* with parallel flood-fill algorithm
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from .rrg import RouteRequest, RouteResult, RoutingResourceGraph

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # Fallback to numpy if CuPy not available
    GPU_AVAILABLE = False

@dataclass
class WavefrontGrid:
    """GPU-optimized wavefront grid for pathfinding"""
    # Grid dimensions
    width: int
    height: int
    layers: int
    
    # GPU arrays
    obstacle_grid: cp.ndarray  # 3D bool array [layer, y, x] - True = blocked
    cost_grid: cp.ndarray      # 3D float array [layer, y, x] - routing costs
    distance_grid: cp.ndarray  # 3D int array [layer, y, x] - wavefront distances
    parent_grid: cp.ndarray    # 3D int array [layer, y, x] - parent pointers
    
    # Node mapping
    grid_to_node: Dict[Tuple[int, int, int], str]  # (layer, y, x) -> node_id
    node_to_grid: Dict[str, Tuple[int, int, int]]  # node_id -> (layer, y, x)
    
    # Pre-computed RRG connectivity for fast access
    neighbors_cache: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]  # grid_pos -> [neighbor_grid_positions]

class GPUWavefrontPathfinder:
    """GPU-accelerated wavefront pathfinder for RRG routing"""
    
    def __init__(self, rrg: RoutingResourceGraph):
        self.rrg = rrg
        self.grid = None
        self.use_gpu = GPU_AVAILABLE
        
        if self.use_gpu:
            logger.info("Initializing GPU-accelerated wavefront pathfinder")
            # Set memory pool for better GPU memory management
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=8 * 1024**3)  # 8GB limit
        else:
            logger.warning("GPU not available, using CPU fallback")
        
    def build_grid(self):
        """Build GPU-optimized grid from RRG"""
        logger.info("Building wavefront grid from RRG...")
        start_time = time.time()
        
        # Analyze RRG structure to determine grid dimensions
        self._analyze_rrg_structure()
        
        # Create GPU arrays
        self._create_gpu_arrays()
        
        # Populate grid from RRG
        self._populate_grid_from_rrg()
        
        build_time = time.time() - start_time
        logger.info(f"Wavefront grid built in {build_time:.2f}s")
        logger.info(f"Grid size: {self.grid.layers}×{self.grid.height}×{self.grid.width}")
        logger.info(f"Memory usage: ~{self._estimate_gpu_memory():.1f}MB")
        
    def _analyze_rrg_structure(self):
        """Analyze RRG to determine optimal grid dimensions"""
        # Find bounding box of all nodes
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        max_layer = 0
        
        for node in self.rrg.nodes.values():
            min_x = min(min_x, node.x)
            max_x = max(max_x, node.x)
            min_y = min(min_y, node.y)  
            max_y = max(max_y, node.y)
            max_layer = max(max_layer, node.layer)
        
        # Calculate grid resolution (0.4mm pitch)
        self.grid_pitch = 0.4  # mm
        self.min_x = min_x
        self.min_y = min_y
        
        # Calculate grid dimensions
        width = int((max_x - min_x) / self.grid_pitch) + 1
        height = int((max_y - min_y) / self.grid_pitch) + 1
        layers = max_layer + 3  # Include F.Cu (-2) and extra layers
        
        logger.info(f"Grid analysis: {width}×{height}×{layers} cells")
        logger.info(f"Physical area: {max_x-min_x:.1f}×{max_y-min_y:.1f}mm")
        
        self.grid_width = width
        self.grid_height = height  
        self.grid_layers = layers
        
    def _create_gpu_arrays(self):
        """Create GPU arrays for wavefront computation"""
        shape = (self.grid_layers, self.grid_height, self.grid_width)
        
        if self.use_gpu:
            # Create CuPy arrays on GPU
            obstacle_grid = cp.zeros(shape, dtype=cp.bool_)
            cost_grid = cp.ones(shape, dtype=cp.float32)
            distance_grid = cp.full(shape, -1, dtype=cp.int32)
            parent_grid = cp.full(shape, -1, dtype=cp.int32)
        else:
            # Create NumPy arrays for CPU fallback
            obstacle_grid = np.zeros(shape, dtype=np.bool_)
            cost_grid = np.ones(shape, dtype=np.float32) 
            distance_grid = np.full(shape, -1, dtype=np.int32)
            parent_grid = np.full(shape, -1, dtype=np.int32)
        
        self.grid = WavefrontGrid(
            width=self.grid_width,
            height=self.grid_height, 
            layers=self.grid_layers,
            obstacle_grid=obstacle_grid,
            cost_grid=cost_grid,
            distance_grid=distance_grid,
            parent_grid=parent_grid,
            grid_to_node={},
            node_to_grid={},
            neighbors_cache={}
        )
        
    def _populate_grid_from_rrg(self):
        """Populate wavefront grid with RRG data"""
        logger.info("Populating grid with RRG nodes and edges...")
        
        # Initialize all cells as obstacles (only mapped nodes will be passable)
        self.grid.obstacle_grid.fill(True)
        
        # Map RRG nodes to grid coordinates
        for node_id, node in self.rrg.nodes.items():
            # Convert world coordinates to grid coordinates
            grid_x = int((node.x - self.min_x) / self.grid_pitch)
            grid_y = int((node.y - self.min_y) / self.grid_pitch)
            grid_layer = node.layer + 2  # Offset for F.Cu = -2
            
            # Clamp to grid bounds
            grid_x = max(0, min(grid_x, self.grid_width - 1))
            grid_y = max(0, min(grid_y, self.grid_height - 1))
            grid_layer = max(0, min(grid_layer, self.grid_layers - 1))
            
            # Create bidirectional mapping
            grid_pos = (grid_layer, grid_y, grid_x)
            self.grid.grid_to_node[grid_pos] = node_id
            self.grid.node_to_grid[node_id] = grid_pos
            
            # Mark this cell as passable (not an obstacle)
            self.grid.obstacle_grid[grid_layer, grid_y, grid_x] = False
            
            # Set base routing cost
            self.grid.cost_grid[grid_layer, grid_y, grid_x] = 1.0
        
        # Mark unavailable nodes as obstacles
        for node_id, node in self.rrg.nodes.items():
            if node_id in self.grid.node_to_grid:
                grid_pos = self.grid.node_to_grid[node_id]
                if not node.is_available():
                    self.grid.obstacle_grid[grid_pos] = True
        
        logger.info(f"Mapped {len(self.grid.node_to_grid)} RRG nodes to grid")
        total_passable = int(cp.sum(~self.grid.obstacle_grid))
        logger.info(f"Grid has {total_passable} passable cells out of {self.grid.obstacle_grid.size} total")
        
        # TEMPORARILY DISABLED: Pre-compute RRG connectivity for fast wavefront expansion
        # This takes too long with 2.48M nodes - disabling for testing
        logger.info("TESTING: Skipping RRG connectivity pre-computation for faster startup")
        # self._precompute_rrg_connectivity()
        # logger.info(f"Cached connectivity for {len(self.grid.neighbors_cache)} grid cells")
        
    def add_node_to_grid(self, node_id: str, node):
        """Add a dynamically created RRG node to the wavefront grid"""
        # Check if grid is fully initialized
        if not hasattr(self, 'min_x') or not hasattr(self, 'grid'):
            logger.warning(f"Wavefront grid not ready, cannot add node {node_id}")
            return None
            
        # Convert world coordinates to grid coordinates
        grid_x = int((node.x - self.min_x) / self.grid_pitch)
        grid_y = int((node.y - self.min_y) / self.grid_pitch)
        grid_layer = node.layer + 2  # Offset for F.Cu = -2
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        grid_layer = max(0, min(grid_layer, self.grid_layers - 1))
        
        # Add to grid mappings
        grid_pos = (grid_layer, grid_y, grid_x)
        self.grid.node_to_grid[node_id] = grid_pos
        self.grid.grid_to_node[grid_pos] = node_id
        
        # Mark as passable (not an obstacle)
        self.grid.obstacle_grid[grid_layer, grid_y, grid_x] = False
        
        logger.debug(f"Added node {node_id} to wavefront grid at {grid_pos}")
        return grid_pos
        
    def _precompute_rrg_connectivity(self):
        """Pre-compute RRG connectivity to avoid runtime lookups"""
        for grid_pos, node_id in self.grid.grid_to_node.items():
            neighbors_list = []
            
            # Get RRG neighbors for this node
            rrg_neighbors = self.rrg.get_neighbors(node_id)
            
            for neighbor_node_id, edge_id in rrg_neighbors:
                # Check if neighbor node is mapped to grid
                if neighbor_node_id in self.grid.node_to_grid:
                    neighbor_grid_pos = self.grid.node_to_grid[neighbor_node_id]
                    neighbors_list.append(neighbor_grid_pos)
            
            # Cache the neighbor list for this grid position
            self.grid.neighbors_cache[grid_pos] = neighbors_list
        
    def _estimate_gpu_memory(self) -> float:
        """Estimate GPU memory usage in MB"""
        total_cells = self.grid_layers * self.grid_height * self.grid_width
        
        # 4 arrays: obstacle(1 byte) + cost(4 bytes) + distance(4 bytes) + parent(4 bytes) 
        memory_bytes = total_cells * (1 + 4 + 4 + 4)
        return memory_bytes / (1024 * 1024)  # Convert to MB
        
    def route_single_net(self, request: RouteRequest) -> RouteResult:
        """Route single net using GPU wavefront"""
        if self.grid is None:
            self.build_grid()
            
        start_time = time.time()
        
        # Check if source and sink exist in grid
        if request.source_pad not in self.grid.node_to_grid:
            logger.error(f"Source {request.source_pad} not found in wavefront grid")
            return RouteResult(net_id=request.net_id, success=False)
            
        if request.sink_pad not in self.grid.node_to_grid:
            logger.error(f"Sink {request.sink_pad} not found in wavefront grid") 
            return RouteResult(net_id=request.net_id, success=False)
        
        # Get grid coordinates
        source_pos = self.grid.node_to_grid[request.source_pad]
        sink_pos = self.grid.node_to_grid[request.sink_pad]
        
        logger.info(f"Routing net {request.net_id}: {request.source_pad}@{source_pos} -> {request.sink_pad}@{sink_pos}")
        
        # Calculate Manhattan distance for comparison
        manhattan_dist = abs(source_pos[0] - sink_pos[0]) + abs(source_pos[1] - sink_pos[1]) + abs(source_pos[2] - sink_pos[2])
        logger.debug(f"Grid Manhattan distance: {manhattan_dist} cells")
        
        # Run GPU wavefront
        path = self._gpu_wavefront(source_pos, sink_pos)
        
        if path:
            # Convert grid path back to RRG nodes
            node_path = self._grid_path_to_nodes(path)
            
            if not node_path:
                route_time = time.time() - start_time
                logger.error(f"Wavefront found grid path but node conversion failed in {route_time:.3f}s")
                return RouteResult(net_id=request.net_id, success=False)
            
            # Calculate route metrics
            total_cost = len(path)  # Simple cost for now
            total_length = len(path) * self.grid_pitch
            via_count = self._count_vias(path)
            
            route_time = time.time() - start_time
            logger.info(f"Wavefront SUCCESS: net {request.net_id}, {len(node_path)} nodes in {route_time:.3f}s")
            
            return RouteResult(
                net_id=request.net_id,
                success=True,
                path=node_path,
                cost=total_cost,
                length_mm=total_length,
                via_count=via_count
            )
        else:
            route_time = time.time() - start_time
            logger.warning(f"Wavefront FAILED: net {request.net_id} in {route_time:.3f}s")
            return RouteResult(net_id=request.net_id, success=False)
    
    def _gpu_wavefront(self, source_pos: Tuple[int, int, int], 
                       sink_pos: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """GPU-accelerated wavefront pathfinding"""
        
        # Reset distance and parent grids
        self.grid.distance_grid.fill(-1)
        self.grid.parent_grid.fill(-1)
        
        # Check if source and sink are passable
        source_layer, source_y, source_x = source_pos
        sink_layer, sink_y, sink_x = sink_pos
        
        if self.grid.obstacle_grid[source_layer, source_y, source_x]:
            logger.error(f"Source position {source_pos} is blocked by obstacle")
            return None
            
        if self.grid.obstacle_grid[sink_layer, sink_y, sink_x]:
            logger.error(f"Sink position {sink_pos} is blocked by obstacle") 
            return None
        
        # Initialize source
        self.grid.distance_grid[source_layer, source_y, source_x] = 0
        
        # Wavefront expansion using GPU parallelization
        current_distance = 0
        max_iterations = 10000  # Increased for large boards
        
        for iteration in range(max_iterations):
            # Find all cells at current distance 
            current_cells = (self.grid.distance_grid == current_distance)
            num_current = int(cp.sum(current_cells))
            
            if num_current == 0:
                logger.debug(f"Wavefront stalled: no cells at distance {current_distance}")
                break  # No more cells to expand
                
            # Check if we reached the sink
            if self.grid.distance_grid[sink_pos] >= 0:
                logger.debug(f"Wavefront reached sink in {iteration} iterations")
                return self._trace_path(sink_pos)
            
            # Expand wavefront to neighbors (GPU parallel operation)
            self._expand_wavefront_gpu(current_cells, current_distance + 1)
            
            current_distance += 1
            
            # Periodic logging for long searches
            if iteration % 50 == 49:
                logger.info(f"Wavefront iteration {iteration}: {num_current} active cells at distance {current_distance}")
        
        logger.warning(f"Wavefront failed to reach sink after {max_iterations} iterations")
        return None
    
    def _expand_wavefront_gpu(self, current_cells: cp.ndarray, new_distance: int):
        """Expand wavefront using pre-computed RRG connectivity"""
        
        # Get coordinates of current cells (convert back to CPU for cache lookups)
        current_coords = cp.where(current_cells)
        current_z = cp.asnumpy(current_coords[0])
        current_y = cp.asnumpy(current_coords[1])
        current_x = cp.asnumpy(current_coords[2])
        
        # Process each active cell using cached connectivity
        for i in range(len(current_z)):
            cell_pos = (current_z[i], current_y[i], current_x[i])
            
            # Skip if no cached neighbors for this cell
            if cell_pos not in self.grid.neighbors_cache:
                continue
                
            # Get pre-computed neighbor positions
            neighbor_positions = self.grid.neighbors_cache[cell_pos]
            
            for neighbor_pos in neighbor_positions:
                neighbor_z, neighbor_y, neighbor_x = neighbor_pos
                
                # Check bounds and if unvisited
                if (0 <= neighbor_z < self.grid.layers and
                    0 <= neighbor_y < self.grid.height and
                    0 <= neighbor_x < self.grid.width and
                    self.grid.distance_grid[neighbor_z, neighbor_y, neighbor_x] == -1 and
                    not self.grid.obstacle_grid[neighbor_z, neighbor_y, neighbor_x]):
                    
                    # Set distance for reachable neighbor
                    self.grid.distance_grid[neighbor_z, neighbor_y, neighbor_x] = new_distance
                    
                    # Set parent pointer (encode parent coordinates)
                    parent_encoded = current_z[i] * 1000000 + current_y[i] * 1000 + current_x[i]
                    self.grid.parent_grid[neighbor_z, neighbor_y, neighbor_x] = parent_encoded
    
    def _trace_path(self, sink_pos: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Trace path from sink back to source using parent pointers"""
        path = []
        current_pos = sink_pos
        
        while True:
            path.append(current_pos)
            
            # Get parent
            layer, y, x = current_pos
            parent_encoded = int(self.grid.parent_grid[layer, y, x])
            
            if parent_encoded == -1:
                break  # Reached source
                
            # Decode parent coordinates
            parent_z = parent_encoded // 1000000
            parent_y = (parent_encoded % 1000000) // 1000
            parent_x = parent_encoded % 1000
            
            current_pos = (parent_z, parent_y, parent_x)
            
            # Sanity check to prevent infinite loops
            if len(path) > 10000:
                logger.warning("Path tracing exceeded maximum length")
                break
        
        # Reverse to get source -> sink order
        path.reverse()
        return path
    
    def _grid_path_to_nodes(self, grid_path: List[Tuple[int, int, int]]) -> List[str]:
        """Convert grid coordinates path to RRG node IDs"""
        node_path = []
        
        for i, grid_pos in enumerate(grid_path):
            if grid_pos in self.grid.grid_to_node:
                node_path.append(self.grid.grid_to_node[grid_pos])
            else:
                # Find nearest mapped node as fallback
                nearest_node = self._find_nearest_mapped_node(grid_pos)
                if nearest_node:
                    logger.debug(f"Grid position {grid_pos} mapped to nearest node {nearest_node}")
                    node_path.append(nearest_node)
                else:
                    logger.error(f"Grid position {grid_pos} has no nearby mapped nodes - path invalid")
                    return []  # Return empty path on failure
                
        return node_path
    
    def _find_nearest_mapped_node(self, grid_pos: Tuple[int, int, int]) -> Optional[str]:
        """Find nearest RRG node to unmapped grid position"""
        layer, y, x = grid_pos
        min_distance = float('inf')
        nearest_node = None
        
        # Search in expanding radius around the position
        for radius in range(1, 10):  # Search up to 10 cells away
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dy) == radius or abs(dx) == radius:  # Only check perimeter
                        check_y = y + dy
                        check_x = x + dx
                        
                        # Check bounds
                        if (0 <= check_y < self.grid.height and 
                            0 <= check_x < self.grid.width):
                            
                            check_pos = (layer, check_y, check_x)
                            if check_pos in self.grid.grid_to_node:
                                distance = abs(dy) + abs(dx)  # Manhattan distance
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_node = self.grid.grid_to_node[check_pos]
            
            if nearest_node:
                break  # Found a node at this radius, stop searching
                
        return nearest_node
    
    def _count_vias(self, grid_path: List[Tuple[int, int, int]]) -> int:
        """Count layer changes (vias) in path"""
        via_count = 0
        
        for i in range(1, len(grid_path)):
            prev_layer = grid_path[i-1][0]
            curr_layer = grid_path[i][0]
            
            if prev_layer != curr_layer:
                via_count += 1
                
        return via_count
    
    def update_costs(self, congestion_map: Dict[str, float]):
        """Update routing costs based on congestion"""
        if self.grid is None:
            return
            
        # Reset to base costs
        self.grid.cost_grid.fill(1.0)
        
        # Apply congestion penalties
        for node_id, congestion_cost in congestion_map.items():
            if node_id in self.grid.node_to_grid:
                grid_pos = self.grid.node_to_grid[node_id]
                self.grid.cost_grid[grid_pos] = congestion_cost
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if self.use_gpu and self.grid:
            # Free GPU arrays
            del self.grid.obstacle_grid
            del self.grid.cost_grid  
            del self.grid.distance_grid
            del self.grid.parent_grid
            
            # Clear memory pool
            cp.get_default_memory_pool().free_all_blocks()
            logger.info("GPU memory cleaned up")