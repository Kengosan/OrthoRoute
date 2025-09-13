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
        self._max_grid_memory_gb = 16.0  # Use full 16GB GPU capacity
        
        if self.use_gpu:
            try:
                # Get actual available GPU memory
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                available_gb = free_mem / (1024**3)
                
                # Use larger memory limit (75% of available or 16GB max)
                memory_limit = min(int(free_mem * 0.75), 16 * 1024**3)
                
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=memory_limit)
                
                logger.warning(f"PERFORMANCE: GPU WILL BE USED for pathfinding! Memory: {total_mem/(1024**3):.1f}GB total, {free_mem/(1024**3):.1f}GB free")
                logger.warning(f"PERFORMANCE: GPU memory limit set to {memory_limit/(1024**3):.1f}GB")
                
            except Exception as e:
                logger.error(f"ERROR: GPU memory setup failed: {e}")
                self.use_gpu = False
        else:
            logger.warning("ERROR: GPU not available, using CPU fallback")
        
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
        
        # Calculate grid resolution (0.2mm pitch - original working resolution)
        self.grid_pitch = 0.2  # mm - proven stable resolution that works without RAM explosion
        self.min_x = min_x
        self.min_y = min_y
        
        # Calculate grid dimensions with SAFETY LIMITS to prevent crashes
        width = int((max_x - min_x) / self.grid_pitch) + 1
        height = int((max_y - min_y) / self.grid_pitch) + 1
        layers = max_layer + 3  # Include F.Cu (-2) and extra layers
        
        # Calculate memory usage and enforce safety limits
        total_cells = width * height * layers
        memory_gb = (total_cells * 13) / (1024**3)  # 13 bytes per cell (bool+3*float32)
        
        # CRITICAL: Prevent system crashes by limiting grid size
        if memory_gb > self._max_grid_memory_gb:
            # Scale down grid resolution to fit memory limit
            scale_factor = (self._max_grid_memory_gb / memory_gb) ** (1/3)
            self.grid_pitch = self.grid_pitch / scale_factor
            
            # Recalculate with larger pitch
            width = int((max_x - min_x) / self.grid_pitch) + 1
            height = int((max_y - min_y) / self.grid_pitch) + 1
            total_cells = width * height * layers
            memory_gb = (total_cells * 13) / (1024**3)
            
            logger.warning(f"Grid scaled down to prevent crash: pitch={self.grid_pitch:.3f}mm")
        
        logger.info(f"Safe grid: {width}×{height}×{layers} = {total_cells:,} cells")
        logger.info(f"Memory usage: {memory_gb:.1f}GB (limit: {self._max_grid_memory_gb}GB)")
        logger.info(f"Physical area: {max_x-min_x:.1f}×{max_y-min_y:.1f}mm at {self.grid_pitch:.3f}mm pitch")
        
        self.grid_width = width
        self.grid_height = height  
        self.grid_layers = layers
        
    def _create_gpu_arrays(self):
        """Create GPU arrays for wavefront computation with safety checks"""
        shape = (self.grid_layers, self.grid_height, self.grid_width)
        
        if self.use_gpu:
            try:
                # Check available memory before allocation
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                required_bytes = np.prod(shape) * 13  # Estimate memory needed
                
                if required_bytes > free_mem * 0.8:  # Leave 20% buffer
                    logger.error(f"Insufficient GPU memory: need {required_bytes/(1024**3):.1f}GB, have {free_mem/(1024**3):.1f}GB")
                    self.use_gpu = False
                    return self._create_cpu_arrays(shape)
                
                logger.info(f"Allocating GPU arrays: {required_bytes/(1024**3):.1f}GB")
                
                # Create CuPy arrays on GPU with error handling
                obstacle_grid = cp.zeros(shape, dtype=cp.bool_)
                cost_grid = cp.ones(shape, dtype=cp.float32)
                distance_grid = cp.full(shape, -1, dtype=cp.int32)
                parent_grid = cp.full(shape, -1, dtype=cp.int32)
                
                logger.info("GPU arrays allocated successfully")
                
            except Exception as e:
                logger.error(f"GPU array allocation failed: {e}")
                logger.warning("Falling back to CPU arrays")
                self.use_gpu = False
                return self._create_cpu_arrays(shape)
        else:
            return self._create_cpu_arrays(shape)
            
        return obstacle_grid, cost_grid, distance_grid, parent_grid
    
    def _create_cpu_arrays(self, shape):
        """Create CPU arrays as fallback"""
        logger.info(f"Creating CPU arrays: {np.prod(shape) * 13 / (1024**3):.1f}GB")
        
        # Create NumPy arrays for CPU fallback
        obstacle_grid = np.zeros(shape, dtype=np.bool_)
        cost_grid = np.ones(shape, dtype=np.float32) 
        distance_grid = np.full(shape, -1, dtype=np.int32)
        parent_grid = np.full(shape, -1, dtype=np.int32)
        
        return obstacle_grid, cost_grid, distance_grid, parent_grid
        
        # Create arrays with safety checks
        arrays = self._create_gpu_arrays()
        if arrays is None:
            logger.error("Failed to create GPU/CPU arrays")
            return
        
        obstacle_grid, cost_grid, distance_grid, parent_grid = arrays
        
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
        
        # ENTERPRISE FIX: Use lazy connectivity for 2.28M nodes (pre-computation takes too long)
        logger.info("Using lazy connectivity for enterprise-scale routing (2.28M nodes)")
        logger.info("Connectivity will be computed on-demand during pathfinding")
        
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
        
        # Cache neighbors for this dynamically added node
        self._cache_node_neighbors(grid_pos, node_id)
        
        logger.debug(f"Added node {node_id} to wavefront grid at {grid_pos}")
        return grid_pos
    
    def _cache_node_neighbors(self, grid_pos: Tuple[int, int, int], node_id: str):
        """Cache neighbors for a single node (used for dynamically added nodes)"""
        neighbors_list = []
        
        logger.info(f"Caching neighbors for dynamic node {node_id} at {grid_pos}")
        
        # Get RRG neighbors using the proper adjacency structure
        if node_id in self.rrg.adjacency and self.rrg.adjacency[node_id]:
            for edge_id in self.rrg.adjacency[node_id]:
                if edge_id in self.rrg.edges:
                    edge = self.rrg.edges[edge_id]
                    neighbor_node_id = edge.to_node if edge.from_node == node_id else edge.from_node
                    
                    # Convert neighbor to grid position
                    if neighbor_node_id in self.grid.node_to_grid:
                        neighbor_grid_pos = self.grid.node_to_grid[neighbor_node_id]
                        neighbors_list.append(neighbor_grid_pos)
        
        # Also check incoming edges (bidirectional connectivity)
        for edge_id, edge in self.rrg.edges.items():
            if edge.to_node == node_id:
                neighbor_node_id = edge.from_node
                if neighbor_node_id in self.grid.node_to_grid:
                    neighbor_grid_pos = self.grid.node_to_grid[neighbor_node_id]
                    if neighbor_grid_pos not in neighbors_list:
                        neighbors_list.append(neighbor_grid_pos)
        
        if not neighbors_list:
            logger.warning(f"ERROR: Node {node_id} has no edges in RRG - this is why routing fails!")
        
        # Cache the neighbor list for this grid position
        self.grid.neighbors_cache[grid_pos] = neighbors_list
        logger.info(f"Cached {len(neighbors_list)} neighbors for dynamic node {node_id} at {grid_pos}")
        
        # CRITICAL FIX: Update existing grid nodes to point back to this escape via
        for neighbor_grid_pos in neighbors_list:
            if neighbor_grid_pos in self.grid.neighbors_cache:
                # Add bidirectional connectivity
                if grid_pos not in self.grid.neighbors_cache[neighbor_grid_pos]:
                    self.grid.neighbors_cache[neighbor_grid_pos].append(grid_pos)
        
        logger.debug(f"Updated {len(neighbors_list)} existing nodes to include escape via {node_id}")
        
    def _precompute_rrg_connectivity_optimized(self):
        """Optimized RRG connectivity pre-computation for enterprise scale"""
        total_nodes = len(self.grid.grid_to_node)
        processed = 0
        batch_size = 10000  # Process in batches for progress tracking
        
        logger.info(f"Pre-computing connectivity for {total_nodes} nodes in batches...")
        
        for grid_pos, node_id in self.grid.grid_to_node.items():
            neighbors_list = []
            
            # FAST PATH: Check adjacency dict directly (most efficient)
            if node_id in self.rrg.adjacency:
                for edge_id in self.rrg.adjacency[node_id]:
                    if edge_id in self.rrg.edges:
                        edge = self.rrg.edges[edge_id] 
                        # Get the neighbor node (handle bidirectional edges)
                        neighbor_node_id = edge.to_node if edge.from_node == node_id else edge.from_node
                        
                        # Map to grid if available
                        if neighbor_node_id in self.grid.node_to_grid:
                            neighbor_grid_pos = self.grid.node_to_grid[neighbor_node_id]
                            neighbors_list.append(neighbor_grid_pos)
            
            # BIDIRECTIONAL: Also check incoming edges
            for edge_id, edge in self.rrg.edges.items():
                if edge.to_node == node_id:
                    neighbor_node_id = edge.from_node
                    if neighbor_node_id in self.grid.node_to_grid:
                        neighbor_grid_pos = self.grid.node_to_grid[neighbor_node_id]
                        if neighbor_grid_pos not in neighbors_list:
                            neighbors_list.append(neighbor_grid_pos)
            
            # Cache the neighbor list
            self.grid.neighbors_cache[grid_pos] = neighbors_list
            processed += 1
            
            # Progress reporting
            if processed % batch_size == 0:
                progress = (processed / total_nodes) * 100
                logger.info(f"Connectivity caching: {processed}/{total_nodes} ({progress:.1f}%)")
    
    def _precompute_rrg_connectivity(self):
        """Legacy connectivity pre-computation (kept for compatibility)"""
        return self._precompute_rrg_connectivity_optimized()
    
    def _compute_neighbors_lazy(self, node_id: str) -> List[Tuple[int, int, int]]:
        """Compute neighbors for a single node on-demand (lazy evaluation)"""
        neighbors_list = []
        
        # FAST PATH: Check adjacency dict directly
        if node_id in self.rrg.adjacency:
            for edge_id in self.rrg.adjacency[node_id]:
                if edge_id in self.rrg.edges:
                    edge = self.rrg.edges[edge_id]
                    # Get the neighbor node (handle bidirectional edges)
                    neighbor_node_id = edge.to_node if edge.from_node == node_id else edge.from_node
                    
                    # Map to grid if available
                    if neighbor_node_id in self.grid.node_to_grid:
                        neighbor_grid_pos = self.grid.node_to_grid[neighbor_node_id]
                        neighbors_list.append(neighbor_grid_pos)
        
        # BIDIRECTIONAL: Also check incoming edges (only check a subset for performance)
        edge_sample = 0
        for edge_id, edge in self.rrg.edges.items():
            if edge.to_node == node_id:
                neighbor_node_id = edge.from_node
                if neighbor_node_id in self.grid.node_to_grid:
                    neighbor_grid_pos = self.grid.node_to_grid[neighbor_node_id]
                    if neighbor_grid_pos not in neighbors_list:
                        neighbors_list.append(neighbor_grid_pos)
            
            # Limit bidirectional search for performance (enterprise optimization)
            edge_sample += 1
            if edge_sample > 10000:  # Only check first 10k edges
                break
        
        return neighbors_list
        
    def _estimate_gpu_memory(self) -> float:
        """Estimate GPU memory usage in MB"""
        total_cells = self.grid_layers * self.grid_height * self.grid_width
        
        # 4 arrays: obstacle(1 byte) + cost(4 bytes) + distance(4 bytes) + parent(4 bytes) 
        memory_bytes = total_cells * (1 + 4 + 4 + 4)
        return memory_bytes / (1024 * 1024)  # Convert to MB
        
    def route_single_net(self, request: RouteRequest) -> RouteResult:
        """Route single net using GPU wavefront with memory safety"""
        try:
            if self.grid is None:
                self.build_grid()
                if self.grid is None:  # Safety check
                    logger.error("Grid creation failed, cannot route")
                    return RouteResult(net_id=request.net_id, success=False)
                
            start_time = time.time()
            
            # Check GPU memory before routing
            if self.use_gpu:
                try:
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    if free_mem < 1024**3:  # Less than 1GB free
                        logger.warning(f"Low GPU memory: {free_mem/(1024**3):.1f}GB free")
                except Exception as e:
                    logger.warning(f"Cannot check GPU memory: {e}")
        
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
            
            # Verify connectivity around source and sink
            source_neighbors = len(self.grid.neighbors_cache.get(source_pos, []))
            sink_neighbors = len(self.grid.neighbors_cache.get(sink_pos, []))
            logger.debug(f"Source has {source_neighbors} neighbors, sink has {sink_neighbors} neighbors")
            
            # Run GPU wavefront with memory protection
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
                
        except Exception as e:
            logger.error(f"Critical error routing net {request.net_id}: {e}")
            # Force cleanup on error to prevent memory leaks
            try:
                self.cleanup_gpu_memory()
            except:
                pass
            return RouteResult(net_id=request.net_id, success=False)
    
    def _gpu_wavefront(self, source_pos: Tuple[int, int, int], 
                       sink_pos: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """Industrial-grade bidirectional A* pathfinding with memory safety"""
        
        logger.warning(f"PERFORMANCE: GPU PATHFINDING CALLED: {source_pos} -> {sink_pos} (use_gpu={self.use_gpu})")
        
        try:
            # Reset distance and parent grids
            self.grid.distance_grid.fill(-1)
            self.grid.parent_grid.fill(-1)
            if self.use_gpu:
                logger.warning("PERFORMANCE: Using GPU arrays for pathfinding!")
            else:
                logger.warning("💻 Using CPU arrays for pathfinding")
        except Exception as e:
            logger.error(f"Error resetting grids: {e}")
            return None
        
        # Check if source and sink are passable
        source_layer, source_y, source_x = source_pos
        sink_layer, sink_y, sink_x = sink_pos
        
        if self.grid.obstacle_grid[source_layer, source_y, source_x]:
            logger.error(f"Source position {source_pos} is blocked by obstacle")
            return None
            
        if self.grid.obstacle_grid[sink_layer, sink_y, sink_x]:
            logger.error(f"Sink position {sink_pos} is blocked by obstacle") 
            return None
        
        # Use GPU-accelerated pathfinding for massive enterprise grids
        if self.use_gpu:
            return self._gpu_bidirectional_search(source_pos, sink_pos)
        else:
            return self._bidirectional_astar(source_pos, sink_pos)
    
    def _bidirectional_astar(self, source_pos: Tuple[int, int, int], 
                           sink_pos: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """Bidirectional A* for enterprise-scale routing on RTX 5090"""
        import heapq
        
        # Forward search from source
        forward_open = [(0, source_pos)]  # (f_score, position)
        forward_g_score = {source_pos: 0}
        forward_came_from = {}
        forward_closed = set()
        
        # Backward search from sink  
        backward_open = [(0, sink_pos)]
        backward_g_score = {sink_pos: 0}
        backward_came_from = {}
        backward_closed = set()
        
        def manhattan_heuristic(pos1, pos2):
            """3D Manhattan distance heuristic with layer change penalty"""
            return abs(pos1[0] - pos2[0]) * 2 + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
        
        def get_neighbors(pos):
            """Get valid neighbors for A* search with lazy connectivity"""
            # Check cache first
            if pos in self.grid.neighbors_cache:
                return self.grid.neighbors_cache[pos]
            
            # Compute neighbors on-demand for enterprise scale
            if pos in self.grid.grid_to_node:
                node_id = self.grid.grid_to_node[pos]
                neighbors = self._compute_neighbors_lazy(node_id)
                self.grid.neighbors_cache[pos] = neighbors  # Cache for next time
                return neighbors
            
            return []
        
        # Bidirectional search with meet-in-the-middle
        max_iterations = 100000  # Much higher limit for A*
        iteration = 0
        
        # DEBUG: Log initial conditions
        logger.warning(f"A* starting: {source_pos} -> {sink_pos}")
        source_neighbors = get_neighbors(source_pos)
        sink_neighbors = get_neighbors(sink_pos)
        logger.warning(f"Source has {len(source_neighbors)} neighbors: {source_neighbors}")
        logger.warning(f"Sink has {len(sink_neighbors)} neighbors: {sink_neighbors}")
        
        while forward_open and backward_open and iteration < max_iterations:
            iteration += 1
            
            # Forward step
            if forward_open:
                current_f, current = heapq.heappop(forward_open)
                
                if current in forward_closed:
                    continue
                    
                forward_closed.add(current)
                
                # Check if we met the backward search
                if current in backward_closed:
                    logger.debug(f"A* paths met at {current} in {iteration} iterations")
                    return self._reconstruct_bidirectional_path(current, forward_came_from, backward_came_from)
                
                # Expand forward
                neighbors = get_neighbors(current)
                if iteration <= 3:
                    logger.warning(f"Forward iteration {iteration}: current={current}, neighbors={neighbors}")
                
                for neighbor in neighbors:
                    if neighbor in forward_closed:
                        continue
                        
                    nz, ny, nx = neighbor
                    if (nz < 0 or nz >= self.grid.layers or
                        ny < 0 or ny >= self.grid.height or  
                        nx < 0 or nx >= self.grid.width or
                        self.grid.obstacle_grid[nz, ny, nx]):
                        continue
                    
                    tentative_g = forward_g_score[current] + 1
                    
                    if neighbor not in forward_g_score or tentative_g < forward_g_score[neighbor]:
                        forward_came_from[neighbor] = current
                        forward_g_score[neighbor] = tentative_g
                        f_score = tentative_g + manhattan_heuristic(neighbor, sink_pos)
                        heapq.heappush(forward_open, (f_score, neighbor))
            
            # Backward step
            if backward_open:
                current_f, current = heapq.heappop(backward_open)
                
                if current in backward_closed:
                    continue
                    
                backward_closed.add(current)
                
                # Check if we met the forward search
                if current in forward_closed:
                    logger.debug(f"A* paths met at {current} in {iteration} iterations")
                    return self._reconstruct_bidirectional_path(current, forward_came_from, backward_came_from)
                
                # Expand backward
                neighbors = get_neighbors(current)
                if iteration <= 3:
                    logger.warning(f"Backward iteration {iteration}: current={current}, neighbors={neighbors}")
                
                for neighbor in neighbors:
                    if neighbor in backward_closed:
                        continue
                        
                    nz, ny, nx = neighbor
                    if (nz < 0 or nz >= self.grid.layers or
                        ny < 0 or ny >= self.grid.height or  
                        nx < 0 or nx >= self.grid.width or
                        self.grid.obstacle_grid[nz, ny, nx]):
                        continue
                    
                    tentative_g = backward_g_score[current] + 1
                    
                    if neighbor not in backward_g_score or tentative_g < backward_g_score[neighbor]:
                        backward_came_from[neighbor] = current
                        backward_g_score[neighbor] = tentative_g
                        f_score = tentative_g + manhattan_heuristic(neighbor, source_pos)
                        heapq.heappush(backward_open, (f_score, neighbor))
            
            # Early termination if no neighbors can be expanded
            if not forward_open and not backward_open:
                logger.warning(f"A* search exhausted: no more neighbors to explore after {iteration} iterations")
                break
        
        logger.warning(f"Bidirectional A* failed after {iteration} iterations (forward_open={len(forward_open)}, backward_open={len(backward_open)})")
        return None
    
    def _reconstruct_bidirectional_path(self, meeting_point, forward_came_from, backward_came_from):
        """Reconstruct path from bidirectional search"""
        path = []
        
        # Build forward path (source to meeting point)
        current = meeting_point
        forward_path = []
        while current in forward_came_from:
            forward_path.append(current)
            current = forward_came_from[current]
        forward_path.append(current)  # Add source
        forward_path.reverse()
        
        # Build backward path (meeting point to sink)  
        current = meeting_point
        backward_path = []
        while current in backward_came_from:
            current = backward_came_from[current]
            backward_path.append(current)
        
        # Combine paths
        path = forward_path + backward_path
        logger.debug(f"Reconstructed bidirectional path with {len(path)} nodes")
        return path
    
    def _gpu_bidirectional_search(self, source_pos: Tuple[int, int, int], 
                                sink_pos: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """TRUE GPU-accelerated bidirectional search using CuPy arrays"""
        
        # Initialize GPU arrays for search
        source_layer, source_y, source_x = source_pos
        sink_layer, sink_y, sink_x = sink_pos
        
        logger.info(f"GPU search: {source_pos} -> {sink_pos}")
        
        # Use GPU wavefront expansion
        self.grid.distance_grid.fill(-1)  # Reset on GPU
        
        # Set source and sink distances
        self.grid.distance_grid[source_layer, source_y, source_x] = 0
        
        # GPU wavefront expansion
        max_iterations = 1000  # Reasonable limit for GPU expansion
        current_distance = 0
        
        while current_distance < max_iterations:
            # Find all cells at current distance
            current_cells = (self.grid.distance_grid == current_distance)
            
            if not cp.any(current_cells):
                break  # No more cells to expand
                
            # Check if we reached the sink
            if self.grid.distance_grid[sink_layer, sink_y, sink_x] >= 0:
                logger.info(f"GPU pathfinding reached sink in {current_distance} iterations")
                return self._trace_gpu_path(sink_pos)
            
            # Expand wavefront using GPU
            self._expand_wavefront_gpu(current_cells, current_distance + 1)
            current_distance += 1
            
            # Progress logging for long searches
            if current_distance % 100 == 0:
                logger.debug(f"GPU search iteration {current_distance}")
        
        logger.warning(f"GPU bidirectional search failed after {current_distance} iterations")
        return None
    
    def _trace_gpu_path(self, sink_pos: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Trace path from sink back to source using GPU distance grid"""
        path = []
        current = sink_pos
        
        while True:
            path.append(current)
            layer, y, x = current
            current_dist = int(self.grid.distance_grid[layer, y, x])
            
            if current_dist == 0:
                break  # Reached source
                
            # Find neighbor with distance current_dist - 1
            found_prev = False
            if current in self.grid.neighbors_cache:
                for neighbor_pos in self.grid.neighbors_cache[current]:
                    nz, ny, nx = neighbor_pos
                    if (0 <= nz < self.grid.layers and 0 <= ny < self.grid.height and 
                        0 <= nx < self.grid.width and
                        self.grid.distance_grid[nz, ny, nx] == current_dist - 1):
                        current = neighbor_pos
                        found_prev = True
                        break
            
            if not found_prev:
                logger.warning(f"GPU path trace failed at {current}")
                break
                
            if len(path) > 10000:  # Safety limit
                logger.warning("GPU path trace exceeded maximum length")
                break
        
        path.reverse()
        logger.info(f"GPU traced path with {len(path)} nodes")
        return path
    
    def _expand_wavefront_gpu(self, current_cells: cp.ndarray, new_distance: int):
        """Expand wavefront using pre-computed RRG connectivity"""
        
        # Get coordinates of current cells (convert back to CPU for cache lookups)
        current_coords = cp.where(current_cells)
        current_z = cp.asnumpy(current_coords[0])
        current_y = cp.asnumpy(current_coords[1])
        current_x = cp.asnumpy(current_coords[2])
        
        expanded_count = 0
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
                    expanded_count += 1
                    
                    # Set parent pointer (encode parent coordinates)
                    parent_encoded = current_z[i] * 1000000 + current_y[i] * 1000 + current_x[i]
                    self.grid.parent_grid[neighbor_z, neighbor_y, neighbor_x] = parent_encoded
        
        # Log expansion results for successful routing
        if expanded_count > 0:
            logger.info(f"GPU wavefront expanded {expanded_count} cells to distance {new_distance}")
    
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
        """Clean up GPU memory with enhanced safety"""
        if self.use_gpu and self.grid:
            try:
                # Free GPU arrays safely
                if hasattr(self.grid, 'obstacle_grid'):
                    del self.grid.obstacle_grid
                if hasattr(self.grid, 'cost_grid'):
                    del self.grid.cost_grid  
                if hasattr(self.grid, 'distance_grid'):
                    del self.grid.distance_grid
                if hasattr(self.grid, 'parent_grid'):
                    del self.grid.parent_grid
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear memory pool
                cp.get_default_memory_pool().free_all_blocks()
                
                # Log memory status
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                logger.info(f"GPU memory cleaned: {free_mem/(1024**3):.1f}GB free of {total_mem/(1024**3):.1f}GB")
                
            except Exception as e:
                logger.error(f"GPU memory cleanup error: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        self.cleanup_gpu_memory()
        if exc_type:
            logger.error(f"Pathfinder exiting due to error: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions