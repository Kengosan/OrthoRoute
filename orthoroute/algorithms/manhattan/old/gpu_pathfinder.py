"""
GPU-Accelerated PathFinder Algorithm for RRG Routing
Implements true PathFinder negotiated congestion on GPU while preserving RRG fabric intelligence
"""

import numpy as np
import logging
import time
import heapq
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from .gpu_rrg import GPURoutingResourceGraph, GPUPathFindingState
from .rrg import RouteRequest, RouteResult, RoutingConfig

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

class GPUPathFinderRouter:
    """GPU-accelerated PathFinder with negotiated congestion for RRG routing"""
    
    def __init__(self, gpu_rrg: GPURoutingResourceGraph, config: RoutingConfig):
        self.gpu_rrg = gpu_rrg
        self.config = config
        self.use_gpu = gpu_rrg.use_gpu
        
        # PathFinder parameters
        self.pres_fac = self.config.pres_fac_init
        self.max_iterations = self.config.max_iterations
        
        # Statistics
        self.routing_stats = {
            'total_nets_routed': 0,
            'successful_routes': 0,
            'iterations_used': 0,
            'total_routing_time': 0.0
        }
        
        logger.info(f"GPU PathFinder router initialized (GPU: {self.use_gpu})")
        
        # Check for CSR adjacency matrix support and initialize advanced components
        if self.use_gpu and hasattr(gpu_rrg, 'adjacency_csr'):
            logger.info("Using optimized GPU CSR pathfinding with parallel PathFinder")
            self.use_csr_pathfinding = True
            
            # Initialize true parallel PathFinder
            from .parallel_pathfinder import TrueParallelPathFinder
            self.parallel_pathfinder = TrueParallelPathFinder(gpu_rrg)
            
            # Initialize F.Cu tap builder  
            from .fcu_tap_builder import FCuVerticalTapBuilder
            fabric_bounds = self._get_fabric_bounds()
            self.fcu_tap_builder = FCuVerticalTapBuilder(fabric_bounds)
            
            # Initialize blind via manager
            from .blind_via_manager import BlindViaManager
            layer_count = gpu_rrg.cpu_rrg.layer_count if hasattr(gpu_rrg, 'cpu_rrg') else 11
            self.blind_via_manager = BlindViaManager(layer_count)
            
            logger.info("Advanced routing components initialized: parallel PathFinder + F.Cu taps + blind vias")
        else:
            logger.info("Using fallback pathfinding implementation") 
            self.use_csr_pathfinding = False
    
    def route_all_nets(self, requests: List[RouteRequest]) -> Dict[str, RouteResult]:
        """Route all nets using PathFinder negotiated congestion"""
        
        logger.info(f"Starting PathFinder routing for {len(requests)} nets")
        start_time = time.time()
        
        # Initialize routing state
        self._reset_routing_state()
        
        # PathFinder main loop with negotiated congestion
        all_routes = {}
        
        for iteration in range(self.max_iterations):
            logger.debug(f"PathFinder iteration {iteration + 1}/{self.max_iterations}")
            
            # Route all nets in current iteration
            current_routes = self._route_nets_iteration(requests)
            all_routes.update(current_routes)
            
            # Check for congestion conflicts
            conflicts = self._detect_resource_conflicts(current_routes)
            
            if not conflicts:
                logger.info(f"PathFinder converged after {iteration + 1} iterations")
                break
                
            # Update congestion costs for next iteration
            self._update_congestion_costs(current_routes)
            
            # Select nets to rip up and re-route
            requests = self._select_rip_up_nets(current_routes, conflicts)
            
            # Update present factor for increasing congestion pressure
            self.pres_fac *= self.config.pres_fac_mult
            
            logger.debug(f"Iteration {iteration + 1}: {len(conflicts)} conflicts, "
                        f"{len(requests)} nets to re-route, pres_fac={self.pres_fac:.2f}")
        
        # Final statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in all_routes.values() if r.success)
        
        self.routing_stats.update({
            'total_nets_routed': len(all_routes),
            'successful_routes': successful,
            'iterations_used': min(iteration + 1, self.max_iterations),
            'total_routing_time': total_time
        })
        
        logger.info(f"PathFinder completed: {successful}/{len(all_routes)} nets routed "
                   f"in {total_time:.2f}s ({self.routing_stats['iterations_used']} iterations)")
        
        return all_routes
    
    def _reset_routing_state(self):
        """Reset PathFinder state for new routing"""
        state = self.gpu_rrg.pathfinder_state
        
        # Reset usage counters
        state.node_usage.fill(0)
        state.edge_usage.fill(0)
        
        # Reset congestion costs
        state.node_pres_cost.fill(0.0)
        state.edge_pres_cost.fill(0.0) 
        state.node_hist_cost.fill(0.0)
        state.edge_hist_cost.fill(0.0)
        
        # Reset routing costs to base costs
        if self.use_gpu:
            cp.copyto(state.node_cost, self.gpu_rrg.node_base_cost)
            cp.copyto(state.edge_cost, self.gpu_rrg.edge_base_cost)
        else:
            np.copyto(state.node_cost, self.gpu_rrg.node_base_cost)
            np.copyto(state.edge_cost, self.gpu_rrg.edge_base_cost)
        
        # Reset present factor
        self.pres_fac = self.config.pres_fac_init
        
        logger.debug("PathFinder routing state reset")
    
    def _route_nets_iteration(self, requests: List[RouteRequest]) -> Dict[str, RouteResult]:
        """Route all nets in a single PathFinder iteration"""
        results = {}
        
        for request in requests:
            # Route single net using current costs
            result = self._route_single_net(request)
            results[request.net_id] = result
            
            # Update resource usage if routing succeeded
            if result.success:
                self._update_resource_usage(result)
        
        return results
    
    def route_single_net(self, request: RouteRequest) -> RouteResult:
        """Route single net using GPU-accelerated pathfinding (public interface)"""
        logger.info(f"GPU PathFinder routing single net: {request.net_id}")
        start_time = time.time()
        
        try:
            # DEBUG: Check if source and sink nodes exist
            source_node = self.gpu_rrg.node_id_to_idx.get(request.source_pad)
            sink_node = self.gpu_rrg.node_id_to_idx.get(request.sink_pad)
            
            logger.info(f"DEBUG: Source pad {request.source_pad} -> node {source_node}")
            logger.info(f"DEBUG: Sink pad {request.sink_pad} -> node {sink_node}")
            
            if source_node is None:
                logger.error(f"ERROR: Source pad {request.source_pad} not found in node mapping")
                return RouteResult(net_id=request.net_id, success=False)
                
            if sink_node is None:
                logger.error(f"ERROR: Sink pad {request.sink_pad} not found in node mapping")
                return RouteResult(net_id=request.net_id, success=False)
            
            # DEBUG: Check connectivity from tap nodes
            if hasattr(self.gpu_rrg, 'tap_edge_connections'):
                source_connections = len(self.gpu_rrg.tap_edge_connections.get(source_node, []))
                sink_connections = len(self.gpu_rrg.tap_edge_connections.get(sink_node, []))
                logger.info(f"DEBUG: Source node has {source_connections} tap connections")
                logger.info(f"DEBUG: Sink node has {sink_connections} tap connections")
            
            result = self._route_single_net(request)
            route_time = time.time() - start_time
            
            if result.success:
                logger.info(f"GPU PathFinder SUCCESS: {request.net_id} routed in {route_time:.3f}s")
            else:
                logger.warning(f"GPU PathFinder FAILED: {request.net_id} after {route_time:.3f}s")
            
            return result
            
        except Exception as e:
            route_time = time.time() - start_time
            logger.error(f"GPU PathFinder ERROR: {request.net_id} failed after {route_time:.3f}s: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return RouteResult(net_id=request.net_id, success=False)
    
    def _route_single_net(self, request: RouteRequest) -> RouteResult:
        """Route single net using GPU-accelerated pathfinding"""
        
        # Get source and sink node indices
        source_idx = self.gpu_rrg.get_node_idx(request.source_pad)
        sink_idx = self.gpu_rrg.get_node_idx(request.sink_pad)
        
        if source_idx is None or sink_idx is None:
            logger.error(f"Cannot find nodes for {request.net_id}: "
                        f"source={request.source_pad}, sink={request.sink_pad}")
            return RouteResult(net_id=request.net_id, success=False)
        
        # Get pathfinding state
        state = self.gpu_rrg.pathfinder_state
        
        # Run pathfinding algorithm - choose implementation based on availability
        if self.use_csr_pathfinding:
            from .gpu_csr_pathfinder import GPUCSRPathFinder
            csr_pathfinder = GPUCSRPathFinder(self.gpu_rrg)
            path_nodes, path_edges = csr_pathfinder.route_single_net_csr(source_idx, sink_idx, state)
        else:
            path_nodes, path_edges = self._dijkstra_pathfinding(source_idx, sink_idx)
        
        if not path_nodes:
            return RouteResult(net_id=request.net_id, success=False)
        
        # Convert indices back to IDs
        node_ids = [self.gpu_rrg.get_node_id(idx) for idx in path_nodes]
        edge_ids = [self.gpu_rrg.get_edge_id(idx) for idx in path_edges]
        
        # Calculate metrics
        total_cost = self._calculate_path_cost(path_nodes, path_edges)
        total_length = self._calculate_path_length(path_edges)
        via_count = self._count_vias(path_nodes)
        
        return RouteResult(
            net_id=request.net_id,
            success=True,
            path=node_ids,
            edges=edge_ids,
            cost=total_cost,
            length_mm=total_length,
            via_count=via_count
        )
    
    def _dijkstra_pathfinding(self, source_idx: int, sink_idx: int) -> Tuple[List[int], List[int]]:
        """Pure GPU wavefront pathfinding using sparse CSR adjacency structure"""
        
        state = self.gpu_rrg.pathfinder_state
        
        # Reset pathfinding workspace
        state.distance.fill(float('inf') if not self.use_gpu else cp.inf)
        state.parent_node.fill(-1)
        state.parent_edge.fill(-1)
        state.visited.fill(False)
        
        # CRITICAL BOUNDS CHECKING: Validate source and sink indices
        state_size = len(state.distance)
        
        if source_idx < 0 or source_idx >= state_size:
            raise IndexError(f"Source index {source_idx} is out of bounds for PathFinder state arrays (size: {state_size}). "
                           f"Valid range: [0, {state_size-1}]. This indicates tap nodes were assigned invalid indices.")
        
        if sink_idx < 0 or sink_idx >= state_size:
            raise IndexError(f"Sink index {sink_idx} is out of bounds for PathFinder state arrays (size: {state_size}). "
                           f"Valid range: [0, {state_size-1}]. This indicates tap nodes were assigned invalid indices.")
        
        # Validate adjacency matrix consistency
        if hasattr(self.gpu_rrg, 'adjacency_matrix') and self.gpu_rrg.adjacency_matrix is not None:
            matrix_size = self.gpu_rrg.adjacency_matrix.shape[0]
            indptr_size = len(self.gpu_rrg.adjacency_matrix.indptr)
            
            if source_idx >= matrix_size or sink_idx >= matrix_size:
                raise IndexError(f"Node indices exceed adjacency matrix size: source={source_idx}, sink={sink_idx}, "
                               f"matrix_size={matrix_size}. PathFinder cannot access node neighbors.")
            
            if indptr_size != state_size + 1:
                raise ValueError(f"Adjacency matrix indptr size {indptr_size} inconsistent with state size {state_size}. "
                               f"Expected indptr size: {state_size + 1}. This WILL cause out-of-bounds errors.")
        
        logger.info(f"PathFinder initialization validated: source={source_idx}, sink={sink_idx}, state_size={state_size}")
        
        state.distance[source_idx] = 0.0
        
        # Priority queue for Dijkstra's algorithm
        # Note: Using CPU heap even for GPU version (GPU heaps are complex)
        pq = [(0.0, source_idx)]
        
        nodes_expanded = 0
        max_expansions = min(10000, self.gpu_rrg.num_nodes)  # Limit for performance
        
        while pq and nodes_expanded < max_expansions:
            current_dist, current_idx = heapq.heappop(pq)
            
            # Bounds checking for array access
            if current_idx >= len(state.visited):
                logger.error(f"Current index {current_idx} is out of bounds for PathFinder state arrays (size: {len(state.visited)})")
                break
            
            if state.visited[current_idx]:
                continue
            
            state.visited[current_idx] = True
            nodes_expanded += 1
            
            # Check if we reached the sink
            if current_idx == sink_idx:
                logger.debug(f"Path found after expanding {nodes_expanded} nodes")
                return self._reconstruct_path(source_idx, sink_idx)
            
            # Expand neighbors using adjacency matrix
            self._expand_neighbors(current_idx, pq, state)
        
        logger.debug(f"Pathfinding failed after expanding {nodes_expanded} nodes")
        return [], []
    
    def _expand_neighbors(self, current_idx: int, pq: List, state: GPUPathFindingState):
        """Expand neighbors of current node with comprehensive bounds checking"""
        
        # CRITICAL BOUNDS CHECK: Validate current_idx before any array access
        if current_idx < 0 or current_idx >= self.gpu_rrg.num_nodes:
            raise IndexError(f"Current node index {current_idx} is out of bounds [0, {self.gpu_rrg.num_nodes-1}]. "
                           f"This indicates a fundamental indexing error in the routing graph.")
        
        # Get neighbors from sparse adjacency matrix
        adj_matrix = self.gpu_rrg.adjacency_matrix
        
        # CRITICAL BOUNDS CHECK: Validate adjacency matrix access
        indptr_size = len(adj_matrix.indptr)
        if current_idx >= indptr_size - 1:  # indptr has size num_nodes + 1
            raise IndexError(f"Cannot access adjacency matrix indptr[{current_idx + 1}]. "
                           f"Matrix indptr size is {indptr_size}, valid access range is [0, {indptr_size-2}]. "
                           f"Node count: {self.gpu_rrg.num_nodes}, current_idx: {current_idx}")
        
        # Get row from sparse matrix (neighbors of current node)
        start_idx = adj_matrix.indptr[current_idx]
        end_idx = adj_matrix.indptr[current_idx + 1]
        
        logger.debug(f"Expanding node {current_idx}: adjacency range [{start_idx}, {end_idx})")
        
        for i in range(start_idx, end_idx):
            # BOUNDS CHECK: Validate adjacency matrix data access
            if i >= len(adj_matrix.indices) or i >= len(adj_matrix.data):
                logger.error(f"Adjacency matrix data index {i} out of bounds (indices: {len(adj_matrix.indices)}, data: {len(adj_matrix.data)})")
                continue
            
            neighbor_idx = adj_matrix.indices[i]
            edge_idx = int(adj_matrix.data[i])  # Convert float32 back to int
            
            # BOUNDS CHECK: Validate neighbor node index
            if neighbor_idx < 0 or neighbor_idx >= len(state.visited):
                logger.warning(f"Neighbor index {neighbor_idx} out of bounds [0, {len(state.visited)-1}] from node {current_idx}, skipping")
                continue
            
            if state.visited[neighbor_idx]:
                continue
            
            # BOUNDS CHECK: Validate edge index
            if edge_idx < 0 or edge_idx >= len(state.edge_cost):
                logger.warning(f"Edge index {edge_idx} out of bounds [0, {len(state.edge_cost)-1}], using default cost")
                edge_cost = 1.0  # Default cost for invalid edges
            else:
                # Calculate edge cost including congestion
                edge_cost = self._calculate_edge_cost(edge_idx, state)
            
            # Process this neighbor
            self._process_neighbor(current_idx, neighbor_idx, edge_idx, edge_cost, pq, state)
        
        # OPTIMIZATION: Also check tap connections (stored separately) with bounds validation
        if hasattr(self.gpu_rrg, 'tap_edge_connections') and current_idx in self.gpu_rrg.tap_edge_connections:
            tap_connections = self.gpu_rrg.tap_edge_connections[current_idx]
            logger.debug(f"Found {len(tap_connections)} tap connections for node {current_idx}")
            
            for connection in tap_connections:
                neighbor_idx = connection['target']
                edge_idx = connection['edge_idx']
                
                # BOUNDS CHECK: Validate tap neighbor index
                if neighbor_idx < 0 or neighbor_idx >= len(state.visited):
                    logger.warning(f"Tap neighbor index {neighbor_idx} out of bounds [0, {len(state.visited)-1}] from node {current_idx}, skipping")
                    continue
                
                if state.visited[neighbor_idx]:
                    continue
                
                # Use base edge cost for tap connections (no congestion data stored separately)
                edge_cost = 1.0  # Tap connections have minimal cost
                
                # Process this tap neighbor with bounds validation
                self._process_neighbor(current_idx, neighbor_idx, edge_idx, edge_cost, pq, state)
    
    def _process_neighbor(self, current_idx: int, neighbor_idx: int, edge_idx: int, edge_cost: float, pq: List, state: GPUPathFindingState):
        """Process a neighbor node during expansion with comprehensive validation"""
        
        # BOUNDS CHECK: Validate all indices before processing
        state_size = len(state.distance)
        if current_idx < 0 or current_idx >= state_size:
            logger.error(f"Current index {current_idx} out of bounds [0, {state_size-1}]")
            return
        
        if neighbor_idx < 0 or neighbor_idx >= state_size:
            logger.error(f"Neighbor index {neighbor_idx} out of bounds [0, {state_size-1}]")
            return
        
        # Check capacity constraint
        if not self._check_capacity(neighbor_idx, edge_idx, state):
            return
        
        # Update distance if better path found
        new_dist = state.distance[current_idx] + edge_cost
        
        if new_dist < state.distance[neighbor_idx]:
            state.distance[neighbor_idx] = new_dist
            state.parent_node[neighbor_idx] = current_idx
            state.parent_edge[neighbor_idx] = edge_idx
            
            heapq.heappush(pq, (float(new_dist), neighbor_idx))
    
    def _calculate_edge_cost(self, edge_idx: int, state: GPUPathFindingState) -> float:
        """Calculate total edge cost including congestion penalties"""
        
        # Base cost (length, via cost, etc.)
        base_cost = float(state.edge_cost[edge_idx])
        
        # Present congestion cost (current iteration conflicts)
        pres_cost = float(state.edge_pres_cost[edge_idx]) * self.pres_fac
        
        # Historical congestion cost (accumulated from previous iterations)
        hist_cost = float(state.edge_hist_cost[edge_idx])
        
        return base_cost + pres_cost + hist_cost
    
    def _check_capacity(self, node_idx: int, edge_idx: int, state: GPUPathFindingState) -> bool:
        """Check if node and edge have available capacity with bounds validation"""
        
        # BOUNDS CHECK: Validate node index
        if node_idx < 0 or node_idx >= len(state.node_usage):
            logger.warning(f"Node index {node_idx} out of bounds for capacity check, assuming no capacity")
            return False
        
        # Check node capacity
        node_available = state.node_usage[node_idx] < state.node_capacity[node_idx]
        
        # BOUNDS CHECK: Validate edge index
        if edge_idx < 0 or edge_idx >= len(state.edge_usage):
            logger.warning(f"Edge index {edge_idx} out of bounds for capacity check, assuming available")
            edge_available = True  # Assume tap edges have capacity
        else:
            # Check edge capacity  
            edge_available = state.edge_usage[edge_idx] < state.edge_capacity[edge_idx]
        
        return node_available and edge_available
    
    def _reconstruct_path(self, source_idx: int, sink_idx: int) -> Tuple[List[int], List[int]]:
        """Reconstruct path from sink back to source"""
        
        state = self.gpu_rrg.pathfinder_state
        path_nodes = []
        path_edges = []
        
        current_idx = sink_idx
        
        while current_idx != source_idx:
            path_nodes.append(current_idx)
            
            parent_idx = int(state.parent_node[current_idx])
            edge_idx = int(state.parent_edge[current_idx])
            
            if parent_idx == -1:
                logger.error("Path reconstruction failed - invalid parent")
                return [], []
            
            path_edges.append(edge_idx)
            current_idx = parent_idx
        
        path_nodes.append(source_idx)  # Add source
        
        # Reverse to get source->sink order
        path_nodes.reverse()
        path_edges.reverse()
        
        return path_nodes, path_edges
    
    def _update_resource_usage(self, result: RouteResult):
        """Update resource usage counters for successful route"""
        
        state = self.gpu_rrg.pathfinder_state
        
        # Update node usage
        for node_id in result.path:
            node_idx = self.gpu_rrg.get_node_idx(node_id)
            if node_idx is not None:
                state.node_usage[node_idx] += 1
        
        # Update edge usage
        for edge_id in result.edges:
            edge_idx = self.gpu_rrg.get_edge_idx(edge_id)
            if edge_idx is not None:
                state.edge_usage[edge_idx] += 1
    
    def _detect_resource_conflicts(self, routes: Dict[str, RouteResult]) -> Set[str]:
        """Detect nets with resource conflicts (over-capacity usage)"""
        
        conflicts = set()
        state = self.gpu_rrg.pathfinder_state
        
        # Find over-capacity resources
        if self.use_gpu:
            node_overuse = cp.where(state.node_usage > state.node_capacity)[0]
            edge_overuse = cp.where(state.edge_usage > state.edge_capacity)[0]
        else:
            node_overuse = np.where(state.node_usage > state.node_capacity)[0]
            edge_overuse = np.where(state.edge_usage > state.edge_capacity)[0]
        
        # Find nets using over-capacity resources
        for net_id, route in routes.items():
            if not route.success:
                continue
                
            # Check if route uses over-capacity nodes
            for node_id in route.path:
                node_idx = self.gpu_rrg.get_node_idx(node_id)
                if node_idx in node_overuse:
                    conflicts.add(net_id)
                    break
            
            # Check if route uses over-capacity edges
            for edge_id in route.edges:
                edge_idx = self.gpu_rrg.get_edge_idx(edge_id)
                if edge_idx in edge_overuse:
                    conflicts.add(net_id)
                    break
        
        return conflicts
    
    def _update_congestion_costs(self, routes: Dict[str, RouteResult]):
        """Update congestion costs based on resource usage"""
        
        state = self.gpu_rrg.pathfinder_state
        
        # Update present congestion costs (reset each iteration)
        if self.use_gpu:
            # GPU version: vectorized updates
            node_overuse = cp.maximum(0, state.node_usage - state.node_capacity)
            edge_overuse = cp.maximum(0, state.edge_usage - state.edge_capacity)
            
            state.node_pres_cost = cp.power(node_overuse, self.config.alpha)
            state.edge_pres_cost = cp.power(edge_overuse, self.config.alpha)
            
            # Update historical costs (accumulate over iterations)
            state.node_hist_cost += (node_overuse > 0) * self.config.hist_cost_step
            state.edge_hist_cost += (edge_overuse > 0) * self.config.hist_cost_step
        else:
            # CPU version
            node_overuse = np.maximum(0, state.node_usage - state.node_capacity)
            edge_overuse = np.maximum(0, state.edge_usage - state.edge_capacity)
            
            state.node_pres_cost = np.power(node_overuse, self.config.alpha)
            state.edge_pres_cost = np.power(edge_overuse, self.config.alpha)
            
            state.node_hist_cost += (node_overuse > 0) * self.config.hist_cost_step
            state.edge_hist_cost += (edge_overuse > 0) * self.config.hist_cost_step
    
    def _select_rip_up_nets(self, routes: Dict[str, RouteResult], conflicts: Set[str]) -> List[RouteRequest]:
        """Select nets to rip up and re-route in next iteration"""
        
        # Simple strategy: re-route all conflicted nets
        rip_up_requests = []
        
        for net_id in conflicts:
            if net_id in routes and routes[net_id].success:
                # Create route request for re-routing
                route = routes[net_id]
                if route.path and len(route.path) >= 2:
                    source_node = route.path[0]
                    sink_node = route.path[-1]
                    
                    request = RouteRequest(
                        net_id=net_id,
                        source_pad=source_node,
                        sink_pad=sink_node
                    )
                    rip_up_requests.append(request)
                    
                    # Remove usage from resources (rip up)
                    self._remove_route_usage(route)
        
        return rip_up_requests
    
    def _remove_route_usage(self, route: RouteResult):
        """Remove resource usage from ripped up route"""
        
        state = self.gpu_rrg.pathfinder_state
        
        # Remove node usage
        for node_id in route.path:
            node_idx = self.gpu_rrg.get_node_idx(node_id)
            if node_idx is not None:
                state.node_usage[node_idx] = max(0, state.node_usage[node_idx] - 1)
        
        # Remove edge usage
        for edge_id in route.edges:
            edge_idx = self.gpu_rrg.get_edge_idx(edge_id)
            if edge_idx is not None:
                state.edge_usage[edge_idx] = max(0, state.edge_usage[edge_idx] - 1)
    
    def _calculate_path_cost(self, path_nodes: List[int], path_edges: List[int]) -> float:
        """Calculate total path cost"""
        total_cost = 0.0
        
        for edge_idx in path_edges:
            total_cost += float(self.gpu_rrg.edge_base_cost[edge_idx])
        
        return total_cost
    
    def _calculate_path_length(self, path_edges: List[int]) -> float:
        """Calculate total path length in mm"""
        total_length = 0.0
        
        for edge_idx in path_edges:
            total_length += float(self.gpu_rrg.edge_lengths[edge_idx])
        
        return total_length
    
    def _count_vias(self, path_nodes: List[int]) -> int:
        """Count layer changes (vias) in path"""
        via_count = 0
        
        for i in range(1, len(path_nodes)):
            prev_layer = int(self.gpu_rrg.node_layers[path_nodes[i-1]])
            curr_layer = int(self.gpu_rrg.node_layers[path_nodes[i]])
            
            if prev_layer != curr_layer:
                via_count += 1
        
        return via_count
    
    def clear_routing_state(self):
        """Clear routing state and resource usage"""
        state = self.gpu_rrg.pathfinder_state
        
        # Reset all usage counters
        state.node_usage.fill(0)
        state.edge_usage.fill(0)
        
        # Reset congestion costs
        state.node_pres_cost.fill(0.0)
        state.edge_pres_cost.fill(0.0)
        state.node_hist_cost.fill(0.0)
        state.edge_hist_cost.fill(0.0)
        
        # Reset PathFinder parameters
        self.pres_fac = self.config.pres_fac_init
        
        # Clear statistics
        self.routing_stats = {
            'total_nets_routed': 0,
            'successful_routes': 0,
            'iterations_used': 0,
            'total_routing_time': 0.0
        }
        
        logger.info("GPU PathFinder routing state cleared")
    
    def _get_fabric_bounds(self) -> Tuple[float, float, float, float]:
        """Get fabric bounding box for tap builder"""
        # Extract bounds from GPU RRG 
        if hasattr(self.gpu_rrg, 'cpu_rrg') and hasattr(self.gpu_rrg.cpu_rrg, 'bounds'):
            bounds = self.gpu_rrg.cpu_rrg.bounds
            return (bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y)
        else:
            # Fallback to reasonable defaults
            return (-50.0, -50.0, 250.0, 250.0)
    
    def route_all_nets_parallel(self, requests: List[RouteRequest]) -> Dict[str, RouteResult]:
        """Route all nets using TRUE parallel PathFinder algorithm"""
        
        if not self.use_csr_pathfinding or not hasattr(self, 'parallel_pathfinder'):
            logger.warning("Parallel PathFinder not available, falling back to sequential routing")
            return self.route_all_nets(requests)
        
        logger.info(f"Starting TRUE PARALLEL PathFinder routing for {len(requests)} nets")
        start_time = time.time()
        
        # Convert route requests to parallel format
        from .parallel_pathfinder import ParallelRouteRequest
        parallel_requests = []
        
        for req in requests:
            # Get source and sink indices (with F.Cu tap integration)
            logger.info(f"ENDPOINT_FIND: Looking for endpoints for net {req.net_id}: source='{req.source_pad}', sink='{req.sink_pad}'")
            source_idx, sink_idx = self._get_routing_endpoints_with_taps(req)
            logger.info(f"ENDPOINT_FIND: Net {req.net_id} endpoints: source_idx={source_idx}, sink_idx={sink_idx}")
            if source_idx is not None and sink_idx is not None:
                parallel_req = ParallelRouteRequest(
                    net_id=req.net_id,
                    source_idx=source_idx,
                    sink_idx=sink_idx,
                    priority=1.0  # Could be adjusted based on net criticality
                )
                parallel_requests.append(parallel_req)
                logger.info(f"ENDPOINT_FIND: Added parallel request for {req.net_id}")
            else:
                logger.warning(f"ENDPOINT_FIND: Failed to find endpoints for {req.net_id} - SKIPPING")
        
        # Execute parallel PathFinder
        logger.info(f"PARALLEL_PF: About to call parallel_pathfinder.route_all_nets_parallel() with {len(parallel_requests)} requests")
        paths = self.parallel_pathfinder.route_all_nets_parallel(parallel_requests)
        logger.info(f"PARALLEL_PF: parallel_pathfinder returned {len(paths) if paths else 0} paths: {list(paths.keys()) if paths else []}")
        
        # Convert back to RouteResult format
        results = {}
        for req in requests:
            path = paths.get(req.net_id, [])
            logger.info(f"PARALLEL_PF: Net {req.net_id} got path with {len(path)} nodes: {path[:5] if path else 'EMPTY'}")
            if path:
                # Convert path to RouteResult
                logger.info(f"PARALLEL_PF: Converting path to RouteResult for {req.net_id}")
                result = self._create_route_result_from_path(req, path)
                logger.info(f"PARALLEL_PF: RouteResult for {req.net_id}: success={result.success}, segments={len(result.segments)}")
                results[req.net_id] = result
            else:
                logger.warning(f"PARALLEL_PF: No path found for {req.net_id} - setting success=False")
                results[req.net_id] = RouteResult(net_id=req.net_id, success=False)
        
        total_time = time.time() - start_time
        successful_routes = len([r for r in results.values() if r.success])
        
        logger.info(f"Parallel PathFinder completed: {successful_routes}/{len(requests)} routes in {total_time:.2f}s")
        
        # Update statistics
        self.routing_stats['total_nets_routed'] += len(requests)
        self.routing_stats['successful_routes'] += successful_routes
        self.routing_stats['total_routing_time'] += total_time
        
        return results
    
    def _get_routing_endpoints_with_taps(self, request: RouteRequest) -> Tuple[Optional[int], Optional[int]]:
        """Get routing endpoints for F.Cu pad nodes (simplified system)"""
        
        # F.CU SIMPLIFIED: Route requests provide F.Cu pad node IDs directly
        # Examples: 'fcu_pad_pad_0_B07B15_000_234.1_229.7', 'fcu_pad_pad_1_B07B15_000_224.4_134.5'
        # These are directly routable - no more complex tap node lookup needed
        source_node_id = request.source_pad
        sink_node_id = request.sink_pad
        
        logger.debug(f"ENDPOINT_LOOKUP: F.Cu pad nodes: source='{source_node_id}', sink='{sink_node_id}'")
        
        # Simple node index lookup using GPU RRG
        if hasattr(self.gpu_rrg, 'get_node_idx'):
            source_idx = self.gpu_rrg.get_node_idx(source_node_id)
            sink_idx = self.gpu_rrg.get_node_idx(sink_node_id)
            
            # CRITICAL DEBUG: Check why lookup is failing
            if source_idx is None:
                logger.error(f"NODE_LOOKUP_FAIL: Source node '{source_node_id}' not found in GPU RRG")
                logger.error(f"NODE_LOOKUP_FAIL: Total nodes in GPU RRG: {self.gpu_rrg.num_nodes}")
                # Check if similar nodes exist
                sample_nodes = [nid for nid in self.gpu_rrg.node_ids[:10] if 'fcu_escape_v_pad_0' in nid]
                logger.error(f"NODE_LOOKUP_FAIL: Sample matching nodes: {sample_nodes}")
            else:
                logger.info(f"NODE_LOOKUP_SUCCESS: Source '{source_node_id}' -> index {source_idx}")
                
            if sink_idx is None:
                logger.error(f"NODE_LOOKUP_FAIL: Sink node '{sink_node_id}' not found in GPU RRG") 
                # Check if similar nodes exist
                sample_nodes = [nid for nid in self.gpu_rrg.node_ids[:10] if 'fcu_escape_v_pad_1' in nid]
                logger.error(f"NODE_LOOKUP_FAIL: Sample matching nodes: {sample_nodes}")
            else:
                logger.info(f"NODE_LOOKUP_SUCCESS: Sink '{sink_node_id}' -> index {sink_idx}")
        else:
            logger.error("NODE_LOOKUP_FAIL: gpu_rrg.get_node_idx method not available")
            source_idx = None
            sink_idx = None
        
        if source_idx is None or sink_idx is None:
            logger.warning(f"ENDPOINT_LOOKUP: Failed to find F.Cu pad nodes: source_idx={source_idx}, sink_idx={sink_idx}")
        else:
            logger.info(f"ENDPOINT_LOOKUP: Found F.Cu pad endpoints: source_idx={source_idx}, sink_idx={sink_idx}")
        
        return source_idx, sink_idx
    
    def _extract_coordinate_segments_from_path(self, path: List[int], net_id: str) -> List:
        """Extract REAL coordinate segments from PathFinder path"""
        from .rrg import RouteSegment
        
        segments = []
        if not path or len(path) < 2:
            logger.warning(f"COORD_EXTRACT [{net_id}]: Empty or too short path: {len(path) if path else 0} nodes")
            return segments
        
        try:
            logger.info(f"COORD_EXTRACT [{net_id}]: Starting extraction for {len(path)} nodes: {path[:5]}...")
            
            # DEBUG: Check GPU RRG state
            if not hasattr(self.gpu_rrg, 'node_positions'):
                logger.error(f"COORD_EXTRACT [{net_id}]: No node_positions attribute in GPU RRG")
                return segments
                
            node_positions = self.gpu_rrg.node_positions
            logger.info(f"COORD_EXTRACT [{net_id}]: node_positions shape: {node_positions.shape if hasattr(node_positions, 'shape') else 'no shape'}")
            
            if hasattr(node_positions, 'get'):  # CuPy array
                node_positions = node_positions.get()  # Convert to numpy
                logger.info(f"COORD_EXTRACT [{net_id}]: Converted CuPy to numpy, shape: {node_positions.shape}")
            
            # DEBUG: Check get_node_id method
            if not hasattr(self.gpu_rrg, 'get_node_id'):
                logger.error(f"COORD_EXTRACT [{net_id}]: No get_node_id method in GPU RRG")
                return segments
            
            # Extract coordinates for each node in the path
            path_coordinates = []
            path_layers = []
            
            for i, node_idx in enumerate(path):
                logger.debug(f"COORD_EXTRACT [{net_id}]: Processing node {i}/{len(path)}: idx={node_idx}")
                
                if node_idx >= len(node_positions):
                    logger.warning(f"COORD_EXTRACT [{net_id}]: Node index {node_idx} >= {len(node_positions)} (out of bounds)")
                    continue
                    
                # Get node coordinates
                try:
                    x, y = node_positions[node_idx]
                    logger.debug(f"COORD_EXTRACT [{net_id}]: Node {node_idx} coordinates: ({x}, {y})")
                except Exception as e:
                    logger.error(f"COORD_EXTRACT [{net_id}]: Failed to get coordinates for node {node_idx}: {e}")
                    continue
                
                # Get node ID to determine layer  
                try:
                    node_id = self.gpu_rrg.get_node_id(node_idx)
                    logger.debug(f"COORD_EXTRACT [{net_id}]: Node {node_idx} ID: '{node_id}'")
                    
                    if not node_id:
                        logger.warning(f"COORD_EXTRACT [{net_id}]: Node {node_idx} returned empty ID")
                        continue
                        
                except Exception as e:
                    logger.error(f"COORD_EXTRACT [{net_id}]: Failed to get node ID for {node_idx}: {e}")
                    continue
                
                # F.CU ARCHITECTURE: Extract layer from new node ID format
                layer_name = "Unknown"
                if 'fcu_pad_' in node_id or 'fcu_tap_' in node_id:
                    layer_name = "F.Cu"  # F.Cu pad/tap nodes
                elif 'rail_h_' in node_id:
                    # Horizontal rail: rail_h_{x_idx}_{y_idx}_{layer_id}
                    parts = node_id.split('_')
                    if len(parts) >= 4:
                        try:
                            layer_id = int(parts[-1])
                            if layer_id == 1:
                                layer_name = "In1.Cu"
                            elif layer_id == 3:  
                                layer_name = "In3.Cu"
                            elif layer_id == 5:
                                layer_name = "B.Cu"
                            else:
                                layer_name = f"In{layer_id}.Cu"
                        except ValueError:
                            logger.warning(f"COORD_EXTRACT [{net_id}]: Invalid layer_id in rail_h node: {node_id}")
                            layer_name = "In1.Cu"
                    else:
                        logger.warning(f"COORD_EXTRACT [{net_id}]: Invalid rail_h format: {node_id}")
                        layer_name = "In1.Cu"
                elif 'rail_v_' in node_id:
                    # Vertical rail: rail_v_{x_idx}_{y_idx}_{layer_id} 
                    parts = node_id.split('_')
                    if len(parts) >= 4:
                        try:
                            layer_id = int(parts[-1])
                            if layer_id == 2:
                                layer_name = "In2.Cu"
                            elif layer_id == 4:
                                layer_name = "In4.Cu" 
                            else:
                                layer_name = f"In{layer_id}.Cu"
                        except ValueError:
                            logger.warning(f"COORD_EXTRACT [{net_id}]: Invalid layer_id in rail_v node: {node_id}")
                            layer_name = "In2.Cu"
                    else:
                        logger.warning(f"COORD_EXTRACT [{net_id}]: Invalid rail_v format: {node_id}")
                        layer_name = "In2.Cu"
                else:
                    logger.warning(f"COORD_EXTRACT [{net_id}]: Unknown node type: {node_id}")
                    layer_name = "In1.Cu"  # Default
                
                path_coordinates.append((float(x), float(y)))
                path_layers.append(layer_name)
                
                logger.info(f"COORD_EXTRACT [{net_id}]: Node {i}: {node_id} -> ({x:.3f}, {y:.3f}) on {layer_name}")
            
            logger.info(f"COORD_EXTRACT [{net_id}]: Extracted {len(path_coordinates)} valid coordinates")
            
            if len(path_coordinates) < 2:
                logger.warning(f"COORD_EXTRACT [{net_id}]: Too few valid coordinates: {len(path_coordinates)}")
                return segments
            
            # MANHATTAN GEOMETRY: Create proper L-shaped routes from path coordinates
            grid_pitch = 0.4  # mm grid spacing
            
            for i in range(len(path_coordinates) - 1):
                start_x, start_y = path_coordinates[i]
                end_x, end_y = path_coordinates[i + 1]
                start_layer = path_layers[i]
                end_layer = path_layers[i + 1]
                
                logger.debug(f"SEGMENT_GEN [{net_id}]: Step {i}: ({start_x:.3f},{start_y:.3f},{start_layer}) -> ({end_x:.3f},{end_y:.3f},{end_layer})")
                
                # SNAP TO GRID: Force all coordinates to 0.4mm grid
                orig_start_x, orig_start_y = start_x, start_y
                orig_end_x, orig_end_y = end_x, end_y
                start_x = round(start_x / grid_pitch) * grid_pitch
                start_y = round(start_y / grid_pitch) * grid_pitch
                end_x = round(end_x / grid_pitch) * grid_pitch
                end_y = round(end_y / grid_pitch) * grid_pitch
                
                logger.debug(f"SEGMENT_GEN [{net_id}]: Grid snap: ({orig_start_x:.3f},{orig_start_y:.3f}) -> ({start_x:.3f},{start_y:.3f})")
                logger.debug(f"SEGMENT_GEN [{net_id}]: Grid snap: ({orig_end_x:.3f},{orig_end_y:.3f}) -> ({end_x:.3f},{end_y:.3f})")
                
                # MANHATTAN ROUTING: Break diagonal moves into L-shaped paths
                dx = end_x - start_x
                dy = end_y - start_y
                
                logger.debug(f"SEGMENT_GEN [{net_id}]: Movement: dx={dx:.3f}, dy={dy:.3f}")
                
                if abs(dx) > 0.001 and abs(dy) > 0.001:
                    # DIAGONAL MOVE: Create L-shaped route
                    
                    # Determine layer assignment based on routing direction
                    if abs(dx) > abs(dy):
                        # Horizontal-dominant: horizontal first, then vertical
                        h_layer = 'In1.Cu'  # Horizontal layer
                        v_layer = 'In2.Cu'  # Vertical layer
                        
                        # 1. Horizontal segment
                        if abs(dx) > 0.001:
                            h_segment = RouteSegment(
                                start_x=float(start_x),
                                start_y=float(start_y),
                                end_x=float(end_x),    # Move horizontally
                                end_y=float(start_y),  # Keep Y constant
                                layer=h_layer,
                                width=0.102,
                                net_id=net_id
                            )
                            segments.append(h_segment)
                            logger.info(f"SEGMENT_ADD [{net_id}]: HORIZONTAL ({start_x:.3f},{start_y:.3f}) -> ({end_x:.3f},{start_y:.3f}) on {h_layer}")
                        
                        # 2. Via at corner
                        if start_layer != v_layer:
                            # Store via info (will be handled by router)
                            logger.info(f"SEGMENT_ADD [{net_id}]: VIA at ({end_x:.3f},{start_y:.3f}) from {h_layer} to {v_layer}")
                        
                        # 3. Vertical segment  
                        if abs(dy) > 0.001:
                            v_segment = RouteSegment(
                                start_x=float(end_x),    # Start from corner
                                start_y=float(start_y),
                                end_x=float(end_x),      # Keep X constant
                                end_y=float(end_y),      # Move vertically
                                layer=v_layer,
                                width=0.102,
                                net_id=net_id
                            )
                            segments.append(v_segment)
                            logger.info(f"SEGMENT_ADD [{net_id}]: VERTICAL ({end_x:.3f},{start_y:.3f}) -> ({end_x:.3f},{end_y:.3f}) on {v_layer}")
                    
                    else:
                        # Vertical-dominant: vertical first, then horizontal
                        v_layer = 'In2.Cu'  # Vertical layer
                        h_layer = 'In1.Cu'  # Horizontal layer
                        
                        # 1. Vertical segment
                        if abs(dy) > 0.001:
                            v_segment = RouteSegment(
                                start_x=float(start_x),
                                start_y=float(start_y),
                                end_x=float(start_x),    # Keep X constant
                                end_y=float(end_y),      # Move vertically
                                layer=v_layer,
                                width=0.102,
                                net_id=net_id
                            )
                            segments.append(v_segment)
                            logger.debug(f"VERTICAL: ({start_x:.3f},{start_y:.3f}) -> ({start_x:.3f},{end_y:.3f}) on {v_layer}")
                        
                        # 2. Via at corner
                        if v_layer != h_layer:
                            logger.debug(f"VIA: at ({start_x:.3f},{end_y:.3f}) from {v_layer} to {h_layer}")
                        
                        # 3. Horizontal segment
                        if abs(dx) > 0.001:
                            h_segment = RouteSegment(
                                start_x=float(start_x),  # Start from corner
                                start_y=float(end_y),
                                end_x=float(end_x),      # Move horizontally
                                end_y=float(end_y),      # Keep Y constant
                                layer=h_layer,
                                width=0.102,
                                net_id=net_id
                            )
                            segments.append(h_segment)
                            logger.debug(f"HORIZONTAL: ({start_x:.3f},{end_y:.3f}) -> ({end_x:.3f},{end_y:.3f}) on {h_layer}")
                
                elif abs(dx) > 0.001:
                    # PURE HORIZONTAL MOVE
                    layer = 'In1.Cu'  # Horizontal layer only
                    segment = RouteSegment(
                        start_x=float(start_x),
                        start_y=float(start_y),
                        end_x=float(end_x),
                        end_y=float(start_y),  # Force horizontal
                        layer=layer,
                        width=0.102,
                        net_id=net_id
                    )
                    segments.append(segment)
                    logger.info(f"SEGMENT_ADD [{net_id}]: PURE HORIZONTAL ({start_x:.3f},{start_y:.3f}) -> ({end_x:.3f},{start_y:.3f}) on {layer}")
                
                elif abs(dy) > 0.001:
                    # PURE VERTICAL MOVE
                    layer = 'In2.Cu'  # Vertical layer only
                    segment = RouteSegment(
                        start_x=float(start_x),
                        start_y=float(start_y),
                        end_x=float(start_x),    # Force vertical
                        end_y=float(end_y),
                        layer=layer,
                        width=0.102,
                        net_id=net_id
                    )
                    segments.append(segment)
                    logger.info(f"SEGMENT_ADD [{net_id}]: PURE VERTICAL ({start_x:.3f},{start_y:.3f}) -> ({start_x:.3f},{end_y:.3f}) on {layer}")
                else:
                    logger.warning(f"SEGMENT_GEN [{net_id}]: Step {i} has no movement (dx={dx:.3f}, dy={dy:.3f}) - SKIPPING")
            
            logger.info(f"COORD_EXTRACT [{net_id}]: FINAL RESULT: Created {len(segments)} segments from {len(path_coordinates)} coordinates")
        
        except Exception as e:
            logger.error(f"COORD_EXTRACT [{net_id}]: EXCEPTION during extraction: {e}")
            import traceback
            logger.error(f"COORD_EXTRACT [{net_id}]: TRACEBACK: {traceback.format_exc()}")
        
        return segments
    
    def _create_route_result_from_path(self, request: RouteRequest, path: List[int]) -> RouteResult:
        """Create RouteResult from node path with REAL coordinate segments"""
        
        if not path:
            return RouteResult(net_id=request.net_id, success=False)
        
        # Convert node indices to IDs
        node_ids = []
        for node_idx in path:
            if hasattr(self.gpu_rrg, 'get_node_id'):
                node_id = self.gpu_rrg.get_node_id(node_idx)
                if node_id:
                    node_ids.append(node_id)
        
        # CREATE REAL COORDINATE SEGMENTS from PathFinder path
        logger.warning(f"COORDINATE_DEBUG [{request.net_id}]: About to extract segments from path of {len(path)} nodes")
        segments = self._extract_coordinate_segments_from_path(path, request.net_id)
        logger.warning(f"COORDINATE_DEBUG [{request.net_id}]: Extraction returned {len(segments)} segments")
        
        # Calculate basic metrics
        cost = len(path) * 1.0  # Simple cost estimate
        length_mm = len(path) * 0.4  # Assume 0.4mm per step
        via_count = 0  # Would need layer analysis
        
        return RouteResult(
            net_id=request.net_id,
            success=True,
            path=node_ids,
            edges=[],  # Could be populated from path
            segments=segments,  # REAL coordinate segments for visualization
            cost=cost,
            length_mm=length_mm,
            via_count=via_count
        )
    
    def get_routing_statistics(self) -> Dict:
        """Get routing performance statistics"""
        return self.routing_stats.copy()
    
    def cleanup(self):
        """Clean up GPU resources"""
        if hasattr(self.gpu_rrg, 'cleanup'):
            self.gpu_rrg.cleanup()