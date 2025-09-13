"""
True Parallel PathFinder Implementation
Routes all nets simultaneously, then negotiates congestion iteratively
"""

import cupy as cp
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParallelRouteRequest:
    """Request for parallel routing"""
    net_id: str
    source_idx: int
    sink_idx: int
    priority: float = 1.0

class TrueParallelPathFinder:
    """True PathFinder - routes all nets simultaneously with negotiated congestion"""
    
    def __init__(self, gpu_rrg):
        self.gpu_rrg = gpu_rrg
        self.max_iterations = 50
        self.pres_fac_base = 1.0
        self.pres_fac_mult = 1.5
        self.acc_fac = 1.0
        self.progress_callback = None  # Callback for GUI updates
        
    def route_all_nets_parallel(self, requests: List[ParallelRouteRequest]) -> Dict[str, List[int]]:
        """Route all nets simultaneously using true PathFinder algorithm with timeout detection"""
        
        logger.info(f"Starting TRUE PARALLEL PathFinder for {len(requests)} nets")
        
        # Handle empty requests
        if not requests:
            logger.warning("No route requests provided to parallel PathFinder")
            return {}
        
        # Initialize congestion tracking
        self._initialize_congestion_state(len(requests))
        
        # Initialize sink positions for heuristic guidance
        self._initialize_sink_positions(requests)
        
        iteration = 0
        routes = {}
        start_time = time.time()
        last_progress_time = start_time
        max_routing_time = 300  # 5 minutes timeout
        iteration_timeout = 30  # 30 seconds per iteration
        
        while iteration < self.max_iterations:
            iteration_start = time.time()
            
            # Check global timeout
            if time.time() - start_time > max_routing_time:
                logger.warning(f"PathFinder global timeout after {max_routing_time}s - stopping at iteration {iteration + 1}")
                if self.progress_callback:
                    self.progress_callback({
                        'type': 'timeout',
                        'message': f"Routing timeout after {max_routing_time}s"
                    })
                break
            logger.info(f"PathFinder iteration {iteration + 1}/{self.max_iterations}")
            
            # Update GUI progress if callback available
            if self.progress_callback:
                self.progress_callback({
                    'type': 'iteration_start',
                    'iteration': iteration + 1,
                    'max_iterations': self.max_iterations,
                    'status': f"Routing iteration {iteration + 1}..."
                })
            
            # PHASE 1: Route ALL nets simultaneously (ignoring conflicts)
            iteration_routes = self._route_all_nets_simultaneously(requests)
            
            # Check iteration timeout
            iteration_time = time.time() - iteration_start
            if iteration_time > iteration_timeout:
                logger.warning(f"Iteration {iteration + 1} timeout after {iteration_time:.1f}s (max {iteration_timeout}s)")
                if self.progress_callback:
                    self.progress_callback({
                        'type': 'iteration_timeout',
                        'iteration': iteration + 1,
                        'time': iteration_time
                    })
                # Continue with current routes but increase timeout for next iteration
                iteration_timeout = min(60, iteration_timeout * 1.5)
            
            # PHASE 2: Calculate congestion from all routes
            congestion_map = self._calculate_congestion(iteration_routes)
            
            # PHASE 3: Update costs based on congestion
            self._update_congestion_costs(congestion_map, iteration)
            
            # Update GUI with routing results and timing
            if self.progress_callback:
                successful_routes = len([r for r in iteration_routes.values() if r])
                failed_routes = len(iteration_routes) - successful_routes
                congested_edges = int(cp.sum(congestion_map > 1)) if hasattr(congestion_map, 'sum') else 0
                total_edges = len(congestion_map) if hasattr(congestion_map, '__len__') else 1
                
                self.progress_callback({
                    'type': 'routing_update',
                    'successful_routes': successful_routes,
                    'failed_routes': failed_routes,
                    'congested_edges': congested_edges,
                    'total_edges': total_edges,
                    'iteration_time': iteration_time,
                    'total_time': time.time() - start_time
                })
            
            # PHASE 4: Check convergence
            if self._check_convergence(congestion_map):
                logger.info(f"PathFinder converged after {iteration + 1} iterations")
                routes = iteration_routes
                
                if self.progress_callback:
                    self.progress_callback({
                        'type': 'convergence',
                        'iteration': iteration + 1,
                        'status': f"Converged after {iteration + 1} iterations!"
                    })
                break
                
            routes = iteration_routes
            iteration += 1
        
        successful_count = len([r for r in routes.values() if r])
        logger.info(f"PathFinder completed: {successful_count}/{len(requests)} successful routes")
        
        # Final update
        if self.progress_callback:
            self.progress_callback({
                'type': 'completion',
                'successful_routes': successful_count,
                'total_routes': len(requests),
                'converged': iteration < self.max_iterations
            })
        
        return routes
    
    def _route_all_nets_simultaneously(self, requests: List[ParallelRouteRequest]) -> Dict[str, List[int]]:
        """Route all nets in parallel - the core of PathFinder"""
        
        # Handle empty requests
        if not requests:
            logger.warning("No route requests provided to parallel PathFinder")
            return {}
        
        # Create GPU arrays for batch processing
        num_nets = len(requests)
        source_indices = cp.array([req.source_idx for req in requests], dtype=cp.int32)
        sink_indices = cp.array([req.sink_idx for req in requests], dtype=cp.int32)
        
        # Debug source/sink indices
        logger.info(f"PathFinder source indices range: {int(cp.min(source_indices).get())} to {int(cp.max(source_indices).get())}")
        logger.info(f"PathFinder sink indices range: {int(cp.min(sink_indices).get())} to {int(cp.max(sink_indices).get())}")
        logger.info(f"Total routing nodes available: {self.gpu_rrg.num_nodes}")
        
        # F.CU SIMPLIFIED: Validate that sources and sinks are F.Cu pad nodes (not tap nodes)
        if hasattr(self.gpu_rrg, 'fcu_pad_nodes'):
            fcu_pad_indices = set(self.gpu_rrg.fcu_pad_nodes.values())
            source_pad_count = sum(1 for idx in source_indices if int(idx) in fcu_pad_indices)
            sink_pad_count = sum(1 for idx in sink_indices if int(idx) in fcu_pad_indices)
            logger.info(f"F.CU PAD VALIDATION: {source_pad_count}/{len(source_indices)} sources are F.Cu pad nodes")
            logger.info(f"F.CU PAD VALIDATION: {sink_pad_count}/{len(sink_indices)} sinks are F.Cu pad nodes")
            
            # Debug: Show specific source/sink F.Cu pad node details
            for i, (src_idx, sink_idx) in enumerate(zip(source_indices[:3], sink_indices[:3])):
                src_is_pad = int(src_idx) in fcu_pad_indices
                sink_is_pad = int(sink_idx) in fcu_pad_indices
                req = requests[i]
                logger.info(f"  Net {req.net_id}: source {int(src_idx)} is_fcu_pad={src_is_pad}, sink {int(sink_idx)} is_fcu_pad={sink_is_pad}")
            
            if source_pad_count == 0 and sink_pad_count == 0:
                logger.warning("NO F.CU PAD NODES IN ROUTING REQUESTS - using general routing nodes")
        else:
            logger.info("F.CU SIMPLIFIED: No fcu_pad_nodes attribute - using general routing approach")
        
        # Validate indices are within bounds
        current_num_nodes = self.gpu_rrg.num_nodes
        max_source = cp.max(source_indices).get() if len(source_indices) > 0 else -1
        max_sink = cp.max(sink_indices).get() if len(sink_indices) > 0 else -1
        logger.info(f"BOUNDS CHECK: current_num_nodes={current_num_nodes}, max_source={max_source}, max_sink={max_sink}")
        
        invalid_sources = cp.sum((source_indices < 0) | (source_indices >= current_num_nodes)).get()
        invalid_sinks = cp.sum((sink_indices < 0) | (sink_indices >= current_num_nodes)).get()
        if invalid_sources > 0 or invalid_sinks > 0:
            logger.error(f"Invalid indices: {invalid_sources} invalid sources, {invalid_sinks} invalid sinks (num_nodes={current_num_nodes})")
            return {}
        
        # Initialize distance arrays for all nets
        # Get current num_nodes AFTER tap nodes have been added
        num_nodes = self.gpu_rrg.num_nodes
        logger.info(f"ARRAY SIZING: Creating PathFinder arrays for {num_nodes} nodes (after tap node addition)")
        
        distances = cp.full((num_nets, num_nodes), cp.inf, dtype=cp.float32)
        parents = cp.full((num_nets, num_nodes), -1, dtype=cp.int32)
        visited = cp.zeros((num_nets, num_nodes), dtype=cp.bool_)
        
        # Set source distances to 0
        net_indices = cp.arange(num_nets)
        distances[net_indices, source_indices] = 0.0
        
        # Parallel wavefront expansion for all nets
        active_frontiers = cp.zeros((num_nets, num_nodes), dtype=cp.bool_)
        active_frontiers[net_indices, source_indices] = True
        
        # Debug initial frontier setup
        initial_active_count = int(cp.sum(active_frontiers).get())
        logger.info(f"Initial active frontier nodes: {initial_active_count} (should be {num_nets})")
        if initial_active_count != num_nets:
            logger.error("Frontier initialization failed - some sources not activated")
        
        max_steps = 1000  # Reduced for efficiency - heuristic guidance should find paths faster
        step = 0
        last_progress_step = 0
        step_start_time = time.time()
        
        while cp.any(active_frontiers) and step < max_steps:
            # Debug main loop conditions for first few steps
            if step <= 5:
                active_count = int(cp.sum(active_frontiers).get())
                logger.error(f"MAIN LOOP Step {step}: {active_count} active frontiers, max_steps={max_steps}")
            
            # EARLY TERMINATION: Check if we're close to sinks
            if step > 10 and step % 50 == 0:  # Check every 50 steps after step 10
                if self._check_proximity_termination(active_frontiers, distances, step):
                    logger.info(f"EARLY_TERMINATION: Stopping at step {step} - close to sinks")
                    break
            
            # Expand all active frontiers simultaneously
            self._expand_all_frontiers_parallel(active_frontiers, distances, parents, visited, step)
            
            # Check if any sinks reached
            sinks_reached = visited[net_indices, sink_indices]
            routed_now = int(cp.sum(sinks_reached).get())
            
            # CRITICAL DEBUG: Check if sink nodes are actually being expanded
            if step <= 3:
                for i in range(min(3, num_nets)):
                    sink_idx = int(sink_indices[i])
                    if sink_idx < visited.shape[1]:
                        is_sink_visited = bool(visited[i, sink_idx])
                        sink_distance = float(distances[i, sink_idx])
                        is_sink_frontier = bool(active_frontiers[i, sink_idx]) if sink_idx < active_frontiers.shape[1] else False
                        logger.error(f"  SINK {sink_idx}: visited={is_sink_visited}, distance={sink_distance:.1f}, frontier={is_sink_frontier}")
                        
                        # DEBUG: Critical check - are the sink node's neighbors bidirectionally connected?
                        if step == 2 and sink_idx in [218496, 218512, 218528]:  # Focus on specific problematic sinks
                            csr_matrix = self.gpu_rrg.adjacency_csr if hasattr(self.gpu_rrg, 'adjacency_csr') else self.gpu_rrg.adjacency_matrix
                            if hasattr(csr_matrix, 'indptr') and sink_idx < len(csr_matrix.indptr) - 1:
                                start_idx = int(csr_matrix.indptr[sink_idx])
                                end_idx = int(csr_matrix.indptr[sink_idx + 1])
                                neighbor_indices = csr_matrix.indices[start_idx:end_idx]
                                if len(neighbor_indices) > 0:
                                    neighbor_distances = [float(distances[i, int(nb)]) if int(nb) < distances.shape[1] else 999 for nb in neighbor_indices[:3]]
                                    logger.error(f"    SINK {sink_idx} neighbors {neighbor_indices[:3].tolist()}: distances {neighbor_distances}")
                                    
                                    # CRITICAL: Check if the neighbor nodes point BACK to the sink
                                    first_neighbor = int(neighbor_indices[0])
                                    if first_neighbor < len(csr_matrix.indptr) - 1:
                                        nb_start = int(csr_matrix.indptr[first_neighbor])
                                        nb_end = int(csr_matrix.indptr[first_neighbor + 1])
                                        nb_neighbors = csr_matrix.indices[nb_start:nb_end]
                                        has_back_connection = sink_idx in nb_neighbors
                                        logger.error(f"      BIDIRECTIONAL: neighbor {first_neighbor} connects back to sink {sink_idx}: {has_back_connection}")
                                        
                                        # CRITICAL: Check what layers these nodes are on
                                        if hasattr(self.gpu_rrg, 'node_layers') and len(self.gpu_rrg.node_layers) > max(sink_idx, first_neighbor):
                                            sink_layer = int(self.gpu_rrg.node_layers[sink_idx]) if hasattr(self.gpu_rrg.node_layers, '__getitem__') else -1
                                            neighbor_layer = int(self.gpu_rrg.node_layers[first_neighbor]) if hasattr(self.gpu_rrg.node_layers, '__getitem__') else -1
                                            logger.error(f"      LAYERS: sink {sink_idx} on layer {sink_layer}, neighbor {first_neighbor} on layer {neighbor_layer}")
            
            if step <= 5:
                logger.error(f"  After expansion: {routed_now}/{num_nets} sinks reached")
                
                # DEBUG: Check progress on sink nodes being reached at any step
                if step <= 5:
                    sample_sink_distances = []
                    for i in range(min(3, num_nets)):  # First 3 nets
                        sink_idx = int(sink_indices[i])
                        if sink_idx < distances.shape[1]:
                            sink_dist = float(distances[i, sink_idx])
                            sample_sink_distances.append(sink_dist)
                    logger.error(f"  SINK DISTANCE PROGRESS Step {step}: {sample_sink_distances}")
                
                if step == 1:
                    # CRITICAL DEBUG: Compare source vs sink tap connectivity
                    source_list = source_indices.get().tolist()[:3]
                    sink_list = sink_indices.get().tolist()[:3]
                    logger.error(f"  SOURCE vs SINK: sources={source_list}, sinks={sink_list}")
                    
                    # Check what nodes the SOURCE taps connect to (these work!)
                    csr_matrix = self.gpu_rrg.adjacency_csr if hasattr(self.gpu_rrg, 'adjacency_csr') else self.gpu_rrg.adjacency_matrix
                    for i, src_idx in enumerate(source_list):
                        if src_idx < len(csr_matrix.indptr) - 1:
                            start_idx = int(csr_matrix.indptr[src_idx])
                            end_idx = int(csr_matrix.indptr[src_idx + 1])
                            neighbor_indices = csr_matrix.indices[start_idx:end_idx]
                            logger.error(f"  SOURCE {src_idx} neighbors: {neighbor_indices[:3].tolist()}")
                    
                    # Check what nodes the SINK taps connect to
                    for i, sink_idx in enumerate(sink_list):
                        if sink_idx < len(csr_matrix.indptr) - 1:
                            start_idx = int(csr_matrix.indptr[sink_idx])
                            end_idx = int(csr_matrix.indptr[sink_idx + 1])
                            neighbor_indices = csr_matrix.indices[start_idx:end_idx]
                            logger.error(f"  SINK {sink_idx} neighbors: {neighbor_indices[:3].tolist()}")
                    
                    logger.error(f"  Problem: Source taps connect to different nodes than sink taps!")
                    
                    # CONNECTIVITY TEST: Check if source neighbors can reach sink neighbors through grid
                    source_neighbors = set()
                    sink_neighbors = set()
                    
                    # Collect all source neighbor nodes  
                    for src_idx in source_list:
                        if src_idx < len(csr_matrix.indptr) - 1:
                            start_idx = int(csr_matrix.indptr[src_idx])
                            end_idx = int(csr_matrix.indptr[src_idx + 1])
                            neighbors = csr_matrix.indices[start_idx:end_idx]
                            source_neighbors.update(neighbors.tolist())
                    
                    # Collect all sink neighbor nodes
                    for sink_idx in sink_list:
                        if sink_idx < len(csr_matrix.indptr) - 1:
                            start_idx = int(csr_matrix.indptr[sink_idx])
                            end_idx = int(csr_matrix.indptr[sink_idx + 1])
                            neighbors = csr_matrix.indices[start_idx:end_idx]
                            sink_neighbors.update(neighbors.tolist())
                    
                    # Check for overlap
                    overlap = source_neighbors.intersection(sink_neighbors)
                    logger.error(f"  CONNECTIVITY: {len(source_neighbors)} source neighbors, {len(sink_neighbors)} sink neighbors, {len(overlap)} shared")
                    if overlap:
                        logger.error(f"  SHARED NODES: {list(overlap)[:3]} (PathFinder should find paths!)")
                    else:
                        logger.error(f"  NO SHARED NODES: Source and sink islands are disconnected")
                        logger.error(f"  ABORTING PATHFINDER: No point expanding if sources and sinks can't connect")
                        # Return early - no point in massive expansion if disconnected
                        return {req.net_id: [] for req in requests}
                    
                    # Check if any of the expected sink nodes are marked as visited
                    visited_sinks = 0
                    for i, sink_idx in enumerate(sink_list[:3]):  # Check first 3
                        if visited[i, sink_idx]:
                            visited_sinks += 1
                            logger.error(f"  VISITED: Sink {sink_idx} for net {i} is already marked visited!")
                    logger.error(f"  VISITED DEBUG: {visited_sinks}/{min(3, len(sink_list))} sink nodes are visited")
                
            if cp.all(sinks_reached):
                logger.info(f"All nets routed in {step} wavefront steps")
                break
                
            # Check if we have active frontiers for next iteration
            active_count_after = int(cp.sum(active_frontiers).get())
            if step <= 5:
                logger.error(f"  Active frontiers after expansion: {active_count_after}")
                
            if active_count_after == 0:
                logger.error(f"TERMINATION: No active frontiers remaining at step {step} (routed {routed_now}/{num_nets})")
                break
            
            # Progress monitoring every 100 steps
            if step > 0 and step % 100 == 0:
                routed_count = int(cp.sum(sinks_reached).get())
                step_time = time.time() - step_start_time
                logger.info(f"Wavefront step {step}: {routed_count}/{num_nets} nets routed in {step_time:.2f}s")
                
                # Check for stuck progress
                if routed_count == last_progress_step:
                    logger.warning(f"No routing progress in last 100 steps (step {step})")
                    # Don't break - some nets might still be expanding
                
                last_progress_step = routed_count
                step_start_time = time.time()
                
            step += 1
            
        if step >= max_steps:
            logger.warning(f"Wavefront expansion reached maximum steps ({max_steps}) - some nets may be unrouted")
        
        # Reconstruct paths for all nets
        routes = {}
        for i, req in enumerate(requests):
            net_idx = i  # Net index matches request index (from cp.arange(num_nets))
            sink_reached = visited[net_idx, req.sink_idx]
            
            if sink_reached:
                path = self._reconstruct_path_parallel(net_idx, req.source_idx, req.sink_idx, parents)
                routes[req.net_id] = path
                logger.info(f"SUCCESS: Net {req.net_id} routed with {len(path)} nodes")
            else:
                routes[req.net_id] = []  # Failed to route
                logger.warning(f"FAILED: Net {req.net_id} - sink {req.sink_idx} not reached")
                
        return routes
    
    def _expand_all_frontiers_parallel(self, frontiers, distances, parents, visited, step):
        """Expand wavefronts for all nets simultaneously using GPU CSR operations"""
        
        # Check for empty frontiers first
        if not cp.any(frontiers):
            logger.warning("No active frontiers - all nets may be routed or stuck")
            return
            
        # Find minimum distance nodes in each frontier
        frontier_distances = cp.where(frontiers, distances, cp.inf)
        
        # Handle case where some nets have no active frontier
        try:
            min_distances = cp.min(frontier_distances, axis=1, keepdims=True)
            
            # DEBUG: Log shapes and values for first few steps
            if step <= 2:
                logger.error(f"SHAPE DEBUG Step {step}: frontier_distances {frontier_distances.shape}")
                logger.error(f"SHAPE DEBUG Step {step}: min_distances {min_distances.shape}")
                logger.error(f"SHAPE DEBUG Step {step}: distances {distances.shape}")
                # Sample values from first net
                if frontiers.shape[0] > 0:
                    sample_frontier = frontiers[0]
                    sample_distances = distances[0]
                    frontier_indices = cp.where(sample_frontier)[0][:3]  # First 3 frontier nodes
                    if len(frontier_indices) > 0:
                        logger.error(f"SHAPE DEBUG: Net 0 has {int(cp.sum(sample_frontier))} frontier nodes")
                        logger.error(f"SHAPE DEBUG: Net 0 frontier sample distances: {sample_distances[frontier_indices].tolist()}")
                        logger.error(f"SHAPE DEBUG: Net 0 min_distance: {float(min_distances[0, 0])}")
        except ValueError as e:
            logger.warning(f"Error computing min distances: {e} - skipping expansion")
            return
        
        # Debug: Check shapes to prevent broadcasting errors
        if min_distances.shape[0] != distances.shape[0]:
            logger.warning(f"Shape mismatch: min_distances {min_distances.shape} vs distances {distances.shape}")
            # If we have fewer minimum distances than nets, pad with inf
            if min_distances.shape[0] < distances.shape[0]:
                padding = cp.full((distances.shape[0] - min_distances.shape[0], 1), cp.inf, dtype=cp.float32)
                min_distances = cp.vstack([min_distances, padding])
            else:
                # Reshape min_distances to match number of nets
                min_distances = min_distances[:distances.shape[0]].reshape(distances.shape[0], -1)
        
        # Select nodes at minimum distance for each net (with tolerance for floating point precision)
        tolerance = 1e-6
        
        # Ensure min_distances broadcasts correctly with distances [num_nets, num_nodes]
        if min_distances.shape[1] == 1:
            # min_distances is [num_nets, 1], broadcast to [num_nets, num_nodes]
            min_distances_broadcasted = cp.broadcast_to(min_distances, distances.shape)
        else:
            min_distances_broadcasted = min_distances
        
        # INTELLIGENT SELECTION: Use heuristic guidance instead of just minimum distance
        current_nodes = self._select_nodes_with_heuristic_guidance(
            frontiers, distances, min_distances_broadcasted, tolerance, step
        )
        
        # CRITICAL FIX: Mark selected nodes as visited (core PathFinder requirement)
        # Without this, sink nodes are never detected as reached!
        visited |= current_nodes
        
        # DEBUG: Check if node selection is failing
        total_frontier = int(cp.sum(frontiers).get())
        total_selected = int(cp.sum(current_nodes).get())
        
        # Always log for first 10 steps to debug issues
        if step <= 10:
            logger.error(f"STEP {step}: {total_frontier} frontier -> {total_selected} selected nodes")
            
            if total_selected == 0:
                logger.error(f"CRITICAL: ZERO NODES SELECTED FOR EXPANSION at step {step}!")
                
                # Sample detailed debugging for first net
                if total_frontier > 0:
                    sample_net = 0
                    frontier_mask = frontiers[sample_net]
                    if cp.any(frontier_mask):
                        frontier_indices = cp.where(frontier_mask)[0][:5]  # First 5 frontier nodes
                        sample_distances = distances[sample_net, frontier_indices]
                        sample_min = min_distances[sample_net, 0] if min_distances.shape[1] > 0 else "INVALID"
                        sample_min_broadcast = min_distances_broadcasted[sample_net, frontier_indices]
                        
                        logger.error(f"  Net 0 frontier distances: {sample_distances.tolist()}")
                        logger.error(f"  Net 0 min_distance: {sample_min}")  
                        logger.error(f"  Broadcasted mins: {sample_min_broadcast.tolist()}")
                        logger.error(f"  Differences: {cp.abs(sample_distances - sample_min_broadcast).tolist()}")
                        logger.error(f"  Tolerance check: {(cp.abs(sample_distances - sample_min_broadcast) < tolerance).tolist()}")
                        logger.error(f"  Tolerance value: {tolerance}")
                        
                        # Check if distances are all inf
                        all_distances = distances[sample_net]
                        finite_distances = cp.isfinite(all_distances)
                        logger.error(f"  {int(cp.sum(finite_distances).get())}/{all_distances.shape[0]} distances are finite")
                        
                        if cp.any(finite_distances):
                            finite_dist_vals = all_distances[finite_distances][:10]
                            logger.error(f"  Sample finite distances: {finite_dist_vals.tolist()}")
                
                # Check if this is termination condition
                if total_frontier == 0:
                    logger.error(f"  TERMINATION: No frontier nodes remaining at step {step}")
                    # Will be handled by the main while loop condition
            else:
                logger.info(f"  Step {step} OK: Selected {total_selected} nodes from {total_frontier} frontier nodes")
        
        # CRITICAL DEBUG: Track what happens to frontiers during expansion
        frontier_before_visited = int(cp.sum(frontiers).get())
        frontier_before_removal = 0
        
        # Mark selected nodes as visited
        visited |= current_nodes
        frontier_before_removal = int(cp.sum(frontiers).get())
        frontiers &= ~current_nodes  # Remove from frontier
        frontier_after_removal = int(cp.sum(frontiers).get())
        
        # CRITICAL DEBUG: Track frontier changes
        if step <= 5:
            logger.error(f"  FRONTIER TRACKING: before_visited={frontier_before_visited}, before_removal={frontier_before_removal}, after_removal={frontier_after_removal}")
            logger.error(f"  Selected nodes removed: {total_selected}, Remaining: {frontier_after_removal}")
        
        # For each net and each current node, expand neighbors
        # Use the appropriate adjacency matrix (GPU uses adjacency_csr, CPU uses adjacency_matrix)
        if hasattr(self.gpu_rrg, 'adjacency_csr') and self.gpu_rrg.adjacency_csr is not None:
            adj_csr = self.gpu_rrg.adjacency_csr
        else:
            adj_csr = self.gpu_rrg.adjacency_matrix
        
        # Debug: Check if we have current nodes to expand
        total_current = int(cp.sum(current_nodes).get())
        if total_current == 0 and step < 10:  # Only log for first few steps
            logger.warning(f"No current nodes to expand in step {step}")
        
        # This is the tricky part - parallel neighbor expansion for multiple nets
        nodes_expanded = 0
        neighbors_found = 0
        tap_nodes_expanded = 0
        tap_neighbors_found = 0
        
        for net_idx in range(current_nodes.shape[0]):
            current_net_nodes = cp.where(current_nodes[net_idx])[0].get()
            
            for node_idx in current_net_nodes:
                node_idx = int(node_idx)
                
                nodes_expanded += 1
                    
                # Get neighbors - check both CSR matrix and tap connections
                csr_neighbors = []
                tap_neighbors = []
                
                # Get CSR neighbors (if node is within CSR bounds)
                if node_idx < len(adj_csr.indptr) - 1:
                    start_idx = int(adj_csr.indptr[node_idx])
                    end_idx = int(adj_csr.indptr[node_idx + 1])
                    if start_idx < end_idx:
                        csr_neighbors = list(adj_csr.indices[start_idx:end_idx].get() if hasattr(adj_csr.indices[start_idx:end_idx], 'get') else adj_csr.indices[start_idx:end_idx])
                
                # Get tap neighbors (separate storage for tap connections)
                if hasattr(self.gpu_rrg, 'tap_edge_connections') and node_idx in self.gpu_rrg.tap_edge_connections:
                    tap_connections = self.gpu_rrg.tap_edge_connections[node_idx]
                    tap_neighbors = [conn['to_node'] for conn in tap_connections]
                
                # Combine all neighbors
                all_neighbors = csr_neighbors + tap_neighbors
                neighbor_count = len(all_neighbors)
                
                # DEBUG: Log neighbor lookup for first few nodes in early steps
                if step <= 1 and nodes_expanded <= 5:
                    csr_count = len(csr_neighbors)
                    tap_count = len(tap_neighbors)
                    has_tap_connections = hasattr(self.gpu_rrg, 'tap_edge_connections')
                    is_in_tap_connections = node_idx in self.gpu_rrg.tap_edge_connections if has_tap_connections else False
                    
                    # Check if this node is a tap node using the optimized lookup
                    is_detected_as_tap = False
                    tap_name = "unknown"
                    if hasattr(self, '_tap_node_reverse_lookup') and node_idx in self._tap_node_reverse_lookup:
                        is_detected_as_tap = True
                        tap_name = self._tap_node_reverse_lookup[node_idx]
                    
                    # CRITICAL: Debug CSR bounds and indexing
                    csr_bounds_ok = node_idx < len(adj_csr.indptr) - 1
                    csr_start = int(adj_csr.indptr[node_idx]) if csr_bounds_ok else -1
                    csr_end = int(adj_csr.indptr[node_idx + 1]) if csr_bounds_ok else -1
                    csr_range_valid = csr_start < csr_end if csr_bounds_ok else False
                    
                    logger.error(f"  NEIGHBOR DEBUG node {node_idx}: CSR={csr_count}, tap={tap_count}, is_tap_node={is_detected_as_tap} ({tap_name})")
                    logger.error(f"    CSR DEBUG: bounds_ok={csr_bounds_ok}, start={csr_start}, end={csr_end}, range_valid={csr_range_valid}")
                    logger.error(f"    CSR matrix size: {adj_csr.indptr.shape[0]-1} nodes, total connections: {adj_csr.indices.shape[0]}")
                    
                    if csr_count == 0 and tap_count == 0:
                        logger.error(f"    ISOLATED: Node {node_idx} has no neighbors in either CSR or tap connections")
                        if is_detected_as_tap:
                            logger.error(f"    CRITICAL: TAP NODE {tap_name} HAS NO CONNECTIVITY! This is the core bug!")
                    elif is_detected_as_tap:
                        logger.error(f"    TAP NODE: {tap_name} being expanded with {neighbor_count} total neighbors")
                
                # Check if this is a tap node and debug its connectivity - OPTIMIZED LOOKUP
                is_tap_node = False
                if hasattr(self.gpu_rrg, 'tap_nodes'):
                    # Create reverse lookup dictionary for efficiency
                    if not hasattr(self, '_tap_node_reverse_lookup'):
                        self._tap_node_reverse_lookup = {idx: tap_id for tap_id, idx in self.gpu_rrg.tap_nodes.items()}
                    
                    if node_idx in self._tap_node_reverse_lookup:
                        is_tap_node = True
                        tap_id = self._tap_node_reverse_lookup[node_idx]
                        tap_nodes_expanded += 1
                        tap_neighbors_found += neighbor_count
                        if step <= 5:  # Only log first few steps to avoid spam
                            # PERFORMANCE FIX: Disabled excessive logging that kills performance
                            # logger.info(f"GRAPH CONNECTIVITY: Expanding tap node {tap_id} (idx {node_idx}) with {neighbor_count} neighbors")
                            pass  # Keep the if block valid
                
                if neighbor_count > 0:
                    neighbors_found += neighbor_count
                
                if neighbor_count > 0:
                    # Create combined neighbor arrays from both CSR and tap sources
                    if csr_neighbors and tap_neighbors:
                        # Both CSR and tap neighbors exist
                        neighbor_indices = cp.array(csr_neighbors + tap_neighbors, dtype=cp.int32)
                        # For edge costs, use CSR costs for CSR neighbors, default cost for tap neighbors
                        if csr_neighbors:
                            raw_csr_costs = adj_csr.data[start_idx:end_idx]
                            csr_costs = cp.clip(raw_csr_costs, 1.0, 10.0)  # CRITICAL FIX: Clamp CSR costs
                        else:
                            csr_costs = cp.array([], dtype=cp.float32)
                        tap_costs = cp.ones(len(tap_neighbors), dtype=cp.float32) * 1.0  # Default tap edge cost
                        edge_costs = cp.concatenate([csr_costs, tap_costs])
                    elif csr_neighbors:
                        # Only CSR neighbors
                        neighbor_indices = cp.array(csr_neighbors, dtype=cp.int32)
                        raw_edge_costs = adj_csr.data[start_idx:end_idx]
                        
                        # CRITICAL FIX: Clamp edge costs to reasonable values (1.0-10.0)
                        # CSR matrix might have corrupted large values from tap integration
                        edge_costs = cp.clip(raw_edge_costs, 1.0, 10.0)
                        
                        # DEBUG: Log if we had to clamp costs
                        if step <= 2 and cp.any(raw_edge_costs > 10.0):
                            max_raw_cost = float(cp.max(raw_edge_costs).get())
                            logger.error(f"COST CLAMP: Node {node_idx} had edge costs up to {max_raw_cost:.0f}, clamped to {float(cp.max(edge_costs).get())}")
                    elif tap_neighbors:
                        # Only tap neighbors
                        neighbor_indices = cp.array(tap_neighbors, dtype=cp.int32)
                        edge_costs = cp.ones(len(tap_neighbors), dtype=cp.float32) * 1.0  # Default tap edge cost
                    else:
                        continue  # No neighbors at all
                    
                    
                    # Add current congestion costs using proper edge mapping
                    try:
                        # Handle congestion costs for combined neighbor approach
                        if csr_neighbors and tap_neighbors:
                            # Mixed CSR + tap neighbors - map congestion costs separately
                            if hasattr(self.gpu_rrg, 'csr_to_edge_mapping') and csr_neighbors:
                                # CRITICAL: Bounds check CSR mapping access
                                if end_idx <= len(self.gpu_rrg.csr_to_edge_mapping):
                                    csr_edge_indices = self.gpu_rrg.csr_to_edge_mapping[start_idx:end_idx]
                                    # Additional bounds check for edge cost array
                                    max_edge_idx = int(cp.max(csr_edge_indices).get()) if len(csr_edge_indices) > 0 else -1
                                    if max_edge_idx < len(self.gpu_rrg.pathfinder_state.edge_cost):
                                        csr_congestion = self.gpu_rrg.pathfinder_state.edge_cost[csr_edge_indices]
                                    else:
                                        logger.warning(f"EDGE COST BOUNDS: Max edge index {max_edge_idx} >= edge cost array size {len(self.gpu_rrg.pathfinder_state.edge_cost)}")
                                        csr_congestion = cp.zeros(len(csr_neighbors), dtype=cp.float32)
                                else:
                                    logger.warning(f"CSR MAPPING BOUNDS: end_idx {end_idx} >= mapping array size {len(self.gpu_rrg.csr_to_edge_mapping)}")
                                    csr_congestion = cp.zeros(len(csr_neighbors), dtype=cp.float32)
                            else:
                                csr_congestion = cp.zeros(len(csr_neighbors), dtype=cp.float32)
                            
                            # Tap edges have no congestion (newly created)
                            tap_congestion = cp.zeros(len(tap_neighbors), dtype=cp.float32)
                            congestion_costs = cp.concatenate([csr_congestion, tap_congestion])
                            
                        elif csr_neighbors:
                            # Only CSR neighbors - use existing logic
                            if hasattr(self.gpu_rrg, 'csr_to_edge_mapping'):
                                # CRITICAL: Bounds check CSR mapping access  
                                if end_idx <= len(self.gpu_rrg.csr_to_edge_mapping):
                                    original_edge_indices = self.gpu_rrg.csr_to_edge_mapping[start_idx:end_idx]
                                    # Additional bounds check for edge cost array
                                    max_edge_idx = int(cp.max(original_edge_indices).get()) if len(original_edge_indices) > 0 else -1
                                    if max_edge_idx < len(self.gpu_rrg.pathfinder_state.edge_cost):
                                        congestion_costs = self.gpu_rrg.pathfinder_state.edge_cost[original_edge_indices]
                                    else:
                                        logger.warning(f"EDGE COST BOUNDS: Max edge index {max_edge_idx} >= edge cost array size {len(self.gpu_rrg.pathfinder_state.edge_cost)}")
                                        congestion_costs = cp.zeros(len(csr_neighbors), dtype=cp.float32)
                                else:
                                    logger.warning(f"CSR MAPPING BOUNDS: end_idx {end_idx} >= mapping array size {len(self.gpu_rrg.csr_to_edge_mapping)}")
                                    congestion_costs = cp.zeros(len(csr_neighbors), dtype=cp.float32)
                            else:
                                congestion_costs = cp.zeros(len(csr_neighbors), dtype=cp.float32)
                                
                        else:
                            # Only tap neighbors - no congestion
                            congestion_costs = cp.zeros(len(tap_neighbors), dtype=cp.float32)
                        
                        # Shape validation and final cost calculation
                        if len(congestion_costs) == len(edge_costs):
                            total_costs = edge_costs + congestion_costs
                        else:
                            logger.warning(f"Congestion cost shape mismatch: {len(congestion_costs)} vs {len(edge_costs)} - using base costs")
                            total_costs = edge_costs
                        
                    except Exception as e:
                        logger.warning(f"Error accessing congestion costs: {e} - using base costs only")
                        total_costs = edge_costs
                    
                    # Calculate new distances
                    current_dist = distances[net_idx, node_idx]
                    new_distances = current_dist + total_costs
                    
                    # Ensure array shapes are compatible
                    if len(new_distances) != len(neighbor_indices):
                        logger.warning(f"Shape mismatch: new_distances {new_distances.shape} vs neighbor_indices {neighbor_indices.shape}")
                        continue
                    
                    # Update for unvisited neighbors with better paths
                    unvisited_mask = ~visited[net_idx, neighbor_indices]
                    better_mask = new_distances < distances[net_idx, neighbor_indices]
                    update_mask = unvisited_mask & better_mask
                    
                    if cp.any(update_mask):
                        update_indices = neighbor_indices[update_mask]
                        
                        # BASIC SAFETY: Only filter indices that would cause immediate array access crash
                        max_valid_idx = distances.shape[1] - 1
                        valid_mask = update_indices <= max_valid_idx
                        if not cp.all(valid_mask):
                            invalid_count = int(cp.sum(~valid_mask).get())
                            logger.debug(f"BOUNDS SAFETY: Skipping {invalid_count} indices beyond distance array bounds")
                            update_indices = update_indices[valid_mask]
                            if len(update_indices) == 0:
                                continue
                            filtered_distances = new_distances[update_mask][valid_mask]
                        else:
                            filtered_distances = new_distances[update_mask]
                        
                        distances[net_idx, update_indices] = filtered_distances
                        parents[net_idx, update_indices] = node_idx
                        frontiers[net_idx, update_indices] = True
                        
                        # CRITICAL DEBUG: Log successful distance updates (first few steps only)
                        if step <= 2:
                            num_updates = int(cp.sum(update_mask).get())
                            min_new_dist = float(cp.min(new_distances[update_mask]).get())
                            max_new_dist = float(cp.max(new_distances[update_mask]).get())
                            logger.error(f"DISTANCE UPDATES: Net {net_idx}, node {node_idx} updated {num_updates} neighbors with distances {min_new_dist:.2f}-{max_new_dist:.2f}")
        
        # CRITICAL DEBUG: Final frontier state after expansion
        final_frontier_count = int(cp.sum(frontiers).get())
        if step <= 5:
            logger.error(f"  FINAL RESULT: Started {frontier_before_visited} -> Removed {total_selected} -> Added neighbors -> Final {final_frontier_count}")
            if final_frontier_count == 0:
                logger.error(f"  CRITICAL: ALL FRONTIERS ELIMINATED! No neighbors added or they were all visited")
                if neighbors_found == 0:
                    logger.error(f"  ROOT CAUSE: ZERO NEIGHBORS FOUND for {nodes_expanded} expanded nodes")
                    logger.error(f"  This indicates tap nodes have NO CONNECTIVITY in routing graph!")
        
        # Debug expansion progress
        if step < 5 or step % 100 == 0:  # Log first few steps and every 100th
            logger.info(f"Step {step}: expanded {nodes_expanded} nodes, found {neighbors_found} total neighbors")
            if tap_nodes_expanded > 0:
                logger.info(f"Step {step}: expanded {tap_nodes_expanded} tap nodes, found {tap_neighbors_found} tap neighbors")
            elif step <= 5:
                logger.warning(f"Step {step}: NO TAP NODES EXPANDED - tap nodes may not be in frontiers")
    
    def _calculate_congestion(self, routes: Dict[str, List[int]]) -> cp.ndarray:
        """Calculate edge congestion from all routes"""
        
        # Count usage for each edge
        edge_usage = cp.zeros(self.gpu_rrg.num_edges, dtype=cp.int32)
        
        for net_id, path in routes.items():
            if len(path) > 1:
                # Convert path to edges and increment usage
                for i in range(len(path) - 1):
                    from_node = path[i]
                    to_node = path[i + 1]
                    
                    # Find edge between these nodes
                    edge_idx = self._find_edge_between_nodes(from_node, to_node)
                    if edge_idx >= 0:
                        # CRITICAL: Bounds check to prevent crash
                        if edge_idx < len(edge_usage):
                            edge_usage[edge_idx] += 1
                        else:
                            logger.debug(f"CONGESTION SKIP: Edge index {edge_idx} beyond edge_usage array size {len(edge_usage)} (likely from tap node expansion)")
        
        return edge_usage
    
    def _update_congestion_costs(self, congestion_map: cp.ndarray, iteration: int):
        """Update edge costs based on congestion using PathFinder formula"""
        
        # PathFinder cost function: base_cost * (1 + pres_fac * (usage - 1))
        state = self.gpu_rrg.pathfinder_state
        
        # Calculate presentation factor for this iteration
        pres_fac = self.pres_fac_base * (self.pres_fac_mult ** iteration)
        
        # Update edge costs
        overused_edges = congestion_map > 1
        congestion_penalty = pres_fac * (congestion_map - 1)
        congestion_penalty = cp.maximum(congestion_penalty, 0)  # No negative costs
        
        # Apply to edge costs (additive congestion costs)
        state.edge_pres_cost[:] = congestion_penalty
        
        # Historical congestion (accumulative)
        state.edge_hist_cost += self.acc_fac * overused_edges.astype(cp.float32)
        
        # Total cost = base + present + historical
        total_congestion = state.edge_pres_cost + state.edge_hist_cost
        state.edge_cost[:] = self.gpu_rrg.edge_base_cost + total_congestion
        
        overused_count = cp.sum(overused_edges).get()
        logger.info(f"Iteration costs updated: {overused_count} overused edges, pres_fac={pres_fac:.2f}")
    
    def _check_convergence(self, congestion_map: cp.ndarray) -> bool:
        """Check if PathFinder has converged (no congestion)"""
        overused_edges = cp.sum(congestion_map > 1).get()
        logger.info(f"Convergence check: {overused_edges} overused edges")
        return overused_edges == 0
    
    def _find_edge_between_nodes(self, from_node: int, to_node: int) -> int:
        """Find edge index between two nodes using CSR structure"""
        # Use the appropriate adjacency matrix
        if hasattr(self.gpu_rrg, 'adjacency_csr') and self.gpu_rrg.adjacency_csr is not None:
            adj_csr = self.gpu_rrg.adjacency_csr
        else:
            adj_csr = self.gpu_rrg.adjacency_matrix
        
        if from_node >= len(adj_csr.indptr) - 1:
            return -1
            
        start_idx = int(adj_csr.indptr[from_node])
        end_idx = int(adj_csr.indptr[from_node + 1])
        
        for i in range(start_idx, end_idx):
            if int(adj_csr.indices[i]) == to_node:
                return i  # Return the edge index in the CSR data structure
        
        return -1  # Edge not found
    
    def _reconstruct_path_parallel(self, net_idx: int, source: int, sink: int, parents: cp.ndarray) -> List[int]:
        """Reconstruct path for a single net"""
        path = []
        current = sink
        parents_shape = parents.shape
        logger.info(f"PATH RECONSTRUCTION [{net_idx}]: source={source}, sink={sink}, parents.shape={parents_shape}")
        
        # Debug: Check initial parent value at sink
        if sink < parents_shape[1]:
            initial_parent = int(parents[net_idx, sink].get())
            logger.info(f"PATH RECONSTRUCTION [{net_idx}]: sink {sink} parent = {initial_parent}")
        else:
            logger.error(f"PATH RECONSTRUCTION [{net_idx}]: sink {sink} >= parents.shape[1]={parents_shape[1]} - OUT OF BOUNDS")
            return []
        
        step = 0
        while current != -1 and current != source and step < 100:  # Limit debug steps
            path.append(int(current))
            if current >= parents_shape[1]:
                logger.error(f"PATH RECONSTRUCTION [{net_idx}]: current={current} >= parents.shape[1]={parents_shape[1]} - OUT OF BOUNDS")
                return []
            
            next_current = int(parents[net_idx, current].get())
            logger.info(f"PATH RECONSTRUCTION [{net_idx}]: step {step}: current={current} -> parent={next_current}")
            
            if next_current == current:  # Infinite loop detection
                logger.error(f"PATH RECONSTRUCTION [{net_idx}]: INFINITE LOOP at node {current}")
                return []
                
            current = next_current
            step += 1
        
        logger.info(f"PATH RECONSTRUCTION [{net_idx}]: Loop ended: current={current}, source={source}, step={step}")
        
        if current == source:
            path.append(source)
            path.reverse()
            logger.info(f"PATH RECONSTRUCTION [{net_idx}]: SUCCESS - reconstructed path with {len(path)} nodes: {path[:10]}")
            return path
        else:
            logger.error(f"PATH RECONSTRUCTION [{net_idx}]: FAILED - current={current} never reached source={source}")
            return []
    
    def _initialize_congestion_state(self, num_nets: int):
        """Initialize congestion tracking state"""
        state = self.gpu_rrg.pathfinder_state
        
        # Reset congestion costs
        state.edge_pres_cost.fill(0.0)
        state.edge_hist_cost.fill(0.0)
        state.edge_cost[:] = self.gpu_rrg.edge_base_cost.copy()
        
        logger.info(f"Congestion state initialized for {num_nets} nets")
    
    def _select_nodes_with_heuristic_guidance(self, frontiers, distances, min_distances_broadcasted, tolerance, step):
        """Select nodes using intelligent heuristic guidance toward sinks"""
        
        # For early steps or fallback, use the original minimum distance selection
        if not hasattr(self, 'sink_positions') or step > 500:  # Fallback after 500 steps
            return frontiers & (cp.abs(distances - min_distances_broadcasted) < tolerance)
        
        # Calculate heuristic scores for all frontier nodes
        heuristic_scores = self._calculate_sink_heuristics_gpu(frontiers)
        
        # Combined score: current_cost + heuristic_to_sink
        heuristic_weight = 0.7  # Weight for heuristic guidance
        combined_scores = distances + (heuristic_scores * heuristic_weight)
        
        # Select nodes with best combined scores (within a reasonable distance range)
        distance_threshold = 2.0  # Allow nodes within 2.0 cost units of minimum
        min_scores = cp.min(cp.where(frontiers, combined_scores, cp.inf), axis=1, keepdims=True)
        min_scores_broadcasted = cp.broadcast_to(min_scores, combined_scores.shape)
        
        # Select nodes that are good candidates: near minimum distance AND have good heuristics
        good_candidates = frontiers & (
            (cp.abs(distances - min_distances_broadcasted) < tolerance) |  # Original minimum distance nodes
            (combined_scores <= min_scores_broadcasted + distance_threshold)  # Or good heuristic nodes
        )
        
        # Limit selection to prevent explosion (max 200 nodes per net)
        selected_nodes = self._limit_node_selection(good_candidates, max_nodes_per_net=200)
        
        # Log progress for early steps
        if step <= 10:
            total_frontier = int(cp.sum(frontiers).get())
            total_selected = int(cp.sum(selected_nodes).get())
            logger.info(f"HEURISTIC_SELECTION Step {step}: {total_frontier} frontier -> {total_selected} guided nodes")
        
        return selected_nodes
    
    def _calculate_sink_heuristics_gpu(self, frontiers):
        """Calculate Manhattan distance heuristics to sinks using GPU"""
        
        if not hasattr(self, 'sink_positions'):
            return cp.zeros_like(frontiers, dtype=cp.float32)
        
        num_nets, num_nodes = frontiers.shape
        heuristics = cp.full((num_nets, num_nodes), cp.inf, dtype=cp.float32)
        
        # For each net, calculate heuristic to its sink
        for net_idx in range(num_nets):
            if net_idx < len(self.sink_positions):
                sink_pos = self.sink_positions[net_idx]
                
                # Calculate Manhattan distances to sink for all nodes
                # Using broadcast operations for GPU efficiency
                node_x = cp.array([self.gpu_rrg.cpu_rrg.nodes[self.gpu_rrg.get_node_id(i)].x 
                                  for i in range(num_nodes)], dtype=cp.float32)
                node_y = cp.array([self.gpu_rrg.cpu_rrg.nodes[self.gpu_rrg.get_node_id(i)].y 
                                  for i in range(num_nodes)], dtype=cp.float32)
                node_layer = cp.array([self.gpu_rrg.cpu_rrg.nodes[self.gpu_rrg.get_node_id(i)].layer 
                                      for i in range(num_nodes)], dtype=cp.float32)
                
                # Manhattan distance + layer penalty
                distance = (cp.abs(node_x - sink_pos['x']) + 
                           cp.abs(node_y - sink_pos['y']) + 
                           cp.abs(node_layer - sink_pos['layer']) * 2.0)
                
                heuristics[net_idx] = distance
        
        return heuristics
    
    def _limit_node_selection(self, candidates, max_nodes_per_net=200):
        """Limit node selection to prevent computational explosion"""
        
        num_nets, num_nodes = candidates.shape
        limited_selection = cp.zeros_like(candidates, dtype=bool)
        
        for net_idx in range(num_nets):
            net_candidates = candidates[net_idx]
            candidate_indices = cp.where(net_candidates)[0]
            
            if len(candidate_indices) <= max_nodes_per_net:
                # If within limit, select all candidates
                limited_selection[net_idx] = net_candidates
            else:
                # If too many candidates, select the best ones randomly
                # (In a more sophisticated implementation, we'd select by best score)
                selected_indices = cp.random.choice(
                    candidate_indices, 
                    size=max_nodes_per_net, 
                    replace=False
                )
                limited_selection[net_idx, selected_indices] = True
        
        return limited_selection
    
    def _initialize_sink_positions(self, requests):
        """Initialize sink positions for heuristic guidance"""
        
        self.sink_positions = []
        
        for req in requests:
            try:
                # Get sink node information
                sink_node_id = self.gpu_rrg.get_node_id(req.sink_idx)
                if sink_node_id in self.gpu_rrg.cpu_rrg.nodes:
                    sink_node = self.gpu_rrg.cpu_rrg.nodes[sink_node_id]
                    sink_pos = {
                        'x': sink_node.x,
                        'y': sink_node.y, 
                        'layer': sink_node.layer,
                        'node_idx': req.sink_idx
                    }
                    self.sink_positions.append(sink_pos)
                    logger.debug(f"HEURISTIC_INIT: Sink {req.net_id} at ({sink_node.x:.1f}, {sink_node.y:.1f}, L{sink_node.layer})")
                else:
                    logger.warning(f"HEURISTIC_INIT: Sink node {sink_node_id} not found for {req.net_id}")
                    self.sink_positions.append({'x': 0, 'y': 0, 'layer': 0, 'node_idx': req.sink_idx})
            except Exception as e:
                logger.warning(f"HEURISTIC_INIT: Error initializing sink position for {req.net_id}: {e}")
                self.sink_positions.append({'x': 0, 'y': 0, 'layer': 0, 'node_idx': req.sink_idx})
        
        logger.info(f"HEURISTIC_INIT: Initialized {len(self.sink_positions)} sink positions for guided PathFinder")
    
    def _check_proximity_termination(self, active_frontiers, distances, step):
        """Check if any frontiers are very close to their target sinks"""
        
        if not hasattr(self, 'sink_positions'):
            return False
        
        proximity_threshold = 5.0  # mm - consider "close enough"
        num_nets = active_frontiers.shape[0]
        
        for net_idx in range(min(num_nets, len(self.sink_positions))):
            if not cp.any(active_frontiers[net_idx]):
                continue  # No active frontier for this net
            
            sink_pos = self.sink_positions[net_idx]
            frontier_nodes = cp.where(active_frontiers[net_idx])[0]
            
            # Check distances from frontier nodes to sink
            for node_idx in frontier_nodes[:20]:  # Check up to 20 frontier nodes per net
                try:
                    node_idx = int(node_idx.get()) if hasattr(node_idx, 'get') else int(node_idx)
                    node_id = self.gpu_rrg.get_node_id(node_idx)
                    
                    if node_id in self.gpu_rrg.cpu_rrg.nodes:
                        node = self.gpu_rrg.cpu_rrg.nodes[node_id]
                        
                        # Manhattan distance to sink
                        distance = (abs(node.x - sink_pos['x']) + 
                                   abs(node.y - sink_pos['y']) + 
                                   abs(node.layer - sink_pos['layer']) * 2.0)
                        
                        if distance <= proximity_threshold:
                            logger.info(f"PROXIMITY: Net {net_idx} frontier node within {distance:.1f}mm of sink")
                            return True
                            
                except Exception as e:
                    continue  # Skip problematic nodes
        
        return False