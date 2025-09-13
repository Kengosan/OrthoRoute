"""
Pure GPU CSR PathFinder Implementation
Optimized pathfinding using sparse CSR adjacency matrices
"""

import cupy as cp
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class GPUCSRPathFinder:
    """Pure GPU pathfinding using CSR sparse matrices"""
    
    def __init__(self, gpu_rrg):
        self.gpu_rrg = gpu_rrg
        
    def route_single_net_csr(self, source_idx: int, sink_idx: int, state) -> Tuple[List[int], List[int]]:
        """Pure GPU pathfinding using CSR sparse matrix - no CPU transfers"""
        
        logger.debug(f"Starting GPU CSR pathfinding: {source_idx} -> {sink_idx}")
        
        # Reset pathfinding state
        state.distance.fill(cp.inf)
        state.parent_node.fill(-1)
        state.visited.fill(False)
        state.distance[source_idx] = 0.0
        
        # Use GPU arrays for everything
        current_frontier = cp.zeros(state.distance.shape[0], dtype=cp.bool_)
        current_frontier[source_idx] = True
        
        max_iterations = 10000  # Safety limit
        iteration = 0
        
        while cp.any(current_frontier) and iteration < max_iterations:
            iteration += 1
            
            # Find nodes in current frontier with minimum distance
            frontier_distances = cp.where(current_frontier, state.distance, cp.inf)
            min_dist = cp.min(frontier_distances)
            
            if cp.isinf(min_dist):
                break  # No more reachable nodes
                
            # Select nodes at minimum distance
            current_nodes = current_frontier & (state.distance == min_dist)
            
            # Mark selected nodes as visited and remove from frontier
            state.visited |= current_nodes
            current_frontier &= ~current_nodes
            
            # Check if we reached the sink
            if state.visited[sink_idx]:
                break
            
            # Expand frontier using CSR matrix
            self._expand_frontier_csr(current_nodes, current_frontier, state)
            
            if iteration % 1000 == 0:
                active_nodes = cp.sum(current_frontier)
                logger.debug(f"GPU CSR iteration {iteration}: {active_nodes} active nodes")
        
        logger.debug(f"GPU CSR pathfinding completed in {iteration} iterations")
        
        # Reconstruct path
        if state.visited[sink_idx]:
            return self._reconstruct_path_csr(source_idx, sink_idx, state)
        else:
            logger.warning(f"GPU CSR pathfinding failed to reach sink {sink_idx}")
            return [], []
    
    def _expand_frontier_csr(self, current_nodes, frontier, state):
        """Expand frontier using GPU CSR matrix operations"""
        
        # Get CSR adjacency matrix
        adj_csr = self.gpu_rrg.adjacency_csr
        
        # For each current node, expand to its neighbors
        current_indices = cp.where(current_nodes)[0]
        
        for node_idx in current_indices:
            node_idx = int(node_idx)  # Convert from numpy/cupy scalar
            if node_idx >= len(adj_csr.indptr) - 1:
                continue
                
            # Get neighbors using CSR indexing
            start_idx = int(adj_csr.indptr[node_idx])
            end_idx = int(adj_csr.indptr[node_idx + 1])
            
            if start_idx < end_idx and end_idx <= len(adj_csr.indices):
                # Get neighbor indices and costs
                neighbor_indices = adj_csr.indices[start_idx:end_idx]
                edge_costs = adj_csr.data[start_idx:end_idx]
                
                # Calculate new distances
                current_dist = state.distance[node_idx]
                new_distances = current_dist + edge_costs + state.node_cost[neighbor_indices]
                
                # Update distances for unvisited neighbors
                unvisited_mask = ~state.visited[neighbor_indices]
                better_mask = new_distances < state.distance[neighbor_indices]
                update_mask = unvisited_mask & better_mask
                
                if cp.any(update_mask):
                    # Update distances and parents
                    update_indices = neighbor_indices[update_mask]
                    state.distance[update_indices] = new_distances[update_mask]
                    state.parent_node[update_indices] = node_idx
                    
                    # Add to frontier
                    frontier[update_indices] = True
    
    def _reconstruct_path_csr(self, source_idx: int, sink_idx: int, state) -> Tuple[List[int], List[int]]:
        """Reconstruct path from parent pointers (GPU version)"""
        
        path_nodes = []
        path_edges = []
        
        current_idx = sink_idx
        while current_idx != -1 and current_idx != source_idx:
            path_nodes.append(current_idx)
            parent_idx = int(state.parent_node[current_idx])
            
            if parent_idx == -1:
                break
                
            current_idx = parent_idx
        
        if current_idx == source_idx:
            path_nodes.append(source_idx)
            path_nodes.reverse()
        else:
            # Path not found
            return [], []
        
        # Convert to CPU lists for compatibility
        if hasattr(path_nodes[0], 'get'):  # CuPy array
            path_nodes = [int(cp.asnumpy(node)) for node in path_nodes]
        
        logger.debug(f"CSR path reconstruction complete: {len(path_nodes)} nodes")
        return path_nodes, path_edges