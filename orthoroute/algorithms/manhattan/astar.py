"""A* pathfinding implementation for Manhattan routing."""
import logging
import heapq
import time
from typing import List, Optional, Set
from collections import defaultdict

from ...domain.services.pathfinder import PathfindingService, GridPoint, ManhattanHeuristic, PathNode
from ..base.grid import RoutingGrid, CellState
from .layer_assignment import LayerDirectionManager

logger = logging.getLogger(__name__)


class ManhattanAStarPathfinder:
    """A* pathfinder specialized for Manhattan routing."""
    
    def __init__(self):
        """Initialize A* pathfinder."""
        self.heuristic = ManhattanHeuristic()
        self.pathfinding_service = PathfindingService(self.heuristic)
    
    def find_path(self, start: GridPoint, end: GridPoint, 
                  grid: RoutingGrid, layer_manager: LayerDirectionManager,
                  net_id: str, timeout: float = 10.0) -> List[GridPoint]:
        """Find path using Manhattan A* algorithm."""
        
        start_time = time.time()
        max_iterations = 50000
        
        # Priority queue for A*
        open_set = []
        heapq.heappush(open_set, PathNode(
            start, 0, self.heuristic.calculate(start, end),
            self.heuristic.calculate(start, end)
        ))
        
        # Tracking sets and maps
        closed_set = set()
        came_from = {}
        g_score = {start: 0.0}
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"A* pathfinding timeout after {iterations} iterations")
                break
            
            current_node = heapq.heappop(open_set)
            current = current_node.point
            
            # Check if reached goal
            if current == end:
                path = self._reconstruct_path(came_from, current, start)
                logger.debug(f"A* found path with {len(path)} points in {iterations} iterations")
                return path
            
            closed_set.add(current)
            
            # Get neighbors based on Manhattan routing rules
            neighbors = self._get_manhattan_neighbors(current, grid, layer_manager)
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                # Check if neighbor is accessible
                if not self._is_accessible(neighbor, grid, net_id):
                    continue
                
                # Calculate movement cost
                movement_cost = self._calculate_movement_cost(current, neighbor, grid, net_id)
                tentative_g = g_score[current] + movement_cost
                
                # Via penalty for layer changes
                if neighbor.layer != current.layer:
                    tentative_g += 3.0  # Higher cost for vias
                
                # Update path if better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self.heuristic.calculate(neighbor, end)
                    f_score = tentative_g + h_score
                    
                    # Add to open set
                    heapq.heappush(open_set, PathNode(
                        neighbor, tentative_g, h_score, f_score
                    ))
        
        logger.debug(f"A* failed to find path after {iterations} iterations")
        return []
    
    def _get_manhattan_neighbors(self, point: GridPoint, grid: RoutingGrid, 
                               layer_manager: LayerDirectionManager) -> List[GridPoint]:
        """Get valid Manhattan neighbors for a grid point."""
        neighbors = []
        
        # Same-layer movement based on Manhattan direction
        if layer_manager.is_horizontal_layer(point.layer):
            # Horizontal movement only
            for dx in [-1, 1]:
                new_x = point.x + dx
                if grid.is_valid_position(new_x, point.y, point.layer):
                    neighbors.append(GridPoint(new_x, point.y, point.layer))
        
        elif layer_manager.is_vertical_layer(point.layer):
            # Vertical movement only
            for dy in [-1, 1]:
                new_y = point.y + dy
                if grid.is_valid_position(point.x, new_y, point.layer):
                    neighbors.append(GridPoint(point.x, new_y, point.layer))
        
        # Layer transitions (vias) - only to adjacent layers
        adjacent_layers = layer_manager.get_adjacent_layers(point.layer)
        for adj_layer in adjacent_layers:
            if grid.is_valid_position(point.x, point.y, adj_layer):
                # Check if via transition is allowed
                if self._is_via_transition_allowed(point.layer, adj_layer, grid):
                    neighbors.append(GridPoint(point.x, point.y, adj_layer))
        
        return neighbors
    
    def _is_accessible(self, point: GridPoint, grid: RoutingGrid, net_id: str) -> bool:
        """Check if grid point is accessible for routing."""
        cell = grid.get_cell(point.x, point.y, point.layer)
        return cell.is_accessible_by_net(net_id)
    
    def _calculate_movement_cost(self, current: GridPoint, neighbor: GridPoint, 
                               grid: RoutingGrid, net_id: str) -> float:
        """Calculate cost of moving to neighbor."""
        base_cost = 1.0
        
        # Get cell at neighbor position
        cell = grid.get_cell(neighbor.x, neighbor.y, neighbor.layer)
        
        # Add congestion penalty
        cost = base_cost + cell.cost_penalty
        
        # Add distance-based cost
        manhattan_dist = abs(neighbor.x - current.x) + abs(neighbor.y - current.y)
        cost += manhattan_dist
        
        # Penalty for obstacles that are accessible (like pad keepouts)
        if cell.state == CellState.OBSTACLE and net_id in cell.accessibility_mask:
            cost += 0.5  # Small penalty for routing through keepout areas
        
        return cost
    
    def _is_via_transition_allowed(self, from_layer: int, to_layer: int, 
                                  grid: RoutingGrid) -> bool:
        """Check if via transition between layers is allowed."""
        # For now, allow all adjacent layer transitions
        # In a full implementation, this would check:
        # - Board stackup constraints
        # - Via type availability (blind, buried, micro)
        # - Manufacturing design rules
        layer_diff = abs(from_layer - to_layer)
        return layer_diff == 1
    
    def _reconstruct_path(self, came_from: dict, current: GridPoint, start: GridPoint) -> List[GridPoint]:
        """Reconstruct path from A* came_from mapping."""
        path = []
        
        while current in came_from:
            path.append(current)
            current = came_from[current]
        
        path.append(start)
        return list(reversed(path))
    
    def find_path_with_alternatives(self, start: GridPoint, end: GridPoint,
                                  grid: RoutingGrid, layer_manager: LayerDirectionManager,
                                  net_id: str, timeout: float = 10.0) -> List[GridPoint]:
        """Find path with alternative layer attempts."""
        
        # Try primary path first
        path = self.find_path(start, end, grid, layer_manager, net_id, timeout * 0.7)
        if path:
            return path
        
        # Try alternative starting layers
        for alt_layer in range(grid.layer_count):
            if alt_layer == start.layer:
                continue
            
            alt_start = GridPoint(start.x, start.y, alt_layer)
            alt_end = GridPoint(end.x, end.y, alt_layer)
            
            # Check if alternative points are accessible
            if (self._is_accessible(alt_start, grid, net_id) and
                self._is_accessible(alt_end, grid, net_id)):
                
                path = self.find_path(alt_start, alt_end, grid, layer_manager, 
                                    net_id, timeout * 0.3)
                if path:
                    logger.debug(f"Found alternative path on layer {alt_layer}")
                    return path
        
        return []
    
    def estimate_path_difficulty(self, start: GridPoint, end: GridPoint,
                               grid: RoutingGrid, layer_manager: LayerDirectionManager,
                               net_id: str) -> float:
        """Estimate routing difficulty for path planning."""
        
        # Base difficulty from Manhattan distance
        base_distance = self.heuristic.calculate(start, end)
        difficulty = base_distance
        
        # Sample points along direct path to estimate obstacles
        steps = min(10, int(base_distance))
        if steps > 1:
            dx = (end.x - start.x) / steps
            dy = (end.y - start.y) / steps
            
            obstacle_count = 0
            for i in range(1, steps):
                sample_x = int(start.x + i * dx)
                sample_y = int(start.y + i * dy)
                
                if grid.is_valid_position(sample_x, sample_y, start.layer):
                    cell = grid.get_cell(sample_x, sample_y, start.layer)
                    if not cell.is_accessible_by_net(net_id):
                        obstacle_count += 1
                    elif cell.state == CellState.OBSTACLE:
                        obstacle_count += 0.5  # Partial penalty for accessible obstacles
            
            # Add obstacle penalty
            difficulty += obstacle_count * 2.0
        
        # Layer mismatch penalty
        if start.layer != end.layer:
            layer_changes = abs(start.layer - end.layer)
            difficulty += layer_changes * 3.0  # Via cost
        
        # Direction alignment bonus/penalty
        direction_start = layer_manager.get_layer_direction(start.layer)
        dx, dy = end.x - start.x, end.y - start.y
        
        if direction_start.value == "horizontal" and abs(dx) > abs(dy):
            difficulty *= 0.9  # Bonus for aligned direction
        elif direction_start.value == "vertical" and abs(dy) > abs(dx):
            difficulty *= 0.9  # Bonus for aligned direction
        else:
            difficulty *= 1.1  # Penalty for misaligned direction
        
        return difficulty