"""
A* pathfinding algorithm for Manhattan routing
"""

import heapq
import logging
from typing import Dict, List, Tuple, Optional, Set
from .types import Track, Via, RoutingConfig
from .grid_manager import GridManager

logger = logging.getLogger(__name__)

class ManhattanPathfinder:
    """A* pathfinding with Manhattan routing constraints"""
    
    def __init__(self, config: RoutingConfig, grid_manager: GridManager):
        self.config = config
        self.grid = grid_manager
    
    def find_path(self, start: Tuple[int, int, int], goal: Tuple[int, int, int], net_id: int) -> Optional[Dict]:
        """
        A* pathfinding with Manhattan routing constraints
        
        Args:
            start: (layer, y, x) start position
            goal: (layer, y, x) goal position  
            net_id: Net ID for DRC checking
            
        Returns:
            Dict with 'path', 'tracks', 'vias' or None if no path found
        """
        start_layer, start_y, start_x = start
        goal_layer, goal_y, goal_x = goal
        
        logger.debug(f"Pathfinding from ({start_layer},{start_y},{start_x}) to ({goal_layer},{goal_y},{goal_x})")
        
        # Priority queue: (f_score, g_score, position, came_from)
        open_set = [(0, 0, start, None)]
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            f_score, g_cost, current, parent = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            came_from[current] = parent
            
            # Goal reached
            if current == goal:
                return self._reconstruct_path(came_from, current, net_id)
            
            # Explore neighbors
            for neighbor, move_cost, move_type in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                neighbor_layer, neighbor_y, neighbor_x = neighbor
                
                # Check if neighbor is valid for this net
                if not self.grid.is_cell_free(neighbor_layer, neighbor_y, neighbor_x, net_id):
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    h_score = self._manhattan_heuristic(neighbor, goal)
                    f_score = tentative_g + h_score
                    
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, current))
        
        return None  # No path found
    
    def _get_neighbors(self, position: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], int, str]]:
        """Get valid neighbors for Manhattan routing with direction preferences"""
        layer, y, x = position
        neighbors = []
        
        current_direction = self.grid.get_layer_direction(layer)
        
        # Horizontal movement (preferred on horizontal layers)
        cost_multiplier = 1 if current_direction == 'H' else 2
        if x > 0:
            neighbors.append(((layer, y, x-1), cost_multiplier, 'move'))
        if x < self.grid.grid_width - 1:
            neighbors.append(((layer, y, x+1), cost_multiplier, 'move'))
        
        # Vertical movement (preferred on vertical layers)  
        cost_multiplier = 1 if current_direction == 'V' else 2
        if y > 0:
            neighbors.append(((layer, y-1, x), cost_multiplier, 'move'))
        if y < self.grid.grid_height - 1:
            neighbors.append(((layer, y+1, x), cost_multiplier, 'move'))
        
        # Layer changes (vias) - allowed between any internal layers
        via_cost = self.config.via_cost
        if layer > 0:
            neighbors.append(((layer-1, y, x), via_cost, 'via'))
        if layer < self.grid.num_layers - 1:
            neighbors.append(((layer+1, y, x), via_cost, 'via'))
        
        return neighbors
    
    def _manhattan_heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> int:
        """Manhattan distance heuristic with layer change penalty"""
        layer1, y1, x1 = pos1
        layer2, y2, x2 = pos2
        
        xy_distance = abs(x2 - x1) + abs(y2 - y1)
        layer_distance = abs(layer2 - layer1) * self.config.via_cost
        
        return xy_distance + layer_distance
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int, int], net_id: int) -> Dict:
        """Reconstruct path from A* search results"""
        path = []
        tracks = []
        vias = []
        
        # Build path from goal to start
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        
        path.reverse()
        
        # Convert path to tracks and vias
        for i in range(len(path) - 1):
            curr_layer, curr_y, curr_x = path[i]
            next_layer, next_y, next_x = path[i + 1]
            
            curr_world_x, curr_world_y = self.grid.grid_to_world(curr_x, curr_y)
            next_world_x, next_world_y = self.grid.grid_to_world(next_x, next_y)
            
            if curr_layer == next_layer:
                # Same layer - create track
                track = Track(
                    start_x=curr_world_x,
                    start_y=curr_world_y,
                    end_x=next_world_x,
                    end_y=next_world_y,
                    layer=curr_layer,
                    width=self.config.track_width,
                    net_id=net_id
                )
                tracks.append(track)
            else:
                # Layer change - create via
                via = Via(
                    x=curr_world_x,
                    y=curr_world_y,
                    from_layer=curr_layer,
                    to_layer=next_layer,
                    size=self.config.via_diameter,
                    drill=self.config.via_drill,
                    net_id=net_id
                )
                vias.append(via)
        
        # Mark path cells as used
        for layer, y, x in path:
            self.grid.mark_cell_used(layer, y, x, net_id)
        
        return {
            'path': path,
            'tracks': tracks,
            'vias': vias,
            'success': True
        }