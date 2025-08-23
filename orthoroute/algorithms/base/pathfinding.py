"""Base pathfinding functionality for routing algorithms."""
import logging
from typing import List, Optional, Set, Tuple
from abc import ABC, abstractmethod

from ...domain.services.pathfinder import GridPoint

logger = logging.getLogger(__name__)


class PathfindingMixin:
    """Mixin class providing common pathfinding functionality."""
    
    def find_shortest_path(self, start: GridPoint, end: GridPoint, 
                          obstacles: Set[GridPoint]) -> List[GridPoint]:
        """Find shortest path avoiding obstacles.
        
        This is a basic implementation using Manhattan distance.
        Subclasses can override for more sophisticated pathfinding.
        """
        if start == end:
            return [start]
        
        # Simple pathfinding - can be overridden by algorithms
        path = []
        current = start
        
        while current != end:
            # Move towards target, avoiding obstacles
            dx = 1 if end.x > current.x else -1 if end.x < current.x else 0
            dy = 1 if end.y > current.y else -1 if end.y < current.y else 0
            
            # Try X movement first
            if dx != 0:
                next_point = GridPoint(current.x + dx, current.y, current.layer)
                if next_point not in obstacles:
                    current = next_point
                    path.append(current)
                    continue
            
            # Try Y movement
            if dy != 0:
                next_point = GridPoint(current.x, current.y + dy, current.layer)
                if next_point not in obstacles:
                    current = next_point
                    path.append(current)
                    continue
            
            # Can't make progress - path blocked
            logger.warning(f"Path blocked from {current} to {end}")
            break
        
        return path
    
    def calculate_manhattan_distance(self, start: GridPoint, end: GridPoint) -> float:
        """Calculate Manhattan distance between two points."""
        return abs(end.x - start.x) + abs(end.y - start.y)
    
    def get_neighbors(self, point: GridPoint, bounds: Tuple[int, int, int, int]) -> List[GridPoint]:
        """Get valid neighboring points within bounds."""
        min_x, min_y, max_x, max_y = bounds
        neighbors = []
        
        # 4-connected neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x = point.x + dx
            new_y = point.y + dy
            
            if min_x <= new_x <= max_x and min_y <= new_y <= max_y:
                neighbors.append(GridPoint(new_x, new_y, point.layer))
        
        return neighbors