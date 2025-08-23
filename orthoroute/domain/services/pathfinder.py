"""Domain service for pathfinding algorithms."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import heapq

from ..models.board import Coordinate


class PathfindingAlgorithm(Enum):
    """Available pathfinding algorithms."""
    ASTAR = "a_star"
    DIJKSTRA = "dijkstra"
    BFS = "breadth_first"


@dataclass(frozen=True)
class GridPoint:
    """A point in 3D routing grid space."""
    x: int
    y: int
    layer: int
    
    def __hash__(self):
        return hash((self.x, self.y, self.layer))


@dataclass
class PathNode:
    """Node for pathfinding algorithms."""
    point: GridPoint
    g_score: float  # Cost from start
    h_score: float  # Heuristic to goal
    f_score: float  # g + h
    parent: Optional['PathNode'] = None
    
    def __lt__(self, other):
        return self.f_score < other.f_score


class HeuristicFunction(ABC):
    """Abstract base class for heuristic functions."""
    
    @abstractmethod
    def calculate(self, start: GridPoint, end: GridPoint) -> float:
        """Calculate heuristic cost between two points."""
        pass


class ManhattanHeuristic(HeuristicFunction):
    """Manhattan distance heuristic."""
    
    def calculate(self, start: GridPoint, end: GridPoint) -> float:
        """Calculate Manhattan distance."""
        return (abs(start.x - end.x) + 
                abs(start.y - end.y) + 
                abs(start.layer - end.layer) * 2)  # Layer changes cost more


class EuclideanHeuristic(HeuristicFunction):
    """Euclidean distance heuristic."""
    
    def calculate(self, start: GridPoint, end: GridPoint) -> float:
        """Calculate Euclidean distance."""
        dx = start.x - end.x
        dy = start.y - end.y
        dl = (start.layer - end.layer) * 2  # Layer changes cost more
        return (dx * dx + dy * dy + dl * dl) ** 0.5


class GridObstacle(ABC):
    """Abstract base class for grid obstacles."""
    
    @abstractmethod
    def is_blocked(self, point: GridPoint, net_id: str) -> bool:
        """Check if a point is blocked for the given net."""
        pass
    
    @abstractmethod
    def get_movement_cost(self, point: GridPoint, net_id: str) -> float:
        """Get movement cost for a point (1.0 = normal, >1.0 = penalty)."""
        pass


class PathfindingService:
    """Domain service for pathfinding operations."""
    
    def __init__(self, heuristic: HeuristicFunction = None):
        """Initialize pathfinding service."""
        self.heuristic = heuristic or ManhattanHeuristic()
    
    def find_path(self, start: GridPoint, end: GridPoint,
                  get_neighbors_func, obstacle_checker: GridObstacle,
                  net_id: str, algorithm: PathfindingAlgorithm = PathfindingAlgorithm.ASTAR,
                  max_iterations: int = 10000) -> List[GridPoint]:
        """Find path between two points using specified algorithm."""
        
        if algorithm == PathfindingAlgorithm.ASTAR:
            return self._astar(start, end, get_neighbors_func, obstacle_checker, 
                             net_id, max_iterations)
        elif algorithm == PathfindingAlgorithm.DIJKSTRA:
            return self._dijkstra(start, end, get_neighbors_func, obstacle_checker,
                                net_id, max_iterations)
        elif algorithm == PathfindingAlgorithm.BFS:
            return self._breadth_first_search(start, end, get_neighbors_func, 
                                            obstacle_checker, net_id, max_iterations)
        else:
            raise ValueError(f"Unknown pathfinding algorithm: {algorithm}")
    
    def _astar(self, start: GridPoint, end: GridPoint, get_neighbors_func,
               obstacle_checker: GridObstacle, net_id: str,
               max_iterations: int) -> List[GridPoint]:
        """A* pathfinding algorithm."""
        open_set = []
        heapq.heappush(open_set, PathNode(
            start, 0, self.heuristic.calculate(start, end),
            self.heuristic.calculate(start, end)
        ))
        
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current_node = heapq.heappop(open_set)
            current = current_node.point
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path))
            
            closed_set.add(current)
            
            for neighbor in get_neighbors_func(current):
                if neighbor in closed_set:
                    continue
                
                if obstacle_checker.is_blocked(neighbor, net_id):
                    continue
                
                # Calculate movement cost
                movement_cost = obstacle_checker.get_movement_cost(neighbor, net_id)
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + movement_cost
                
                # Add layer transition penalty
                if neighbor.layer != current.layer:
                    tentative_g += 3.0  # Via cost penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self.heuristic.calculate(neighbor, end)
                    f_score = tentative_g + h_score
                    
                    heapq.heappush(open_set, PathNode(
                        neighbor, tentative_g, h_score, f_score
                    ))
        
        return []  # No path found
    
    def _dijkstra(self, start: GridPoint, end: GridPoint, get_neighbors_func,
                  obstacle_checker: GridObstacle, net_id: str,
                  max_iterations: int) -> List[GridPoint]:
        """Dijkstra's algorithm (A* with zero heuristic)."""
        # Use A* with zero heuristic
        original_heuristic = self.heuristic
        self.heuristic = ZeroHeuristic()
        
        try:
            return self._astar(start, end, get_neighbors_func, obstacle_checker,
                             net_id, max_iterations)
        finally:
            self.heuristic = original_heuristic
    
    def _breadth_first_search(self, start: GridPoint, end: GridPoint,
                            get_neighbors_func, obstacle_checker: GridObstacle,
                            net_id: str, max_iterations: int) -> List[GridPoint]:
        """Breadth-first search algorithm."""
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            current, path = queue.popleft()
            
            if current == end:
                return path
            
            for neighbor in get_neighbors_func(current):
                if neighbor in visited:
                    continue
                
                if obstacle_checker.is_blocked(neighbor, net_id):
                    continue
                
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def estimate_path_cost(self, start: GridPoint, end: GridPoint) -> float:
        """Estimate the cost of a path between two points."""
        return self.heuristic.calculate(start, end)
    
    def validate_path(self, path: List[GridPoint]) -> List[str]:
        """Validate that a path is continuous and valid."""
        issues = []
        
        if len(path) < 2:
            return issues  # Single point or empty path is valid
        
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            # Check if points are adjacent
            dx = abs(next_point.x - current.x)
            dy = abs(next_point.y - current.y)
            dl = abs(next_point.layer - current.layer)
            
            # Valid moves: same layer (dx=1,dy=0 or dx=0,dy=1) or layer change (dx=0,dy=0,dl=1)
            if dl > 0:  # Layer change
                if dx != 0 or dy != 0 or dl != 1:
                    issues.append(f"Invalid layer transition at step {i}: {current} -> {next_point}")
            else:  # Same layer
                if (dx + dy) != 1:
                    issues.append(f"Invalid movement at step {i}: {current} -> {next_point}")
        
        return issues


class ZeroHeuristic(HeuristicFunction):
    """Zero heuristic for Dijkstra's algorithm."""
    
    def calculate(self, start: GridPoint, end: GridPoint) -> float:
        """Always return zero."""
        return 0.0