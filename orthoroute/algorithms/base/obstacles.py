"""Obstacle management for routing algorithms."""
import logging
from typing import Set, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from ...domain.services.pathfinder import GridPoint

logger = logging.getLogger(__name__)


class ObstacleType(Enum):
    """Types of obstacles in routing."""
    COMPONENT = "component"
    TRACE = "trace" 
    VIA = "via"
    KEEPOUT = "keepout"
    BOARD_EDGE = "board_edge"


@dataclass
class Obstacle:
    """Represents an obstacle in the routing grid."""
    position: GridPoint
    obstacle_type: ObstacleType
    size: Tuple[int, int] = (1, 1)  # width, height in grid units
    net_id: str = None  # Net that created this obstacle (for traces/vias)


class ObstacleManager:
    """Manages obstacles in the routing grid."""
    
    def __init__(self):
        """Initialize obstacle manager."""
        self.obstacles: Dict[GridPoint, Obstacle] = {}
        self.obstacle_sets: Dict[ObstacleType, Set[GridPoint]] = {
            obstacle_type: set() for obstacle_type in ObstacleType
        }
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the grid."""
        # Add main position
        self.obstacles[obstacle.position] = obstacle
        self.obstacle_sets[obstacle.obstacle_type].add(obstacle.position)
        
        # Add extended positions if size > 1x1
        if obstacle.size != (1, 1):
            width, height = obstacle.size
            for dx in range(width):
                for dy in range(height):
                    if dx == 0 and dy == 0:
                        continue  # Already added main position
                    
                    extended_pos = GridPoint(
                        obstacle.position.x + dx,
                        obstacle.position.y + dy,
                        obstacle.position.layer
                    )
                    
                    extended_obstacle = Obstacle(
                        position=extended_pos,
                        obstacle_type=obstacle.obstacle_type,
                        size=(1, 1),
                        net_id=obstacle.net_id
                    )
                    
                    self.obstacles[extended_pos] = extended_obstacle
                    self.obstacle_sets[obstacle.obstacle_type].add(extended_pos)
    
    def remove_obstacle(self, position: GridPoint):
        """Remove obstacle at position."""
        if position in self.obstacles:
            obstacle = self.obstacles[position]
            del self.obstacles[position]
            self.obstacle_sets[obstacle.obstacle_type].discard(position)
    
    def is_obstacle(self, position: GridPoint) -> bool:
        """Check if position has an obstacle."""
        return position in self.obstacles
    
    def get_obstacle(self, position: GridPoint) -> Obstacle:
        """Get obstacle at position."""
        return self.obstacles.get(position)
    
    def get_obstacles_by_type(self, obstacle_type: ObstacleType) -> Set[GridPoint]:
        """Get all obstacles of a specific type."""
        return self.obstacle_sets[obstacle_type].copy()
    
    def get_obstacles_by_net(self, net_id: str) -> List[Obstacle]:
        """Get all obstacles created by a specific net."""
        return [
            obstacle for obstacle in self.obstacles.values()
            if obstacle.net_id == net_id
        ]
    
    def clear_obstacles_by_net(self, net_id: str):
        """Clear all obstacles created by a specific net."""
        to_remove = []
        
        for position, obstacle in self.obstacles.items():
            if obstacle.net_id == net_id:
                to_remove.append(position)
        
        for position in to_remove:
            self.remove_obstacle(position)
    
    def clear_obstacles_by_type(self, obstacle_type: ObstacleType):
        """Clear all obstacles of a specific type."""
        to_remove = list(self.obstacle_sets[obstacle_type])
        
        for position in to_remove:
            self.remove_obstacle(position)
    
    def clear_all_obstacles(self):
        """Clear all obstacles."""
        self.obstacles.clear()
        for obstacle_set in self.obstacle_sets.values():
            obstacle_set.clear()
    
    def get_obstacle_count(self) -> Dict[ObstacleType, int]:
        """Get count of obstacles by type."""
        return {
            obstacle_type: len(obstacle_set)
            for obstacle_type, obstacle_set in self.obstacle_sets.items()
        }
    
    def is_position_blocked(self, position: GridPoint, net_id: str = None) -> bool:
        """Check if position is blocked for routing.
        
        Args:
            position: Position to check
            net_id: Net attempting to route (can route over own traces)
        
        Returns:
            True if position is blocked for this net
        """
        if position not in self.obstacles:
            return False
        
        obstacle = self.obstacles[position]
        
        # Can route over own traces/vias
        if net_id and obstacle.net_id == net_id:
            if obstacle.obstacle_type in [ObstacleType.TRACE, ObstacleType.VIA]:
                return False
        
        return True