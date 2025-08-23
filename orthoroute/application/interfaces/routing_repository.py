"""Abstract routing repository interface."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from ...domain.models.routing import Route, RoutingStatistics


class RoutingRepository(ABC):
    """Abstract repository for routing data access."""
    
    @abstractmethod
    def get_route(self, route_id: str) -> Optional[Route]:
        """Get route by ID."""
        pass
    
    @abstractmethod
    def get_route_by_net(self, net_id: str) -> Optional[Route]:
        """Get route by net ID."""
        pass
    
    @abstractmethod
    def save_route(self, route: Route) -> None:
        """Save route data."""
        pass
    
    @abstractmethod
    def delete_route(self, route_id: str) -> bool:
        """Delete route by ID."""
        pass
    
    @abstractmethod
    def delete_routes_by_net(self, net_id: str) -> int:
        """Delete all routes for a net. Returns count deleted."""
        pass
    
    @abstractmethod
    def get_all_routes(self, include_failed: bool = False) -> List[Route]:
        """Get all routes."""
        pass
    
    @abstractmethod
    def get_routes_by_board(self, board_id: str) -> List[Route]:
        """Get all routes for a specific board."""
        pass
    
    @abstractmethod
    def clear_all_routes(self) -> int:
        """Clear all routes. Returns count deleted."""
        pass
    
    @abstractmethod
    def get_routing_statistics(self, board_id: Optional[str] = None) -> RoutingStatistics:
        """Get routing statistics."""
        pass
    
    @abstractmethod
    def route_exists(self, route_id: str) -> bool:
        """Check if route exists."""
        pass