"""In-memory routing repository implementation."""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from ...application.interfaces.routing_repository import RoutingRepository
from ...domain.models.routing import Route, RoutingStatistics

logger = logging.getLogger(__name__)


class MemoryRoutingRepository(RoutingRepository):
    """In-memory implementation of routing repository."""
    
    def __init__(self):
        """Initialize memory routing repository."""
        self._routes: Dict[str, Route] = {}
        self._routes_by_net: Dict[str, str] = {}  # net_id -> route_id mapping
        self._statistics = RoutingStatistics()
    
    def get_route(self, route_id: str) -> Optional[Route]:
        """Get route by ID."""
        return self._routes.get(route_id)
    
    def get_route_by_net(self, net_id: str) -> Optional[Route]:
        """Get route by net ID."""
        route_id = self._routes_by_net.get(net_id)
        if route_id:
            return self._routes.get(route_id)
        return None
    
    def save_route(self, route: Route) -> None:
        """Save route data."""
        self._routes[route.id] = route
        self._routes_by_net[route.net_id] = route.id
        
        # Update statistics
        self._update_statistics()
        
        logger.debug(f"Saved route {route.id} for net {route.net_id}")
    
    def delete_route(self, route_id: str) -> bool:
        """Delete route by ID."""
        route = self._routes.get(route_id)
        if route:
            del self._routes[route_id]
            
            # Remove from net mapping
            if route.net_id in self._routes_by_net:
                del self._routes_by_net[route.net_id]
            
            # Update statistics
            self._update_statistics()
            
            logger.debug(f"Deleted route {route_id}")
            return True
        
        return False
    
    def delete_routes_by_net(self, net_id: str) -> int:
        """Delete all routes for a net. Returns count deleted."""
        route_id = self._routes_by_net.get(net_id)
        if route_id and route_id in self._routes:
            del self._routes[route_id]
            del self._routes_by_net[net_id]
            
            # Update statistics
            self._update_statistics()
            
            logger.debug(f"Deleted route for net {net_id}")
            return 1
        
        return 0
    
    def get_all_routes(self, include_failed: bool = False) -> List[Route]:
        """Get all routes."""
        routes = list(self._routes.values())
        
        if not include_failed:
            # Filter out failed routes (routes with no segments)
            routes = [route for route in routes if len(route.segments) > 0]
        
        return routes
    
    def get_routes_by_board(self, board_id: str) -> List[Route]:
        """Get all routes for a specific board."""
        # In this simple implementation, all routes belong to current board
        # A more sophisticated implementation would track board associations
        return self.get_all_routes()
    
    def clear_all_routes(self) -> int:
        """Clear all routes. Returns count deleted."""
        count = len(self._routes)
        self._routes.clear()
        self._routes_by_net.clear()
        
        # Reset statistics
        self._statistics = RoutingStatistics()
        
        logger.info(f"Cleared {count} routes")
        return count
    
    def get_routing_statistics(self, board_id: Optional[str] = None) -> RoutingStatistics:
        """Get routing statistics."""
        return self._statistics
    
    def route_exists(self, route_id: str) -> bool:
        """Check if route exists."""
        return route_id in self._routes
    
    def _update_statistics(self):
        """Update routing statistics based on current routes."""
        routes = list(self._routes.values())
        
        if not routes:
            self._statistics = RoutingStatistics()
            return
        
        # Calculate statistics
        total_length = sum(route.total_length for route in routes)
        total_vias = sum(route.via_count for route in routes)
        
        # Count successful routes (routes with segments)
        successful_routes = [route for route in routes if len(route.segments) > 0]
        
        self._statistics = RoutingStatistics(
            nets_attempted=len(routes),
            nets_routed=len(successful_routes),
            nets_failed=len(routes) - len(successful_routes),
            total_length=total_length,
            total_vias=total_vias,
            total_time=0.0,  # Time tracking would need to be implemented
            algorithm_used="mixed"
        )
    
    def get_routes_summary(self) -> Dict[str, Any]:
        """Get summary of routes in repository."""
        routes = list(self._routes.values())
        
        if not routes:
            return {
                'total_routes': 0,
                'successful_routes': 0,
                'failed_routes': 0,
                'total_length': 0.0,
                'total_vias': 0,
                'layers_used': []
            }
        
        successful_routes = [route for route in routes if len(route.segments) > 0]
        failed_routes = [route for route in routes if len(route.segments) == 0]
        
        # Get all layers used
        all_layers = set()
        for route in successful_routes:
            all_layers.update(route.layers_used)
        
        return {
            'total_routes': len(routes),
            'successful_routes': len(successful_routes),
            'failed_routes': len(failed_routes),
            'total_length': sum(route.total_length for route in successful_routes),
            'total_vias': sum(route.via_count for route in successful_routes),
            'layers_used': list(all_layers),
            'nets_routed': list(self._routes_by_net.keys())
        }