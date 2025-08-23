"""Query handlers for routing data retrieval."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

from ...domain.models.routing import RoutingStatistics, Route
from ...domain.services.routing_engine import RoutingEngine


@dataclass
class Query:
    """Base query class."""
    query_id: str
    timestamp: datetime
    user_id: Optional[str] = None


@dataclass
class GetRoutingStatsQuery(Query):
    """Query to get routing statistics."""
    board_id: Optional[str] = None


@dataclass
class GetNetRoutesQuery(Query):
    """Query to get routes for specific nets."""
    net_ids: Optional[List[str]] = None  # None means all nets
    include_failed: bool = False


@dataclass
class GetRouteQuery(Query):
    """Query to get a specific route."""
    route_id: str


@dataclass
class GetRoutingVisualizationQuery(Query):
    """Query to get visualization data."""
    layer_filter: Optional[List[str]] = None
    net_filter: Optional[List[str]] = None
    show_active_only: bool = False


class QueryHandler(ABC):
    """Base class for query handlers."""
    
    @abstractmethod
    def handle(self, query: Query) -> Any:
        """Handle the query."""
        pass


class GetRoutingStatsQueryHandler(QueryHandler):
    """Handler for GetRoutingStatsQuery."""
    
    def __init__(self, routing_engine: RoutingEngine):
        self.routing_engine = routing_engine
    
    def handle(self, query: GetRoutingStatsQuery) -> RoutingStatistics:
        """Handle routing statistics query."""
        return self.routing_engine.get_routing_statistics()


class GetNetRoutesQueryHandler(QueryHandler):
    """Handler for GetNetRoutesQuery."""
    
    def __init__(self, route_repository):
        self.route_repository = route_repository
    
    def handle(self, query: GetNetRoutesQuery) -> List[Route]:
        """Handle net routes query."""
        if query.net_ids:
            routes = []
            for net_id in query.net_ids:
                route = self.route_repository.get_route_by_net(net_id)
                if route:
                    routes.append(route)
            return routes
        else:
            return self.route_repository.get_all_routes(include_failed=query.include_failed)


class GetRouteQueryHandler(QueryHandler):
    """Handler for GetRouteQuery."""
    
    def __init__(self, route_repository):
        self.route_repository = route_repository
    
    def handle(self, query: GetRouteQuery) -> Optional[Route]:
        """Handle specific route query."""
        return self.route_repository.get_route(query.route_id)


class GetRoutingVisualizationQueryHandler(QueryHandler):
    """Handler for GetRoutingVisualizationQuery."""
    
    def __init__(self, routing_engine: RoutingEngine):
        self.routing_engine = routing_engine
    
    def handle(self, query: GetRoutingVisualizationQuery) -> Dict[str, Any]:
        """Handle routing visualization query."""
        visualization_data = {
            'tracks': [],
            'vias': [],
            'progress': {},
            'statistics': {}
        }
        
        # Get tracks
        all_tracks = self.routing_engine.get_routed_tracks()
        
        # Apply layer filter
        if query.layer_filter:
            all_tracks = [t for t in all_tracks if t.get('layer') in query.layer_filter]
        
        # Apply net filter
        if query.net_filter:
            all_tracks = [t for t in all_tracks if t.get('net') in query.net_filter]
        
        visualization_data['tracks'] = all_tracks
        
        # Get vias
        all_vias = self.routing_engine.get_routed_vias()
        
        # Apply same filters to vias
        if query.layer_filter:
            all_vias = [v for v in all_vias 
                       if any(layer in query.layer_filter for layer in v.get('layers', []))]
        
        if query.net_filter:
            all_vias = [v for v in all_vias if v.get('net') in query.net_filter]
        
        visualization_data['vias'] = all_vias
        
        # Get statistics
        stats = self.routing_engine.get_routing_statistics()
        visualization_data['statistics'] = stats.to_dict()
        
        return visualization_data