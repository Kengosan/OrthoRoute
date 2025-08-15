"""
Routing engines for the OrthoRoute autorouter.

This module provides different routing algorithms that implement
the common BaseRouter interface:
- Lee's wavefront expansion algorithm
- Manhattan routing (future)
- A* pathfinding (future)
"""
from routing_engines.base_router import BaseRouter, RoutingResult, RouteSegment, RoutingStats
from routing_engines.lees_router import LeeRouter

__all__ = [
    'BaseRouter',
    'RoutingResult', 
    'RouteSegment',
    'RoutingStats',
    'LeeRouter'
]
