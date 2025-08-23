"""
Routing engines for the OrthoRoute autorouter.

This module provides different routing algorithms that implement
the common BaseRouter interface:
- Lee's wavefront expansion algorithm
- Manhattan routing (future)
- A* pathfinding (future)
"""
import sys
import os

# Add src directory to Python path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try absolute imports first, fall back to relative
try:
    from routing_engines.base_router import BaseRouter, RoutingResult, RouteSegment, RoutingStats
    from routing_engines.lees_router import LeeRouter
except ImportError:
    from .base_router import BaseRouter, RoutingResult, RouteSegment, RoutingStats
    from .lees_router import LeeRouter

__all__ = [
    'BaseRouter',
    'RoutingResult', 
    'RouteSegment',
    'RoutingStats',
    'LeeRouter'
]
