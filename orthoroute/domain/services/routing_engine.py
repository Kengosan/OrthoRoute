"""Abstract routing engine interface and strategy pattern."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum

from ..models.board import Board, Net, Pad
from ..models.routing import Route, RoutingResult, RoutingStatistics
from ..models.constraints import DRCConstraints


class RoutingStrategy(Enum):
    """Available routing strategies."""
    LEE_WAVEFRONT = "lee_wavefront"
    MANHATTAN_ASTAR = "manhattan_astar"
    GENETIC_ALGORITHM = "genetic"


class RoutingEngine(ABC):
    """Abstract base class for routing engines."""
    
    def __init__(self, constraints: DRCConstraints):
        """Initialize routing engine with DRC constraints."""
        self.constraints = constraints
    
    @abstractmethod
    def initialize(self, board: Board) -> None:
        """Initialize routing engine with board data."""
        pass
    
    @abstractmethod
    def route_net(self, net: Net, timeout: float = 10.0) -> RoutingResult:
        """Route a single net."""
        pass
    
    @abstractmethod
    def route_two_pads(self, pad_a: Pad, pad_b: Pad, net_id: str, 
                      timeout: float = 5.0) -> RoutingResult:
        """Route between two specific pads."""
        pass
    
    @abstractmethod
    def route_all_nets(self, nets: List[Net], 
                      timeout_per_net: float = 5.0,
                      total_timeout: float = 300.0) -> RoutingStatistics:
        """Route all provided nets."""
        pass
    
    @abstractmethod
    def clear_routes(self) -> None:
        """Clear all routing data."""
        pass
    
    @abstractmethod
    def get_routed_tracks(self) -> List[Dict[str, Any]]:
        """Get all routed tracks."""
        pass
    
    @abstractmethod
    def get_routed_vias(self) -> List[Dict[str, Any]]:
        """Get all routed vias."""
        pass
    
    @abstractmethod
    def get_routing_statistics(self) -> RoutingStatistics:
        """Get current routing statistics."""
        pass
    
    @property
    @abstractmethod
    def strategy(self) -> RoutingStrategy:
        """Get the routing strategy used by this engine."""
        pass
    
    @property
    @abstractmethod
    def supports_gpu(self) -> bool:
        """Check if engine supports GPU acceleration."""
        pass
    
    def validate_net(self, net: Net) -> List[str]:
        """Validate that a net can be routed."""
        issues = []
        
        if not net.is_routable:
            issues.append(f"Net {net.name} has fewer than 2 pads")
        
        # Check pad positions are valid
        for pad in net.pads:
            if pad.position.x == 0 and pad.position.y == 0:
                issues.append(f"Pad {pad.id} in net {net.name} has invalid position (0,0)")
        
        return issues
    
    def estimate_routing_complexity(self, net: Net) -> float:
        """Estimate routing complexity score for a net."""
        if not net.is_routable:
            return 0.0
        
        # Base complexity from number of pads
        complexity = len(net.pads) ** 2
        
        # Add distance factor
        bounds = net.get_bounds()
        area = bounds.width * bounds.height
        complexity *= (1 + area / 100.0)  # Normalize by 10cm x 10cm area
        
        # Add minimum distance factor (closer pads are harder to route)
        min_distance = net.calculate_min_distance()
        if min_distance < 1.0:  # Less than 1mm
            complexity *= (2.0 - min_distance)  # Penalty for very close pads
        
        return complexity
    
    def sort_nets_by_routing_priority(self, nets: List[Net]) -> List[Net]:
        """Sort nets by routing priority (simpler nets first)."""
        return sorted(nets, key=lambda net: (
            self.estimate_routing_complexity(net),
            net.name  # Secondary sort by name for consistency
        ))


class RoutingEngineFactory:
    """Factory for creating routing engines."""
    
    _engines: Dict[RoutingStrategy, type] = {}
    
    @classmethod
    def register_engine(cls, strategy: RoutingStrategy, engine_class: type) -> None:
        """Register a routing engine implementation."""
        cls._engines[strategy] = engine_class
    
    @classmethod
    def create_engine(cls, strategy: RoutingStrategy, 
                     constraints: DRCConstraints,
                     **kwargs) -> RoutingEngine:
        """Create a routing engine instance."""
        if strategy not in cls._engines:
            raise ValueError(f"Unknown routing strategy: {strategy}")
        
        engine_class = cls._engines[strategy]
        return engine_class(constraints, **kwargs)
    
    @classmethod
    def get_available_strategies(cls) -> List[RoutingStrategy]:
        """Get list of available routing strategies."""
        return list(cls._engines.keys())
    
    @classmethod
    def is_strategy_available(cls, strategy: RoutingStrategy) -> bool:
        """Check if a strategy is available."""
        return strategy in cls._engines