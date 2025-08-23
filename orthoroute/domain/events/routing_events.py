"""Domain events related to routing operations."""
from dataclasses import dataclass
from typing import Optional, List, Any
from datetime import datetime

from ..models.routing import Route, RoutingResult, RoutingStatistics


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events."""
    timestamp: datetime
    event_id: str
    
    def __post_init__(self):
        if not self.timestamp:
            object.__setattr__(self, 'timestamp', datetime.now())


@dataclass(frozen=True)
class RoutingStarted(DomainEvent):
    """Event fired when routing session begins."""
    total_nets: int
    algorithm: str
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class RoutingCompleted(DomainEvent):
    """Event fired when routing session completes."""
    statistics: RoutingStatistics
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class RoutingFailed(DomainEvent):
    """Event fired when routing session fails."""
    error_message: str
    partial_statistics: Optional[RoutingStatistics] = None
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class NetRouted(DomainEvent):
    """Event fired when a single net is successfully routed."""
    net_id: str
    net_name: str
    route: Route
    execution_time: float
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class NetRoutingFailed(DomainEvent):
    """Event fired when a single net fails to route."""
    net_id: str
    net_name: str
    error_message: str
    attempts_made: int
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class RouteCleared(DomainEvent):
    """Event fired when routes are cleared."""
    net_id: Optional[str] = None  # None means all routes cleared
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class RipupStarted(DomainEvent):
    """Event fired when rip-up and repair begins."""
    conflicting_nets: List[str]
    target_net: str
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class VisualizationUpdate(DomainEvent):
    """Event fired for visualization updates."""
    update_type: str  # 'track', 'via', 'progress'
    data: Any
    net_id: str
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class RoutingProgress(DomainEvent):
    """Event fired for routing progress updates."""
    current_net: str
    nets_completed: int
    nets_failed: int
    total_nets: int
    current_algorithm: str
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_nets == 0:
            return 0.0
        return (self.nets_completed + self.nets_failed) / self.total_nets * 100
    
    def __post_init__(self):
        super().__post_init__()