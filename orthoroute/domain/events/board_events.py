"""Domain events related to board operations."""
from dataclasses import dataclass
from typing import List, Any
from datetime import datetime

from .routing_events import DomainEvent
from ..models.board import Board, Component, Net, Layer
from ..models.constraints import DRCConstraints


@dataclass(frozen=True)
class BoardLoaded(DomainEvent):
    """Event fired when board data is loaded."""
    board_id: str
    board_name: str
    component_count: int
    net_count: int
    layer_count: int
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class ComponentsChanged(DomainEvent):
    """Event fired when components are added/removed/modified."""
    board_id: str
    added_components: List[str]
    removed_components: List[str]
    modified_components: List[str]
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class NetsChanged(DomainEvent):
    """Event fired when nets are modified."""
    board_id: str
    added_nets: List[str]
    removed_nets: List[str]
    modified_nets: List[str]
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class LayersChanged(DomainEvent):
    """Event fired when layer stack is modified."""
    board_id: str
    old_layer_count: int
    new_layer_count: int
    added_layers: List[str]
    removed_layers: List[str]
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class ConstraintsUpdated(DomainEvent):
    """Event fired when DRC constraints are updated."""
    board_id: str
    constraints: DRCConstraints
    changed_fields: List[str]
    
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class BoardValidated(DomainEvent):
    """Event fired when board validation completes."""
    board_id: str
    is_valid: bool
    issues: List[str]
    
    def __post_init__(self):
        super().__post_init__()