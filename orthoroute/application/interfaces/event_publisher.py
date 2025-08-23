"""Abstract event publisher interface."""
from abc import ABC, abstractmethod
from typing import Callable, Type, List

from ...domain.events.routing_events import DomainEvent


class EventPublisher(ABC):
    """Abstract interface for publishing domain events."""
    
    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        """Publish a domain event."""
        pass
    
    @abstractmethod
    def subscribe(self, event_type: Type[DomainEvent], handler: Callable[[DomainEvent], None]) -> None:
        """Subscribe to an event type."""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: Type[DomainEvent], handler: Callable[[DomainEvent], None]) -> None:
        """Unsubscribe from an event type."""
        pass
    
    @abstractmethod
    def clear_subscribers(self, event_type: Type[DomainEvent] = None) -> None:
        """Clear subscribers for event type (or all if None)."""
        pass
    
    @abstractmethod
    def get_event_history(self, count: int = 100) -> List[DomainEvent]:
        """Get recent event history."""
        pass