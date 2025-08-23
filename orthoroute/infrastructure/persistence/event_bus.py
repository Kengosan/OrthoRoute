"""Event bus implementation."""
import logging
from typing import Callable, Type, List, Dict, Any
from collections import defaultdict
from datetime import datetime

from ...application.interfaces.event_publisher import EventPublisher
from ...domain.events.routing_events import DomainEvent

logger = logging.getLogger(__name__)


class EventBus(EventPublisher):
    """In-memory event bus implementation."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize event bus."""
        self._subscribers: Dict[Type[DomainEvent], List[Callable]] = defaultdict(list)
        self._event_history: List[DomainEvent] = []
        self._max_history = max_history
    
    def publish(self, event: DomainEvent) -> None:
        """Publish a domain event."""
        try:
            # Add to history
            self._event_history.append(event)
            
            # Trim history if needed
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            
            # Notify subscribers
            event_type = type(event)
            subscribers = self._subscribers.get(event_type, [])
            
            logger.debug(f"Publishing event {event_type.__name__} to {len(subscribers)} subscribers")
            
            for handler in subscribers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type.__name__}: {e}")
            
            # Also notify any subscribers to the base DomainEvent type
            base_subscribers = self._subscribers.get(DomainEvent, [])
            for handler in base_subscribers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in base event handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error publishing event {type(event).__name__}: {e}")
    
    def subscribe(self, event_type: Type[DomainEvent], handler: Callable[[DomainEvent], None]) -> None:
        """Subscribe to an event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Added subscriber for {event_type.__name__}")
    
    def unsubscribe(self, event_type: Type[DomainEvent], handler: Callable[[DomainEvent], None]) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Removed subscriber for {event_type.__name__}")
            except ValueError:
                logger.warning(f"Handler not found in subscribers for {event_type.__name__}")
    
    def clear_subscribers(self, event_type: Type[DomainEvent] = None) -> None:
        """Clear subscribers for event type (or all if None)."""
        if event_type is None:
            self._subscribers.clear()
            logger.info("Cleared all event subscribers")
        else:
            self._subscribers[event_type].clear()
            logger.debug(f"Cleared subscribers for {event_type.__name__}")
    
    def get_event_history(self, count: int = 100) -> List[DomainEvent]:
        """Get recent event history."""
        return self._event_history[-count:] if count < len(self._event_history) else self._event_history.copy()
    
    def get_subscriber_count(self, event_type: Type[DomainEvent]) -> int:
        """Get number of subscribers for event type."""
        return len(self._subscribers.get(event_type, []))
    
    def get_all_subscriber_counts(self) -> Dict[str, int]:
        """Get subscriber counts for all event types."""
        return {
            event_type.__name__: len(handlers)
            for event_type, handlers in self._subscribers.items()
        }
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.debug("Cleared event history")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        event_type_counts = defaultdict(int)
        for event in self._event_history:
            event_type_counts[type(event).__name__] += 1
        
        return {
            'total_events_published': len(self._event_history),
            'event_types': dict(event_type_counts),
            'total_subscribers': sum(len(handlers) for handlers in self._subscribers.values()),
            'subscriber_counts': self.get_all_subscriber_counts(),
            'history_size': len(self._event_history),
            'max_history': self._max_history
        }