"""Command handlers for routing operations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ...domain.models.board import Board, Net
from ...domain.models.routing import RoutingResult, RoutingStatistics
from ...domain.services.routing_engine import RoutingEngine, RoutingStrategy
from ...domain.events.routing_events import (
    RoutingStarted, RoutingCompleted, RoutingFailed, 
    NetRouted, NetRoutingFailed, RouteCleared
)

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """Base command class."""
    command_id: str
    timestamp: datetime
    user_id: Optional[str] = None


@dataclass
class RouteNetCommand(Command):
    """Command to route a single net."""
    net_id: str
    timeout: float = 10.0
    strategy: Optional[RoutingStrategy] = None


@dataclass 
class RouteAllNetsCommand(Command):
    """Command to route all nets on the board."""
    timeout_per_net: float = 5.0
    total_timeout: float = 300.0
    strategy: Optional[RoutingStrategy] = None
    priority_nets: Optional[List[str]] = None


@dataclass
class ClearRoutesCommand(Command):
    """Command to clear routes."""
    net_ids: Optional[List[str]] = None  # None means clear all


@dataclass
class RipupRepairCommand(Command):
    """Command to perform rip-up and repair."""
    target_net_id: str
    conflicting_area: Optional[Dict[str, Any]] = None


class CommandHandler(ABC):
    """Base class for command handlers."""
    
    @abstractmethod
    def handle(self, command: Command) -> Any:
        """Handle the command."""
        pass


class RouteNetCommandHandler(CommandHandler):
    """Handler for RouteNetCommand."""
    
    def __init__(self, routing_engine: RoutingEngine, event_publisher):
        self.routing_engine = routing_engine
        self.event_publisher = event_publisher
    
    def handle(self, command: RouteNetCommand) -> RoutingResult:
        """Handle net routing command."""
        logger.info(f"Routing net {command.net_id}")
        
        try:
            # Get net from board (this would come from repository)
            # For now, we'll assume the routing engine can handle net by ID
            result = self.routing_engine.route_net_by_id(command.net_id, command.timeout)
            
            if result.success:
                # Publish success event
                event = NetRouted(
                    event_id=f"net_routed_{command.net_id}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    net_id=command.net_id,
                    net_name=command.net_id,  # Would get actual name
                    route=result.route,
                    execution_time=result.execution_time
                )
                self.event_publisher.publish(event)
                logger.info(f"Successfully routed net {command.net_id}")
            else:
                # Publish failure event
                event = NetRoutingFailed(
                    event_id=f"net_failed_{command.net_id}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    net_id=command.net_id,
                    net_name=command.net_id,
                    error_message=result.error_message or "Unknown error",
                    attempts_made=1
                )
                self.event_publisher.publish(event)
                logger.warning(f"Failed to route net {command.net_id}: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling RouteNetCommand: {e}")
            return RoutingResult.failure_result(str(e))


class RouteAllNetsCommandHandler(CommandHandler):
    """Handler for RouteAllNetsCommand."""
    
    def __init__(self, routing_engine: RoutingEngine, event_publisher, board_repository):
        self.routing_engine = routing_engine
        self.event_publisher = event_publisher
        self.board_repository = board_repository
    
    def handle(self, command: RouteAllNetsCommand) -> RoutingStatistics:
        """Handle route all nets command."""
        logger.info("Starting route all nets operation")
        
        try:
            # Get board and nets
            board = self.board_repository.get_current_board()
            if not board:
                raise ValueError("No board loaded")
            
            routable_nets = board.get_routable_nets()
            
            # Apply priority ordering if specified
            if command.priority_nets:
                priority_set = set(command.priority_nets)
                routable_nets.sort(key=lambda net: (
                    0 if net.id in priority_set else 1,
                    net.name
                ))
            else:
                # Use default priority (complexity-based)
                routable_nets = self.routing_engine.sort_nets_by_routing_priority(routable_nets)
            
            # Publish start event
            start_event = RoutingStarted(
                event_id=f"routing_started_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                total_nets=len(routable_nets),
                algorithm=command.strategy.value if command.strategy else "default"
            )
            self.event_publisher.publish(start_event)
            
            # Route all nets
            statistics = self.routing_engine.route_all_nets(
                routable_nets,
                command.timeout_per_net,
                command.total_timeout
            )
            
            # Publish completion event
            completion_event = RoutingCompleted(
                event_id=f"routing_completed_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                statistics=statistics
            )
            self.event_publisher.publish(completion_event)
            
            logger.info(f"Routing completed: {statistics.nets_routed}/{statistics.nets_attempted} nets")
            return statistics
            
        except Exception as e:
            logger.error(f"Error handling RouteAllNetsCommand: {e}")
            
            # Publish failure event
            failure_event = RoutingFailed(
                event_id=f"routing_failed_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                error_message=str(e)
            )
            self.event_publisher.publish(failure_event)
            
            # Return empty statistics
            return RoutingStatistics(algorithm_used="failed")


class ClearRoutesCommandHandler(CommandHandler):
    """Handler for ClearRoutesCommand."""
    
    def __init__(self, routing_engine: RoutingEngine, event_publisher):
        self.routing_engine = routing_engine
        self.event_publisher = event_publisher
    
    def handle(self, command: ClearRoutesCommand) -> bool:
        """Handle clear routes command."""
        try:
            if command.net_ids:
                # Clear specific nets
                for net_id in command.net_ids:
                    self.routing_engine.clear_net_routes(net_id)
                    
                    # Publish event for each cleared net
                    event = RouteCleared(
                        event_id=f"route_cleared_{net_id}_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        net_id=net_id
                    )
                    self.event_publisher.publish(event)
                
                logger.info(f"Cleared routes for {len(command.net_ids)} nets")
            else:
                # Clear all routes
                self.routing_engine.clear_routes()
                
                # Publish global clear event
                event = RouteCleared(
                    event_id=f"all_routes_cleared_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    net_id=None
                )
                self.event_publisher.publish(event)
                
                logger.info("Cleared all routes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling ClearRoutesCommand: {e}")
            return False


class RipupRepairCommandHandler(CommandHandler):
    """Handler for RipupRepairCommand."""
    
    def __init__(self, routing_engine: RoutingEngine, event_publisher):
        self.routing_engine = routing_engine
        self.event_publisher = event_publisher
    
    def handle(self, command: RipupRepairCommand) -> RoutingResult:
        """Handle rip-up and repair command."""
        logger.info(f"Starting rip-up and repair for net {command.target_net_id}")
        
        try:
            # This would interact with the routing engine's rip-up functionality
            # For now, simulate the process
            conflicting_nets = self.routing_engine.find_conflicting_nets(
                command.target_net_id, 
                command.conflicting_area
            )
            
            if conflicting_nets:
                # Publish rip-up start event
                event = RipupStarted(
                    event_id=f"ripup_started_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    conflicting_nets=conflicting_nets,
                    target_net=command.target_net_id
                )
                self.event_publisher.publish(event)
                
                # Perform rip-up and repair
                result = self.routing_engine.ripup_and_repair(
                    command.target_net_id, 
                    conflicting_nets
                )
                
                return result
            else:
                # No conflicts found, try normal routing
                return self.routing_engine.route_net_by_id(command.target_net_id)
                
        except Exception as e:
            logger.error(f"Error handling RipupRepairCommand: {e}")
            return RoutingResult.failure_result(str(e))