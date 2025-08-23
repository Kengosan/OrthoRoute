"""Application service for orchestrating routing operations."""
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from ...domain.models.board import Board, Net
from ...domain.models.routing import RoutingResult, RoutingStatistics
from ...domain.services.routing_engine import RoutingEngine, RoutingStrategy
from ...domain.events.routing_events import RoutingProgress
from ..interfaces.board_repository import BoardRepository
from ..interfaces.routing_repository import RoutingRepository
from ..interfaces.event_publisher import EventPublisher

logger = logging.getLogger(__name__)


class RoutingOrchestrator:
    """Application service that orchestrates routing operations."""
    
    def __init__(self, 
                 routing_engine: RoutingEngine,
                 board_repository: BoardRepository,
                 routing_repository: RoutingRepository,
                 event_publisher: EventPublisher):
        """Initialize routing orchestrator."""
        self.routing_engine = routing_engine
        self.board_repository = board_repository
        self.routing_repository = routing_repository
        self.event_publisher = event_publisher
        
        # State
        self.current_board: Optional[Board] = None
        self.routing_active = False
        self.cancelled = False
    
    def initialize_with_board(self, board_id: str) -> bool:
        """Initialize orchestrator with a specific board."""
        try:
            board = self.board_repository.get_board(board_id)
            if not board:
                logger.error(f"Board {board_id} not found")
                return False
            
            self.current_board = board
            self.routing_engine.initialize(board)
            logger.info(f"Initialized routing orchestrator with board {board.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize with board {board_id}: {e}")
            return False
    
    def route_net_sync(self, net_id: str, timeout: float = 10.0) -> RoutingResult:
        """Route a single net synchronously."""
        if not self.current_board:
            return RoutingResult.failure_result("No board loaded")
        
        net = self.current_board.get_net(net_id)
        if not net:
            return RoutingResult.failure_result(f"Net {net_id} not found")
        
        try:
            result = self.routing_engine.route_net(net, timeout)
            
            if result.success and result.route:
                # Save route
                self.routing_repository.save_route(result.route)
                logger.info(f"Successfully routed net {net.name}")
            else:
                logger.warning(f"Failed to route net {net.name}: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing net {net.name}: {e}")
            return RoutingResult.failure_result(str(e))
    
    async def route_net_async(self, net_id: str, timeout: float = 10.0) -> RoutingResult:
        """Route a single net asynchronously."""
        # Run synchronous routing in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.route_net_sync, net_id, timeout)
    
    def route_all_nets_sync(self, 
                           timeout_per_net: float = 5.0,
                           total_timeout: float = 300.0,
                           strategy: Optional[RoutingStrategy] = None,
                           priority_nets: Optional[List[str]] = None) -> RoutingStatistics:
        """Route all nets synchronously with progress reporting."""
        if not self.current_board:
            return RoutingStatistics(algorithm_used="failed")
        
        try:
            self.routing_active = True
            self.cancelled = False
            
            routable_nets = self.current_board.get_routable_nets()
            
            # Apply priority ordering
            if priority_nets:
                priority_set = set(priority_nets)
                routable_nets.sort(key=lambda net: (
                    0 if net.id in priority_set else 1,
                    net.name
                ))
            else:
                routable_nets = self.routing_engine.sort_nets_by_routing_priority(routable_nets)
            
            logger.info(f"Starting to route {len(routable_nets)} nets")
            
            start_time = datetime.now()
            nets_completed = 0
            nets_failed = 0
            total_length = 0.0
            total_vias = 0
            
            # Route each net with progress reporting
            for i, net in enumerate(routable_nets):
                if self.cancelled:
                    logger.info("Routing cancelled by user")
                    break
                
                # Check total timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > total_timeout:
                    logger.warning("Total timeout reached, stopping routing")
                    break
                
                # Publish progress event
                progress_event = RoutingProgress(
                    event_id=f"progress_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    current_net=net.name,
                    nets_completed=nets_completed,
                    nets_failed=nets_failed,
                    total_nets=len(routable_nets),
                    current_algorithm=strategy.value if strategy else "default"
                )
                self.event_publisher.publish(progress_event)
                
                # Route the net
                result = self.routing_engine.route_net(net, timeout_per_net)
                
                if result.success and result.route:
                    nets_completed += 1
                    total_length += result.route.total_length
                    total_vias += result.route.via_count
                    
                    # Save route
                    self.routing_repository.save_route(result.route)
                else:
                    nets_failed += 1
                    logger.warning(f"Failed to route net {net.name}: {result.error_message}")
                
                # Progress reporting every 10 nets
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {nets_completed + nets_failed}/{len(routable_nets)} nets processed")
            
            # Final statistics
            total_time = (datetime.now() - start_time).total_seconds()
            
            statistics = RoutingStatistics(
                nets_attempted=len(routable_nets),
                nets_routed=nets_completed,
                nets_failed=nets_failed,
                total_length=total_length,
                total_vias=total_vias,
                total_time=total_time,
                algorithm_used=strategy.value if strategy else "default"
            )
            
            logger.info(f"Routing completed: {nets_completed}/{len(routable_nets)} nets "
                       f"({statistics.success_rate:.1%} success rate)")
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error in route_all_nets: {e}")
            return RoutingStatistics(algorithm_used="failed")
        finally:
            self.routing_active = False
    
    async def route_all_nets_async(self, 
                                  timeout_per_net: float = 5.0,
                                  total_timeout: float = 300.0,
                                  strategy: Optional[RoutingStrategy] = None,
                                  priority_nets: Optional[List[str]] = None) -> RoutingStatistics:
        """Route all nets asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.route_all_nets_sync, 
            timeout_per_net, 
            total_timeout, 
            strategy, 
            priority_nets
        )
    
    def cancel_routing(self) -> None:
        """Cancel ongoing routing operation."""
        self.cancelled = True
        logger.info("Routing cancellation requested")
    
    def clear_routes(self, net_ids: Optional[List[str]] = None) -> int:
        """Clear routes."""
        try:
            if net_ids:
                count = 0
                for net_id in net_ids:
                    count += self.routing_repository.delete_routes_by_net(net_id)
                logger.info(f"Cleared routes for {len(net_ids)} nets")
                return count
            else:
                count = self.routing_repository.clear_all_routes()
                self.routing_engine.clear_routes()
                logger.info(f"Cleared all routes ({count} removed)")
                return count
                
        except Exception as e:
            logger.error(f"Error clearing routes: {e}")
            return 0
    
    def get_routing_status(self) -> Dict[str, Any]:
        """Get current routing status."""
        statistics = self.routing_repository.get_routing_statistics()
        
        return {
            'routing_active': self.routing_active,
            'board_loaded': self.current_board is not None,
            'board_name': self.current_board.name if self.current_board else None,
            'statistics': statistics.to_dict(),
            'engine_strategy': self.routing_engine.strategy.value,
            'engine_supports_gpu': self.routing_engine.supports_gpu
        }
    
    def validate_routing_readiness(self) -> List[str]:
        """Validate that system is ready for routing."""
        issues = []
        
        if not self.current_board:
            issues.append("No board loaded")
        else:
            # Check board integrity
            board_issues = self.current_board.validate_integrity()
            issues.extend(board_issues)
            
            # Check if there are routable nets
            routable_nets = self.current_board.get_routable_nets()
            if not routable_nets:
                issues.append("No routable nets found")
        
        return issues