"""Manhattan routing engine implementation."""
import logging
import time
import math
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime

from ...domain.models.board import Board, Net, Pad, Bounds, Coordinate
from ...domain.models.routing import Route, Segment, Via, RoutingResult, RoutingStatistics, SegmentType, ViaType
from ...domain.models.constraints import DRCConstraints
from ...domain.services.routing_engine import RoutingEngine, RoutingStrategy
from ...domain.services.pathfinder import PathfindingService, GridPoint, ManhattanHeuristic
from ...application.interfaces.gpu_provider import GPUProvider

from ..base.grid import RoutingGrid, CellState
from .layer_assignment import LayerDirectionManager
from .astar import ManhattanAStarPathfinder

logger = logging.getLogger(__name__)


class ManhattanRoutingEngine(RoutingEngine):
    """Manhattan routing engine with GPU acceleration."""
    
    def __init__(self, constraints: DRCConstraints, gpu_provider: Optional[GPUProvider] = None):
        """Initialize Manhattan routing engine."""
        super().__init__(constraints)
        
        self.gpu_provider = gpu_provider
        self.board: Optional[Board] = None
        self.routing_grid: Optional[RoutingGrid] = None
        self.layer_manager: Optional[LayerDirectionManager] = None
        self.pathfinder = ManhattanAStarPathfinder()
        
        # Routing state
        self.routed_nets: Dict[str, Route] = {}
        self.failed_nets: Set[str] = set()
        self.nets_attempted = 0
        self.nets_routed = 0
        self.nets_failed = 0
        
        # Configuration
        self.default_grid_resolution = 0.4  # mm
        self.default_board_margin = 3.0  # mm
        
        logger.info("Manhattan routing engine initialized")
    
    @property
    def strategy(self) -> RoutingStrategy:
        """Get the routing strategy."""
        return RoutingStrategy.MANHATTAN_ASTAR
    
    @property
    def supports_gpu(self) -> bool:
        """Check if engine supports GPU acceleration."""
        return self.gpu_provider is not None and self.gpu_provider.is_available()
    
    def initialize(self, board: Board) -> None:
        """Initialize routing engine with board data."""
        try:
            self.board = board
            
            # Get routing layers (exclude F.Cu which is for escape routing)
            routing_layers = [layer.name for layer in board.get_routing_layers() 
                            if layer.name != 'F.Cu']
            
            if not routing_layers:
                # Default layer stack
                routing_layers = ['In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'In5.Cu',
                                'In6.Cu', 'In7.Cu', 'In8.Cu', 'In9.Cu', 'In10.Cu', 'B.Cu']
                logger.warning("No routing layers found, using default 11-layer stack")
            
            # Calculate board bounds with margin
            board_bounds = board.get_bounds()
            routing_bounds = Bounds(
                board_bounds.min_x - self.default_board_margin,
                board_bounds.min_y - self.default_board_margin,
                board_bounds.max_x + self.default_board_margin,
                board_bounds.max_y + self.default_board_margin
            )
            
            # Initialize routing grid
            self.routing_grid = RoutingGrid(
                bounds=routing_bounds,
                layer_names=routing_layers,
                resolution=self.default_grid_resolution,
                gpu_provider=self.gpu_provider
            )
            
            # Initialize layer direction manager
            self.layer_manager = LayerDirectionManager(routing_layers)
            
            # Mark obstacles from board
            self.routing_grid.mark_obstacles_from_board(board)
            
            # Clear previous routing state
            self.routed_nets.clear()
            self.failed_nets.clear()
            self.nets_attempted = 0
            self.nets_routed = 0
            self.nets_failed = 0
            
            logger.info(f"Initialized Manhattan router for board with {len(routing_layers)} layers")
            
        except Exception as e:
            logger.error(f"Failed to initialize Manhattan routing engine: {e}")
            raise
    
    def route_net(self, net: Net, timeout: float = 10.0) -> RoutingResult:
        """Route a single net."""
        if not self.routing_grid or not self.layer_manager:
            return RoutingResult.failure_result("Routing engine not initialized")
        
        start_time = time.time()
        self.nets_attempted += 1
        
        try:
            # Validate net can be routed
            validation_issues = self.validate_net(net)
            if validation_issues:
                error_msg = f"Net validation failed: {'; '.join(validation_issues)}"
                return RoutingResult.failure_result(error_msg)
            
            # Create escape routes on F.Cu
            escape_segments = self._create_escape_routes(net)
            
            # Route between grid entry points
            grid_segments, vias = self._route_grid_connections(net, timeout)
            
            if not grid_segments:
                self.nets_failed += 1
                self.failed_nets.add(net.id)
                return RoutingResult.failure_result("Failed to find grid path")
            
            # Create complete route
            all_segments = escape_segments + grid_segments
            route = Route(
                id=f"route_{net.id}_{datetime.now().timestamp()}",
                net_id=net.id,
                segments=all_segments,
                vias=vias
            )
            
            # Store successful route
            self.routed_nets[net.id] = route
            self.nets_routed += 1
            
            execution_time = time.time() - start_time
            
            logger.info(f"Successfully routed net {net.name} in {execution_time:.3f}s")
            
            return RoutingResult.success_result(
                route=route,
                execution_time=execution_time,
                algorithm=self.strategy.value
            )
            
        except Exception as e:
            self.nets_failed += 1
            self.failed_nets.add(net.id)
            execution_time = time.time() - start_time
            
            logger.error(f"Error routing net {net.name}: {e}")
            
            return RoutingResult.failure_result(
                error_message=str(e),
                execution_time=execution_time,
                algorithm=self.strategy.value
            )
    
    def route_two_pads(self, pad_a: Pad, pad_b: Pad, net_id: str, 
                      timeout: float = 5.0) -> RoutingResult:
        """Route between two specific pads."""
        if not self.routing_grid or not self.layer_manager:
            return RoutingResult.failure_result("Routing engine not initialized")
        
        try:
            # Convert pads to grid coordinates
            grid_a = self._pad_to_grid_point(pad_a)
            grid_b = self._pad_to_grid_point(pad_b)
            
            # Find path using A*
            path = self.pathfinder.find_path(
                start=grid_a,
                end=grid_b,
                grid=self.routing_grid,
                layer_manager=self.layer_manager,
                net_id=net_id,
                timeout=timeout
            )
            
            if not path:
                return RoutingResult.failure_result("No path found between pads")
            
            # Convert path to segments
            segments = self._path_to_segments(path, net_id)
            
            # Create route
            route = Route(
                id=f"two_pad_route_{net_id}_{datetime.now().timestamp()}",
                net_id=net_id,
                segments=segments
            )
            
            return RoutingResult.success_result(route=route, algorithm=self.strategy.value)
            
        except Exception as e:
            logger.error(f"Error routing two pads for {net_id}: {e}")
            return RoutingResult.failure_result(str(e))
    
    def route_all_nets(self, nets: List[Net], 
                      timeout_per_net: float = 5.0,
                      total_timeout: float = 300.0) -> RoutingStatistics:
        """Route all provided nets."""
        if not nets:
            return RoutingStatistics(algorithm_used=self.strategy.value)
        
        start_time = time.time()
        
        # Sort nets by routing priority
        sorted_nets = self.sort_nets_by_routing_priority(nets)
        
        logger.info(f"Starting to route {len(sorted_nets)} nets with Manhattan algorithm")
        
        nets_completed = 0
        nets_failed = 0
        
        for net in sorted_nets:
            # Check total timeout
            elapsed = time.time() - start_time
            if elapsed > total_timeout:
                logger.warning("Total timeout reached, stopping routing")
                break
            
            # Route the net
            result = self.route_net(net, timeout_per_net)
            
            if result.success:
                nets_completed += 1
            else:
                nets_failed += 1
                logger.warning(f"Failed to route net {net.name}: {result.error_message}")
        
        # Calculate statistics
        total_time = time.time() - start_time
        total_length = sum(route.total_length for route in self.routed_nets.values())
        total_vias = sum(route.via_count for route in self.routed_nets.values())
        
        statistics = RoutingStatistics(
            nets_attempted=len(sorted_nets),
            nets_routed=nets_completed,
            nets_failed=nets_failed,
            total_length=total_length,
            total_vias=total_vias,
            total_time=total_time,
            algorithm_used=self.strategy.value
        )
        
        logger.info(f"Manhattan routing completed: {nets_completed}/{len(sorted_nets)} nets "
                   f"({statistics.success_rate:.1%} success rate)")
        
        return statistics
    
    def clear_routes(self) -> None:
        """Clear all routing data."""
        if self.routing_grid:
            self.routing_grid.clear_all_routes()
        
        self.routed_nets.clear()
        self.failed_nets.clear()
        self.nets_attempted = 0
        self.nets_routed = 0
        self.nets_failed = 0
        
        logger.info("Cleared all routes from Manhattan routing engine")
    
    def get_routed_tracks(self) -> List[Dict[str, Any]]:
        """Get all routed tracks in display format."""
        tracks = []
        
        for route in self.routed_nets.values():
            for segment in route.segments:
                if segment.type == SegmentType.TRACK:
                    tracks.append({
                        'start': (segment.start.x, segment.start.y),
                        'end': (segment.end.x, segment.end.y),
                        'layer': segment.layer,
                        'width': segment.width,
                        'net': segment.net_id
                    })
        
        return tracks
    
    def get_routed_vias(self) -> List[Dict[str, Any]]:
        """Get all routed vias in display format."""
        vias = []
        
        for route in self.routed_nets.values():
            for via in route.vias:
                vias.append({
                    'x': via.position.x,
                    'y': via.position.y,
                    'size': via.diameter,
                    'drill': via.drill_size,
                    'layers': [via.from_layer, via.to_layer],
                    'net': via.net_id,
                    'type': via.via_type.value
                })
        
        return vias
    
    def get_routing_statistics(self) -> RoutingStatistics:
        """Get current routing statistics."""
        total_length = sum(route.total_length for route in self.routed_nets.values())
        total_vias = sum(route.via_count for route in self.routed_nets.values())
        
        return RoutingStatistics(
            nets_attempted=self.nets_attempted,
            nets_routed=self.nets_routed,
            nets_failed=self.nets_failed,
            total_length=total_length,
            total_vias=total_vias,
            algorithm_used=self.strategy.value
        )
    
    def _create_escape_routes(self, net: Net) -> List[Segment]:
        """Create F.Cu escape routes from pads to grid entry points."""
        segments = []
        
        for pad in net.pads:
            # Find escape point near pad
            escape_point = self._find_escape_point(pad, net.id)
            
            if escape_point and (escape_point.x != pad.position.x or escape_point.y != pad.position.y):
                # Create escape segment
                segment = Segment(
                    type=SegmentType.TRACK,
                    start=pad.position,
                    end=escape_point,
                    width=self.constraints.default_trace_width,
                    layer='F.Cu',
                    net_id=net.id
                )
                segments.append(segment)
        
        return segments
    
    def _find_escape_point(self, pad: Pad, net_id: str) -> Optional[Coordinate]:
        """Find escape point from pad to grid."""
        if not self.routing_grid:
            return None
        
        # Convert pad position to grid
        pad_grid_x, pad_grid_y = self.routing_grid.world_to_grid(pad.position.x, pad.position.y)
        
        # Search for accessible point nearby
        for radius in range(1, 4):
            for dx in [-radius, 0, radius]:
                for dy in [-radius, 0, radius]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    test_x, test_y = pad_grid_x + dx, pad_grid_y + dy
                    
                    if (self.routing_grid.is_valid_position(test_x, test_y, 0) and
                        self.routing_grid.get_cell(test_x, test_y, 0).is_accessible_by_net(net_id)):
                        
                        return self.routing_grid.grid_to_world(test_x, test_y)
        
        # Fallback to pad position
        return pad.position
    
    def _route_grid_connections(self, net: Net, timeout: float) -> Tuple[List[Segment], List[Via]]:
        """Route connections between pads on the routing grid."""
        segments = []
        vias = []
        
        if len(net.pads) < 2:
            return segments, vias
        
        # Route between consecutive pads (star routing)
        for i in range(len(net.pads) - 1):
            pad_a = net.pads[i]
            pad_b = net.pads[i + 1]
            
            # Convert to grid points
            grid_a = self._pad_to_grid_point(pad_a)
            grid_b = self._pad_to_grid_point(pad_b)
            
            # Find path
            path = self.pathfinder.find_path(
                start=grid_a,
                end=grid_b,
                grid=self.routing_grid,
                layer_manager=self.layer_manager,
                net_id=net.id,
                timeout=timeout / len(net.pads)
            )
            
            if path:
                path_segments, path_vias = self._path_to_segments_and_vias(path, net.id)
                segments.extend(path_segments)
                vias.extend(path_vias)
                
                # Mark path in grid
                self._mark_path_in_grid(path, net.id)
        
        return segments, vias
    
    def _pad_to_grid_point(self, pad: Pad) -> GridPoint:
        """Convert pad to grid point."""
        if not self.routing_grid:
            raise RuntimeError("Routing grid not initialized")
        
        grid_x, grid_y = self.routing_grid.world_to_grid(pad.position.x, pad.position.y)
        
        # Start on first routing layer (In1.Cu)
        layer_index = 0
        
        return GridPoint(grid_x, grid_y, layer_index)
    
    def _path_to_segments(self, path: List[GridPoint], net_id: str) -> List[Segment]:
        """Convert grid path to segments."""
        segments, _ = self._path_to_segments_and_vias(path, net_id)
        return segments
    
    def _path_to_segments_and_vias(self, path: List[GridPoint], net_id: str) -> Tuple[List[Segment], List[Via]]:
        """Convert grid path to segments and vias."""
        segments = []
        vias = []
        
        if not self.routing_grid or len(path) < 2:
            return segments, vias
        
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            current_world = self.routing_grid.grid_to_world(current.x, current.y)
            next_world = self.routing_grid.grid_to_world(next_point.x, next_point.y)
            
            if current.layer == next_point.layer:
                # Same layer - create track segment
                layer_name = self.routing_grid.get_layer_name(current.layer)
                
                segment = Segment(
                    type=SegmentType.TRACK,
                    start=current_world,
                    end=next_world,
                    width=self.constraints.default_trace_width,
                    layer=layer_name,
                    net_id=net_id
                )
                segments.append(segment)
            else:
                # Layer change - create via
                from_layer = self.routing_grid.get_layer_name(current.layer)
                to_layer = self.routing_grid.get_layer_name(next_point.layer)
                
                via = Via(
                    position=current_world,
                    diameter=self.constraints.default_via_diameter,
                    drill_size=self.constraints.default_via_drill,
                    from_layer=from_layer,
                    to_layer=to_layer,
                    net_id=net_id,
                    via_type=self._determine_via_type(current.layer, next_point.layer)
                )
                vias.append(via)
        
        return segments, vias
    
    def _determine_via_type(self, from_layer: int, to_layer: int) -> ViaType:
        """Determine via type based on layer transition."""
        if not self.routing_grid:
            return ViaType.BURIED
        
        total_layers = self.routing_grid.layer_count
        
        # Simple logic: first/last layer = blind, otherwise buried
        if from_layer == 0 or to_layer == 0:
            return ViaType.BLIND
        elif from_layer == total_layers - 1 or to_layer == total_layers - 1:
            return ViaType.BLIND
        else:
            return ViaType.BURIED
    
    def _mark_path_in_grid(self, path: List[GridPoint], net_id: str):
        """Mark path as routed in the grid."""
        if not self.routing_grid:
            return
        
        for point in path:
            cell = self.routing_grid.get_cell(point.x, point.y, point.layer)
            cell.set_routed(net_id)
            
            # Also mark with spacing halos
            spacing_cells = int(math.ceil(self.constraints.min_trace_spacing / self.routing_grid.resolution))
            
            for dx in range(-spacing_cells, spacing_cells + 1):
                for dy in range(-spacing_cells, spacing_cells + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    halo_x, halo_y = point.x + dx, point.y + dy
                    if self.routing_grid.is_valid_position(halo_x, halo_y, point.layer):
                        halo_cell = self.routing_grid.get_cell(halo_x, halo_y, point.layer)
                        if halo_cell.state == CellState.EMPTY:
                            halo_cell.cost_penalty += 0.5  # Add congestion penalty