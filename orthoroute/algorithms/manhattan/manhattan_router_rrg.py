"""
RRG-based Manhattan Routing Engine
Replaces cell-based A* with FPGA-style PathFinder routing
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime

from ...domain.models.board import Board, Net, Pad, Bounds, Coordinate
from ...domain.models.routing import Route, Segment, Via, RoutingResult, RoutingStatistics, SegmentType, ViaType
from ...domain.models.constraints import DRCConstraints
from ...domain.services.routing_engine import RoutingEngine, RoutingStrategy
from ...application.interfaces.gpu_provider import GPUProvider

from .rrg import RoutingConfig, RouteRequest, RouteResult
from .sparse_rrg_builder import SparseRRGBuilder
from .rrg import PathFinderRouter
from .types import Pad as RRGPad

logger = logging.getLogger(__name__)


class ManhattanRRGRoutingEngine(RoutingEngine):
    """Manhattan routing engine using RRG PathFinder algorithm"""
    
    def __init__(self, constraints: DRCConstraints, gpu_provider: Optional[GPUProvider] = None):
        """Initialize RRG-based Manhattan routing engine."""
        super().__init__(constraints)
        
        self.gpu_provider = gpu_provider
        self.board: Optional[Board] = None
        
        # RRG components
        self.fabric_builder: Optional[RRGFabricBuilder] = None
        self.pathfinder_router: Optional[PathFinderRouter] = None
        
        # Create routing configuration from DRC constraints
        self.routing_config = RoutingConfig(
            grid_pitch=0.4,  # mm - standard Manhattan grid pitch
            track_width=constraints.default_trace_width if hasattr(constraints, 'default_trace_width') else 0.0889,
            clearance=constraints.min_trace_spacing if hasattr(constraints, 'min_trace_spacing') else 0.0889,
            via_diameter=constraints.default_via_diameter if hasattr(constraints, 'default_via_diameter') else 0.25,
            via_drill=constraints.default_via_drill if hasattr(constraints, 'default_via_drill') else 0.15,
        )
        
        # Routing state
        self.routed_nets: Dict[str, Route] = {}
        self.failed_nets: Set[str] = set()
        self.nets_attempted = 0
        self.nets_routed = 0
        self.nets_failed = 0
        
        # Configuration
        self.default_board_margin = 3.0  # mm
        
        logger.info("🚀 RRG-based Manhattan routing engine initialized")
    
    @property
    def strategy(self) -> RoutingStrategy:
        """Get the routing strategy."""
        return RoutingStrategy.MANHATTAN_ASTAR  # Keep same enum value
    
    @property 
    def supports_gpu(self) -> bool:
        """Check if engine supports GPU acceleration."""
        return self.gpu_provider is not None and self.gpu_provider.is_available()
    
    def initialize(self, board: Board) -> None:
        """Initialize routing engine with board data."""
        try:
            self.board = board
            
            logger.info("🏗️ Building RRG fabric for Manhattan routing...")
            
            # Calculate board bounds with margin
            board_bounds = board.get_bounds()
            routing_bounds = (
                board_bounds.min_x - self.default_board_margin,
                board_bounds.min_y - self.default_board_margin,
                board_bounds.max_x + self.default_board_margin,
                board_bounds.max_y + self.default_board_margin
            )
            
            # Convert board pads to RRG format
            rrg_pads = self._convert_pads_to_rrg_format(board)
            
            # Extract airwires from board if available
            airwires = self._extract_airwires_from_board(board)
            
            # Build RRG fabric using sparse builder with airwire-derived bounds
            self.fabric_builder = SparseRRGBuilder(self.routing_config)
            rrg = self.fabric_builder.build_fabric(routing_bounds, rrg_pads, airwires)
            
            # Initialize PathFinder router
            self.pathfinder_router = PathFinderRouter(rrg)
            
            # Clear previous routing state
            self.routed_nets.clear()
            self.failed_nets.clear()
            self.nets_attempted = 0
            self.nets_routed = 0
            self.nets_failed = 0
            
            logger.info("✅ RRG Manhattan router initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RRG Manhattan routing engine: {e}")
            raise
    
    def _convert_pads_to_rrg_format(self, board: Board) -> List[RRGPad]:
        """Convert board pads to RRG format"""
        rrg_pads = []
        
        # Collect all pads from all nets
        for net in board.nets:
            for pad in net.pads:
                # Determine layer set
                layer_set = set()
                if hasattr(pad, 'layers'):
                    layer_set = set(pad.layers)
                else:
                    # Default to F.Cu for surface mount, or THRU for through-hole
                    layer_set = {"F.Cu"}
                
                rrg_pad = RRGPad(
                    net_name=net.name,
                    x_mm=pad.position.x,
                    y_mm=pad.position.y,
                    width_mm=getattr(pad, 'width', 1.0),
                    height_mm=getattr(pad, 'height', 1.0),
                    layer_set=layer_set,
                    is_through_hole=getattr(pad, 'is_through_hole', False)
                )
                rrg_pads.append(rrg_pad)
        
        logger.debug(f"Converted {len(rrg_pads)} pads to RRG format")
        return rrg_pads
    
    def _extract_airwires_from_board(self, board: Board) -> List[Dict]:
        """Extract airwires from board for routing area calculation"""
        if hasattr(board, '_airwires'):
            airwires = board._airwires
            logger.info(f"📡 Extracted {len(airwires)} airwires from board data")
            return airwires
        else:
            logger.warning("No airwires found in board data")
            return []
    
    def route_net(self, net: Net, timeout: float = 10.0) -> RoutingResult:
        """Route a single net using RRG PathFinder."""
        if not self.pathfinder_router:
            return RoutingResult.failure_result("RRG routing engine not initialized")
        
        start_time = time.time()
        self.nets_attempted += 1
        
        try:
            # Validate net can be routed
            validation_issues = self.validate_net(net)
            if validation_issues:
                error_msg = f"Net validation failed: {'; '.join(validation_issues)}"
                return RoutingResult.failure_result(error_msg)
            
            # Convert net to route requests
            requests = self._create_route_requests(net)
            
            if not requests:
                return RoutingResult.failure_result("No valid route requests created")
            
            # Route using PathFinder (single net, but still uses negotiated congestion internally)
            results = self.pathfinder_router.route_all_nets(requests)
            
            # Check if routing was successful
            successful_routes = [r for r in results.values() if r.success]
            if not successful_routes:
                self.nets_failed += 1
                self.failed_nets.add(net.id)
                return RoutingResult.failure_result("PathFinder failed to route net")
            
            # Convert RRG route results to domain route
            route = self._convert_rrg_results_to_route(net, successful_routes)
            
            # Store successful route
            self.routed_nets[net.id] = route
            self.nets_routed += 1
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Successfully routed net {net.name} in {execution_time:.3f}s")
            
            return RoutingResult.success_result(
                route=route,
                execution_time=execution_time,
                algorithm=self.strategy.value
            )
            
        except Exception as e:
            self.nets_failed += 1
            self.failed_nets.add(net.id)
            execution_time = time.time() - start_time
            
            logger.error(f"❌ Error routing net {net.name}: {e}")
            
            return RoutingResult.failure_result(
                error_message=str(e),
                execution_time=execution_time,
                algorithm=self.strategy.value
            )
    
    def route_two_pads(self, pad_a, pad_b, net_id: str, timeout: float = 5.0) -> RoutingResult:
        """Route between two specific pads using RRG PathFinder."""
        if not self.pathfinder_router:
            return RoutingResult.failure_result("RRG routing engine not initialized")
        
        try:
            start_time = time.time()
            
            # Create a temporary net with these two pads
            from ...domain.models.board import Net
            temp_net = Net(
                id=net_id,
                name=f"temp_net_{net_id}",
                pads=[pad_a, pad_b]
            )
            
            # Route using the standard net routing
            result = self.route_net(temp_net, timeout)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Error routing two pads: {e}")
            return RoutingResult.failure_result(
                error_message=str(e),
                execution_time=execution_time,
                algorithm=self.strategy.value
            )
    
    def _create_route_requests(self, net: Net) -> List[RouteRequest]:
        """Convert net to RRG route requests"""
        requests = []
        
        if len(net.pads) < 2:
            return requests
        
        # Create star topology: route from first pad to all others
        # In RRG, each pad should have a corresponding entry node
        source_pad_id = f"pad_entry_{net.name}_0"
        
        for i in range(1, len(net.pads)):
            sink_pad_id = f"pad_entry_{net.name}_{i}"
            
            request = RouteRequest(
                net_id=f"{net.id}_{i}",
                source_pad=source_pad_id,
                sink_pad=sink_pad_id
            )
            requests.append(request)
        
        return requests
    
    def _convert_rrg_results_to_route(self, net: Net, rrg_results: List[RouteResult]) -> Route:
        """Convert RRG route results back to domain Route"""
        all_segments = []
        all_vias = []
        
        if not self.pathfinder_router:
            raise RuntimeError("PathFinder router not initialized")
        
        rrg = self.pathfinder_router.rrg
        
        for result in rrg_results:
            if not result.success:
                continue
                
            # Convert path to segments and vias
            segments, vias = self._convert_rrg_path_to_segments_vias(
                result.path, result.edges, net.id, rrg
            )
            
            all_segments.extend(segments)
            all_vias.extend(vias)
        
        # Create route
        route = Route(
            id=f"rrg_route_{net.id}_{datetime.now().timestamp()}",
            net_id=net.id,
            segments=all_segments,
            vias=all_vias
        )
        
        return route
    
    def _convert_rrg_path_to_segments_vias(self, path: List[str], edges: List[str], 
                                         net_id: str, rrg) -> Tuple[List[Segment], List[Via]]:
        """Convert RRG path to domain segments and vias"""
        segments = []
        vias = []
        
        if len(path) < 2:
            return segments, vias
        
        # Process each edge in the path
        for edge_id in edges:
            edge = rrg.edges[edge_id]
            from_node = rrg.nodes[edge.from_node]
            to_node = rrg.nodes[edge.to_node]
            
            if edge.edge_type.value in ['track', 'entry', 'exit']:
                # Create track segment
                segment = Segment(
                    type=SegmentType.TRACK,
                    start=Coordinate(from_node.x, from_node.y),
                    end=Coordinate(to_node.x, to_node.y),
                    width=self.routing_config.track_width,
                    layer=self._get_layer_name(from_node.layer),
                    net_id=net_id
                )
                segments.append(segment)
                
            elif edge.edge_type.value == 'switch':
                # Create via for layer changes
                via = Via(
                    position=Coordinate(from_node.x, from_node.y),
                    diameter=self.routing_config.via_diameter,
                    drill_size=self.routing_config.via_drill,
                    from_layer=self._get_layer_name(from_node.layer),
                    to_layer=self._get_layer_name(to_node.layer),
                    net_id=net_id,
                    via_type=self._determine_via_type(from_node.layer, to_node.layer)
                )
                vias.append(via)
        
        return segments, vias
    
    def _get_layer_name(self, layer_index: int) -> str:
        """Convert layer index to layer name"""
        if layer_index == -2:
            return "F.Cu"
        elif layer_index == -1:
            return "Switch"  # Special case
        elif 0 <= layer_index <= 10:
            if layer_index == 10:
                return "B.Cu"
            else:
                return f"In{layer_index + 1}.Cu"
        else:
            return f"Layer{layer_index}"
    
    def _determine_via_type(self, from_layer: int, to_layer: int) -> ViaType:
        """Determine via type based on layer transition"""
        # Simple logic for now - could be enhanced based on stack-up
        if from_layer == -2 or to_layer == -2:  # F.Cu involved
            return ViaType.BLIND
        elif from_layer == 10 or to_layer == 10:  # B.Cu involved
            return ViaType.BLIND
        else:
            return ViaType.BURIED
    
    def route_all_nets(self, nets: List[Net], 
                      timeout_per_net: float = 5.0,
                      total_timeout: float = 300.0) -> RoutingStatistics:
        """Route all provided nets using RRG PathFinder."""
        if not nets:
            return RoutingStatistics(algorithm_used=self.strategy.value)
        
        start_time = time.time()
        
        logger.info(f"🚀 Starting RRG PathFinder routing for {len(nets)} nets")
        
        # TESTING: Limit to first 3 nets for faster debugging
        test_nets = nets[:3]
        logger.info(f"TESTING: Processing only {len(test_nets)} nets for debugging")
        
        # Create all route requests
        all_requests = []
        for net in test_nets:
            requests = self._create_route_requests(net)
            all_requests.extend(requests)
        
        if not all_requests:
            logger.warning("No valid route requests created")
            return RoutingStatistics(algorithm_used=self.strategy.value)
        
        # Route all nets simultaneously with negotiated congestion
        if self.pathfinder_router:
            results = self.pathfinder_router.route_all_nets(all_requests)
        else:
            logger.error("PathFinder router not initialized")
            return RoutingStatistics(algorithm_used=self.strategy.value)
        
        # Process results and create routes
        nets_completed = 0
        nets_failed = 0
        
        for net in test_nets:
            # Find results for this net
            net_results = [r for r in results.values() 
                          if r.net_id.startswith(f"{net.id}_")]
            
            if net_results and any(r.success for r in net_results):
                try:
                    # Convert to domain route
                    successful_results = [r for r in net_results if r.success]
                    route = self._convert_rrg_results_to_route(net, successful_results)
                    
                    self.routed_nets[net.id] = route
                    nets_completed += 1
                    self.nets_routed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to convert RRG result for net {net.name}: {e}")
                    nets_failed += 1
                    self.nets_failed += 1
                    self.failed_nets.add(net.id)
            else:
                nets_failed += 1
                self.nets_failed += 1
                self.failed_nets.add(net.id)
                logger.warning(f"Failed to route net {net.name}: {[r.success for r in net_results]}")
        
        # Calculate statistics
        total_time = time.time() - start_time
        total_length = sum(route.total_length for route in self.routed_nets.values())
        total_vias = sum(route.via_count for route in self.routed_nets.values())
        
        statistics = RoutingStatistics(
            nets_attempted=len(nets),
            nets_routed=nets_completed,
            nets_failed=nets_failed,
            total_length=total_length,
            total_vias=total_vias,
            total_time=total_time,
            algorithm_used=self.strategy.value
        )
        
        logger.info(f"🏁 RRG PathFinder routing completed: {nets_completed}/{len(nets)} nets "
                   f"({statistics.success_rate:.1%} success rate)")
        
        return statistics
    
    def clear_routes(self) -> None:
        """Clear all routing data."""
        if self.pathfinder_router:
            self.pathfinder_router.rrg.clear_usage()
        
        self.routed_nets.clear()
        self.failed_nets.clear()
        self.nets_attempted = 0
        self.nets_routed = 0
        self.nets_failed = 0
        
        logger.info("Cleared all routes from RRG Manhattan routing engine")
    
    def get_routed_tracks(self) -> List[Dict[str, Any]]:
        """Get all routed tracks in display format."""
        tracks = []
        
        for net_id, route in self.routed_nets.items():
            logger.debug(f"Converting route for net {net_id}: {len(route.segments)} segments")
            for segment in route.segments:
                if segment.type == SegmentType.TRACK:
                    track = {
                        'start_x': segment.start.x,
                        'start_y': segment.start.y,
                        'end_x': segment.end.x,
                        'end_y': segment.end.y,
                        'layer': segment.layer,
                        'width': segment.width,
                        'net': segment.net_id
                    }
                    tracks.append(track)
                    logger.debug(f"Added track: {track}")
        
        logger.info(f"Generated {len(tracks)} display tracks from {len(self.routed_nets)} routes")
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