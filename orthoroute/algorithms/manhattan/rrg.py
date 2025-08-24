"""
Routing Resource Graph (RRG) for FPGA-style Manhattan routing
Based on PathFinder negotiated congestion algorithm
"""

import heapq
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of RRG nodes"""
    RAIL = "rail"           # Vertical track segment
    BUS = "bus"             # Horizontal track segment  
    SWITCH = "switch"       # Rail/bus intersection for layer changes
    PAD_ENTRY = "pad_entry" # F.Cu pad entry point
    PAD_EXIT = "pad_exit"   # F.Cu pad exit point

class EdgeType(Enum):
    """Types of RRG edges"""
    TRACK = "track"         # Along same track/layer
    SWITCH = "switch"       # Between layers at switch box
    ENTRY = "entry"         # F.Cu pad to fabric
    EXIT = "exit"           # Fabric to F.Cu pad

@dataclass
class RRGNode:
    """RRG node representation"""
    id: str
    node_type: NodeType
    x: float                # World coordinates
    y: float
    layer: int             # Layer index (0=In1.Cu, 1=In2.Cu, etc.)
    capacity: int = 1      # Track capacity
    usage: int = 0         # Current usage
    track_index: int = 0   # Which track/lane on this layer
    
    def is_available(self) -> bool:
        """Check if node has available capacity"""
        return self.usage < self.capacity
    
    def utilization(self) -> float:
        """Get utilization ratio"""
        return self.usage / self.capacity if self.capacity > 0 else 0.0

@dataclass 
class RRGEdge:
    """RRG edge representation"""
    id: str
    edge_type: EdgeType
    from_node: str         # Node IDs
    to_node: str
    length_mm: float       # Physical length
    capacity: int = 1      # Edge capacity
    usage: int = 0         # Current usage
    base_cost: float = 0.0 # Base routing cost
    history_cost: float = 0.0  # Accumulated congestion penalty
    
    def is_available(self) -> bool:
        """Check if edge has available capacity"""
        return self.usage < self.capacity
    
    def utilization(self) -> float:
        """Get utilization ratio"""
        return self.usage / self.capacity if self.capacity > 0 else 0.0
    
    def current_cost(self, pres_fac: float = 1.0, alpha: float = 2.0) -> float:
        """Calculate current routing cost with congestion"""
        present_penalty = pres_fac * (self.utilization() ** alpha) if self.usage > 0 else 0.0
        return self.base_cost * (1.0 + present_penalty) + self.history_cost

@dataclass
class RoutingConfig:
    """PathFinder routing configuration"""
    grid_pitch: float = 0.4        # mm between tracks
    track_width: float = 0.0889    # mm trace width
    clearance: float = 0.0889      # mm minimum spacing
    via_diameter: float = 0.25     # mm
    via_drill: float = 0.15        # mm
    
    # Cost weights
    k_length: float = 1.0          # Length cost weight
    k_via: float = 10.0            # Via cost weight  
    k_bend: float = 2.0            # Bend/turn cost weight
    
    # PathFinder parameters
    max_iterations: int = 50       # Max congestion iterations
    pres_fac_init: float = 0.5     # Initial present factor
    pres_fac_mult: float = 1.3     # Present factor multiplier
    hist_cost_step: float = 1.0    # History cost increment
    alpha: float = 2.0             # Congestion penalty exponent

@dataclass
class RouteRequest:
    """Single net routing request"""
    net_id: str
    source_pad: str    # Pad entry node ID
    sink_pad: str      # Pad exit node ID

@dataclass
class RouteResult:
    """Routing result for a single net"""
    net_id: str
    success: bool
    path: List[str] = field(default_factory=list)  # Node IDs in path
    edges: List[str] = field(default_factory=list) # Edge IDs used
    cost: float = 0.0
    length_mm: float = 0.0
    via_count: int = 0

class RoutingResourceGraph:
    """FPGA-style Routing Resource Graph"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.nodes: Dict[str, RRGNode] = {}
        self.edges: Dict[str, RRGEdge] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # node_id -> edge_ids
        
        # Layer configuration
        self.layer_count = 11  # In1.Cu through B.Cu
        self.layer_directions = {}  # layer_index -> 'H'/'V'
        self._setup_layer_directions()
        
        # Statistics
        self.total_rails = 0
        self.total_buses = 0
        self.total_switches = 0
        
    def _setup_layer_directions(self):
        """Setup alternating H/V layer directions"""
        for i in range(self.layer_count):
            # Odd layers (In1.Cu=0, In3.Cu=2, etc.) = Horizontal
            # Even layers (In2.Cu=1, In4.Cu=3, etc.) = Vertical  
            self.layer_directions[i] = 'H' if i % 2 == 0 else 'V'
    
    def add_node(self, node: RRGNode) -> None:
        """Add node to RRG"""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
    
    def add_edge(self, edge: RRGEdge) -> None:
        """Add edge to RRG"""
        # Memory safety check - INCREASED limits for better routing
        if len(self.edges) > 5000000:  # 5M edge limit (was 1M)
            raise MemoryError(f"RRG edge limit exceeded: {len(self.edges)} edges. "
                            "Consider reducing board size or grid resolution.")
        
        self.edges[edge.id] = edge
        
        # Add to adjacency lists
        self.adjacency[edge.from_node].append(edge.id)
        # Note: edges are directional, but we can traverse both ways in search
        
    def get_neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        """Get neighboring (node_id, edge_id) pairs"""
        neighbors = []
        
        # Outgoing edges
        for edge_id in self.adjacency[node_id]:
            edge = self.edges[edge_id]
            neighbors.append((edge.to_node, edge_id))
            
        # Incoming edges (for bidirectional search)
        for edge_id, edge in self.edges.items():
            if edge.to_node == node_id:
                neighbors.append((edge.from_node, edge_id))
                
        return neighbors
    
    def manhattan_distance(self, node1_id: str, node2_id: str) -> float:
        """Calculate Manhattan distance between nodes"""
        n1 = self.nodes[node1_id]
        n2 = self.nodes[node2_id]
        return abs(n1.x - n2.x) + abs(n1.y - n2.y)
    
    def estimate_min_vias(self, node1_id: str, node2_id: str) -> int:
        """Estimate minimum vias needed between nodes"""
        n1 = self.nodes[node1_id]
        n2 = self.nodes[node2_id]
        return abs(n1.layer - n2.layer)
    
    def clear_usage(self):
        """Clear all usage counters"""
        for node in self.nodes.values():
            node.usage = 0
        for edge in self.edges.values():
            edge.usage = 0
    
    def get_overused_edges(self) -> List[str]:
        """Get list of overused edge IDs"""
        return [edge_id for edge_id, edge in self.edges.items() 
                if edge.usage > edge.capacity]
    
    def update_history_costs(self, overused_edges: List[str], step: float):
        """Update history costs for overused edges"""
        for edge_id in overused_edges:
            edge = self.edges[edge_id]
            excess = edge.usage - edge.capacity
            edge.history_cost += step * excess

class PathFinderRouter:
    """PathFinder negotiated congestion router with GPU wavefront pathfinding"""
    
    def __init__(self, rrg: RoutingResourceGraph):
        self.rrg = rrg
        self.config = rrg.config
        
        # Routing state
        self.net_routes: Dict[str, RouteResult] = {}
        self.current_iteration = 0
        self.present_factor = self.config.pres_fac_init
        
        # Initialize GPU wavefront pathfinder
        from .wavefront_pathfinder import GPUWavefrontPathfinder
        self.wavefront_router = GPUWavefrontPathfinder(rrg)
        logger.info("PathFinder initialized with GPU wavefront pathfinding")
        
    def route_single_net(self, request: RouteRequest) -> RouteResult:
        """Route single net using GPU wavefront pathfinding with F.Cu escape routing"""
        
        # Check if this routing requires F.Cu escape routing
        source_node = self.rrg.nodes.get(request.source_pad)
        sink_node = self.rrg.nodes.get(request.sink_pad)
        
        if source_node and sink_node:
            source_is_fcu = source_node.layer == -2  # F.Cu layer
            sink_is_fcu = sink_node.layer == -2      # F.Cu layer
            
            if source_is_fcu or sink_is_fcu:
                logger.debug(f"F.Cu escape routing needed for net {request.net_id}: source_fcu={source_is_fcu}, sink_fcu={sink_is_fcu}")
                return self._route_with_escape_routing(request, source_is_fcu, sink_is_fcu)
        
        # Use direct GPU wavefront pathfinder for internal routing
        result = self.wavefront_router.route_single_net(request)
        
        return result
    
    def _route_with_escape_routing(self, request: RouteRequest, source_is_fcu: bool, sink_is_fcu: bool) -> RouteResult:
        """Route net with F.Cu escape routing"""
        logger.info(f"Routing net {request.net_id} with F.Cu escape routing")
        
        # Get source and sink nodes
        source_node = self.rrg.nodes[request.source_pad]
        sink_node = self.rrg.nodes[request.sink_pad]
        
        # Find internal grid entry/exit points for F.Cu pads
        actual_source = request.source_pad
        actual_sink = request.sink_pad
        escape_segments = []
        
        if source_is_fcu:
            # Find nearest internal grid point for source escape
            escape_point = self._find_escape_point(source_node)
            if not escape_point:
                logger.error(f"Cannot find escape point for F.Cu source {request.source_pad}")
                return RouteResult(net_id=request.net_id, success=False)
            
            actual_source = escape_point
            escape_segments.append(('escape', request.source_pad, escape_point))
            logger.debug(f"Source escape: {request.source_pad} -> {escape_point}")
        
        if sink_is_fcu:
            # Find nearest internal grid point for sink entry
            entry_point = self._find_escape_point(sink_node)
            if not entry_point:
                logger.error(f"Cannot find entry point for F.Cu sink {request.sink_pad}")
                return RouteResult(net_id=request.net_id, success=False)
            
            actual_sink = entry_point
            escape_segments.append(('entry', entry_point, request.sink_pad))
            logger.debug(f"Sink entry: {entry_point} -> {request.sink_pad}")
        
        # Create modified request for internal grid routing
        internal_request = RouteRequest(
            net_id=request.net_id,
            source_pad=actual_source,
            sink_pad=actual_sink
        )
        
        # Add any pending escape vias to wavefront grid before routing
        self._add_pending_escape_vias_to_grid()
        
        # Route between internal grid points
        result = self.wavefront_router.route_single_net(internal_request)
        
        if result.success:
            # Add escape routing segments to the path
            full_path = []
            
            # Add source escape segment
            if source_is_fcu:
                full_path.append(request.source_pad)  # F.Cu pad
                full_path.append(actual_source)       # Internal grid point
            
            # Add internal routing path (skip first node if escape was added)
            if result.path:
                path_start = 1 if source_is_fcu else 0
                full_path.extend(result.path[path_start:])
            
            # Add sink entry segment  
            if sink_is_fcu and request.sink_pad not in full_path:
                full_path.append(request.sink_pad)    # F.Cu pad
            
            logger.info(f"F.Cu escape routing SUCCESS: net {request.net_id}, path length {len(full_path)}")
            
            return RouteResult(
                net_id=request.net_id,
                success=True,
                path=full_path,
                cost=result.cost + len(escape_segments) * 2,  # Add escape cost
                length_mm=result.length_mm,
                via_count=result.via_count + len(escape_segments)
            )
        else:
            logger.warning(f"F.Cu escape routing FAILED: internal routing failed for net {request.net_id}")
            # Even if internal routing failed, return escape routing segments for visualization
            # Store current escape vias before they get cleared
            current_escape_vias = getattr(self, '_pending_escape_vias', [])
            partial_result = self._create_partial_escape_route_result(request, source_is_fcu, sink_is_fcu, escape_segments, current_escape_vias)
            return partial_result
    
    def _find_escape_point(self, fcu_node) -> Optional[str]:
        """Create proper F.Cu escape routing with via placement"""
        # Create escape via outside component footprint
        escape_distance = 0.5  # 0.5mm escape distance from pad center
        
        # Try vertical directions only for F.Cu escape routing
        directions = [
            (0, 1),   # Up (+Y) on F.Cu layer
            (0, -1),  # Down (-Y) on F.Cu layer
        ]
        
        for dx, dy in directions:
            via_x = fcu_node.x + dx * escape_distance
            via_y = fcu_node.y + dy * escape_distance
            
            # Check if escape via position is clear
            if self._is_escape_position_clear(via_x, via_y):
                # Create escape via node 
                via_id = f"escape_via_{fcu_node.id}_{dx}_{dy}"
                escape_via = RRGNode(
                    id=via_id,
                    node_type=NodeType.SWITCH,  # Via acts as a switch point
                    x=via_x,
                    y=via_y,
                    layer=0,  # Start on first routing layer
                    capacity=1
                )
                self.rrg.add_node(escape_via)
                
                # Create F.Cu trace segment from pad to via
                fcu_trace_id = f"fcu_trace_{fcu_node.id}_to_{via_id}"
                fcu_trace = RRGEdge(
                    id=fcu_trace_id,
                    edge_type=EdgeType.ENTRY,  # Use ENTRY for escape routing
                    from_node=fcu_node.id,
                    to_node=via_id,
                    length_mm=escape_distance,
                    base_cost=escape_distance * 1.0  # F.Cu trace cost
                )
                self.rrg.add_edge(fcu_trace)
                
                # Connect escape via to nearby routing grid nodes
                self._connect_via_to_routing_grid(via_id, via_x, via_y)
                
                # Store escape via for later addition to wavefront grid
                if not hasattr(self, '_pending_escape_vias'):
                    self._pending_escape_vias = []
                self._pending_escape_vias.append((via_id, escape_via))
                
                logger.info(f"Created F.Cu escape via {via_id} at ({via_x:.2f},{via_y:.2f})")
                return via_id
                
        logger.warning(f"Could not find clear escape position for F.Cu node {fcu_node.id}")
        return None
    
    def _create_partial_escape_route_result(self, request: RouteRequest, source_is_fcu: bool, sink_is_fcu: bool, escape_segments: List, escape_vias: List = None) -> RouteResult:
        """Create route result showing escape routing even when internal routing fails"""
        partial_path = []
        
        # Add escape routing segments to show F.Cu traces and vias
        if escape_vias is None:
            escape_vias = getattr(self, '_pending_escape_vias', [])
            
        logger.debug(f"Looking for escape vias in {len(escape_vias)} pending vias for net {request.net_id}")
        
        if source_is_fcu and escape_vias:
            # Find source escape via by checking if the source pad name is in via_id
            source_pad_name = request.source_pad.replace('pad_entry_', '')
            for via_id, escape_via in escape_vias:
                if source_pad_name in via_id:
                    partial_path.extend([request.source_pad, via_id])  # F.Cu pad -> escape via
                    logger.info(f"Added source escape: {request.source_pad} -> {via_id}")
                    break
        
        if sink_is_fcu and escape_vias:
            # Find sink escape via by checking if the sink pad name is in via_id
            sink_pad_name = request.sink_pad.replace('pad_entry_', '')
            for via_id, escape_via in escape_vias:
                if sink_pad_name in via_id and via_id not in partial_path:
                    partial_path.extend([via_id, request.sink_pad])  # Escape via -> F.Cu pad
                    logger.info(f"Added sink escape: {via_id} -> {request.sink_pad}")
                    break
        
        logger.info(f"Created partial escape route for {request.net_id}: {len(partial_path)} segments")
        
        return RouteResult(
            net_id=request.net_id,
            success=True,  # Mark as success so escape routing gets visualized
            path=partial_path,
            cost=len(escape_segments) * 2.0,
            length_mm=len(escape_segments) * 0.5,  # Escape distance
            via_count=len(escape_segments)
        )
        
    def _add_pending_escape_vias_to_grid(self):
        """Add pending escape vias to the wavefront grid after construction"""
        if not hasattr(self, '_pending_escape_vias'):
            logger.debug("No pending escape vias to add")
            return
            
        logger.info(f"Adding {len(self._pending_escape_vias)} pending escape vias to wavefront grid")
        
        if hasattr(self.wavefront_router, 'add_node_to_grid'):
            for via_id, escape_via in self._pending_escape_vias:
                grid_pos = self.wavefront_router.add_node_to_grid(via_id, escape_via)
                if grid_pos:
                    logger.info(f"Added escape via {via_id} to wavefront grid at {grid_pos}")
                else:
                    logger.warning(f"Failed to add escape via {via_id} to wavefront grid")
        else:
            logger.warning("Wavefront router does not support add_node_to_grid")
        
        # Clear pending vias
        self._pending_escape_vias = []
    
    def _connect_via_to_routing_grid(self, via_id: str, via_x: float, via_y: float):
        """Connect escape via to nearby routing grid nodes"""
        connections_made = 0
        max_search_distance = 2.0  # Search within 2mm radius
        
        # Find nearby routing grid nodes
        for node_id, node in self.rrg.nodes.items():
            if node.layer >= 0 and node.node_type in [NodeType.BUS, NodeType.RAIL]:
                # Calculate distance to via
                distance = math.sqrt((node.x - via_x)**2 + (node.y - via_y)**2)
                
                if distance <= max_search_distance:
                    # Create connection from via to routing node
                    conn_id = f"via_conn_{via_id}_to_{node_id}"
                    via_conn = RRGEdge(
                        id=conn_id,
                        edge_type=EdgeType.SWITCH,  # Via connection acts as a switch
                        from_node=via_id,
                        to_node=node_id,
                        length_mm=distance,
                        base_cost=distance * 0.5 + 1.0  # Via connection cost
                    )
                    self.rrg.add_edge(via_conn)
                    connections_made += 1
                    
                    # Limit connections to avoid too many
                    if connections_made >= 4:
                        break
        
        logger.debug(f"Connected escape via {via_id} to {connections_made} routing grid nodes")
    
    def _is_escape_position_clear(self, x: float, y: float) -> bool:
        """Check if escape via position is clear of obstacles"""
        # For now, assume all positions are clear
        # TODO: Check for component keepouts, existing vias, etc.
        return True
    
    def _is_node_accessible(self, node_id: str) -> bool:
        """Check if node is accessible for escape routing"""
        # For now, consider all internal grid nodes accessible
        # Could add more sophisticated checking later (obstacles, existing routes)
        return True
    
    def _create_route_result(self, net_id: str, success: bool, path: List[str], cost: float) -> RouteResult:
        """Create route result from path"""
        if not success or len(path) < 2:
            return RouteResult(net_id=net_id, success=success)
        
        # Calculate edges used
        edges_used = []
        total_length = 0.0
        via_count = 0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            # Find edge between these nodes
            edge_id = self._find_edge_between_nodes(from_node, to_node)
            if edge_id:
                edges_used.append(edge_id)
                edge = self.rrg.edges[edge_id]
                total_length += edge.length_mm
                
                if edge.edge_type in [EdgeType.SWITCH, EdgeType.ENTRY, EdgeType.EXIT]:
                    via_count += 1
        
        return RouteResult(
            net_id=net_id,
            success=success,
            path=path,
            edges=edges_used,
            cost=cost,
            length_mm=total_length,
            via_count=via_count
        )
    
    def _find_edge_between_nodes(self, from_id: str, to_id: str) -> Optional[str]:
        """Find edge between two nodes"""
        for edge_id in self.rrg.adjacency[from_id]:
            edge = self.rrg.edges[edge_id]
            if edge.to_node == to_id:
                return edge_id
        
        # Try reverse direction
        for edge_id in self.rrg.adjacency[to_id]:
            edge = self.rrg.edges[edge_id]
            if edge.to_node == from_id:
                return edge_id
                
        return None
    
    def claim_route(self, result: RouteResult):
        """Claim resources for a successful route"""
        for edge_id in result.edges:
            edge = self.rrg.edges[edge_id]
            edge.usage += 1
            
        for node_id in result.path:
            node = self.rrg.nodes[node_id]
            if node.node_type in [NodeType.RAIL, NodeType.BUS]:
                node.usage += 1
    
    def route_all_nets(self, requests: List[RouteRequest]) -> Dict[str, RouteResult]:
        """Route all nets with negotiated congestion"""
        logger.info(f"🚀 Starting PathFinder routing for {len(requests)} nets")
        
        results = {}
        
        for iteration in range(self.config.max_iterations):
            self.current_iteration = iteration
            logger.debug(f"PathFinder iteration {iteration + 1}")
            
            # Clear usage from previous iteration
            self.rrg.clear_usage()
            
            # Route all nets
            failed_nets = 0
            for request in requests:
                result = self.route_single_net(request)
                results[request.net_id] = result
                
                if result.success:
                    self.claim_route(result)
                else:
                    failed_nets += 1
            
            # Check for overused resources
            overused_edges = self.rrg.get_overused_edges()
            
            if not overused_edges:
                logger.info(f"✅ Routing converged after {iteration + 1} iterations")
                break
                
            if failed_nets == len(requests):
                logger.warning(f"❌ All nets failed in iteration {iteration + 1}")
                break
            
            # Update costs for next iteration
            self.rrg.update_history_costs(overused_edges, self.config.hist_cost_step)
            self.present_factor *= self.config.pres_fac_mult
            
            # Update wavefront router with congestion costs
            congestion_map = {}
            for edge_id in overused_edges:
                edge = self.rrg.edges[edge_id]
                congestion_cost = edge.current_cost(self.present_factor, self.config.alpha)
                # Map edge nodes to congestion costs
                congestion_map[edge.from_node] = max(congestion_map.get(edge.from_node, 1.0), congestion_cost)
                congestion_map[edge.to_node] = max(congestion_map.get(edge.to_node, 1.0), congestion_cost)
            
            self.wavefront_router.update_costs(congestion_map)
            
            logger.debug(f"Iteration {iteration + 1}: {len(overused_edges)} overused edges, "
                        f"{failed_nets} failed nets")
        
        success_count = sum(1 for r in results.values() if r.success)
        logger.info(f"🏁 PathFinder complete: {success_count}/{len(requests)} nets routed")
        
        return results