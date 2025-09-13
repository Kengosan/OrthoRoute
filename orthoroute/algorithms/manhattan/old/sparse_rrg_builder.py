"""
Sparse Hierarchical RRG Builder - Memory-efficient fabric for large boards
Uses coarse global routing grid with fine local connections
"""

import logging
import math
from typing import Dict, List, Tuple, Set, Any
from .rrg import (
    RoutingResourceGraph, RRGNode, RRGEdge, NodeType, EdgeType, RoutingConfig
)
from .types import Pad

logger = logging.getLogger(__name__)

class SparseRRGBuilder:
    """Memory-efficient sparse RRG builder for large boards"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.rrg = RoutingResourceGraph(config)
        
        # F.Cu pad node tracking (simplified system)
        self.fcu_pad_nodes = {}  # pad_id -> node_index mapping
        
        # Board bounds
        self.min_x = 0.0
        self.min_y = 0.0  
        self.max_x = 0.0
        self.max_y = 0.0
        
        # Hierarchical grid parameters  
        self.coarse_pitch = 0.2   # mm - Original working pitch
        self.fine_pitch = 0.2     # mm - Stable resolution with manageable RAM usage
        
        self.coarse_cols = 0
        self.coarse_rows = 0
        
        # Track local routing areas around pads
        self.local_areas = []  # List of (min_x, min_y, max_x, max_y) fine grid areas
        
        # Store grid positions for tap generation
        self.grid_x_positions = []  # Vertical rail X positions
        
    def build_fabric(self, board_bounds: Tuple[float, float, float, float], 
                    pads: List[Pad], airwires: List[Dict] = None) -> RoutingResourceGraph:
        """Build proper F.Cu routing architecture with structured inner layer grid"""
        
        # Calculate active routing area from airwires if provided
        if airwires and len(airwires) > 0:
            self.min_x, self.min_y, self.max_x, self.max_y = self._calculate_airwire_bounds(airwires)
            logger.info(f"Active routing area from {len(airwires)} airwires: "
                       f"({self.min_x:.1f},{self.min_y:.1f}) to ({self.max_x:.1f},{self.max_y:.1f})")
        else:
            # Fallback to board bounds
            self.min_x, self.min_y, self.max_x, self.max_y = board_bounds
            logger.warning("No airwires provided, using full board bounds")
        
        board_width = self.max_x - self.min_x
        board_height = self.max_y - self.min_y
        
        logger.info(f"Building F.Cu Routing Architecture for {board_width:.1f}x{board_height:.1f}mm active area")
        
        # F.CU SIMPLIFIED: F.Cu as vertical-only routing layer with direct pad connections
        logger.info("F.CU SIMPLIFIED: Building F.Cu vertical grid with DRC-compliant pad connections")
        
        # Step 1: Build complete orthogonal grid (F.Cu + inner layers)
        logger.error("DEBUG: About to call _build_complete_orthogonal_grid()")
        self._build_complete_orthogonal_grid()
        logger.error("DEBUG: Finished _build_complete_orthogonal_grid()")
        
        # Step 2: Connect component pads to F.Cu vertical grid with DRC rules
        self._connect_pads_to_fcu_grid(pads)
        
        # Step 3: Build inter-layer vias for grid routing
        self._build_grid_layer_connections()
        
        # Validate fabric connectivity
        self._validate_fabric_connectivity()
        
        self._report_fabric_stats()
        
        # Sort and deduplicate grid X positions
        self.grid_x_positions = sorted(list(set(self.grid_x_positions)))
        logger.info(f"Generated {len(self.grid_x_positions)} unique vertical rail positions")
        
        return self.rrg
    
    def _build_complete_orthogonal_grid(self):
        """Build complete orthogonal routing grid including F.Cu vertical layer"""
        logger.info("COMPLETE GRID: Building full orthogonal routing grid (F.Cu + inner layers)")
        
        # Grid parameters
        grid_pitch = 0.4  # mm - 0.4mm grid spacing
        
        # PRECISION FIX: Align grid bounds to exact grid coordinates
        grid_min_x = round(self.min_x / grid_pitch) * grid_pitch
        grid_max_x = round(self.max_x / grid_pitch) * grid_pitch  
        grid_min_y = round(self.min_y / grid_pitch) * grid_pitch
        grid_max_y = round(self.max_y / grid_pitch) * grid_pitch
        
        logger.error(f"3D_LATTICE_BOUNDS: raw bounds=({self.min_x:.6f}, {self.min_y:.6f}, {self.max_x:.6f}, {self.max_y:.6f})")
        logger.error(f"3D_LATTICE_BOUNDS: grid bounds=({grid_min_x:.6f}, {grid_min_y:.6f}, {grid_max_x:.6f}, {grid_max_y:.6f})")
        logger.error(f"3D_LATTICE_BOUNDS: grid_pitch={grid_pitch:.6f}")
        
        # DYNAMIC LAYER ASSIGNMENTS: F.Cu=0 (vertical), then alternating H/V for inner layers
        layer_assignments = [
            {'id': 0, 'name': 'F.Cu', 'direction': 'v'},  # F.Cu is vertical-only
        ]
        
        # Add inner layers dynamically based on board layer count
        total_layers = self.rrg.layer_count
        for layer_id in range(1, total_layers):
            layer_name = f"In{layer_id}.Cu" if layer_id < total_layers-1 else "B.Cu"
            direction = 'h' if (layer_id % 2 == 1) else 'v'  # In1=h, In2=v, In3=h, etc.
            layer_assignments.append({
                'id': layer_id, 
                'name': layer_name, 
                'direction': direction
            })
        
        logger.error("3D LATTICE: Creating COMPLETE lattice - every (x,y) gets nodes on ALL layers")
        
        logger.info(f"LAYER ASSIGNMENTS: {len(layer_assignments)} layers")
        for layer in layer_assignments:
            logger.info(f"  Layer {layer['id']}: {layer['name']} ({layer['direction'].upper()})")
        
        # CALCULATE 3D LATTICE DIMENSIONS
        x_steps = int((grid_max_x - grid_min_x) / grid_pitch) + 1
        y_steps = int((grid_max_y - grid_min_y) / grid_pitch) + 1
        total_layers = self.rrg.layer_count
        
        logger.error(f"3D LATTICE: Building {x_steps} x {y_steps} x {total_layers} = {x_steps * y_steps * total_layers:,} lattice points")
        
        # CREATE COMPLETE 3D LATTICE: Every (x,y,z) gets a rail node
        total_nodes = 0
        for z in range(total_layers):
            # Determine layer properties
            if z == 0:
                layer_name = 'F.Cu'
            elif z == total_layers - 1:
                layer_name = 'B.Cu'
            else:
                layer_name = f'In{z}.Cu'
                
            # Even Z = vertical, Odd Z = horizontal  
            direction = 'v' if (z % 2 == 0) else 'h'
            layer_nodes = 0
            
            # Create lattice point at every (x,y) coordinate on this layer
            for x_idx in range(x_steps):
                x = grid_min_x + (x_idx * grid_pitch)
                for y_idx in range(y_steps):
                    y = grid_min_y + (y_idx * grid_pitch)
                    
                    # Create rail node at lattice point (x_idx, y_idx, z)
                    node_id = f"rail_{direction}_{x_idx}_{y_idx}_{z}"
                    rail_node = RRGNode(
                        id=node_id,
                        node_type=NodeType.RAIL,
                        x=x, y=y,
                        layer=z,
                        capacity=1
                    )
                    self.rrg.add_node(rail_node)
                    layer_nodes += 1
                    
                    # Log first few F.Cu rail coordinates for verification
                    if z == 0 and x_idx < 3 and y_idx < 3:
                        logger.error(f"3D_LATTICE_SAMPLE: {node_id} at ({x:.6f}, {y:.6f}) indices=({x_idx}, {y_idx})")
                    
                    # Store F.Cu positions for escape route reference
                    if z == 0 and y_idx == 0:  # F.Cu layer, once per X
                        self.grid_x_positions.append(x)
            
            logger.error(f"3D LATTICE: Layer {z} ({layer_name}, {direction.upper()}): {layer_nodes:,} nodes")
            total_nodes += layer_nodes
        
        # CRITICAL FIX: Add missing intra-layer rail-to-rail connectivity
        self._build_intra_layer_rail_connections()
        
        # Sort and deduplicate F.Cu vertical rail positions
        self.grid_x_positions = sorted(list(set(self.grid_x_positions)))
        logger.info(f"GRID: Generated {len(self.grid_x_positions)} F.Cu vertical rail positions")
        logger.info(f"COMPLETE GRID: Created {total_nodes} total routing nodes")
    
    def _connect_pads_to_fcu_grid(self, pads: List[Pad]):
        """Connect component pads to F.Cu vertical grid with DRC-compliant routing"""
        logger.info(f"PAD CONNECTIONS: Connecting {len(pads)} pads to F.Cu vertical grid")
        
        grid_pitch = 0.4  # mm
        max_escape_distance = 0.2  # mm - maximum horizontal escape
        direct_connection_threshold = 0.5  # mm - direct connection if within this distance
        
        connections_made = 0
        escape_connections = 0
        
        # CRITICAL FIX: Create escape routes for ALL pads with consistent per-net indexing
        # Group pads by net and create consistent pad IDs that match route requests
        pads_by_net = {}
        for pad in pads:
            if pad.net_name not in pads_by_net:
                pads_by_net[pad.net_name] = []
            pads_by_net[pad.net_name].append(pad)
        
        for net_name, net_pads in pads_by_net.items():
            for pad_idx, pad in enumerate(net_pads):
                # Create pad node with per-net indexing (matches route requests)
                pad_id = f"pad_{pad_idx}_{net_name}_{pad.x_mm:.1f}_{pad.y_mm:.1f}"
                pad_node_id = f"fcu_pad_{pad_id}"
                
                pad_node = RRGNode(
                    id=pad_node_id,
                    node_type=NodeType.PAD_ENTRY,
                    x=pad.x_mm,
                    y=pad.y_mm,
                    layer=0,  # F.Cu layer
                    capacity=1
                )
                # Add node to RRG
                self.rrg.add_node(pad_node)
                
                # Track F.Cu pad node for parallel PathFinder validation
                # Get actual node index from RRG (since add_node returns None)
                if hasattr(self.rrg, 'node_count'):
                    node_idx = self.rrg.node_count - 1  # Last added node
                else:
                    node_idx = len(self.rrg.nodes) - 1  # Fallback: count of nodes - 1
                
                self.fcu_pad_nodes[pad_node_id] = node_idx
                logger.info(f"FCU_PAD_TRACK: {pad_node_id} -> node_idx {node_idx} (manual tracking)")
                
                # DRC-COMPLIANT F.CU ESCAPE ROUTING per specification:
                # 1. Vertical trace from pad (up to 5mm)
                # 2. 45° dogleg to grid rail (max 0.2mm horizontal)
                # 3. DRC-aware (no interference with other pads)
                
                # Find nearest F.Cu vertical rail
                nearest_rail_x = min(self.grid_x_positions, key=lambda x: abs(x - pad.x_mm))
                horizontal_distance = abs(pad.x_mm - nearest_rail_x)
                
                # MULTI-OPTION F.CU ESCAPE ROUTING: Pre-compute ALL valid escape options
                escape_options = self._generate_all_escape_options(pad, pad_id, grid_pitch)
                
                if escape_options:
                    # Create all escape route nodes and connections
                    for option_idx, escape_option in enumerate(escape_options):
                        self._create_escape_route_nodes(escape_option, pad_node_id, option_idx)
                        connections_made += escape_option['connection_count']
                    
                    escape_connections += len(escape_options)
                    logger.info(f"MULTI_ESCAPE: Pad {pad_id} -> {len(escape_options)} escape options created")
                else:
                    logger.warning(f"MULTI_ESCAPE: No valid escape routes found for pad {pad_id}")
        
        logger.info(f"PAD CONNECTIONS: {connections_made} total connections, {escape_connections} pads with escape options")
    
    def _generate_all_escape_options(self, pad: Pad, pad_id: str, grid_pitch: float) -> List[Dict]:
        """Generate all valid DRC-compliant escape routing options for a pad"""
        escape_options = []
        
        # Grid bounds for precise calculation
        grid_min_x = round(self.min_x / grid_pitch) * grid_pitch
        grid_min_y = round(self.min_y / grid_pitch) * grid_pitch
        
        logger.error(f"ESCAPE_BOUNDS: raw bounds=({self.min_x:.6f}, {self.min_y:.6f})")
        logger.error(f"ESCAPE_BOUNDS: grid bounds=({grid_min_x:.6f}, {grid_min_y:.6f})")
        logger.error(f"ESCAPE_BOUNDS: grid_pitch={grid_pitch:.6f}")
        
        # Escape parameters
        vertical_distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # mm
        horizontal_offsets = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]  # mm
        escape_directions = [1, -1]  # Up and down
        
        option_count = 0
        for direction in escape_directions:
            for vertical_dist in vertical_distances:
                for horizontal_offset in horizontal_offsets:
                    # Calculate escape endpoint
                    escape_y = pad.y_mm + (direction * vertical_dist)
                    escape_x = pad.x_mm + horizontal_offset
                    
                    # Find nearest grid rail position
                    target_rail_x = round(escape_x / grid_pitch) * grid_pitch
                    target_rail_y = round(escape_y / grid_pitch) * grid_pitch
                    
                    # Calculate actual horizontal dogleg distance
                    actual_horizontal_offset = abs(target_rail_x - pad.x_mm)
                    
                    # DRC check: horizontal dogleg must be ≤ 0.2mm
                    if actual_horizontal_offset <= 0.2:
                        # Calculate grid indices
                        rail_x_idx = round((target_rail_x - grid_min_x) / grid_pitch)
                        rail_y_idx = round((target_rail_y - grid_min_y) / grid_pitch)
                        
                        # Check if F.Cu rail node exists at this position
                        rail_node_id = f"rail_v_{rail_x_idx}_{rail_y_idx}_0"
                        
                        # Log coordinate verification for debugging
                        if option_count < 5:  # Only log first few for debugging
                            logger.error(f"ESCAPE_COORD_CHECK: pad_id={pad_id}")
                            logger.error(f"  pad=({pad.x_mm:.6f}, {pad.y_mm:.6f}) -> target=({target_rail_x:.6f}, {target_rail_y:.6f})")
                            logger.error(f"  indices=({rail_x_idx}, {rail_y_idx}) -> rail_node_id={rail_node_id}")
                            logger.error(f"  exists_in_rrg={rail_node_id in self.rrg.nodes}")
                        
                        if rail_node_id in self.rrg.nodes:
                            # Valid escape option found
                            escape_option = {
                                'option_id': option_count,
                                'pad_id': pad_id,
                                'vertical_distance': vertical_dist,
                                'vertical_direction': direction,
                                'horizontal_offset': actual_horizontal_offset,
                                'escape_endpoint': (pad.x_mm, escape_y),
                                'dogleg_endpoint': (target_rail_x, target_rail_y),
                                'rail_node_id': rail_node_id,
                                'rail_indices': (rail_x_idx, rail_y_idx),
                                'total_length': vertical_dist + actual_horizontal_offset,
                                'connection_count': 0  # Will be calculated during creation
                            }
                            escape_options.append(escape_option)
                            option_count += 1
                            
                            # Limit to prevent explosion of options
                            if len(escape_options) >= 30:
                                break
                
                if len(escape_options) >= 30:
                    break
            if len(escape_options) >= 30:
                break
        
        return escape_options
    
    def _create_escape_route_nodes(self, escape_option: Dict, pad_node_id: str, option_idx: int):
        """Create RRG nodes and edges for a specific escape route option"""
        option_id = escape_option['option_id']
        pad_id = escape_option['pad_id']
        
        # 1. Create vertical escape endpoint node
        vertical_node_id = f"fcu_escape_v_{pad_id}_{option_id}"
        vertical_node = RRGNode(
            id=vertical_node_id,
            node_type=NodeType.RAIL,
            x=escape_option['escape_endpoint'][0],
            y=escape_option['escape_endpoint'][1],
            layer=0,  # F.Cu
            capacity=1
        )
        self.rrg.add_node(vertical_node)
        
        # 2. Create dogleg endpoint node (if horizontal offset > 0)
        if escape_option['horizontal_offset'] > 0.01:  # Only create if significant horizontal movement
            dogleg_node_id = f"fcu_escape_d_{pad_id}_{option_id}"
            dogleg_node = RRGNode(
                id=dogleg_node_id,
                node_type=NodeType.RAIL,
                x=escape_option['dogleg_endpoint'][0],
                y=escape_option['dogleg_endpoint'][1],
                layer=0,  # F.Cu
                capacity=1
            )
            self.rrg.add_node(dogleg_node)
        else:
            # Direct connection to rail (no dogleg needed)
            dogleg_node_id = vertical_node_id
        
        # 3. Create edges: Pad -> Vertical escape
        vertical_edge = RRGEdge(
            id=f"escape_vertical_{pad_id}_{option_id}",
            from_node=pad_node_id,
            to_node=vertical_node_id,
            edge_type=EdgeType.TRACK,
            base_cost=0.5,  # Low cost for escape routing
            length_mm=escape_option['vertical_distance']
        )
        self.rrg.add_edge(vertical_edge)
        
        # Bidirectional
        vertical_edge_reverse = RRGEdge(
            id=f"escape_vertical_rev_{pad_id}_{option_id}",
            from_node=vertical_node_id,
            to_node=pad_node_id,
            edge_type=EdgeType.TRACK,
            base_cost=0.5,
            length_mm=escape_option['vertical_distance']
        )
        self.rrg.add_edge(vertical_edge_reverse)
        connection_count = 2
        
        # 4. Create dogleg edge (if needed)
        if escape_option['horizontal_offset'] > 0.01:
            dogleg_edge = RRGEdge(
                id=f"escape_dogleg_{pad_id}_{option_id}",
                from_node=vertical_node_id,
                to_node=dogleg_node_id,
                edge_type=EdgeType.TRACK,
                base_cost=0.7,  # Slightly higher for dogleg
                length_mm=escape_option['horizontal_offset']
            )
            self.rrg.add_edge(dogleg_edge)
            
            # Bidirectional
            dogleg_edge_reverse = RRGEdge(
                id=f"escape_dogleg_rev_{pad_id}_{option_id}",
                from_node=dogleg_node_id,
                to_node=vertical_node_id,
                edge_type=EdgeType.TRACK,
                base_cost=0.7,
                length_mm=escape_option['horizontal_offset']
            )
            self.rrg.add_edge(dogleg_edge_reverse)
            connection_count += 2
        
        # 5. Connect dogleg endpoint to grid rail
        rail_node_id = escape_option['rail_node_id']
        
        # CRITICAL DEBUG: Verify rail node exists
        if rail_node_id not in self.rrg.nodes:
            logger.error(f"CONNECTIVITY BUG: Rail node {rail_node_id} missing for escape route {pad_id}_{option_id}")
            logger.error(f"CONNECTIVITY BUG: Available nodes sample: {list(self.rrg.nodes.keys())[:5]}")
            return  # Skip this escape route - can't connect to missing rail
        
        logger.info(f"ESCAPE_CONNECT: Connecting escape route {pad_id}_{option_id} ({dogleg_node_id}) to rail {rail_node_id}")
        
        rail_connection = RRGEdge(
            id=f"escape_to_rail_{pad_id}_{option_id}",
            from_node=dogleg_node_id,
            to_node=rail_node_id,
            edge_type=EdgeType.TRACK,
            base_cost=0.3,  # Very low cost to connect to grid
            length_mm=0.05  # Minimal connection distance
        )
        self.rrg.add_edge(rail_connection)
        
        # Bidirectional
        rail_connection_reverse = RRGEdge(
            id=f"escape_from_rail_{pad_id}_{option_id}",
            from_node=rail_node_id,
            to_node=dogleg_node_id,
            edge_type=EdgeType.TRACK,
            base_cost=0.3,
            length_mm=0.05
        )
        self.rrg.add_edge(rail_connection_reverse)
        connection_count += 2
        
        # CRITICAL FIX: Also connect vertical escape endpoint directly to rail (for PathFinder routing)
        if vertical_node_id != dogleg_node_id:  # Only if dogleg exists
            logger.info(f"DIRECT_RAIL_CONNECT: Connecting vertical endpoint {vertical_node_id} directly to rail {rail_node_id}")
            
            vertical_to_rail = RRGEdge(
                id=f"vertical_to_rail_{pad_id}_{option_id}",
                from_node=vertical_node_id,
                to_node=rail_node_id,
                edge_type=EdgeType.TRACK,
                base_cost=0.2,  # Lower cost for direct connection
                length_mm=escape_option['horizontal_offset']
            )
            self.rrg.add_edge(vertical_to_rail)
            
            rail_to_vertical = RRGEdge(
                id=f"rail_to_vertical_{pad_id}_{option_id}",
                from_node=rail_node_id,
                to_node=vertical_node_id,
                edge_type=EdgeType.TRACK,
                base_cost=0.2,
                length_mm=escape_option['horizontal_offset']
            )
            self.rrg.add_edge(rail_to_vertical)
            connection_count += 2
        
        # 6. Create blind/buried vias from rail to ALL inner layers
        rail_x_idx, rail_y_idx = escape_option['rail_indices']
        total_layers = self.rrg.layer_count
        
        for target_layer in range(1, total_layers):  # All inner layers (skip F.Cu)
            # Determine rail direction for target layer  
            target_direction = 'h' if (target_layer % 2 == 1) else 'v'
            target_rail_id = f"rail_{target_direction}_{rail_x_idx}_{rail_y_idx}_{target_layer}"
            
            if target_rail_id in self.rrg.nodes:
                logger.debug(f"VIA_CONNECT: Creating via from {rail_node_id} to {target_rail_id}")
                # Blind/buried via from F.Cu rail to inner layer
                via_to_inner = RRGEdge(
                    id=f"via_{pad_id}_{option_id}_to_L{target_layer}",
                    from_node=rail_node_id,
                    to_node=target_rail_id,
                    edge_type=EdgeType.SWITCH,
                    base_cost=2.0 + target_layer * 0.5,  # Reduced via costs for PathFinder
                    length_mm=0.3 + target_layer * 0.05
                )
                self.rrg.add_edge(via_to_inner)
                
                # Bidirectional via
                via_from_inner = RRGEdge(
                    id=f"via_{pad_id}_{option_id}_from_L{target_layer}",
                    from_node=target_rail_id,
                    to_node=rail_node_id,
                    edge_type=EdgeType.SWITCH,
                    base_cost=2.0 + target_layer * 0.5,
                    length_mm=0.3 + target_layer * 0.05
                )
                self.rrg.add_edge(via_from_inner)
                connection_count += 2
            else:
                logger.warning(f"VIA_MISSING: Target rail {target_rail_id} not found for via connection")
        
        # Update connection count in escape option
        escape_option['connection_count'] = connection_count
    
    def _calculate_airwire_bounds(self, airwires: List[Dict]) -> Tuple[float, float, float, float]:
        """Calculate bounding box from airwire endpoints"""
        if not airwires:
            return (0.0, 0.0, 0.0, 0.0)
        
        min_x = min(min(aw['start_x'], aw['end_x']) for aw in airwires)
        max_x = max(max(aw['start_x'], aw['end_x']) for aw in airwires)
        min_y = min(min(aw['start_y'], aw['end_y']) for aw in airwires)
        max_y = max(max(aw['start_y'], aw['end_y']) for aw in airwires)
        
        # Add margin for routing
        margin = 3.0  # mm
        return (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
    
    def _build_intra_layer_rail_connections(self):
        """Create rail-to-rail connections within each layer for continuous routing paths"""
        logger.error("INTRA_LAYER: Building rail-to-rail connections within each layer")
        
        grid_pitch = 0.4  # mm
        intra_connections = 0
        
        # Grid bounds (same as used in grid generation)
        grid_min_x = round(self.min_x / grid_pitch) * grid_pitch
        grid_max_x = round(self.max_x / grid_pitch) * grid_pitch
        grid_min_y = round(self.min_y / grid_pitch) * grid_pitch
        grid_max_y = round(self.max_y / grid_pitch) * grid_pitch
        
        # Calculate grid dimensions for iteration
        x_steps = int(round((grid_max_x - grid_min_x) / grid_pitch)) + 1
        y_steps = int(round((grid_max_y - grid_min_y) / grid_pitch)) + 1
        
        total_layers = self.rrg.layer_count
        
        # For each layer, connect adjacent rail nodes
        for layer_id in range(total_layers):
            layer_direction = 'h' if (layer_id % 2 == 1) else 'v'
            layer_connections = 0
            logger.error(f"INTRA_LAYER: Processing layer {layer_id} ({layer_direction}) with {x_steps}x{y_steps} grid")
            
            if layer_direction == 'h':
                # HORIZONTAL LAYER: Connect rails left-to-right (increasing X)
                for y_idx in range(y_steps):
                    for x_idx in range(x_steps - 1):  # -1 because we connect to x_idx+1
                        from_node_id = f"rail_h_{x_idx}_{y_idx}_{layer_id}"
                        to_node_id = f"rail_h_{x_idx+1}_{y_idx}_{layer_id}"
                        
                        if from_node_id in self.rrg.nodes and to_node_id in self.rrg.nodes:
                            logger.error(f"INTRA_CONNECT: Connecting {from_node_id} -> {to_node_id}")
                            # Forward edge: left to right
                            forward_edge = RRGEdge(
                                id=f"h_rail_{layer_id}_{x_idx}_{y_idx}_to_{x_idx+1}",
                                from_node=from_node_id,
                                to_node=to_node_id,
                                edge_type=EdgeType.TRACK,
                                base_cost=1.0,
                                length_mm=grid_pitch
                            )
                            self.rrg.add_edge(forward_edge)
                            
                            # Backward edge: right to left (bidirectional)
                            backward_edge = RRGEdge(
                                id=f"h_rail_{layer_id}_{x_idx+1}_{y_idx}_to_{x_idx}",
                                from_node=to_node_id,
                                to_node=from_node_id,
                                edge_type=EdgeType.TRACK,
                                base_cost=1.0,
                                length_mm=grid_pitch
                            )
                            self.rrg.add_edge(backward_edge)
                            layer_connections += 2
                            
                            # Log first few connections for verification
                            if layer_connections <= 10:
                                logger.error(f"INTRA_H_CONNECT: Layer {layer_id} connected {from_node_id} <-> {to_node_id}")
                                logger.error(f"  Forward edge: {forward_edge.id} cost={forward_edge.base_cost}")
                                logger.error(f"  Backward edge: {backward_edge.id} cost={backward_edge.base_cost}")
                            
            else:  # layer_direction == 'v'
                # VERTICAL LAYER: Connect rails top-to-bottom (increasing Y)
                for x_idx in range(x_steps):
                    for y_idx in range(y_steps - 1):  # -1 because we connect to y_idx+1
                        from_node_id = f"rail_v_{x_idx}_{y_idx}_{layer_id}"
                        to_node_id = f"rail_v_{x_idx}_{y_idx+1}_{layer_id}"
                        
                        if from_node_id in self.rrg.nodes and to_node_id in self.rrg.nodes:
                            logger.error(f"INTRA_CONNECT: Connecting {from_node_id} -> {to_node_id}")
                            # Forward edge: top to bottom
                            forward_edge = RRGEdge(
                                id=f"v_rail_{layer_id}_{x_idx}_{y_idx}_to_{y_idx+1}",
                                from_node=from_node_id,
                                to_node=to_node_id,
                                edge_type=EdgeType.TRACK,
                                base_cost=1.0,
                                length_mm=grid_pitch
                            )
                            self.rrg.add_edge(forward_edge)
                            
                            # Backward edge: bottom to top (bidirectional)
                            backward_edge = RRGEdge(
                                id=f"v_rail_{layer_id}_{x_idx}_{y_idx+1}_to_{y_idx}",
                                from_node=to_node_id,
                                to_node=from_node_id,
                                edge_type=EdgeType.TRACK,
                                base_cost=1.0,
                                length_mm=grid_pitch
                            )
                            self.rrg.add_edge(backward_edge)
                            layer_connections += 2
                            
                            # Log first few connections for verification
                            if layer_connections <= 10:
                                logger.error(f"INTRA_V_CONNECT: Layer {layer_id} connected {from_node_id} <-> {to_node_id}")
                                logger.error(f"  Forward edge: {forward_edge.id} cost={forward_edge.base_cost}")
                                logger.error(f"  Backward edge: {backward_edge.id} cost={backward_edge.base_cost}")
            
            logger.error(f"INTRA_LAYER: Added {layer_connections} rail connections on layer {layer_id} ({layer_direction})")
            intra_connections += layer_connections
        
        logger.info(f"INTRA_LAYER: Created {intra_connections} total intra-layer rail connections")
    
    def _build_grid_layer_connections(self):
        """Create vias between grid layers for layer transitions"""
        logger.info("LAYER VIAS: Creating inter-layer connections in routing grid")
        
        grid_pitch = 0.4
        via_count = 0
        
        # PRECISION FIX: Use same grid bounds as grid generation
        grid_min_x = round(self.min_x / grid_pitch) * grid_pitch
        grid_max_x = round(self.max_x / grid_pitch) * grid_pitch
        grid_min_y = round(self.min_y / grid_pitch) * grid_pitch
        grid_max_y = round(self.max_y / grid_pitch) * grid_pitch
        
        # COMPLETE LAYER CONNECTIVITY: Connect all grid layers with vias
        total_layers = self.rrg.layer_count
        
        # Connect grid points between layers using integer indices
        x_idx = 0
        x = grid_min_x
        while x <= grid_max_x + 0.001:
            y_idx = 0
            y = grid_min_y
            while y <= grid_max_y + 0.001:
                # COMPLETE LAYER CONNECTIVITY: Connect ALL layers including F.Cu to inner layers
                for from_layer in range(0, total_layers - 1):  # 0 to total_layers-2 (includes F.Cu)
                    to_layer = from_layer + 1  # 1 to total_layers-1 (includes B.Cu)
                    
                    # DEBUG: Log critical F.Cu to In1.Cu via creation
                    if from_layer == 0 and to_layer == 1 and x_idx < 5 and y_idx < 5:
                        logger.error(f"CRITICAL VIA: Creating F.Cu->In1.Cu via at grid ({x_idx},{y_idx}) between rail_v and rail_h")
                    
                    # Determine node types based on layer direction pattern
                    from_direction = 'h' if (from_layer % 2 == 1) else 'v'
                    to_direction = 'h' if (to_layer % 2 == 1) else 'v'
                    
                    from_node = f"rail_{from_direction}_{x_idx}_{y_idx}_{from_layer}"
                    to_node = f"rail_{to_direction}_{x_idx}_{y_idx}_{to_layer}"
                    
                    if from_node in self.rrg.nodes and to_node in self.rrg.nodes:
                        via_edge = RRGEdge(
                            id=f"grid_via_{x_idx}_{y_idx}_{from_layer}_{to_layer}",
                            from_node=from_node,
                            to_node=to_node,
                            edge_type=EdgeType.SWITCH,
                            base_cost=3.0 + from_layer + to_layer,
                            length_mm=0.4 + (abs(from_layer - to_layer) * 0.05)
                        )
                        self.rrg.add_edge(via_edge)
                        
                        # Bidirectional via
                        via_edge_reverse = RRGEdge(
                            id=f"grid_via_{x_idx}_{y_idx}_{to_layer}_{from_layer}",
                            from_node=to_node,
                            to_node=from_node,
                            edge_type=EdgeType.SWITCH,
                            base_cost=3.0 + from_layer + to_layer,
                            length_mm=0.4 + (abs(from_layer - to_layer) * 0.05)
                        )
                        self.rrg.add_edge(via_edge_reverse)
                        via_count += 2
                
                y += grid_pitch
                y_idx += 1
            x += grid_pitch
            x_idx += 1
        
        logger.info(f"LAYER VIAS: Created {via_count} inter-layer grid connections")
    
    def _validate_fabric_connectivity(self):
        """Check if fabric has basic connectivity"""
        pad_nodes = [node for node in self.rrg.nodes.values() 
                    if node.node_type == NodeType.PAD_ENTRY]
        logger.info(f"Fabric validation: {len(pad_nodes)} pads connected")
    
    def _report_fabric_stats(self):
        """Report final fabric statistics"""
        node_count = len(self.rrg.nodes)
        edge_count = len(self.rrg.edges)
        avg_connectivity = edge_count / node_count if node_count > 0 else 0
        
        # Memory estimates
        estimated_memory_mb = (node_count * 64 + edge_count * 32) / (1024 * 1024)
        
        logger.info(f"Constrained Sparse RRG built: {node_count:,} nodes, {edge_count:,} edges")
        logger.info(f"   Global grid: 0×0 @ 0.2mm")
        logger.info(f"   Local areas: 0 (using constrained fabric)")
        logger.info(f"   Estimated memory: {estimated_memory_mb:.1f}MB")
        logger.info(f"   Average connectivity: {avg_connectivity:.1f} edges per node (ideal for sparse CSR)")
        logger.info(f"   Dense matrix would require: {(node_count * node_count * 4) / (1024*1024*1024):.1f}GB")
        logger.info(f"   Sparse CSR requires: {(edge_count * 8) / (1024*1024):.1f}MB ({int((node_count * node_count) / edge_count):,}x smaller)")
        logger.info(f"Memory usage: {estimated_memory_mb:.1f}MB - within reasonable limits")

    # === END OF SPARSE RRG BUILDER ===
