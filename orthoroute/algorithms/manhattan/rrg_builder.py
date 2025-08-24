"""
RRG Builder - Constructs the routing fabric network
"""

import logging
import math
from typing import Dict, List, Tuple, Set, Any
from .rrg import (
    RoutingResourceGraph, RRGNode, RRGEdge, NodeType, EdgeType, RoutingConfig
)
from .types import Pad

logger = logging.getLogger(__name__)

class RRGFabricBuilder:
    """Builds the orthogonal routing fabric as an RRG"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.rrg = RoutingResourceGraph(config)
        
        # Board bounds
        self.min_x = 0.0
        self.min_y = 0.0  
        self.max_x = 0.0
        self.max_y = 0.0
        
        # Grid parameters
        self.grid_cols = 0
        self.grid_rows = 0
        self.rails_per_col = 1  # Number of vertical tracks per column
        self.buses_per_row = 1  # Number of horizontal tracks per row
        
        # Track spacing (center-to-center)
        self.track_spacing = self.config.track_width + self.config.clearance
        
    def build_fabric(self, board_bounds: Tuple[float, float, float, float], 
                    pads: List[Pad]) -> RoutingResourceGraph:
        """
        Build complete routing fabric
        
        Args:
            board_bounds: (min_x, min_y, max_x, max_y) in mm
            pads: List of pads to connect
        """
        self.min_x, self.min_y, self.max_x, self.max_y = board_bounds
        
        logger.info(f"🏗️ Building RRG fabric for {self.max_x-self.min_x:.1f}x{self.max_y-self.min_y:.1f}mm board")
        
        # Calculate grid dimensions
        self._calculate_grid_dimensions()
        
        # Build fabric layers
        self._build_rails_and_buses()
        
        # Add switch boxes at intersections
        self._build_switch_boxes()
        
        # Connect pads with F.Cu entry/exit stubs
        self._connect_pads(pads)
        
        # Report fabric statistics
        self._report_fabric_stats()
        
        return self.rrg
    
    def _calculate_grid_dimensions(self):
        """Calculate grid dimensions based on board size and pitch"""
        width_mm = self.max_x - self.min_x
        height_mm = self.max_y - self.min_y
        
        # Grid spacing is the pitch
        self.grid_cols = max(1, int(math.ceil(width_mm / self.config.grid_pitch)))
        self.grid_rows = max(1, int(math.ceil(height_mm / self.config.grid_pitch)))
        
        # MEMORY OPTIMIZATION: Limit grid size for large boards
        MAX_GRID_CELLS = 10000  # Reasonable limit for memory usage
        total_cells = self.grid_cols * self.grid_rows
        
        if total_cells > MAX_GRID_CELLS:
            # Scale down the grid resolution
            scale_factor = math.sqrt(MAX_GRID_CELLS / total_cells)
            self.grid_cols = max(10, int(self.grid_cols * scale_factor))
            self.grid_rows = max(10, int(self.grid_rows * scale_factor))
            
            # Recalculate effective pitch
            self.config.grid_pitch = max(width_mm / self.grid_cols, height_mm / self.grid_rows)
            
            logger.warning(f"Large board detected. Scaling grid to {self.grid_cols}x{self.grid_rows} "
                          f"with {self.config.grid_pitch:.2f}mm pitch for memory efficiency")
        
        # MEMORY OPTIMIZATION: Single track per grid cell for now
        self.rails_per_col = 1  # Reduced from multiple tracks per cell
        self.buses_per_row = 1
        
        logger.info(f"RRG Grid: {self.grid_cols}×{self.grid_rows} cells, pitch={self.config.grid_pitch:.2f}mm")
        
        # Estimate memory usage
        estimated_nodes = self.grid_cols * self.grid_rows * 2  # rails + buses
        estimated_edges = estimated_nodes * 4  # rough estimate
        estimated_mb = (estimated_nodes + estimated_edges) * 0.001  # rough KB to MB
        
        logger.info(f"Estimated RRG memory: ~{estimated_mb:.1f}MB "
                   f"({estimated_nodes} nodes, ~{estimated_edges} edges)")
    
    def _build_rails_and_buses(self):
        """Build vertical rails and horizontal buses"""
        logger.debug("Building rails and buses...")
        
        # Build vertical rails (on V layers: In2.Cu=1, In4.Cu=3, etc.)
        for layer_idx in range(self.rrg.layer_count):
            if self.rrg.layer_directions[layer_idx] == 'V':
                self._build_vertical_rails(layer_idx)
        
        # Build horizontal buses (on H layers: In1.Cu=0, In3.Cu=2, etc.)
        for layer_idx in range(self.rrg.layer_count):
            if self.rrg.layer_directions[layer_idx] == 'H':
                self._build_horizontal_buses(layer_idx)
    
    def _build_vertical_rails(self, layer_idx: int):
        """Build vertical rail segments for a layer"""
        for col in range(self.grid_cols):
            x_center = self.min_x + (col + 0.5) * self.config.grid_pitch
            
            for track_idx in range(self.rails_per_col):
                # Offset tracks within the grid pitch
                track_offset = (track_idx - (self.rails_per_col - 1) / 2) * self.track_spacing
                x_pos = x_center + track_offset
                
                # Create rail segments between switch boxes (every grid row)
                for row in range(self.grid_rows - 1):
                    y_start = self.min_y + row * self.config.grid_pitch
                    y_end = self.min_y + (row + 1) * self.config.grid_pitch
                    y_center = (y_start + y_end) / 2
                    
                    # Rail segment node
                    rail_id = f"rail_L{layer_idx}_C{col}_T{track_idx}_R{row}"
                    rail_node = RRGNode(
                        id=rail_id,
                        node_type=NodeType.RAIL,
                        x=x_pos,
                        y=y_center, 
                        layer=layer_idx,
                        track_index=track_idx,
                        capacity=1
                    )
                    self.rrg.add_node(rail_node)
                    self.rrg.total_rails += 1
                    
                    # Connect to next rail segment (if exists)
                    if row < self.grid_rows - 2:
                        next_rail_id = f"rail_L{layer_idx}_C{col}_T{track_idx}_R{row+1}"
                        edge_id = f"track_{rail_id}_to_{next_rail_id}"
                        edge = RRGEdge(
                            id=edge_id,
                            edge_type=EdgeType.TRACK,
                            from_node=rail_id,
                            to_node=next_rail_id,
                            length_mm=self.config.grid_pitch,
                            base_cost=self.config.k_length * self.config.grid_pitch
                        )
                        self.rrg.add_edge(edge)
    
    def _build_horizontal_buses(self, layer_idx: int):
        """Build horizontal bus segments for a layer"""  
        for row in range(self.grid_rows):
            y_center = self.min_y + (row + 0.5) * self.config.grid_pitch
            
            for track_idx in range(self.buses_per_row):
                # Offset tracks within the grid pitch
                track_offset = (track_idx - (self.buses_per_row - 1) / 2) * self.track_spacing
                y_pos = y_center + track_offset
                
                # Create bus segments between switch boxes (every grid column)
                for col in range(self.grid_cols - 1):
                    x_start = self.min_x + col * self.config.grid_pitch
                    x_end = self.min_x + (col + 1) * self.config.grid_pitch
                    x_center = (x_start + x_end) / 2
                    
                    # Bus segment node
                    bus_id = f"bus_L{layer_idx}_R{row}_T{track_idx}_C{col}"
                    bus_node = RRGNode(
                        id=bus_id,
                        node_type=NodeType.BUS,
                        x=x_center,
                        y=y_pos,
                        layer=layer_idx,
                        track_index=track_idx,
                        capacity=1
                    )
                    self.rrg.add_node(bus_node)
                    self.rrg.total_buses += 1
                    
                    # Connect to next bus segment (if exists)
                    if col < self.grid_cols - 2:
                        next_bus_id = f"bus_L{layer_idx}_R{row}_T{track_idx}_C{col+1}"
                        edge_id = f"track_{bus_id}_to_{next_bus_id}"
                        edge = RRGEdge(
                            id=edge_id,
                            edge_type=EdgeType.TRACK,
                            from_node=bus_id,
                            to_node=next_bus_id,
                            length_mm=self.config.grid_pitch,
                            base_cost=self.config.k_length * self.config.grid_pitch
                        )
                        self.rrg.add_edge(edge)
    
    def _build_switch_boxes(self):
        """Build switch boxes at rail/bus intersections"""
        logger.debug("Building switch boxes...")
        
        switch_count = 0
        
        for col in range(self.grid_cols):
            for row in range(self.grid_rows):
                x_grid = self.min_x + col * self.config.grid_pitch
                y_grid = self.min_y + row * self.config.grid_pitch
                
                # Create switch box at this grid intersection
                switch_id = f"switch_C{col}_R{row}"
                switch_node = RRGNode(
                    id=switch_id,
                    node_type=NodeType.SWITCH,
                    x=x_grid,
                    y=y_grid,
                    layer=-1,  # Switch boxes span layers
                    capacity=999  # High capacity for switching
                )
                self.rrg.add_node(switch_node)
                switch_count += 1
                
                # Connect adjacent layers through this switch box
                self._connect_adjacent_layers_at_switch(switch_id, col, row)
        
        self.rrg.total_switches = switch_count
        logger.debug(f"Created {switch_count} switch boxes")
    
    def _connect_adjacent_layers_at_switch(self, switch_id: str, col: int, row: int):
        """Connect adjacent layers through a switch box"""
        # MEMORY OPTIMIZATION: Only connect every Nth switch box to reduce edge count
        if (col + row) % 2 != 0:  # Skip every other switch box
            return
            
        # Only allow vias between adjacent layers
        for layer_idx in range(self.rrg.layer_count - 1):
            from_layer = layer_idx
            to_layer = layer_idx + 1
            
            # Get layer directions
            from_dir = self.rrg.layer_directions[from_layer]
            to_dir = self.rrg.layer_directions[to_layer]
            
            # Find tracks that intersect at this switch point
            from_tracks = self._get_tracks_at_switch(from_layer, from_dir, col, row)
            to_tracks = self._get_tracks_at_switch(to_layer, to_dir, col, row)
            
            # MEMORY OPTIMIZATION: Only connect first track from each layer
            if from_tracks and to_tracks:
                from_track = from_tracks[0]  # Just take first track
                to_track = to_tracks[0]     # Just take first track
                
                # Calculate switch cost (via + bend if direction change)
                switch_cost = self.config.k_via
                if from_dir != to_dir:
                    switch_cost += self.config.k_bend
                
                # Switch edge from_layer -> to_layer
                edge_id = f"switch_{from_track}_to_{to_track}"
                edge = RRGEdge(
                    id=edge_id,
                    edge_type=EdgeType.SWITCH,
                    from_node=from_track,
                    to_node=to_track,
                    length_mm=0.0,  # No horizontal distance
                    base_cost=switch_cost
                )
                self.rrg.add_edge(edge)
                
                # Reverse direction edge
                reverse_edge_id = f"switch_{to_track}_to_{from_track}"
                reverse_edge = RRGEdge(
                    id=reverse_edge_id,
                    edge_type=EdgeType.SWITCH,
                    from_node=to_track,
                    to_node=from_track,
                    length_mm=0.0,
                    base_cost=switch_cost
                )
                self.rrg.add_edge(reverse_edge)
    
    def _get_tracks_at_switch(self, layer_idx: int, direction: str, col: int, row: int) -> List[str]:
        """Get track node IDs that pass through a switch point"""
        tracks = []
        
        if direction == 'V':  # Vertical rails
            for track_idx in range(self.rails_per_col):
                # Rails exist between switch boxes
                if row > 0:
                    rail_id = f"rail_L{layer_idx}_C{col}_T{track_idx}_R{row-1}"
                    if rail_id in self.rrg.nodes:
                        tracks.append(rail_id)
                if row < self.grid_rows - 1:
                    rail_id = f"rail_L{layer_idx}_C{col}_T{track_idx}_R{row}"
                    if rail_id in self.rrg.nodes:
                        tracks.append(rail_id)
        
        elif direction == 'H':  # Horizontal buses
            for track_idx in range(self.buses_per_row):
                # Buses exist between switch boxes  
                if col > 0:
                    bus_id = f"bus_L{layer_idx}_R{row}_T{track_idx}_C{col-1}"
                    if bus_id in self.rrg.nodes:
                        tracks.append(bus_id)
                if col < self.grid_cols - 1:
                    bus_id = f"bus_L{layer_idx}_R{row}_T{track_idx}_C{col}"
                    if bus_id in self.rrg.nodes:
                        tracks.append(bus_id)
        
        return tracks
    
    def _connect_pads(self, pads: List[Pad]):
        """Connect pads with F.Cu escape stubs to fabric entry points"""
        logger.debug(f"Connecting {len(pads)} pads to fabric...")
        
        for i, pad in enumerate(pads):
            # Create pad entry node
            entry_id = f"pad_entry_{pad.net_name}_{i}"
            entry_node = RRGNode(
                id=entry_id,
                node_type=NodeType.PAD_ENTRY,
                x=pad.x_mm,
                y=pad.y_mm,
                layer=-2,  # F.Cu level
                capacity=1
            )
            self.rrg.add_node(entry_node)
            
            # Find nearest fabric access point
            access_point = self._find_nearest_fabric_access(pad.x_mm, pad.y_mm)
            
            if access_point:
                # Create escape stub edge from pad to fabric
                stub_length = math.sqrt(
                    (access_point[1] - pad.x_mm)**2 + (access_point[2] - pad.y_mm)**2
                )
                
                escape_edge_id = f"escape_{entry_id}_to_{access_point[0]}"
                escape_edge = RRGEdge(
                    id=escape_edge_id,
                    edge_type=EdgeType.ENTRY,
                    from_node=entry_id,
                    to_node=access_point[0],
                    length_mm=stub_length,
                    base_cost=self.config.k_length * stub_length + self.config.k_via
                )
                self.rrg.add_edge(escape_edge)
                
                logger.debug(f"Connected pad {pad.net_name} to fabric at {access_point[0]}")
    
    def _find_nearest_fabric_access(self, x: float, y: float) -> Tuple[str, float, float]:
        """Find nearest fabric access point for a pad"""
        min_distance = float('inf')
        best_access = None
        
        # Check all rail and bus nodes for proximity
        for node in self.rrg.nodes.values():
            if node.node_type in [NodeType.RAIL, NodeType.BUS]:
                distance = math.sqrt((node.x - x)**2 + (node.y - y)**2)
                if distance < min_distance:
                    min_distance = distance
                    best_access = (node.id, node.x, node.y)
        
        return best_access
    
    def _report_fabric_stats(self):
        """Report fabric construction statistics"""
        total_nodes = len(self.rrg.nodes)
        total_edges = len(self.rrg.edges)
        
        logger.info(f"🏁 RRG Fabric built: {total_nodes} nodes, {total_edges} edges")
        logger.info(f"   Rails: {self.rrg.total_rails}, Buses: {self.rrg.total_buses}, "
                   f"Switches: {self.rrg.total_switches}")
        logger.info(f"   Grid: {self.grid_cols}x{self.grid_rows} with {self.config.grid_pitch}mm pitch")