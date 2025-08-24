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
        
        # Board bounds
        self.min_x = 0.0
        self.min_y = 0.0  
        self.max_x = 0.0
        self.max_y = 0.0
        
        # Hierarchical grid parameters  
        self.coarse_pitch = 0.4   # mm - Same as fine pitch for maximum routing quality
        self.fine_pitch = 0.4     # mm - fine local routing grid around pads
        
        self.coarse_cols = 0
        self.coarse_rows = 0
        
        # Track local routing areas around pads
        self.local_areas = []  # List of (min_x, min_y, max_x, max_y) fine grid areas
        
    def build_fabric(self, board_bounds: Tuple[float, float, float, float], 
                    pads: List[Pad], airwires: List[Dict] = None) -> RoutingResourceGraph:
        """Build sparse hierarchical routing fabric"""
        
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
        
        logger.info(f"Building Sparse RRG for {board_width:.1f}x{board_height:.1f}mm active area")
        
        # Step 1: Build coarse global routing grid
        self._build_coarse_global_grid()
        
        # Step 2: Identify local routing areas around pads
        self._identify_local_areas(pads)
        
        # Step 3: Build fine local grids in pad areas
        self._build_fine_local_grids()
        
        # Step 4: Connect local areas to global grid
        self._connect_local_to_global()
        
        # Step 5: Connect pads to local grids
        self._connect_pads_to_local_grids(pads)
        
        # Validate fabric connectivity
        self._validate_fabric_connectivity()
        
        self._report_fabric_stats()
        
        return self.rrg
    
    def _validate_fabric_connectivity(self):
        """Check if fabric has basic connectivity"""
        pad_nodes = [node for node in self.rrg.nodes.values() 
                    if node.node_type == NodeType.PAD_ENTRY]
        
        if not pad_nodes:
            logger.warning("No pad entry nodes found in fabric!")
            return
            
        # Check if pad nodes have connections
        isolated_pads = 0
        for pad_node in pad_nodes[:10]:  # Check first 10 pads
            neighbors = self.rrg.get_neighbors(pad_node.id)
            if not neighbors:
                isolated_pads += 1
                logger.warning(f"Isolated pad: {pad_node.id}")
        
        if isolated_pads > 0:
            logger.error(f"{isolated_pads} isolated pad nodes found!")
        else:
            logger.info(f"Fabric validation: {len(pad_nodes)} pads connected")
    
    def _calculate_airwire_bounds(self, airwires: List[Dict]) -> Tuple[float, float, float, float]:
        """Calculate bounding box from airwires with 3mm margin"""
        if not airwires:
            return (0, 0, 100, 100)  # Default bounds
        
        # Extract all airwire endpoints
        all_x = []
        all_y = []
        
        for airwire in airwires:
            try:
                all_x.extend([airwire['start_x'], airwire['end_x']])
                all_y.extend([airwire['start_y'], airwire['end_y']])
            except KeyError:
                logger.warning(f"Malformed airwire: {airwire}")
                continue
        
        if not all_x or not all_y:
            logger.warning("No valid airwire coordinates found")
            return (0, 0, 100, 100)
        
        # Calculate bounding box with 3mm margin
        margin = 3.0  # mm
        min_x = min(all_x) - margin
        max_x = max(all_x) + margin
        min_y = min(all_y) - margin 
        max_y = max(all_y) + margin
        
        width = max_x - min_x
        height = max_y - min_y
        
        logger.info(f"Airwire bounding box: {width:.1f}x{height:.1f}mm "
                   f"from ({min_x:.1f},{min_y:.1f}) to ({max_x:.1f},{max_y:.1f})")
        
        return (min_x, min_y, max_x, max_y)
    
    def _build_coarse_global_grid(self):
        """Build coarse global routing grid for long-distance routing"""
        width_mm = self.max_x - self.min_x
        height_mm = self.max_y - self.min_y
        
        self.coarse_cols = max(5, int(math.ceil(width_mm / self.coarse_pitch)))
        self.coarse_rows = max(5, int(math.ceil(height_mm / self.coarse_pitch)))
        
        # Sanity check - limit global grid size for very large boards
        max_global_cells = 2000000  # 2M cells = ~1GB for 16GB GPU, enabling 0.4mm pitch
        if self.coarse_cols * self.coarse_rows > max_global_cells:
            scale_factor = math.sqrt(max_global_cells / (self.coarse_cols * self.coarse_rows))
            self.coarse_cols = max(10, int(self.coarse_cols * scale_factor))
            self.coarse_rows = max(10, int(self.coarse_rows * scale_factor))
            
            # Recalculate effective pitch
            self.coarse_pitch = max(width_mm / self.coarse_cols, height_mm / self.coarse_rows)
            logger.warning(f"Limited global grid to {self.coarse_cols}×{self.coarse_rows} "
                          f"@ {self.coarse_pitch:.1f}mm pitch for memory")
        
        logger.info(f"Global grid: {self.coarse_cols}×{self.coarse_rows} @ {self.coarse_pitch:.1f}mm pitch")
        
        # Build global rails and buses
        for layer_idx in range(self.rrg.layer_count):
            direction = self.rrg.layer_directions[layer_idx]
            
            if direction == 'H':  # Horizontal buses
                for row in range(self.coarse_rows):
                    y_pos = self.min_y + (row + 0.5) * self.coarse_pitch
                    
                    for col in range(self.coarse_cols - 1):
                        x_center = self.min_x + (col + 0.5) * self.coarse_pitch
                        
                        # Global bus segment
                        bus_id = f"global_bus_L{layer_idx}_R{row}_C{col}"
                        bus_node = RRGNode(
                            id=bus_id,
                            node_type=NodeType.BUS,
                            x=x_center,
                            y=y_pos,
                            layer=layer_idx,
                            capacity=4  # Higher capacity for global routes
                        )
                        self.rrg.add_node(bus_node)
                        
                        # Connect to next segment
                        if col < self.coarse_cols - 2:
                            next_bus_id = f"global_bus_L{layer_idx}_R{row}_C{col+1}"
                            edge_id = f"global_track_{bus_id}_to_{next_bus_id}"
                            edge = RRGEdge(
                                id=edge_id,
                                edge_type=EdgeType.TRACK,
                                from_node=bus_id,
                                to_node=next_bus_id,
                                length_mm=self.coarse_pitch,
                                base_cost=self.config.k_length * self.coarse_pitch
                            )
                            self.rrg.add_edge(edge)
            
            elif direction == 'V':  # Vertical rails
                for col in range(self.coarse_cols):
                    x_pos = self.min_x + (col + 0.5) * self.coarse_pitch
                    
                    for row in range(self.coarse_rows - 1):
                        y_center = self.min_y + (row + 0.5) * self.coarse_pitch
                        
                        # Global rail segment
                        rail_id = f"global_rail_L{layer_idx}_C{col}_R{row}"
                        rail_node = RRGNode(
                            id=rail_id,
                            node_type=NodeType.RAIL,
                            x=x_pos,
                            y=y_center,
                            layer=layer_idx,
                            capacity=4  # Higher capacity for global routes
                        )
                        self.rrg.add_node(rail_node)
                        
                        # Connect to next segment
                        if row < self.coarse_rows - 2:
                            next_rail_id = f"global_rail_L{layer_idx}_C{col}_R{row+1}"
                            edge_id = f"global_track_{rail_id}_to_{next_rail_id}"
                            edge = RRGEdge(
                                id=edge_id,
                                edge_type=EdgeType.TRACK,
                                from_node=rail_id,
                                to_node=next_rail_id,
                                length_mm=self.coarse_pitch,
                                base_cost=self.config.k_length * self.coarse_pitch
                            )
                            self.rrg.add_edge(edge)
        
        # Add sparse global switch boxes (every 4th intersection to reduce memory)
        for col in range(0, self.coarse_cols, 4):
            for row in range(0, self.coarse_rows, 4):
                self._add_global_switch_box(col, row)
    
    def _add_global_switch_box(self, col: int, row: int):
        """Add switch box at global grid intersection"""
        for layer_idx in range(self.rrg.layer_count - 1):
            from_layer = layer_idx
            to_layer = layer_idx + 1
            
            from_dir = self.rrg.layer_directions[from_layer]
            to_dir = self.rrg.layer_directions[to_layer]
            
            # Find nodes at this intersection
            from_node_id = None
            to_node_id = None
            
            if from_dir == 'H' and row < self.coarse_rows:
                from_node_id = f"global_bus_L{from_layer}_R{row}_C{col}"
            elif from_dir == 'V' and col < self.coarse_cols:
                from_node_id = f"global_rail_L{from_layer}_C{col}_R{row}"
                
            if to_dir == 'H' and row < self.coarse_rows:
                to_node_id = f"global_bus_L{to_layer}_R{row}_C{col}"
            elif to_dir == 'V' and col < self.coarse_cols:
                to_node_id = f"global_rail_L{to_layer}_C{col}_R{row}"
            
            # Create switch connections
            if (from_node_id and to_node_id and 
                from_node_id in self.rrg.nodes and to_node_id in self.rrg.nodes):
                
                switch_cost = self.config.k_via
                if from_dir != to_dir:
                    switch_cost += self.config.k_bend
                
                # Bidirectional switch edges
                for direction in ['up', 'down']:
                    src = from_node_id if direction == 'up' else to_node_id
                    dst = to_node_id if direction == 'up' else from_node_id
                    
                    edge_id = f"global_switch_{src}_to_{dst}"
                    edge = RRGEdge(
                        id=edge_id,
                        edge_type=EdgeType.SWITCH,
                        from_node=src,
                        to_node=dst,
                        length_mm=0.0,
                        base_cost=switch_cost
                    )
                    self.rrg.add_edge(edge)
    
    def _identify_local_areas(self, pads: List[Pad]):
        """Identify areas that need fine routing grids around pads"""
        self.local_areas = []
        
        # Group pads by proximity - LARGER clusters for better connectivity
        pad_clusters = self._cluster_pads(pads, cluster_radius=10.0)  # 10mm clusters (was 5mm)
        
        for cluster in pad_clusters:
            # Calculate bounding box for cluster - LARGER margins
            min_x = min(pad.x_mm for pad in cluster) - 4.0  # 4mm margin (was 2mm)
            max_x = max(pad.x_mm for pad in cluster) + 4.0
            min_y = min(pad.y_mm for pad in cluster) - 4.0
            max_y = max(pad.y_mm for pad in cluster) + 4.0
            
            # Clamp to board bounds
            min_x = max(min_x, self.min_x)
            max_x = min(max_x, self.max_x)
            min_y = max(min_y, self.min_y)
            max_y = min(max_y, self.max_y)
            
            self.local_areas.append((min_x, min_y, max_x, max_y))
        
        logger.info(f"Created {len(self.local_areas)} local routing areas")
    
    def _cluster_pads(self, pads: List[Pad], cluster_radius: float) -> List[List[Pad]]:
        """Cluster pads by proximity"""
        clusters = []
        unclustered_pads = list(pads)
        
        while unclustered_pads:
            # Start new cluster with first unclustered pad
            seed_pad = unclustered_pads.pop(0)
            cluster = [seed_pad]
            
            # Find all pads within cluster radius
            i = 0
            while i < len(unclustered_pads):
                pad = unclustered_pads[i]
                
                # Check distance to any pad in current cluster
                min_dist = min(
                    math.sqrt((pad.x_mm - c_pad.x_mm)**2 + (pad.y_mm - c_pad.y_mm)**2)
                    for c_pad in cluster
                )
                
                if min_dist <= cluster_radius:
                    cluster.append(unclustered_pads.pop(i))
                else:
                    i += 1
            
            clusters.append(cluster)
        
        return clusters
    
    def _build_fine_local_grids(self):
        """Build fine routing grids in local areas"""
        for area_idx, (min_x, min_y, max_x, max_y) in enumerate(self.local_areas):
            width = max_x - min_x
            height = max_y - min_y
            
            # Calculate local grid dimensions
            local_cols = max(3, int(math.ceil(width / self.fine_pitch)))
            local_rows = max(3, int(math.ceil(height / self.fine_pitch)))
            
            # Limit local area size for memory efficiency - INCREASED limits
            if local_cols * local_rows > 1600:  # Max 40x40 local grid (was 20x20)
                scale = math.sqrt(1600 / (local_cols * local_rows))
                local_cols = max(10, int(local_cols * scale))  # Min 10x10 (was 5x5)
                local_rows = max(10, int(local_rows * scale))
            
            logger.debug(f"Local area {area_idx}: {local_cols}×{local_rows} @ {self.fine_pitch}mm")
            
            # Build local rails and buses
            for layer_idx in range(self.rrg.layer_count):
                direction = self.rrg.layer_directions[layer_idx]
                
                if direction == 'H':
                    for row in range(local_rows):
                        y_pos = min_y + (row + 0.5) * (height / local_rows)
                        
                        for col in range(local_cols - 1):
                            x_center = min_x + (col + 0.5) * (width / local_cols)
                            
                            bus_id = f"local_{area_idx}_bus_L{layer_idx}_R{row}_C{col}"
                            bus_node = RRGNode(
                                id=bus_id,
                                node_type=NodeType.BUS,
                                x=x_center,
                                y=y_pos,
                                layer=layer_idx,
                                capacity=1
                            )
                            self.rrg.add_node(bus_node)
                            
                            # Connect to next segment
                            if col < local_cols - 2:
                                next_bus_id = f"local_{area_idx}_bus_L{layer_idx}_R{row}_C{col+1}"
                                edge_id = f"local_track_{bus_id}_to_{next_bus_id}"
                                edge = RRGEdge(
                                    id=edge_id,
                                    edge_type=EdgeType.TRACK,
                                    from_node=bus_id,
                                    to_node=next_bus_id,
                                    length_mm=width / local_cols,
                                    base_cost=self.config.k_length * (width / local_cols)
                                )
                                self.rrg.add_edge(edge)
                
                elif direction == 'V':
                    for col in range(local_cols):
                        x_pos = min_x + (col + 0.5) * (width / local_cols)
                        
                        for row in range(local_rows - 1):
                            y_center = min_y + (row + 0.5) * (height / local_rows)
                            
                            rail_id = f"local_{area_idx}_rail_L{layer_idx}_C{col}_R{row}"
                            rail_node = RRGNode(
                                id=rail_id,
                                node_type=NodeType.RAIL,
                                x=x_pos,
                                y=y_center,
                                layer=layer_idx,
                                capacity=1
                            )
                            self.rrg.add_node(rail_node)
                            
                            # Connect to next segment
                            if row < local_rows - 2:
                                next_rail_id = f"local_{area_idx}_rail_L{layer_idx}_C{col}_R{row+1}"
                                edge_id = f"local_track_{rail_id}_to_{next_rail_id}"
                                edge = RRGEdge(
                                    id=edge_id,
                                    edge_type=EdgeType.TRACK,
                                    from_node=rail_id,
                                    to_node=next_rail_id,
                                    length_mm=height / local_rows,
                                    base_cost=self.config.k_length * (height / local_rows)
                                )
                                self.rrg.add_edge(edge)
            
            # Add local switch boxes (every other intersection)
            for col in range(0, local_cols, 2):
                for row in range(0, local_rows, 2):
                    self._add_local_switch_box(area_idx, col, row, local_cols, local_rows)
    
    def _add_local_switch_box(self, area_idx: int, col: int, row: int, 
                             local_cols: int, local_rows: int):
        """Add switch box in local area"""
        for layer_idx in range(self.rrg.layer_count - 1):
            from_layer = layer_idx
            to_layer = layer_idx + 1
            
            from_dir = self.rrg.layer_directions[from_layer]
            to_dir = self.rrg.layer_directions[to_layer]
            
            from_node_id = None
            to_node_id = None
            
            if from_dir == 'H' and row < local_rows:
                from_node_id = f"local_{area_idx}_bus_L{from_layer}_R{row}_C{col}"
            elif from_dir == 'V' and col < local_cols:
                from_node_id = f"local_{area_idx}_rail_L{from_layer}_C{col}_R{row}"
                
            if to_dir == 'H' and row < local_rows:
                to_node_id = f"local_{area_idx}_bus_L{to_layer}_R{row}_C{col}"
            elif to_dir == 'V' and col < local_cols:
                to_node_id = f"local_{area_idx}_rail_L{to_layer}_C{col}_R{row}"
            
            # Create switch connections
            if (from_node_id and to_node_id and 
                from_node_id in self.rrg.nodes and to_node_id in self.rrg.nodes):
                
                switch_cost = self.config.k_via + self.config.k_bend
                
                # Bidirectional switch edges
                for direction in ['up', 'down']:
                    src = from_node_id if direction == 'up' else to_node_id
                    dst = to_node_id if direction == 'up' else from_node_id
                    
                    edge_id = f"local_switch_{src}_to_{dst}"
                    edge = RRGEdge(
                        id=edge_id,
                        edge_type=EdgeType.SWITCH,
                        from_node=src,
                        to_node=dst,
                        length_mm=0.0,
                        base_cost=switch_cost
                    )
                    self.rrg.add_edge(edge)
    
    def _connect_local_to_global(self):
        """Connect local areas to global grid"""
        for area_idx, (min_x, min_y, max_x, max_y) in enumerate(self.local_areas):
            # Find nearest global grid points
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            # Find closest global nodes
            global_col = int((center_x - self.min_x) / self.coarse_pitch)
            global_row = int((center_y - self.min_y) / self.coarse_pitch)
            
            global_col = max(0, min(global_col, self.coarse_cols - 1))
            global_row = max(0, min(global_row, self.coarse_rows - 1))
            
            # Connect a few local nodes to global grid
            for layer_idx in range(self.rrg.layer_count):
                direction = self.rrg.layer_directions[layer_idx]
                
                # Find local and global node IDs
                local_node_id = None
                global_node_id = None
                
                if direction == 'H':
                    local_node_id = f"local_{area_idx}_bus_L{layer_idx}_R0_C0"
                    if global_row < self.coarse_rows:
                        global_node_id = f"global_bus_L{layer_idx}_R{global_row}_C{global_col}"
                elif direction == 'V':
                    local_node_id = f"local_{area_idx}_rail_L{layer_idx}_C0_R0"
                    if global_col < self.coarse_cols:
                        global_node_id = f"global_rail_L{layer_idx}_C{global_col}_R{global_row}"
                
                # Create connection if both nodes exist
                if (local_node_id and global_node_id and 
                    local_node_id in self.rrg.nodes and global_node_id in self.rrg.nodes):
                    
                    # Bidirectional connection
                    for src, dst in [(local_node_id, global_node_id), (global_node_id, local_node_id)]:
                        edge_id = f"connector_{src}_to_{dst}"
                        edge = RRGEdge(
                            id=edge_id,
                            edge_type=EdgeType.TRACK,
                            from_node=src,
                            to_node=dst,
                            length_mm=1.0,  # Small connector cost
                            base_cost=self.config.k_length * 1.0
                        )
                        self.rrg.add_edge(edge)
    
    def _connect_pads_to_local_grids(self, pads: List[Pad]):
        """Connect pads to their local grids"""
        logger.info(f"Connecting {len(pads)} pads to fabric with {len(self.local_areas)} local areas")
        
        pad_stats = {'connected': 0, 'isolated': 0, 'no_area': 0}
        
        # Group pads by net to create proper net-specific indices
        net_pad_counts = {}
        for pad in pads:
            if pad.net_name not in net_pad_counts:
                net_pad_counts[pad.net_name] = 0
        
        for i, pad in enumerate(pads):
            net_pad_index = net_pad_counts[pad.net_name]
            net_pad_counts[pad.net_name] += 1
            
            if i < 10:  # Debug first 10 pads
                logger.info(f"Pad {i}: {pad.net_name} at ({pad.x_mm:.1f},{pad.y_mm:.1f}) - net index {net_pad_index}")
            # Find which local area contains this pad
            containing_area = None
            for area_idx, (min_x, min_y, max_x, max_y) in enumerate(self.local_areas):
                if min_x <= pad.x_mm <= max_x and min_y <= pad.y_mm <= max_y:
                    containing_area = area_idx
                    break
            
            if containing_area is None:
                pad_stats['no_area'] += 1
                if i < 10:
                    logger.error(f"Pad {pad.net_name} not in any local area!")
                    logger.error(f"   Pad location: ({pad.x_mm:.1f},{pad.y_mm:.1f})")
                    logger.error(f"   Local areas: {[(f'Area{j}: ({a:.1f},{b:.1f})-({c:.1f},{d:.1f})') for j,(a,b,c,d) in enumerate(self.local_areas[:3])]}")
                continue
            
            # Create pad entry node with net-specific index
            entry_id = f"pad_entry_{pad.net_name}_{net_pad_index}"
            entry_node = RRGNode(
                id=entry_id,
                node_type=NodeType.PAD_ENTRY,
                x=pad.x_mm,
                y=pad.y_mm,
                layer=-2,  # F.Cu level
                capacity=1
            )
            self.rrg.add_node(entry_node)
            
            # Connect to multiple local grid nodes for better connectivity
            connections_made = 0
            for layer_idx in range(min(4, self.rrg.layer_count)):  # Connect to first 4 layers
                direction = self.rrg.layer_directions[layer_idx]
                
                # Try multiple connection points
                possible_nodes = []
                if direction == 'H':
                    for r in range(3):  # Try first 3 rows
                        for c in range(3):  # Try first 3 cols
                            node_id = f"local_{containing_area}_bus_L{layer_idx}_R{r}_C{c}"
                            if node_id in self.rrg.nodes:
                                possible_nodes.append(node_id)
                elif direction == 'V':
                    for c in range(3):  # Try first 3 cols  
                        for r in range(3):  # Try first 3 rows
                            node_id = f"local_{containing_area}_rail_L{layer_idx}_C{c}_R{r}"
                            if node_id in self.rrg.nodes:
                                possible_nodes.append(node_id)
                
                # Connect to closest nodes
                if possible_nodes:
                    # Sort by distance and connect to closest
                    node_distances = []
                    for node_id in possible_nodes:
                        local_node = self.rrg.nodes[node_id]
                        distance = math.sqrt(
                            (pad.x_mm - local_node.x)**2 + (pad.y_mm - local_node.y)**2
                        )
                        node_distances.append((distance, node_id))
                    
                    node_distances.sort()  # Sort by distance
                    
                    # Connect to closest node
                    distance, local_node_id = node_distances[0]
                    
                    # Create escape stub
                    edge_id = f"escape_{entry_id}_to_{local_node_id}"
                    edge = RRGEdge(
                        id=edge_id,
                        edge_type=EdgeType.ENTRY,
                        from_node=entry_id,
                        to_node=local_node_id,
                        length_mm=distance,
                        base_cost=self.config.k_length * distance + self.config.k_via
                    )
                    self.rrg.add_edge(edge)
                    connections_made += 1
                    
                    if connections_made >= 2:  # Limit to 2 connections per pad
                        break
                        
            if connections_made == 0:
                pad_stats['isolated'] += 1
                logger.error(f"ISOLATED PAD: {pad.net_name} at ({pad.x_mm:.1f},{pad.y_mm:.1f}) in area {containing_area}")
                # Show available local nodes for debugging
                local_nodes = [node_id for node_id in self.rrg.nodes.keys() 
                              if f"local_{containing_area}_" in node_id]
                logger.error(f"Available local nodes in area {containing_area}: {len(local_nodes)}")
                if local_nodes:
                    sample_node = self.rrg.nodes[local_nodes[0]]
                    logger.error(f"Sample node: {local_nodes[0]} at ({sample_node.x:.1f},{sample_node.y:.1f})")
                else:
                    logger.error("NO LOCAL NODES FOUND IN AREA!")
            else:
                pad_stats['connected'] += 1
                if i < 10:
                    logger.info(f"Connected pad {pad.net_name} with {connections_made} connections")
        
        # Report final statistics
        logger.info(f"Pad Connection Statistics:")
        logger.info(f"   Connected: {pad_stats['connected']}/{len(pads)} ({100.0*pad_stats['connected']/len(pads):.1f}%)")
        logger.info(f"   Isolated: {pad_stats['isolated']} ({100.0*pad_stats['isolated']/len(pads):.1f}%)")
        logger.info(f"   No Area: {pad_stats['no_area']} ({100.0*pad_stats['no_area']/len(pads):.1f}%)")
        
        if pad_stats['isolated'] > 0 or pad_stats['no_area'] > 0:
            logger.error(f"WARNING: {pad_stats['isolated'] + pad_stats['no_area']} pads have connectivity issues!")
        else:
            logger.info(f"SUCCESS: All {len(pads)} pads successfully connected to fabric!")
    
    def _report_fabric_stats(self):
        """Report fabric construction statistics"""
        total_nodes = len(self.rrg.nodes)
        total_edges = len(self.rrg.edges)
        
        # Estimate memory usage (fix calculation)
        node_memory_bytes = total_nodes * 200   # ~200 bytes per node
        edge_memory_bytes = total_edges * 300   # ~300 bytes per edge
        total_memory_mb = (node_memory_bytes + edge_memory_bytes) / (1024 * 1024)  # Convert bytes to MB
        
        logger.info(f"Sparse RRG built: {total_nodes} nodes, {total_edges} edges")
        logger.info(f"   Global grid: {self.coarse_cols}×{self.coarse_rows} @ {self.coarse_pitch}mm")
        logger.info(f"   Local areas: {len(self.local_areas)} @ {self.fine_pitch}mm")
        logger.info(f"   Estimated memory: {total_memory_mb:.1f}MB")
        
        if total_memory_mb > 1000:  # Warn if approaching 1GB
            logger.warning(f"Memory usage high: {total_memory_mb:.1f}MB")
        else:
            logger.info(f"Memory usage: {total_memory_mb:.1f}MB - within reasonable limits")