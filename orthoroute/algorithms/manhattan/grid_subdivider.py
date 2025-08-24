"""
Grid subdivision and trace breaking logic for Manhattan routing
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from .types import Track, Via, RoutingConfig
from .grid_manager import GridManager

logger = logging.getLogger(__name__)

class GridSubdivider:
    """Handles trace subdivision and grid breaking for dynamic routing"""
    
    def __init__(self, config: RoutingConfig, grid_manager: GridManager):
        self.config = config
        self.grid = grid_manager
    
    def subdivide_grid_for_via(self, gx: int, gy: int, layer: int, net_id: int) -> bool:
        """
        Subdivide grid to make space for a via connection
        
        When connecting to a net, subdivide the traces you use. When connecting with a via, 
        'break' the trace from the grid. Redraw what you need, and redraw ALMOST the complete 
        trace, leaving 0.4mm of space. The bit you're using will be used by the net, the 
        'broken' pieces may be used by other nets.
        
        Args:
            gx, gy: Grid coordinates for via
            layer: Layer index
            net_id: Net ID that needs the via space
            
        Returns:
            True if subdivision successful, False otherwise
        """
        if not (0 <= gx < self.grid.grid_width and 0 <= gy < self.grid.grid_height):
            return False
        
        # Check if cell is already free or used by same net
        if self.grid.is_cell_free(layer, gy, gx, net_id):
            return True
        
        # Find the existing net using this cell
        existing_net = self.grid.net_id_grid[layer, gy, gx]
        if existing_net == 0:
            return True  # Cell is actually free
        
        logger.debug(f"Subdividing trace from net {existing_net} at ({gx},{gy}) layer {layer}")
        
        # Find the trace segment that uses this cell
        trace_segments = self._find_trace_segments(existing_net, layer, gx, gy)
        
        if not trace_segments:
            # Just clear the cell if no continuous trace found
            self.grid.occupancy[layer, gy, gx] = 0
            self.grid.net_id_grid[layer, gy, gx] = 0
            return True
        
        # Break and rebuild trace segments
        success = True
        for segment in trace_segments:
            if not self._subdivide_trace_segment(segment, gx, gy, layer, existing_net):
                success = False
        
        # Mark the via cell as available for the requesting net
        self.grid.mark_cell_used(layer, gy, gx, net_id)
        
        return success
    
    def _find_trace_segments(self, net_id: int, layer: int, center_gx: int, center_gy: int) -> List[Dict]:
        """Find continuous trace segments around a grid cell"""
        segments = []
        visited = set()
        
        # Search for connected trace cells
        to_visit = [(center_gx, center_gy)]
        
        while to_visit:
            gx, gy = to_visit.pop()
            
            if (gx, gy) in visited:
                continue
            if not (0 <= gx < self.grid.grid_width and 0 <= gy < self.grid.grid_height):
                continue
            if self.grid.net_id_grid[layer, gy, gx] != net_id:
                continue
            
            visited.add((gx, gy))
            
            # Find connected neighbors
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = gx + dx, gy + dy
                if (0 <= nx < self.grid.grid_width and 0 <= ny < self.grid.grid_height and
                    self.grid.net_id_grid[layer, ny, nx] == net_id and (nx, ny) not in visited):
                    neighbors.append((nx, ny))
                    to_visit.append((nx, ny))
            
            if neighbors:  # This cell is part of a trace
                # Create segment info
                segment = {
                    'cells': [(gx, gy)] + neighbors,
                    'layer': layer,
                    'net_id': net_id,
                    'center': (center_gx, center_gy)
                }
                segments.append(segment)
        
        return segments
    
    def _subdivide_trace_segment(self, segment: Dict, break_gx: int, break_gy: int, layer: int, net_id: int) -> bool:
        """
        Subdivide a trace segment, leaving 0.4mm space around the break point
        
        Per spec: "redraw ALMOST the complete trace, leaving 0.4mm of space"
        """
        cells = segment['cells']
        
        # Calculate break radius in grid cells (0.4mm spacing requirement)
        break_radius = max(1, int(0.4 / self.config.grid_pitch))
        
        # Find cells that need to be cleared (within break radius)
        clear_cells = []
        keep_cells = []
        
        for gx, gy in cells:
            distance = max(abs(gx - break_gx), abs(gy - break_gy))  # Chebyshev distance
            if distance <= break_radius:
                clear_cells.append((gx, gy))
            else:
                keep_cells.append((gx, gy))
        
        # Clear the break area
        for gx, gy in clear_cells:
            if 0 <= gx < self.grid.grid_width and 0 <= gy < self.grid.grid_height:
                self.grid.occupancy[layer, gy, gx] = 0
                self.grid.net_id_grid[layer, gy, gx] = 0
        
        # Rebuild remaining trace segments
        if keep_cells:
            # Group remaining cells into continuous segments
            remaining_segments = self._group_continuous_cells(keep_cells, layer)
            
            # Mark remaining segments back into grid
            for remaining_cells in remaining_segments:
                if len(remaining_cells) >= 2:  # Only keep segments with 2+ cells
                    for gx, gy in remaining_cells:
                        if 0 <= gx < self.grid.grid_width and 0 <= gy < self.grid.grid_height:
                            self.grid.mark_cell_used(layer, gy, gx, net_id)
        
        logger.debug(f"Subdivided trace: cleared {len(clear_cells)} cells, kept {len(keep_cells)} cells")
        return True
    
    def _group_continuous_cells(self, cells: List[Tuple[int, int]], layer: int) -> List[List[Tuple[int, int]]]:
        """Group cells into continuous segments"""
        if not cells:
            return []
        
        cell_set = set(cells)
        groups = []
        remaining = set(cells)
        
        while remaining:
            # Start a new group
            start_cell = remaining.pop()
            group = [start_cell]
            to_visit = [start_cell]
            
            while to_visit:
                gx, gy = to_visit.pop()
                
                # Check neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = gx + dx, gy + dy
                    if (nx, ny) in remaining:
                        remaining.remove((nx, ny))
                        group.append((nx, ny))
                        to_visit.append((nx, ny))
            
            if group:
                groups.append(group)
        
        return groups
    
    def break_trace_for_crossing(self, gx: int, gy: int, layer1: int, layer2: int, net_id: int) -> bool:
        """
        Break traces at a crossing point where a via needs to connect two layers
        
        Args:
            gx, gy: Grid coordinates
            layer1, layer2: Layers that need to be connected
            net_id: Net ID that needs the crossing
            
        Returns:
            True if crossing space created successfully
        """
        success = True
        
        # Subdivide both layers
        success &= self.subdivide_grid_for_via(gx, gy, layer1, net_id)
        success &= self.subdivide_grid_for_via(gx, gy, layer2, net_id)
        
        # Ensure routing direction compatibility
        layer1_dir = self.grid.get_layer_direction(layer1)
        layer2_dir = self.grid.get_layer_direction(layer2)
        
        if layer1_dir == layer2_dir:
            logger.warning(f"Creating crossing between same-direction layers {layer1}({layer1_dir}) and {layer2}({layer2_dir})")
        
        return success
    
    def can_subdivide_cell(self, gx: int, gy: int, layer: int, requesting_net: int) -> bool:
        """Check if a cell can be subdivided for a requesting net"""
        if not (0 <= gx < self.grid.grid_width and 0 <= gy < self.grid.grid_height):
            return False
        
        # Already free or same net
        if self.grid.is_cell_free(layer, gy, gx, requesting_net):
            return True
        
        existing_net = self.grid.net_id_grid[layer, gy, gx]
        
        # Can't subdivide keepout areas or pads of other nets
        if self.grid.occupancy[layer, gy, gx] in [self.grid.PAD, self.grid.KEEPOUT]:
            return existing_net == requesting_net
        
        # Can subdivide traces of other nets
        return self.grid.occupancy[layer, gy, gx] == self.grid.TRACE