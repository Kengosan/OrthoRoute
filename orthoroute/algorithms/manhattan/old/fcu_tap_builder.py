"""
F.Cu Vertical Tap Escape Builder
Creates vertical traces on F.Cu layer to connect pads to fabric via blind vias
"""

import cupy as cp
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FCuTapEscape:
    """Represents a vertical tap escape on F.Cu"""
    pad_id: str
    pad_position: Tuple[float, float]
    escape_start: Tuple[float, float]  # Where vertical trace starts on F.Cu
    fabric_connection: Tuple[float, float]  # Where blind via connects to fabric
    trace_length: float
    via_position: Tuple[float, float]
    fabric_layer: int  # Which fabric layer (In1.Cu, In2.Cu, etc)

class FCuVerticalTapBuilder:
    """Builds vertical tap escapes on F.Cu for pad connections"""
    
    def __init__(self, fabric_bounds, fabric_pitch=0.4, escape_length=2.0):
        self.fabric_bounds = fabric_bounds  # (min_x, min_y, max_x, max_y)
        self.fabric_pitch = fabric_pitch
        self.escape_length = escape_length  # Length of vertical escape trace
        
        # F.Cu is layer 0, fabric starts at layer 1 (In1.Cu)
        self.fcu_layer = 0
        self.first_fabric_layer = 1
        
    def generate_fcu_tap_escapes(self, pads: List[Dict]) -> List[FCuTapEscape]:
        """Generate F.Cu vertical tap escapes for all pads"""
        
        logger.info(f"Generating F.Cu vertical tap escapes for {len(pads)} pads")
        
        escapes = []
        fabric_grid = self._build_fabric_grid_map()
        
        for pad in pads:
            escape = self._create_pad_escape(pad, fabric_grid)
            if escape:
                escapes.append(escape)
        
        logger.info(f"Generated {len(escapes)} F.Cu tap escapes")
        return escapes
    
    def _create_pad_escape(self, pad: Dict, fabric_grid: Dict) -> Optional[FCuTapEscape]:
        """Create vertical tap escape for a single pad"""
        
        pad_x = pad['x']
        pad_y = pad['y']
        pad_id = pad.get('name', f"pad_{pad_x}_{pad_y}")
        
        # Strategy: Create short vertical trace on F.Cu, then blind via to fabric
        
        # 1. Choose escape direction (up/down/left/right) to avoid conflicts
        escape_direction = self._choose_escape_direction(pad_x, pad_y, fabric_grid)
        
        # 2. Calculate escape trace endpoint
        escape_end = self._calculate_escape_endpoint(pad_x, pad_y, escape_direction)
        
        # 3. Find nearest fabric grid point
        fabric_connection = self._find_nearest_fabric_point(escape_end[0], escape_end[1])
        
        # 4. Choose fabric layer (prefer In1.Cu for short blind vias)
        fabric_layer = self._choose_fabric_layer(fabric_connection)
        
        # 5. Calculate via position (at escape endpoint)
        via_position = escape_end
        
        trace_length = self._calculate_distance((pad_x, pad_y), escape_end)
        
        return FCuTapEscape(
            pad_id=pad_id,
            pad_position=(pad_x, pad_y),
            escape_start=(pad_x, pad_y),
            fabric_connection=fabric_connection,
            trace_length=trace_length,
            via_position=via_position,
            fabric_layer=fabric_layer
        )
    
    def _choose_escape_direction(self, pad_x: float, pad_y: float, fabric_grid: Dict) -> str:
        """Choose optimal escape direction for pad"""
        
        # Simple strategy: escape towards center of fabric to minimize trace length
        fabric_center_x = (self.fabric_bounds[0] + self.fabric_bounds[2]) / 2
        fabric_center_y = (self.fabric_bounds[1] + self.fabric_bounds[3]) / 2
        
        dx = fabric_center_x - pad_x
        dy = fabric_center_y - pad_y
        
        # Choose dominant direction
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "up" if dy > 0 else "down"
    
    def _calculate_escape_endpoint(self, pad_x: float, pad_y: float, direction: str) -> Tuple[float, float]:
        """Calculate where the vertical escape trace ends"""
        
        if direction == "up":
            return (pad_x, pad_y + self.escape_length)
        elif direction == "down":
            return (pad_x, pad_y - self.escape_length)
        elif direction == "left":
            return (pad_x - self.escape_length, pad_y)
        elif direction == "right":
            return (pad_x + self.escape_length, pad_y)
        else:
            # Default to upward escape
            return (pad_x, pad_y + self.escape_length)
    
    def _find_nearest_fabric_point(self, x: float, y: float) -> Tuple[float, float]:
        """Find nearest fabric grid intersection"""
        
        # Snap to fabric grid
        fabric_x = round((x - self.fabric_bounds[0]) / self.fabric_pitch) * self.fabric_pitch + self.fabric_bounds[0]
        fabric_y = round((y - self.fabric_bounds[1]) / self.fabric_pitch) * self.fabric_pitch + self.fabric_bounds[1]
        
        # Ensure within fabric bounds
        fabric_x = max(self.fabric_bounds[0], min(fabric_x, self.fabric_bounds[2]))
        fabric_y = max(self.fabric_bounds[1], min(fabric_y, self.fabric_bounds[3]))
        
        return (fabric_x, fabric_y)
    
    def _choose_fabric_layer(self, fabric_connection: Tuple[float, float]) -> int:
        """Choose which fabric layer to connect to"""
        
        # For now, always connect to In1.Cu (layer 1) for simplicity
        # In a more sophisticated system, we'd analyze congestion and choose optimal layer
        return self.first_fabric_layer
    
    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def _build_fabric_grid_map(self) -> Dict:
        """Build map of fabric grid points for reference"""
        
        grid_points = {}
        
        min_x, min_y, max_x, max_y = self.fabric_bounds
        
        x = min_x
        while x <= max_x:
            y = min_y
            while y <= max_y:
                grid_points[(x, y)] = True
                y += self.fabric_pitch
            x += self.fabric_pitch
        
        logger.debug(f"Built fabric grid map with {len(grid_points)} points")
        return grid_points
    
    def generate_rrg_nodes_for_escapes(self, escapes: List[FCuTapEscape]) -> List[Dict]:
        """Generate RRG nodes for F.Cu tap escapes"""
        
        nodes = []
        
        for escape in escapes:
            # F.Cu trace node (start of escape)
            trace_start_node = {
                'id': f"fcu_trace_start_{escape.pad_id}",
                'type': 'trace',
                'layer': self.fcu_layer,
                'position': escape.escape_start,
                'connects_to_pad': escape.pad_id
            }
            nodes.append(trace_start_node)
            
            # F.Cu trace node (end of escape, via location)
            trace_end_node = {
                'id': f"fcu_trace_end_{escape.pad_id}",
                'type': 'trace',
                'layer': self.fcu_layer,
                'position': escape.via_position,
                'connects_to_via': f"blind_via_{escape.pad_id}"
            }
            nodes.append(trace_end_node)
            
            # Blind via node (F.Cu to fabric layer)
            via_node = {
                'id': f"blind_via_{escape.pad_id}",
                'type': 'blind_via',
                'from_layer': self.fcu_layer,
                'to_layer': escape.fabric_layer,
                'position': escape.via_position,
                'connects_to_fabric': True
            }
            nodes.append(via_node)
            
            # Fabric connection node
            fabric_node = {
                'id': f"fabric_connection_{escape.pad_id}",
                'type': 'fabric_connection',
                'layer': escape.fabric_layer,
                'position': escape.fabric_connection,
                'connects_to_fabric': True
            }
            nodes.append(fabric_node)
        
        logger.info(f"Generated {len(nodes)} RRG nodes for F.Cu tap escapes")
        return nodes
    
    def generate_rrg_edges_for_escapes(self, escapes: List[FCuTapEscape]) -> List[Dict]:
        """Generate RRG edges connecting F.Cu tap escapes"""
        
        edges = []
        
        for escape in escapes:
            # Edge: Pad to trace start
            pad_to_trace = {
                'id': f"edge_pad_to_fcu_{escape.pad_id}",
                'from_node': escape.pad_id,
                'to_node': f"fcu_trace_start_{escape.pad_id}",
                'type': 'pad_connection',
                'cost': 0.1  # Low cost for pad connections
            }
            edges.append(pad_to_trace)
            
            # Edge: Trace start to trace end (F.Cu trace)
            trace_edge = {
                'id': f"edge_fcu_trace_{escape.pad_id}",
                'from_node': f"fcu_trace_start_{escape.pad_id}",
                'to_node': f"fcu_trace_end_{escape.pad_id}",
                'type': 'fcu_trace',
                'cost': escape.trace_length * 0.1,  # Cost proportional to length
                'length': escape.trace_length
            }
            edges.append(trace_edge)
            
            # Edge: Trace end to blind via
            trace_to_via = {
                'id': f"edge_trace_to_via_{escape.pad_id}",
                'from_node': f"fcu_trace_end_{escape.pad_id}",
                'to_node': f"blind_via_{escape.pad_id}",
                'type': 'via_connection',
                'cost': 1.0  # Standard via cost
            }
            edges.append(trace_to_via)
            
            # Edge: Blind via to fabric connection
            via_to_fabric = {
                'id': f"edge_via_to_fabric_{escape.pad_id}",
                'from_node': f"blind_via_{escape.pad_id}",
                'to_node': f"fabric_connection_{escape.pad_id}",
                'type': 'fabric_connection',
                'cost': 0.1  # Low cost for fabric connections
            }
            edges.append(via_to_fabric)
        
        logger.info(f"Generated {len(edges)} RRG edges for F.Cu tap escapes")
        return edges
    
    def get_escape_statistics(self, escapes: List[FCuTapEscape]) -> Dict:
        """Get statistics about generated tap escapes"""
        
        if not escapes:
            return {}
        
        trace_lengths = [e.trace_length for e in escapes]
        fabric_layers = [e.fabric_layer for e in escapes]
        
        stats = {
            'total_escapes': len(escapes),
            'avg_trace_length': sum(trace_lengths) / len(trace_lengths),
            'max_trace_length': max(trace_lengths),
            'min_trace_length': min(trace_lengths),
            'fabric_layers_used': list(set(fabric_layers)),
            'blind_vias_required': len(escapes)  # One per escape
        }
        
        logger.info(f"F.Cu Tap Escape Statistics: {stats}")
        return stats