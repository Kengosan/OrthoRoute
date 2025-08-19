#!/usr/bin/env python3
"""
Board Interface Management

Handles board data extraction, processing, and provides a clean interface 
for accessing PCB geometry data across all routing algorithms.
"""
import logging
import math
import sys
import os
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np

# Add src directory to Python path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try absolute imports first, fall back to relative
try:
    from data_structures.grid_config import GridConfig
except ImportError:
    from ..data_structures.grid_config import GridConfig

logger = logging.getLogger(__name__)


class BoardInterface:
    """Manages PCB board data and provides routing-specific access patterns"""
    
    def __init__(self, board_data: Dict, kicad_interface, grid_config: GridConfig):
        """
        Initialize board interface
        
        Args:
            board_data: Raw board data from KiCad
            kicad_interface: KiCad interface instance
            grid_config: Grid configuration for coordinate conversion
        """
        self.board_data = board_data
        self.kicad_interface = kicad_interface
        self.grid_config = grid_config
        
        # Performance optimization: Cache frequently accessed data
        self._pad_net_cache = {}
        self._routable_nets = {}
        self._layer_map = {'F.Cu': 0, 'B.Cu': 1}  # Layer mapping
        
        self._build_caches()
        
        # Statistics
        self.stats = {
            'total_pads': len(self.board_data.get('pads', [])),
            'total_nets': len(self._pad_net_cache),
            'routable_nets': len(self._routable_nets),
            'tracks': len(self.board_data.get('tracks', [])),
            'vias': len(self.board_data.get('vias', []))
        }
        
        logger.info(f"🏗️ Board Interface initialized:")
        logger.info(f"  Pads: {self.stats['total_pads']}")
        logger.info(f"  Nets: {self.stats['total_nets']} total, {self.stats['routable_nets']} routable")
        logger.info(f"  Existing: {self.stats['tracks']} tracks, {self.stats['vias']} vias")
    
    def _build_caches(self):
        """Build performance caches for commonly accessed data"""
        logger.debug("🏗️ Building board data caches...")
        
        # Build pad-to-net mapping cache
        pads = self.board_data.get('pads', [])
        net_to_pads = {}
        
        for i, pad in enumerate(pads):
            pad_net = pad.get('net')
            pad_net_name = self._extract_net_name(pad_net)
            
            if pad_net_name:
                if pad_net_name not in net_to_pads:
                    net_to_pads[pad_net_name] = []
                net_to_pads[pad_net_name].append(i)
        
        self._pad_net_cache = net_to_pads
        
        # Build routable nets cache (nets with 2+ pads)
        # Get existing routing status from KiCad data
        kicad_nets = self.board_data.get('nets', [])
        kicad_net_status = {}
        for net in kicad_nets:
            net_name = net.get('name', '')
            kicad_net_status[net_name] = net.get('routed', False)
        
        for net_name, pad_indices in net_to_pads.items():
            if len(pad_indices) >= 2 and net_name.strip() and net_name not in ['', 'NC']:
                # Use existing routing status from KiCad data if available
                already_routed = kicad_net_status.get(net_name, False)
                self._routable_nets[net_name] = {
                    'pad_indices': pad_indices,
                    'pad_count': len(pad_indices),
                    'routed': already_routed
                }
        
        logger.info(f"🏗️ Caches built: {len(net_to_pads)} nets, {len(self._routable_nets)} routable")
    
    def _extract_net_name(self, pad_net) -> str:
        """Extract net name from various pad net object types"""
        if hasattr(pad_net, 'name'):
            return pad_net.name
        elif isinstance(pad_net, dict):
            return pad_net.get('name', '')
        elif isinstance(pad_net, str):
            return pad_net
        return ''
    
    def get_routable_nets(self) -> Dict[str, Dict]:
        """Get all nets that can be routed (have 2+ pads)"""
        return self._routable_nets.copy()
    
    def get_pads_for_net(self, net_name: str) -> List[Dict]:
        """Get all pads for a specific net"""
        if net_name not in self._pad_net_cache:
            return []
        
        pads = self.board_data.get('pads', [])
        pad_indices = self._pad_net_cache[net_name]
        
        return [pads[i] for i in pad_indices if i < len(pads)]
    
    def get_all_pads(self) -> List[Dict]:
        """Get all pads on the board"""
        return self.board_data.get('pads', [])
    
    def get_all_tracks(self) -> List[Dict]:
        """Get all existing tracks on the board"""
        return self.board_data.get('tracks', [])
    
    def get_all_vias(self) -> List[Dict]:
        """Get all existing vias on the board"""
        return self.board_data.get('vias', [])
    
    def get_all_zones(self) -> List[Dict]:
        """Get all copper zones on the board"""
        return self.board_data.get('zones', [])
    
    def get_board_bounds(self) -> Tuple[float, float, float, float]:
        """Get board boundaries as (min_x, min_y, max_x, max_y)"""
        return self.board_data.get('bounds', [-50, -50, 50, 50])
    
    def get_layers(self) -> List[str]:
        """Get available routing layers"""
        return ['F.Cu', 'B.Cu']  # Start with 2-layer support
    
    def get_layer_id(self, layer_name: str) -> int:
        """Convert layer name to KiCad layer ID"""
        layer_map = {
            'F.Cu': 3,   # KiCad IPC API uses layer 3 for front copper
            'B.Cu': 34   # KiCad IPC API uses layer 34 for back copper
        }
        return layer_map.get(layer_name, 3)
    
    def is_pad_on_layer(self, pad: Dict, layer: str) -> bool:
        """Check if a pad exists on a specific layer"""
        pad_layers = pad.get('layers', [])
        
        # Check if pad is on this layer
        # For through-hole pads, they exist on both F.Cu and B.Cu
        is_on_layer = (layer in pad_layers or 
                      (not pad_layers and pad.get('drill_diameter', 0) > 0) or  # Through-hole with empty layers
                      ('F.Cu' in pad_layers and 'B.Cu' in pad_layers))  # Through-hole explicit
        
        return is_on_layer
    
    def get_pad_geometry(self, pad: Dict) -> Dict[str, float]:
        """Get pad geometry information"""
        return {
            'x': pad.get('x', 0),
            'y': pad.get('y', 0),
            'size_x': pad.get('size_x', 1.0),
            'size_y': pad.get('size_y', 1.0),
            'rotation': pad.get('rotation', 0),
            'drill_diameter': pad.get('drill_diameter', 0),
            'uuid': pad.get('uuid', '')
        }
    
    def get_track_geometry(self, track: Dict) -> Dict[str, Any]:
        """Get track geometry information"""
        return {
            'start_x': track.get('start_x', 0),
            'start_y': track.get('start_y', 0),
            'end_x': track.get('end_x', 0),
            'end_y': track.get('end_y', 0),
            'width': track.get('width', 0.25),
            'layer': track.get('layer', 0),
            'net_name': track.get('net', ''),  # FIX: Use 'net' field from KiCad interface
            'uuid': track.get('uuid', '')
        }
    
    def calculate_connection_distance(self, pad_a: Dict, pad_b: Dict) -> float:
        """Calculate Euclidean distance between two pads"""
        dx = pad_b.get('x', 0) - pad_a.get('x', 0)
        dy = pad_b.get('y', 0) - pad_a.get('y', 0)
        return math.sqrt(dx * dx + dy * dy)
    
    def get_pad_clearance_zone(self, pad: Dict, trace_width: float, required_clearance: float) -> Dict:
        """Calculate the obstacle zone around a pad including clearance"""
        pad_x, pad_y = pad.get('x', 0), pad.get('y', 0)
        size_x, size_y = pad.get('size_x', 1.0), pad.get('size_y', 1.0)
        
        # Calculate total obstacle zone with edge-to-edge clearance
        pad_radius_x = size_x / 2
        pad_radius_y = size_y / 2
        track_radius = trace_width / 2
        
        obstacle_radius_x = pad_radius_x + required_clearance + track_radius
        obstacle_radius_y = pad_radius_y + required_clearance + track_radius
        
        return {
            'center_x': pad_x,
            'center_y': pad_y,
            'radius_x': obstacle_radius_x,
            'radius_y': obstacle_radius_y
        }
    
    def mark_net_as_routed(self, net_name: str):
        """Mark a net as successfully routed"""
        if net_name in self._routable_nets:
            self._routable_nets[net_name]['routed'] = True
    
    def get_routing_statistics(self) -> Dict:
        """Get routing statistics"""
        routed_count = sum(1 for net_data in self._routable_nets.values() if net_data['routed'])
        
        return {
            'total_nets': len(self._routable_nets),
            'routed_nets': routed_count,
            'unrouted_nets': len(self._routable_nets) - routed_count,
            'success_rate': (routed_count / len(self._routable_nets)) * 100 if self._routable_nets else 0,
            **self.stats
        }
    
    def get_board_outline(self) -> List[Dict]:
        """Get board outline points for routing boundary"""
        try:
            # Try to get board outline from zones or other sources
            # For now, return a rectangle based on board bounds
            bounds = self.board_data.get('bounds', (0, 0, 100, 80))
            min_x, min_y, max_x, max_y = bounds
            
            # Return as polygon points
            return [
                {'x': min_x, 'y': min_y},
                {'x': max_x, 'y': min_y},
                {'x': max_x, 'y': max_y},
                {'x': min_x, 'y': max_y}
            ]
        except Exception as e:
            logger.warning(f"Could not get board outline: {e}")
            return []
    
    def get_via_geometry(self) -> List[Dict]:
        """Get existing via geometries for obstacle avoidance"""
        try:
            vias = self.board_data.get('vias', [])
            via_obstacles = []
            
            for via in vias:
                via_obstacles.append({
                    'x': via['x'],
                    'y': via['y'],
                    'diameter': via.get('via_diameter', 0.6),
                    'drill': via.get('drill_diameter', 0.3)
                })
            
            return via_obstacles
        except Exception as e:
            logger.warning(f"Could not get via geometry: {e}")
            return []
    
    def get_board_holes(self) -> List[Dict]:
        """Get board holes and cutouts for obstacle avoidance"""
        try:
            # For now, return empty list - could be enhanced to get actual holes
            # from board outline or special pads
            return []
        except Exception as e:
            logger.warning(f"Could not get board holes: {e}")
            return []
