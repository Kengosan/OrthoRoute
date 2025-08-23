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
                
                # Find airwires for this net
                net_airwires = []
                all_airwires = self.board_data.get('airwires', [])
                for airwire in all_airwires:
                    if airwire.get('net') == net_name:
                        net_airwires.append(airwire)
                
                self._routable_nets[net_name] = {
                    'pad_indices': pad_indices,
                    'pad_count': len(pad_indices),
                    'routed': already_routed,
                    'airwires': net_airwires,
                    'pads': [pads[i] for i in pad_indices if i < len(pads)]
                }
        
        logger.info(f"🏗️ Caches built: {len(net_to_pads)} nets, {len(self._routable_nets)} routable")
        
        # Debug airwires distribution
        total_airwires = sum(len(net_data['airwires']) for net_data in self._routable_nets.values())
        nets_with_airwires = sum(1 for net_data in self._routable_nets.values() if len(net_data['airwires']) > 0)
        logger.info(f"🔗 Airwires distribution: {total_airwires} total airwires, {nets_with_airwires} nets have airwires")
    
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
    
    def get_airwire_bounds(self, margin_mm: float = 3.0) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding rectangle that contains all airwires plus margin.
        This is the optimal area for routing fabric placement.
        
        Args:
            margin_mm: Extra margin around airwires in millimeters (default 3.0mm)
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y) including margin
        """
        airwires = self.board_data.get('airwires', [])
        
        if not airwires:
            logger.warning("⚠️ No airwires found - falling back to board bounds for fabric area")
            return self.get_board_bounds()
        
        # Find the bounding box of all airwire endpoints
        min_x = float('inf')
        min_y = float('inf') 
        max_x = float('-inf')
        max_y = float('-inf')
        
        airwire_count = 0
        for airwire in airwires:
            # Check both start and end points of each airwire
            points = [
                (airwire.get('start_x'), airwire.get('start_y')),
                (airwire.get('end_x'), airwire.get('end_y'))
            ]
            
            for x, y in points:
                if x is not None and y is not None:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    airwire_count += 1
        
        if min_x == float('inf'):
            logger.warning("⚠️ No valid airwire coordinates found - falling back to board bounds")
            return self.get_board_bounds()
        
        # Add margin around the airwire bounding box
        min_x -= margin_mm
        min_y -= margin_mm
        max_x += margin_mm
        max_y += margin_mm
        
        airwire_width = max_x - min_x
        airwire_height = max_y - min_y
        
        logger.info(f"📊 Airwire routing area: {airwire_width:.1f}×{airwire_height:.1f}mm with {margin_mm}mm margin")
        logger.info(f"📍 Airwire bounds: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
        logger.info(f"🔗 Analyzed {len(airwires)} airwires, {airwire_count} coordinate pairs")
        
        return (min_x, min_y, max_x, max_y)
    
    def get_copper_layer_count(self) -> int:
        """Get the actual number of copper layers from KiCad"""
        try:
            # First try to get from board data
            if 'copper_layers' in self.board_data:
                layer_count = self.board_data['copper_layers']
                if isinstance(layer_count, int) and layer_count > 0:
                    logger.info(f"📋 Board has {layer_count} copper layers (from board data)")
                    return layer_count
            
            # Try to get from KiCad stackup if available
            stackup_info = self.get_board_stackup_info()
            if stackup_info:
                layer_count = stackup_info['layer_count']
                if layer_count > 0:
                    logger.info(f"📋 Board has {layer_count} copper layers (from KiCad stackup)")
                    return layer_count
            
            # Fallback: count available copper layers
            copper_layers = self.get_copper_layer_names()
            if copper_layers:
                layer_count = len(copper_layers)
                logger.info(f"📋 Board has {layer_count} copper layers (counted from layer names)")
                return layer_count
            
            # Final fallback: assume 2-layer board
            logger.warning("⚠️ Could not determine copper layer count, assuming 2-layer board")
            return 2
            
        except Exception as e:
            logger.warning(f"⚠️ Error getting copper layer count: {e}, assuming 2-layer board")
            return 2

    def get_layers(self) -> List[Tuple[str, int]]:
        """Get available routing layers as (name, id) tuples"""
        try:
            # Get actual copper layers from KiCad
            copper_layers = self.get_copper_layer_names()
            if copper_layers:
                # Map layer names to IDs
                layer_tuples = []
                for layer_name in copper_layers:
                    layer_id = self.get_layer_id(layer_name)
                    layer_tuples.append((layer_name, layer_id))
                return layer_tuples
            
            # Fallback based on layer count
            layer_count = self.get_copper_layer_count()
            if layer_count == 2:
                return [('F.Cu', 0), ('B.Cu', 31)]
            elif layer_count == 4:
                return [('F.Cu', 0), ('In1.Cu', 1), ('In2.Cu', 2), ('B.Cu', 31)]
            else:
                # Generate for multi-layer board
                layers = [('F.Cu', 0)]
                for i in range(1, layer_count - 1):
                    layers.append((f'In{i}.Cu', i))
                layers.append(('B.Cu', 31))
                return layers
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting layers: {e}, using 2-layer fallback")
            return [('F.Cu', 0), ('B.Cu', 31)]
    
    def get_copper_layer_names(self) -> List[str]:
        """Get the actual copper layer names from the board"""
        try:
            if (hasattr(self, 'kicad_interface') and 
                hasattr(self.kicad_interface, 'board') and 
                self.kicad_interface.board):
                
                kicad_board = self.kicad_interface.board
                
                # Try to get stackup information
                if hasattr(kicad_board, 'get_stackup'):
                    try:
                        stackup = kicad_board.get_stackup()
                        copper_layer_names = []
                        
                        for stackup_layer in stackup.layers:
                            if (stackup_layer.enabled and 
                                stackup_layer.layer != -1):
                                
                                # Get layer name
                                layer_name = getattr(stackup_layer, 'user_name', '')
                                if not layer_name:
                                    # Generate standard layer name
                                    layer_id = stackup_layer.layer
                                    if layer_id == 0:
                                        layer_name = 'F.Cu'
                                    elif layer_id == 31:
                                        layer_name = 'B.Cu'
                                    elif 1 <= layer_id <= 30:
                                        layer_name = f'In{layer_id}.Cu'
                                    else:
                                        continue  # Skip non-copper layers
                                
                                # Only include copper layers
                                if '.Cu' in layer_name or 'copper' in layer_name.lower():
                                    copper_layer_names.append(layer_name)
                        
                        if copper_layer_names:
                            logger.info(f"📋 Board copper layers: {copper_layer_names}")
                            return copper_layer_names
                            
                    except Exception as e:
                        logger.debug(f"Stackup layer names failed: {e}")
                
                # Fallback: try to determine from board layer count
                layer_count = self.board_data.get('copper_layers', 2)
                if isinstance(layer_count, int) and layer_count > 2:
                    # Generate standard copper layer names for multi-layer board
                    copper_layers = ['F.Cu']
                    for i in range(1, layer_count - 1):
                        copper_layers.append(f'In{i}.Cu')
                    copper_layers.append('B.Cu')
                    logger.info(f"📋 Generated copper layer names for {layer_count} layers: {copper_layers}")
                    return copper_layers
                        
        except Exception as e:
            logger.debug(f"Could not get copper layer names: {e}")
        
        return []  # Return empty if detection failed
    
    def get_layer_id(self, layer_name: str) -> int:
        """Convert layer name to KiCad layer ID for IPC API"""
        # Standard KiCad layer mapping for IPC API
        layer_map = {
            'F.Cu': 0,      # Front copper
            'In1.Cu': 1,    # Inner layer 1
            'In2.Cu': 2,    # Inner layer 2
            'In3.Cu': 3,    # Inner layer 3
            'In4.Cu': 4,    # Inner layer 4
            'In5.Cu': 5,    # Inner layer 5
            'In6.Cu': 6,    # Inner layer 6
            'In7.Cu': 7,    # Inner layer 7
            'In8.Cu': 8,    # Inner layer 8
            'In9.Cu': 9,    # Inner layer 9
            'In10.Cu': 10,  # Inner layer 10
            'In11.Cu': 11,  # Inner layer 11
            'In12.Cu': 12,  # Inner layer 12
            'In13.Cu': 13,  # Inner layer 13
            'In14.Cu': 14,  # Inner layer 14
            'In15.Cu': 15,  # Inner layer 15
            'In16.Cu': 16,  # Inner layer 16
            'In17.Cu': 17,  # Inner layer 17
            'In18.Cu': 18,  # Inner layer 18
            'In19.Cu': 19,  # Inner layer 19
            'In20.Cu': 20,  # Inner layer 20
            'In21.Cu': 21,  # Inner layer 21
            'In22.Cu': 22,  # Inner layer 22
            'In23.Cu': 23,  # Inner layer 23
            'In24.Cu': 24,  # Inner layer 24
            'In25.Cu': 25,  # Inner layer 25
            'In26.Cu': 26,  # Inner layer 26
            'In27.Cu': 27,  # Inner layer 27
            'In28.Cu': 28,  # Inner layer 28
            'In29.Cu': 29,  # Inner layer 29
            'In30.Cu': 30,  # Inner layer 30
            'B.Cu': 31      # Back copper
        }
        return layer_map.get(layer_name, 0)  # Default to F.Cu if not found
    
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
    
    def get_board_stackup_info(self) -> Optional[Dict]:
        """Get detailed board stackup information using KiCad API"""
        try:
            if (hasattr(self, 'kicad_interface') and 
                hasattr(self.kicad_interface, 'board') and 
                self.kicad_interface.board):
                
                kicad_board = self.kicad_interface.board
                
                # Get board stackup using KiCad Python API
                stackup = kicad_board.get_stackup()
                
                stackup_info = {
                    'layers': [],
                    'copper_layers': [],
                    'layer_count': 0,
                    'board_thickness': 0.0
                }
                
                for stackup_layer in stackup.layers:
                    layer_info = {
                        'layer_id': stackup_layer.layer,
                        'enabled': stackup_layer.enabled,
                        'type': getattr(stackup_layer, 'type', 'unknown'),
                        'thickness': getattr(stackup_layer, 'thickness', 0.0),
                        'name': getattr(stackup_layer, 'user_name', f'Layer_{stackup_layer.layer}')
                    }
                    
                    stackup_info['layers'].append(layer_info)
                    
                    # Track copper layers specifically
                    if (stackup_layer.enabled and 
                        stackup_layer.layer != -1 and  # BL_UNDEFINED = -1
                        layer_info['type'] in ['signal', 'power', 'ground', 'mixed']):
                        stackup_info['copper_layers'].append(stackup_layer.layer)
                        stackup_info['board_thickness'] += layer_info['thickness']
                
                stackup_info['layer_count'] = len(stackup_info['copper_layers'])
                
                logger.info(f"📋 Board stackup: {stackup_info['layer_count']} copper layers, {stackup_info['board_thickness']:.3f}mm thick")
                return stackup_info
                
        except Exception as e:
            logger.debug(f"Could not get KiCad board stackup: {e}")
        
        return None
    
    def get_visible_copper_layers(self) -> List[int]:
        """Get visible copper layers using KiCad API"""
        try:
            if (hasattr(self, 'kicad_interface') and 
                hasattr(self.kicad_interface, 'board') and 
                self.kicad_interface.board):
                
                kicad_board = self.kicad_interface.board
                visible_layers = kicad_board.get_visible_layers()
                
                # Filter for copper layers (F.Cu=0, In1.Cu=1, ..., B.Cu=31)
                copper_layers = []
                for layer_id in visible_layers:
                    if (layer_id == 0 or    # F.Cu (front copper)
                        layer_id == 31 or   # B.Cu (back copper)
                        (1 <= layer_id <= 30)):  # Inner copper layers
                        copper_layers.append(layer_id)
                
                logger.info(f"📋 Visible copper layers: {copper_layers}")
                return copper_layers
                
        except Exception as e:
            logger.debug(f"Could not get KiCad visible layers: {e}")
        
        return []
    
    def get_airwires(self) -> List[Dict]:
        """
        Get all airwires from the board data.
        
        Returns:
            List of airwire dictionaries with start/end positions and net info
        """
        airwires = self.board_data.get('airwires', [])
        logger.debug(f"🔗 Retrieved {len(airwires)} airwires from board data")
        return airwires
        
    def get_footprints(self) -> List[Dict]:
        """
        Get all footprints from the board data.
        
        Returns:
            List of footprint dictionaries with components and pads
        """
        # Get components (footprints)
        components = self.board_data.get('components', [])
        
        # Group pads by footprint reference
        pads = self.board_data.get('pads', [])
        pads_by_footprint = {}
        
        for pad in pads:
            # Get the parent footprint reference
            footprint_ref = pad.get('footprint_ref', '')
            if not footprint_ref:
                continue
                
            if footprint_ref not in pads_by_footprint:
                pads_by_footprint[footprint_ref] = []
            pads_by_footprint[footprint_ref].append(pad)
        
        # Create footprints with their pads
        footprints = []
        for comp in components:
            ref = comp.get('reference', '')
            footprint = {
                'reference': ref,
                'value': comp.get('value', ''),
                'position': (comp.get('x', 0), comp.get('y', 0)),
                'rotation': comp.get('rotation', 0),
                'layer': comp.get('layer', 'F.Cu'),
                'pads': pads_by_footprint.get(ref, [])
            }
            footprints.append(footprint)
            
        logger.debug(f"📋 Retrieved {len(footprints)} footprints from board data")
        return footprints
        
    def get_nets(self) -> Dict[str, Dict]:
        """
        Get all nets from the board.
        
        Returns:
            Dictionary of nets (net_name -> net_data)
        """
        return self.get_routable_nets()
