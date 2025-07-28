"""
Board Export for OrthoRoute KiCad Plugin
Convert KiCad board data to OrthoRoute GPU format

This module extracts all necessary information from a KiCad PCB design
and converts it to the JSON format expected by the OrthoRoute GPU engine.
"""

import pcbnew
import json
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

@dataclass
class ExportedPin:
    """Exported pin information"""
    x: int  # Nanometers
    y: int  # Nanometers
    layer: int
    size_x: int = 0
    size_y: int = 0
    drill_size: int = 0
    pad_type: str = "smd"  # "smd", "through_hole", "np_through_hole"

@dataclass
class ExportedNet:
    """Exported net information"""
    id: int
    name: str
    pins: List[ExportedPin]
    priority: int = 5
    width_nm: int = 200000
    via_size_nm: int = 200000
    net_class: str = "default"

@dataclass
class ExportedObstacle:
    """Exported obstacle/keepout information"""
    type: str  # "track", "via", "keepout", "component", "board_edge"
    x1: int
    y1: int
    x2: int
    y2: int
    layer: int = -1  # -1 means all layers
    width: int = 0

class BoardExporter:
    """Export KiCad board data to OrthoRoute format"""
    def __init__(self, board: pcbnew.BOARD):
        self.board = board
        self.unit_nm = 1000000  # KiCad units to nanometers
        
    def export_board(self) -> Dict:
        """Export board to OrthoRoute format"""
        # Get board bounds
        bounds = self._get_board_bounds()
        
        # Get net information
        nets = self._export_nets()
        
        # Get board grid settings
        grid = self._get_grid_settings()
        
        # Create board data
        board_data = {
            "bounds": bounds,
            "grid": grid,
            "nets": [net.__dict__ for net in nets],
            "design_rules": self._get_design_rules()
        }
        
        return board_data
        
    def _get_board_bounds(self) -> Dict:
        """Get board dimensions in nm"""
        bbox = self.board.GetBoardEdgesBoundingBox()
        width = bbox.GetWidth() * self.unit_nm // pcbnew.IU_PER_MM
        height = bbox.GetHeight() * self.unit_nm // pcbnew.IU_PER_MM
        
        # Count copper layers
        layers = 0
        for i in range(pcbnew.PCB_LAYER_ID_COUNT):
            if self.board.GetEnabledLayers().Contains(i):
                if pcbnew.IsCopperLayer(i):
                    layers += 1
        
        return {
            "width_nm": width,
            "height_nm": height,
            "layers": layers
        }
        
    def _get_grid_settings(self) -> Dict:
        """Get grid and via settings"""
        return {
            "pitch_nm": 100000,  # Default 0.1mm grid
            "via_size_nm": 200000  # Default 0.2mm vias
        }
        
    def _get_design_rules(self) -> Dict:
        """Get design rules from board"""
        rules = self.board.GetDesignSettings()
        
        return {
            "min_track_width_nm": rules.GetMinTrackWidth() * self.unit_nm // pcbnew.IU_PER_MM,
            "min_clearance_nm": rules.GetMinClearance() * self.unit_nm // pcbnew.IU_PER_MM,
            "min_via_size_nm": rules.GetMinViaDiameter() * self.unit_nm // pcbnew.IU_PER_MM
        }
        
    def _export_nets(self) -> List[ExportedNet]:
        """Export all nets that need routing"""
        exported_nets = []
        
        # Process each net
        for net_info in self.board.GetNetInfo().NetsByNetcode():
            net_code = net_info[0]
            net = net_info[1]
            
            # Skip power nets and nets with no pads
            if net.IsPowerNet() or net.GetNodesCount() < 2:
                continue
            
            # Get net pins
            pins = self._get_net_pins(net)
            if len(pins) < 2:
                continue
            
            # Create exported net
            exported_net = ExportedNet(
                id=net_code,
                name=net.GetNetname(),
                pins=pins,
                width_nm=200000,  # Default 0.2mm traces
                net_class=net.GetClassName()
            )
            
            exported_nets.append(exported_net)
            
        return exported_nets
        
    def _get_net_pins(self, net: pcbnew.NETINFO_ITEM) -> List[ExportedPin]:
        """Get all pins for a net"""
        pins = []
        
        # Process all pads in net
        for pad in self.board.GetPads():
            if pad.GetNetCode() != net.GetNetCode():
                continue
                
            # Get pad position
            pos = pad.GetPosition()
            x = pos.x * self.unit_nm // pcbnew.IU_PER_MM
            y = pos.y * self.unit_nm // pcbnew.IU_PER_MM
            
            # Get pad layer
            if pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD:
                layer = pad.GetLayer()
            else:
                # Through-hole pad, use top layer
                layer = pcbnew.F_Cu
            
            # Create exported pin
            pin = ExportedPin(
                x=x,
                y=y,
                layer=layer,
                size_x=pad.GetSize().x * self.unit_nm // pcbnew.IU_PER_MM,
                size_y=pad.GetSize().y * self.unit_nm // pcbnew.IU_PER_MM,
                drill_size=pad.GetDrillSize().x * self.unit_nm // pcbnew.IU_PER_MM,
                pad_type="smd" if pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD else "through_hole"
            )
            
            pins.append(pin)
            
        return pins
    def __init__(self, board: pcbnew.BOARD):
        self.board = board
        self.layer_map = self._build_layer_map()
        self.design_settings = board.GetDesignSettings()
        
        # Track export statistics
        self.stats = {
            'nets_exported': 0,
            'nets_filtered': 0,
            'pins_exported': 0,
            'obstacles_exported': 0,
            'tracks_existing': 0,
            'vias_existing': 0
        }
    
    def export_board(self, config: Dict) -> Dict:
        """
        Export complete board data for OrthoRoute GPU engine.
        
        Args:
            config: Export configuration from user dialog
            
        Returns:
            Complete board data in OrthoRoute JSON format
        """
        print("📤 Exporting KiCad board data...")
        
        # Get board bounds
        bounds = self._get_board_bounds()
        
        # Extract nets
        nets = self._extract_nets(config)
        
        # Extract obstacles
        obstacles = self._extract_obstacles(config)
        
        # Extract design rules
        design_rules = self._extract_design_rules()
        
        # Build complete board data
        board_data = {
            'version': '0.1.0',
            'source': 'KiCad',
            'timestamp': self._get_timestamp(),
            'bounds': bounds,
            'grid': {
                'pitch_mm': config.get('grid_pitch_mm', 0.1),
                'layers': config.get('max_layers', 4)
            },
            'config': {
                'max_iterations': config.get('max_iterations', 20),
                'batch_size': config.get('batch_size', 256),
                'verbose': config.get('verbose', False)
            },
            'nets': nets,
            'obstacles': obstacles,
            'design_rules': design_rules,
            'layer_stackup': self._get_layer_stackup(),
            'export_stats': self.stats
        }
        
        print(f"✅ Export complete: {len(nets)} nets, {len(obstacles)} obstacles")
        return board_data
    
    def _build_layer_map(self) -> Dict[int, int]:
        """Build mapping from KiCad layer IDs to sequential indices"""
        layer_map = {}
        copper_layers = []
        
        # Get all enabled copper layers
        for layer_id in range(pcbnew.PCB_LAYER_ID_COUNT):
            if (self.board.IsLayerEnabled(layer_id) and 
                pcbnew.IsCopperLayer(layer_id)):
                copper_layers.append(layer_id)
        
        # Sort layers (front to back)
        copper_layers.sort()
        
        # Map to sequential indices
        for idx, layer_id in enumerate(copper_layers):
            layer_map[layer_id] = idx
        
        return layer_map
    
    def _get_board_bounds(self) -> Dict:
        """Get board bounding box with margins"""
        bbox = self.board.GetBoardEdgesBoundingBox()
        
        # Add routing margin (2mm default)
        margin_nm = 2000000  # 2mm in nanometers
        
        bounds = {
            'min_x': int(bbox.GetLeft()) - margin_nm,
            'min_y': int(bbox.GetTop()) - margin_nm,
            'max_x': int(bbox.GetRight()) + margin_nm,
            'max_y': int(bbox.GetBottom()) + margin_nm
        }
        
        # Calculate board dimensions
        width_mm = (bounds['max_x'] - bounds['min_x']) / 1000000.0
        height_mm = (bounds['max_y'] - bounds['min_y']) / 1000000.0
        
        print(f"📏 Board bounds: {width_mm:.1f}mm × {height_mm:.1f}mm")
        
        return bounds
    
    def _extract_nets(self, config: Dict) -> List[Dict]:
        """Extract nets that need routing"""
        nets = []
        
        # Get filtering options
        net_pattern = config.get('net_pattern', '')
        skip_power = config.get('skip_power_nets', True)
        skip_routed = config.get('skip_routed_nets', True)
        min_pins = config.get('min_pins', 2)
        max_pins = config.get('max_pins', 100)
        
        print(f"🔍 Filtering nets with pattern: '{net_pattern}'")
        
        for net_id in range(1, self.board.GetNetCount()):  # Skip net 0 (no connection)
            net = self.board.FindNet(net_id)
            if not net:
                continue
            
            net_name = net.GetNetname()
            
            # Apply filters
            if net_pattern and net_pattern not in net_name:
                self.stats['nets_filtered'] += 1
                continue
            
            if skip_power and self._is_power_net(net_name):
                self.stats['nets_filtered'] += 1
                continue
            
            if skip_routed and self._is_net_routed(net):
                self.stats['nets_filtered'] += 1
                continue
            
            # Get all pads for this net
            pins = self._get_net_pins(net)
            
            if len(pins) < min_pins or len(pins) > max_pins:
                self.stats['nets_filtered'] += 1
                continue
            
            # Determine net properties
            net_class = self._classify_net(net_name)
            priority = self._get_net_priority(net_name, net_class)
            width_nm = self._get_net_width(net_name, net_class)
            via_size_nm = self._get_net_via_size(net_name, net_class)
            
            # Create net data
            net_data = {
                'id': net_id,
                'name': net_name,
                'pins': [self._pin_to_dict(pin) for pin in pins],
                'priority': priority,
                'width_nm': width_nm,
                'via_size_nm': via_size_nm,
                'net_class': net_class
            }
            
            nets.append(net_data)
            self.stats['nets_exported'] += 1
            self.stats['pins_exported'] += len(pins)
        
        # Sort by priority (high priority first)
        nets.sort(key=lambda n: n['priority'])
        
        print(f"📊 Net extraction: {len(nets)} nets exported, {self.stats['nets_filtered']} filtered")
        return nets
    
    def _get_net_pins(self, net: pcbnew.NETINFO_ITEM) -> List[ExportedPin]:
        """Get all pins for a net"""
        pins = []
        net_code = net.GetNetCode()
        
        # Iterate through all pads
        for footprint in self.board.GetFootprints():
            for pad in footprint.Pads():
                if pad.GetNetCode() == net_code:
                    pin = self._extract_pad_info(pad)
                    if pin:
                        pins.append(pin)
        
        return pins
    
    def _extract_pad_info(self, pad: pcbnew.PAD) -> Optional[ExportedPin]:
        """Extract information from a KiCad pad"""
        pos = pad.GetPosition()
        
        # Get pad layer
        pad_layers = []
        for layer_id in self.layer_map.keys():
            if pad.IsOnLayer(layer_id):
                pad_layers.append(self.layer_map[layer_id])
        
        if not pad_layers:
            return None  # Pad not on any routing layer
        
        # Use the first available layer (could be optimized)
        layer = pad_layers[0]
        
        # Determine pad type
        pad_type = "smd"
        if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
            pad_type = "through_hole"
        elif pad.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH:
            pad_type = "np_through_hole"
        
        # Get sizes
        size = pad.GetSize()
        drill_size = pad.GetDrillSize()
        
        return ExportedPin(
            x=int(pos.x),
            y=int(pos.y),
            layer=layer,
            size_x=int(size.x),
            size_y=int(size.y),
            drill_size=int(drill_size.x) if drill_size.x > 0 else 0,
            pad_type=pad_type
        )
    
    def _extract_obstacles(self, config: Dict) -> List[Dict]:
        """Extract obstacles and keepouts"""
        obstacles = []
        
        # Extract existing tracks
        if config.get('include_existing_tracks', True):
            obstacles.extend(self._extract_existing_tracks())
        
        # Extract component keepouts
        if config.get('include_component_keepouts', True):
            obstacles.extend(self._extract_component_keepouts())
        
        # Extract board edge keepouts
        if config.get('include_edge_keepouts', True):
            obstacles.extend(self._extract_edge_keepouts())
        
        # Extract user-defined keepouts
        if config.get('include_keepout_zones', True):
            obstacles.extend(self._extract_keepout_zones())
        
        self.stats['obstacles_exported'] = len(obstacles)
        return obstacles
    
    def _extract_existing_tracks(self) -> List[Dict]:
        """Extract existing tracks and vias as obstacles"""
        obstacles = []
        
        for track in self.board.GetTracks():
            if isinstance(track, pcbnew.PCB_TRACK):
                start = track.GetStart()
                end = track.GetEnd()
                width = track.GetWidth()
                layer_id = track.GetLayer()
                
                if layer_id in self.layer_map:
                    obstacles.append({
                        'type': 'track',
                        'x1': int(start.x - width//2),
                        'y1': int(start.y - width//2),
                        'x2': int(end.x + width//2),
                        'y2': int(end.y + width//2),
                        'layer': self.layer_map[layer_id],
                        'width': int(width)
                    })
                    self.stats['tracks_existing'] += 1
            
            elif isinstance(track, pcbnew.PCB_VIA):
                pos = track.GetPosition()
                size = track.GetWidth()
                
                obstacles.append({
                    'type': 'via',
                    'x1': int(pos.x - size//2),
                    'y1': int(pos.y - size//2),
                    'x2': int(pos.x + size//2),
                    'y2': int(pos.y + size//2),
                    'layer': -1,  # All layers
                    'width': int(size)
                })
                self.stats['vias_existing'] += 1
        
        return obstacles
    
    def _extract_component_keepouts(self) -> List[Dict]:
        """Extract component keepout areas"""
        obstacles = []
        keepout_margin = 200000  # 0.2mm margin around components
        
        for footprint in self.board.GetFootprints():
            bbox = footprint.GetBoundingBox()
            
            obstacles.append({
                'type': 'component',
                'x1': int(bbox.GetLeft()) - keepout_margin,
                'y1': int(bbox.GetTop()) - keepout_margin,
                'x2': int(bbox.GetRight()) + keepout_margin,
                'y2': int(bbox.GetBottom()) + keepout_margin,
                'layer': -1,  # All layers
                'width': 0
            })
        
        return obstacles
    
    def _extract_edge_keepouts(self) -> List[Dict]:
        """Extract board edge keepouts"""
        obstacles = []
        edge_clearance = 500000  # 0.5mm from board edge
        
        bbox = self.board.GetBoardEdgesBoundingBox()
        
        # Top edge
        obstacles.append({
            'type': 'board_edge',
            'x1': int(bbox.GetLeft()),
            'y1': int(bbox.GetTop()),
            'x2': int(bbox.GetRight()),
            'y2': int(bbox.GetTop()) + edge_clearance,
            'layer': -1,
            'width': 0
        })
        
        # Bottom edge
        obstacles.append({
            'type': 'board_edge',
            'x1': int(bbox.GetLeft()),
            'y1': int(bbox.GetBottom()) - edge_clearance,
            'x2': int(bbox.GetRight()),
            'y2': int(bbox.GetBottom()),
            'layer': -1,
            'width': 0
        })
        
        # Left edge
        obstacles.append({
            'type': 'board_edge',
            'x1': int(bbox.GetLeft()),
            'y1': int(bbox.GetTop()),
            'x2': int(bbox.GetLeft()) + edge_clearance,
            'y2': int(bbox.GetBottom()),
            'layer': -1,
            'width': 0
        })
        
        # Right edge
        obstacles.append({
            'type': 'board_edge',
            'x1': int(bbox.GetRight()) - edge_clearance,
            'y1': int(bbox.GetTop()),
            'x2': int(bbox.GetRight()),
            'y2': int(bbox.GetBottom()),
            'layer': -1,
            'width': 0
        })
        
        return obstacles
    
    def _extract_keepout_zones(self) -> List[Dict]:
        """Extract user-defined keepout zones"""
        obstacles = []
        
        # KiCad 7.0+ zone handling
        for zone in self.board.Zones():
            if zone.GetIsRuleArea() and zone.GetDoNotAllowTracks():
                layer_id = zone.GetLayer()
                
                if layer_id in self.layer_map:
                    bbox = zone.GetBoundingBox()
                    
                    obstacles.append({
                        'type': 'keepout',
                        'x1': int(bbox.GetLeft()),
                        'y1': int(bbox.GetTop()),
                        'x2': int(bbox.GetRight()),
                        'y2': int(bbox.GetBottom()),
                        'layer': self.layer_map[layer_id],
                        'width': 0
                    })
        
        return obstacles
    
    def _extract_design_rules(self) -> Dict:
        """Extract design rules from KiCad"""
        ds = self.design_settings
        
        rules = {
            'min_track_width': int(ds.m_TrackMinWidth),
            'min_via_size': int(ds.m_ViasMinSize),
            'min_via_drill': int(ds.m_ViasMinDrill),
            'min_clearance': int(ds.GetSmallestClearanceValue()),
            'copper_layers': self.board.GetCopperLayerCount(),
            'board_thickness': 1600000,  # Default 1.6mm (could be extracted from stackup)
        }
        
        # Via size rules
        rules['via_rules'] = []
        if hasattr(ds, 'm_ViasDimensionsList'):
            for via_size in ds.m_ViasDimensionsList:
                rules['via_rules'].append({
                    'diameter': int(via_size.m_Diameter),
                    'drill': int(via_size.m_Drill)
                })
        
        # Track width rules  
        rules['track_rules'] = []
        if hasattr(ds, 'm_TrackWidthList'):
            for width in ds.m_TrackWidthList:
                rules['track_rules'].append({
                    'width': int(width)
                })
        
        return rules
    
    def _get_layer_stackup(self) -> List[Dict]:
        """Get layer stackup information"""
        stackup = []
        
        for layer_id, layer_idx in self.layer_map.items():
            layer_name = self.board.GetLayerName(layer_id)
            
            stackup.append({
                'index': layer_idx,
                'kicad_id': layer_id,
                'name': layer_name,
                'type': 'signal',  # Could be extracted from stackup
                'thickness': 35000  # Default 35µm copper
            })
        
        return sorted(stackup, key=lambda x: x['index'])
    
    def _is_power_net(self, net_name: str) -> bool:
        """Check if net is a power/ground net"""
        power_keywords = [
            'VCC', 'VDD', 'VIN', 'VOUT', 'VREF', 'VPP', 'VPWR',
            'GND', 'VSS', 'GNDA', 'GNDD', 'AGND', 'DGND', 'PGND',
            '+3V3', '+5V', '+12V', '-12V', '+24V', '+1V8', '+2V5',
            'PWR', 'POWER', 'SUPPLY'
        ]
        
        net_upper = net_name.upper()
        return any(keyword in net_upper for keyword in power_keywords)
    
    def _is_net_routed(self, net: pcbnew.NETINFO_ITEM) -> bool:
        """Check if net already has routing"""
        net_code = net.GetNetCode()
        
        # Check for existing tracks
        for track in self.board.GetTracks():
            if isinstance(track, pcbnew.PCB_TRACK) and track.GetNetCode() == net_code:
                return True
        
        return False
    
    def _classify_net(self, net_name: str) -> str:
        """Classify net into categories"""
        net_upper = net_name.upper()
        
        if self._is_power_net(net_name):
            return 'power'
        elif any(keyword in net_upper for keyword in ['CLK', 'CLOCK']):
            return 'clock'
        elif any(keyword in net_upper for keyword in ['USB', 'PCIE', 'HDMI', 'LVDS']):
            return 'high_speed'
        elif any(suffix in net_name for suffix in ['_P', '_N', '+', '-']):
            return 'differential'
        else:
            return 'default'
    
    def _get_net_priority(self, net_name: str, net_class: str) -> int:
        """Get routing priority for net"""
        priority_map = {
            'clock': 0,      # Highest priority
            'power': 1,
            'high_speed': 2,
            'differential': 2,
            'default': 5     # Normal priority
        }
        
        return priority_map.get(net_class, 5)
    
    def _get_net_width(self, net_name: str, net_class: str) -> int:
        """Get track width for net"""
        width_map = {
            'power': 500000,      # 0.5mm
            'clock': 150000,      # 0.15mm
            'high_speed': 100000, # 0.1mm
            'differential': 100000, # 0.1mm
            'default': 200000     # 0.2mm
        }
        
        return width_map.get(net_class, 200000)
    
    def _get_net_via_size(self, net_name: str, net_class: str) -> int:
        """Get via size for net"""
        via_map = {
            'power': 400000,      # 0.4mm
            'clock': 200000,      # 0.2mm
            'high_speed': 200000, # 0.2mm
            'differential': 200000, # 0.2mm
            'default': 200000     # 0.2mm
        }
        
        return via_map.get(net_class, 200000)
    
    def _pin_to_dict(self, pin: ExportedPin) -> Dict:
        """Convert ExportedPin to dictionary"""
        return {
            'x': pin.x,
            'y': pin.y,
            'layer': pin.layer,
            'size_x': pin.size_x,
            'size_y': pin.size_y,
            'drill_size': pin.drill_size,
            'type': pin.pad_type
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()

def export_board_to_file(board: pcbnew.BOARD, filename: str, config: Dict = None) -> bool:
    """
    Export board to JSON file.
    
    Args:
        board: KiCad board object
        filename: Output JSON filename
        config: Export configuration
        
    Returns:
        True if export successful
    """
    try:
        exporter = BoardExporter(board)
        
        if config is None:
            config = {
                'grid_pitch_mm': 0.1,
                'max_layers': 4,
                'skip_power_nets': False,
                'skip_routed_nets': True
            }
        
        board_data = exporter.export_board(config)
        
        with open(filename, 'w') as f:
            json.dump(board_data, f, indent=2)
        
        print(f"✅ Board exported to: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def get_export_stats(board: pcbnew.BOARD) -> Dict:
    """Get statistics about what would be exported"""
    exporter = BoardExporter(board)
    
    # Quick analysis without full export
    total_nets = board.GetNetCount() - 1  # Exclude net 0
    routed_nets = 0
    power_nets = 0
    
    for net_id in range(1, board.GetNetCount()):
        net = board.FindNet(net_id)
        if net:
            net_name = net.GetNetname()
            if exporter._is_power_net(net_name):
                power_nets += 1
            if exporter._is_net_routed(net):
                routed_nets += 1
    
    return {
        'total_nets': total_nets,
        'routed_nets': routed_nets,
        'power_nets': power_nets,
        'signal_nets': total_nets - power_nets,
        'unrouted_nets': total_nets - routed_nets,
        'copper_layers': board.GetCopperLayerCount(),
        'total_tracks': len([t for t in board.GetTracks() if isinstance(t, pcbnew.PCB_TRACK)]),
        'total_vias': len([t for t in board.GetTracks() if isinstance(t, pcbnew.PCB_VIA)])
    }