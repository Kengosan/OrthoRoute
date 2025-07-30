"""
OrthoRoute Board Data Exporter
Handles extraction and processing of KiCad board data for routing.
"""

import pcbnew
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class BoardBounds:
    """Board boundary information"""
    x_min_nm: int
    y_min_nm: int
    x_max_nm: int
    y_max_nm: int
    width_nm: int
    height_nm: int
    layers: int

@dataclass
class NetPin:
    """Pin information for routing"""
    x_nm: int
    y_nm: int
    layer: int
    pad_name: str
    footprint_name: str
    net_code: int
    drill_nm: int = 0
    size_nm: int = 0

@dataclass
class RoutingNet:
    """Net information for routing"""
    id: int
    name: str
    pins: List[NetPin]
    width_nm: int
    priority: int = 5
    routed: bool = False

@dataclass
class BoardObstacle:
    """Obstacle on the board"""
    x_nm: int
    y_nm: int
    width_nm: int
    height_nm: int
    layer: int
    obstacle_type: str  # 'track', 'via', 'pad', 'zone', 'keepout'

class BoardDataExporter:
    """Exports comprehensive board data for routing"""
    
    def __init__(self):
        self.obstacles = []
        self.nets = []
        self.bounds = None
        self.design_rules = {}
        
    def export_board_data(self, board, config: Dict = None, progress_callback=None) -> Dict:
        """Export comprehensive board data"""
        print("🔍 BoardDataExporter: Starting board analysis...")
        
        try:
            # Update progress
            if progress_callback:
                progress_callback("Analyzing board geometry...")
            
            # Extract board bounds
            self.bounds = self._extract_board_bounds(board)
            print(f"📐 Board: {self.bounds.width_nm/1000000:.1f}mm × {self.bounds.height_nm/1000000:.1f}mm, {self.bounds.layers} layers")
            
            # Extract design rules
            if progress_callback:
                progress_callback("Extracting design rules...")
            self.design_rules = self._extract_design_rules(board)
            print(f"📏 Design rules: {len(self.design_rules)} rules extracted")
            
            # Extract obstacles
            if progress_callback:
                progress_callback("Mapping board obstacles...")
            self.obstacles = self._extract_obstacles(board)
            print(f"🚧 Obstacles: {len(self.obstacles)} obstacles mapped")
            
            # Extract nets for routing
            if progress_callback:
                progress_callback("Analyzing nets and connectivity...")
            self.nets = self._extract_routing_nets(board)
            unrouted_nets = [n for n in self.nets if len(n.pins) > 1]
            print(f"🔗 Nets: {len(unrouted_nets)} nets need routing")
            
            # Calculate optimal grid parameters
            if progress_callback:
                progress_callback("Calculating routing grid...")
            grid_config = self._calculate_grid_parameters(config or {})
            print(f"📊 Grid: {grid_config['width']}×{grid_config['height']}×{grid_config['layers']} @ {grid_config['pitch_nm']}nm")
            
            # Build complete board data structure
            board_data = {
                'bounds': {
                    'x_min_nm': self.bounds.x_min_nm,
                    'y_min_nm': self.bounds.y_min_nm,
                    'x_max_nm': self.bounds.x_max_nm,
                    'y_max_nm': self.bounds.y_max_nm,
                    'width_nm': self.bounds.width_nm,
                    'height_nm': self.bounds.height_nm,
                    'layers': self.bounds.layers
                },
                'grid': grid_config,
                'design_rules': self.design_rules,
                'obstacles': [self._obstacle_to_dict(obs) for obs in self.obstacles],
                'nets': [self._net_to_dict(net) for net in unrouted_nets],
                'board_info': {
                    'total_obstacles': len(self.obstacles),
                    'routable_nets': len(unrouted_nets),
                    'board_area_mm2': (self.bounds.width_nm * self.bounds.height_nm) / (1000000 * 1000000)
                }
            }
            
            print(f"✅ Board data export complete: {len(board_data['nets'])} nets, {len(board_data['obstacles'])} obstacles")
            return board_data
            
        except Exception as e:
            print(f"❌ Board export error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_board_bounds(self, board) -> BoardBounds:
        """Extract board boundary information"""
        bbox = board.GetBoundingBox()
        layers = board.GetCopperLayerCount()
        
        return BoardBounds(
            x_min_nm=int(bbox.GetX()),
            y_min_nm=int(bbox.GetY()),
            x_max_nm=int(bbox.GetX() + bbox.GetWidth()),
            y_max_nm=int(bbox.GetY() + bbox.GetHeight()),
            width_nm=int(bbox.GetWidth()),
            height_nm=int(bbox.GetHeight()),
            layers=layers
        )
    
    def _extract_design_rules(self, board) -> Dict:
        """Extract design rules from board"""
        rules = {}
        
        try:
            design_settings = board.GetDesignSettings()
            rules.update({
                'track_width_nm': design_settings.GetCurrentTrackWidth(),
                'via_size_nm': design_settings.GetCurrentViaSize(),
                'via_drill_nm': design_settings.GetCurrentViaDrill(),
                'min_track_width_nm': design_settings.m_TrackMinWidth,
                'min_via_size_nm': design_settings.m_ViasMinSize,
                'min_spacing_nm': design_settings.m_TrackMinWidth  # Approximate
            })
        except Exception as e:
            print(f"⚠️ Could not extract all design rules: {e}")
            # Fallback values
            rules.update({
                'track_width_nm': 200000,  # 0.2mm
                'via_size_nm': 400000,     # 0.4mm
                'via_drill_nm': 200000,    # 0.2mm
                'min_track_width_nm': 100000,  # 0.1mm
                'min_via_size_nm': 300000,     # 0.3mm
                'min_spacing_nm': 150000       # 0.15mm
            })
        
        return rules
    
    def _extract_obstacles(self, board) -> List[BoardObstacle]:
        """Extract all obstacles on the board"""
        obstacles = []
        
        # Existing tracks
        for track in board.GetTracks():
            if track.GetClass() == "PCB_TRACK":
                obstacles.append(BoardObstacle(
                    x_nm=int(track.GetStart().x),
                    y_nm=int(track.GetStart().y),
                    width_nm=int(track.GetWidth()),
                    height_nm=int((track.GetEnd() - track.GetStart()).EuclideanNorm()),
                    layer=track.GetLayer(),
                    obstacle_type='track'
                ))
            elif track.GetClass() == "PCB_VIA":
                obstacles.append(BoardObstacle(
                    x_nm=int(track.GetPosition().x),
                    y_nm=int(track.GetPosition().y),
                    width_nm=int(track.GetWidth()),
                    height_nm=int(track.GetWidth()),
                    layer=-1,  # Via spans all layers
                    obstacle_type='via'
                ))
        
        # Pads
        for footprint in board.GetFootprints():
            for pad in footprint.Pads():
                pos = pad.GetPosition()
                size = pad.GetSize()
                obstacles.append(BoardObstacle(
                    x_nm=int(pos.x),
                    y_nm=int(pos.y),
                    width_nm=int(size.x),
                    height_nm=int(size.y),
                    layer=pad.GetLayer() if pad.GetAttribute() != pcbnew.PAD_ATTRIB_PTH else -1,
                    obstacle_type='pad'
                ))
        
        # Zones
        for zone in board.Zones():
            bbox = zone.GetBoundingBox()
            obstacles.append(BoardObstacle(
                x_nm=int(bbox.GetX()),
                y_nm=int(bbox.GetY()),
                width_nm=int(bbox.GetWidth()),
                height_nm=int(bbox.GetHeight()),
                layer=zone.GetLayer(),
                obstacle_type='zone'
            ))
        
        return obstacles
    
    def _extract_routing_nets(self, board) -> List[RoutingNet]:
        """Extract nets that need routing"""
        nets = []
        net_info = board.GetNetInfo()
        
        for net_code in range(1, net_info.GetNetCount()):
            net_info_item = net_info.GetNetItem(net_code)
            if not net_info_item:
                continue
            
            net_name = net_info_item.GetNetname()
            pins = self._extract_pins_for_net(board, net_code)
            
            if len(pins) >= 2:  # Only nets with multiple pins need routing
                nets.append(RoutingNet(
                    id=net_code,
                    name=net_name,
                    pins=pins,
                    width_nm=self.design_rules.get('track_width_nm', 200000),
                    priority=self._calculate_net_priority(net_name, pins)
                ))
        
        return nets
    
    def _extract_pins_for_net(self, board, net_code: int) -> List[NetPin]:
        """Extract pins for a specific net"""
        pins = []
        
        for footprint in board.GetFootprints():
            for pad in footprint.Pads():
                if pad.GetNetCode() == net_code:
                    pos = pad.GetPosition()
                    size = pad.GetSize()
                    
                    # Determine layer
                    if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                        layer = 0  # Through-hole, start at top layer
                    else:
                        layer = 0 if pad.GetLayer() == pcbnew.F_Cu else board.GetCopperLayerCount() - 1
                    
                    pins.append(NetPin(
                        x_nm=int(pos.x),
                        y_nm=int(pos.y),
                        layer=layer,
                        pad_name=pad.GetName(),
                        footprint_name=footprint.GetReference(),
                        net_code=net_code,
                        drill_nm=int(pad.GetDrillSize().x) if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH else 0,
                        size_nm=max(int(size.x), int(size.y))
                    ))
        
        return pins
    
    def _calculate_net_priority(self, net_name: str, pins: List[NetPin]) -> int:
        """Calculate routing priority for a net"""
        # Higher priority for power/ground nets
        if any(keyword in net_name.upper() for keyword in ['VCC', 'VDD', 'GND', 'POWER']):
            return 1
        # Medium priority for clock signals
        elif any(keyword in net_name.upper() for keyword in ['CLK', 'CLOCK', 'OSC']):
            return 3
        # Lower priority for many-pin nets (likely buses)
        elif len(pins) > 10:
            return 7
        else:
            return 5
    
    def _calculate_grid_parameters(self, config: Dict) -> Dict:
        """Calculate optimal grid parameters"""
        # Base pitch from config or design rules
        base_pitch = config.get('grid_size', self.design_rules.get('min_track_width_nm', 100000))
        
        # Adjust based on board complexity
        obstacle_density = len(self.obstacles) / ((self.bounds.width_nm * self.bounds.height_nm) / (1000000 * 1000000))
        
        if obstacle_density > 100:  # High density
            pitch_nm = max(base_pitch // 2, 50000)  # Finer grid
        elif obstacle_density < 10:  # Low density
            pitch_nm = min(base_pitch * 2, 500000)  # Coarser grid
        else:
            pitch_nm = base_pitch
        
        # Calculate grid dimensions
        width = max(int(self.bounds.width_nm / pitch_nm) + 20, 100)
        height = max(int(self.bounds.height_nm / pitch_nm) + 20, 100)
        
        return {
            'width': width,
            'height': height,
            'layers': self.bounds.layers,
            'pitch_nm': pitch_nm,
            'origin_x_nm': self.bounds.x_min_nm,
            'origin_y_nm': self.bounds.y_min_nm
        }
    
    def _obstacle_to_dict(self, obstacle: BoardObstacle) -> Dict:
        """Convert obstacle to dictionary"""
        return {
            'x_nm': obstacle.x_nm,
            'y_nm': obstacle.y_nm,
            'width_nm': obstacle.width_nm,
            'height_nm': obstacle.height_nm,
            'layer': obstacle.layer,
            'type': obstacle.obstacle_type
        }
    
    def _net_to_dict(self, net: RoutingNet) -> Dict:
        """Convert net to dictionary"""
        return {
            'id': net.id,
            'name': net.name,
            'width_nm': net.width_nm,
            'priority': net.priority,
            'pins': [
                {
                    'x': pin.x_nm,
                    'y': pin.y_nm,
                    'layer': pin.layer,
                    'pad_name': pin.pad_name,
                    'footprint': pin.footprint_name,
                    'size_nm': pin.size_nm,
                    'drill_nm': pin.drill_nm
                }
                for pin in net.pins
            ]
        }
