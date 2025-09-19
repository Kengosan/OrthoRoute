"""KiCad SWIG API adapter (fallback)."""
import logging
from typing import Dict, List, Optional, Any

from ...domain.models.board import Board, Component, Net, Pad, Layer, Coordinate
from ...domain.models.constraints import DRCConstraints, NetClass

logger = logging.getLogger(__name__)


class KiCadSWIGAdapter:
    """Adapter for legacy KiCad SWIG API integration."""
    
    def __init__(self):
        """Initialize SWIG adapter."""
        self.board = None
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to KiCad via SWIG API."""
        try:
            # Import SWIG-based KiCad Python API
            import pcbnew
            
            # Get the current board from KiCad
            self.board = pcbnew.GetBoard()
            
            if self.board is None:
                logger.warning("No board loaded in KiCad")
                return False
            
            self.is_connected = True
            logger.info("Connected to KiCad via SWIG API")
            return True
            
        except ImportError:
            logger.warning("KiCad SWIG API (pcbnew) not available")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to KiCad SWIG API: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from KiCad."""
        self.board = None
        self.is_connected = False
        logger.info("Disconnected from KiCad SWIG API")
    
    def is_available(self) -> bool:
        """Check if KiCad SWIG API is available."""
        try:
            import pcbnew
            return True
        except ImportError:
            return False
    
    def load_board(self) -> Optional[Board]:
        """Load board from KiCad via SWIG API."""
        if not self.is_connected or not self.board:
            logger.error("Not connected to KiCad or no board loaded")
            return None
        
        try:
            import pcbnew
            
            # Create board object
            board_name = self.board.GetTitleBlock().GetTitle() or "KiCad Board"
            
            board = Board(
                id='kicad_swig_board',
                name=board_name,
                thickness=self.board.GetDesignSettings().GetBoardThickness() / 1000000.0,  # Convert to mm
                layer_count=self.board.GetCopperLayerCount()
            )
            
            # Add layers
            layer_names = []
            for layer_id in range(pcbnew.PCB_LAYER_ID_COUNT):
                if self.board.IsLayerEnabled(layer_id):
                    layer_name = self.board.GetLayerName(layer_id)
                    layer_type = 'copper' if layer_name.endswith('.Cu') else 'other'
                    
                    layer = Layer(
                        name=layer_name,
                        type=layer_type,
                        stackup_position=layer_id
                    )
                    board.add_layer(layer)
                    layer_names.append(layer_name)
            
            # Add components (footprints in KiCad terminology)
            for footprint in self.board.GetFootprints():
                component = self._create_component_from_footprint(footprint)
                board.add_component(component)
            
            # Add nets
            netlist = self.board.GetNetInfo()
            for net_code in range(netlist.GetNetCount()):
                net_info = netlist.GetNetItem(net_code)
                if net_info and net_info.GetNetname():
                    net = self._create_net_from_netinfo(net_info, board)
                    if net:
                        board.add_net(net)
            
            logger.info(f"Loaded board via SWIG: {len(board.components)} components, {len(board.nets)} nets")
            return board
            
        except Exception as e:
            logger.error(f"Error loading board via SWIG API: {e}")
            return None
    
    def get_drc_constraints(self) -> Optional[DRCConstraints]:
        """Extract DRC constraints from KiCad via SWIG API."""
        if not self.is_connected or not self.board:
            return None
        
        try:
            import pcbnew
            
            design_settings = self.board.GetDesignSettings()
            
            # Convert from internal units (nanometers) to mm
            def to_mm(value):
                return value / 1000000.0
            
            constraints = DRCConstraints(
                min_track_width=to_mm(design_settings.m_TrackMinWidth),
                min_via_diameter=to_mm(design_settings.m_ViasMinSize),
                min_via_drill=to_mm(design_settings.m_ViasMinDrill),
                default_track_width=to_mm(design_settings.GetCurrentTrackWidth()),
                default_via_diameter=to_mm(design_settings.GetCurrentViaSize()),
                default_via_drill=to_mm(design_settings.GetCurrentViaDrill())
            )
            
            # Extract netclasses
            netclasses = design_settings.GetNetClasses()
            default_netclass = netclasses.GetDefault()
            
            # Add default netclass
            if default_netclass:
                default_nc = NetClass(
                    name="Default",
                    track_width=to_mm(default_netclass.GetTrackWidth()),
                    via_diameter=to_mm(default_netclass.GetViaDiameter()),
                    via_drill=to_mm(default_netclass.GetViaDrill()),
                    clearance=to_mm(default_netclass.GetClearance())
                )
                constraints.add_netclass(default_nc)
            
            # Add custom netclasses
            for nc_name in netclasses.NetClasses():
                netclass = netclasses.Find(nc_name)
                if netclass:
                    custom_nc = NetClass(
                        name=nc_name,
                        track_width=to_mm(netclass.GetTrackWidth()),
                        via_diameter=to_mm(netclass.GetViaDiameter()),
                        via_drill=to_mm(netclass.GetViaDrill()),
                        clearance=to_mm(netclass.GetClearance())
                    )
                    constraints.add_netclass(custom_nc)
            
            logger.info(f"Extracted DRC constraints via SWIG with {len(constraints.netclasses)} netclasses")
            return constraints
            
        except Exception as e:
            logger.error(f"Error extracting DRC constraints via SWIG: {e}")
            return None
    
    def create_track(self, start_x: float, start_y: float, end_x: float, end_y: float,
                    layer: str, width: float, net_name: str) -> bool:
        """Create a track in KiCad via SWIG API."""
        if not self.is_connected or not self.board:
            return False
        
        try:
            import pcbnew
            
            # Convert mm to internal units (nanometers)
            def to_internal(value_mm):
                return int(value_mm * 1000000)
            
            # Find the layer ID
            layer_id = None
            for lid in range(pcbnew.PCB_LAYER_ID_COUNT):
                if self.board.GetLayerName(lid) == layer:
                    layer_id = lid
                    break
            
            if layer_id is None:
                logger.error(f"Layer {layer} not found")
                return False
            
            # Find the net
            netinfo = self.board.FindNet(net_name)
            if not netinfo:
                logger.error(f"Net {net_name} not found")
                return False
            
            # Quantization assert: Check for collapse after conversion to KiCad internal units
            x0i, y0i = to_internal(start_x), to_internal(start_y)
            x1i, y1i = to_internal(end_x), to_internal(end_y)
            if x0i == x1i and y0i == y1i:
                logger.error("[GUI-QUANT] collapse after quantization: (%.6f,%.6f)->(%.6f,%.6f) mm",
                           start_x, start_y, end_x, end_y)

            # Create track
            track = pcbnew.PCB_TRACK(self.board)
            track.SetStart(pcbnew.VECTOR2I(x0i, y0i))
            track.SetEnd(pcbnew.VECTOR2I(x1i, y1i))
            track.SetWidth(to_internal(width))
            track.SetLayer(layer_id)
            track.SetNet(netinfo)
            
            self.board.Add(track)
            logger.debug(f"Created track on {layer} for net {net_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating track via SWIG: {e}")
            return False
    
    def create_via(self, x: float, y: float, diameter: float, drill: float,
                  from_layer: str, to_layer: str, net_name: str) -> bool:
        """Create a via in KiCad via SWIG API."""
        if not self.is_connected or not self.board:
            return False
        
        try:
            import pcbnew
            
            # Convert mm to internal units
            def to_internal(value_mm):
                return int(value_mm * 1000000)
            
            # Find layer IDs
            from_layer_id = None
            to_layer_id = None
            
            for lid in range(pcbnew.PCB_LAYER_ID_COUNT):
                layer_name = self.board.GetLayerName(lid)
                if layer_name == from_layer:
                    from_layer_id = lid
                elif layer_name == to_layer:
                    to_layer_id = lid
            
            if from_layer_id is None or to_layer_id is None:
                logger.error(f"Layer not found: {from_layer} or {to_layer}")
                return False
            
            # Find the net
            netinfo = self.board.FindNet(net_name)
            if not netinfo:
                logger.error(f"Net {net_name} not found")
                return False
            
            # Create via
            via = pcbnew.PCB_VIA(self.board)
            via.SetPosition(pcbnew.VECTOR2I(to_internal(x), to_internal(y)))
            via.SetWidth(to_internal(diameter))
            via.SetDrill(to_internal(drill))
            via.SetNet(netinfo)
            
            # Set layer pair for blind/buried vias
            via.SetLayerPair(from_layer_id, to_layer_id)
            
            self.board.Add(via)
            logger.debug(f"Created via from {from_layer} to {to_layer} for net {net_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating via via SWIG: {e}")
            return False
    
    def _create_component_from_footprint(self, footprint) -> Component:
        """Create component from KiCad footprint."""
        import pcbnew
        
        # Convert internal units to mm
        def to_mm(value):
            return value / 1000000.0
        
        position = footprint.GetPosition()
        
        component = Component(
            id=footprint.GetFPID().GetLibItemName().ToString(),
            reference=footprint.GetReference(),
            value=footprint.GetValue(),
            footprint=footprint.GetFPID().Format(),
            position=Coordinate(to_mm(position.x), to_mm(position.y)),
            angle=footprint.GetOrientationDegrees(),
            layer=self.board.GetLayerName(footprint.GetLayer())
        )
        
        # Add pads
        for pad in footprint.Pads():
            pad_pos = pad.GetPosition()
            pad_size = pad.GetSize()
            
            net_info = pad.GetNet()
            net_id = str(net_info.GetNetCode()) if net_info else None
            
            component_pad = Pad(
                id=f"{component.reference}_{pad.GetNumber()}",
                component_id=component.id,
                net_id=net_id,
                position=Coordinate(to_mm(pad_pos.x), to_mm(pad_pos.y)),
                size=(to_mm(pad_size.x), to_mm(pad_size.y)),
                drill_size=to_mm(pad.GetDrillSize().x) if pad.GetDrillSize().x > 0 else None,
                layer=self.board.GetLayerName(pad.GetPrincipalLayer()),
                shape=pad.GetShape().name if hasattr(pad.GetShape(), 'name') else 'circle',
                angle=pad.GetOrientationDegrees()
            )
            component.pads.append(component_pad)
        
        return component
    
    def _create_net_from_netinfo(self, net_info, board: Board) -> Optional[Net]:
        """Create net from KiCad netinfo."""
        net_name = net_info.GetNetname()
        net_code = net_info.GetNetCode()
        
        if not net_name or net_code == 0:  # Skip unconnected nets
            return None
        
        # Find pads for this net
        net_pads = []
        for component in board.components:
            for pad in component.pads:
                if pad.net_id == str(net_code):
                    net_pads.append(pad)
        
        if not net_pads:
            return None
        
        # Get netclass
        netclass_name = "Default"
        if hasattr(net_info, 'GetNetClass'):
            netclass = net_info.GetNetClass()
            if netclass:
                netclass_name = netclass.GetName()
        
        net = Net(
            id=str(net_code),
            name=net_name,
            netclass=netclass_name,
            pads=net_pads
        )
        
        return net