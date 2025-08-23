#!/usr/bin/env python3
"""
KiCad IPC Interface - Handles communication with KiCad via IPC API (kipy)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import sys
import time

logger = logging.getLogger(__name__)

@dataclass
class DRCRules:
    """Container for Design Rule Check information"""
    netclasses: Dict[str, Dict]  # netclass_name -> rules dict
    default_track_width: float  # mm
    default_via_size: float  # mm
    default_via_drill: float  # mm
    default_clearance: float  # mm
    minimum_track_width: float  # mm
    minimum_via_size: float  # mm

@dataclass
class BoardData:
    """Container for board data extracted from KiCad"""
    filename: str
    width: float  # mm
    height: float  # mm
    layers: int
    nets: List[Dict]
    components: List[Dict]
    tracks: List[Dict]
    vias: List[Dict]
    pads: List[Dict]
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    drc_rules: Optional[DRCRules] = None


def _ipc_retry(func, desc: str, max_retries: int = 3, sleep_s: float = 0.5):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            msg = str(e)
            last_err = e
            logger.warning(f"IPC '{desc}' failed (attempt {attempt}/{max_retries}): {msg}")
            if "Timed out" in msg or "AS_BUSY" in msg or "busy" in msg.lower():
                time.sleep(sleep_s)
                continue
            break
    if last_err:
        raise last_err


class KiCadInterface:
    """Interface to KiCad via IPC API (kicad-python -> kipy)"""

    def __init__(self):
        self.client = None
        self.board = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to KiCad via IPC API"""
        try:
            # Ensure kipy is importable from user site (common when KiCad launches process)
            try:
                from kipy import KiCad  # type: ignore
            except ImportError:
                import site
                user_site = site.getusersitepackages()
                if user_site and user_site not in sys.path:
                    sys.path.insert(0, user_site)
                from kipy import KiCad  # retry

            # Gather credentials if provided by KiCad runtime
            api_socket = os.environ.get('KICAD_API_SOCKET')
            api_token = os.environ.get('KICAD_API_TOKEN')
            timeout_ms = 25000
            if api_socket or api_token:
                self.client = KiCad(socket_path=api_socket, kicad_token=api_token, timeout_ms=timeout_ms)
            else:
                self.client = KiCad(timeout_ms=timeout_ms)

            # Get board to confirm connection - try different methods
            try:
                # Method 1: Try get_board directly
                self.board = _ipc_retry(self.client.get_board, "get_board", max_retries=2, sleep_s=0.5)
            except Exception as e1:
                logger.warning(f"get_board failed: {e1}")
                try:
                    # Method 2: Try getting open documents first
                    docs = self.client.get_open_documents()
                    if docs and len(docs) > 0:
                        # Use first open document
                        self.board = docs[0]
                        logger.info(f"✅ Retrieved board from open documents: {getattr(self.board, 'name', 'Unknown')}")
                    else:
                        raise Exception("No open documents found")
                except Exception as e2:
                    logger.warning(f"get_open_documents failed: {e2}")
                    # Method 3: Try direct board access
                    self.board = self.client.board
                    if not self.board:
                        raise Exception("No board available through any method")
                    
            self.connected = True
            logger.info("Connected to KiCad IPC API and retrieved board")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to KiCad: {e}")
            self.connected = False
            return False

    def get_board_filename(self) -> str:
        """Get the current board filename using KiCad Python API"""
        if not self.connected or not self.board:
            logger.warning("Not connected to KiCad board")
            return "Unknown"
        
        board = self.board
        try:
            # Try multiple methods to get the board filename
            if hasattr(board, 'GetFileName') and board.GetFileName():
                # Use KiCad Python API GetFileName method (preferred)
                filename = board.GetFileName()
            elif hasattr(board, 'filename') and board.filename:
                filename = board.filename
            elif hasattr(board, 'name') and board.name:
                filename = board.name
            elif hasattr(board, '_board') and hasattr(board._board, 'GetFileName'):
                filename = board._board.GetFileName()
            elif hasattr(board, 'board') and hasattr(board.board, 'GetFileName'):
                filename = board.board.GetFileName()
            elif hasattr(board, 'document') and board.document:
                if hasattr(board.document, 'filename') and board.document.filename:
                    filename = board.document.filename
                elif hasattr(board.document, 'name') and board.document.name:
                    filename = board.document.name
                else:
                    filename = "Unknown"
            else:
                filename = "Unknown"
            
            # If we got a full path, extract just the filename
            if filename and filename != "Unknown" and ('\\' in filename or '/' in filename):
                import os
                filename = os.path.basename(filename)
                
            logger.info(f"Retrieved board filename: {filename}")
            return filename
            
        except Exception as e:
            logger.warning(f"Error getting board filename: {e}")
            return "Unknown"

    def get_board_data(self) -> Dict:
        """Extract comprehensive board data from KiCad"""
        if not self.connected or not self.board:
            logger.error("Not connected to KiCad")
            return self._get_fallback_board_data()

        board = self.board

        # File/name - Enhanced extraction using KiCad Python API
        filename = 'Untitled'
        try:
            # Try multiple methods to get the board filename
            if hasattr(board, 'name') and board.name:
                filename = board.name
            elif hasattr(board, 'filename') and board.filename:
                filename = board.filename
            elif hasattr(board, 'GetFileName') and board.GetFileName():
                # Use KiCad Python API GetFileName method
                filename = board.GetFileName()
            elif hasattr(board, 'document') and board.document:
                if hasattr(board.document, 'name') and board.document.name:
                    filename = board.document.name
                elif hasattr(board.document, 'filename') and board.document.filename:
                    filename = board.document.filename
            # Try accessing underlying board object
            elif hasattr(board, '_board') and hasattr(board._board, 'GetFileName'):
                filename = board._board.GetFileName()
            elif hasattr(board, 'board') and hasattr(board.board, 'GetFileName'):
                filename = board.board.GetFileName()
            
            # If we got a full path, extract just the filename
            if filename and ('\\' in filename or '/' in filename):
                import os
                filename = os.path.basename(filename)
                
            logger.info(f"Board filename: {filename}")
        except Exception as e:
            logger.warning(f"Could not get board filename: {e}")
            filename = 'Untitled'

        # Components
        components = []
        try:
            fps = _ipc_retry(board.get_footprints, "get_footprints", max_retries=3, sleep_s=0.7)
            for i, fp in enumerate(fps):
                try:
                    pos = getattr(fp, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    y = float(getattr(pos, 'y', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    ref = None
                    try:
                        ref = getattr(getattr(getattr(fp, 'reference_field', None), 'text', None), 'value', None)
                    except Exception:
                        pass
                    val = None
                    try:
                        val = getattr(getattr(getattr(fp, 'value_field', None), 'text', None), 'value', None)
                    except Exception:
                        pass
                    rot = getattr(getattr(fp, 'orientation', None), 'degrees', 0.0)
                    layer = getattr(fp, 'layer', 'F.Cu')
                    components.append({
                        'reference': ref or f'U{i}',
                        'value': val or '',
                        'x': x,
                        'y': y,
                        'rotation': float(rot) if isinstance(rot, (int, float)) else 0.0,
                        'layer': layer
                    })
                except Exception as e:
                    logger.warning(f"Footprint parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting footprints: {e}")

        # Pads (used to derive net pins) - Enhanced with full geometric data for accurate rendering
        pads = []
        try:
            all_pads = _ipc_retry(board.get_pads, "get_pads", max_retries=3, sleep_s=0.7)
            
            # Extract pad data with polygon shapes for accurate geometry
            logger.info(f"Found {len(all_pads)} pads - extracting with polygon shapes")
            
            # Get pad shapes as polygons for both front and back copper
            try:
                front_pad_shapes = board.get_pad_shapes_as_polygons(all_pads, 3)  # Front copper (layer 3)
                back_pad_shapes = board.get_pad_shapes_as_polygons(all_pads, 34)  # Back copper (layer 34)
                logger.info(f"Got pad polygon shapes: {len(front_pad_shapes)} front, {len(back_pad_shapes)} back")
            except Exception as e:
                logger.error(f"Error getting pad polygon shapes: {e}")
                front_pad_shapes = [None] * len(all_pads)
                back_pad_shapes = [None] * len(all_pads)
            
            for i, p in enumerate(all_pads):
                try:
                    # Basic position and net data
                    pos = getattr(p, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    y = float(getattr(pos, 'y', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    net_obj = getattr(p, 'net', None)
                    net = getattr(net_obj, 'name', None) if net_obj else None  # Extract net name from Net object
                    num = getattr(p, 'number', None)
                    
                    # Enhanced geometric data from padstack for accurate rendering
                    padstack = getattr(p, 'padstack', None)
                    size_x = 1.3  # Default
                    size_y = 1.3  # Default  
                    drill_diameter = 0.0
                    shape = 1  # Default to rectangle
                    
                    if padstack:
                        # Get drill diameter from padstack
                        drill = getattr(padstack, 'drill', None)
                        if drill:
                            drill_dia = getattr(drill, 'diameter', None)
                            if drill_dia and hasattr(drill_dia, 'x'):
                                drill_diameter = float(getattr(drill_dia, 'x', 0.0)) / 1000000.0  # Convert nm to mm
                        
                        # Get pad size and shape from copper layers
                        copper_layers = getattr(padstack, 'copper_layers', [])
                        if copper_layers and len(copper_layers) > 0:
                            first_layer = copper_layers[0]
                            
                            # Size
                            size = getattr(first_layer, 'size', None)
                            if size:
                                size_x = float(getattr(size, 'x', 1300000.0)) / 1000000.0  # Convert nm to mm
                                size_y = float(getattr(size, 'y', 1300000.0)) / 1000000.0  # Convert nm to mm
                            
                            # Shape (PSS_CIRCLE=0, PSS_RECTANGLE=1, PSS_OVAL=2, etc.)
                            shape_val = getattr(first_layer, 'shape', 1)
                            if hasattr(shape_val, 'value'):
                                shape = shape_val.value
                            else:
                                shape = int(shape_val)
                    
                    # Extract polygon shapes for this pad
                    polygons = {}
                    
                    # Process front copper polygon
                    if i < len(front_pad_shapes) and front_pad_shapes[i] is not None:
                        polygon_shape = front_pad_shapes[i]
                        outline_points = []
                        
                        if hasattr(polygon_shape, 'outline'):
                            outline = polygon_shape.outline
                            for point_node in outline:
                                if hasattr(point_node, 'point'):
                                    point = point_node.point
                                    outline_points.append({
                                        'x': point.x / 1000000.0,
                                        'y': point.y / 1000000.0
                                    })
                        
                        if outline_points:
                            polygons['F.Cu'] = {
                                'outline': outline_points,
                                'holes': []  # Pads typically don't have holes in their shape
                            }
                    
                    # Process back copper polygon
                    if i < len(back_pad_shapes) and back_pad_shapes[i] is not None:
                        polygon_shape = back_pad_shapes[i]
                        outline_points = []
                        
                        if hasattr(polygon_shape, 'outline'):
                            outline = polygon_shape.outline
                            for point_node in outline:
                                if hasattr(point_node, 'point'):
                                    point = point_node.point
                                    outline_points.append({
                                        'x': point.x / 1000000.0,
                                        'y': point.y / 1000000.0
                                    })
                        
                        if outline_points:
                            polygons['B.Cu'] = {
                                'outline': outline_points,
                                'holes': []
                            }
                    
                    # Determine actual layers this pad exists on
                    actual_pad_layers = []
                    if drill_diameter > 0:
                        # Through-hole pads appear on both layers
                        actual_pad_layers = ['F.Cu', 'B.Cu']
                    else:
                        # SMD pads - determine actual layer from padstack, not polygon keys
                        # (polygon keys include both layers even for SMD pads)
                        actual_pad_layers = []
                        
                        if padstack:
                            # Method 1: Check padstack copper layers to find the actual layer
                            copper_layers = getattr(padstack, 'copper_layers', [])
                            for layer in copper_layers:
                                layer_id = getattr(layer, 'layer', None)
                                if hasattr(layer_id, 'value'):
                                    layer_id = layer_id.value
                                elif hasattr(layer_id, 'layer_id'):
                                    layer_id = layer_id.layer_id
                                
                                # Map KiCad layer IDs to layer names
                                if layer_id == 3:  # F.Cu
                                    actual_pad_layers.append('F.Cu')
                                elif layer_id == 34:  # B.Cu
                                    actual_pad_layers.append('B.Cu')
                                else:
                                    # Check if this copper layer corresponds to F.Cu or B.Cu
                                    layer_name = str(layer_id)
                                    if 'front' in layer_name.lower() or 'f.cu' in layer_name.lower():
                                        actual_pad_layers.append('F.Cu')
                                    elif 'back' in layer_name.lower() or 'b.cu' in layer_name.lower():
                                        actual_pad_layers.append('B.Cu')
                        
                        # Remove duplicates
                        actual_pad_layers = list(set(actual_pad_layers))
                        
                        # Fallback: If no layer detected, use polygon presence to make best guess
                        if not actual_pad_layers:
                            # Only include a layer if it has significant polygon data
                            for layer_name, polygon_data in polygons.items():
                                outline = polygon_data.get('outline', [])
                                if len(outline) >= 3:  # Valid polygon
                                    # For SMD pads, typically only one layer should have valid data
                                    # But if both have data, prefer F.Cu (component side)
                                    if layer_name == 'F.Cu':
                                        actual_pad_layers = ['F.Cu']
                                        break
                                    elif layer_name == 'B.Cu' and not actual_pad_layers:
                                        actual_pad_layers = ['B.Cu']
                        
                        # Final fallback for SMD pads
                        if not actual_pad_layers:
                            actual_pad_layers = ['F.Cu']  # Default for SMD
                    
                    # === COMPREHENSIVE PAD DATA EXTRACTION ===
                    pad_data = {
                        'net': net, 
                        'number': num, 
                        'x': x, 
                        'y': y,
                        'size_x': size_x,
                        'size_y': size_y,
                        'shape': shape,
                        'drill_diameter': drill_diameter,
                        'layers': actual_pad_layers,
                        'polygons': polygons  # Add exact polygon shapes from KiCad
                    }
                    
                    # Pad classification
                    if drill_diameter > 0:
                        pad_data['pad_type'] = 'through-hole'  # THT
                    else:
                        pad_data['pad_type'] = 'smd'  # Surface mount if no drill
                    
                    # Debug log first few pads to verify data extraction
                    if i < 20:  # Show more pads to catch any SMD ones
                        polygon_info = f", polygons: {list(polygons.keys())}" if polygons else ""
                        drill_info = f", drill: {drill_diameter:.2f}mm" if drill_diameter > 0 else " (SMD)"
                        pad_type = "Through-hole" if drill_diameter > 0 else "Surface-mount"
                        layer_info = f", layers: {actual_pad_layers}"
                        
                        # Extra detailed logging for SMD pads
                        if drill_diameter == 0:
                            layer_detection_info = ""
                            if padstack:
                                copper_layers = getattr(padstack, 'copper_layers', [])
                                layer_ids = []
                                for layer in copper_layers:
                                    layer_id = getattr(layer, 'layer', None)
                                    if hasattr(layer_id, 'value'):
                                        layer_ids.append(layer_id.value)
                                    elif hasattr(layer_id, 'layer_id'):
                                        layer_ids.append(layer_id.layer_id)
                                    else:
                                        layer_ids.append(str(layer_id))
                                layer_detection_info = f", copper_layer_ids: {layer_ids}"
                            
                            logger.info(f"SMD Pad {i}: pos=({x:.2f}, {y:.2f}), size=({size_x:.2f}x{size_y:.2f}){drill_info} [{pad_type}]{polygon_info}{layer_info}{layer_detection_info}")
                        else:
                            logger.info(f"Pad {i}: pos=({x:.2f}, {y:.2f}), size=({size_x:.2f}x{size_y:.2f}){drill_info} [{pad_type}]{polygon_info}{layer_info}")
                    
                    # Quick stats for debugging
                    if i == 0:
                        smd_count = 0
                        th_count = 0
                        smd_front_count = 0
                        smd_back_count = 0
                        smd_both_count = 0
                    if drill_diameter > 0:
                        th_count += 1
                    else:
                        smd_count += 1
                        if actual_pad_layers == ['F.Cu']:
                            smd_front_count += 1
                        elif actual_pad_layers == ['B.Cu']:
                            smd_back_count += 1
                        elif 'F.Cu' in actual_pad_layers and 'B.Cu' in actual_pad_layers:
                            smd_both_count += 1
                    
                    # Show pad type breakdown after processing significant number
                    if i == 99 or i == len(all_pads) - 1:
                        logger.info(f"📊 Pad analysis (first {i+1} pads): {th_count} through-hole, {smd_count} SMD")
                        logger.info(f"📊 SMD breakdown: {smd_front_count} F.Cu only, {smd_back_count} B.Cu only, {smd_both_count} both layers (SHOULD BE 0!)")
                    
                    pads.append(pad_data)
                except Exception as e:
                    logger.warning(f"Pad parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting pads: {e}")

        # Tracks - with proper coordinate conversion
        tracks = []
        tracks_by_net = {}  # Track which nets have tracks
        try:
            trs = _ipc_retry(board.get_tracks, "get_tracks", max_retries=3, sleep_s=0.7)
            for i, tr in enumerate(trs):
                try:
                    start = getattr(tr, 'start', None)
                    end = getattr(tr, 'end', None)
                    net = getattr(getattr(tr, 'net', None), 'name', None)
                    s = (float(getattr(start, 'x', 0.0)) / 1000000.0, float(getattr(start, 'y', 0.0)) / 1000000.0) if start else (0.0, 0.0)  # Convert nm to mm
                    e = (float(getattr(end, 'x', 0.0)) / 1000000.0, float(getattr(end, 'y', 0.0)) / 1000000.0) if end else (0.0, 0.0)  # Convert nm to mm
                    
                    # Get track width and layer
                    width = float(getattr(tr, 'width', 200000)) / 1000000.0 if hasattr(tr, 'width') else 0.2  # Convert nm to mm, default 0.2mm
                    layer = getattr(tr, 'layer', 0)  # Get layer ID
                    
                    track_data = {
                        'start_x': s[0], 
                        'start_y': s[1],
                        'end_x': e[0], 
                        'end_y': e[1],
                        'width': width,
                        'layer': layer,
                        'net': net
                    }
                    tracks.append(track_data)
                    
                    # Track which nets have routing
                    if net:
                        tracks_by_net.setdefault(net, 0)
                        tracks_by_net[net] += 1
                        
                except Exception as e:
                    logger.warning(f"Track parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting tracks: {e}")

        logger.info(f"✅ Loaded {len(tracks)} existing tracks from KiCad")

        # Vias - with proper coordinate conversion  
        vias = []
        try:
            vs = _ipc_retry(board.get_vias, "get_vias", max_retries=3, sleep_s=0.7)
            for i, v in enumerate(vs):
                try:
                    pos = getattr(v, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    y = float(getattr(pos, 'y', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    
                    # Get via size
                    size = float(getattr(v, 'size', 0.6)) / 1000000.0 if hasattr(v, 'size') else 0.6  # Convert nm to mm, default 0.6mm
                    drill = float(getattr(v, 'drill', 0.3)) / 1000000.0 if hasattr(v, 'drill') else 0.3  # Convert nm to mm, default 0.3mm
                    
                    vias.append({
                        'x': x, 
                        'y': y, 
                        'via_diameter': size,
                        'drill_diameter': drill
                    })
                except Exception as e:
                    logger.warning(f"Via parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting vias: {e}")

        # Get copper zones/planes for plane-aware routing AND complete thermal relief extraction
        copper_zones = []
        zones_for_visualization = []
        try:
            zones = _ipc_retry(board.get_zones, "get_zones", max_retries=2, sleep_s=0.5)
            logger.info(f"Found {len(zones)} zones")
            
            for i, zone in enumerate(zones):
                try:
                    zone_net = getattr(zone, 'net', None)
                    zone_layers = getattr(zone, 'layers', [])
                    zone_net_name = zone_net.name if zone_net and hasattr(zone_net, 'name') else None
                    
                    logger.info(f"Zone {i}: net={zone_net_name}, layers={zone_layers}")
                    
                    # Basic zone info for routing
                    if zone_net_name:
                        copper_zones.append({
                            'net': zone_net_name,
                            'layers': list(zone_layers),
                            'filled': getattr(zone, 'is_filled', True)
                        })
                        logger.debug(f"Found copper zone: {zone_net_name} on {zone_layers}")
                    
                    # === COMPLETE THERMAL RELIEF EXTRACTION ===
                    # Extract filled polygons - this contains thermal relief geometry
                    filled_polygons = getattr(zone, 'filled_polygons', {})
                    
                    zone_data = {
                        'net': zone_net_name or 'unnamed',
                        'layers': list(zone_layers),
                        'filled_polygons': {}
                    }
                    
                    for layer_id, polygon_list in filled_polygons.items():
                        logger.info(f"  Layer {layer_id}: {len(polygon_list)} polygons")
                        layer_polygons = []
                        
                        for j, polygon in enumerate(polygon_list):
                            outline = getattr(polygon, 'outline', None)
                            holes = getattr(polygon, 'holes', [])
                            
                            if outline:
                                logger.info(f"    Polygon {j}: {len(outline)} outline points, {len(holes)} holes")
                                
                                # Convert outline points to our format
                                outline_points = []
                                for point in outline:
                                    # Handle PolyLineNode structure properly
                                    if hasattr(point, 'point'):
                                        actual_point = point.point  # Get Vector2 from PolyLineNode
                                        outline_points.append({
                                            'x': actual_point.x / 1000000.0,  # Convert to mm
                                            'y': actual_point.y / 1000000.0
                                        })
                                    elif hasattr(point, 'x') and hasattr(point, 'y'):
                                        # Direct point access
                                        outline_points.append({
                                            'x': point.x / 1000000.0,  # Convert to mm
                                            'y': point.y / 1000000.0
                                        })
                                
                                # Convert hole points to our format
                                hole_data = []
                                for hole in holes:
                                    hole_points = []
                                    for point in hole:
                                        # Handle PolyLineNode structure properly
                                        if hasattr(point, 'point'):
                                            actual_point = point.point  # Get Vector2 from PolyLineNode
                                            hole_points.append({
                                                'x': actual_point.x / 1000000.0,
                                                'y': actual_point.y / 1000000.0
                                            })
                                        elif hasattr(point, 'x') and hasattr(point, 'y'):
                                            # Direct point access
                                            hole_points.append({
                                                'x': point.x / 1000000.0,
                                                'y': point.y / 1000000.0
                                            })
                                    if hole_points:
                                        hole_data.append(hole_points)
                                
                                polygon_data = {
                                    'outline': outline_points,
                                    'holes': hole_data
                                }
                                
                                layer_polygons.append(polygon_data)
                                
                                # Log thermal relief detection
                                if len(outline_points) > 1000:
                                    logger.info(f"     🎯 THERMAL RELIEF DETECTED: {len(outline_points)} points trace around pads!")
                                
                        zone_data['filled_polygons'][layer_id] = layer_polygons
                    
                    zones_for_visualization.append(zone_data)
                    logger.info(f"  Added thermal relief zone {i} with layers: {list(zone_layers)}")
                    
                except Exception as e:
                    logger.error(f"Error processing zone {i}: {e}")
        
        except Exception as e:
            logger.error(f"Error getting zones: {e}")
        
        logger.info(f"Processed {len(zones_for_visualization)} zones with thermal relief data")

        # Nets (with pins derived from pads)
        nets = []
        try:
            board_nets = _ipc_retry(board.get_nets, "get_nets", max_retries=3, sleep_s=0.7)
            # Group pads by net - coordinates already converted to mm
            pins_by_net: Dict[str, List[Dict]] = {}
            for pad in pads:
                net_obj = pad.get('net')
                if not net_obj:
                    continue
                
                # Extract net name from Net object or string
                if hasattr(net_obj, 'name'):
                    net_name = net_obj.name
                elif isinstance(net_obj, str):
                    net_name = net_obj
                else:
                    continue
                    
                if not net_name or net_name == "":
                    continue
                    
                pins_by_net.setdefault(net_name, []).append({'x': pad['x'], 'y': pad['y'], 'layer': 0, 'pad_name': pad.get('number')})
            
            logger.info(f"Found {len(pins_by_net)} nets with pads")
            
            # Create set of nets that have copper planes (should be skipped)
            plane_nets = set()
            for zone in copper_zones:
                if zone.get('filled', True):
                    net_name = zone.get('net')
                    if net_name and isinstance(net_name, str):
                        plane_nets.add(net_name)
            
            if plane_nets:
                logger.info(f"🏭 Found nets with copper planes: {', '.join(sorted(plane_nets))}")
            
            # Store all nets for debugging
            all_nets_debug = []
            routable_nets = []
            
            for i, net in enumerate(board_nets):
                try:
                    name = getattr(net, 'name', f'Net_{i}')
                    net_pins = pins_by_net.get(name, [])
                    
                    # Add to debug list
                    all_nets_debug.append({
                        'name': name,
                        'pins': len(net_pins),
                        'has_plane': name in plane_nets,
                        'routable': len(net_pins) >= 2 and name not in plane_nets
                    })
                    
                    # Skip nets with copper planes (already connected via plane)
                    if name in plane_nets:
                        logger.info(f"⚡ Skipping '{name}' - connected via copper plane")
                        continue
                    
                    # Only include nets with 2+ pins for routing
                    if len(net_pins) >= 2:
                        # Check if this net has tracks - but don't assume fully routed
                        has_tracks = name in tracks_by_net and tracks_by_net[name] > 0
                        
                        # For airwire purposes, only mark as "routed" if it's a simple 2-pin net with tracks
                        # Multi-pin nets might be partially routed and still need airwires
                        is_fully_routed = has_tracks and len(net_pins) == 2
                        
                        routable_nets.append({
                            'id': i,
                            'name': name,
                            'pins': net_pins,
                            'routed': is_fully_routed,  # Only mark fully routed if 2 pins + tracks
                            'has_tracks': has_tracks,   # Track if it has any tracks
                            'priority': 1
                        })
                        logger.debug(f"Added net '{name}' with {len(net_pins)} pins (tracks: {has_tracks}, fully_routed: {is_fully_routed})")
                    else:
                        logger.debug(f"Skipped net '{name}' - only {len(net_pins)} pins")
                        
                except Exception as e:
                    logger.warning(f"Net parse error #{i}: {e}")
            
            nets = routable_nets
            logger.info(f"✅ Created {len(nets)} routable nets (excluding {len(plane_nets)} plane-connected nets)")
            
        except Exception as e:
            logger.error(f"Error getting nets: {e}")
            # Initialize empty nets and debug data on error
            nets = []
            all_nets_debug = []
            plane_nets = set()

        # Get board dimensions using computed bounds from all geometry
        # This is more reliable than trying to get dimensions from KiCad directly
        all_x = []
        all_y = []
        
        # Collect pad coordinates (already in mm)
        for p in pads:
            all_x.extend([p['x'] - p['size_x']/2, p['x'] + p['size_x']/2])
            all_y.extend([p['y'] - p['size_y']/2, p['y'] + p['size_y']/2])
            
        # Collect track coordinates (already in mm)
        for track in tracks:
            all_x.extend([track['start_x'], track['end_x']])
            all_y.extend([track['start_y'], track['end_y']])
            
        # Collect via coordinates (already in mm)
        for via in vias:
            all_x.extend([via['x'] - via['via_diameter']/2, via['x'] + via['via_diameter']/2])
            all_y.extend([via['y'] - via['via_diameter']/2, via['y'] + via['via_diameter']/2])
        
        # Collect component positions
        for comp in components:
            all_x.append(comp['x'])
            all_y.append(comp['y'])
            
        # Set bounds with defaults if no geometry found
        if all_x and all_y:
            min_x = min(all_x)
            min_y = min(all_y)
            max_x = max(all_x)
            max_y = max(all_y)
            
            # Add small margin
            margin = 5.0
            bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
            
            # Calculate actual board dimensions from geometry
            width = max_x - min_x + 2 * margin
            height = max_y - min_y + 2 * margin
            
            logger.info(f"Board dimensions calculated from geometry: {width:.1f} x {height:.1f} mm")
        else:
            # Fallback if no geometry
            bounds = (0.0, 0.0, 100.0, 80.0)
            width = 100.0
            height = 80.0
            logger.warning("No geometry found - using default bounds 100x80mm")

        # Get copper layer count
        copper_layers = self._get_layer_count()

        board_data = {
            'filename': filename,
            'width': width,  # Use calculated width
            'height': height,  # Use calculated height
            'copper_layers': copper_layers,  # Changed from 'layers' to 'copper_layers'
            'layers': copper_layers,  # Keep for backward compatibility
            'nets': nets,
            'components': components,
            'tracks': tracks,
            'vias': vias,
            'pads': pads,
            'bounds': bounds,
            'copper_zones': copper_zones,  # Include copper zone information
            'zones': zones_for_visualization,  # Include zones for visualization with filled polygons
            'copper_pours': zones_for_visualization,  # ALSO provide thermal reliefs as copper_pours for UI rendering
            'all_nets_debug': all_nets_debug,  # For debugging net filtering
            'unrouted_count': len([n for n in nets if not n.get('routed', False)]),
            'routed_count': len([n for n in nets if n.get('routed', False)])
        }

        # Generate airwires for unrouted nets
        airwires = self._generate_airwires(nets, plane_nets)
        board_data['airwires'] = airwires
        
        # Extract DRC rules
        logger.info("Extracting DRC rules...")
        drc_rules = self.extract_drc_rules()
        board_data['drc_rules'] = drc_rules
        
        logger.info(f"Extracted board data: {filename} ({width:.1f}x{height:.1f}mm, {copper_layers} copper layers)")
        logger.info(f"  {len(nets)} routable nets, {len(components)} components, {len(tracks)} tracks, {len(copper_zones)} zones")
        logger.info(f"  Generated {len(airwires)} airwires for visualization")
        if drc_rules:
            logger.info(f"  Extracted {len(drc_rules.netclasses)} netclasses with design rules")
        return board_data

    def _get_layer_count(self) -> int:
        """Get the number of copper layers in the board using KiCad stackup API"""
        try:
            # For large backplane boards, make intelligent assumptions first
            if hasattr(self.board, 'get_pads'):
                try:
                    pads = self.board.get_pads()
                    pad_count = len(pads)
                    
                    # This is clearly a large backplane - assume 12 layers
                    if pad_count > 15000:
                        logger.info(f"📋 Large backplane detected ({pad_count} pads), using 12 copper layers")
                        return 12
                    elif pad_count > 5000:
                        logger.info(f"📋 Medium complex board detected ({pad_count} pads), using 8 copper layers") 
                        return 8
                        
                except Exception as e:
                    logger.debug(f"Pad count analysis failed: {e}")
            
            # Method 1: Use KiCad board stackup API (most reliable)
            if hasattr(self.board, 'get_stackup'):
                try:
                    stackup = self.board.get_stackup()
                    copper_layers = []
                    layer_names = []
                    
                    for stackup_layer in stackup.layers:
                        # Check if this is an enabled copper layer
                        if (stackup_layer.enabled and 
                            stackup_layer.layer != -1):  # BL_UNDEFINED = -1
                            
                            # Get layer name to check if it's a copper layer
                            layer_name = getattr(stackup_layer, 'user_name', '')
                            if not layer_name:
                                # Try to get standard layer name
                                layer_id = stackup_layer.layer
                                if layer_id == 0:
                                    layer_name = 'F.Cu'
                                elif layer_id == 31:
                                    layer_name = 'B.Cu'
                                elif 1 <= layer_id <= 30:
                                    layer_name = f'In{layer_id}.Cu'
                                else:
                                    layer_name = f'Layer_{layer_id}'
                            
                            # Check if this is actually a copper layer by name
                            if '.Cu' in layer_name or 'copper' in layer_name.lower():
                                copper_layers.append(stackup_layer.layer)
                                layer_names.append(layer_name)
                    
                    if len(copper_layers) > 0:
                        logger.info(f"📋 Board stackup detected {len(copper_layers)} copper layers: {layer_names}")
                        logger.info(f"📋 Layer IDs: {copper_layers}")
                        return len(copper_layers)
                        
                except Exception as e:
                    logger.debug(f"Board stackup method failed: {e}")
            
            # Method 2: Check board layer_count property first  
            if hasattr(self.board, 'layer_count'):
                layer_count = getattr(self.board, 'layer_count')
                if isinstance(layer_count, int) and layer_count > 0:
                    logger.info(f"📋 Board layer_count property: {layer_count} layers")
                    return layer_count
            
            # Method 3: Use visible layers API but filter more carefully
            if hasattr(self.board, 'get_visible_layers'):
                try:
                    visible_layers = self.board.get_visible_layers()
                    
                    # First try to find the actual board's copper layers by analyzing existing pads/tracks
                    actual_copper_layers = set()
                    
                    # Check tracks to see what layers are actually used
                    try:
                        tracks = self.board.get_tracks()
                        for track in tracks:
                            layer = getattr(track, 'layer', None)
                            if layer is not None:
                                actual_copper_layers.add(layer)
                    except:
                        pass
                    
                    # Check pads to see what copper layers exist
                    try:
                        pads = self.board.get_pads()
                        for pad in pads[:100]:  # Sample pads
                            padstack = getattr(pad, 'padstack', None)
                            if padstack:
                                copper_layers_attr = getattr(padstack, 'copper_layers', [])
                                for layer in copper_layers_attr:
                                    actual_copper_layers.add(layer)
                    except:
                        pass
                    
                    # Filter visible layers to only include copper layers that exist in the design
                    copper_layer_ids = set()
                    for layer_id in visible_layers:
                        # KiCad copper layer IDs: F.Cu=0, In1.Cu=1-30, B.Cu=31
                        if (layer_id == 0 or layer_id == 31 or (1 <= layer_id <= 30)):
                            # Only include if we found evidence this layer is actually used
                            if actual_copper_layers and layer_id in actual_copper_layers:
                                copper_layer_ids.add(layer_id)
                            elif not actual_copper_layers:  # Fallback if no track/pad analysis worked
                                copper_layer_ids.add(layer_id)
                    
                    if len(copper_layer_ids) > 0:
                        sorted_layers = sorted(copper_layer_ids)
                        layer_names = []
                        for layer_id in sorted_layers:
                            if layer_id == 0:
                                layer_names.append('F.Cu')
                            elif layer_id == 31:
                                layer_names.append('B.Cu')
                            else:
                                layer_names.append(f'In{layer_id}.Cu')
                        
                        logger.info(f"📋 Visible layers detected {len(sorted_layers)} copper layers: {layer_names}")
                        return len(sorted_layers)
                        
                except Exception as e:
                    logger.debug(f"Visible layers method failed: {e}")
            
            # Method 4: Analyze pad data to find unique copper layers
            copper_layers_found = set()
            layer_names_found = []
            try:
                pads = self.board.get_pads()
                for pad in pads[:100]:  # Check more pads for better detection
                    padstack = getattr(pad, 'padstack', None)
                    if padstack:
                        padstack_layers = getattr(padstack, 'layers', [])
                        for layer in padstack_layers:
                            layer_id = int(layer) if str(layer).isdigit() else layer
                            # F.Cu=0, B.Cu=31, Inner layers 1-30
                            if (layer_id == 0 or layer_id == 31 or (1 <= layer_id <= 30)):
                                copper_layers_found.add(layer_id)
                
                if len(copper_layers_found) > 0:
                    sorted_layers = sorted(copper_layers_found)
                    for layer_id in sorted_layers:
                        if layer_id == 0:
                            layer_names_found.append('F.Cu')
                        elif layer_id == 31:
                            layer_names_found.append('B.Cu')
                        else:
                            layer_names_found.append(f'In{layer_id}.Cu')
                    
                    logger.info(f"📋 Pad analysis detected {len(sorted_layers)} copper layers: {layer_names_found}")
                    return len(sorted_layers)
                    
            except Exception as e:
                logger.debug(f"Pad analysis for copper layers failed: {e}")
            
            # Default assumption for PCBs
            logger.warning("⚠️ Could not determine layer count - defaulting to 2 layers")
            return 2
            
        except Exception as e:
            logger.warning(f"Error determining layer count: {e}")
            return 2

    def _generate_airwires(self, nets: List[Dict], plane_nets: set) -> List[Dict]:
        """Generate airwires (ratsnest lines) for unrouted nets using minimum spanning tree"""
        airwires = []
        filtered_airwires = 0
        partial_airwires = 0
        
        try:
            for net in nets:
                net_name = net.get('name', '')
                pins = net.get('pins', [])
                is_routed = net.get('routed', False)
                has_tracks = net.get('has_tracks', False)
                
                # Skip if fully routed or has copper plane connection
                if is_routed or net_name in plane_nets:
                    if net_name in plane_nets:
                        filtered_airwires += 1
                        logger.debug(f"Skipping airwires for net '{net_name}' - has copper pour")
                    continue
                
                # Need at least 2 pins to create airwires
                if len(pins) < 2:
                    continue
                
                # Generate airwires even for partially routed nets
                if has_tracks and len(pins) > 2:
                    partial_airwires += 1
                    logger.debug(f"Generating airwires for partially routed net '{net_name}' ({len(pins)} pins)")
                
                # Generate minimum spanning tree for this net's pins
                mst_airwires = self._generate_mst_airwires(pins, net_name)
                airwires.extend(mst_airwires)
            
            logger.info(f"Generated {len(airwires)} airwires from {len(nets)} nets")
            logger.info(f"  Including {partial_airwires} partially routed nets")
            logger.info(f"  Filtered out {filtered_airwires} nets with copper pours")
            
        except Exception as e:
            logger.error(f"Error generating airwires: {e}")
            
        return airwires
    
    def _generate_mst_airwires(self, pins: List[Dict], net_name: str) -> List[Dict]:
        """Generate minimum spanning tree airwires for a single net"""
        if len(pins) < 2:
            return []
        
        # If only 2 pins, just connect them directly
        if len(pins) == 2:
            return [{
                'start_x': pins[0]['x'],
                'start_y': pins[0]['y'],
                'end_x': pins[1]['x'],
                'end_y': pins[1]['y'],
                'net': net_name
            }]
        
        # For 3+ pins, use Prim's minimum spanning tree algorithm
        import math
        
        def distance(p1, p2):
            """Calculate Euclidean distance between two pins"""
            dx = p1['x'] - p2['x']
            dy = p1['y'] - p2['y']
            return math.sqrt(dx*dx + dy*dy)
        
        # Prim's MST algorithm
        n = len(pins)
        visited = [False] * n
        min_edge = [float('inf')] * n
        parent = [-1] * n
        
        # Start with pin 0
        min_edge[0] = 0
        mst_edges = []
        
        for _ in range(n):
            # Find minimum weight edge from visited to unvisited
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or min_edge[v] < min_edge[u]):
                    u = v
            
            visited[u] = True
            
            # Add edge to MST (except for the first vertex)
            if parent[u] != -1:
                mst_edges.append({
                    'start_x': pins[parent[u]]['x'],
                    'start_y': pins[parent[u]]['y'],
                    'end_x': pins[u]['x'],
                    'end_y': pins[u]['y'],
                    'net': net_name
                })
            
            # Update minimum edges for unvisited vertices
            for v in range(n):
                if not visited[v]:
                    dist = distance(pins[u], pins[v])
                    if dist < min_edge[v]:
                        min_edge[v] = dist
                        parent[v] = u
        
        return mst_edges

    def _get_board_dimensions(self) -> Tuple[float, float]:
        """Get board width and height in mm"""
        try:
            # Method 1: Try to get dimensions directly from board
            if hasattr(self.board, 'get_dimensions'):
                dimensions = self.board.get_dimensions()
                if dimensions and hasattr(dimensions, 'width') and hasattr(dimensions, 'height'):
                    width_mm = float(dimensions.width) / 1000000.0  # Convert nm to mm
                    height_mm = float(dimensions.height) / 1000000.0  # Convert nm to mm
                    if width_mm > 0 and height_mm > 0:
                        logger.info(f"Board dimensions from get_dimensions(): {width_mm:.1f} x {height_mm:.1f} mm")
                        return width_mm, height_mm
            
            # Method 2: Try to get board outline/edge cuts
            if hasattr(self.board, 'get_board_outline'):
                try:
                    outline = self.board.get_board_outline()
                    if outline:
                        # Extract bounding box from outline
                        min_x = min_y = float('inf')
                        max_x = max_y = float('-inf')
                        
                        # Process outline points
                        for point in outline:
                            if hasattr(point, 'x') and hasattr(point, 'y'):
                                x_mm = float(point.x) / 1000000.0
                                y_mm = float(point.y) / 1000000.0
                                min_x = min(min_x, x_mm)
                                max_x = max(max_x, x_mm)
                                min_y = min(min_y, y_mm)
                                max_y = max(max_y, y_mm)
                        
                        if min_x != float('inf'):
                            width_mm = max_x - min_x
                            height_mm = max_y - min_y
                            logger.info(f"Board dimensions from outline: {width_mm:.1f} x {height_mm:.1f} mm")
                            return width_mm, height_mm
                except Exception as e:
                    logger.debug(f"Could not get board outline: {e}")
            
            # Method 3: Calculate from component and pad positions
            all_x = []
            all_y = []
            
            # Get footprint positions
            try:
                footprints = self.board.get_footprints()
                for fp in footprints:
                    pos = getattr(fp, 'position', None)
                    if pos:
                        x_mm = float(getattr(pos, 'x', 0.0)) / 1000000.0
                        y_mm = float(getattr(pos, 'y', 0.0)) / 1000000.0
                        all_x.append(x_mm)
                        all_y.append(y_mm)
            except Exception as e:
                logger.debug(f"Could not get footprint positions: {e}")
            
            # Get pad positions
            try:
                pads = self.board.get_pads()
                for pad in pads[:50]:  # Sample to avoid performance issues
                    pos = getattr(pad, 'position', None)
                    if pos:
                        x_mm = float(getattr(pos, 'x', 0.0)) / 1000000.0
                        y_mm = float(getattr(pos, 'y', 0.0)) / 1000000.0
                        all_x.append(x_mm)
                        all_y.append(y_mm)
            except Exception as e:
                logger.debug(f"Could not get pad positions: {e}")
            
            if all_x and all_y:
                # Add margin around components
                margin = 5.0  # 5mm margin
                width_mm = max(all_x) - min(all_x) + 2 * margin
                height_mm = max(all_y) - min(all_y) + 2 * margin
                logger.info(f"Board dimensions estimated from components: {width_mm:.1f} x {height_mm:.1f} mm")
                return width_mm, height_mm
            
            # Default fallback
            logger.warning("Could not determine board dimensions - using default 100x80mm")
            return 100.0, 80.0
            
        except Exception as e:
            logger.error(f"Error getting board dimensions: {e}")
            return 100.0, 80.0

    def _layer_name_to_id(self, layer_name: str) -> int:
        """Convert layer name to layer ID for KiCad IPC API"""
        # Standard KiCad layer mapping
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
            'B.Cu': 31,     # Back copper
        }
        return layer_map.get(layer_name, 0)  # Default to F.Cu if unknown

    # The following are stubs; real creation via IPC will be added later
    def create_track(self, start_x: float, start_y: float, end_x: float, end_y: float,
                     layer: str, width: float, net_name: str) -> bool:
        """Create a straight track on the board via IPC.
        Units: mm for coordinates and width. Layer by name (e.g., 'F.Cu').
        """
        if not self.connected or not self.board:
            logger.warning("create_track called while not connected; ignoring")
            return False
        try:
            from kipy.board_types import Track, Vector2
            
            # Log the track creation attempt
            logger.debug(f"🔧 Creating track: {net_name} from ({start_x:.3f}, {start_y:.3f}) to ({end_x:.3f}, {end_y:.3f}) on {layer}, width={width:.3f}mm")
            
            # Find the net by name
            target_net = None
            nets = self.board.get_nets()
            for net in nets:
                if net.name == net_name:
                    target_net = net
                    break
            
            if not target_net:
                logger.warning(f"❌ Net '{net_name}' not found")
                return False
            
            # Convert mm to nanometers (KiCad internal units)
            start_x_nm = int(start_x * 1_000_000)
            start_y_nm = int(start_y * 1_000_000) 
            end_x_nm = int(end_x * 1_000_000)
            end_y_nm = int(end_y * 1_000_000)
            width_nm = int(width * 1_000_000)
            
            # Convert layer name to layer ID
            layer_id = self._layer_name_to_id(layer)
            
            # Create track object
            track = Track()
            track.start = Vector2(start_x_nm, start_y_nm)
            track.end = Vector2(end_x_nm, end_y_nm)
            track.layer = layer_id
            track.width = width_nm
            track.net = target_net
            
            # Create the track via IPC API
            result = self.board.create_items([track])
            
            if result and len(result) > 0:
                logger.debug(f"✅ Created track {net_name} from ({start_x:.3f}, {start_y:.3f}) to ({end_x:.3f}, {end_y:.3f}) on {layer}")
                return True
            else:
                logger.warning(f"❌ Track creation returned empty result for {net_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ IPC create_track failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def create_via(self, x: float, y: float, size: float, drill: float,
                   from_layer: str, to_layer: str, net_name: str) -> bool:
        """Create a via via IPC; size/drill in mm, layer pair by names.
        """
        if not self.connected or not self.board:
            logger.warning("create_via called while not connected; ignoring")
            return False
        try:
            layer_pair = (from_layer or 'F.Cu', to_layer or 'B.Cu')
            
            # Log the via creation attempt
            logger.debug(f"🔧 Creating via: {net_name} at ({x:.3f}, {y:.3f}), size={size:.3f}, drill={drill:.3f}, layers={layer_pair}")
            
            # Create via via IPC API
            via = self.board.add_via((float(x), float(y)), layer_pair, size=float(size) or 0.4, drill=float(drill) or 0.2, net=net_name or None)
            
            if via:
                logger.info(f"✅ Created via {net_name} at ({x},{y}) size={size} drill={drill} layers={layer_pair}")
                return True
            else:
                logger.warning(f"❌ Via creation returned None for {net_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ IPC create_via failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def refresh_board(self):
        try:
            # Some IPC implementations require an explicit refresh/commit; attempt if available
            if hasattr(self.board, 'refresh'):
                self.board.refresh()
            elif hasattr(self.client, 'refresh_board'):
                self.client.refresh_board()
        except Exception as e:
            logger.error(f"Error refreshing board: {e}")

    def get_pad_polygons(self) -> List[Dict]:
        """
        Get exact pad polygon shapes using get_pad_shapes_as_polygons().
        Returns list of pad dictionaries with precise polygon geometry.
        """
        if not self.connected or not self.board:
            return []
        
        try:
            # Get all pads
            all_pads = _ipc_retry(self.board.get_pads, "get_pads", max_retries=3, sleep_s=0.7)
            
            # Get exact polygon shapes for all pads on front copper layer
            import kipy
            front_cu_layer = kipy.board_types.BoardLayer.BL_F_Cu
            pad_shapes = self.board.get_pad_shapes_as_polygons(all_pads, front_cu_layer)
            
            pad_data = []
            
            for pad, polygon_shape in zip(all_pads, pad_shapes):
                try:
                    # Get basic pad info
                    pad_pos = pad.position
                    x_mm = pad_pos.x / 1e6  # Convert nanometers to mm
                    y_mm = pad_pos.y / 1e6
                    
                    net_name = pad.net.name if pad.net else "No Net"
                    
                    # Extract drill diameter information
                    drill_diameter = 0.0
                    if hasattr(pad, 'padstack') and pad.padstack:
                        padstack = pad.padstack
                        if hasattr(padstack, 'drill') and padstack.drill:
                            drill = padstack.drill
                            if hasattr(drill, 'diameter'):
                                drill_raw = drill.diameter
                                # Handle Vector2 drill diameter
                                if hasattr(drill_raw, 'x') and hasattr(drill_raw, 'y'):
                                    drill_x = float(drill_raw.x) / 1e6  # Convert nm to mm
                                    drill_y = float(drill_raw.y) / 1e6
                                    drill_diameter = max(drill_x, drill_y)  # Use larger dimension
                                elif isinstance(drill_raw, (int, float)):
                                    drill_diameter = float(drill_raw) / 1e6
                    
                    pad_info = {
                        'x': x_mm,
                        'y': y_mm,
                        'net': net_name,
                        'pad_number': getattr(pad, 'number', ''),
                        'position': (x_mm, y_mm),
                        'drill_diameter': drill_diameter
                    }
                    
                    # Extract exact polygon geometry
                    if polygon_shape is not None and hasattr(polygon_shape, 'outline'):
                        outline = polygon_shape.outline
                        
                        if hasattr(outline, 'nodes'):
                            # Extract all polygon points
                            polygon_points = []
                            for node in outline.nodes:
                                if hasattr(node, 'point'):
                                    point = node.point
                                    point_x_mm = point.x / 1e6
                                    point_y_mm = point.y / 1e6
                                    polygon_points.append((point_x_mm, point_y_mm))
                            
                            pad_info['shape'] = 'polygon'
                            pad_info['polygon_points'] = polygon_points
                            pad_info['point_count'] = len(polygon_points)
                            
                            # Extract holes if present
                            holes = []
                            if hasattr(polygon_shape, 'holes'):
                                for hole in polygon_shape.holes:
                                    if hasattr(hole, 'nodes'):
                                        hole_points = []
                                        for node in hole.nodes:
                                            if hasattr(node, 'point'):
                                                point = node.point
                                                hole_x_mm = point.x / 1e6
                                                hole_y_mm = point.y / 1e6
                                                hole_points.append((hole_x_mm, hole_y_mm))
                                        holes.append(hole_points)
                            
                            pad_info['holes'] = holes
                        
                        else:
                            # Fallback: no nodes available
                            pad_info['shape'] = 'unknown'
                            pad_info['polygon_points'] = []
                            pad_info['holes'] = []
                    
                    else:
                        # No polygon shape on this layer
                        pad_info['shape'] = 'none'
                        pad_info['polygon_points'] = []
                        pad_info['holes'] = []
                    
                    pad_data.append(pad_info)
                    
                except Exception as e:
                    logger.warning(f"Error processing pad polygon: {e}")
                    continue
            
            logger.info(f"Extracted {len(pad_data)} pad polygons")
            
            # Log complexity summary
            shape_counts = {}
            for pad in pad_data:
                point_count = pad.get('point_count', 0)
                if point_count in shape_counts:
                    shape_counts[point_count] += 1
                else:
                    shape_counts[point_count] = 1
            
            logger.info(f"Pad polygon complexity: {shape_counts}")
            
            return pad_data
            
        except Exception as e:
            logger.error(f"Error extracting pad polygons: {e}")
            return []

    def extract_drc_rules(self) -> Optional[DRCRules]:
        """Extract Design Rule Check information from KiCad board and project"""
        if not self.board:
            logger.error("No board connection available for DRC extraction")
            return None
            
        try:
            logger.info("🔍 Extracting DRC rules from KiCad...")
            
            # Get project for netclass information
            project = _ipc_retry(self.board.get_project, "get_project")
            
            # Get all netclasses from project
            netclasses_list = _ipc_retry(project.get_net_classes, "get_net_classes")
            
            # Convert netclasses to dict format
            netclasses = {}
            default_track_width = 0.2  # mm fallback
            default_via_size = 0.8     # mm fallback
            default_via_drill = 0.4    # mm fallback
            default_clearance = 0.2    # mm fallback
            minimum_track_width = 0.1  # mm fallback
            minimum_via_size = 0.4     # mm fallback
            
            for netclass in netclasses_list:
                try:
                    # Extract netclass properties
                    netclass_name = getattr(netclass, 'name', 'Default')
                    
                    # Try to extract design rule parameters
                    rules = {
                        'name': netclass_name,
                        'track_width': 0.2,  # fallback
                        'via_size': 0.8,     # fallback
                        'via_drill': 0.4,    # fallback
                        'clearance': 0.2,    # fallback
                    }
                    
                    # Extract actual values if available
                    # Note: NetClass properties may vary based on KiCad API version
                    if hasattr(netclass, 'track_width'):
                        rules['track_width'] = getattr(netclass, 'track_width', 0.2) / 1000000.0  # convert from nm to mm
                    elif hasattr(netclass, 'TrackWidth'):
                        rules['track_width'] = getattr(netclass, 'TrackWidth', 200000) / 1000000.0
                        
                    if hasattr(netclass, 'via_size'):
                        rules['via_size'] = getattr(netclass, 'via_size', 0.8) / 1000000.0
                    elif hasattr(netclass, 'ViaSize'):
                        rules['via_size'] = getattr(netclass, 'ViaSize', 800000) / 1000000.0
                        
                    if hasattr(netclass, 'via_drill'):
                        rules['via_drill'] = getattr(netclass, 'via_drill', 0.4) / 1000000.0
                    elif hasattr(netclass, 'ViaDrill'):
                        rules['via_drill'] = getattr(netclass, 'ViaDrill', 400000) / 1000000.0
                        
                    if hasattr(netclass, 'clearance'):
                        rules['clearance'] = getattr(netclass, 'clearance', 0.2) / 1000000.0
                    elif hasattr(netclass, 'Clearance'):
                        rules['clearance'] = getattr(netclass, 'Clearance', 200000) / 1000000.0
                    
                    netclasses[netclass_name] = rules
                    
                    # Set defaults from "Default" netclass
                    if netclass_name.lower() == 'default':
                        default_track_width = rules['track_width']
                        default_via_size = rules['via_size']
                        default_via_drill = rules['via_drill']
                        default_clearance = rules['clearance']
                    
                    logger.info(f"  NetClass '{netclass_name}': track={rules['track_width']:.3f}mm via={rules['via_size']:.3f}mm clearance={rules['clearance']:.3f}mm")
                    
                except Exception as e:
                    logger.warning(f"Error processing netclass {getattr(netclass, 'name', 'Unknown')}: {e}")
                    continue
            
            # Get additional constraints from board if available
            try:
                # Try to get board-level design rules
                # This may not be available in all KiCad API versions
                if hasattr(self.board, 'get_design_rules'):
                    board_rules = self.board.get_design_rules()
                    logger.info(f"Found board-level design rules: {board_rules}")
                elif hasattr(self.board, 'GetDesignSettings'):
                    design_settings = self.board.GetDesignSettings()
                    logger.info(f"Found design settings: {design_settings}")
            except Exception as e:
                logger.debug(f"Board design rules not available: {e}")
            
            # Create DRC rules object
            drc_rules = DRCRules(
                netclasses=netclasses,
                default_track_width=default_track_width,
                default_via_size=default_via_size,
                default_via_drill=default_via_drill,
                default_clearance=default_clearance,
                minimum_track_width=min(minimum_track_width, default_track_width * 0.5),
                minimum_via_size=min(minimum_via_size, default_via_size * 0.5)
            )
            
            logger.info(f"✅ Extracted DRC rules: {len(netclasses)} netclasses")
            logger.info(f"   Default track width: {default_track_width:.3f}mm")
            logger.info(f"   Default via size: {default_via_size:.3f}mm")
            logger.info(f"   Default clearance: {default_clearance:.3f}mm")
            
            return drc_rules
            
        except Exception as e:
            logger.error(f"Error extracting DRC rules: {e}")
            logger.info("Will use fallback DRC rules")
            
            # Return fallback DRC rules
            return DRCRules(
                netclasses={'Default': {
                    'name': 'Default',
                    'track_width': 0.2,
                    'via_size': 0.8,
                    'via_drill': 0.4,
                    'clearance': 0.2
                }},
                default_track_width=0.2,
                default_via_size=0.8,
                default_via_drill=0.4,
                default_clearance=0.2,
                minimum_track_width=0.1,
                minimum_via_size=0.4
            )

    def get_net_constraints(self, net_name: str) -> Dict[str, float]:
        """Get routing constraints for a specific net"""
        if not self.board:
            logger.error("No board connection available")
            return {}
            
        try:
            # Get nets and their netclass assignments
            nets = _ipc_retry(self.board.get_nets, "get_nets")
            target_net = None
            
            for net in nets:
                if net.name == net_name:
                    target_net = net
                    break
            
            if not target_net:
                logger.warning(f"Net '{net_name}' not found")
                return {}
            
            # Get netclass for this net
            netclass_mapping = _ipc_retry(lambda: self.board.get_netclass_for_nets([target_net]), "get_netclass_for_nets")
            
            if target_net in netclass_mapping:
                netclass = netclass_mapping[target_net]
                netclass_name = getattr(netclass, 'name', 'Default')
                
                # Extract constraints
                constraints = {
                    'netclass': netclass_name,
                    'track_width': getattr(netclass, 'track_width', 200000) / 1000000.0,  # nm to mm
                    'via_size': getattr(netclass, 'via_size', 800000) / 1000000.0,
                    'via_drill': getattr(netclass, 'via_drill', 400000) / 1000000.0,
                    'clearance': getattr(netclass, 'clearance', 200000) / 1000000.0
                }
                
                logger.info(f"Net '{net_name}' constraints: {constraints}")
                return constraints
            
        except Exception as e:
            logger.error(f"Error getting constraints for net '{net_name}': {e}")
        
        # Return defaults
        return {
            'netclass': 'Default',
            'track_width': 0.2,
            'via_size': 0.8,
            'via_drill': 0.4,
            'clearance': 0.2
        }

    def _get_fallback_board_data(self) -> Dict:
        """Fallback mock data if IPC connection fails"""
        fallback_drc = DRCRules(
            netclasses={'Default': {
                'name': 'Default',
                'track_width': 0.2,
                'via_size': 0.8,
                'via_drill': 0.4,
                'clearance': 0.2
            }},
            default_track_width=0.2,
            default_via_size=0.8,
            default_via_drill=0.4,
            default_clearance=0.2,
            minimum_track_width=0.1,
            minimum_via_size=0.4
        )
        
        return {
            'filename': 'Mock_Board.kicad_pcb',
            'width': 100.0,
            'height': 80.0,
            'layers': 4,
            'nets': [],
            'components': [],
            'tracks': [],
            'vias': [],
            'pads': [],
            'bounds': (0.0, 0.0, 100.0, 80.0),
            'unrouted_count': 0,
            'routed_count': 0,
            'drc_rules': fallback_drc
        }
