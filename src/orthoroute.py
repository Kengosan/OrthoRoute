#!/usr/bin/env python3
"""
OrthoRoute - Professional PCB Autorouting Plugin for KiCad
GPU-Accelerated routing with real-time visualization and IPC API integration
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

# Setup logging
log_file = Path.home() / "Documents" / "orthoroute.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check for required dependencies"""
    required_packages = ['kipy', 'PyQt6', 'numpy']
    optional_packages = ['cupy']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            if package == 'kipy':
                import kipy
                logger.info("✓ Found kipy at: " + str(Path(kipy.__file__).parent))
            elif package == 'PyQt6':
                import PyQt6
                logger.info("✓ PyQt6 fully available and functional - using current environment")
            elif package == 'numpy':
                import numpy
                logger.info("✓ NumPy available for array operations")
        except ImportError:
            missing_required.append(package)
            logger.warning(f"Missing required package: {package}")
    
    for package in optional_packages:
        try:
            if package == 'cupy':
                import cupy
                logger.info("✓ CuPy available for GPU acceleration")
        except ImportError:
            missing_optional.append(package)
            logger.info(f"Optional package not found: {package}")
    
    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        return False
    
    return True

def connect_to_kicad() -> Optional[Any]:
    """Connect to KiCad via IPC API"""
    try:
        from kipy import KiCad
        
        # Try to get IPC credentials from environment
        api_socket = os.environ.get('KICAD_API_SOCKET')
        api_token = os.environ.get('KICAD_API_TOKEN')
        
        if api_socket and api_token:
            logger.info("✓ Found IPC credentials in environment")
            kicad = KiCad(socket_path=api_socket, kicad_token=api_token, timeout_ms=15000)
        else:
            logger.warning("⚠ IPC credentials not found - attempting direct connection")
            logger.info("  This may work for local KiCad instances without authentication")
            logger.info("⚠ Attempting connection without explicit credentials")
            kicad = KiCad(timeout_ms=15000)
        
        # Test connection by getting board
        board = kicad.get_board()
        if board:
            logger.info("✓ KiCad IPC connection established")
            return kicad
        else:
            logger.error("❌ Could not retrieve board from KiCad")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to KiCad: {e}")
        return None

def analyze_board_data(board) -> Dict[str, Any]:
    """Analyze board data using IPC APIs"""
    try:
        logger.info("✓ Board object retrieved")
        board_name = getattr(board, 'name', 'PCB')
        logger.info(f"Board: {board_name}")
        
        # Discover available methods
        available_methods = [method for method in dir(board) if not method.startswith('_')]
        logger.info(f"Available board methods: {', '.join(available_methods[:20])}...")
        
        # Look for color/theme related methods
        color_methods = [method for method in available_methods if 'color' in method.lower() or 'theme' in method.lower() or 'appearance' in method.lower()]
        if color_methods:
            logger.info(f"Color/theme related methods: {', '.join(color_methods)}")
        else:
            logger.info("No obvious color/theme methods found on board object")
        
        # Try to find KiCad color theme data
        logger.info("Searching for KiCad color theme data...")
        
        # Check if board has access to color theme data
        theme_methods = [method for method in available_methods if any(keyword in method.lower() for keyword in ['color', 'theme', 'palette', 'settings'])]
        if theme_methods:
            logger.info(f"Potential theme methods: {', '.join(theme_methods)}")
            
            # Try some common methods that might expose color data
            for method_name in theme_methods:
                try:
                    method = getattr(board, method_name)
                    if callable(method):
                        logger.info(f"Found callable method: {method_name}")
                        # Could try calling it here if safe
                except Exception as e:
                    logger.debug(f"Error checking method {method_name}: {e}")
        
        # Get KiCad theme/appearance settings
        kicad_theme_colors = {}
        try:
            if hasattr(board, 'get_editor_appearance_settings'):
                appearance = board.get_editor_appearance_settings()
                logger.info("✓ Retrieved KiCad appearance settings")
                logger.info(f"Appearance object type: {type(appearance)}")
                
                # Log available appearance attributes
                if appearance:
                    appearance_attrs = [attr for attr in dir(appearance) if not attr.startswith('_')]
                    logger.info(f"Appearance attributes: {', '.join(appearance_attrs[:20])}")
                    
                    # Extract appearance settings from BoardEditorAppearanceSettings
                    # NOTE: This class only provides display settings, not actual color values
                    kicad_theme_colors['board_flip'] = appearance.board_flip
                    logger.info(f"Board flip setting: {appearance.board_flip}")
                    
                    kicad_theme_colors['inactive_layer_display'] = appearance.inactive_layer_display
                    logger.info(f"Inactive layer display mode: {appearance.inactive_layer_display}")
                    
                    kicad_theme_colors['net_color_display'] = appearance.net_color_display
                    logger.info(f"Net color display mode: {appearance.net_color_display}")
                    
                    kicad_theme_colors['ratsnest_display'] = appearance.ratsnest_display
                    logger.info(f"Ratsnest display setting: {appearance.ratsnest_display}")
                    
                    # Try to load authentic KiCad theme data
                    theme_file = Path(__file__).parent / "assets" / "kicad_theme.json"
                    if theme_file.exists():
                        try:
                            with open(theme_file, 'r') as f:
                                kicad_json_theme = json.load(f)
                            kicad_theme_colors['json_theme'] = kicad_json_theme
                            logger.info("✓ Loaded authentic KiCad theme from assets/kicad_theme.json")
                        except Exception as e:
                            logger.warning(f"Could not load KiCad theme file: {e}")
                    else:
                        # Fallback: Add minimal sample theme data
                        sample_kicad_theme = {
                            "board": {
                                "background": "rgb(0, 16, 35)",
                                "copper": {
                                    "f": "rgb(200, 52, 52)",
                                    "b": "rgb(77, 127, 196)"
                                },
                                "edge_cuts": "rgb(208, 210, 205)",
                                "through_via": "rgb(236, 236, 236)",
                                "ratsnest": "rgba(245, 255, 213, 0.702)"
                            }
                        }
                        kicad_theme_colors['json_theme'] = sample_kicad_theme
                        logger.info("✓ Using minimal KiCad theme (no theme file found)")
                        
            else:
                logger.info("⚠ get_editor_appearance_settings not available")
        except Exception as e:
            logger.warning(f"Could not get appearance settings: {e}")
        
        # Get board components
        footprints = board.get_footprints()
        logger.info(f"✓ Retrieved {len(footprints)} footprints")
        
        tracks = board.get_tracks()
        logger.info(f"✓ Retrieved {len(tracks)} tracks")
        
        vias = board.get_vias()
        logger.info(f"✓ Retrieved {len(vias)} vias")
        
        pads = board.get_pads()
        logger.info(f"✓ Retrieved {len(pads)} pads")
        
        nets = board.get_nets()
        logger.info(f"✓ Retrieved {len(nets)} nets")
        
        # Get board outline and other geometry
        logger.info("Getting board outline...")
        
        # Get zones (copper pours, keepouts, etc.)
        zones = board.get_zones()
        logger.info(f"✓ Retrieved {len(zones)} zones")
        
        # Analyze zones for keepout areas and copper pours
        keepout_areas = []
        copper_pours = []
        
        for i, zone in enumerate(zones):
            try:
                zone_attrs = [attr for attr in dir(zone) if not attr.startswith('_')]
                logger.info(f"Zone {i} attributes: {', '.join(zone_attrs[:20])}")
                
                outline = zone.outline
                outline_attrs = [attr for attr in dir(outline) if not attr.startswith('_')]
                logger.info(f"Zone {i} outline attributes: {', '.join(outline_attrs)}")
                
                # Check filled polygons - this is where the actual copper pour with thermal reliefs is!
                filled_polygons = zone.filled_polygons
                logger.info(f"Zone {i} filled_polygons type: {type(filled_polygons)}")
                
                # Get zone outline points for fallback
                outline_points = []
                try:
                    outline_obj = outline.outline
                    for node in outline_obj.nodes:
                        point = node.point
                        outline_points.append((point.x, point.y))
                    logger.info(f"Zone {i} outline: {len(outline_points)} points")
                except Exception as e:
                    logger.debug(f"Error getting outline for zone {i}: {e}")
                
                # Extract actual filled polygons with thermal reliefs
                filled_polygon_data = {}
                
                # Process filled polygons for each layer
                for layer_id, polygons in filled_polygons.items():
                    logger.info(f"Zone {i} layer {layer_id}: {len(polygons)} filled polygons")
                    
                    layer_polygons = []
                    for j, polygon in enumerate(polygons):
                        try:
                            # Extract polygon points
                            polygon_points = []
                            if hasattr(polygon, 'nodes'):
                                for node in polygon.nodes:
                                    if hasattr(node, 'point'):
                                        point = node.point
                                        polygon_points.append((point.x, point.y))
                            elif hasattr(polygon, 'outline') and hasattr(polygon.outline, 'nodes'):
                                for node in polygon.outline.nodes:
                                    if hasattr(node, 'point'):
                                        point = node.point
                                        polygon_points.append((point.x, point.y))
                            
                            # Extract holes (thermal reliefs around pads, vias, etc.)
                            polygon_holes = []
                            if hasattr(polygon, 'holes'):
                                for hole in polygon.holes:
                                    hole_points = []
                                    if hasattr(hole, 'nodes'):
                                        for node in hole.nodes:
                                            if hasattr(node, 'point'):
                                                point = node.point
                                                hole_points.append((point.x, point.y))
                                    elif hasattr(hole, 'outline') and hasattr(hole.outline, 'nodes'):
                                        for node in hole.outline.nodes:
                                            if hasattr(node, 'point'):
                                                point = node.point
                                                hole_points.append((point.x, point.y))
                                    if hole_points:
                                        polygon_holes.append(hole_points)
                            
                            if polygon_points:
                                layer_polygons.append({
                                    'outline': polygon_points,
                                    'holes': polygon_holes
                                })
                                logger.info(f"  Polygon {j}: {len(polygon_points)} outline points, {len(polygon_holes)} holes")
                        
                        except Exception as e:
                            logger.debug(f"Error processing polygon {j} in zone {i}: {e}")
                    
                    if layer_polygons:
                        filled_polygon_data[layer_id] = layer_polygons
                
                # Determine zone type - improve classification logic
                # Check if zone has copper data (filled polygons) to determine if it's copper pour vs keepout
                has_filled_polygons = bool(filled_polygon_data)
                net = getattr(zone, 'net', None)
                has_net = net is not None
                is_rule_area = hasattr(zone, 'is_rule_area') and zone.is_rule_area
                
                # Copper zones have filled polygons, even if they're marked as rule areas
                if has_filled_polygons:
                    zone_info = f"Zone_{i}"
                    layers = getattr(zone, 'layers', [])
                    net_name = getattr(net, 'name', 'Unknown') if net else 'No Net'
                    logger.info(f"  Found copper pour: {zone_info} on {layers}, net: {net_name}")
                    logger.info(f"    Filled polygons on {len(filled_polygon_data)} layers")
                    
                    copper_pours.append({
                        'name': zone_info,
                        'layers': layers,
                        'net': net,
                        'net_name': net_name,
                        'outline': outline_points,
                        'filled_polygons': filled_polygon_data  # This contains the actual copper with thermal reliefs!
                    })
                elif is_rule_area:
                    zone_info = f"Zone_{i}"
                    layers = getattr(zone, 'layers', [])
                    logger.info(f"  Found keepout area: {zone_info} on {layers}")
                    keepout_areas.append({
                        'name': zone_info,
                        'layers': layers,
                        'outline': outline_points
                    })
                else:
                    zone_info = f"Zone_{i}"
                    layers = getattr(zone, 'layers', [])
                    net_name = getattr(net, 'name', 'Unknown') if net else 'No Net'
                    logger.info(f"  Found zone (no fill): {zone_info} on {layers}, net: {net_name}")
                    
                    copper_pours.append({
                        'name': zone_info,
                        'layers': layers,
                        'net': net,
                        'net_name': net_name,
                        'outline': outline_points,
                        'filled_polygons': filled_polygon_data  # Empty but structure preserved
                    })
                    
            except Exception as e:
                logger.warning(f"Error analyzing zone {i}: {e}")
                continue
        
        logger.info(f"✓ Parsed {len(copper_pours)} zones and {len(keepout_areas)} keepout areas")
        
        # Get layer information
        logger.info("Getting layer information...")
        
        # Get detailed footprint information
        logger.info("Getting detailed footprint information...")
        
        footprint_details = []
        for footprint in footprints:
            try:
                footprint_info = {
                    'reference': getattr(footprint, 'reference', 'Unknown'),
                    'value': getattr(footprint, 'value', ''),
                    'layer': getattr(footprint, 'layer', 0),
                    'position': getattr(footprint, 'position', {'x': 0, 'y': 0}),
                    'orientation': getattr(footprint, 'orientation', 0),
                    'library': getattr(footprint, 'library', ''),
                    'footprint_name': getattr(footprint, 'footprint_name', '')
                }
                footprint_details.append(footprint_info)
            except Exception as e:
                logger.debug(f"Error getting footprint details: {e}")
        
        logger.info(f"✓ Retrieved detailed info for {len(footprint_details)} footprints")
        
        # Success message
        logger.info("🎉 SUCCESS: Board data retrieved via KiCad IPC API!")
        logger.info(f"Board: {board_name}")
        logger.info(f"Components: {len(footprints)}")
        logger.info(f"Tracks: {len(tracks)}")
        logger.info(f"Vias: {len(vias)}")
        logger.info(f"Pads: {len(pads)}")
        logger.info(f"Nets: {len(nets)}")
        
        return {
            'board_name': board_name,
            'footprints': footprints,
            'tracks': tracks,
            'vias': vias,
            'pads': pads,
            'nets': nets,
            'zones': zones,
            'keepout_areas': keepout_areas,
            'copper_pours': copper_pours,
            'footprint_details': footprint_details,
            'kicad_theme_colors': kicad_theme_colors
        }
        
    except Exception as e:
        logger.error(f"❌ Board analysis failed: {e}")
        return None

def launch_qt_interface(board_data: Dict[str, Any], kicad_interface) -> None:
    """Launch the Qt interface with board data"""
    try:
        logger.info("Launching Qt UI with safety checks...")
        
        # Add src directory to path for imports
        src_dir = Path(__file__).parent / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            logger.info(f"Added src dir to Python path: {src_dir}")
        
        # Import Qt modules with error handling
        logger.info("Importing PyQt6...")
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QIcon, QPixmap
        logger.info("PyQt6 imported successfully")
        
        # Import our main window
        logger.info("Importing OrthoRouteWindow...")
        from orthoroute_window import OrthoRouteWindow
        logger.info("OrthoRouteWindow imported successfully")
        
        # Create QApplication
        logger.info("Creating QApplication...")
        app = QApplication(sys.argv)
        
        # Set application icon if available
        icon_path = Path(__file__).parent / "assets" / "BigIcon.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            logger.info(f"App icon set: {icon_path}")
        
        # Convert board data to format expected by the window
        logger.info("Converting board data for fast window launch...")
        start_time = time.time()
        
        # Convert to the format expected by the GUI (fast visual elements only)
        gui_board_data = convert_board_data_for_gui(board_data)
        
        conversion_time = time.time() - start_time
        logger.info(f"Fast GUI conversion completed in {conversion_time:.2f} seconds")
        logger.info(f"Board data ready: {len(board_data.get('footprints', []))} components, {len(board_data.get('tracks', []))} tracks")
        logger.info("Net connectivity analysis will be processed progressively in background")
        
        # Create and show main window
        logger.info("Creating main window...")
        window = OrthoRouteWindow(gui_board_data, kicad_interface)
        
        logger.info("Showing window...")
        window.show()
        
        logger.info("Qt UI launched successfully - entering event loop")
        app.exec()
        
    except Exception as e:
        logger.error(f"❌ Qt interface launch failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Fallback: print board data
        print("🎉 OrthoRoute Board Analysis Complete!")
        print(f"Board: {board_data.get('board_name', 'Unknown')}")
        print(f"Components: {len(board_data.get('footprints', []))}")
        print(f"Tracks: {len(board_data.get('tracks', []))}")
        print(f"Nets: {len(board_data.get('nets', []))}")

def convert_board_data_for_gui(board_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert IPC board data to format expected by GUI"""
    try:
        footprints = board_data.get('footprints', [])
        tracks = board_data.get('tracks', [])
        pads = board_data.get('pads', [])
        nets = board_data.get('nets', [])
        vias = board_data.get('vias', [])
        zones = board_data.get('zones', [])
        keepout_areas = board_data.get('keepout_areas', [])
        copper_pours = board_data.get('copper_pours', [])
        
        # Convert components (footprints)
        components = []
        for fp in footprints:
            try:
                component = {
                    'reference': getattr(fp, 'reference', 'Unknown'),
                    'value': getattr(fp, 'value', ''),
                    'x': getattr(getattr(fp, 'position', {}), 'x', 0) / 1000000,  # Convert to mm
                    'y': getattr(getattr(fp, 'position', {}), 'y', 0) / 1000000,
                    'rotation': getattr(fp, 'orientation', 0),
                    'layer': 'F.Cu',  # Default to front copper
                    'library': getattr(fp, 'library', ''),
                    'footprint': getattr(fp, 'footprint_name', ''),
                    'bounding_box': getattr(fp, 'bounding_box', None)
                }
                components.append(component)
            except Exception as e:
                logger.debug(f"Error converting component: {e}")
        
        # Convert tracks
        gui_tracks = []
        for track in tracks:
            try:
                track_data = {
                    'start_x': getattr(getattr(track, 'start', {}), 'x', 0) / 1000000,
                    'start_y': getattr(getattr(track, 'start', {}), 'y', 0) / 1000000,
                    'end_x': getattr(getattr(track, 'end', {}), 'x', 0) / 1000000,
                    'end_y': getattr(getattr(track, 'end', {}), 'y', 0) / 1000000,
                    'width': getattr(track, 'width', 0.2),
                    'layer': getattr(track, 'layer', 0),
                    'net': getattr(track, 'net', None)
                }
                gui_tracks.append(track_data)
            except Exception as e:
                logger.debug(f"Error converting track: {e}")
        
        # Convert vias
        gui_vias = []
        for via in vias:
            try:
                via_data = {
                    'x': getattr(getattr(via, 'position', {}), 'x', 0) / 1000000,
                    'y': getattr(getattr(via, 'position', {}), 'y', 0) / 1000000,
                    'drill_diameter': getattr(via, 'drill_diameter', 0.3) / 1000000,
                    'via_diameter': getattr(via, 'via_diameter', 0.6) / 1000000,
                    'start_layer': getattr(via, 'start_layer', 0),
                    'end_layer': getattr(via, 'end_layer', 31),
                    'net': getattr(via, 'net', None)
                }
                gui_vias.append(via_data)
            except Exception as e:
                logger.debug(f"Error converting via: {e}")
        
        # Create comprehensive footprint lookup system - map each footprint instance to its definition
        footprint_instances = {}  # Maps footprint reference to footprint object and definition
        footprint_pad_definitions = {}  # Maps footprint definition ID to pad definitions
        
        logger.info(f"Processing {len(footprints)} footprints for instance mapping...")
        
        for footprint in footprints:
            try:
                # Get footprint instance info - try multiple reference attributes
                reference = None
                for ref_attr in ['reference', 'ref', 'designator', 'id']:
                    if hasattr(footprint, ref_attr):
                        ref_value = getattr(footprint, ref_attr, None)
                        if ref_value and ref_value != 'Unknown':
                            reference = str(ref_value)  # Convert to string to handle KIID objects
                            break
                
                # If no reference found, create a unique one based on position
                if not reference or reference == 'Unknown':
                    position = getattr(footprint, 'position', None)
                    if position:
                        pos_x = getattr(position, 'x', 0) / 1000000
                        pos_y = getattr(position, 'y', 0) / 1000000
                        reference = f"FP_{pos_x:.2f}_{pos_y:.2f}"
                    else:
                        reference = f"FP_{len(footprint_instances)}"
                
                position = getattr(footprint, 'position', None)
                orientation = getattr(footprint, 'orientation', 0)
                layer = getattr(footprint, 'layer', 0)
                
                if len(footprint_instances) < 5:  # Debug first few
                    logger.info(f"Processing footprint '{reference}' at position {getattr(position, 'x', 0)/1000000 if position else 'None'}, {getattr(position, 'y', 0)/1000000 if position else 'None'}")
                
                # Get footprint definition
                definition = getattr(footprint, 'definition', None)
                if definition:
                    footprint_id = getattr(definition, 'id', 'unknown')
                    
                    if len(footprint_instances) < 5:  # Debug first few
                        logger.info(f"  Found definition with ID: '{footprint_id}'")
                    
                    # Store footprint instance data
                    footprint_instances[reference] = {
                        'footprint': footprint,
                        'definition': definition,
                        'definition_id': footprint_id,
                        'position': position,
                        'orientation': orientation,
                        'layer': layer
                    }
                    
                    # Process definition pads if not already done
                    if footprint_id not in footprint_pad_definitions and hasattr(definition, 'pads'):
                        def_pads = getattr(definition, 'pads', [])
                        
                        if len(footprint_instances) < 5:  # Debug first few
                            logger.info(f"  Processing {len(def_pads)} definition pads for {footprint_id}")
                        
                        # Create comprehensive pad definition mapping
                        pad_definitions = {}
                        for def_pad in def_pads:
                            pad_number = getattr(def_pad, 'number', '')
                            pad_position = getattr(def_pad, 'position', None)
                            
                            # Store by pad number (primary key)
                            if pad_number:
                                pad_definitions[pad_number] = def_pad
                            
                            # Also store by relative position within footprint for backup matching
                            if pad_position:
                                rel_x = getattr(pad_position, 'x', 0)
                                rel_y = getattr(pad_position, 'y', 0)
                                pos_key = f"pos_{rel_x}_{rel_y}"
                                pad_definitions[pos_key] = def_pad
                        
                        footprint_pad_definitions[footprint_id] = {
                            'pads': pad_definitions,
                            'definition': definition,
                            'pad_count': len(def_pads)
                        }
                        
                        if len(footprint_pad_definitions) < 5:  # Log first few for debugging
                            logger.info(f"Footprint definition '{footprint_id}': {len(def_pads)} pads mapped")
                else:
                    if len(footprint_instances) < 5:
                        logger.info(f"  No definition found for footprint '{reference}'")
                            
            except Exception as e:
                logger.info(f"Error processing footprint {reference}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"Created footprint instance lookup for {len(footprint_instances)} footprints")
        logger.info(f"Created pad definition lookup for {len(footprint_pad_definitions)} unique footprint types")

        # Convert pads with enhanced footprint definition lookup
        gui_pads = []
        for i, pad in enumerate(pads):
            try:
                # Debug the first few pads to understand the structure
                if i < 5:
                    logger.info(f"Debug pad {i} attributes: {[attr for attr in dir(pad) if not attr.startswith('_')]}")
                    pos = getattr(pad, 'position', None)
                    if pos:
                        logger.info(f"  position type: {type(pos)}, attrs: {[attr for attr in dir(pos) if not attr.startswith('_')]}")
                    padstack = getattr(pad, 'padstack', None)
                    if padstack:
                        logger.info(f"  padstack type: {type(padstack)}, attrs: {[attr for attr in dir(padstack) if not attr.startswith('_')]}")
                        
                        # Check for drill-related attributes more thoroughly
                        drill_attrs = [attr for attr in dir(padstack) if 'drill' in attr.lower() or 'hole' in attr.lower()]
                        if drill_attrs:
                            logger.info(f"  drill-related attributes: {drill_attrs}")
                            for attr in drill_attrs:
                                try:
                                    value = getattr(padstack, attr)
                                    logger.info(f"    {attr}: {value} (type: {type(value)})")
                                except Exception as e:
                                    logger.info(f"    {attr}: error getting value - {e}")
                        
                        # Check for size-related attributes
                        size_attrs = [attr for attr in dir(padstack) if 'size' in attr.lower()]
                        if size_attrs:
                            logger.info(f"  size-related attributes: {size_attrs}")
                        
                        # Check copper layers in detail
                        if hasattr(padstack, 'copper_layers'):
                            copper_layers = getattr(padstack, 'copper_layers')
                            logger.info(f"  copper_layers type: {type(copper_layers)}")
                            if hasattr(copper_layers, '__len__'):
                                logger.info(f"  copper_layers length: {len(copper_layers)}")
                            if copper_layers and hasattr(copper_layers, '__iter__'):
                                try:
                                    first_layer = list(copper_layers)[0] if hasattr(copper_layers, '__iter__') else copper_layers
                                    logger.info(f"  first copper layer type: {type(first_layer)}")
                                    layer_attrs = [attr for attr in dir(first_layer) if not attr.startswith('_')]
                                    logger.info(f"  first layer attrs: {layer_attrs}")
                                except Exception as e:
                                    logger.info(f"  error accessing first copper layer: {e}")
                
                # Extract pad position (absolute position on board)
                position = getattr(pad, 'position', None)
                pad_x = getattr(position, 'x', 0) / 1000000 if position else 0
                pad_y = getattr(position, 'y', 0) / 1000000 if position else 0
                
                # CRITICAL: Find the parent footprint for this specific pad
                parent_footprint = None
                parent_definition = None
                definition_pad = None
                pad_number = getattr(pad, 'number', '')
                
                # Method 1: Direct parent reference (if available in API)
                if hasattr(pad, 'parent') or hasattr(pad, 'footprint') or hasattr(pad, 'owner'):
                    for attr_name in ['parent', 'footprint', 'owner']:
                        if hasattr(pad, attr_name):
                            parent_ref = getattr(pad, attr_name)
                            if parent_ref:
                                # Try to find this in our footprint instances
                                for ref, fp_data in footprint_instances.items():
                                    if fp_data['footprint'] == parent_ref:
                                        parent_footprint = fp_data
                                        parent_definition = fp_data['definition']
                                        if i < 3:
                                            logger.info(f"  Found parent footprint '{ref}' via {attr_name}")
                                        break
                                if parent_footprint:
                                    break
                
                # Method 2: Geometric proximity matching (find closest footprint)
                if not parent_footprint and position:
                    min_distance = float('inf')
                    closest_footprint = None
                    
                    for ref, fp_data in footprint_instances.items():
                        fp_pos = fp_data['position']
                        if fp_pos:
                            fp_x = getattr(fp_pos, 'x', 0) / 1000000
                            fp_y = getattr(fp_pos, 'y', 0) / 1000000
                            
                            # Calculate distance from pad to footprint center
                            distance = ((pad_x - fp_x) ** 2 + (pad_y - fp_y) ** 2) ** 0.5
                            
                            # Check if this pad could belong to this footprint (reasonable distance)
                            if distance < min_distance and distance < 20:  # 20mm max reasonable distance
                                # Verify this footprint actually has a pad with this number
                                def_id = fp_data['definition_id']
                                if def_id in footprint_pad_definitions:
                                    pad_defs = footprint_pad_definitions[def_id]['pads']
                                    if pad_number and pad_number in pad_defs:
                                        min_distance = distance
                                        closest_footprint = fp_data
                    
                    if closest_footprint:
                        parent_footprint = closest_footprint
                        parent_definition = closest_footprint['definition']
                        if i < 3:
                            logger.info(f"  Found parent footprint '{closest_footprint}' via proximity ({min_distance:.2f}mm)")
                
                # Method 3: Find definition pad within the parent footprint
                if parent_footprint and parent_definition:
                    def_id = parent_footprint['definition_id']
                    if def_id in footprint_pad_definitions:
                        pad_defs = footprint_pad_definitions[def_id]['pads']
                        
                        # Try to find by pad number first
                        if pad_number and pad_number in pad_defs:
                            definition_pad = pad_defs[pad_number]
                            if i < 3:
                                logger.info(f"  Found definition pad by number '{pad_number}' in footprint type '{def_id}'")
                        
                        # Fallback: try to find by relative position within footprint
                        elif parent_footprint['position']:
                            fp_pos = parent_footprint['position']
                            fp_x = getattr(fp_pos, 'x', 0)
                            fp_y = getattr(fp_pos, 'y', 0)
                            
                            # Calculate relative position within footprint
                            rel_x = getattr(position, 'x', 0) - fp_x
                            rel_y = getattr(position, 'y', 0) - fp_y
                            
                            # Try to find pad at this relative position
                            pos_key = f"pos_{rel_x}_{rel_y}"
                            if pos_key in pad_defs:
                                definition_pad = pad_defs[pos_key]
                                if i < 3:
                                    logger.info(f"  Found definition pad by relative position in footprint type '{def_id}'")
                
                # Extract comprehensive pad properties from parent footprint
                size_x = 1.0  # Default size in mm
                size_y = 1.0
                drill_diameter = 0.0
                shape = 1  # Default to rectangle (0=circle, 1=rectangle, 2=oval)
                pad_layers = []
                pad_type = 'SMD'  # SMD, PTH, NPTH
                
                # Extract all footprint-specific properties
                parent_reference = None
                if parent_footprint:
                    # Find the reference (key) that maps to this footprint data
                    for ref, fp_data in footprint_instances.items():
                        if fp_data == parent_footprint:
                            parent_reference = ref
                            break
                
                footprint_properties = {
                    'parent_reference': parent_reference,
                    'footprint_type': parent_footprint['definition_id'] if parent_footprint else None,
                    'footprint_layer': parent_footprint['layer'] if parent_footprint else None,
                    'footprint_orientation': parent_footprint['orientation'] if parent_footprint else 0,
                    'pad_number': pad_number,
                    'relative_position': None
                }
                
                if parent_footprint and parent_footprint['position'] and position:
                    # Calculate pad position relative to footprint
                    fp_pos = parent_footprint['position']
                    fp_x = getattr(fp_pos, 'x', 0) / 1000000
                    fp_y = getattr(fp_pos, 'y', 0) / 1000000
                    footprint_properties['relative_position'] = {
                        'x': pad_x - fp_x,
                        'y': pad_y - fp_y
                    }
                
                
                # PRIORITY 1: Extract ALL properties from footprint definition pad (most accurate)
                if definition_pad:
                    definition_padstack = getattr(definition_pad, 'padstack', None)
                    if definition_padstack:
                        if i < 3:
                            logger.info(f"  Using footprint definition padstack for accurate properties")
                        
                        # Get comprehensive pad properties from definition
                        if hasattr(definition_padstack, 'copper_layers'):
                            def_copper_layers = definition_padstack.copper_layers
                            if def_copper_layers and len(def_copper_layers) > 0:
                                def_first_layer = def_copper_layers[0] if isinstance(def_copper_layers, list) else def_copper_layers
                                
                                # Extract accurate size from footprint definition
                                if hasattr(def_first_layer, 'size'):
                                    size_obj = def_first_layer.size
                                    if hasattr(size_obj, 'x') and hasattr(size_obj, 'y'):
                                        size_x = float(getattr(size_obj, 'x', 1000000)) / 1000000
                                        size_y = float(getattr(size_obj, 'y', 1000000)) / 1000000
                                        if i < 3:
                                            logger.info(f"  Definition size: ({size_x:.3f}, {size_y:.3f})")
                                    elif isinstance(size_obj, (int, float)):
                                        size_x = size_y = float(size_obj) / 1000000
                                        if i < 3:
                                            logger.info(f"  Definition circular size: {size_x:.3f}")
                                
                                # Extract accurate shape from footprint definition
                                if hasattr(def_first_layer, 'shape'):
                                    shape_raw = getattr(def_first_layer, 'shape', 1)
                                    if isinstance(shape_raw, int):
                                        shape = shape_raw
                                        if i < 3:
                                            shape_names = {
                                                0: 'circle', 1: 'rectangle', 2: 'oval', 3: 'trapezoid', 
                                                4: 'roundrect', 5: 'chamfered_rect', 6: 'custom'
                                            }
                                            logger.info(f"  Definition shape: {shape} ({shape_names.get(shape, 'unknown')})")
                                
                                # Extract layer information
                                if hasattr(def_first_layer, 'layer'):
                                    layer_info = getattr(def_first_layer, 'layer', None)
                                    if layer_info:
                                        pad_layers.append(layer_info)
                        
                        # Extract drill information from definition
                        if hasattr(definition_padstack, 'drill'):
                            def_drill = getattr(definition_padstack, 'drill')
                            if def_drill:
                                if hasattr(def_drill, 'diameter'):
                                    drill_raw = getattr(def_drill, 'diameter')
                                    if hasattr(drill_raw, 'x') and hasattr(drill_raw, 'y'):
                                        drill_x = float(getattr(drill_raw, 'x', 0)) / 1000000
                                        drill_y = float(getattr(drill_raw, 'y', 0)) / 1000000
                                        drill_diameter = min(drill_x, drill_y) if drill_x > 0 and drill_y > 0 else max(drill_x, drill_y)
                                        if drill_diameter > 0:
                                            pad_type = 'PTH'  # Through-hole
                                        if i < 3:
                                            logger.info(f"  Definition drill: {drill_diameter:.3f}mm (PTH)")
                                    elif isinstance(drill_raw, (int, float)) and drill_raw > 0:
                                        drill_diameter = float(drill_raw) / 1000000
                                        if drill_diameter > 0:
                                            pad_type = 'PTH'  # Through-hole
                                        if i < 3:
                                            logger.info(f"  Definition drill: {drill_diameter:.3f}mm (PTH)")
                                else:
                                    # No drill = SMD pad
                                    pad_type = 'SMD'
                                    if i < 3:
                                        logger.info(f"  No drill = SMD pad")
                        else:
                            # No drill object = SMD pad
                            pad_type = 'SMD'
                            if i < 3:
                                logger.info(f"  No drill object = SMD pad")
                        
                        # Extract pad type from definition
                        if hasattr(definition_pad, 'pad_type'):
                            pad_type_raw = getattr(definition_pad, 'pad_type', None)
                            if pad_type_raw:
                                pad_type_str = str(pad_type_raw).upper()
                                if 'PTH' in pad_type_str:
                                    pad_type = 'PTH'
                                elif 'SMD' in pad_type_str:
                                    pad_type = 'SMD'
                                elif 'NPTH' in pad_type_str:
                                    pad_type = 'NPTH'
                                if i < 3:
                                    logger.info(f"  Definition pad type: {pad_type}")
                        
                        # Extract additional properties from definition padstack
                        additional_properties = {}
                        for prop in ['angle', 'is_masked', 'zone_settings']:
                            if hasattr(definition_padstack, prop):
                                additional_properties[prop] = getattr(definition_padstack, prop)
                        
                        footprint_properties['definition_properties'] = additional_properties
                
                # PRIORITY 2: Fall back to instance padstack if definition not available
                padstack = getattr(pad, 'padstack', None)
                if padstack and (not definition_pad or size_x <= 0):
                    # [Previous fallback logic for instance padstack - keeping existing code]
                    # Enhanced drill diameter detection - try multiple approaches
                    drill_found = False
                    
                    # Method 1: Direct drill_diameter attribute
                    if hasattr(padstack, 'drill_diameter'):
                        drill_raw = getattr(padstack, 'drill_diameter', 0)
                        if drill_raw and drill_raw > 0:
                            drill_diameter = float(drill_raw) / 1000000  # Convert from nanometers to mm
                            drill_found = True
                            if i < 3:
                                logger.info(f"  Found drill via drill_diameter: {drill_diameter:.3f}mm (raw: {drill_raw})")
                    
                    # [Rest of the existing fallback logic...]
                    # Try to get size and shape from padstack copper layers if not from definition
                    if hasattr(padstack, 'copper_layers') and size_x <= 0:
                        copper_layers = padstack.copper_layers
                        if copper_layers and len(copper_layers) > 0:
                            first_layer = copper_layers[0] if isinstance(copper_layers, list) else copper_layers
                            if hasattr(first_layer, 'size'):
                                size_obj = first_layer.size
                                if hasattr(size_obj, 'x') and hasattr(size_obj, 'y'):
                                    size_x = float(getattr(size_obj, 'x', 1000000)) / 1000000
                                    size_y = float(getattr(size_obj, 'y', 1000000)) / 1000000
                                    if i < 3:
                                        logger.info(f"  Found size from instance copper layer: ({size_x:.3f}, {size_y:.3f})")
                
                # Use default minimum size if still zero
                if size_x <= 0:
                    size_x = 1.0  # 1mm default
                if size_y <= 0:
                    size_y = 1.0  # 1mm default
                
                # Create comprehensive pad data with all footprint properties
                pad_data = {
                    # Basic pad geometry
                    'x': pad_x,
                    'y': pad_y,
                    'size_x': size_x,
                    'size_y': size_y,
                    'shape': shape,
                    'drill_diameter': drill_diameter,
                    
                    # Pad identification
                    'pad_name': pad_number,
                    'pad_type': pad_type,  # SMD, PTH, NPTH
                    'layers': getattr(pad, 'layers', pad_layers),
                    'net': getattr(pad, 'net', None),
                    
                    # Comprehensive footprint properties
                    'footprint_reference': footprint_properties['parent_reference'],
                    'footprint_type': footprint_properties['footprint_type'],
                    'footprint_layer': footprint_properties['footprint_layer'],
                    'footprint_orientation': footprint_properties['footprint_orientation'],
                    'relative_position': footprint_properties['relative_position'],
                    
                    # Additional properties from definition
                    'definition_properties': footprint_properties.get('definition_properties', {}),
                    
                    # Source information for debugging
                    'data_source': 'footprint_definition' if definition_pad else 'instance_padstack'
                }
                
                if i < 3:
                    fp_ref = footprint_properties['parent_reference']
                    fp_type = footprint_properties['footprint_type']
                    logger.info(f"  Converted pad {i}: pos=({pad_x:.2f},{pad_y:.2f}), size=({size_x:.3f},{size_y:.3f}), drill={drill_diameter:.3f}")
                    logger.info(f"    Footprint: {fp_ref} (type: {fp_type}), pad_type: {pad_type}, source: {pad_data['data_source']}")
                
                gui_pads.append(pad_data)
            except Exception as e:
                logger.error(f"Error converting pad {i}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
        
        logger.info(f"Successfully converted {len(gui_pads)} pads from {len(pads)} original pads")
        
        # Convert zones (copper pours and keepouts) - enhanced for filled polygons with thermal reliefs
        gui_zones = []
        for zone in zones:
            try:
                zone_data = {
                    'zone_type': 'copper_pour' if not getattr(zone, 'is_rule_area', False) else 'keepout',
                    'net': getattr(zone, 'net', None),
                    'layers': getattr(zone, 'layers', []),
                    'outline': [],
                    'filled_polygons': {},  # Will be properly processed
                    'priority': getattr(zone, 'priority', 0)
                }
                
                # Get zone outline points using the correct API structure
                outline_obj = getattr(zone, 'outline', None)
                if outline_obj:
                    try:
                        # Access the outline points correctly
                        if hasattr(outline_obj, 'outline'):
                            actual_outline = outline_obj.outline
                            if hasattr(actual_outline, 'nodes'):
                                for node in actual_outline.nodes:
                                    if hasattr(node, 'point'):
                                        point = node.point
                                        zone_data['outline'].append({
                                            'x': float(getattr(point, 'x', 0)) / 1000000,
                                            'y': float(getattr(point, 'y', 0)) / 1000000
                                        })
                    except Exception as e:
                        logger.debug(f"Error getting zone outline: {e}")
                
                # Process filled polygons with thermal reliefs
                filled_polygons_raw = getattr(zone, 'filled_polygons', {})
                if filled_polygons_raw:
                    for layer_id, polygons in filled_polygons_raw.items():
                        layer_polygons = []
                        for polygon in polygons:
                            try:
                                # Extract polygon outline
                                polygon_outline = []
                                if hasattr(polygon, 'nodes'):
                                    for node in polygon.nodes:
                                        if hasattr(node, 'point'):
                                            point = node.point
                                            polygon_outline.append({
                                                'x': float(getattr(point, 'x', 0)) / 1000000,
                                                'y': float(getattr(point, 'y', 0)) / 1000000
                                            })
                                elif hasattr(polygon, 'outline') and hasattr(polygon.outline, 'nodes'):
                                    for node in polygon.outline.nodes:
                                        if hasattr(node, 'point'):
                                            point = node.point
                                            polygon_outline.append({
                                                'x': float(getattr(point, 'x', 0)) / 1000000,
                                                'y': float(getattr(point, 'y', 0)) / 1000000
                                            })
                                
                                # Extract holes (thermal reliefs around pads)
                                polygon_holes = []
                                if hasattr(polygon, 'holes'):
                                    for hole in polygon.holes:
                                        hole_outline = []
                                        if hasattr(hole, 'nodes'):
                                            for node in hole.nodes:
                                                if hasattr(node, 'point'):
                                                    point = node.point
                                                    hole_outline.append({
                                                        'x': float(getattr(point, 'x', 0)) / 1000000,
                                                        'y': float(getattr(point, 'y', 0)) / 1000000
                                                    })
                                        elif hasattr(hole, 'outline') and hasattr(hole.outline, 'nodes'):
                                            for node in hole.outline.nodes:
                                                if hasattr(node, 'point'):
                                                    point = node.point
                                                    hole_outline.append({
                                                        'x': float(getattr(point, 'x', 0)) / 1000000,
                                                        'y': float(getattr(point, 'y', 0)) / 1000000
                                                    })
                                        if hole_outline:
                                            polygon_holes.append(hole_outline)
                                
                                if polygon_outline:
                                    layer_polygons.append({
                                        'outline': polygon_outline,
                                        'holes': polygon_holes  # Thermal reliefs around pads!
                                    })
                            
                            except Exception as e:
                                logger.debug(f"Error processing polygon in zone: {e}")
                        
                        if layer_polygons:
                            zone_data['filled_polygons'][layer_id] = layer_polygons
                
                # Fallback: if no filled polygons but have outline, use zone outline
                if not zone_data['filled_polygons'] and zone_data['outline']:
                    # If this is a copper pour, try to get outline from first filled polygon
                    if filled_polygons_raw and not getattr(zone, 'is_rule_area', False):
                        for layer_id, polygons in filled_polygons_raw.items():
                            if polygons and len(polygons) > 0:
                                polygon = polygons[0]  # Use first polygon
                                if hasattr(polygon, 'nodes'):
                                    for node in polygon.nodes:
                                        if hasattr(node, 'point'):
                                            point = node.point
                                            zone_data['outline'].append({
                                                'x': float(getattr(point, 'x', 0)) / 1000000,
                                                'y': float(getattr(point, 'y', 0)) / 1000000
                                            })
                                break  # Only use first polygon for outline
                
                gui_zones.append(zone_data)
            except Exception as e:
                logger.debug(f"Error converting zone: {e}")
        
        # Convert keepout areas from the parsed data
        gui_keepouts = []
        for keepout in keepout_areas:
            try:
                keepout_data = {
                    'layers': keepout.get('layers', []),
                    'outline': []
                }
                
                # Convert outline points from the parsed format
                outline_points = keepout.get('outline', [])
                for point in outline_points:
                    if isinstance(point, (tuple, list)) and len(point) >= 2:
                        keepout_data['outline'].append({
                            'x': float(point[0]) / 1000000,
                            'y': float(point[1]) / 1000000
                        })
                
                gui_keepouts.append(keepout_data)
            except Exception as e:
                logger.debug(f"Error converting keepout: {e}")
        
        # Convert copper pours from the enhanced parsed data with thermal reliefs
        gui_copper_pours = []
        for copper_pour in copper_pours:
            try:
                copper_data = {
                    'name': copper_pour.get('name', 'Unknown'),
                    'layers': copper_pour.get('layers', []),
                    'net': copper_pour.get('net'),
                    'net_name': copper_pour.get('net_name', 'Unknown'),
                    'outline': [],
                    'filled_polygons': {}  # Enhanced with thermal reliefs
                }
                
                # Convert outline points
                outline_points = copper_pour.get('outline', [])
                for point in outline_points:
                    if isinstance(point, (tuple, list)) and len(point) >= 2:
                        copper_data['outline'].append({
                            'x': float(point[0]) / 1000000,
                            'y': float(point[1]) / 1000000
                        })
                
                # Convert filled polygons with holes (thermal reliefs)
                filled_polygons = copper_pour.get('filled_polygons', {})
                for layer_id, layer_polygons in filled_polygons.items():
                    converted_polygons = []
                    for polygon_data in layer_polygons:
                        converted_polygon = {
                            'outline': [],
                            'holes': []
                        }
                        
                        # Convert outline points
                        outline = polygon_data.get('outline', [])
                        for point in outline:
                            if isinstance(point, (tuple, list)) and len(point) >= 2:
                                converted_polygon['outline'].append({
                                    'x': float(point[0]) / 1000000,
                                    'y': float(point[1]) / 1000000
                                })
                        
                        # Convert holes (thermal reliefs around pads)
                        holes = polygon_data.get('holes', [])
                        for hole in holes:
                            converted_hole = []
                            for point in hole:
                                if isinstance(point, (tuple, list)) and len(point) >= 2:
                                    converted_hole.append({
                                        'x': float(point[0]) / 1000000,
                                        'y': float(point[1]) / 1000000
                                    })
                            if converted_hole:
                                converted_polygon['holes'].append(converted_hole)
                        
                        if converted_polygon['outline']:
                            converted_polygons.append(converted_polygon)
                    
                    if converted_polygons:
                        copper_data['filled_polygons'][layer_id] = converted_polygons
                
                gui_copper_pours.append(copper_data)
            except Exception as e:
                logger.debug(f"Error converting copper pour: {e}")
        
        # Calculate board bounds including all elements
        all_x = []
        all_y = []
        
        for comp in components:
            all_x.extend([comp['x']])
            all_y.extend([comp['y']])
        
        for pad in gui_pads:
            all_x.extend([pad['x']])
            all_y.extend([pad['y']])
        
        for via in gui_vias:
            all_x.extend([via['x']])
            all_y.extend([via['y']])
        
        for zone in gui_zones:
            for point in zone['outline']:
                all_x.extend([point['x']])
                all_y.extend([point['y']])
        
        for keepout in gui_keepouts:
            for point in keepout['outline']:
                all_x.extend([point['x']])
                all_y.extend([point['y']])
        
        if all_x and all_y:
            bounds = (min(all_x) - 5, min(all_y) - 5, max(all_x) + 5, max(all_y) + 5)
        else:
            bounds = (0, 0, 100, 100)
        
        # Fast GUI data for immediate window display (no net analysis)
        gui_data = {
            'components': components,
            'tracks': gui_tracks,
            'vias': gui_vias,
            'pads': gui_pads,
            'zones': gui_zones,
            'keepouts': gui_keepouts,
            'copper_pours': gui_copper_pours,  # Enhanced copper pours with thermal reliefs
            'nets': {},  # Empty initially - will be populated progressively
            'airwires': [],  # Empty initially - will be populated progressively
            'bounds': bounds,
            'layers': ['F.Cu', 'B.Cu', 'F.Mask', 'B.Mask'],
            'kicad_theme_colors': board_data.get('kicad_theme_colors', {}),
            # Store raw data for progressive processing
            '_raw_nets': nets,
            '_raw_pads': pads,
            '_gui_tracks': gui_tracks
        }
        
        logger.info(f"Fast GUI data ready: {len(components)} components, {len(gui_tracks)} tracks")
        logger.info(f"  Vias: {len(gui_vias)}, Pads: {len(gui_pads)}")
        logger.info(f"  Zones: {len(gui_zones)}, Keepouts: {len(gui_keepouts)}")
        logger.info(f"  Copper pours: {len(gui_copper_pours)} with thermal reliefs")
        logger.info(f"Nets and airwires will be processed progressively...")
        
        return gui_data
        
    except Exception as e:
        logger.error(f"Error converting board data: {e}")
        return {
            'components': [],
            'tracks': [],
            'nets': {},
            'airwires': [],
            'bounds': (0, 0, 100, 100),
            'layers': ['F.Cu', 'B.Cu']
        }

def process_nets_progressively(gui_data, progress_callback=None):
    """Process nets and airwires progressively in batches"""
    try:
        nets = gui_data.get('_raw_nets', [])
        pads = gui_data.get('_raw_pads', [])
        gui_tracks = gui_data.get('_gui_tracks', [])
        
        net_info = {}
        net_pins = {}
        airwires = []
        
        logger.info(f"Starting progressive net processing: {len(nets)} nets to process")
        
        # Process nets in batches
        batch_size = 50
        for batch_start in range(0, len(nets), batch_size):
            batch_end = min(batch_start + batch_size, len(nets))
            batch_nets = nets[batch_start:batch_end]
            
            # Only log every 10th batch to reduce terminal spam
            batch_num = batch_start // batch_size + 1
            if batch_num % 10 == 1 or batch_num <= 5:
                logger.info(f"Processing net batch {batch_num}: nets {batch_start}-{batch_end}")
            elif batch_num % 50 == 0:
                logger.info(f"Processing net batch {batch_num}: nets {batch_start}-{batch_end} (milestone)")
            
            # Process each net in the batch
            for net in batch_nets:
                try:
                    net_name = getattr(net, 'name', f'Net_{id(net)}')
                    # Count pads on this net
                    net_pads = [pad for pad in pads if hasattr(pad, 'net') and getattr(pad, 'net', None) == net]
                    
                    net_info[net_name] = {
                        'name': net_name,
                        'pin_count': len(net_pads),
                        'routed': len([t for t in gui_tracks if getattr(t.get('net', None), 'name', '') == net_name]) > 0
                    }
                    
                    # Create pin positions for airwires
                    pins = []
                    for pad in net_pads[:10]:  # Limit for performance
                        try:
                            pos = getattr(pad, 'position', {'x': 0, 'y': 0})
                            pins.append({
                                'x': pos.x / 1000000 if hasattr(pos, 'x') else 0,
                                'y': pos.y / 1000000 if hasattr(pos, 'y') else 0
                            })
                        except:
                            pass
                    
                    net_pins[net_name] = pins
                    
                    # Create airwires for unrouted nets
                    if len(pins) > 1 and not net_info.get(net_name, {}).get('routed', False):
                        # Create airwires between pins (minimum spanning tree approach)
                        for i in range(len(pins) - 1):
                            airwires.append({
                                'start_x': pins[i]['x'],
                                'start_y': pins[i]['y'],
                                'end_x': pins[i + 1]['x'],
                                'end_y': pins[i + 1]['y'],
                                'net': net_name
                            })
                    
                except Exception as e:
                    logger.debug(f"Error processing net: {e}")
            
            # Update progress
            progress = (batch_end) / len(nets) * 100
            if progress_callback:
                progress_callback(int(progress), len(airwires), len(net_info))
        
        # Update GUI data with processed nets and airwires
        gui_data['nets'] = net_info
        gui_data['airwires'] = airwires
        
        # Clean up raw data
        if '_raw_nets' in gui_data:
            del gui_data['_raw_nets']
        if '_raw_pads' in gui_data:
            del gui_data['_raw_pads']
        if '_gui_tracks' in gui_data:
            del gui_data['_gui_tracks']
        
        logger.info(f"Progressive processing complete: {len(net_info)} nets, {len(airwires)} airwires")
        unrouted_count = len([n for n in net_info.values() if not n.get('routed', False) and n['pin_count'] > 1])
        logger.info(f"Unrouted nets: {unrouted_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in progressive net processing: {e}")
        return False

def main():
    """Main entry point"""
    logger.info("=== OrthoRoute IPC API Client ===")
    logger.info("Plugin launched - testing KiCad IPC connection!")
    
    # Check environment and credentials
    if len(sys.argv) > 1:
        logger.info(f"Command line args: {sys.argv[1:]}")
    else:
        logger.info("⚠ No command line credentials, will try environment variables")
    
    # Check dependencies
    logger.info("Testing PyQt6 availability...")
    if not check_dependencies():
        return 1
    
    logger.info("Skipping venv re-execution - using current environment")
    
    # Connect to KiCad
    logger.info("Attempting to connect to KiCad via IPC API...")
    kicad = connect_to_kicad()
    if not kicad:
        logger.error("Failed to connect to KiCad")
        return 1
    
    # Get and analyze board data
    board = kicad.get_board()
    if not board:
        logger.error("No board available")
        return 1
    
    board_data = analyze_board_data(board)
    if not board_data:
        logger.error("Failed to analyze board data")
        return 1
    
    # Launch Qt interface
    launch_qt_interface(board_data, kicad)
    return 0

if __name__ == "__main__":
    sys.exit(main())
