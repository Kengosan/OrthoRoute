"""
OrthoRoute Route Importer
Handles importing routing results back into KiCad board.
"""

import pcbnew
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RouteSegment:
    """A single route segment"""
    start_x_nm: int
    start_y_nm: int
    end_x_nm: int
    end_y_nm: int
    layer: int
    width_nm: int
    net_code: int

@dataclass
class RouteVia:
    """A via in the route"""
    x_nm: int
    y_nm: int
    size_nm: int
    drill_nm: int
    net_code: int
    from_layer: int
    to_layer: int

@dataclass
class ImportedRoute:
    """Complete imported route information"""
    net_id: int
    net_name: str
    segments: List[RouteSegment]
    vias: List[RouteVia]
    total_length_mm: float
    success: bool

class RouteImporter:
    """Imports routing results back to KiCad board"""
    
    def __init__(self):
        self.imported_routes = []
        self.errors = []
        
    def import_routes(self, board, routing_results: Dict) -> List[ImportedRoute]:
        """Import routing results back to the board"""
        print("📥 RouteImporter: Starting route import...")
        
        self.imported_routes = []
        self.errors = []
        
        if not routing_results.get('nets'):
            print("⚠️ No routing results to import")
            return []
        
        # Get net info for mapping
        net_map = self._build_net_map(board)
        
        # Import each routed net
        for net_result in routing_results['nets']:
            try:
                imported_route = self._import_single_net(board, net_result, net_map)
                if imported_route.success:
                    self.imported_routes.append(imported_route)
                    print(f"✅ Imported net '{imported_route.net_name}': {len(imported_route.segments)} segments, {len(imported_route.vias)} vias")
                else:
                    print(f"❌ Failed to import net '{net_result.get('name', 'Unknown')}'")
                    
            except Exception as e:
                error_msg = f"Error importing net {net_result.get('name', 'Unknown')}: {e}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")
                continue
        
        # Refresh board display
        self._refresh_board()
        
        total_segments = sum(len(route.segments) for route in self.imported_routes)
        total_vias = sum(len(route.vias) for route in self.imported_routes)
        
        print(f"✅ Import complete: {len(self.imported_routes)} nets, {total_segments} segments, {total_vias} vias")
        return self.imported_routes
    
    def _build_net_map(self, board) -> Dict[int, object]:
        """Build mapping from net codes to net info objects"""
        net_map = {}
        net_info_list = board.GetNetInfo()
        
        for net_code in range(1, net_info_list.GetNetCount()):
            net_info = net_info_list.GetNetItem(net_code)
            if net_info:
                net_map[net_code] = net_info
        
        print(f"📋 Built net map: {len(net_map)} nets available")
        return net_map
    
    def _import_single_net(self, board, net_result: Dict, net_map: Dict) -> ImportedRoute:
        """Import a single net's routing results"""
        net_id = net_result['id']
        net_name = net_result.get('name', f'Net_{net_id}')
        path = net_result.get('path', [])
        
        # Initialize route
        route = ImportedRoute(
            net_id=net_id,
            net_name=net_name,
            segments=[],
            vias=[],
            total_length_mm=0.0,
            success=False
        )
        
        # Validate net and path
        if net_id not in net_map:
            print(f"⚠️ Net {net_id} ({net_name}) not found in board")
            return route
        
        if len(path) < 2:
            print(f"⚠️ Net {net_name}: insufficient path points ({len(path)})")
            return route
        
        net_info = net_map[net_id]
        track_width = net_result.get('width_nm', 200000)  # Default 0.2mm
        
        # Process path and create segments/vias
        segments_created = 0
        vias_created = 0
        total_length = 0.0
        
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            
            # Validate coordinates
            if not self._validate_coordinates(start_point, end_point):
                print(f"⚠️ Invalid coordinates in path for net '{net_name}' at segment {i}")
                continue
            
            start_layer = self._routing_layer_to_kicad_layer(start_point['layer'], board)
            end_layer = self._routing_layer_to_kicad_layer(end_point['layer'], board)
            
            # Create via if layer changes
            if start_layer != end_layer:
                via = self._create_via_object(start_point, net_info, net_id, start_layer, end_layer)
                if via:
                    # Add via to board
                    kicad_via = self._create_kicad_via(board, via)
                    if kicad_via:
                        board.Add(kicad_via)
                        route.vias.append(via)
                        vias_created += 1
            
            # Create track segment
            segment = RouteSegment(
                start_x_nm=start_point['x'],
                start_y_nm=start_point['y'],
                end_x_nm=end_point['x'],
                end_y_nm=end_point['y'],
                layer=start_layer,
                width_nm=track_width,
                net_code=net_id
            )
            
            # Add track to board
            kicad_track = self._create_kicad_track(board, segment, net_info)
            if kicad_track:
                board.Add(kicad_track)
                route.segments.append(segment)
                segments_created += 1
                
                # Calculate length
                dx = end_point['x'] - start_point['x']
                dy = end_point['y'] - start_point['y']
                length = (dx*dx + dy*dy) ** 0.5
                total_length += length
        
        route.total_length_mm = total_length / 1000000.0  # Convert nm to mm
        route.success = segments_created > 0
        
        return route
    
    def _validate_coordinates(self, point1: Dict, point2: Dict) -> bool:
        """Validate that coordinates are reasonable"""
        try:
            for point in [point1, point2]:
                if not isinstance(point.get('x'), (int, float)):
                    return False
                if not isinstance(point.get('y'), (int, float)):
                    return False
                if not isinstance(point.get('layer'), int):
                    return False
                # Check for reasonable coordinate ranges (±1 meter)
                if abs(point['x']) > 1000000000 or abs(point['y']) > 1000000000:
                    return False
            return True
        except Exception:
            return False
    
    def _routing_layer_to_kicad_layer(self, routing_layer: int, board) -> int:
        """Convert routing layer index back to KiCad layer"""
        try:
            copper_layers = board.GetCopperLayerCount()
            
            if routing_layer == 0:
                return pcbnew.F_Cu  # Top layer
            elif routing_layer == copper_layers - 1:
                return pcbnew.B_Cu  # Bottom layer
            elif 1 <= routing_layer < copper_layers - 1:
                # Inner layers
                return pcbnew.In1_Cu + (routing_layer - 1)
            else:
                # Invalid layer, default to top
                return pcbnew.F_Cu
        except Exception:
            return pcbnew.F_Cu
    
    def _create_via_object(self, point: Dict, net_info, net_code: int, 
                          from_layer: int, to_layer: int) -> RouteVia:
        """Create a via object"""
        return RouteVia(
            x_nm=int(point['x']),
            y_nm=int(point['y']),
            size_nm=400000,  # 0.4mm diameter
            drill_nm=200000,  # 0.2mm drill
            net_code=net_code,
            from_layer=from_layer,
            to_layer=to_layer
        )
    
    def _create_kicad_via(self, board, via: RouteVia):
        """Create a KiCad via object"""
        try:
            kicad_via = pcbnew.PCB_VIA(board)
            kicad_via.SetPosition(pcbnew.VECTOR2I(via.x_nm, via.y_nm))
            kicad_via.SetWidth(via.size_nm)
            kicad_via.SetDrill(via.drill_nm)
            kicad_via.SetNetCode(via.net_code)
            kicad_via.SetViaType(pcbnew.VIATYPE_THROUGH)
            return kicad_via
        except Exception as e:
            print(f"⚠️ Could not create via: {e}")
            return None
    
    def _create_kicad_track(self, board, segment: RouteSegment, net_info):
        """Create a KiCad track segment"""
        try:
            track = pcbnew.PCB_TRACK(board)
            track.SetStart(pcbnew.VECTOR2I(segment.start_x_nm, segment.start_y_nm))
            track.SetEnd(pcbnew.VECTOR2I(segment.end_x_nm, segment.end_y_nm))
            track.SetWidth(segment.width_nm)
            track.SetLayer(segment.layer)
            track.SetNetCode(segment.net_code)
            return track
        except Exception as e:
            print(f"⚠️ Could not create track: {e}")
            return None
    
    def _refresh_board(self):
        """Refresh board display"""
        try:
            pcbnew.Refresh()
            print("🔄 Board display refreshed")
        except Exception as e:
            print(f"⚠️ Could not refresh display: {e}")
    
    def get_import_summary(self) -> Dict:
        """Get summary of import operation"""
        total_segments = sum(len(route.segments) for route in self.imported_routes)
        total_vias = sum(len(route.vias) for route in self.imported_routes)
        total_length = sum(route.total_length_mm for route in self.imported_routes)
        
        return {
            'success': len(self.imported_routes) > 0,
            'nets_imported': len(self.imported_routes),
            'total_segments': total_segments,
            'total_vias': total_vias,
            'total_length_mm': total_length,
            'errors': self.errors,
            'routes': self.imported_routes
        }

def import_routing_results(board, routing_results: Dict) -> Dict:
    """Convenience function to import routing results"""
    importer = RouteImporter()
    importer.import_routes(board, routing_results)
    return importer.get_import_summary()
