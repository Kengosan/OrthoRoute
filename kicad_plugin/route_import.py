"""
Route Import for OrthoRoute KiCad Plugin
Convert GPU routing results back to KiCad tracks and vias

This module takes the JSON results from the OrthoRoute GPU engine and
creates the corresponding tracks and vias in the KiCad PCB design.
"""

import pcbnew
import wx
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import math

@dataclass
class ImportedTrack:
    """Track segment to be imported"""
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    layer: int
    width: int
    net_code: int
    net_name: str

@dataclass
class ImportedVia:
    """Via to be imported"""
    x: int
    y: int
    size: int
    drill: int
    net_code: int
    net_name: str
    layer_start: int
    layer_end: int

@dataclass
class ImportStats:
    """Import operation statistics"""
    tracks_created: int = 0
    vias_created: int = 0
    tracks_failed: int = 0
    vias_failed: int = 0
    nets_processed: int = 0
    nets_imported: int = 0
    total_length_nm: int = 0
    existing_tracks_removed: int = 0
    existing_vias_removed: int = 0

class RouteImporter:
    """Import GPU routing results into KiCad board"""
    
    def __init__(self, board: pcbnew.BOARD):
        self.board = board
        self.layer_map = self._build_reverse_layer_map()
        self.net_map = self._build_net_map()
        self.stats = ImportStats()
        
        # Import options
        self.options = {
            'remove_existing_routes': True,
            'validate_design_rules': True,
            'merge_overlapping_tracks': True,
            'optimize_via_placement': True,
            'create_teardrops': False,
            'add_test_points': False
        }
    
    def apply_routes(self, routed_nets: List[Dict], options: Dict = None) -> int:
        """
        Apply GPU routing results to KiCad board.
        
        Args:
            routed_nets: List of routed nets from OrthoRoute
            options: Import options (optional)
            
        Returns:
            Number of successfully imported nets
        """
        if options:
            self.options.update(options)
        
        print(f"📥 Importing {len(routed_nets)} routed nets...")
        
        # Start modification group for undo support
        self.board.BeginModify()
        
        try:
            # Remove existing routes if requested
            if self.options['remove_existing_routes']:
                self._remove_existing_routes(routed_nets)
            
            # Process each routed net
            for net_data in routed_nets:
                if self._import_net_routes(net_data):
                    self.stats.nets_imported += 1
                self.stats.nets_processed += 1
            
            # Post-processing optimizations
            if self.options['merge_overlapping_tracks']:
                self._merge_overlapping_tracks()
            
            if self.options['optimize_via_placement']:
                self._optimize_via_placement()
            
            # Commit changes
            self.board.EndModify()
            
            # Update board
            pcbnew.Refresh()
            
            print(f"✅ Import complete: {self.stats.nets_imported}/{self.stats.nets_processed} nets")
            self._print_import_summary()
            
            return self.stats.nets_imported
            
        except Exception as e:
            # Rollback on error
            self.board.EndModify()
            print(f"❌ Import failed: {e}")
            raise
    
    def _build_reverse_layer_map(self) -> Dict[int, int]:
        """Build mapping from sequential indices to KiCad layer IDs"""
        layer_map = {}
        copper_layers = []
        
        # Get all enabled copper layers
        for layer_id in range(pcbnew.PCB_LAYER_ID_COUNT):
            if (self.board.IsLayerEnabled(layer_id) and 
                pcbnew.IsCopperLayer(layer_id)):
                copper_layers.append(layer_id)
        
        # Sort layers (front to back)
        copper_layers.sort()
        
        # Map sequential indices to KiCad layer IDs
        for idx, layer_id in enumerate(copper_layers):
            layer_map[idx] = layer_id
        
        return layer_map
    
    def _build_net_map(self) -> Dict[int, pcbnew.NETINFO_ITEM]:
        """Build mapping from net IDs to KiCad net objects"""
        net_map = {}
        
        for net_id in range(self.board.GetNetCount()):
            net = self.board.FindNet(net_id)
            if net:
                net_map[net_id] = net
        
        return net_map
    
    def _remove_existing_routes(self, routed_nets: List[Dict]):
        """Remove existing routes for nets that will be imported"""
        print("🧹 Removing existing routes...")
        
        # Get net codes for routed nets
        routed_net_codes = set()
        for net_data in routed_nets:
            net_id = net_data.get('id')
            if net_id in self.net_map:
                routed_net_codes.add(net_id)
        
        # Remove existing tracks and vias
        tracks_to_remove = []
        
        for track in self.board.GetTracks():
            net_code = track.GetNetCode()
            if net_code in routed_net_codes:
                tracks_to_remove.append(track)
        
        # Remove tracks
        for track in tracks_to_remove:
            self.board.RemoveNative(track)
            if isinstance(track, pcbnew.PCB_TRACK):
                self.stats.existing_tracks_removed += 1
            elif isinstance(track, pcbnew.PCB_VIA):
                self.stats.existing_vias_removed += 1
        
        print(f"   Removed {self.stats.existing_tracks_removed} tracks, "
              f"{self.stats.existing_vias_removed} vias")
    
    def _import_net_routes(self, net_data: Dict) -> bool:
        """Import routes for a single net"""
        net_id = net_data.get('id')
        net_name = net_data.get('name', f'Net_{net_id}')
        path = net_data.get('path', [])
        width_nm = net_data.get('width_nm', 200000)
        via_size_nm = net_data.get('via_size_nm', 200000)
        
        if not path or len(path) < 2:
            return False
        
        # Get KiCad net object
        if net_id not in self.net_map:
            print(f"⚠️  Warning: Net {net_name} (ID {net_id}) not found in board")
            return False
        
        kicad_net = self.net_map[net_id]
        
        try:
            # Convert path to tracks and vias
            tracks, vias = self._path_to_tracks_and_vias(
                path, width_nm, via_size_nm, net_id, net_name
            )
            
            # Create tracks
            for track_data in tracks:
                if self._create_track(track_data, kicad_net):
                    self.stats.tracks_created += 1
                    self.stats.total_length_nm += self._calculate_track_length(track_data)
                else:
                    self.stats.tracks_failed += 1
            
            # Create vias
            for via_data in vias:
                if self._create_via(via_data, kicad_net):
                    self.stats.vias_created += 1
                else:
                    self.stats.vias_failed += 1
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to import net {net_name}: {e}")
            return False
    
    def _path_to_tracks_and_vias(self, path: List[Dict], width_nm: int, 
                                via_size_nm: int, net_id: int, net_name: str) -> Tuple[List[ImportedTrack], List[ImportedVia]]:
        """Convert path points to track segments and vias"""
        tracks = []
        vias = []
        
        for i in range(len(path) - 1):
            current_point = path[i]
            next_point = path[i + 1]
            
            current_layer = current_point['layer']
            next_layer = next_point['layer']
            
            # Handle layer changes (vias)
            if current_layer != next_layer:
                # Create via at current position
                via = ImportedVia(
                    x=current_point['x'],
                    y=current_point['y'],
                    size=via_size_nm,
                    drill=int(via_size_nm * 0.6),  # 60% drill ratio
                    net_code=net_id,
                    net_name=net_name,
                    layer_start=min(current_layer, next_layer),
                    layer_end=max(current_layer, next_layer)
                )
                vias.append(via)
                
                # If there's a position change along with layer change,
                # create track on the new layer
                if (current_point['x'] != next_point['x'] or 
                    current_point['y'] != next_point['y']):
                    
                    track = ImportedTrack(
                        start_x=current_point['x'],
                        start_y=current_point['y'],
                        end_x=next_point['x'],
                        end_y=next_point['y'],
                        layer=next_layer,
                        width=width_nm,
                        net_code=net_id,
                        net_name=net_name
                    )
                    tracks.append(track)
            
            else:
                # Same layer - create track segment
                track = ImportedTrack(
                    start_x=current_point['x'],
                    start_y=current_point['y'],
                    end_x=next_point['x'],
                    end_y=next_point['y'],
                    layer=current_layer,
                    width=width_nm,
                    net_code=net_id,
                    net_name=net_name
                )
                tracks.append(track)
        
        return tracks, vias
    
    def _create_track(self, track_data: ImportedTrack, net: pcbnew.NETINFO_ITEM) -> bool:
        """Create KiCad track from track data"""
        try:
            # Get KiCad layer ID
            if track_data.layer not in self.layer_map:
                print(f"⚠️  Warning: Invalid layer {track_data.layer} for track")
                return False
            
            kicad_layer = self.layer_map[track_data.layer]
            
            # Create track object
            track = pcbnew.PCB_TRACK(self.board)
            
            # Set properties
            track.SetStart(pcbnew.VECTOR2I(track_data.start_x, track_data.start_y))
            track.SetEnd(pcbnew.VECTOR2I(track_data.end_x, track_data.end_y))
            track.SetWidth(track_data.width)
            track.SetLayer(kicad_layer)
            track.SetNet(net)
            
            # Validate track
            if not self._validate_track(track):
                return False
            
            # Add to board
            self.board.Add(track)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create track: {e}")
            return False
    
    def _create_via(self, via_data: ImportedVia, net: pcbnew.NETINFO_ITEM) -> bool:
        """Create KiCad via from via data"""
        try:
            # Create via object
            via = pcbnew.PCB_VIA(self.board)
            
            # Set properties
            via.SetPosition(pcbnew.VECTOR2I(via_data.x, via_data.y))
            via.SetWidth(via_data.size)
            via.SetDrill(via_data.drill)
            via.SetNet(net)
            
            # Set layer span
            if (via_data.layer_start in self.layer_map and 
                via_data.layer_end in self.layer_map):
                
                start_layer = self.layer_map[via_data.layer_start]
                end_layer = self.layer_map[via_data.layer_end]
                
                via.SetLayerPair(start_layer, end_layer)
            else:
                # Default to through-hole via
                via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
            
            # Validate via
            if not self._validate_via(via):
                return False
            
            # Add to board
            self.board.Add(via)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create via: {e}")
            return False
    
    def _validate_track(self, track: pcbnew.PCB_TRACK) -> bool:
        """Validate track against design rules"""
        if not self.options['validate_design_rules']:
            return True
        
        try:
            # Check minimum track width
            design_settings = self.board.GetDesignSettings()
            min_width = design_settings.m_TrackMinWidth
            
            if track.GetWidth() < min_width:
                print(f"⚠️  Track width {track.GetWidth()} below minimum {min_width}")
                track.SetWidth(min_width)
            
            # Check track length (avoid zero-length tracks)
            start = track.GetStart()
            end = track.GetEnd()
            if start.x == end.x and start.y == end.y:
                return False
            
            # Check layer validity
            layer = track.GetLayer()
            if not self.board.IsLayerEnabled(layer) or not pcbnew.IsCopperLayer(layer):
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  Track validation failed: {e}")
            return False
    
    def _validate_via(self, via: pcbnew.PCB_VIA) -> bool:
        """Validate via against design rules"""
        if not self.options['validate_design_rules']:
            return True
        
        try:
            # Check minimum via size
            design_settings = self.board.GetDesignSettings()
            min_via_size = design_settings.m_ViasMinSize
            min_drill = design_settings.m_ViasMinDrill
            
            if via.GetWidth() < min_via_size:
                print(f"⚠️  Via size {via.GetWidth()} below minimum {min_via_size}")
                via.SetWidth(min_via_size)
            
            if via.GetDrillValue() < min_drill:
                print(f"⚠️  Via drill {via.GetDrillValue()} below minimum {min_drill}")
                via.SetDrill(min_drill)
            
            # Check drill/size ratio
            if via.GetDrillValue() >= via.GetWidth():
                print(f"⚠️  Via drill {via.GetDrillValue()} >= size {via.GetWidth()}")
                via.SetDrill(int(via.GetWidth() * 0.6))
            
            return True
            
        except Exception as e:
            print(f"⚠️  Via validation failed: {e}")
            return False
    
    def _calculate_track_length(self, track_data: ImportedTrack) -> int:
        """Calculate track length in nanometers"""
        dx = track_data.end_x - track_data.start_x
        dy = track_data.end_y - track_data.start_y
        return int(math.sqrt(dx*dx + dy*dy))
    
    def _merge_overlapping_tracks(self):
        """Merge overlapping track segments on same net/layer"""
        print("🔧 Optimizing track segments...")
        
        # Group tracks by net and layer
        track_groups = {}
        
        for track in self.board.GetTracks():
            if isinstance(track, pcbnew.PCB_TRACK):
                key = (track.GetNetCode(), track.GetLayer())
                if key not in track_groups:
                    track_groups[key] = []
                track_groups[key].append(track)
        
        merged_count = 0
        
        # Process each group
        for (net_code, layer), tracks in track_groups.items():
            if len(tracks) < 2:
                continue
            
            # Find tracks that can be merged (collinear and adjacent)
            to_merge = self._find_mergeable_tracks(tracks)
            
            for track_pair in to_merge:
                if self._merge_track_pair(track_pair[0], track_pair[1]):
                    merged_count += 1
        
        if merged_count > 0:
            print(f"   Merged {merged_count} track segments")
    
    def _find_mergeable_tracks(self, tracks: List[pcbnew.PCB_TRACK]) -> List[Tuple[pcbnew.PCB_TRACK, pcbnew.PCB_TRACK]]:
        """Find pairs of tracks that can be merged"""
        mergeable = []
        
        for i, track1 in enumerate(tracks):
            for track2 in tracks[i+1:]:
                if self._can_merge_tracks(track1, track2):
                    mergeable.append((track1, track2))
        
        return mergeable
    
    def _can_merge_tracks(self, track1: pcbnew.PCB_TRACK, track2: pcbnew.PCB_TRACK) -> bool:
        """Check if two tracks can be merged"""
        # Same width and net
        if (track1.GetWidth() != track2.GetWidth() or 
            track1.GetNetCode() != track2.GetNetCode()):
            return False
        
        # Check if tracks are collinear and adjacent
        start1 = track1.GetStart()
        end1 = track1.GetEnd()
        start2 = track2.GetStart()
        end2 = track2.GetEnd()
        
        # Simple adjacency check (endpoints touch)
        tolerance = track1.GetWidth() // 2
        
        adjacent = (
            self._points_close(end1, start2, tolerance) or
            self._points_close(end1, end2, tolerance) or
            self._points_close(start1, start2, tolerance) or
            self._points_close(start1, end2, tolerance)
        )
        
        if not adjacent:
            return False
        
        # Check collinearity (simplified)
        return self._tracks_collinear(track1, track2)
    
    def _points_close(self, p1: pcbnew.VECTOR2I, p2: pcbnew.VECTOR2I, tolerance: int) -> bool:
        """Check if two points are close within tolerance"""
        dx = abs(p1.x - p2.x)
        dy = abs(p1.y - p2.y)
        return dx <= tolerance and dy <= tolerance
    
    def _tracks_collinear(self, track1: pcbnew.PCB_TRACK, track2: pcbnew.PCB_TRACK) -> bool:
        """Check if two tracks are approximately collinear"""
        # Simplified collinearity check
        # In a real implementation, this would check angle between track vectors
        
        # For now, just check if tracks are horizontal or vertical
        start1 = track1.GetStart()
        end1 = track1.GetEnd()
        start2 = track2.GetStart()
        end2 = track2.GetEnd()
        
        # Track 1 direction
        dx1 = end1.x - start1.x
        dy1 = end1.y - start1.y
        
        # Track 2 direction  
        dx2 = end2.x - start2.x
        dy2 = end2.y - start2.y
        
        # Check if both horizontal or both vertical
        if (dx1 == 0 and dx2 == 0) or (dy1 == 0 and dy2 == 0):
            return True
        
        # Check if directions are parallel (simplified)
        if dx1 != 0 and dx2 != 0 and dy1 != 0 and dy2 != 0:
            ratio1 = dy1 / dx1 if dx1 != 0 else float('inf')
            ratio2 = dy2 / dx2 if dx2 != 0 else float('inf')
            return abs(ratio1 - ratio2) < 0.1  # 10% tolerance
        
        return False
    
    def _merge_track_pair(self, track1: pcbnew.PCB_TRACK, track2: pcbnew.PCB_TRACK) -> bool:
        """Merge two tracks into one"""
        try:
            # Determine new start and end points
            points = [track1.GetStart(), track1.GetEnd(), track2.GetStart(), track2.GetEnd()]
            
            # Find the two points that are furthest apart
            max_dist = 0
            new_start = None
            new_end = None
            
            for i, p1 in enumerate(points):
                for p2 in points[i+1:]:
                    dist = (p1.x - p2.x)**2 + (p1.y - p2.y)**2
                    if dist > max_dist:
                        max_dist = dist
                        new_start = p1
                        new_end = p2
            
            if new_start and new_end:
                # Update first track with merged geometry
                track1.SetStart(new_start)
                track1.SetEnd(new_end)
                
                # Remove second track
                self.board.RemoveNative(track2)
                
                return True
            
        except Exception as e:
            print(f"⚠️  Failed to merge tracks: {e}")
        
        return False
    
    def _optimize_via_placement(self):
        """Optimize via placement to reduce count and improve routing"""
        print("🔧 Optimizing via placement...")
        
        # Find vias that might be redundant
        vias_to_remove = []
        
        for via in self.board.GetTracks():
            if isinstance(via, pcbnew.PCB_VIA):
                if self._is_redundant_via(via):
                    vias_to_remove.append(via)
        
        # Remove redundant vias
        for via in vias_to_remove:
            self.board.RemoveNative(via)
        
        if vias_to_remove:
            print(f"   Removed {len(vias_to_remove)} redundant vias")
    
    def _is_redundant_via(self, via: pcbnew.PCB_VIA) -> bool:
        """Check if via is redundant (connects layers with no tracks)"""
        # Simplified check - in practice this would be more sophisticated
        via_pos = via.GetPosition()
        via_net = via.GetNetCode()
        
        # Count tracks connected to this via
        connected_tracks = 0
        
        for track in self.board.GetTracks():
            if (isinstance(track, pcbnew.PCB_TRACK) and 
                track.GetNetCode() == via_net):
                
                start = track.GetStart()
                end = track.GetEnd()
                tolerance = via.GetWidth()
                
                if (self._points_close(start, via_pos, tolerance) or 
                    self._points_close(end, via_pos, tolerance)):
                    connected_tracks += 1
        
        # Via is redundant if it connects fewer than 2 tracks
        return connected_tracks < 2
    
    def _print_import_summary(self):
        """Print detailed import summary"""
        print(f"\n📋 Import Summary:")
        print(f"  Nets processed: {self.stats.nets_processed}")
        print(f"  Nets imported: {self.stats.nets_imported}")
        print(f"  Success rate: {self.stats.nets_imported/self.stats.nets_processed*100:.1f}%")
        
        print(f"\n🛤️  Tracks:")
        print(f"  Created: {self.stats.tracks_created}")
        print(f"  Failed: {self.stats.tracks_failed}")
        print(f"  Removed (existing): {self.stats.existing_tracks_removed}")
        print(f"  Total length: {self.stats.total_length_nm/1000000:.1f} mm")
        
        print(f"\n🔌 Vias:")
        print(f"  Created: {self.stats.vias_created}")
        print(f"  Failed: {self.stats.vias_failed}")
        print(f"  Removed (existing): {self.stats.existing_vias_removed}")

def import_routes_from_file(board: pcbnew.BOARD, results_file: str, 
                           options: Dict = None) -> bool:
    """
    Import routes from JSON results file.
    
    Args:
        board: KiCad board object
        results_file: Path to OrthoRoute results JSON file
        options: Import options
        
    Returns:
        True if import successful
    """
    try:
        import json
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if not results.get('success', False):
            print(f"❌ Results file indicates routing failure")
            return False
        
        routed_nets = results.get('routed_nets', [])
        if not routed_nets:
            print(f"❌ No routed nets found in results file")
            return False
        
        importer = RouteImporter(board)
        imported_count = importer.apply_routes(routed_nets, options)
        
        print(f"✅ Imported {imported_count} nets from {results_file}")
        return imported_count > 0
        
    except Exception as e:
        print(f"❌ Import from file failed: {e}")
        return False

def get_import_options() -> Dict:
    """Get default import options"""
    return {
        'remove_existing_routes': True,
        'validate_design_rules': True,
        'merge_overlapping_tracks': True,
        'optimize_via_placement': True,
        'create_teardrops': False,
        'add_test_points': False
    }

def validate_routing_results(results: Dict) -> Tuple[bool, str]:
    """
    Validate routing results format.
    
    Args:
        results: OrthoRoute results dictionary
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(results, dict):
        return False, "Results must be a dictionary"
    
    if not results.get('success', False):
        return False, f"Routing failed: {results.get('error', 'Unknown error')}"
    
    routed_nets = results.get('routed_nets', [])
    if not routed_nets:
        return False, "No routed nets found"
    
    # Validate net format
    for i, net in enumerate(routed_nets):
        if not isinstance(net, dict):
            return False, f"Net {i} is not a dictionary"
        
        required_fields = ['id', 'name', 'path']
        for field in required_fields:
            if field not in net:
                return False, f"Net {i} missing required field: {field}"
        
        path = net['path']
        if not isinstance(path, list) or len(path) < 2:
            return False, f"Net {i} has invalid path"
        
        # Validate path points
        for j, point in enumerate(path):
            if not isinstance(point, dict):
                return False, f"Net {i} path point {j} is not a dictionary"
            
            required_coords = ['x', 'y', 'layer']
            for coord in required_coords:
                if coord not in point:
                    return False, f"Net {i} path point {j} missing {coord}"
    
    return True, "Results format is valid"