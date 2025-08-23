"""
Virtual Copper Pour Generator

Creates a virtual routable area polygon for each layer, independent of KiCad's copper pours.
This polygon defines where traces can be routed based on:
- Board outline (maximum area)
- Pad clearances (keepout zones)
- Existing tracks/vias (obstacles)
- Board features (holes, cutouts)

The virtual copper pour exists only in our GPU routing model.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional
import numpy as np

# Import dependencies with fallback for different import contexts
try:
    from core.board_interface import BoardInterface
    from core.drc_rules import DRCRules
    from data_structures.grid_config import GridConfig
except ImportError:
    try:
        from ..core.board_interface import BoardInterface
        from ..core.drc_rules import DRCRules
        from ..data_structures.grid_config import GridConfig
    except ImportError:
        # Absolute import fallback
        import sys
        import os
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, src_dir)
        from core.board_interface import BoardInterface
        from core.drc_rules import DRCRules
        from data_structures.grid_config import GridConfig

logger = logging.getLogger(__name__)


"""
Virtual Copper Pour Generator

Creates a virtual routable area grid for each layer, independent of KiCad's copper pours.
This defines where traces can be routed based on:
- Board outline (maximum area)
- Pad clearances (keepout zones)
- Existing tracks/vias (obstacles)

The virtual copper pour exists only in our GPU routing model.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
import os

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

logger = logging.getLogger(__name__)


class VirtualCopperGenerator:
    """Generates virtual copper pour grids for PCB routing"""
    
    def __init__(self, board_interface, drc_rules, grid_config):
        self.board_interface = board_interface
        self.drc_rules = drc_rules
        self.grid_config = grid_config
        
        # Cache for generated grids
        self._virtual_copper_cache: Dict[str, np.ndarray] = {}
        
        logger.info("🔧 Virtual Copper Generator initialized")
    
    def generate_virtual_copper_grid(self, layer: str) -> np.ndarray:
        """
        Generate virtual copper pour grid for a specific layer
        
        Returns:
            np.ndarray: Boolean grid where True = routable area, False = obstacle
        """
        if layer in self._virtual_copper_cache:
            return self._virtual_copper_cache[layer]
        
        logger.info(f"🌟 Generating virtual copper pour for layer {layer}")
        
        # Initialize grid as all routable (True = free space)
        height, width = self.grid_config.height, self.grid_config.width
        virtual_copper_grid = np.ones((height, width), dtype=bool)
        
        # Step 1: Apply board outline constraints
        virtual_copper_grid = self._apply_board_outline_constraints(virtual_copper_grid)
        
        # Step 2: Apply pad keepout zones
        virtual_copper_grid = self._apply_pad_keepouts(virtual_copper_grid, layer)
        
        # Step 3: Apply existing track obstacles
        virtual_copper_grid = self._apply_track_obstacles(virtual_copper_grid, layer)
        
        # Step 4: Apply via obstacles
        virtual_copper_grid = self._apply_via_obstacles(virtual_copper_grid, layer)
        
        # Step 5: Apply board feature obstacles
        virtual_copper_grid = self._apply_board_feature_obstacles(virtual_copper_grid)
        
        # Cache the result
        self._virtual_copper_cache[layer] = virtual_copper_grid
        
        routable_cells = np.sum(virtual_copper_grid)
        total_cells = height * width
        percentage = (routable_cells / total_cells) * 100
        
        logger.info(f"✅ Virtual copper pour for {layer}: {routable_cells}/{total_cells} cells routable ({percentage:.1f}%)")
        
        return virtual_copper_grid
    
    def _apply_board_outline_constraints(self, grid: np.ndarray) -> np.ndarray:
        """Apply board outline as routing boundary"""
        try:
            # Get board outline from KiCad
            outline_points = self.board_interface.get_board_outline()
            
            if not outline_points or len(outline_points) < 3:
                logger.warning("⚠️ No valid board outline found, using full grid")
                return grid  # Keep full grid as routable
            
            # Convert dict format to tuple format for _point_in_polygon
            polygon_points = [(point['x'], point['y']) for point in outline_points]
            
            # For each grid cell, check if it's inside the board outline
            height, width = grid.shape
            
            for gy in range(height):
                for gx in range(width):
                    # Convert grid coordinates to world coordinates
                    world_x, world_y = self.grid_config.grid_to_world(gx, gy)
                    
                    # Simple point-in-polygon test using winding number
                    if not self._point_in_polygon(world_x, world_y, polygon_points):
                        grid[gy, gx] = False  # Outside board = not routable
            
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying board outline constraints: {e}")
            return grid  # Return original grid as fallback
    
    def _apply_pad_keepouts(self, grid: np.ndarray, layer: str) -> np.ndarray:
        """Apply keepout zones around all pads"""
        try:
            logger.info(f"🚀 Starting pad keepouts for {layer}")
            pads = self.board_interface.get_all_pads()
            logger.info(f"🚀 Got {len(pads)} total pads")
            height, width = grid.shape
            
            pad_count = 0
            skipped_count = 0
            for pad in pads:
                # Check if pad affects this layer
                if not self._pad_affects_layer(pad, layer):
                    skipped_count += 1
                    continue
                
                # Get pad geometry using the correct method
                pad_geom = self.board_interface.get_pad_geometry(pad)
                pad_x = pad_geom['x']
                pad_y = pad_geom['y']
                size_x = pad_geom['size_x']
                size_y = pad_geom['size_y']
                
                # Calculate keepout radius with enhanced logic for fine-pitch components
                pad_radius = max(size_x, size_y) / 2.0
                
                # Enhanced keepout calculation for fine-pitch components
                base_clearance = self.drc_rules.min_trace_spacing
                
                # Detect fine-pitch components and apply enhanced spacing
                if self._is_fine_pitch_component(pad, pads):
                    # For fine-pitch components (IC packages), use 1.5x clearance instead of 2x
                    # This provides better clearance while not being overly restrictive
                    enhanced_clearance = base_clearance * 1.5
                    keepout_radius = pad_radius + enhanced_clearance
                    logger.debug(f"Fine-pitch component detected: enhanced clearance {enhanced_clearance:.3f}mm")
                else:
                    # Standard components use normal clearance
                    keepout_radius = pad_radius + base_clearance
                
                # Convert to grid coordinates
                pad_gx, pad_gy = self.grid_config.world_to_grid(pad_x, pad_y)
                keepout_radius_grid = keepout_radius / self.grid_config.resolution
                
                # Mark cells within keepout radius as obstacles
                for gy in range(max(0, int(pad_gy - keepout_radius_grid)), 
                               min(height, int(pad_gy + keepout_radius_grid) + 1)):
                    for gx in range(max(0, int(pad_gx - keepout_radius_grid)), 
                                   min(width, int(pad_gx + keepout_radius_grid) + 1)):
                        
                        # Calculate distance from pad center
                        distance = np.sqrt((gx - pad_gx)**2 + (gy - pad_gy)**2) * self.grid_config.resolution
                        
                        if distance <= keepout_radius:
                            grid[gy, gx] = False  # Mark as obstacle
                
                pad_count += 1
            
            logger.info(f"📦 Applied keepouts for {pad_count} pads on {layer} (skipped {skipped_count})")
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying pad keepouts: {e}")
            return grid
    
    def _is_fine_pitch_component(self, target_pad, all_pads) -> bool:
        """
        Detect if a pad belongs to a fine-pitch component (like QFN/BGA ICs)
        by analyzing the spacing to nearby pads from the same component.
        
        Args:
            target_pad: The pad to analyze
            all_pads: List of all pads on the board
            
        Returns:
            bool: True if this is a fine-pitch component
        """
        try:
            # Get target pad geometry
            target_geom = self.board_interface.get_pad_geometry(target_pad)
            target_x, target_y = target_geom['x'], target_geom['y']
            target_size = max(target_geom['size_x'], target_geom['size_y'])
            
            # Simple heuristic: if pad is smaller than 0.7mm, likely fine-pitch
            # This catches QFN, BGA, and other dense IC packages
            if target_size < 0.7:  # mm
                logger.debug(f"Small pad detected: {target_size:.3f}mm - treating as fine-pitch")
                return True
            
            # Also check pad density in local area
            nearby_pads = 0
            search_radius = 3.0  # mm - smaller search radius for local density
            
            for pad in all_pads:
                if pad == target_pad:  # Skip self
                    continue
                    
                # Calculate distance between pads
                pad_geom = self.board_interface.get_pad_geometry(pad)
                pad_x, pad_y = pad_geom['x'], pad_geom['y']
                distance = math.sqrt((pad_x - target_x)**2 + (pad_y - target_y)**2)
                
                # Count nearby pads within search radius
                if distance <= search_radius:
                    nearby_pads += 1
            
            # If there are many pads nearby (>6), likely a dense IC package
            if nearby_pads > 6:
                logger.debug(f"Dense component detected: {nearby_pads} pads within {search_radius}mm - treating as fine-pitch")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Error in fine-pitch detection: {e}")
            return False  # Default to standard spacing on error
    
    def _get_pad_component(self, pad) -> Optional[str]:
        """Get the component reference designator for a pad"""
        try:
            # Try to get component from pad object
            if hasattr(pad, 'GetParentFootprint'):
                footprint = pad.GetParentFootprint()
                if footprint and hasattr(footprint, 'GetReference'):
                    return footprint.GetReference()
            elif hasattr(pad, 'GetParent'):
                parent = pad.GetParent()
                if parent and hasattr(parent, 'GetReference'):
                    return parent.GetReference()
            
            # Fallback: try board interface method if available
            if hasattr(self.board_interface, 'get_pad_component'):
                return self.board_interface.get_pad_component(pad)
                
            return None
            
        except Exception as e:
            logger.debug(f"Could not get component for pad: {e}")
            return None
    
    def _apply_track_obstacles(self, grid: np.ndarray, layer: str) -> np.ndarray:
        """Apply obstacles for existing tracks"""
        try:
            tracks = self.board_interface.get_all_tracks()
            layer_id = self.board_interface.get_layer_id(layer)
            
            track_count = 0
            for track in tracks:
                track_geom = self.board_interface.get_track_geometry(track)
                
                # Only process tracks on this layer
                if track_geom['layer'] != layer_id:
                    continue
                
                # Mark track area as obstacle using line drawing with clearance
                # Add required clearance to track width for proper DRC compliance
                track_width_with_clearance = track_geom['width'] + (2 * self.drc_rules.min_trace_spacing)
                self._mark_line_obstacle(
                    grid,
                    track_geom['start_x'], track_geom['start_y'],
                    track_geom['end_x'], track_geom['end_y'],
                    track_width_with_clearance
                )
                
                track_count += 1
            
            logger.info(f"🛤️ Applied {track_count} track obstacles on {layer}")
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying track obstacles: {e}")
            return grid
    
    def _apply_via_obstacles(self, grid: np.ndarray, layer: str) -> np.ndarray:
        """Apply obstacles for vias"""
        try:
            vias = self.board_interface.get_all_vias()
            layer_id = self.board_interface.get_layer_id(layer)
            
            via_count = 0
            for via in vias:
                # Via already contains geometry information
                via_geom = {
                    'x': via['x'],
                    'y': via['y'],
                    'diameter': via.get('via_diameter', 0.6),
                    'drill': via.get('drill_diameter', 0.3)
                }
                
                # Check if via affects this layer
                if not self._via_affects_layer(via_geom, layer_id):
                    continue
                
                # Mark circular area around via as obstacle
                via_gx, via_gy = self.grid_config.world_to_grid(via_geom['x'], via_geom['y'])
                via_radius = (via_geom['drill_diameter'] / 2.0 + self.drc_rules.min_via_spacing)
                via_radius_grid = via_radius / self.grid_config.resolution
                
                height, width = grid.shape
                for gy in range(max(0, int(via_gy - via_radius_grid)), 
                               min(height, int(via_gy + via_radius_grid) + 1)):
                    for gx in range(max(0, int(via_gx - via_radius_grid)), 
                                   min(width, int(via_gx + via_radius_grid) + 1)):
                        
                        distance = np.sqrt((gx - via_gx)**2 + (gy - via_gy)**2) * self.grid_config.resolution
                        if distance <= via_radius:
                            grid[gy, gx] = False
                
                via_count += 1
            
            logger.debug(f"🔘 Applied {via_count} via obstacles on {layer}")
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying via obstacles: {e}")
            return grid
    
    def _apply_board_feature_obstacles(self, grid: np.ndarray) -> np.ndarray:
        """Apply obstacles for board features like holes"""
        try:
            holes = self.board_interface.get_board_holes()
            
            hole_count = 0
            for hole in holes:
                hole_gx, hole_gy = self.grid_config.world_to_grid(hole['x'], hole['y'])
                hole_radius = (hole['diameter'] / 2.0 + self.drc_rules.min_hole_clearance)
                hole_radius_grid = hole_radius / self.grid_config.resolution
                
                height, width = grid.shape
                for gy in range(max(0, int(hole_gy - hole_radius_grid)), 
                               min(height, int(hole_gy + hole_radius_grid) + 1)):
                    for gx in range(max(0, int(hole_gx - hole_radius_grid)), 
                                   min(width, int(hole_gx + hole_radius_grid) + 1)):
                        
                        distance = np.sqrt((gx - hole_gx)**2 + (gy - hole_gy)**2) * self.grid_config.resolution
                        if distance <= hole_radius:
                            grid[gy, gx] = False
                
                hole_count += 1
            
            logger.debug(f"🕳️ Applied {hole_count} board feature obstacles")
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying board feature obstacles: {e}")
            return grid
    
    def _mark_line_obstacle(self, grid: np.ndarray, start_x: float, start_y: float, 
                           end_x: float, end_y: float, width: float):
        """Mark a line as an obstacle in the grid"""
        # Convert to grid coordinates
        start_gx, start_gy = self.grid_config.world_to_grid(start_x, start_y)
        end_gx, end_gy = self.grid_config.world_to_grid(end_x, end_y)
        width_grid = width / self.grid_config.resolution
        
        # Use Bresenham's line algorithm with width
        dx = abs(end_gx - start_gx)
        dy = abs(end_gy - start_gy)
        sx = 1 if start_gx < end_gx else -1
        sy = 1 if start_gy < end_gy else -1
        err = dx - dy
        
        x, y = start_gx, start_gy
        height, width_cells = grid.shape
        
        while True:
            # Mark circle around current point
            for dy_offset in range(int(-width_grid), int(width_grid) + 1):
                for dx_offset in range(int(-width_grid), int(width_grid) + 1):
                    if dx_offset**2 + dy_offset**2 <= width_grid**2:
                        px, py = int(x + dx_offset), int(y + dy_offset)
                        if 0 <= px < width_cells and 0 <= py < height:
                            grid[py, px] = False
            
            if x == end_gx and y == end_gy:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def _point_in_polygon(self, x: float, y: float, polygon_points: List[Tuple[float, float]]) -> bool:
        """Test if point is inside polygon using ray casting"""
        n = len(polygon_points)
        inside = False
        
        p1x, p1y = polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _pad_affects_layer(self, pad, layer: str) -> bool:
        """Check if a pad affects routing on the given layer"""
        try:
            pad_layers = self.board_interface.get_pad_layers(pad)
            layer_id = self.board_interface.get_layer_id(layer)
            
            # Pad affects layer if it's on that layer or is through-hole
            return layer_id in pad_layers or self.board_interface.is_pad_through_hole(pad)
            
        except Exception:
            return True  # Conservative: assume it affects the layer
    
    def _via_affects_layer(self, via_geom: dict, layer_id: int) -> bool:
        """Check if a via affects routing on the given layer"""
        try:
            start_layer = via_geom.get('start_layer', 0)
            end_layer = via_geom.get('end_layer', 31)
            
            # Via affects layer if it spans through it
            return start_layer <= layer_id <= end_layer
            
        except Exception:
            return True  # Conservative: assume it affects the layer
    
    def clear_cache(self):
        """Clear the virtual copper cache"""
        self._virtual_copper_cache.clear()
        logger.debug("🧹 Virtual copper cache cleared")
