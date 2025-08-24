"""
Escape routing from F.Cu pads to internal routing grid
"""

import logging
from typing import Dict, List, Tuple, Optional
from .types import Pad, Via, Track, RoutingConfig
from .grid_manager import GridManager

logger = logging.getLogger(__name__)

class EscapeRouter:
    """Handles escape routing from F.Cu pads to internal routing layers"""
    
    def __init__(self, config: RoutingConfig, grid_manager: GridManager):
        self.config = config
        self.grid = grid_manager
    
    def create_escape_route(self, pad: Dict, net_id: int, target_layer: int) -> Optional[Dict]:
        """
        Create escape route from F.Cu pad to internal routing layer
        
        Args:
            pad: Pad dictionary with position and layer info
            net_id: Net ID for DRC
            target_layer: Target internal layer (0-based index)
            
        Returns:
            Dict with 'tracks', 'vias', 'grid_entry_point' or None if failed
        """
        logger.debug(f"Creating escape route for pad to layer {target_layer}")
        
        # Get pad position
        pad_x = pad.get('x_mm', 0.0)
        pad_y = pad.get('y_mm', 0.0)
        
        # Find nearest grid point
        grid_x, grid_y = self.grid.world_to_grid(pad_x, pad_y)
        
        # Ensure grid coordinates are valid
        grid_x = max(0, min(grid_x, self.grid.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid.grid_height - 1))
        
        world_x, world_y = self.grid.grid_to_world(grid_x, grid_y)
        
        # Check if target grid cell is available
        if not self.grid.is_cell_free(target_layer, grid_y, grid_x, net_id):
            # Try to find nearby free cell
            for radius in range(1, 5):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                            test_x = grid_x + dx
                            test_y = grid_y + dy
                            
                            if (0 <= test_x < self.grid.grid_width and 
                                0 <= test_y < self.grid.grid_height and
                                self.grid.is_cell_free(target_layer, test_y, test_x, net_id)):
                                
                                grid_x, grid_y = test_x, test_y
                                world_x, world_y = self.grid.grid_to_world(grid_x, grid_y)
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                logger.warning(f"No free grid cell found near pad at ({pad_x:.2f},{pad_y:.2f})")
                return None
        
        tracks = []
        vias = []
        
        # Create F.Cu escape track (from pad to via)
        if abs(pad_x - world_x) > 0.001 or abs(pad_y - world_y) > 0.001:
            escape_track = Track(
                start_x=pad_x,
                start_y=pad_y,
                end_x=world_x,
                end_y=world_y,
                layer=-1,  # F.Cu layer (special marker)
                width=self.config.track_width,
                net_id=net_id
            )
            tracks.append(escape_track)
        
        # Create blind via from F.Cu to target layer
        if target_layer > 0:  # Only create via if not already on target layer
            via = Via(
                x=world_x,
                y=world_y,
                from_layer=-1,  # F.Cu
                to_layer=target_layer,
                size=self.config.via_diameter,
                drill=self.config.via_drill,
                net_id=net_id
            )
            vias.append(via)
        
        # Mark grid cell as used
        self.grid.mark_cell_used(target_layer, grid_y, grid_x, net_id)
        
        return {
            'tracks': tracks,
            'vias': vias,
            'grid_entry_point': (target_layer, grid_y, grid_x),
            'world_point': (world_x, world_y),
            'success': True
        }
    
    def create_entry_route(self, pad: Dict, net_id: int, source_layer: int) -> Optional[Dict]:
        """
        Create entry route from internal routing layer to F.Cu pad
        
        Args:
            pad: Pad dictionary with position and layer info
            net_id: Net ID for DRC
            source_layer: Source internal layer (0-based index)
            
        Returns:
            Dict with 'tracks', 'vias', 'grid_exit_point' or None if failed
        """
        logger.debug(f"Creating entry route from layer {source_layer} to pad")
        
        # Get pad position
        pad_x = pad.get('x_mm', 0.0)
        pad_y = pad.get('y_mm', 0.0)
        
        # Find nearest grid point
        grid_x, grid_y = self.grid.world_to_grid(pad_x, pad_y)
        
        # Ensure grid coordinates are valid
        grid_x = max(0, min(grid_x, self.grid.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid.grid_height - 1))
        
        world_x, world_y = self.grid.grid_to_world(grid_x, grid_y)
        
        # Check if source grid cell is available
        if not self.grid.is_cell_free(source_layer, grid_y, grid_x, net_id):
            # Try to find nearby free cell
            for radius in range(1, 5):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                            test_x = grid_x + dx
                            test_y = grid_y + dy
                            
                            if (0 <= test_x < self.grid.grid_width and 
                                0 <= test_y < self.grid.grid_height and
                                self.grid.is_cell_free(source_layer, test_y, test_x, net_id)):
                                
                                grid_x, grid_y = test_x, test_y
                                world_x, world_y = self.grid.grid_to_world(grid_x, grid_y)
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                logger.warning(f"No free grid cell found near pad at ({pad_x:.2f},{pad_y:.2f})")
                return None
        
        tracks = []
        vias = []
        
        # Create blind via from source layer to F.Cu
        if source_layer > 0:  # Only create via if not already on F.Cu
            via = Via(
                x=world_x,
                y=world_y,
                from_layer=source_layer,
                to_layer=-1,  # F.Cu
                size=self.config.via_diameter,
                drill=self.config.via_drill,
                net_id=net_id
            )
            vias.append(via)
        
        # Create F.Cu entry track (from via to pad)
        if abs(world_x - pad_x) > 0.001 or abs(world_y - pad_y) > 0.001:
            entry_track = Track(
                start_x=world_x,
                start_y=world_y,
                end_x=pad_x,
                end_y=pad_y,
                layer=-1,  # F.Cu layer (special marker)
                width=self.config.track_width,
                net_id=net_id
            )
            tracks.append(entry_track)
        
        # Mark grid cell as used
        self.grid.mark_cell_used(source_layer, grid_y, grid_x, net_id)
        
        return {
            'tracks': tracks,
            'vias': vias,
            'grid_exit_point': (source_layer, grid_y, grid_x),
            'world_point': (world_x, world_y),
            'success': True
        }