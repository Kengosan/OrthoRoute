"""Layer direction assignment for Manhattan routing."""
import logging
from typing import List, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class LayerDirection(Enum):
    """Layer routing directions for Manhattan routing."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class LayerDirectionManager:
    """Manages layer direction assignments for Manhattan routing."""
    
    def __init__(self, layer_names: List[str]):
        """Initialize layer direction manager."""
        self.layer_names = layer_names
        self.layer_directions = {}
        self.name_to_index = {name: i for i, name in enumerate(layer_names)}
        self.index_to_name = {i: name for i, name in enumerate(layer_names)}
        
        self._assign_layer_directions()
        
        logger.info(f"Initialized layer direction manager for {len(layer_names)} layers")
    
    def _assign_layer_directions(self):
        """Assign directions to layers based on Manhattan routing rules."""
        for i, layer_name in enumerate(self.layer_names):
            direction = self._get_layer_direction(layer_name)
            self.layer_directions[i] = direction
            
            direction_str = "horizontal" if direction == LayerDirection.HORIZONTAL else "vertical"
            logger.debug(f"Layer {layer_name} (index {i}): {direction_str}")
    
    def _get_layer_direction(self, layer_name: str) -> LayerDirection:
        """Get layer direction based on Manhattan routing rules."""
        # B.Cu is always vertical per specification
        if layer_name == 'B.Cu':
            return LayerDirection.VERTICAL
        
        # For inner layers: odd numbers horizontal, even numbers vertical
        if layer_name.startswith('In') and layer_name.endswith('.Cu'):
            try:
                layer_num = int(layer_name[2:-3])  # Extract number from "In1.Cu"
                if layer_num % 2 == 1:  # In1, In3, In5, In7, In9 = horizontal
                    return LayerDirection.HORIZONTAL
                else:  # In2, In4, In6, In8, In10 = vertical
                    return LayerDirection.VERTICAL
            except ValueError:
                logger.warning(f"Could not parse layer number from {layer_name}, defaulting to vertical")
                return LayerDirection.VERTICAL
        
        # F.Cu should not be in routing layers, but default to horizontal
        if layer_name == 'F.Cu':
            logger.warning("F.Cu should not be in routing layers")
            return LayerDirection.HORIZONTAL
        
        # Unknown layers default to alternating pattern based on index
        index = self.name_to_index.get(layer_name, 0)
        return LayerDirection.HORIZONTAL if index % 2 == 0 else LayerDirection.VERTICAL
    
    def get_layer_direction(self, layer_index: int) -> LayerDirection:
        """Get direction for a layer by index."""
        return self.layer_directions.get(layer_index, LayerDirection.HORIZONTAL)
    
    def get_layer_direction_by_name(self, layer_name: str) -> LayerDirection:
        """Get direction for a layer by name."""
        index = self.name_to_index.get(layer_name)
        if index is not None:
            return self.get_layer_direction(index)
        return LayerDirection.HORIZONTAL
    
    def is_horizontal_layer(self, layer_index: int) -> bool:
        """Check if layer allows horizontal routing."""
        return self.get_layer_direction(layer_index) == LayerDirection.HORIZONTAL
    
    def is_vertical_layer(self, layer_index: int) -> bool:
        """Check if layer allows vertical routing."""
        return self.get_layer_direction(layer_index) == LayerDirection.VERTICAL
    
    def get_valid_movements(self, layer_index: int) -> List[tuple]:
        """Get valid movement directions for a layer."""
        direction = self.get_layer_direction(layer_index)
        
        if direction == LayerDirection.HORIZONTAL:
            return [(-1, 0), (1, 0)]  # Left, Right
        else:
            return [(0, -1), (0, 1)]  # Down, Up
    
    def can_move_in_direction(self, layer_index: int, dx: int, dy: int) -> bool:
        """Check if movement is allowed in layer direction."""
        valid_moves = self.get_valid_movements(layer_index)
        return (dx, dy) in valid_moves
    
    def get_preferred_layers_for_direction(self, dx: int, dy: int) -> List[int]:
        """Get layers that prefer the given movement direction."""
        preferred_layers = []
        
        for layer_index, direction in self.layer_directions.items():
            if direction == LayerDirection.HORIZONTAL and dx != 0 and dy == 0:
                preferred_layers.append(layer_index)
            elif direction == LayerDirection.VERTICAL and dx == 0 and dy != 0:
                preferred_layers.append(layer_index)
        
        return preferred_layers
    
    def get_adjacent_layers(self, layer_index: int) -> List[int]:
        """Get adjacent layers for via transitions."""
        adjacent = []
        
        if layer_index > 0:
            adjacent.append(layer_index - 1)
        
        if layer_index < len(self.layer_names) - 1:
            adjacent.append(layer_index + 1)
        
        return adjacent
    
    def get_statistics(self) -> Dict[str, any]:
        """Get layer direction statistics."""
        horizontal_count = sum(1 for d in self.layer_directions.values() 
                             if d == LayerDirection.HORIZONTAL)
        vertical_count = len(self.layer_directions) - horizontal_count
        
        return {
            'total_layers': len(self.layer_names),
            'horizontal_layers': horizontal_count,
            'vertical_layers': vertical_count,
            'layer_assignments': {
                self.index_to_name[i]: direction.value 
                for i, direction in self.layer_directions.items()
            }
        }