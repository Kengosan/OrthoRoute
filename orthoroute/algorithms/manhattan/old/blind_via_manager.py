"""
Blind and Buried Via Manager
Handles advanced via types that don't go through all layers
"""

import cupy as cp
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ViaType(Enum):
    """Types of vias supported"""
    THROUGH = "through"      # Goes through all layers (F.Cu to B.Cu)
    BLIND = "blind"          # Starts/ends at outer layer, doesn't go through
    BURIED = "buried"        # Entirely internal, doesn't touch outer layers

@dataclass
class Via:
    """Represents any type of via"""
    id: str
    via_type: ViaType
    position: Tuple[float, float]
    start_layer: int         # 0 = F.Cu, 1 = In1.Cu, etc.
    end_layer: int          # Last layer this via connects
    drill_size: float       # Via drill diameter
    pad_size: float         # Via pad diameter
    cost: float             # Routing cost
    blocks_layers: Set[int] # Which layers this via occupies/blocks

class BlindViaManager:
    """Manages blind and buried vias for advanced PCB routing"""
    
    def __init__(self, layer_count: int, layer_thickness: float = 0.1):
        self.layer_count = layer_count
        self.layer_thickness = layer_thickness
        
        # Standard via properties
        self.min_via_drill = 0.15  # mm
        self.max_via_drill = 0.8   # mm
        self.standard_via_drill = 0.2  # mm
        self.via_pad_size = 0.4    # mm (pad around drill)
        
        # Via costs (blind/buried vias are more expensive to manufacture)
        self.through_via_cost = 1.0
        self.blind_via_cost = 1.5    # 50% more expensive
        self.buried_via_cost = 2.0   # 100% more expensive
        
        logger.info(f"Blind via manager initialized for {layer_count} layer PCB")
    
    def create_blind_via(self, position: Tuple[float, float], outer_layer: int, 
                        inner_layer: int, via_id: Optional[str] = None) -> Via:
        """Create blind via (outer layer to inner layer)"""
        
        if via_id is None:
            via_id = f"blind_via_{position[0]:.3f}_{position[1]:.3f}_{outer_layer}_{inner_layer}"
        
        # Determine start/end layers
        start_layer = min(outer_layer, inner_layer)
        end_layer = max(outer_layer, inner_layer)
        
        # Blind vias must touch at least one outer layer (0 or layer_count-1)
        if start_layer != 0 and end_layer != self.layer_count - 1:
            raise ValueError(f"Blind via must connect to outer layer: {outer_layer} -> {inner_layer}")
        
        # Calculate blocked layers
        blocked_layers = set(range(start_layer, end_layer + 1))
        
        via = Via(
            id=via_id,
            via_type=ViaType.BLIND,
            position=position,
            start_layer=start_layer,
            end_layer=end_layer,
            drill_size=self.standard_via_drill,
            pad_size=self.via_pad_size,
            cost=self.blind_via_cost,
            blocks_layers=blocked_layers
        )
        
        logger.debug(f"Created blind via {via_id}: L{start_layer}->L{end_layer}")
        return via
    
    def create_buried_via(self, position: Tuple[float, float], layer1: int, 
                         layer2: int, via_id: Optional[str] = None) -> Via:
        """Create buried via (inner layer to inner layer)"""
        
        if via_id is None:
            via_id = f"buried_via_{position[0]:.3f}_{position[1]:.3f}_{layer1}_{layer2}"
        
        # Determine start/end layers
        start_layer = min(layer1, layer2)
        end_layer = max(layer1, layer2)
        
        # Buried vias cannot touch outer layers
        if start_layer == 0 or end_layer == self.layer_count - 1:
            raise ValueError(f"Buried via cannot touch outer layers: {layer1} -> {layer2}")
        
        # Calculate blocked layers
        blocked_layers = set(range(start_layer, end_layer + 1))
        
        via = Via(
            id=via_id,
            via_type=ViaType.BURIED,
            position=position,
            start_layer=start_layer,
            end_layer=end_layer,
            drill_size=self.standard_via_drill,
            pad_size=self.via_pad_size,
            cost=self.buried_via_cost,
            blocks_layers=blocked_layers
        )
        
        logger.debug(f"Created buried via {via_id}: L{start_layer}->L{end_layer}")
        return via
    
    def create_through_via(self, position: Tuple[float, float], 
                          via_id: Optional[str] = None) -> Via:
        """Create through via (F.Cu to B.Cu)"""
        
        if via_id is None:
            via_id = f"through_via_{position[0]:.3f}_{position[1]:.3f}"
        
        # Through vias block all layers
        blocked_layers = set(range(self.layer_count))
        
        via = Via(
            id=via_id,
            via_type=ViaType.THROUGH,
            position=position,
            start_layer=0,
            end_layer=self.layer_count - 1,
            drill_size=self.standard_via_drill,
            pad_size=self.via_pad_size,
            cost=self.through_via_cost,
            blocks_layers=blocked_layers
        )
        
        logger.debug(f"Created through via {via_id}: L0->L{self.layer_count-1}")
        return via
    
    def get_optimal_via_type(self, from_layer: int, to_layer: int) -> ViaType:
        """Determine optimal via type for layer transition"""
        
        if from_layer == to_layer:
            raise ValueError("Cannot create via within same layer")
        
        # Sort layers
        start_layer = min(from_layer, to_layer)
        end_layer = max(from_layer, to_layer)
        
        # Check if it touches outer layers
        touches_top = start_layer == 0
        touches_bottom = end_layer == self.layer_count - 1
        
        if touches_top or touches_bottom:
            if touches_top and touches_bottom:
                return ViaType.THROUGH  # F.Cu to B.Cu
            else:
                return ViaType.BLIND   # Outer to inner layer
        else:
            return ViaType.BURIED      # Inner to inner layer
    
    def create_optimal_via(self, position: Tuple[float, float], from_layer: int, 
                          to_layer: int, via_id: Optional[str] = None) -> Via:
        """Create optimal via type for given layer transition"""
        
        via_type = self.get_optimal_via_type(from_layer, to_layer)
        
        if via_type == ViaType.THROUGH:
            return self.create_through_via(position, via_id)
        elif via_type == ViaType.BLIND:
            outer_layer = 0 if from_layer == 0 or to_layer == 0 else self.layer_count - 1
            inner_layer = to_layer if from_layer == outer_layer else from_layer
            return self.create_blind_via(position, outer_layer, inner_layer, via_id)
        elif via_type == ViaType.BURIED:
            return self.create_buried_via(position, from_layer, to_layer, via_id)
    
    def check_via_conflicts(self, new_via: Via, existing_vias: List[Via], 
                           min_spacing: float = 0.3) -> List[str]:
        """Check for conflicts between vias"""
        
        conflicts = []
        
        for existing_via in existing_vias:
            # Check spacing
            distance = self._calculate_distance(new_via.position, existing_via.position)
            if distance < min_spacing:
                conflicts.append(f"Too close to {existing_via.id}: {distance:.3f}mm < {min_spacing}mm")
                continue
            
            # Check layer conflicts - vias can share position if they don't overlap layers
            overlapping_layers = new_via.blocks_layers & existing_via.blocks_layers
            if overlapping_layers and distance < self.via_pad_size:
                conflicts.append(f"Layer overlap with {existing_via.id} on layers {overlapping_layers}")
        
        return conflicts
    
    def generate_via_alternatives(self, from_layer: int, to_layer: int, 
                                 position: Tuple[float, float]) -> List[Via]:
        """Generate alternative via options for layer transition"""
        
        alternatives = []
        
        # Direct via (optimal)
        direct_via = self.create_optimal_via(position, from_layer, to_layer, 
                                           f"direct_{from_layer}_{to_layer}")
        alternatives.append(direct_via)
        
        # Multi-hop alternatives for complex transitions
        if abs(from_layer - to_layer) > 3:  # Long layer transitions
            # Option 1: Via to middle layer, then another via
            middle_layer = (from_layer + to_layer) // 2
            
            via1 = self.create_optimal_via(position, from_layer, middle_layer, 
                                         f"hop1_{from_layer}_{middle_layer}")
            via2 = self.create_optimal_via(position, middle_layer, to_layer,
                                         f"hop2_{middle_layer}_{to_layer}")
            
            # Note: Multi-hop vias would need separate positions to avoid conflicts
            # This is just conceptual - in practice you'd offset the positions
            alternatives.extend([via1, via2])
        
        return alternatives
    
    def get_via_manufacturing_cost(self, vias: List[Via]) -> Dict[str, int]:
        """Calculate manufacturing complexity from via usage"""
        
        via_counts = {
            ViaType.THROUGH: 0,
            ViaType.BLIND: 0,
            ViaType.BURIED: 0
        }
        
        for via in vias:
            via_counts[via.via_type] += 1
        
        # Manufacturing cost factors
        cost_analysis = {
            'through_vias': via_counts[ViaType.THROUGH],
            'blind_vias': via_counts[ViaType.BLIND],
            'buried_vias': via_counts[ViaType.BURIED],
            'total_drill_hits': sum(via_counts.values()),
            'manufacturing_complexity': (
                via_counts[ViaType.THROUGH] * 1.0 +
                via_counts[ViaType.BLIND] * 1.5 +
                via_counts[ViaType.BURIED] * 2.0
            ),
            'requires_sequential_lamination': via_counts[ViaType.BURIED] > 0
        }
        
        logger.info(f"Via manufacturing analysis: {cost_analysis}")
        return cost_analysis
    
    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def convert_vias_to_rrg_nodes(self, vias: List[Via]) -> List[Dict]:
        """Convert vias to RRG nodes for routing"""
        
        nodes = []
        
        for via in vias:
            # Create a node for each layer the via connects
            for layer in range(via.start_layer, via.end_layer + 1):
                node = {
                    'id': f"{via.id}_L{layer}",
                    'type': 'via_node',
                    'via_type': via.via_type.value,
                    'layer': layer,
                    'position': via.position,
                    'parent_via': via.id,
                    'cost': via.cost / (via.end_layer - via.start_layer + 1)  # Distribute cost
                }
                nodes.append(node)
        
        logger.debug(f"Converted {len(vias)} vias to {len(nodes)} RRG nodes")
        return nodes
    
    def convert_vias_to_rrg_edges(self, vias: List[Via]) -> List[Dict]:
        """Convert vias to RRG edges for layer transitions"""
        
        edges = []
        
        for via in vias:
            # Create edges between consecutive layers in the via
            for layer in range(via.start_layer, via.end_layer):
                edge = {
                    'id': f"{via.id}_edge_L{layer}_L{layer+1}",
                    'from_node': f"{via.id}_L{layer}",
                    'to_node': f"{via.id}_L{layer+1}",
                    'type': 'via_transition',
                    'via_type': via.via_type.value,
                    'cost': via.cost
                }
                edges.append(edge)
        
        logger.debug(f"Converted {len(vias)} vias to {len(edges)} RRG edges")
        return edges