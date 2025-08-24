"""
Board utilities for Manhattan routing
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from .types import Pad

logger = logging.getLogger(__name__)

def get_pos_mm(pad_data: Dict) -> Tuple[float, float]:
    """Extract position from pad data (convert from IU to mm)"""
    # Board data has 'x' and 'y' in internal units, convert to mm
    x_iu = pad_data.get('x', 0.0)
    y_iu = pad_data.get('y', 0.0)
    
    # Convert from IU to mm (assuming 1 IU = 1 nm, so 1mm = 1e6 IU)
    IU_PER_MM = 1_000_000.0
    x_mm = x_iu / IU_PER_MM
    y_mm = y_iu / IU_PER_MM
    
    return x_mm, y_mm

def get_net_name(pad_data: Dict) -> Optional[str]:
    """Extract net name from pad data"""
    net_name = pad_data.get('net_name', '').strip()
    return net_name if net_name else None

def is_through_hole(pad_data: Dict) -> bool:
    """Check if pad is through hole"""
    return pad_data.get('is_through_hole', False)

def get_layer_set(pad_data: Dict) -> Union[Set[str], str]:
    """Extract layer set from pad data"""
    return pad_data.get('layer_set', {'F.Cu'})

def snapshot_board(board_data: Dict[str, Any]) -> Tuple[List[Pad], Dict[str, List[Pad]]]:
    """
    Create normalized snapshot of board for Manhattan routing
    
    Args:
        board_data: Raw board data dictionary
        
    Returns:
        Tuple of (all_pads, routable_nets)
    """
    logger.info("Creating board snapshot for Manhattan routing")
    
    all_pads = []
    routable_nets = {}
    
    # Extract pads from board data
    pads_data = board_data.get('pads', [])
    logger.info(f"Processing {len(pads_data)} pads from board")
    
    for pad_data in pads_data:
        try:
            # Extract pad information
            net_name = get_net_name(pad_data)
            if not net_name:
                continue  # Skip pads without nets
            
            x_mm, y_mm = get_pos_mm(pad_data)
            
            # Create normalized pad (convert dimensions from IU to mm)
            IU_PER_MM = 1_000_000.0
            width_mm = pad_data.get('width', 0.0) / IU_PER_MM
            height_mm = pad_data.get('height', 0.0) / IU_PER_MM
            
            pad = Pad(
                net_name=net_name,
                x_mm=x_mm,
                y_mm=y_mm,
                width_mm=width_mm,
                height_mm=height_mm,
                layer_set=get_layer_set(pad_data),
                is_through_hole=is_through_hole(pad_data)
            )
            
            all_pads.append(pad)
            
            # Group by net
            if net_name not in routable_nets:
                routable_nets[net_name] = []
            routable_nets[net_name].append(pad)
            
        except Exception as e:
            logger.warning(f"Error processing pad data: {e}")
            continue
    
    # Filter out single-pad nets
    routable_nets = {net_name: pads for net_name, pads in routable_nets.items() 
                    if len(pads) >= 2}
    
    logger.info(f"Board snapshot complete: {len(all_pads)} pads, {len(routable_nets)} routable nets")
    
    return all_pads, routable_nets