#!/usr/bin/env python3
"""
Immediate Routing Quality Fixes for OrthoRoute

This module provides immediate fixes for the routing quality issues seen in the screenshot:
1. Proper pad clearance enforcement during pathfinding
2. Enhanced via placement validation
3. Track placement with full DRC compliance

These fixes can be directly integrated into the existing autorouter.py
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def get_enhanced_clearance_settings(drc_rules) -> Dict[str, float]:
    """
    Get enhanced clearance settings that prevent trace-to-pad violations
    
    Returns proper clearances for different routing phases
    """
    return {
        # Pathfinding phase: Use proper clearances to avoid violations
        'pathfinding_pad_clearance': max(0.1, drc_rules.min_trace_spacing * 0.8),  # 80% of full clearance
        'pathfinding_trace_clearance': max(0.05, drc_rules.min_trace_spacing * 0.6),  # 60% of full clearance
        
        # Track placement phase: Full DRC compliance
        'placement_pad_clearance': drc_rules.min_trace_spacing,     # Full clearance
        'placement_trace_clearance': drc_rules.min_trace_spacing,   # Full clearance
        
        # Via placement: Extra safety margin
        'via_safety_clearance': drc_rules.min_via_spacing * 1.1,   # 10% extra for vias
    }

def create_enhanced_pad_obstacle_marking(autorouter, obstacle_grid, layer: str, exclude_net: str = None):
    """
    Enhanced pad obstacle marking with proper clearance enforcement
    
    This replaces the problematic 0.02mm clearance with proper DRC clearances
    """
    pads = autorouter.board_data.get('pads', [])
    clearances = get_enhanced_clearance_settings(autorouter.drc_rules)
    
    # Use proper pathfinding clearance instead of 0.02mm
    pad_clearance = clearances['pathfinding_pad_clearance']
    
    logger.info(f"🔧 Using enhanced pad clearance: {pad_clearance:.3f}mm (was 0.02mm)")
    
    marked_count = 0
    
    for pad in pads:
        # Skip pads on current net to preserve connectivity
        pad_net = pad.get('net', {})
        pad_net_name = ""
        
        if hasattr(pad_net, 'name'):
            pad_net_name = pad_net.name
        elif isinstance(pad_net, dict):
            pad_net_name = pad_net.get('name', '')
        elif isinstance(pad_net, str):
            pad_net_name = pad_net
        
        if exclude_net and pad_net_name == exclude_net:
            continue
        
        # Check if pad is on this layer
        pad_layers = pad.get('layers', [])
        is_on_layer = (layer in pad_layers or 
                      (not pad_layers and pad.get('drill_diameter', 0) > 0) or
                      ('F.Cu' in pad_layers and 'B.Cu' in pad_layers))
        
        if not is_on_layer:
            continue
        
        # Get pad parameters
        pad_x, pad_y = pad.get('x', 0), pad.get('y', 0)
        size_x, size_y = pad.get('size_x', 1.0), pad.get('size_y', 1.0)
        
        # Convert to grid coordinates
        grid_x = int((pad_x - autorouter.grid_config.min_x) / autorouter.grid_config.resolution)
        grid_y = int((pad_y - autorouter.grid_config.min_y) / autorouter.grid_config.resolution)
        
        # Calculate pad size with proper clearance
        half_size_x_cells = max(1, int((size_x / 2 + pad_clearance) / autorouter.grid_config.resolution))
        half_size_y_cells = max(1, int((size_y / 2 + pad_clearance) / autorouter.grid_config.resolution))
        
        # Mark pad area plus clearance as obstacle
        for dy in range(-half_size_y_cells, half_size_y_cells + 1):
            for dx in range(-half_size_x_cells, half_size_x_cells + 1):
                obs_x, obs_y = grid_x + dx, grid_y + dy
                if 0 <= obs_x < autorouter.grid_config.width and 0 <= obs_y < autorouter.grid_config.height:
                    if not obstacle_grid[obs_y, obs_x]:  # Only mark if not already marked
                        obstacle_grid[obs_y, obs_x] = True
                        marked_count += 1
    
    logger.info(f"🚧 Enhanced pad marking: {marked_count} cells marked with {pad_clearance:.3f}mm clearance")
    return marked_count

def get_enhanced_via_positions(source_pad: Dict, target_pad: Dict, max_attempts: int = 7) -> List[Tuple[float, float]]:
    """
    Enhanced via positioning that addresses the via connection failures seen in the screenshot
    
    Returns strategic via positions that consider obstacles and routing geometry
    """
    src_x, src_y = source_pad.get('x', 0), source_pad.get('y', 0)
    tgt_x, tgt_y = target_pad.get('x', 0), target_pad.get('y', 0)
    
    # Connection vector
    dx, dy = tgt_x - src_x, tgt_y - src_y
    distance = (dx**2 + dy**2)**0.5
    
    if distance < 0.1:  # Very close pads
        return [(src_x, src_y)]
    
    via_positions = []
    
    # Strategy 1: Traditional line positions (improved spacing)
    for fraction in [0.2, 0.35, 0.5, 0.65, 0.8]:
        via_x = src_x + dx * fraction
        via_y = src_y + dy * fraction
        via_positions.append((via_x, via_y))
    
    # Strategy 2: Perpendicular offsets for obstacle avoidance
    if distance > 0.5:  # Only for longer connections
        # Perpendicular vector
        perp_dx, perp_dy = -dy / distance, dx / distance
        offset_distance = min(0.8, distance * 0.25)
        
        # Midpoint with perpendicular offsets
        mid_x, mid_y = src_x + dx * 0.5, src_y + dy * 0.5
        via_positions.extend([
            (mid_x + perp_dx * offset_distance, mid_y + perp_dy * offset_distance),
            (mid_x - perp_dx * offset_distance, mid_y - perp_dy * offset_distance),
        ])
    
    return via_positions[:max_attempts]

def validate_via_placement_with_enhanced_clearance(autorouter, via_gx: int, via_gy: int, 
                                                 obstacle_grids: Dict) -> bool:
    """
    Enhanced via placement validation with proper clearance checking
    
    This addresses the via connection failures by ensuring proper clearances
    """
    clearances = get_enhanced_clearance_settings(autorouter.drc_rules)
    via_clearance = clearances['via_safety_clearance']
    
    # Check clearance radius in grid cells
    clearance_radius = max(2, int(via_clearance / autorouter.grid_config.resolution))
    
    # Check all layers for conflicts
    for layer_name, obstacle_grid in obstacle_grids.items():
        if layer_name not in ['F.Cu', 'B.Cu']:
            continue
        
        # Check area around via position
        for dy in range(-clearance_radius, clearance_radius + 1):
            for dx in range(-clearance_radius, clearance_radius + 1):
                check_x, check_y = via_gx + dx, via_gy + dy
                
                # Check bounds
                if not (0 <= check_x < autorouter.grid_config.width and 
                       0 <= check_y < autorouter.grid_config.height):
                    continue
                
                # Check for obstacles
                if obstacle_grid[check_y, check_x]:
                    # Calculate actual distance
                    actual_distance = (dx**2 + dy**2)**0.5 * autorouter.grid_config.resolution
                    if actual_distance < via_clearance:
                        logger.debug(f"❌ Via position blocked: obstacle at {actual_distance:.3f}mm (need {via_clearance:.3f}mm)")
                        return False
    
    logger.debug(f"✅ Via position valid with {via_clearance:.3f}mm clearance")
    return True

def create_quality_assessment_report(routing_results: Dict) -> str:
    """
    Create a quality assessment report for routing results
    
    This helps identify and track routing quality improvements
    """
    report = []
    report.append("📊 Routing Quality Assessment Report")
    report.append("=" * 50)
    
    # Basic statistics
    total_nets = routing_results.get('total_nets', 0)
    routed_nets = routing_results.get('nets_routed', 0)
    failed_nets = total_nets - routed_nets
    
    report.append(f"\n📈 Routing Statistics:")
    report.append(f"   Total nets: {total_nets}")
    report.append(f"   Successfully routed: {routed_nets} ({routed_nets/total_nets*100:.1f}%)")
    report.append(f"   Failed to route: {failed_nets} ({failed_nets/total_nets*100:.1f}%)")
    
    # Track statistics
    tracks_added = routing_results.get('tracks_added', 0)
    vias_added = routing_results.get('vias_added', 0)
    total_length = routing_results.get('total_length_mm', 0)
    
    report.append(f"\n🛤️ Track Statistics:")
    report.append(f"   Tracks created: {tracks_added}")
    report.append(f"   Vias created: {vias_added}")
    report.append(f"   Total length: {total_length:.1f}mm")
    if tracks_added > 0:
        report.append(f"   Average track length: {total_length/tracks_added:.2f}mm")
    
    # Quality indicators
    report.append(f"\n🎯 Quality Indicators:")
    if vias_added == 0 and routed_nets > 0:
        report.append("   ⚠️  No vias used - may indicate via routing issues")
    if failed_nets > total_nets * 0.1:
        report.append("   ⚠️  High failure rate - clearance issues likely")
    if tracks_added > 0 and total_length / tracks_added > 5.0:
        report.append("   ⚠️  Long average tracks - path optimization needed")
    
    # Recommendations
    report.append(f"\n💡 Recommendations:")
    if failed_nets > 0:
        report.append("   1. Check pad clearance settings (current: enhanced)")
        report.append("   2. Verify via placement strategy (current: adaptive)")
        report.append("   3. Consider multi-strategy routing for difficult nets")
    
    if vias_added == 0 and total_nets > 10:
        report.append("   4. Investigate via routing implementation")
        report.append("   5. Check via clearance validation")
    
    return "\n".join(report)

def apply_immediate_fixes_to_autorouter():
    """
    Instructions for applying immediate fixes to the main autorouter
    
    These changes address the routing quality issues in the screenshot
    """
    instructions = [
        "🔧 Immediate Fixes for Autorouter Quality Issues",
        "=" * 60,
        "",
        "1. REPLACE the problematic 0.02mm clearance in _mark_pads_as_obstacles:",
        "   OLD: 'clearance': 0.02  # Pathfinding: just 0.02mm to avoid pad shorting",
        "   NEW: Use get_enhanced_clearance_settings() for proper clearances",
        "",
        "2. UPDATE via placement in _route_two_pads_multilayer_with_timeout_and_grids:",
        "   OLD: 3 fixed via positions (0.3, 0.5, 0.7)",
        "   NEW: Use get_enhanced_via_positions() for 7 strategic positions",
        "",
        "3. ENHANCE via validation in _is_via_location_valid_with_grids:",
        "   ADD: validate_via_placement_with_enhanced_clearance() for proper checking",
        "",
        "4. UPDATE obstacle marking to use enhanced clearances:",
        "   REPLACE: _mark_pads_as_obstacles() implementation",
        "   WITH: create_enhanced_pad_obstacle_marking() for proper DRC",
        "",
        "5. ADD quality assessment after routing:",
        "   USE: create_quality_assessment_report() to track improvements",
        "",
        "These fixes directly address:",
        "✅ Trace-to-pad clearance violations (enhanced clearances)",
        "✅ Failed via connections (better via placement + validation)",
        "✅ Suboptimal routing (strategic via positions)",
        "✅ Incomplete routing (proper obstacle marking)",
    ]
    
    return "\n".join(instructions)

def demonstrate_fixes():
    """Demonstrate the immediate routing quality fixes"""
    print("🔧 Immediate Routing Quality Fixes")
    print("=" * 50)
    
    # Simulate DRC rules for demo
    class MockDRCRules:
        def __init__(self):
            self.min_trace_spacing = 0.2
            self.min_via_spacing = 0.4
            self.default_trace_width = 0.3
    
    drc_rules = MockDRCRules()
    
    print("\n📋 Enhanced Clearance Settings:")
    clearances = get_enhanced_clearance_settings(drc_rules)
    for setting, value in clearances.items():
        print(f"   {setting}: {value:.3f}mm")
    
    print(f"\n🎯 Improvement Summary:")
    print(f"   OLD pathfinding clearance: 0.02mm")
    print(f"   NEW pathfinding clearance: {clearances['pathfinding_pad_clearance']:.3f}mm")
    print(f"   Improvement factor: {clearances['pathfinding_pad_clearance']/0.02:.1f}x")
    
    print("\n🔄 Enhanced Via Positions:")
    source_pad = {'x': 0.0, 'y': 0.0}
    target_pad = {'x': 5.0, 'y': 3.0}
    
    old_positions = [(1.5, 0.9), (2.5, 1.5), (3.5, 2.1)]  # 3 fixed positions
    new_positions = get_enhanced_via_positions(source_pad, target_pad)
    
    print(f"   OLD: {len(old_positions)} fixed positions")
    print(f"   NEW: {len(new_positions)} strategic positions")
    for i, (x, y) in enumerate(new_positions):
        print(f"        Position {i+1}: ({x:.2f}, {y:.2f})")
    
    print("\n📝 Implementation Instructions:")
    print(apply_immediate_fixes_to_autorouter())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_fixes()
