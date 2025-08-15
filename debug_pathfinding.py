#!/usr/bin/env python3
"""
Debug the pathfinding issue with simplified routing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import logging
import numpy as np
from autorouter import AutorouterEngine

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_pathfinding():
    """Debug why pathfinding is failing"""
    
    print("🔍 Debugging Pathfinding Issue")
    print("=" * 40)
    
    # Extremely simple test case
    test_board_data = {
        'pads': [
            # Just two pads, nothing else!
            {'x': 10.0, 'y': 10.0, 'size_x': 1.0, 'size_y': 1.0, 'net': {'name': 'TEST'}, 
             'layers': ['F.Cu'], 'drill': 0.8},
            {'x': 15.0, 'y': 10.0, 'size_x': 1.0, 'size_y': 1.0, 'net': {'name': 'TEST'}, 
             'layers': ['F.Cu'], 'drill': 0.8},
        ],
        'tracks': [], 'vias': [], 'zones': [],
        'nets': {'TEST': {'name': 'TEST', 'pads': [0, 1]}},
        'bounds': (0.0, 0.0, 25.0, 20.0)
    }
    
    class MockKiCadInterface:
        def add_track(self, track): pass
        def add_via(self, via): pass
        def refresh_display(self): pass
    
    autorouter = AutorouterEngine(test_board_data, MockKiCadInterface())
    
    print(f"📋 Setup:")
    print(f"   Grid: {autorouter.grid_config.width}x{autorouter.grid_config.height}")
    
    src_grid_x, src_grid_y = autorouter.grid_config.world_to_grid(10.0, 10.0)
    tgt_grid_x, tgt_grid_y = autorouter.grid_config.world_to_grid(15.0, 10.0)
    
    print(f"   From: (10, 10) -> Grid ({src_grid_x}, {src_grid_y})")
    print(f"   To: (15, 10) -> Grid ({tgt_grid_x}, {tgt_grid_y})")
    
    # Initialize and get obstacle grid
    autorouter._initialize_obstacle_grids()
    
    net_constraints = {'trace_width': 0.25, 'clearance': 0.1, 'via_size': 0.6, 'via_drill': 0.3}
    
    # Create net-specific grid
    obstacle_grid = autorouter._create_net_specific_obstacle_grid('F.Cu', 'TEST', net_constraints)
    
    # Convert coordinates
    src_x = int(src_grid_x)
    src_y = int(src_grid_y)
    tgt_x = int(tgt_grid_x)
    tgt_y = int(tgt_grid_y)
    
    print(f"🎯 Grid coordinates: Source=({src_x}, {src_y}), Target=({tgt_x}, {tgt_y})")
    
    # Check obstacle grid around source and target
    print(f"🔍 Obstacle grid inspection:")
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            check_x, check_y = src_x + dx, src_y + dy
            if 0 <= check_x < autorouter.grid_config.width and 0 <= check_y < autorouter.grid_config.height:
                is_obstacle = bool(obstacle_grid[check_y, check_x])
                marker = "🚫" if is_obstacle else "✅"
                if dx == 0 and dy == 0:
                    marker = "🎯"  # Source
                print(f"   ({check_x:3d}, {check_y:3d}): {marker}")
            else:
                print(f"   ({check_x:3d}, {check_y:3d}): 🌎 (out of bounds)")
    
    # Check if source can expand
    print(f"\n🔍 Can source expand?")
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_x, new_y = src_x + dx, src_y + dy
        if 0 <= new_x < autorouter.grid_config.width and 0 <= new_y < autorouter.grid_config.height:
            is_blocked = bool(obstacle_grid[new_y, new_x])
            direction = {(0,1): "→", (0,-1): "←", (1,0): "↓", (-1,0): "↑"}[dy, dx]
            status = "BLOCKED" if is_blocked else "FREE"
            print(f"   {direction} ({new_x}, {new_y}): {status}")
        else:
            direction = {(0,1): "→", (0,-1): "←", (1,0): "↓", (-1,0): "↑"}[dy, dx]
            print(f"   {direction} OUT OF BOUNDS")
    
    # Test actual pathfinding
    print(f"\n🎯 Testing pathfinding...")
    import time
    start_time = time.time()
    path = autorouter._lee_algorithm_gpu_with_timeout(src_x, src_y, tgt_x, tgt_y, obstacle_grid, timeout=30.0, start_time=start_time)
    
    if path:
        print(f"✅ Path found! Length: {len(path)}")
        for i, (x, y) in enumerate(path[:10]):  # Show first 10 points
            print(f"   {i}: ({x}, {y})")
        if len(path) > 10:
            print(f"   ... and {len(path)-10} more points")
    else:
        print(f"❌ No path found")
    
    return path is not None

if __name__ == "__main__":
    debug_pathfinding()
