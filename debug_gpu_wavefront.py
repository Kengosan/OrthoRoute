#!/usr/bin/env python3
"""
Debug script to test GPU wavefront expansion in isolation
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_gpu_wavefront():
    """Test GPU wavefront expansion in isolation"""
    
    if not HAS_CUPY:
        print("CuPy not available - cannot test GPU")
        return
    
    print("🔥 Testing GPU wavefront expansion...")
    
    # Create a simple 10x10 grid
    height, width = 10, 10
    
    # Initialize grids
    distance_grid = cp.full((height, width), -1, dtype=cp.int32)
    obstacle_grid = cp.zeros((height, width), dtype=cp.bool_)
    current_wave = cp.zeros((height, width), dtype=cp.bool_)
    next_wave = cp.zeros((height, width), dtype=cp.bool_)
    
    # Add some obstacles
    obstacle_grid[5, 3:7] = True  # Horizontal wall
    
    # Set source
    src_x, src_y = 1, 1
    tgt_x, tgt_y = 8, 8
    
    distance_grid[src_y, src_x] = 0
    current_wave[src_y, src_x] = True
    
    print(f"Source: ({src_x}, {src_y}), Target: ({tgt_x}, {tgt_y})")
    print("Initial state:")
    print("Distance grid:")
    print(cp.asnumpy(distance_grid))
    print("Obstacles:")
    print(cp.asnumpy(obstacle_grid).astype(int))
    print("Current wave:")
    print(cp.asnumpy(current_wave).astype(int))
    
    # Run wavefront expansion
    distance = 0
    max_iterations = width + height
    
    for iteration in range(max_iterations):
        if not cp.any(current_wave):
            print(f"No more cells to expand at iteration {iteration}")
            break
        
        distance += 1
        next_wave.fill(False)
        
        # GPU wavefront expansion
        print(f"\n--- Iteration {iteration + 1} (distance {distance}) ---")
        
        # Create convolution kernel for 4-connected neighbors
        kernel = cp.array([[0, 1, 0],
                          [1, 0, 1], 
                          [0, 1, 0]], dtype=cp.bool_)
        
        # Find all neighbors of current wave front
        from cupyx.scipy import ndimage
        neighbor_mask = ndimage.binary_dilation(current_wave, kernel, border_value=False)
        
        print("Neighbor mask:")
        print(cp.asnumpy(neighbor_mask).astype(int))
        
        # Only consider unvisited cells that aren't obstacles
        valid_expansion = (neighbor_mask & 
                          (distance_grid == -1) & 
                          (~obstacle_grid))
        
        print("Valid expansion:")
        print(cp.asnumpy(valid_expansion).astype(int))
        
        # Set distance for newly reached cells
        distance_grid[valid_expansion] = distance
        
        # Update next wave
        next_wave[:] = valid_expansion
        
        print("Updated distance grid:")
        print(cp.asnumpy(distance_grid))
        
        # Check if target reached
        if next_wave[tgt_y, tgt_x]:
            print(f"🎯 Target reached at iteration {iteration + 1}!")
            break
        
        # Swap waves
        current_wave, next_wave = next_wave, current_wave
        
        active_cells = int(cp.sum(current_wave))
        print(f"Active cells for next iteration: {active_cells}")
    
    print("\nFinal distance grid:")
    print(cp.asnumpy(distance_grid))
    
    if distance_grid[tgt_y, tgt_x] > 0:
        print(f"✅ Path found! Distance: {distance_grid[tgt_y, tgt_x]}")
        return True
    else:
        print("❌ No path found")
        return False

if __name__ == "__main__":
    test_gpu_wavefront()
