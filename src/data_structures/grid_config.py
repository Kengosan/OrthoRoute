#!/usr/bin/env python3
"""
Grid Configuration and Utilities

Handles the conversion between world coordinates and grid coordinates for routing.
Provides fundamental grid operations used across all routing algorithms.
"""
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


class GridConfig:
    """Configuration for the routing grid with coordinate conversion utilities"""
    
    def __init__(self, board_bounds: Tuple[float, float, float, float], grid_resolution: float = 0.1):
        """
        Initialize grid configuration
        
        Args:
            board_bounds: (min_x, min_y, max_x, max_y) in mm
            grid_resolution: mm per grid cell
        """
        self.min_x, self.min_y, self.max_x, self.max_y = board_bounds
        self.resolution = grid_resolution  # mm per grid cell
        
        # Calculate grid dimensions
        self.width = int(np.ceil((self.max_x - self.min_x) / self.resolution))
        self.height = int(np.ceil((self.max_y - self.min_y) / self.resolution))
        
        logger.info(f"Grid: {self.width}x{self.height} cells at {self.resolution}mm resolution")
        logger.info(f"Board bounds: ({self.min_x:.2f}, {self.min_y:.2f}) to ({self.max_x:.2f}, {self.max_y:.2f}) mm")
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.min_x) / self.resolution)
        grid_y = int((y - self.min_y) / self.resolution)
        return np.clip(grid_x, 0, self.width - 1), np.clip(grid_y, 0, self.height - 1)
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x = self.min_x + grid_x * self.resolution
        y = self.min_y + grid_y * self.resolution
        return x, y
    
    def is_valid_grid_position(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid position is within bounds"""
        return 0 <= grid_x < self.width and 0 <= grid_y < self.height
    
    def get_grid_dimensions(self) -> Tuple[int, int]:
        """Get grid dimensions as (width, height)"""
        return self.width, self.height
    
    def get_total_cells(self) -> int:
        """Get total number of cells in the grid"""
        return self.width * self.height
