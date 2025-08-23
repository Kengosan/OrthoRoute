#!/usr/bin/env python3
"""
Visualization package for OrthoRoute

Provides visualization capabilities for routing algorithms including:
- Real-time routing progress display
- Manhattan routing visualization
- Track and via visualization
- Progress reporting
"""

from .manhattan_visualizer import ManhattanVisualizer, create_manhattan_visualizer

__all__ = ['ManhattanVisualizer', 'create_manhattan_visualizer']