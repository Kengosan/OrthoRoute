"""
Core infrastructure components for the OrthoRoute autorouter.

This module provides the fundamental building blocks that are shared
across all routing algorithms:
- DRC rules management with KiCad integration
- GPU resource management and acceleration
- Board data interface and abstraction
"""
import sys
import os

# Add src directory to Python path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try absolute imports first, fall back to relative
try:
    from core.drc_rules import DRCRules
    from core.gpu_manager import GPUManager
    from core.board_interface import BoardInterface
except ImportError:
    from .drc_rules import DRCRules
    from .gpu_manager import GPUManager
    from .board_interface import BoardInterface

__all__ = [
    'DRCRules',
    'GPUManager', 
    'BoardInterface'
]
