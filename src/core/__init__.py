"""
Core infrastructure components for the OrthoRoute autorouter.

This module provides the fundamental building blocks that are shared
across all routing algorithms:
- DRC rules management with KiCad integration
- GPU resource management and acceleration
- Board data interface and abstraction
"""
from core.drc_rules import DRCRules
from core.gpu_manager import GPUManager
from core.board_interface import BoardInterface

__all__ = [
    'DRCRules',
    'GPUManager', 
    'BoardInterface'
]
