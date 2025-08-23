"""OrthoRoute - Advanced PCB Autorouter with Manhattan routing and GPU acceleration."""

__version__ = "1.0.0"
__author__ = "OrthoRoute Team"
__description__ = "Advanced PCB Autorouter with Manhattan routing and GPU acceleration"

# Core domain exports
from .domain.models.board import Board, Component, Net, Layer
from .domain.models.routing import Route, Segment, Via
from .domain.models.constraints import DRCConstraints, NetClass

# Service exports
from .domain.services.routing_engine import RoutingEngine
from .application.services.routing_orchestrator import RoutingOrchestrator

# Configuration
from .shared.configuration.config_manager import ConfigManager, get_config, initialize_config
from .shared.configuration.settings import ApplicationSettings

# Main application classes
from .presentation.plugin.kicad_plugin import KiCadPlugin

# Algorithm exports
from .algorithms.manhattan.manhattan_router import ManhattanRoutingEngine

# Infrastructure exports
from .infrastructure.kicad.ipc_adapter import KiCadIPCAdapter
from .infrastructure.gpu.cuda_provider import CUDAProvider
from .infrastructure.gpu.cpu_fallback import CPUProvider

__all__ = [
    # Version info
    "__version__", "__author__", "__description__",
    
    # Domain models
    "Board", "Component", "Net", "Layer",
    "Route", "Segment", "Via",
    "DRCConstraints", "NetClass",
    
    # Services
    "RoutingEngine", "RoutingOrchestrator",
    
    # Configuration
    "ConfigManager", "get_config", "initialize_config", "ApplicationSettings",
    
    # Applications
    "KiCadPlugin",
    
    # Algorithms
    "ManhattanRoutingEngine",
    
    # Infrastructure
    "KiCadIPCAdapter", "CUDAProvider", "CPUProvider"
]