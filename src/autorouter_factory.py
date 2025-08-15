#!/usr/bin/env python3
"""
Autorouter Factory and Main Interface

Provides the main interface for the refactored modular autorouting system.
Creates and configures the appropriate routing engines and infrastructure.
"""
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from core.drc_rules import DRCRules
from core.gpu_manager import GPUManager
from core.board_interface import BoardInterface
from data_structures.grid_config import GridConfig
from routing_engines.lees_router import LeeRouter
from routing_engines.base_router import BaseRouter, RoutingStats

logger = logging.getLogger(__name__)


class RoutingAlgorithm(Enum):
    """Available routing algorithms"""
    LEE_WAVEFRONT = "lee_wavefront"
    MANHATTAN = "manhattan"  # Future implementation
    ASTAR = "astar"  # Future implementation


class AutorouterEngine:
    """
    Main autorouter engine that manages the modular routing system
    
    This is the new modular architecture that replaces the monolithic AutorouterEngine.
    It provides a clean interface while allowing multiple routing algorithms.
    """
    
    def __init__(self, board_data: Dict, kicad_interface, use_gpu: bool = True, 
                 progress_callback=None, track_callback=None):
        """
        Initialize the modular autorouter engine
        
        Args:
            board_data: Board geometry and component data
            kicad_interface: KiCad interface for DRC extraction
            use_gpu: Whether to enable GPU acceleration
            progress_callback: Callback for progress updates
            track_callback: Callback for real-time track updates
        """
        self.board_data = board_data
        self.kicad_interface = kicad_interface
        self.progress_callback = progress_callback
        self.track_callback = track_callback
        
        logger.info("🚀 Initializing Modular Autorouter Engine")
        
        # Initialize core infrastructure
        self._initialize_core_infrastructure(board_data, kicad_interface, use_gpu)
        
        # Initialize routing engines
        self._routing_engines = {}
        self._initialize_routing_engines()
        
        # Default routing algorithm
        self.current_algorithm = RoutingAlgorithm.LEE_WAVEFRONT
        self.current_router = self._routing_engines[self.current_algorithm]
        
        # Legacy compatibility properties
        self.routed_tracks = []
        self.routed_vias = []
        self.routing_stats = {
            'nets_routed': 0,
            'nets_failed': 0,
            'tracks_added': 0,
            'vias_added': 0,
            'total_length_mm': 0.0,
            'routing_time': 0.0
        }
        
        logger.info("✅ Modular Autorouter Engine initialized successfully")
        logger.info(f"   Available algorithms: {[alg.value for alg in RoutingAlgorithm]}")
        logger.info(f"   Current algorithm: {self.current_algorithm.value}")
    
    def _initialize_core_infrastructure(self, board_data: Dict, kicad_interface, use_gpu: bool):
        """Initialize the core infrastructure components"""
        
        # Grid configuration
        bounds = board_data.get('bounds', [-50, -50, 50, 50])
        self.grid_config = GridConfig(bounds, grid_resolution=0.1)
        
        # DRC rules with KiCad interface
        board_data_with_interface = board_data.copy()
        board_data_with_interface['kicad_interface'] = kicad_interface
        self.drc_rules = DRCRules(board_data_with_interface)
        
        # GPU manager
        self.gpu_manager = GPUManager(use_gpu=use_gpu)
        
        # Board interface
        self.board_interface = BoardInterface(board_data, kicad_interface, self.grid_config)
        
        logger.info("🏗️ Core infrastructure initialized:")
        logger.info(f"   Grid: {self.grid_config.width}x{self.grid_config.height} cells")
        logger.info(f"   GPU: {'Enabled' if self.gpu_manager.is_gpu_enabled() else 'Disabled'}")
        logger.info(f"   DRC: {len(self.drc_rules.netclasses)} netclasses")
        logger.info(f"   Board: {self.board_interface.stats['routable_nets']} routable nets")
    
    def _initialize_routing_engines(self):
        """Initialize available routing engines"""
        
        # Lee's Algorithm Router
        self._routing_engines[RoutingAlgorithm.LEE_WAVEFRONT] = LeeRouter(
            self.board_interface, 
            self.drc_rules, 
            self.gpu_manager, 
            self.grid_config
        )
        
        # Set callbacks for all routers
        for router in self._routing_engines.values():
            router.set_progress_callback(self.progress_callback)
            router.set_track_callback(self.track_callback)
        
        logger.info(f"🔧 Initialized {len(self._routing_engines)} routing engines")
    
    def set_routing_algorithm(self, algorithm: RoutingAlgorithm):
        """Switch to a different routing algorithm"""
        if algorithm not in self._routing_engines:
            raise ValueError(f"Routing algorithm {algorithm.value} not available")
        
        self.current_algorithm = algorithm
        self.current_router = self._routing_engines[algorithm]
        
        logger.info(f"🔄 Switched to {algorithm.value} routing algorithm")
    
    def route_single_net(self, net_name: str, timeout: float = 10.0) -> bool:
        """
        Route a single net using the current algorithm
        
        Args:
            net_name: Name of the net to route
            timeout: Maximum routing time in seconds
            
        Returns:
            True if routing succeeded, False otherwise
        """
        from .routing_engines.base_router import RoutingResult
        
        result = self.current_router.route_net(net_name, timeout)
        success = (result == RoutingResult.SUCCESS)
        
        if success:
            self._update_legacy_stats()
        
        return success
    
    def route_all_nets(self, timeout_per_net: float = 5.0, total_timeout: float = 300.0) -> Dict:
        """
        Route all nets on the board using the current algorithm
        
        Args:
            timeout_per_net: Maximum time per net in seconds
            total_timeout: Maximum total routing time in seconds
            
        Returns:
            Routing statistics dictionary
        """
        stats = self.current_router.route_all_nets(timeout_per_net, total_timeout)
        
        # Update legacy statistics format
        self._update_legacy_stats_from_router_stats(stats)
        
        return self._convert_stats_to_legacy_format(stats)
    
    def get_routable_nets(self) -> Dict[str, Dict]:
        """Get all nets that can be routed"""
        return self.board_interface.get_routable_nets()
    
    def get_routed_tracks(self) -> List[Dict]:
        """Get all routed tracks in KiCad format"""
        tracks = self.current_router.get_routed_tracks()
        self.routed_tracks = tracks  # Update legacy property
        return tracks
    
    def get_routed_vias(self) -> List[Dict]:
        """Get all routed vias in KiCad format"""
        vias = self.current_router.get_routed_vias()
        self.routed_vias = vias  # Update legacy property
        return vias
    
    def clear_routes(self):
        """Clear all routed segments"""
        for router in self._routing_engines.values():
            router.clear_routes()
        
        # Clear legacy properties
        self.routed_tracks = []
        self.routed_vias = []
        self.routing_stats = {
            'nets_routed': 0,
            'nets_failed': 0,
            'tracks_added': 0,
            'vias_added': 0,
            'total_length_mm': 0.0,
            'routing_time': 0.0
        }
        
        logger.info("🗑️ Cleared all routes from all routing engines")
    
    def get_routing_statistics(self) -> Dict:
        """Get current routing statistics"""
        stats = self.current_router.get_routing_statistics()
        return self._convert_stats_to_legacy_format(stats)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about available algorithms"""
        return {
            'available_algorithms': [alg.value for alg in RoutingAlgorithm],
            'current_algorithm': self.current_algorithm.value,
            'algorithm_descriptions': {
                RoutingAlgorithm.LEE_WAVEFRONT.value: "Lee's wavefront expansion with GPU acceleration",
                RoutingAlgorithm.MANHATTAN.value: "Manhattan routing (H/V grid layers) - Future",
                RoutingAlgorithm.ASTAR.value: "A* pathfinding algorithm - Future"
            }
        }
    
    def _update_legacy_stats(self):
        """Update legacy statistics from current router"""
        stats = self.current_router.get_routing_statistics()
        self._update_legacy_stats_from_router_stats(stats)
    
    def _update_legacy_stats_from_router_stats(self, stats: RoutingStats):
        """Update legacy stats format from router stats"""
        self.routing_stats = {
            'nets_routed': stats.nets_routed,
            'nets_failed': stats.nets_failed,
            'tracks_added': stats.tracks_added,
            'vias_added': stats.vias_added,
            'total_length_mm': stats.total_length_mm,
            'routing_time': stats.routing_time
        }
    
    def _convert_stats_to_legacy_format(self, stats: RoutingStats) -> Dict:
        """Convert router stats to legacy dictionary format"""
        return {
            'nets_routed': stats.nets_routed,
            'nets_failed': stats.nets_failed,
            'nets_attempted': stats.nets_attempted,
            'tracks_added': stats.tracks_added,
            'vias_added': stats.vias_added,
            'total_length_mm': stats.total_length_mm,
            'routing_time': stats.routing_time,
            'success_rate': stats.success_rate
        }
    
    # Legacy compatibility methods for existing code
    def _route_single_net(self, net_name: str, timeout: float = 10.0) -> bool:
        """Legacy compatibility wrapper"""
        return self.route_single_net(net_name, timeout)
    
    @property
    def layers(self) -> List[str]:
        """Legacy compatibility - get available layers"""
        return self.board_interface.get_layers()
    
    @property
    def use_gpu(self) -> bool:
        """Legacy compatibility - check if GPU is enabled"""
        return self.gpu_manager.is_gpu_enabled()
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'gpu_manager'):
            self.gpu_manager.cleanup()
        
        logger.info("🧹 Autorouter engine cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()


# Factory function for creating autorouter instances
def create_autorouter(board_data: Dict, kicad_interface, use_gpu: bool = True, 
                     algorithm: RoutingAlgorithm = RoutingAlgorithm.LEE_WAVEFRONT,
                     progress_callback=None, track_callback=None) -> AutorouterEngine:
    """
    Factory function to create an autorouter instance
    
    Args:
        board_data: Board geometry and component data
        kicad_interface: KiCad interface for DRC extraction
        use_gpu: Whether to enable GPU acceleration
        algorithm: Initial routing algorithm to use
        progress_callback: Callback for progress updates
        track_callback: Callback for real-time track updates
        
    Returns:
        Configured AutorouterEngine instance
    """
    engine = AutorouterEngine(
        board_data=board_data,
        kicad_interface=kicad_interface,
        use_gpu=use_gpu,
        progress_callback=progress_callback,
        track_callback=track_callback
    )
    
    if algorithm != RoutingAlgorithm.LEE_WAVEFRONT:
        engine.set_routing_algorithm(algorithm)
    
    return engine
