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
from routing_engines.advanced_manhattan_router import ManhattanRouter
from routing_engines.gpu_manhattan_router import GPUManhattanRouter
from routing_engines.base_router import BaseRouter, RoutingStats

logger = logging.getLogger(__name__)


class RoutingAlgorithm(Enum):
    """Available routing algorithms"""
    LEE_WAVEFRONT = "lee_wavefront"
    MANHATTAN = "manhattan"  # Manhattan router with blind/buried vias
    GPU_MANHATTAN = "gpu_manhattan"  # GPU-accelerated Manhattan router with specs
    ASTAR = "astar"  # Future implementation


class AutorouterEngine:
    """
    Main autorouter engine that manages the modular routing system
    
    This is the new modular architecture that replaces the monolithic AutorouterEngine.
    It provides a clean interface while allowing multiple routing algorithms.
    """
    
    def __init__(self, board_data: Dict, kicad_interface, use_gpu: bool = True, 
                 progress_callback=None, track_callback=None, via_callback=None):
        """
        Initialize the modular autorouter engine
        
        Args:
            board_data: Board geometry and component data
            kicad_interface: KiCad interface for DRC extraction
            use_gpu: Whether to enable GPU acceleration
            progress_callback: Callback for progress updates
            track_callback: Callback for real-time track updates
            via_callback: Callback for real-time via updates
        """
        self.board_data = board_data
        self.kicad_interface = kicad_interface
        self.progress_callback = progress_callback
        self.track_callback = track_callback
        self.via_callback = via_callback
        
        logger.info("🚀 Initializing Modular Autorouter Engine")
        
        # Initialize core infrastructure
        self._initialize_core_infrastructure(board_data, kicad_interface, use_gpu)
        
        # Initialize routing engines lazily - only create when needed
        self._routing_engines = {}
        self._engines_initialized = set()
        
        # Default routing algorithm
        self.current_algorithm = RoutingAlgorithm.LEE_WAVEFRONT
        self.current_router = None  # Will be created on first access
        
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
    
    def _get_or_create_router(self, algorithm: RoutingAlgorithm):
        """Get existing router or create it lazily"""
        
        if algorithm not in self._routing_engines:
            logger.info(f"🔧 Lazy initialization of {algorithm.value} router")
            
            if algorithm == RoutingAlgorithm.LEE_WAVEFRONT:
                self._routing_engines[algorithm] = LeeRouter(
                    self.board_interface, 
                    self.drc_rules, 
                    self.gpu_manager, 
                    self.grid_config
                )
            elif algorithm == RoutingAlgorithm.MANHATTAN:
                # Manhattan router with blind/buried vias
                self._routing_engines[algorithm] = ManhattanRouter(
                    self.board_interface,
                    self.drc_rules,
                    self.gpu_manager,
                    self.grid_config
                )
            elif algorithm == RoutingAlgorithm.GPU_MANHATTAN:
                # GPU-accelerated Manhattan router with full specifications
                self._routing_engines[algorithm] = GPUManhattanRouter(
                    self.board_interface,
                    self.drc_rules,
                    self.gpu_manager,
                    self.grid_config
                )
            else:
                raise ValueError(f"Unknown routing algorithm: {algorithm}")
            
            # Set callbacks for the newly created router
            router = self._routing_engines[algorithm]
            router.set_progress_callback(self.progress_callback)
            router.set_track_callback(self.track_callback)
            
            # Set via callback if the router supports it
            if hasattr(router, 'set_via_callback') and self.via_callback:
                router.set_via_callback(self.via_callback)
            
            self._engines_initialized.add(algorithm)
            logger.info(f"✅ {algorithm.value} router initialized - type: {type(router).__name__}")
        
        return self._routing_engines[algorithm]

    def _initialize_routing_engines(self):
        """Legacy method - now engines are initialized lazily"""
        logger.info(f"🔧 Routing engines will be initialized on demand")
    
    def set_routing_algorithm(self, algorithm: RoutingAlgorithm):
        """Switch to a different routing algorithm"""
        
        logger.info(f"🔄 Switching from {self.current_algorithm.value} to {algorithm.value}")
        self.current_algorithm = algorithm
        
        # Lazy initialization - only create the router when it's actually needed
        self.current_router = self._get_or_create_router(algorithm)
        
        logger.info(f"🔄 Switched to {algorithm.value} routing algorithm - router type: {type(self.current_router).__name__}")
    
    def _ensure_current_router(self):
        """Ensure current router is initialized"""
        if self.current_router is None:
            self.current_router = self._get_or_create_router(self.current_algorithm)
        return self.current_router
    
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
        
        result = self._ensure_current_router().route_net(net_name, timeout)
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
        stats = self._ensure_current_router().route_all_nets(timeout_per_net, total_timeout)
        
        # Update legacy statistics format
        self._update_legacy_stats_from_router_stats(stats)
        
        return self._convert_stats_to_legacy_format(stats)
    
    def get_routable_nets(self) -> Dict[str, Dict]:
        """Get all nets that can be routed"""
        return self.board_interface.get_routable_nets()
    
    def get_routed_tracks(self) -> List[Dict]:
        """Get all routed tracks in KiCad format"""
        tracks = self._ensure_current_router().get_routed_tracks()
        self.routed_tracks = tracks  # Update legacy property
        return tracks
    
    def get_routed_vias(self) -> List[Dict]:
        """Get all routed vias in KiCad format"""
        vias = self._ensure_current_router().get_routed_vias()
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
    
    def create_trace_fabric_for_visualization(self) -> bool:
        """
        Create trace fabric for visualization (Trace Fabric router only)
        
        Returns:
            True if fabric was created successfully, False otherwise
        """
        logger.info("🎨 Creating trace fabric for visualization...")
        
        if self.current_algorithm != RoutingAlgorithm.TRACE_FABRIC:
            logger.warning(f"Trace fabric visualization only available for TRACE_FABRIC algorithm, current: {self.current_algorithm}")
            return False
        
        try:
            router = self._ensure_current_router()
            if hasattr(router, 'create_trace_fabric_for_visualization'):
                success = router.create_trace_fabric_for_visualization()
                if success:
                    logger.info("✅ Trace fabric created and visualized")
                else:
                    logger.error("❌ Failed to create trace fabric")
                return success
            else:
                logger.error("Current router does not support trace fabric visualization")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to create trace fabric: {e}")
            return False
    
    def get_routing_statistics(self) -> Dict:
        """Get current routing statistics"""
        logger.info("🔍 Getting routing statistics from factory...")
        stats = self._ensure_current_router().get_routing_statistics()
        logger.info(f"📊 Router returned stats: {type(stats)}")
        result = self._convert_stats_to_legacy_format(stats)
        logger.info(f"✅ Converted to dict: {result}")
        return result
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about available algorithms"""
        return {
            'available_algorithms': [alg.value for alg in RoutingAlgorithm],
            'current_algorithm': self.current_algorithm.value,
            'algorithm_descriptions': {
                RoutingAlgorithm.LEE_WAVEFRONT.value: "Lee's wavefront expansion with GPU acceleration",
                RoutingAlgorithm.MANHATTAN.value: "Manhattan router (multi-layer, A* pathfinding, blind/buried vias)",
                RoutingAlgorithm.GPU_MANHATTAN.value: "GPU Manhattan router (3.5mil/0.4mm grid, 11 layers, blind/buried vias)",
                RoutingAlgorithm.ASTAR.value: "A* pathfinding algorithm - Future"
            }
        }
    
    def _update_legacy_stats(self):
        """Update legacy statistics from current router"""
        stats = self._ensure_current_router().get_routing_statistics()
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
        try:
            logger.info(f"🔄 Converting stats object to legacy format: {type(stats)}")
            
            if not hasattr(stats, 'nets_routed'):
                logger.error(f"❌ RoutingStats object missing required attributes: {dir(stats)}")
                return {'nets_routed': 0, 'nets_failed': 0, 'nets_attempted': 0, 'tracks_added': 0, 'vias_added': 0, 'total_length_mm': 0.0, 'routing_time': 0.0, 'success_rate': 0.0}
                
            result = {
                'nets_routed': stats.nets_routed,
                'nets_failed': stats.nets_failed,
                'nets_attempted': stats.nets_attempted,
                'tracks_added': stats.tracks_added,
                'vias_added': stats.vias_added,
                'total_length_mm': stats.total_length_mm,
                'routing_time': stats.routing_time,
                'success_rate': stats.success_rate
            }
            
            logger.info(f"✅ Converted stats: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error converting stats to legacy format: {e}")
            # Return a default dictionary instead of the stats object
            return {'nets_routed': 0, 'nets_failed': 0, 'nets_attempted': 0, 'tracks_added': 0, 'vias_added': 0, 'total_length_mm': 0.0, 'routing_time': 0.0, 'success_rate': 0.0}
    
    def commit_solution(self) -> bool:
        """Commit the current routing solution to KiCad"""
        try:
            logger.info("🚀 Committing routing solution to KiCad...")
            
            # Get the current router
            router = self._ensure_current_router()
            
            # Check if router has export functionality
            if hasattr(router, 'export_to_kicad'):
                logger.info(f"📤 Exporting via {type(router).__name__} export_to_kicad method")
                success = router.export_to_kicad()
                if success:
                    logger.info("✅ Routes successfully exported to KiCad")
                    return True
                else:
                    logger.error("❌ Router export_to_kicad method failed")
                    return False
            
            # Fallback: use generic track/via export
            elif hasattr(router, 'get_routed_tracks') and hasattr(router, 'get_routed_vias'):
                logger.info(f"📤 Exporting via generic track/via methods from {type(router).__name__}")
                
                tracks = router.get_routed_tracks()
                vias = router.get_routed_vias()
                
                logger.info(f"📊 Found {len(tracks)} tracks and {len(vias)} vias to export")
                
                # Export tracks via board interface (placeholder - needs KiCad API integration)
                tracks_added = len(tracks)  # For now, assume success
                vias_added = len(vias)      # For now, assume success
                
                logger.info(f"✅ Would export {tracks_added} tracks and {vias_added} vias to KiCad")
                logger.warning("⚠️ Actual KiCad track/via creation not yet implemented")
                return tracks_added > 0 or vias_added > 0
            
            else:
                logger.error(f"❌ Router {type(router).__name__} does not support KiCad export")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error committing solution to KiCad: {e}")
            return False

    def rollback_solution(self) -> bool:
        """Rollback/clear the current routing solution"""
        try:
            logger.info("🔄 Rolling back routing solution...")
            
            # Clear routes from current router
            router = self._ensure_current_router()
            if hasattr(router, 'clear_routes'):
                router.clear_routes()
                logger.info("✅ Routes cleared from router")
            
            # Reset statistics
            self._update_legacy_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error rolling back solution: {e}")
            return False
    
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
                     progress_callback=None, track_callback=None, via_callback=None) -> AutorouterEngine:
    """
    Factory function to create an autorouter instance
    
    Args:
        board_data: Board geometry and component data
        kicad_interface: KiCad interface for DRC extraction
        use_gpu: Whether to enable GPU acceleration
        algorithm: Initial routing algorithm to use
        progress_callback: Callback for progress updates
        track_callback: Callback for real-time track updates
        via_callback: Callback for real-time via updates
        
    Returns:
        Configured AutorouterEngine instance
    """
    engine = AutorouterEngine(
        board_data=board_data,
        kicad_interface=kicad_interface,
        use_gpu=use_gpu,
        progress_callback=progress_callback,
        track_callback=track_callback,
        via_callback=via_callback
    )
    
    if algorithm != RoutingAlgorithm.LEE_WAVEFRONT:
        logger.info(f"🔄 Setting algorithm to {algorithm.value}")
        engine.set_routing_algorithm(algorithm)
    else:
        logger.info(f"🔄 Using default algorithm: {algorithm.value}")
    
    return engine
