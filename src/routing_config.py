#!/usr/bin/env python3
"""
Centralized configuration management for OrthoRoute autorouter

This module provides a centralized configuration system for routing parameters,
performance settings, and algorithm tuning, addressing the configuration 
management improvement identified in code analysis.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class RoutingConfig:
    """Centralized configuration for routing parameters"""
    
    # Grid and Resolution Settings
    grid_resolution: float = 0.1  # mm per grid cell
    max_grid_size: int = 2000     # Maximum grid dimension (safety limit)
    
    # Routing Strategy Timeouts
    timeout_per_net: float = 5.0          # seconds per net
    timeout_strategy_1: float = 0.4       # 40% for single-layer on best layer
    timeout_strategy_2: float = 0.4       # 40% for multi-layer with vias
    timeout_strategy_3: float = 0.2       # 20% for fallback single-layer
    
    # Via Routing Configuration
    max_via_attempts: int = 3             # Number of via locations to try
    via_placement_positions: list = None  # Strategic via positions (0.3, 0.5, 0.7)
    enable_proactive_vias: bool = True    # Use enhanced via-aware routing
    
    # Performance Settings
    gpu_memory_limit: float = 0.8         # Fraction of GPU memory to use
    max_pathfinding_iterations: int = 10000  # Lee's algorithm iteration limit
    enable_gpu_acceleration: bool = True   # Allow GPU usage when available
    
    # Emergency Mode Thresholds
    emergency_net_threshold: int = 5000   # Switch to emergency mode above this
    large_design_threshold: int = 1000    # Consider "large" above this
    
    # Debug and Logging
    enable_debug_timing: bool = True      # Log performance metrics
    enable_pathfinding_debug: bool = False  # Detailed Lee's algorithm logging
    log_routing_decisions: bool = True    # Log strategy selections
    
    # DRC and Safety
    safety_clearance_pathfinding: float = 0.0  # Zero-clearance for connectivity
    minimum_track_width: float = 0.1      # mm, absolute minimum
    minimum_via_size: float = 0.4         # mm, absolute minimum
    
    def __post_init__(self):
        """Initialize computed fields and validate configuration"""
        if self.via_placement_positions is None:
            self.via_placement_positions = [0.3, 0.5, 0.7]  # Default strategic positions
        
        # Validate timeouts sum to 1.0
        total_timeout = self.timeout_strategy_1 + self.timeout_strategy_2 + self.timeout_strategy_3
        if abs(total_timeout - 1.0) > 0.01:
            logger.warning(f"⚠️ Timeout strategies sum to {total_timeout:.3f}, not 1.0")
        
        # Validate ranges
        if self.grid_resolution <= 0 or self.grid_resolution > 1.0:
            raise ValueError(f"Invalid grid resolution: {self.grid_resolution}mm")
        
        if self.gpu_memory_limit <= 0 or self.gpu_memory_limit > 1.0:
            raise ValueError(f"Invalid GPU memory limit: {self.gpu_memory_limit}")
        
        logger.info(f"🔧 Routing configuration initialized:")
        logger.info(f"   Grid: {self.grid_resolution}mm resolution")
        logger.info(f"   Strategy timeouts: {self.timeout_strategy_1:.0%}/{self.timeout_strategy_2:.0%}/{self.timeout_strategy_3:.0%}")
        logger.info(f"   Via settings: {len(self.via_placement_positions)} positions, proactive={self.enable_proactive_vias}")

@dataclass 
class PerformanceConfig:
    """Performance monitoring and optimization settings"""
    
    # Grid Optimization
    enable_incremental_grids: bool = True     # Use optimized grid updates
    enable_pad_net_caching: bool = True       # Cache pad-to-net mappings
    grid_copy_method: str = "cupy"            # "cupy", "numpy", or "python"
    
    # Memory Management
    enable_memory_pooling: bool = True        # Use GPU memory pools
    clear_gpu_cache_frequency: int = 100     # Clear every N nets
    
    # Algorithm Optimizations  
    enable_early_termination: bool = True     # Stop when target reached
    diagonal_cost_factor: float = 1.414      # √2 for diagonal moves
    
    # Monitoring
    track_memory_usage: bool = True           # Monitor GPU/CPU memory
    log_performance_metrics: bool = True      # Log timing data
    
@dataclass
class AlgorithmConfig:
    """Configuration for routing algorithms and strategies"""
    
    # Lee's Algorithm Settings
    wavefront_expansion_method: str = "gpu"   # "gpu", "cpu", or "auto"
    neighbor_connectivity: int = 8            # 4-connected or 8-connected
    distance_metric: str = "euclidean"        # "manhattan" or "euclidean"
    
    # Multi-layer Strategy
    layer_selection_method: str = "obstacle_density"  # "obstacle_density", "random", "round_robin"
    prefer_single_layer: bool = True          # Try single-layer first
    via_cost_penalty: float = 1.5            # Cost multiplier for using vias
    
    # Multi-pad Net Handling
    topology_method: str = "mst"              # "mst", "star", or "chain"
    connection_order: str = "shortest_first"  # "shortest_first", "random"
    
    # Fallback Strategies
    enable_rip_up_retry: bool = False         # Rip up conflicting routes
    max_retry_attempts: int = 2               # Maximum retry attempts
    
def create_default_config() -> Dict[str, object]:
    """Create default configuration set for OrthoRoute"""
    return {
        'routing': RoutingConfig(),
        'performance': PerformanceConfig(), 
        'algorithm': AlgorithmConfig()
    }

def create_high_performance_config() -> Dict[str, object]:
    """Create configuration optimized for maximum performance"""
    routing = RoutingConfig(
        timeout_per_net=3.0,  # Faster timeouts
        enable_proactive_vias=True,
        enable_debug_timing=False  # Reduce logging overhead
    )
    
    performance = PerformanceConfig(
        enable_incremental_grids=True,
        enable_pad_net_caching=True,
        clear_gpu_cache_frequency=50  # More frequent cleanup
    )
    
    algorithm = AlgorithmConfig(
        wavefront_expansion_method="gpu",
        neighbor_connectivity=8,
        prefer_single_layer=True
    )
    
    return {
        'routing': routing,
        'performance': performance,
        'algorithm': algorithm
    }

def create_via_focused_config() -> Dict[str, object]:
    """Create configuration that prioritizes multi-layer routing with vias"""
    routing = RoutingConfig(
        timeout_strategy_1=0.2,  # Reduce single-layer time
        timeout_strategy_2=0.6,  # Increase via routing time  
        timeout_strategy_3=0.2,
        enable_proactive_vias=True,
        max_via_attempts=5,      # Try more via locations
        via_placement_positions=[0.2, 0.3, 0.5, 0.7, 0.8]  # More positions
    )
    
    performance = PerformanceConfig(
        enable_incremental_grids=True,
        enable_pad_net_caching=True
    )
    
    algorithm = AlgorithmConfig(
        prefer_single_layer=False,  # Prefer multi-layer 
        via_cost_penalty=1.0,       # No penalty for vias
        layer_selection_method="obstacle_density"
    )
    
    return {
        'routing': routing,
        'performance': performance,
        'algorithm': algorithm
    }

# Configuration presets for different use cases
CONFIGURATION_PRESETS = {
    'default': create_default_config,
    'high_performance': create_high_performance_config,
    'via_focused': create_via_focused_config
}

def load_config(preset: str = 'default') -> Dict[str, object]:
    """Load a configuration preset"""
    if preset not in CONFIGURATION_PRESETS:
        available = list(CONFIGURATION_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    config = CONFIGURATION_PRESETS[preset]()
    logger.info(f"📋 Loaded '{preset}' configuration preset")
    return config

if __name__ == "__main__":
    # Demo the configuration system
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 OrthoRoute Configuration System Demo")
    print("=" * 50)
    
    for preset_name in CONFIGURATION_PRESETS:
        print(f"\n📋 {preset_name.upper()} Configuration:")
        config = load_config(preset_name)
        
        print(f"   Grid resolution: {config['routing'].grid_resolution}mm")
        print(f"   Strategy timeouts: {config['routing'].timeout_strategy_1:.0%}/{config['routing'].timeout_strategy_2:.0%}/{config['routing'].timeout_strategy_3:.0%}")
        print(f"   Via positions: {len(config['routing'].via_placement_positions)}")
        print(f"   GPU acceleration: {config['performance'].enable_incremental_grids}")
