"""
orthoroute - GPU-accelerated PCB autorouter
"""
import sys
import warnings
from typing import Dict, List, Optional, Tuple
import cupy as cp
import numpy as np

# Version information
__version__ = "0.1.0"
__author__ = "Brian Benchoff"
__email__ = "bbenchoff@gmail.com"
__license__ = "MIT"
__description__ = "GPU-accelerated PCB autorouter using CuPy"
__url__ = "https://github.com/bbenchoff/OrthoRoute"
    
# Test basic CuPy functionality
try:
    test_array = cp.array([1, 2, 3])
    _CUPY_AVAILABLE = True
    
    # Import components - explicitly use relative imports
    from .gpu_engine import OrthoRouteEngine, GPUGrid, Point3D, Net
    from .wave_router import WaveRouter
    # Define what should be available when importing from the package
    __all__ = ['OrthoRouteEngine', 'GPUGrid', 'Point3D', 'Net', 'WaveRouter', 'gpu_engine']
    _CUPY_AVAILABLE = True
except ImportError as e:
    _CUPY_ERROR = f"CuPy not available: {e}"
    _CUPY_AVAILABLE = False
except Exception as e:
    _CUPY_ERROR = f"CuPy test failed: {e}"
    _CUPY_AVAILABLE = False

# Import main classes (with graceful degradation)
if _CUPY_AVAILABLE:
    try:
        from .gpu_engine import OrthoRouteEngine
        from .grid_manager import GPUGrid, Point3D, Net, TileManager
        from .routing_algorithms import WavefrontRouter, ConflictResolver
        
        # Optional imports (may not be available in minimal installations)
        try:
            from .steiner_tree import SteinerTreeBuilder
        except ImportError:
            SteinerTreeBuilder = None
        
        try:
            from .visualization import RoutingVisualizer
        except ImportError:
            RoutingVisualizer = None
            
    except ImportError as e:
        # Core modules not available - provide dummy classes
        warnings.warn(f"OrthoRoute core modules not available: {e}")
        OrthoRouteEngine = None
        GPUGrid = None
        Point3D = None
        Net = None
        
else:
    # CuPy not available - provide informative error classes
    class _CuPyNotAvailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"CuPy is required for OrthoRoute GPU functionality.\n"
                f"Error: {_CUPY_ERROR}\n\n"
                f"Installation instructions:\n"
                f"1. Ensure NVIDIA GPU with CUDA support\n"
                f"2. Install CUDA Toolkit 11.8+ or 12.x\n"
                f"3. Install CuPy:\n"
                f"   pip install cupy-cuda12x  # For CUDA 12.x\n"
                f"   pip install cupy-cuda11x  # For CUDA 11.x\n\n"
                f"For more details: https://docs.cupy.dev/en/stable/install.html"
            )
    
    OrthoRouteEngine = _CuPyNotAvailable
    GPUGrid = _CuPyNotAvailable
    Point3D = None
    Net = None
    WavefrontRouter = _CuPyNotAvailable
    ConflictResolver = _CuPyNotAvailable
    SteinerTreeBuilder = None
    RoutingVisualizer = None

# Utility functions
def check_gpu_availability() -> dict:
    """
    Check GPU and CUDA availability for OrthoRoute.
    
    Returns:
        dict: Status information including GPU details, CUDA version, memory info
    """
    status = {
        'cupy_available': _CUPY_AVAILABLE,
        'gpu_detected': False,
        'cuda_version': None,
        'gpu_count': 0,
        'gpu_name': None,
        'memory_total': None,
        'memory_free': None,
        'error': _CUPY_ERROR if not _CUPY_AVAILABLE else None
    }
    
    if _CUPY_AVAILABLE:
        try:
            import cupy as cp
            
            # Get GPU information
            device = cp.cuda.Device()
            attrs = device.attributes
            mem_info = device.mem_info
            
            status.update({
                'gpu_detected': True,
                'gpu_count': cp.cuda.runtime.getDeviceCount(),
                'gpu_name': device.name,
                'compute_capability': f"{attrs['major']}.{attrs['minor']}",
                'multiprocessors': attrs['multiProcessorCount'],
                'cuda_cores_estimate': attrs['multiProcessorCount'] * 128,  # Rough estimate
                'memory_total': mem_info[1],
                'memory_free': mem_info[0],
                'memory_used': mem_info[1] - mem_info[0],
                'cuda_version': cp.cuda.runtime.runtimeGetVersion()
            })
            
        except Exception as e:
            status['error'] = f"GPU detection failed: {e}"
    
    return status

def print_system_info():
    """Print detailed system information for debugging."""
    print("OrthoRoute System Information")
    print("=" * 40)
    print(f"Version: {__version__}")
    print(f"Python: {sys.version}")
    
    status = check_gpu_availability()
    
    if status['cupy_available']:
        print("✅ CuPy: Available")
        
        if status['gpu_detected']:
            print(f"✅ GPU: {status['gpu_name']}")
            print(f"   Compute Capability: {status['compute_capability']}")
            print(f"   Multiprocessors: {status['multiprocessors']}")
            print(f"   Estimated CUDA Cores: {status['cuda_cores_estimate']}")
            print(f"   Memory: {status['memory_total'] / (1024**3):.1f} GB total, "
                  f"{status['memory_free'] / (1024**3):.1f} GB free")
            print(f"   CUDA Version: {status['cuda_version']}")
        else:
            print("❌ GPU: Not detected")
    else:
        print(f"❌ CuPy: Not available - {status['error']}")
    
    print("=" * 40)

def get_performance_recommendations(net_count: int) -> dict:
    """
    Get performance recommendations based on board complexity.
    
    Args:
        net_count: Number of nets to route
        
    Returns:
        dict: Recommended settings and hardware requirements
    """
    if net_count < 500:
        complexity = "Simple"
        gpu_rec = "RTX 3060+ (8GB)"
        grid_pitch = 0.1
        batch_size = 256
        expected_time = "1-5 seconds"
    elif net_count < 2000:
        complexity = "Medium"
        gpu_rec = "RTX 4070+ (12GB)"
        grid_pitch = 0.1
        batch_size = 512
        expected_time = "10-30 seconds"
    elif net_count < 8000:
        complexity = "Complex"
        gpu_rec = "RTX 4080+ (16GB)"
        grid_pitch = 0.15
        batch_size = 1024
        expected_time = "1-5 minutes"
    else:
        complexity = "Extreme"
        gpu_rec = "RTX 5080+ (16GB+)"
        grid_pitch = 0.2
        batch_size = 1024
        expected_time = "5-20 minutes"
    
    return {
        'complexity': complexity,
        'recommended_gpu': gpu_rec,
        'grid_pitch_mm': grid_pitch,
        'batch_size': batch_size,
        'expected_time': expected_time,
        'memory_estimate_gb': (net_count * 0.001) + 1  # Rough estimate
    }

# Configuration defaults
DEFAULT_CONFIG = {
    'grid_pitch_mm': 0.1,
    'max_layers': 8,
    'max_iterations': 20,
    'batch_size': 256,
    'tile_size': 64,
    'congestion_factor': 1.5,
    'via_cost': 10,
    'trace_cost': 1,
    'direction_change_cost': 2
}

# Export main API
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Main classes
    'OrthoRouteEngine',
    'GPUGrid',
    'Point3D', 
    'Net',
    'TileManager',
    'WavefrontRouter',
    'ConflictResolver',
    'SteinerTreeBuilder',
    'RoutingVisualizer',
    
    # Utility functions
    'check_gpu_availability',
    'print_system_info',
    'get_performance_recommendations',
    
    # Configuration
    'DEFAULT_CONFIG'
]

# Package-level docstring for help()
__doc__ = """
OrthoRoute GPU-Accelerated PCB Autorouter
=========================================

Installation:
    pip install cupy-cuda12x  # For CUDA 12.x
    pip install orthoroute

Quick Start:
    from orthoroute import OrthoRouteEngine, check_gpu_availability
    
    # Check system compatibility
    status = check_gpu_availability()
    if status['cupy_available']:
        print(f"GPU ready: {status['gpu_name']}")
        
        # Route a board
        engine = OrthoRouteEngine()
        results = engine.route_board(board_data)
    else:
        print(f"Setup required: {status['error']}")

Main Classes:
    OrthoRouteEngine - Main routing engine
    GPUGrid - GPU memory management
    Point3D, Net - Data structures
    WavefrontRouter - Parallel routing algorithm
    ConflictResolver - Congestion handling

Utilities:
    check_gpu_availability() - System compatibility check
    print_system_info() - Detailed system information
    get_performance_recommendations(net_count) - Optimize settings

For complete documentation and examples:
https://github.com/bbenchoff/OrthoRoute
"""

# Initialize package
def _initialize_package():
    """Initialize package with environment checks."""
    if _CUPY_AVAILABLE:
        try:
            # Verify GPU access
            import cupy as cp
            test = cp.array([1])
            del test
        except Exception as e:
            warnings.warn(f"GPU initialization failed: {e}")

# Run initialization
_initialize_package()