#!/usr/bin/env python3
"""
GPU Infrastructure Management

Handles GPU initialization, memory management, and provides GPU/CPU fallback functionality.
Shared across all routing algorithms that can benefit from GPU acceleration.
"""
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import cupy.cuda.profiler
    HAS_CUPY = True
    logger.info("🔥 CuPy available - GPU acceleration enabled")
except ImportError:
    HAS_CUPY = False
    cp = None
    logger.info("💻 CuPy not available - using CPU fallback")


class GPUManager:
    """Manages GPU resources and provides GPU/CPU abstraction"""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPU manager
        
        Args:
            use_gpu: Whether to attempt GPU usage (if available)
        """
        self.use_gpu = use_gpu and HAS_CUPY
        self.device_info = {}
        self.mempool = None
        
        if self.use_gpu:
            try:
                self._setup_gpu_environment()
                logger.info("🔥 GPU environment initialized successfully")
            except Exception as e:
                logger.error(f"GPU setup failed: {e}")
                logger.warning("Falling back to CPU mode")
                self.use_gpu = False
        else:
            logger.info("💻 Using CPU implementation")
    
    def _setup_gpu_environment(self):
        """Initialize GPU environment and memory pools"""
        # Set up CuPy memory pool for better performance
        self.mempool = cp.get_default_memory_pool()
        
        # Get available memory and set limit (leave some headroom)
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        memory_limit = int(free_mem * 0.8)  # Use 80% of free memory
        self.mempool.set_limit(size=memory_limit)
        
        # Enable CuPy profiling if in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            cp.cuda.profiler.start()
        
        # Get GPU device info
        device = cp.cuda.Device()
        
        try:
            device_name = device.attributes.get('Name', 'Unknown GPU')
        except:
            device_name = f"GPU Device {device.id}"
        
        self.device_info = {
            'name': device_name,
            'compute_capability': device.compute_capability,
            'total_memory': total_mem,
            'free_memory': free_mem,
            'memory_limit': memory_limit
        }
        
        logger.info(f"🔥 GPU Device: {device_name}")
        logger.info(f"🔥 Compute Capability: {device.compute_capability}")
        logger.info(f"🔥 Memory: {total_mem / 1024**3:.1f}GB total, {free_mem / 1024**3:.1f}GB free")
        logger.info(f"🔥 Memory limit set to: {memory_limit / 1024**3:.1f}GB")
        
        # Test basic GPU operations
        test_array = cp.ones((1000, 1000), dtype=cp.float32)
        result = cp.sum(test_array)
        logger.debug(f"🔥 GPU test passed: sum = {result}")
    
    def create_array(self, shape: Tuple[int, ...], dtype=None, fill_value=None) -> Any:
        """
        Create array on GPU or CPU depending on configuration
        
        Args:
            shape: Array dimensions
            dtype: Data type (defaults to appropriate type)
            fill_value: Initial value (None for uninitialized, 0 for zeros, etc.)
        
        Returns:
            CuPy array if GPU enabled, NumPy array otherwise
        """
        import numpy as np
        
        if dtype is None:
            dtype = cp.bool_ if self.use_gpu else np.bool_
        
        if self.use_gpu:
            if fill_value is None:
                return cp.empty(shape, dtype=dtype)
            elif fill_value == 0:
                return cp.zeros(shape, dtype=dtype)
            elif fill_value == 1:
                return cp.ones(shape, dtype=dtype)
            else:
                return cp.full(shape, fill_value, dtype=dtype)
        else:
            if fill_value is None:
                return np.empty(shape, dtype=dtype)
            elif fill_value == 0:
                return np.zeros(shape, dtype=dtype)
            elif fill_value == 1:
                return np.ones(shape, dtype=dtype)
            else:
                return np.full(shape, fill_value, dtype=dtype)
    
    def copy_array(self, array) -> Any:
        """Create a copy of an array on the same device"""
        if self.use_gpu:
            # Ensure array is on GPU first
            gpu_array = self.to_gpu(array)
            return cp.copy(gpu_array)
        else:
            import numpy as np
            return np.copy(array)
    
    def to_cpu(self, array) -> Any:
        """Convert GPU array to CPU (no-op if already CPU)"""
        if self.use_gpu and hasattr(array, 'get'):
            return array.get()
        return array
    
    def to_gpu(self, array) -> Any:
        """Convert CPU array to GPU (no-op if already GPU or GPU disabled)"""
        if not self.use_gpu:
            return array
            
        # More reliable CuPy array detection
        if hasattr(cp, 'ndarray') and isinstance(array, cp.ndarray):
            # Already a CuPy array
            return array
        else:
            # Convert NumPy array or other type to CuPy array
            return cp.asarray(array)
    
    def get_memory_info(self) -> dict:
        """Get current memory usage information"""
        if self.use_gpu:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            used_mem = total_mem - free_mem
            
            return {
                'total_memory': total_mem,
                'free_memory': free_mem,
                'used_memory': used_mem,
                'memory_pool_used': self.mempool.used_bytes() if self.mempool else 0,
                'memory_pool_total': self.mempool.total_bytes() if self.mempool else 0
            }
        else:
            return {
                'total_memory': 0,
                'free_memory': 0,
                'used_memory': 0,
                'memory_pool_used': 0,
                'memory_pool_total': 0
            }
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if self.use_gpu and self.mempool:
            self.mempool.free_all_blocks()
            logger.debug("🔥 GPU memory pool cleaned up")
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU is enabled and available"""
        return self.use_gpu
    
    def get_device_info(self) -> dict:
        """Get GPU device information"""
        return self.device_info.copy()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
