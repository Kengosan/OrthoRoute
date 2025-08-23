"""CUDA GPU provider implementation."""
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ...application.interfaces.gpu_provider import GPUProvider

logger = logging.getLogger(__name__)


class CUDAProvider(GPUProvider):
    """CUDA/CuPy implementation of GPU provider."""
    
    def __init__(self):
        """Initialize CUDA provider."""
        self._cupy = None
        self._memory_pool = None
        self._device_info = {}
        self._initialized = False
    
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import cupy as cp
            # Test basic CUDA functionality
            test_array = cp.array([1, 2, 3])
            _ = cp.sum(test_array)
            # Log successful GPU detection
            logger.info("CUDA GPU detected and working")
            return True
        except ImportError:
            logger.warning("CuPy not installed - GPU acceleration unavailable")
            return False
        except Exception as e:
            logger.warning(f"CUDA error: {str(e)} - GPU acceleration unavailable")
            return False
    
    def initialize(self) -> bool:
        """Initialize CUDA resources."""
        if self._initialized:
            return True
        
        try:
            import cupy as cp
            self._cupy = cp
            
            # Set up memory pool
            self._memory_pool = cp.get_default_memory_pool()
            
            # Get device information
            device = cp.cuda.Device()
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            
            # Set memory limit (80% of free memory)
            memory_limit = int(free_mem * 0.8)
            self._memory_pool.set_limit(size=memory_limit)
            
            # Get device name
            try:
                device_name = device.attributes.get('Name', f'CUDA Device {device.id}')
            except:
                device_name = f'CUDA Device {device.id}'
            
            self._device_info = {
                'name': device_name,
                'compute_capability': device.compute_capability,
                'total_memory': total_mem,
                'free_memory': free_mem,
                'memory_limit': memory_limit,
                'device_id': device.id
            }
            
            # Test basic operations
            test_array = cp.ones((100, 100), dtype=cp.float32)
            result = cp.sum(test_array)
            
            self._initialized = True
            logger.info(f"CUDA initialized: {device_name}")
            logger.info(f"Memory: {total_mem / 1024**3:.1f}GB total, {free_mem / 1024**3:.1f}GB free")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CUDA: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup CUDA resources."""
        if self._memory_pool:
            try:
                self._memory_pool.free_all_blocks()
                logger.debug("CUDA memory pool cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up CUDA memory pool: {e}")
        
        self._initialized = False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get CUDA device information."""
        return self._device_info.copy()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get CUDA memory usage information."""
        if not self._initialized or not self._cupy:
            return {
                'total_memory': 0,
                'free_memory': 0,
                'used_memory': 0,
                'memory_pool_used': 0,
                'memory_pool_total': 0
            }
        
        try:
            free_mem, total_mem = self._cupy.cuda.runtime.memGetInfo()
            used_mem = total_mem - free_mem
            
            pool_used = self._memory_pool.used_bytes() if self._memory_pool else 0
            pool_total = self._memory_pool.total_bytes() if self._memory_pool else 0
            
            return {
                'total_memory': total_mem,
                'free_memory': free_mem,
                'used_memory': used_mem,
                'memory_pool_used': pool_used,
                'memory_pool_total': pool_total
            }
            
        except Exception as e:
            logger.error(f"Error getting CUDA memory info: {e}")
            return {
                'total_memory': 0,
                'free_memory': 0,
                'used_memory': 0,
                'memory_pool_used': 0,
                'memory_pool_total': 0
            }
    
    def create_array(self, shape: Tuple[int, ...], dtype=None, fill_value=None) -> Any:
        """Create array on GPU."""
        if not self._initialized or not self._cupy:
            raise RuntimeError("CUDA not initialized")
        
        if dtype is None:
            dtype = self._cupy.float32
        
        try:
            if fill_value is None:
                return self._cupy.empty(shape, dtype=dtype)
            elif fill_value == 0:
                return self._cupy.zeros(shape, dtype=dtype)
            elif fill_value == 1:
                return self._cupy.ones(shape, dtype=dtype)
            else:
                return self._cupy.full(shape, fill_value, dtype=dtype)
                
        except Exception as e:
            logger.error(f"Error creating CUDA array: {e}")
            raise
    
    def copy_array(self, array: Any) -> Any:
        """Create copy of array on GPU."""
        if not self._initialized or not self._cupy:
            raise RuntimeError("CUDA not initialized")
        
        try:
            return self._cupy.copy(array)
        except Exception as e:
            logger.error(f"Error copying CUDA array: {e}")
            raise
    
    def to_cpu(self, array: Any) -> np.ndarray:
        """Convert GPU array to CPU."""
        if not self._initialized or not self._cupy:
            return array
        
        try:
            if hasattr(array, 'get'):
                return array.get()
            else:
                return array
        except Exception as e:
            logger.error(f"Error converting array to CPU: {e}")
            return array
    
    def to_gpu(self, array: np.ndarray) -> Any:
        """Convert CPU array to GPU."""
        if not self._initialized or not self._cupy:
            raise RuntimeError("CUDA not initialized")
        
        try:
            return self._cupy.asarray(array)
        except Exception as e:
            logger.error(f"Error converting array to GPU: {e}")
            raise
    
    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        if not self._initialized or not self._cupy:
            return
        
        try:
            self._cupy.cuda.Stream.null.synchronize()
        except Exception as e:
            logger.error(f"Error synchronizing CUDA: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()