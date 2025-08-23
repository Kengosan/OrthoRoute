"""Abstract GPU provider interface."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np


class GPUProvider(ABC):
    """Abstract interface for GPU operations."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if GPU is available."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize GPU resources."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup GPU resources."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        pass
    
    @abstractmethod
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory usage information."""
        pass
    
    @abstractmethod
    def create_array(self, shape: Tuple[int, ...], dtype=None, fill_value=None) -> Any:
        """Create array on GPU or CPU."""
        pass
    
    @abstractmethod
    def copy_array(self, array: Any) -> Any:
        """Create copy of array."""
        pass
    
    @abstractmethod
    def to_cpu(self, array: Any) -> np.ndarray:
        """Convert GPU array to CPU."""
        pass
    
    @abstractmethod
    def to_gpu(self, array: np.ndarray) -> Any:
        """Convert CPU array to GPU."""
        pass
    
    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        pass