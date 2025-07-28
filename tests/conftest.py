"""Test configuration and fixtures for OrthoRoute."""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cupy as cp

@pytest.fixture
def mock_gpu_device():
    """Mock GPU device with proper memory info."""
    device = MagicMock()
    # Set memory info to return tuple of (used, total) in bytes
    device.mem_info.return_value = (2 * 1024**3, 16 * 1024**3)  # 2GB used, 16GB total
    device.id = 0
    return device

@pytest.fixture
def mock_cupy_arrays():
    """Mock CuPy arrays for grid operations."""
    def mock_array(shape, dtype=np.float32):
        return MagicMock(
            shape=shape,
            dtype=dtype,
            nbytes=np.prod(shape) * np.dtype(dtype).itemsize
        )
    return mock_array

@pytest.fixture
def mock_gpu_grid(mock_cupy_arrays):
    """Create a mock GPU grid with proper array attributes."""
    with patch('cupy.ones', mock_cupy_arrays), \
         patch('cupy.zeros', mock_cupy_arrays), \
         patch('cupy.full', mock_cupy_arrays):
        from orthoroute.grid_manager import GPUGrid
        grid = GPUGrid(20, 20, 2, 0.1)
        return grid

@pytest.fixture
def sample_board_data():
    """Create sample board data for testing."""
    return {
        'width_nm': 100000000,  # 100mm in nanometers
        'height_nm': 100000000,
        'layer_count': 2,
        'grid_pitch_nm': 100000,  # 0.1mm in nanometers
        'nets': [
            {
                'name': 'net1',
                'pins': [
                    {'x': 1000000, 'y': 1000000, 'layer': 0},
                    {'x': 9000000, 'y': 9000000, 'layer': 0}
                ],
                'width_nm': 200000,
                'priority': 1
            }
        ],
        'obstacles': [
            {
                'x1': 4000000, 'y1': 4000000,
                'x2': 6000000, 'y2': 6000000,
                'layer': 0
            }
        ]
    }

@pytest.fixture
def mock_engine():
    """Create a mock routing engine."""
    with patch('cupy.cuda.Device') as mock_device:
        mock_device.return_value.mem_info = lambda: (2 * 1024**3, 16 * 1024**3)
        mock_device.return_value.id = 0
        from orthoroute.gpu_engine import OrthoRouteEngine
        engine = OrthoRouteEngine()
        return engine
