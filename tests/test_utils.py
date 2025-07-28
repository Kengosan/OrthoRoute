import pytest
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def mock_gpu_device():
    """Create a mock GPU device with proper memory info."""
    device = MagicMock()
    device.mem_info = MagicMock(return_value=[2 * 1024**3, 16 * 1024**3])
    device.mem_info.__getitem__ = lambda self, idx: device.mem_info.return_value[idx]
    device.id = 0
    device.properties = {
        'totalMemory': 16 * 1024**3,
        'maxThreadsPerBlock': 1024,
        'name': 'Test GPU'
    }
    return device

@pytest.fixture
def mock_gpu_array():
    """Create a mock GPU array with proper attributes."""
    def _create_mock_array(shape, dtype=None):
        arr = MagicMock()
        if isinstance(shape, (tuple, list)):
            arr.shape = shape
        else:
            # If called with a single value, make it a 3D array
            arr.shape = (2, shape[0], shape[1]) if len(shape) == 2 else shape
        arr.nbytes = np.prod(arr.shape) * 4  # Assuming float32
        return arr
    return _create_mock_array

def create_test_board(width_mm, height_mm, net_count, layer_count=2):
    """Create a test board with the specified dimensions and number of nets.
    
    Args:
        width_mm: Board width in millimeters
        height_mm: Board height in millimeters
        net_count: Number of nets to create
        layer_count: Number of routing layers
        
    Returns:
        Dictionary containing board data
    """
    # Convert mm to nm for internal representation
    width_nm = int(width_mm * 1_000_000)
    height_nm = int(height_mm * 1_000_000)
    grid_pitch_nm = 100_000  # 0.1mm grid pitch
    
    # Create basic board data
    board_data = {
        'width_nm': width_nm,
        'height_nm': height_nm,
        'layer_count': layer_count,
        'grid_pitch_nm': grid_pitch_nm,
        'nets': [],
        'obstacles': []
    }
    
    # Generate test nets
    import random
    random.seed(42)  # For reproducible tests
    
    for i in range(net_count):
        # Create random start and end points
        x1 = random.randint(grid_pitch_nm, width_nm - grid_pitch_nm)
        y1 = random.randint(grid_pitch_nm, height_nm - grid_pitch_nm)
        x2 = random.randint(grid_pitch_nm, width_nm - grid_pitch_nm)
        y2 = random.randint(grid_pitch_nm, height_nm - grid_pitch_nm)
        
        net = {
            'name': f'net_{i}',
            'pins': [
                {'x': x1, 'y': y1, 'layer': 0},
                {'x': x2, 'y': y2, 'layer': 0}
            ],
            'width_nm': 200_000,  # 0.2mm trace width
            'priority': random.randint(1, 3)
        }
        board_data['nets'].append(net)
    
    # Add some test obstacles
    obstacle = {
        'x1': width_nm // 4,
        'y1': height_nm // 4,
        'x2': width_nm // 2,
        'y2': height_nm // 2,
        'layer': 0
    }
    board_data['obstacles'].append(obstacle)
    
    return board_data
