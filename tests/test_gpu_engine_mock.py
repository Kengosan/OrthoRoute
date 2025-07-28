"""Tests for GPU engine using mock objects."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from orthoroute.gpu_engine import OrthoRouteEngine
from orthoroute.grid_manager import GPUGrid, Point3D, Net
from tests.test_utils import create_test_board, mock_gpu_device, mock_gpu_array

def test_gpu_engine_init(mock_gpu_device):
    """Test GPU engine initialization with mock device."""
    with patch('cupy.cuda.Device', return_value=mock_gpu_device):
        engine = OrthoRouteEngine()
        assert isinstance(engine, OrthoRouteEngine)

def test_memory_management(mock_gpu_device, mock_gpu_array):
    """Test memory tracking and management."""
    with patch('cupy.cuda.Device', return_value=mock_gpu_device):
        with patch('cupy.ones') as mock_ones:
            def mock_array_factory(*args, **kwargs):
                # First arg is shape tuple
                shape = args[0] if args else kwargs.get('shape', None)
                return mock_gpu_array(shape)
            mock_ones.side_effect = mock_array_factory
            
            engine = OrthoRouteEngine()
            grid_mgr = GPUGrid(width=500, height=500, layers=2)
            
            # Verify grid dimensions
            assert grid_mgr.width == 500
            assert grid_mgr.height == 500
            assert grid_mgr.layers == 2

def test_grid_allocation(mock_gpu_device, mock_gpu_array):
    """Test grid allocation with mock arrays."""
    with patch('cupy.cuda.Device', return_value=mock_gpu_device):
        with patch('cupy.ones') as mock_ones:
            def mock_array_factory(*args, **kwargs):
                # First arg is shape tuple
                shape = args[0] if args else kwargs.get('shape', None)
                return mock_gpu_array(shape)
            mock_ones.side_effect = mock_array_factory
            
            grid_mgr = GPUGrid(width=10, height=10, layers=2)
            
            # Verify grid dimensions and arrays
            assert grid_mgr.width == 10
            assert grid_mgr.height == 10
            assert grid_mgr.layers == 2
            assert grid_mgr.availability.shape == (2, 10, 10)
