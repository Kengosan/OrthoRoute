"""
Core GPU Engine Tests for OrthoRoute
Unit tests for the main GPU routing engine functionality

These tests verify the core GPU routing engine components including
initialization, board loading, routing execution, and result generation.
"""

import pytest
import numpy as np
import json
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

# Test if CuPy is available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    # Create mock CuPy for testing without GPU
    cp = Mock()

# Import modules under test
try:
    from orthoroute.gpu_engine import OrthoRouteEngine
    from orthoroute.grid_manager import GPUGrid, Point3D, Net
    from orthoroute.routing_algorithms import WavefrontRouter, ConflictResolver
    ENGINE_AVAILABLE = True
except ImportError as e:
    ENGINE_AVAILABLE = False
    print(f"Warning: Engine modules not available for testing: {e}")


class TestGPUEngineInitialization:
    """Test GPU engine initialization and setup"""
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_engine_initialization_success(self):
        """Test successful engine initialization with valid GPU"""
        engine = OrthoRouteEngine(gpu_id=0)
        
        assert engine is not None
        assert hasattr(engine, 'grid')
        assert hasattr(engine, 'config')
        assert engine.config['gpu_id'] == 0
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_engine_initialization_invalid_gpu(self):
        """Test engine initialization with invalid GPU ID"""
        with pytest.raises((RuntimeError, Exception)):
            OrthoRouteEngine(gpu_id=99)  # Non-existent GPU
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_engine_default_config(self):
        """Test default configuration values"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            expected_defaults = {
                'grid_pitch_mm': 0.1,
                'max_layers': 8,
                'max_iterations': 20,
                'batch_size': 256,
                'tile_size': 64
            }
            
            for key, expected_value in expected_defaults.items():
                assert engine.config[key] == expected_value


class TestBoardDataLoading:
    """Test board data loading and validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sample_board_data = {
            'bounds': {
                'min_x': 0,
                'min_y': 0,
                'max_x': 50000000,  # 50mm in nanometers
                'max_y': 50000000
            },
            'grid': {
                'pitch_mm': 0.1,
                'layers': 4
            },
            'nets': [
                {
                    'id': 1,
                    'name': 'TEST_NET_1',
                    'pins': [
                        {'x': 5000000, 'y': 5000000, 'layer': 0},
                        {'x': 45000000, 'y': 45000000, 'layer': 0}
                    ],
                    'priority': 5,
                    'width_nm': 200000,
                    'via_size_nm': 200000
                }
            ],
            'obstacles': [
                {
                    'type': 'keepout',
                    'x1': 20000000,
                    'y1': 20000000,
                    'x2': 30000000,
                    'y2': 30000000,
                    'layer': -1
                }
            ]
        }
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_load_valid_board_data(self):
        """Test loading valid board data"""
        with patch('cupy.cuda.Device'), patch('orthoroute.gpu_engine.GPUGrid') as mock_grid:
            engine = OrthoRouteEngine()
            result = engine.load_board_data(self.sample_board_data)
            
            assert result is True
            mock_grid.assert_called_once()
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_load_invalid_board_data(self):
        """Test loading invalid board data"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            # Missing bounds
            invalid_data = {'nets': []}
            result = engine.load_board_data(invalid_data)
            assert result is False
            
            # Invalid bounds
            invalid_data = {'bounds': {'min_x': 100, 'max_x': 50}}  # max < min
            result = engine.load_board_data(invalid_data)
            assert result is False
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_board_bounds_calculation(self):
        """Test board bounds calculation and grid sizing"""
        with patch('cupy.cuda.Device'), patch('orthoroute.gpu_engine.GPUGrid') as mock_grid:
            engine = OrthoRouteEngine()
            engine.load_board_data(self.sample_board_data)
            
            # Verify grid was created with correct dimensions
            # 50mm / 0.1mm = 500 + 1 = 501 grid points
            mock_grid.assert_called_with(501, 501, 4, 0.1)


class TestNetParsing:
    """Test net parsing and validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_nets_data = [
            {
                'id': 1,
                'name': 'NET_1',
                'pins': [
                    {'x': 1000000, 'y': 1000000, 'layer': 0},
                    {'x': 2000000, 'y': 2000000, 'layer': 1}
                ],
                'priority': 1,
                'width_nm': 150000,
                'via_size_nm': 200000
            },
            {
                'id': 2,
                'name': 'NET_2',
                'pins': [
                    {'x': 3000000, 'y': 3000000, 'layer': 0},
                    {'x': 4000000, 'y': 4000000, 'layer': 0},
                    {'x': 5000000, 'y': 5000000, 'layer': 1}
                ],
                'priority': 5
            }
        ]
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_parse_valid_nets(self):
        """Test parsing valid net data"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            # Mock grid for coordinate conversion
            engine.grid = Mock()
            engine.grid.world_to_grid = Mock(side_effect=lambda x, y: (x // 100000, y // 100000))
            
            nets = engine._parse_nets(self.test_nets_data)
            
            assert len(nets) == 2
            assert all(isinstance(net, Net) for net in nets)
            assert nets[0].priority == 1  # Should be sorted by priority
            assert nets[1].priority == 5
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_parse_nets_with_insufficient_pins(self):
        """Test parsing nets with insufficient pins"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            engine.grid = Mock()
            engine.grid.world_to_grid = Mock(side_effect=lambda x, y: (x // 100000, y // 100000))
            
            # Net with only one pin
            invalid_nets = [
                {
                    'id': 1,
                    'name': 'INVALID_NET',
                    'pins': [{'x': 1000000, 'y': 1000000, 'layer': 0}]
                }
            ]
            
            nets = engine._parse_nets(invalid_nets)
            assert len(nets) == 0  # Should be filtered out
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_net_priority_sorting(self):
        """Test that nets are sorted by priority"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            engine.grid = Mock()
            engine.grid.world_to_grid = Mock(side_effect=lambda x, y: (x // 100000, y // 100000))
            
            # Create nets with different priorities
            mixed_priority_nets = [
                {
                    'id': 1, 'name': 'LOW_PRIORITY',
                    'pins': [{'x': 1000000, 'y': 1000000, 'layer': 0}, {'x': 2000000, 'y': 2000000, 'layer': 0}],
                    'priority': 10
                },
                {
                    'id': 2, 'name': 'HIGH_PRIORITY',
                    'pins': [{'x': 3000000, 'y': 3000000, 'layer': 0}, {'x': 4000000, 'y': 4000000, 'layer': 0}],
                    'priority': 1
                }
            ]
            
            nets = engine._parse_nets(mixed_priority_nets)
            assert nets[0].priority == 1  # High priority first
            assert nets[1].priority == 10  # Low priority second


class TestRoutingExecution:
    """Test routing execution and workflow"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.simple_board = {
            'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 10000000, 'max_y': 10000000},
            'grid': {'pitch_mm': 0.2, 'layers': 2},
            'config': {'max_iterations': 5, 'batch_size': 64},
            'nets': [
                {
                    'id': 1, 'name': 'TEST_NET',
                    'pins': [
                        {'x': 1000000, 'y': 1000000, 'layer': 0},
                        {'x': 9000000, 'y': 9000000, 'layer': 0}
                    ]
                }
            ],
            'obstacles': []
        }
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_route_board_success(self):
        """Test successful board routing"""
        with patch('cupy.cuda.Device'), \
             patch('orthoroute.gpu_engine.ConflictResolver') as mock_resolver:
            
            # Mock successful routing result
            mock_net = Mock()
            mock_net.routed = True
            mock_net.total_length = 10.5
            mock_net.via_count = 2
            mock_resolver.return_value.resolve_conflicts.return_value = [mock_net]
            
            engine = OrthoRouteEngine()
            result = engine.route_board(self.simple_board)
            
            assert result['success'] is True
            assert 'stats' in result
            assert 'routed_nets' in result
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_route_board_no_nets(self):
        """Test routing board with no nets"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            empty_board = self.simple_board.copy()
            empty_board['nets'] = []
            
            result = engine.route_board(empty_board)
            assert result['success'] is False
            assert 'No nets to route' in result['error']
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_route_board_invalid_data(self):
        """Test routing with invalid board data"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            result = engine.route_board({'invalid': 'data'})
            assert result['success'] is False
            assert 'error' in result


class TestResultsGeneration:
    """Test routing results generation and formatting"""
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_generate_results_success(self):
        """Test successful results generation"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            # Mock grid for coordinate conversion
            engine.grid = Mock()
            engine.grid.grid_to_world = Mock(side_effect=lambda x, y: (x * 100000, y * 100000))
            
            # Create mock routed nets
            mock_nets = []
            for i in range(3):
                net = Mock()
                net.net_id = i + 1
                net.name = f'NET_{i+1}'
                net.routed = True
                net.route_path = [
                    Mock(x=i, y=i, layer=0),
                    Mock(x=i+1, y=i+1, layer=0)
                ]
                net.total_length = 5.0 + i
                net.via_count = i
                net.width_nm = 200000
                net.via_size_nm = 200000
                mock_nets.append(net)
            
            routing_time = 2.5
            result = engine._generate_results(mock_nets, routing_time)
            
            assert result['success'] is True
            assert result['stats']['total_nets'] == 3
            assert result['stats']['successful_nets'] == 3
            assert result['stats']['success_rate'] == 100.0
            assert result['stats']['routing_time_seconds'] == 2.5
            assert len(result['routed_nets']) == 3
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_generate_results_partial_failure(self):
        """Test results generation with some failed nets"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            engine.grid = Mock()
            engine.grid.grid_to_world = Mock(side_effect=lambda x, y: (x * 100000, y * 100000))
            
            # Mix of successful and failed nets
            mock_nets = []
            for i in range(5):
                net = Mock()
                net.net_id = i + 1
                net.name = f'NET_{i+1}'
                net.routed = i < 3  # First 3 succeed, last 2 fail
                if net.routed:
                    net.route_path = [Mock(x=i, y=i, layer=0)]
                    net.total_length = 5.0
                    net.via_count = 1
                else:
                    net.route_path = []
                    net.total_length = 0.0
                    net.via_count = 0
                net.width_nm = 200000
                net.via_size_nm = 200000
                mock_nets.append(net)
            
            result = engine._generate_results(mock_nets, 3.0)
            
            assert result['success'] is True
            assert result['stats']['total_nets'] == 5
            assert result['stats']['successful_nets'] == 3
            assert result['stats']['failed_nets'] == 2
            assert result['stats']['success_rate'] == 60.0


class TestMemoryManagement:
    """Test GPU memory management and efficiency"""
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_memory_cleanup(self):
        """Test that GPU memory is properly cleaned up"""
        initial_memory = cp.get_default_memory_pool().used_bytes()
        
        engine = OrthoRouteEngine()
        
        # Force some memory allocation
        large_board = {
            'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 100000000, 'max_y': 100000000},
            'grid': {'pitch_mm': 0.1, 'layers': 4},
            'nets': [],
            'obstacles': []
        }
        
        engine.load_board_data(large_board)
        
        # Memory should increase
        allocated_memory = cp.get_default_memory_pool().used_bytes()
        assert allocated_memory >= initial_memory
        
        # Clean up
        del engine
        cp.get_default_memory_pool().free_all_blocks()
        
        # Memory should be freed (approximately)
        final_memory = cp.get_default_memory_pool().used_bytes()
        assert final_memory <= allocated_memory
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_memory_estimation(self):
        """Test memory usage estimation"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            # Small board
            small_board = {
                'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 10000000, 'max_y': 10000000},
                'grid': {'pitch_mm': 0.2, 'layers': 2}
            }
            
            # Large board
            large_board = {
                'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 100000000, 'max_y': 100000000},
                'grid': {'pitch_mm': 0.05, 'layers': 8}
            }
            
            with patch('orthoroute.gpu_engine.GPUGrid') as mock_grid:
                mock_grid.return_value._calculate_memory_usage = Mock()
                
                engine.load_board_data(small_board)
                small_calls = mock_grid.call_count
                
                engine.load_board_data(large_board)
                large_calls = mock_grid.call_count
                
                # Both should create grids
                assert small_calls > 0
                assert large_calls > small_calls


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_cuda_error_handling(self):
        """Test handling of CUDA errors"""
        with patch('cupy.cuda.Device', side_effect=RuntimeError("CUDA error")):
            with pytest.raises(RuntimeError):
                OrthoRouteEngine()
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_malformed_input_handling(self):
        """Test handling of malformed input data"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            # Test various malformed inputs
            malformed_inputs = [
                None,
                "not a dict",
                {'bounds': 'invalid'},
                {'bounds': {'min_x': 'not_a_number'}},
                {'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 10, 'max_y': 10}, 'nets': 'invalid'}
            ]
            
            for malformed_input in malformed_inputs:
                result = engine.route_board(malformed_input)
                assert result['success'] is False
                assert 'error' in result
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_routing_timeout_simulation(self):
        """Test routing timeout handling (simulated)"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            # Mock a slow routing operation
            with patch('orthoroute.gpu_engine.ConflictResolver') as mock_resolver:
                def slow_resolve(*args, **kwargs):
                    time.sleep(0.1)  # Simulate slow operation
                    return []
                
                mock_resolver.return_value.resolve_conflicts.side_effect = slow_resolve
                
                board_data = {
                    'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 10000000, 'max_y': 10000000},
                    'grid': {'pitch_mm': 0.1, 'layers': 2},
                    'nets': [{'id': 1, 'name': 'TEST', 'pins': [
                        {'x': 1000000, 'y': 1000000, 'layer': 0},
                        {'x': 2000000, 'y': 2000000, 'layer': 0}
                    ]}],
                    'obstacles': []
                }
                
                start_time = time.time()
                result = engine.route_board(board_data)
                elapsed_time = time.time() - start_time
                
                # Should complete (even if slowly)
                assert elapsed_time >= 0.1
                assert 'stats' in result


class TestPerformanceMetrics:
    """Test performance measurement and optimization"""
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_timing_accuracy(self):
        """Test that timing measurements are accurate"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            with patch('orthoroute.gpu_engine.ConflictResolver') as mock_resolver:
                # Mock routing that takes known time
                def timed_resolve(*args, **kwargs):
                    time.sleep(0.05)  # 50ms
                    mock_net = Mock()
                    mock_net.routed = True
                    mock_net.total_length = 1.0
                    mock_net.via_count = 0
                    return [mock_net]
                
                mock_resolver.return_value.resolve_conflicts.side_effect = timed_resolve
                
                board_data = {
                    'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 10000000, 'max_y': 10000000},
                    'grid': {'pitch_mm': 0.2, 'layers': 2},
                    'nets': [{'id': 1, 'name': 'TEST', 'pins': [
                        {'x': 1000000, 'y': 1000000, 'layer': 0},
                        {'x': 2000000, 'y': 2000000, 'layer': 0}
                    ]}],
                    'obstacles': []
                }
                
                result = engine.route_board(board_data)
                
                # Timing should be approximately 50ms (with some tolerance)
                routing_time = result['stats']['routing_time_seconds']
                assert 0.04 <= routing_time <= 0.1  # 40-100ms tolerance
    
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_nets_per_second_calculation(self):
        """Test nets per second calculation"""
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            mock_nets = []
            for i in range(10):
                net = Mock()
                net.net_id = i + 1
                net.name = f'NET_{i+1}'
                net.routed = True
                net.route_path = [Mock(x=i, y=i, layer=0)]
                net.total_length = 1.0
                net.via_count = 0
                net.width_nm = 200000
                net.via_size_nm = 200000
                mock_nets.append(net)
            
            # Test with known timing
            routing_time = 2.0  # 2 seconds
            result = engine._generate_results(mock_nets, routing_time)
            
            expected_rate = 10 / 2.0  # 5 nets per second
            assert result['stats']['nets_per_second'] == expected_rate


# Integration test helpers
def create_test_board(width_mm: float, height_mm: float, net_count: int, layer_count: int = 4) -> Dict:
    """Create test board data with specified parameters"""
    width_nm = int(width_mm * 1000000)
    height_nm = int(height_mm * 1000000)
    
    nets = []
    for i in range(net_count):
        # Create random two-pin nets
        x1 = int(width_nm * 0.1 + (width_nm * 0.8) * (i / net_count))
        y1 = int(height_nm * 0.1)
        x2 = int(width_nm * 0.1 + (width_nm * 0.8) * (i / net_count))
        y2 = int(height_nm * 0.9)
        
        nets.append({
            'id': i + 1,
            'name': f'NET_{i+1:04d}',
            'pins': [
                {'x': x1, 'y': y1, 'layer': 0},
                {'x': x2, 'y': y2, 'layer': layer_count - 1}
            ],
            'priority': 5,
            'width_nm': 200000,
            'via_size_nm': 200000
        })
    
    return {
        'bounds': {'min_x': 0, 'min_y': 0, 'max_x': width_nm, 'max_y': height_nm},
        'grid': {'pitch_mm': 0.1, 'layers': layer_count},
        'config': {'max_iterations': 10, 'batch_size': 128},
        'nets': nets,
        'obstacles': []
    }


@pytest.mark.integration
class TestEngineIntegration:
    """Integration tests for complete routing workflow"""
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_complete_routing_workflow(self):
        """Test complete routing workflow from start to finish"""
        # Create small test board
        board_data = create_test_board(10.0, 10.0, 5, 2)
        
        engine = OrthoRouteEngine()
        result = engine.route_board(board_data)
        
        # Should complete successfully
        assert result['success'] is True
        assert result['stats']['total_nets'] == 5
        assert 'routed_nets' in result
        assert 'routing_time_seconds' in result['stats']
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available") 
    @pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine modules not available")
    def test_scaling_performance(self):
        """Test performance scaling with different board sizes"""
        test_cases = [
            (5.0, 5.0, 10),    # Small
            (10.0, 10.0, 25),  # Medium
            (20.0, 20.0, 50)   # Large
        ]
        
        engine = OrthoRouteEngine()
        results = []
        
        for width, height, net_count in test_cases:
            board_data = create_test_board(width, height, net_count)
            result = engine.route_board(board_data)
            
            if result['success']:
                results.append({
                    'size': f"{width}x{height}",
                    'nets': net_count,
                    'time': result['stats']['routing_time_seconds'],
                    'rate': result['stats']['nets_per_second']
                })
        
        # Should have results for all test cases
        assert len(results) >= 2
        
        # Performance should be reasonable
        for result in results:
            assert result['rate'] > 0  # Should process some nets per second
            assert result['time'] < 60  # Should complete within 1 minute


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])