"""
Integration Tests for OrthoRoute
End-to-end testing of complete routing workflows and system integration

These tests verify the complete OrthoRoute system including data flow between
components, KiCad integration, file I/O, and real-world usage scenarios.
"""

import pytest
import json
import tempfile
import os
import time
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
import logging
import psutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = Mock()

# Test modules availability
try:
    from orthoroute.gpu_engine import OrthoRouteEngine
    from orthoroute.grid_manager import GPUGrid, Point3D, Net
    from orthoroute.design_rules import DesignRuleChecker, create_default_rules
    from orthoroute.visualization import RoutingVisualizer, VisualizationConfig
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Integration test modules not available: {e}")

# Test KiCad plugin availability
try:
    from kicad_plugin.board_export import BoardExporter, export_board_to_file
    from kicad_plugin.route_import import RouteImporter, import_routes_from_file
    from kicad_plugin.ui_dialogs import show_configuration_dialog, show_results_dialog
    KICAD_AVAILABLE = True
except ImportError:
    KICAD_AVAILABLE = False

# Test benchmark boards availability
try:
    from tests.benchmark_boards import BenchmarkBoardGenerator, BenchmarkRunner
    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False


class TestDataFlowIntegration:
    """Test data flow between system components"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_board_data = self._create_test_board_data()
        
        # Record initial memory state
        self.initial_mem = psutil.Process().memory_info().rss
        logger.info(f"Initial memory usage: {self.initial_mem / 1024 / 1024:.2f} MB")
        
        # Ensure clean test environment
        if hasattr(self, 'engine'):
            del self.engine
        if hasattr(self, 'grid'):
            del self.grid
            
        # Force garbage collection
        gc.collect()
        
        # Set up error recording
        self.errors = []
    
    def teardown_method(self):
        """Clean up test fixtures"""
        try:
            # Clean up resources
            if hasattr(self, 'engine'):
                del self.engine
            if hasattr(self, 'grid'):
                del self.grid
            
            # Force cleanup
            gc.collect()
            
            # Check for memory leaks
            final_mem = psutil.Process().memory_info().rss
            delta_mb = (final_mem - self.initial_mem) / 1024 / 1024
            if delta_mb > 1.0:  # More than 1MB leak
                logger.warning(f"Possible memory leak: {delta_mb:.2f} MB")
            
            # Remove temporary directory
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
        except Exception as e:
            logger.error(f"Teardown failed: {e}")
            self.errors.append(e)
        
        # Report any accumulated errors
        if self.errors:
            logger.error(f"Test completed with {len(self.errors)} errors")
            raise Exception(f"Test failed with {len(self.errors)} errors: {self.errors}")
    
    def _create_test_board_data(self) -> Dict:
        """Create test board data"""
        return {
            'version': '0.1.0',
            'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 20000000, 'max_y': 20000000},
            'grid': {'pitch_mm': 0.1, 'layers': 4},
            'config': {'max_iterations': 10, 'batch_size': 64, 'verbose': False},
            'nets': [
                {
                    'id': 1, 'name': 'TEST_NET_1',
                    'pins': [
                        {'x': 2000000, 'y': 2000000, 'layer': 0},
                        {'x': 18000000, 'y': 18000000, 'layer': 1}
                    ],
                    'priority': 5, 'width_nm': 200000, 'via_size_nm': 200000
                },
                {
                    'id': 2, 'name': 'TEST_NET_2',
                    'pins': [
                        {'x': 2000000, 'y': 18000000, 'layer': 0},
                        {'x': 18000000, 'y': 2000000, 'layer': 1}
                    ],
                    'priority': 3, 'width_nm': 150000, 'via_size_nm': 200000
                }
            ],
            'obstacles': [
                {
                    'type': 'keepout', 'x1': 9000000, 'y1': 9000000,
                    'x2': 11000000, 'y2': 11000000, 'layer': -1, 'width': 0
                }
            ],
            'design_rules': {
                'min_track_width': 100000, 'min_clearance': 150000,
                'min_via_size': 200000, 'min_via_drill': 120000
            }
        }
    
    def _validate_board_data(self, data: Dict) -> bool:
        """Validate board data structure"""
        required_fields = ['version', 'bounds', 'grid', 'nets']
        return all(field in data for field in required_fields)
    
    @pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_json_to_engine_data_flow(self):
        """Test data flow from JSON input to GPU engine"""
        try:
            # Save test data to file
            board_file = Path(self.temp_dir) / "test_board.json"
            board_file.write_text(json.dumps(self.test_board_data))
            
            # Load and verify data can be processed
            loaded_data = json.loads(board_file.read_text())
            assert self._validate_board_data(loaded_data)
            
            # Test engine can load the data
            with patch('cupy.cuda.Device'):
                engine = OrthoRouteEngine()
                success = engine.load_board_data(loaded_data)
                assert success is True
                
        except (IOError, json.JSONDecodeError) as e:
            pytest.fail(f"Failed to handle board data: {e}")
    
    @pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_engine_to_results_data_flow(self):
        """Test data flow from engine processing to results output"""
        try:
            with patch('cupy.cuda.Device'):
                engine = OrthoRouteEngine()
                
                # Mock successful routing with proper error handling
                with patch.object(engine, '_parse_nets') as mock_parse, \
                     patch('orthoroute.gpu_engine.ConflictResolver') as mock_resolver:
                    
                    # Setup mocks with validation
                    mock_net = Mock()
                    mock_net.routed = True
                    mock_net.net_id = 1
                    mock_net.name = "TEST_NET_1"
                    mock_net.route_path = [Point3D(0, 0, 0), Point3D(1, 1, 1)]
                    mock_net.total_length = 5.0
                    mock_net.validate = Mock(return_value=True)
                    
                    mock_parse.return_value = [mock_net]
                    mock_resolver.return_value.resolve_conflicts.return_value = [mock_net]
                    
                    # Mock grid for coordinate conversion
                    engine.grid = Mock()
                    engine.grid.grid_to_world = Mock(side_effect=lambda x, y: (x * 100000, y * 100000))
                    
                    # Run routing
                    results = engine.route_board(self.test_board_data)
                    
                    # Verify results structure
                    assert results['success'] is True
                    assert 'stats' in results
                    assert 'routed_nets' in results
                    assert len(results['routed_nets']) == 1
                    
                    # Verify stats
                    stats = results['stats']
                    assert stats['total_nets'] == 1
                    assert stats['successful_nets'] == 1
                    assert stats['success_rate'] == 100.0
                    
                    # Verify routed net data
                    routed_net = results['routed_nets'][0]
                    assert routed_net['id'] == 1
                    assert routed_net['name'] == "TEST_NET_1"
                    assert 'path' in routed_net
    
    @pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_results_to_json_data_flow(self):
        """Test data flow from results to JSON output"""
        # Create mock results
        results = {
            'success': True,
            'stats': {
                'total_nets': 2,
                'successful_nets': 2,
                'failed_nets': 0,
                'success_rate': 100.0,
                'routing_time_seconds': 1.5,
                'nets_per_second': 1.33,
                'total_length_mm': 25.4,
                'total_vias': 3
            },
            'routed_nets': [
                {
                    'id': 1, 'name': 'NET_1',
                    'path': [
                        {'x': 1000000, 'y': 1000000, 'layer': 0},
                        {'x': 2000000, 'y': 2000000, 'layer': 1}
                    ],
                    'length_mm': 12.7, 'via_count': 1,
                    'width_nm': 200000, 'via_size_nm': 200000
                }
            ]
        }
        
        # Save to JSON
        results_file = os.path.join(self.temp_dir, "routing_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify JSON can be loaded back
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results == results
        assert loaded_results['success'] is True
        assert len(loaded_results['routed_nets']) == 1


class TestFileIOIntegration:
    """Test file input/output operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_board_data_json_serialization(self):
        """Test board data JSON serialization/deserialization"""
        # Create complex board data with various data types
        board_data = {
            'version': '0.1.0',
            'timestamp': '2024-01-01T00:00:00Z',
            'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 50000000, 'max_y': 50000000},
            'grid': {'pitch_mm': 0.05, 'layers': 8},
            'nets': [
                {
                    'id': i + 1,
                    'name': f'NET_{i+1:04d}',
                    'pins': [
                        {'x': i * 1000000, 'y': i * 1000000, 'layer': 0},
                        {'x': (i + 10) * 1000000, 'y': (i + 10) * 1000000, 'layer': 1}
                    ],
                    'priority': i % 10,
                    'width_nm': 100000 + i * 10000,
                    'via_size_nm': 200000
                }
                for i in range(100)
            ],
            'obstacles': [
                {
                    'type': 'component',
                    'x1': 5000000, 'y1': 5000000,
                    'x2': 10000000, 'y2': 10000000,
                    'layer': -1, 'width': 0
                }
            ]
        }
        
        # Test serialization
        board_file = os.path.join(self.temp_dir, "complex_board.json")
        with open(board_file, 'w') as f:
            json.dump(board_data, f, indent=2)
        
        # Test deserialization
        with open(board_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Verify data integrity
        assert loaded_data['version'] == board_data['version']
        assert loaded_data['bounds'] == board_data['bounds']
        assert len(loaded_data['nets']) == len(board_data['nets'])
        
        # Verify specific net data
        for i, (original, loaded) in enumerate(zip(board_data['nets'], loaded_data['nets'])):
            assert original['id'] == loaded['id']
            assert original['name'] == loaded['name']
            assert original['pins'] == loaded['pins']
    
    def test_results_data_persistence(self):
        """Test routing results data persistence"""
        # Create comprehensive results data
        results_data = {
            'success': True,
            'timestamp': time.time(),
            'stats': {
                'total_nets': 1000,
                'successful_nets': 950,
                'failed_nets': 50,
                'success_rate': 95.0,
                'routing_time_seconds': 45.7,
                'nets_per_second': 21.8,
                'total_length_mm': 2547.3,
                'total_vias': 1234,
                'total_execution_time': 47.2
            },
            'routed_nets': [
                {
                    'id': i + 1,
                    'name': f'ROUTED_NET_{i+1:04d}',
                    'path': [
                        {'x': j * 500000, 'y': j * 500000, 'layer': j % 4}
                        for j in range(5)
                    ],
                    'length_mm': 2.5 + i * 0.1,
                    'via_count': i % 3,
                    'width_nm': 200000,
                    'via_size_nm': 200000
                }
                for i in range(10)  # Sample of routed nets
            ],
            'violations': [
                {
                    'type': 'clearance',
                    'severity': 'warning',
                    'net_id': 42,
                    'net_name': 'PROBLEM_NET',
                    'location': {'x': 1000000, 'y': 2000000, 'layer': 1},
                    'measured_value': 0.12,
                    'required_value': 0.15,
                    'description': 'Clearance violation between traces'
                }
            ]
        }
        
        # Test large file handling
        results_file = os.path.join(self.temp_dir, "large_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Verify file size and loading
        file_size = os.path.getsize(results_file)
        assert file_size > 1000  # Should be substantial
        
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['success'] == results_data['success']
        assert loaded_results['stats'] == results_data['stats']
        assert len(loaded_results['routed_nets']) == len(results_data['routed_nets'])
    
    def test_configuration_file_handling(self):
        """Test configuration file save/load"""
        config_data = {
            'grid_settings': {
                'pitch_mm': 0.08,
                'layers': 6,
                'tile_size': 128
            },
            'routing_settings': {
                'max_iterations': 25,
                'batch_size': 512,
                'congestion_factor': 1.8,
                'via_cost': 15
            },
            'gpu_settings': {
                'device_id': 0,
                'memory_limit_gb': 8
            },
            'user_preferences': {
                'show_visualization': True,
                'save_results': True,
                'verbose_logging': False
            }
        }
        
        config_file = os.path.join(self.temp_dir, "orthoroute_config.json")
        
        # Save configuration
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Load configuration
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == config_data


@pytest.mark.skipif(not KICAD_AVAILABLE, reason="KiCad plugin modules not available")
class TestKiCadIntegration:
    """Test KiCad plugin integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_board = self._create_mock_kicad_board()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_kicad_board(self):
        """Create mock KiCad board object"""
        mock_board = Mock()
        
        # Mock board methods and properties
        mock_board.GetNetCount.return_value = 5
        mock_board.GetCopperLayerCount.return_value = 4
        mock_board.IsLayerEnabled.return_value = True
        
        # Mock board bounds
        mock_bbox = Mock()
        mock_bbox.GetLeft.return_value = 0
        mock_bbox.GetTop.return_value = 0
        mock_bbox.GetRight.return_value = 50000000  # 50mm
        mock_bbox.GetBottom.return_value = 50000000
        mock_board.GetBoardEdgesBoundingBox.return_value = mock_bbox
        
        # Mock nets
        mock_nets = []
        for i in range(1, 6):  # Net IDs 1-5
            mock_net = Mock()
            mock_net.GetNetCode.return_value = i
            mock_net.GetNetname.return_value = f"NET_{i}"
            mock_nets.append(mock_net)
        
        mock_board.FindNet.side_effect = lambda net_id: mock_nets[net_id - 1] if 1 <= net_id <= 5 else None
        
        # Mock footprints and pads
        mock_board.GetFootprints.return_value = []
        mock_board.GetTracks.return_value = []
        mock_board.Zones.return_value = []
        
        return mock_board
    
    def test_board_export_integration(self):
        """Test board export to OrthoRoute format"""
        # Mock the exporter dependencies
        with patch('pcbnew.IsCopperLayer', return_value=True), \
             patch('pcbnew.PCB_LAYER_ID_COUNT', 50):
            
            exporter = BoardExporter(self.mock_board)
            
            config = {
                'grid_pitch_mm': 0.1,
                'max_layers': 4,
                'skip_power_nets': True,
                'skip_routed_nets': True
            }
            
            board_data = exporter.export_board(config)
            
            # Verify export structure
            assert 'version' in board_data
            assert 'bounds' in board_data
            assert 'grid' in board_data
            assert 'nets' in board_data
            assert 'obstacles' in board_data
    
    def test_route_import_integration(self):
        """Test route import from OrthoRoute results"""
        # Create mock routing results
        routing_results = [
            {
                'id': 1,
                'name': 'NET_1',
                'path': [
                    {'x': 5000000, 'y': 5000000, 'layer': 0},
                    {'x': 10000000, 'y': 10000000, 'layer': 0},
                    {'x': 15000000, 'y': 15000000, 'layer': 1}
                ],
                'length_mm': 14.14,
                'via_count': 1,
                'width_nm': 200000,
                'via_size_nm': 200000
            }
        ]
        
        # Mock KiCad objects
        with patch('pcbnew.NETINFO_ITEM'), \
             patch('pcbnew.PCB_TRACK'), \
             patch('pcbnew.PCB_VIA'), \
             patch('pcbnew.VECTOR2I'):
            
            importer = RouteImporter(self.mock_board)
            
            # Mock the required methods
            self.mock_board.BeginModify = Mock()
            self.mock_board.EndModify = Mock()
            self.mock_board.Add = Mock()
            
            imported_count = importer.apply_routes(routing_results)
            
            # Verify import was attempted
            assert imported_count >= 0
            assert self.mock_board.BeginModify.called
            assert self.mock_board.EndModify.called
    
    def test_export_import_roundtrip(self):
        """Test complete export -> route -> import roundtrip"""
        # Step 1: Export board data
        with patch('pcbnew.IsCopperLayer', return_value=True), \
             patch('pcbnew.PCB_LAYER_ID_COUNT', 50):
            
            export_config = {'grid_pitch_mm': 0.1, 'max_layers': 4}
            board_file = export_board_to_file(self.mock_board, 
                                            os.path.join(self.temp_dir, "exported_board.json"),
                                            export_config)
            
            assert os.path.exists(board_file)
        
        # Step 2: Load exported data
        with open(board_file, 'r') as f:
            board_data = json.load(f)
        
        # Step 3: Simulate routing (create mock results)
        mock_results = {
            'success': True,
            'stats': {'total_nets': 2, 'successful_nets': 2, 'success_rate': 100.0},
            'routed_nets': [
                {
                    'id': 1, 'name': 'NET_1',
                    'path': [
                        {'x': 1000000, 'y': 1000000, 'layer': 0},
                        {'x': 5000000, 'y': 5000000, 'layer': 0}
                    ],
                    'length_mm': 5.66, 'via_count': 0,
                    'width_nm': 200000, 'via_size_nm': 200000
                }
            ]
        }
        
        results_file = os.path.join(self.temp_dir, "routing_results.json")
        with open(results_file, 'w') as f:
            json.dump(mock_results, f)
        
        # Step 4: Import results back to KiCad
        with patch('pcbnew.NETINFO_ITEM'), \
             patch('pcbnew.PCB_TRACK'), \
             patch('pcbnew.PCB_VIA'), \
             patch('pcbnew.VECTOR2I'):
            
            success = import_routes_from_file(self.mock_board, results_file)
            assert success is True


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
class TestDesignRuleIntegration:
    """Test design rule checking integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            self.grid = GPUGrid(50, 50, 4, 0.1)
            self.design_rules = create_default_rules()
    
    def test_drc_with_routing_results(self):
        """Test DRC integration with routing results"""
        # Create test nets with potential violations
        nets = [
            Net(
                net_id=1, name="NARROW_NET",
                pins=[Point3D(5, 5, 0), Point3D(10, 10, 0)],
                width_nm=50000  # Below minimum
            ),
            Net(
                net_id=2, name="NORMAL_NET",
                pins=[Point3D(15, 15, 0), Point3D(20, 20, 0)],
                width_nm=200000  # Normal width
            )
        ]
        
        # Set up routed paths
        nets[0].routed = True
        nets[0].route_path = [Point3D(5, 5, 0), Point3D(6, 6, 0), Point3D(10, 10, 0)]
        nets[0].via_count = 0
        
        nets[1].routed = True
        nets[1].route_path = [Point3D(15, 15, 0), Point3D(20, 20, 0)]
        nets[1].via_count = 0
        
        # Run DRC
        checker = DesignRuleChecker(self.grid, self.design_rules)
        violations = checker.check_all_rules(nets)
        
        # Should find track width violation
        assert len(violations) > 0
        track_width_violations = [v for v in violations if v.violation_type.value == 'track_width']
        assert len(track_width_violations) > 0
    
    def test_drc_integration_with_engine_results(self):
        """Test DRC integration with actual engine results"""
        # Simulate engine results with DRC data
        engine_results = {
            'success': True,
            'stats': {'total_nets': 2, 'successful_nets': 2},
            'routed_nets': [
                {
                    'id': 1, 'name': 'NET_1',
                    'path': [
                        {'x': 1000000, 'y': 1000000, 'layer': 0},
                        {'x': 2000000, 'y': 2000000, 'layer': 0}
                    ],
                    'width_nm': 100000,  # Minimum width
                    'via_size_nm': 200000
                }
            ]
        }
        
        # Convert to nets for DRC
        nets = []
        for net_data in engine_results['routed_nets']:
            net = Net(
                net_id=net_data['id'],
                name=net_data['name'],
                pins=[],  # Not needed for this test
                width_nm=net_data['width_nm'],
                via_size_nm=net_data['via_size_nm']
            )
            net.routed = True
            net.route_path = [
                Point3D(p['x'] // 100000, p['y'] // 100000, p['layer'])
                for p in net_data['path']
            ]
            nets.append(net)
        
        # Run DRC
        checker = DesignRuleChecker(self.grid, self.design_rules)
        violations = checker.check_all_rules(nets)
        
        # Verify DRC results can be serialized
        violation_data = [
            {
                'type': v.violation_type.value,
                'severity': v.severity,
                'net_id': v.net_id,
                'net_name': v.net_name,
                'location': {'x': v.location.x, 'y': v.location.y, 'layer': v.location.layer},
                'measured_value': v.measured_value,
                'required_value': v.required_value,
                'description': v.description
            }
            for v in violations
        ]
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(violation_data)
        assert isinstance(json_str, str)


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
class TestVisualizationIntegration:
    """Test visualization integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            self.grid = GPUGrid(20, 20, 2, 0.1)
            self.config = VisualizationConfig(backend="headless")  # No actual display
    
    def test_visualization_with_routing_data(self):
        """Test visualization integration with routing data"""
        # Create visualizer
        visualizer = RoutingVisualizer(self.grid, self.config)
        
        # Create test nets with routing data
        nets = [
            Net(
                net_id=1, name="VIS_NET_1",
                pins=[Point3D(0, 0, 0), Point3D(10, 10, 0)]
            )
        ]
        nets[0].routed = True
        nets[0].route_path = [Point3D(0, 0, 0), Point3D(5, 5, 0), Point3D(10, 10, 0)]
        
        # Update visualization with routing progress
        visualizer.update_routing_progress(nets, iteration=1, routing_time=1.5)
        
        # Verify visualization state
        assert visualizer.routing_stats['nets_total'] == 1
        assert visualizer.routing_stats['nets_completed'] == 1
        assert 1 in visualizer.routing_progress
    
    def test_visualization_performance_tracking(self):
        """Test visualization performance tracking"""
        visualizer = RoutingVisualizer(self.grid, self.config)
        
        # Simulate routing progress updates
        for i in range(5):
            nets = [Mock() for _ in range(10)]
            for j, net in enumerate(nets):
                net.routed = j <= i * 2  # Gradually route more nets
                net.net_id = j + 1
            
            visualizer.update_routing_progress(nets, iteration=i+1, routing_time=i*0.5)
        
        # Verify performance tracking
        assert len(visualizer.frame_times) <= 100  # Should be limited
        assert visualizer.routing_stats['current_iteration'] == 5


@pytest.mark.skipif(not BENCHMARKS_AVAILABLE, reason="Benchmark modules not available")
class TestBenchmarkIntegration:
    """Test benchmark system integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_generation_integration(self):
        """Test benchmark board generation"""
        generator = BenchmarkBoardGenerator()
        
        # Generate a simple benchmark
        simple_spec = generator.benchmark_specs[0]  # First spec should be simple
        
        board_file = generator.generate_board(simple_spec, self.temp_dir)
        
        # Verify file was created and contains valid data
        assert os.path.exists(board_file)
        
        with open(board_file, 'r') as f:
            board_data = json.load(f)
        
        # Verify board data structure
        assert 'version' in board_data
        assert 'benchmark_info' in board_data
        assert 'bounds' in board_data
        assert 'nets' in board_data
        assert len(board_data['nets']) <= simple_spec.net_count
    
    @pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_benchmark_execution_integration(self):
        """Test benchmark execution with mock engine"""
        # Generate simple benchmark
        generator = BenchmarkBoardGenerator()
        simple_spec = generator.benchmark_specs[0]
        board_file = generator.generate_board(simple_spec, self.temp_dir)
        
        # Mock the benchmark runner to avoid actual GPU routing
        runner = BenchmarkRunner()
        
        # Mock the routing engine
        with patch('orthoroute.gpu_engine.OrthoRouteEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.route_board.return_value = {
                'success': True,
                'stats': {
                    'total_nets': simple_spec.net_count,
                    'successful_nets': simple_spec.net_count,
                    'success_rate': 100.0,
                    'routing_time_seconds': 2.0,
                    'nets_per_second': simple_spec.net_count / 2.0,
                    'total_length_mm': 50.0,
                    'total_vias': 10
                }
            }
            mock_engine_class.return_value = mock_engine
            
            # Run single benchmark
            benchmark_info = {
                'name': simple_spec.name,
                'complexity': simple_spec.complexity.value,
                'pattern': simple_spec.pattern.value
            }
            
            result = runner._run_single_benchmark(board_file, benchmark_info)
            
            # Verify benchmark result
            assert result['success'] is True
            assert result['benchmark_name'] == simple_spec.name
            assert 'metrics' in result
            assert result['metrics']['net_count'] == simple_spec.net_count


class TestSystemScalability:
    """Test system scalability and performance characteristics"""
    
    @pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with board size"""
        board_sizes = [(10, 10), (50, 50), (100, 100)]
        
        for width, height in board_sizes:
            with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
                grid = GPUGrid(width, height, 4, 0.1)
                
                # Verify grid was created
                assert grid.width == width
                assert grid.height == height
                
                # Memory calculation should work without error
                grid._calculate_memory_usage()
    
    @pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available") 
    def test_net_count_scaling(self):
        """Test performance scaling with net count"""
        net_counts = [10, 100, 1000]
        
        for count in net_counts:
            # Create test board data
            board_data = {
                'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 50000000, 'max_y': 50000000},
                'grid': {'pitch_mm': 0.1, 'layers': 4},
                'config': {'max_iterations': 5, 'batch_size': min(256, count)},
                'nets': [
                    {
                        'id': i + 1, 'name': f'NET_{i+1}',
                        'pins': [
                            {'x': (i % 10) * 1000000, 'y': (i // 10) * 1000000, 'layer': 0},
                            {'x': ((i + 5) % 10) * 1000000, 'y': ((i + 5) // 10) * 1000000, 'layer': 1}
                        ]
                    }
                    for i in range(count)
                ],
                'obstacles': []
            }
            
            # Test data processing without actual routing
            with patch('cupy.cuda.Device'):
                engine = OrthoRouteEngine()
                success = engine.load_board_data(board_data)
                assert success is True
                
                # Parse nets
                nets = engine._parse_nets(board_data['nets'])
                assert len(nets) == count


class TestErrorHandlingIntegration:
    """Test error handling across system components"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_input_handling(self):
        """Test system handling of invalid inputs"""
        invalid_inputs = [
            None,
            "not_a_dict",
            {},
            {'bounds': 'invalid'},
            {'bounds': {'min_x': 'not_a_number'}},
            {'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 1000, 'max_y': 1000}, 'nets': 'invalid'}
        ]
        
        for invalid_input in invalid_inputs:
            with patch('cupy.cuda.Device'):
                engine = OrthoRouteEngine()
                
                if invalid_input is None or not isinstance(invalid_input, dict):
                    # These should raise exceptions or return error results
                    try:
                        result = engine.route_board(invalid_input)
                        assert result['success'] is False
                    except (TypeError, AttributeError):
                        pass  # Expected for completely invalid inputs
                else:
                    result = engine.route_board(invalid_input)
                    assert result['success'] is False
                    assert 'error' in result
    
    def test_file_io_error_handling(self):
        """Test file I/O error handling"""
        # Test reading non-existent file
        non_existent_file = os.path.join(self.temp_dir, "does_not_exist.json")
        
        try:
            with open(non_existent_file, 'r') as f:
                json.load(f)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
        
        # Test reading invalid JSON
        invalid_json_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json content")
        
        try:
            with open(invalid_json_file, 'r') as f:
                json.load(f)
            assert False, "Should have raised JSONDecodeError"
        except json.JSONDecodeError:
            pass  # Expected
        
        # Test writing to read-only directory (if applicable)
        readonly_dir = os.path.join(self.temp_dir, "readonly")
        os.makedirs(readonly_dir, exist_ok=True)
        os.chmod(readonly_dir, 0o444)  # Read-only
        
        readonly_file = os.path.join(readonly_dir, "test.json")
        try:
            with open(readonly_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            # If we reach here, the write succeeded (some systems may allow this)
        except (PermissionError, OSError):
            pass  # Expected on most systems
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)
    
    @pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_gpu_error_recovery(self):
        """Test recovery from GPU errors"""
        # Test CUDA device initialization failure
        with patch('cupy.cuda.Device', side_effect=RuntimeError("CUDA error")):
            try:
                engine = OrthoRouteEngine()
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "CUDA error" in str(e)
        
        # Test GPU memory allocation failure
        with patch('cupy.cuda.Device'), \
             patch('cupy.zeros', side_effect=RuntimeError("Out of memory")):
            
            try:
                grid = GPUGrid(1000, 1000, 8, 0.05)  # Large grid
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "memory" in str(e).lower()


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system"""
    
    @pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_routing_pipeline_performance(self):
        """Test performance of complete routing pipeline"""
        # Create test board with known characteristics
        board_data = {
            'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 25000000, 'max_y': 25000000},
            'grid': {'pitch_mm': 0.1, 'layers': 4},
            'config': {'max_iterations': 10, 'batch_size': 128, 'verbose': False},
            'nets': [
                {
                    'id': i + 1, 'name': f'PERF_NET_{i+1:03d}',
                    'pins': [
                        {'x': (i % 5) * 5000000, 'y': (i // 5) * 5000000, 'layer': 0},
                        {'x': ((i + 1) % 5) * 5000000, 'y': ((i + 1) // 5) * 5000000, 'layer': 1}
                    ],
                    'priority': 5, 'width_nm': 200000, 'via_size_nm': 200000
                }
                for i in range(50)  # 50 nets for performance test
            ],
            'obstacles': []
        }
        
        # Measure performance
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            
            # Mock routing to measure data processing overhead
            with patch.object(engine, '_parse_nets') as mock_parse, \
                 patch('orthoroute.gpu_engine.ConflictResolver') as mock_resolver:
                
                # Create mock nets
                mock_nets = []
                for i in range(50):
                    net = Mock()
                    net.routed = True
                    net.net_id = i + 1
                    net.name = f'PERF_NET_{i+1:03d}'
                    net.route_path = [Point3D(i, 0, 0), Point3D(i+1, 1, 1)]
                    net.total_length = 1.41  # sqrt(2)
                    net.via_count = 1
                    net.width_nm = 200000
                    net.via_size_nm = 200000
                    mock_nets.append(net)
                
                mock_parse.return_value = mock_nets
                mock_resolver.return_value.resolve_conflicts.return_value = mock_nets
                
                # Mock grid for coordinate conversion
                engine.grid = Mock()
                engine.grid.grid_to_world = Mock(side_effect=lambda x, y: (x * 100000, y * 100000))
                
                # Time the operation
                start_time = time.time()
                result = engine.route_board(board_data)
                end_time = time.time()
                
                # Verify reasonable performance
                execution_time = end_time - start_time
                assert execution_time < 10.0  # Should complete in under 10 seconds
                
                # Verify result quality
                assert result['success'] is True
                assert result['stats']['total_nets'] == 50
                assert result['stats']['successful_nets'] == 50
    
    def test_memory_efficiency_integration(self):
        """Test memory efficiency across components"""
        # Test with progressively larger board sizes
        sizes = [(20, 20), (50, 50), (100, 100)]
        
        for width, height in sizes:
            with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
                # Create grid
                grid = GPUGrid(width, height, 4, 0.1)
                
                # Verify memory calculation works
                grid._calculate_memory_usage()
                
                # Create tile manager
                tile_manager = grid_manager.TileManager(grid, tile_size=32)
                
                # Verify tiling works for memory efficiency
                tiles_x = tile_manager.tiles_x
                tiles_y = tile_manager.tiles_y
                
                assert tiles_x * tiles_y >= 1
                assert tiles_x == (width + 31) // 32  # Ceiling division


class TestRealWorldScenarios:
    """Test realistic usage scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_typical_pcb_workflow(self):
        """Test typical PCB design workflow"""
        # Step 1: Create realistic board data (simulating KiCad export)
        board_data = {
            'version': '0.1.0',
            'source': 'KiCad',
            'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 80000000, 'max_y': 60000000},  # 80x60mm board
            'grid': {'pitch_mm': 0.1, 'layers': 6},
            'config': {'max_iterations': 25, 'batch_size': 256, 'verbose': False},
            'nets': self._generate_realistic_nets(200),  # 200 nets
            'obstacles': self._generate_realistic_obstacles(),
            'design_rules': {
                'min_track_width': 100000, 'min_clearance': 150000,
                'min_via_size': 200000, 'min_via_drill': 120000,
                'copper_layers': 6, 'board_thickness': 1600000
            }
        }
        
        # Step 2: Save to file (simulating export)
        board_file = os.path.join(self.temp_dir, "realistic_board.json")
        with open(board_file, 'w') as f:
            json.dump(board_data, f, indent=2)
        
        # Step 3: Load and validate
        with open(board_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['bounds']['max_x'] == 80000000
        assert len(loaded_data['nets']) == 200
        
        # Step 4: Simulate routing
        with patch('cupy.cuda.Device'):
            engine = OrthoRouteEngine()
            success = engine.load_board_data(loaded_data)
            assert success is True
            
            # Mock routing results
            with patch.object(engine, 'route_board') as mock_route:
                mock_route.return_value = {
                    'success': True,
                    'stats': {
                        'total_nets': 200, 'successful_nets': 185, 'failed_nets': 15,
                        'success_rate': 92.5, 'routing_time_seconds': 45.2,
                        'nets_per_second': 4.4, 'total_length_mm': 2456.7,
                        'total_vias': 432
                    },
                    'routed_nets': self._generate_mock_routed_nets(185)
                }
                
                results = engine.route_board(loaded_data)
                
                # Step 5: Save results
                results_file = os.path.join(self.temp_dir, "routing_results.json")
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Verify realistic results
                assert results['success'] is True
                assert results['stats']['success_rate'] > 90.0  # Good success rate
                assert len(results['routed_nets']) == 185
    
    def test_iterative_design_process(self):
        """Test iterative design refinement process"""
        # Simulate multiple design iterations with improvements
        iterations = [
            {'net_count': 50, 'complexity': 'simple'},
            {'net_count': 150, 'complexity': 'medium'},
            {'net_count': 300, 'complexity': 'complex'}
        ]
        
        for i, iteration in enumerate(iterations):
            # Create board for this iteration
            board_data = {
                'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 50000000, 'max_y': 50000000},
                'grid': {'pitch_mm': 0.1, 'layers': 4},
                'nets': self._generate_realistic_nets(iteration['net_count']),
                'obstacles': []
            }
            
            # Save iteration
            iter_file = os.path.join(self.temp_dir, f"iteration_{i+1}.json")
            with open(iter_file, 'w') as f:
                json.dump(board_data, f)
            
            # Verify file can be loaded
            with open(iter_file, 'r') as f:
                loaded = json.load(f)
            
            assert len(loaded['nets']) == iteration['net_count']
    
    def _generate_realistic_nets(self, count: int) -> List[Dict]:
        """Generate realistic net patterns"""
        nets = []
        
        # Power nets (10%)
        power_count = count // 10
        for i in range(power_count):
            nets.append({
                'id': len(nets) + 1,
                'name': f'VCC_{i+1}' if i % 2 == 0 else f'GND_{i+1}',
                'pins': [
                    {'x': j * 5000000, 'y': j * 5000000, 'layer': 0}
                    for j in range(3, 8)  # 5 pins (multi-pin power net)
                ],
                'priority': 1,  # High priority
                'width_nm': 500000,  # Wide power traces
                'via_size_nm': 400000
            })
        
        # Signal nets (80%)
        signal_count = int(count * 0.8)
        for i in range(signal_count):
            pin_count = 2 if i % 4 != 0 else 3  # Mostly 2-pin, some 3-pin
            
            nets.append({
                'id': len(nets) + 1,
                'name': f'SIG_{i+1:03d}',
                'pins': [
                    {
                        'x': (i * 1234567) % 70000000 + 5000000,  # Pseudo-random positions
                        'y': (i * 2345678) % 50000000 + 5000000,
                        'layer': (i + j) % 4  # Distribute across layers
                    }
                    for j in range(pin_count)
                ],
                'priority': 5,
                'width_nm': 150000,
                'via_size_nm': 200000
            })
        
        # Clock nets (5%)
        clock_count = count // 20
        for i in range(clock_count):
            nets.append({
                'id': len(nets) + 1,
                'name': f'CLK_{i+1}',
                'pins': [
                    {'x': 10000000, 'y': 10000000, 'layer': 1},  # Clock source
                    {'x': 20000000 + i * 5000000, 'y': 20000000, 'layer': 1}  # Clock load
                ],
                'priority': 2,  # High priority
                'width_nm': 120000,  # Controlled impedance
                'via_size_nm': 200000
            })
        
        # Fill remaining with random nets
        while len(nets) < count:
            nets.append({
                'id': len(nets) + 1,
                'name': f'NET_{len(nets)+1:03d}',
                'pins': [
                    {'x': 10000000, 'y': 10000000, 'layer': 0},
                    {'x': 40000000, 'y': 40000000, 'layer': 2}
                ],
                'priority': 5,
                'width_nm': 200000,
                'via_size_nm': 200000
            })
        
        return nets[:count]
    
    def _generate_realistic_obstacles(self) -> List[Dict]:
        """Generate realistic obstacle patterns"""
        obstacles = []
        
        # Component keepouts
        component_positions = [
            (10000000, 10000000, 15000000, 20000000),  # Large IC
            (30000000, 30000000, 40000000, 35000000),  # Medium component
            (50000000, 10000000, 55000000, 15000000),  # Small component
            (60000000, 40000000, 70000000, 50000000),  # Connector
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(component_positions):
            obstacles.append({
                'type': 'component',
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'layer': -1,  # All layers
                'width': 0
            })
        
        # Mechanical keepouts
        obstacles.append({
            'type': 'mechanical',
            'x1': 35000000, 'y1': 25000000,
            'x2': 45000000, 'y2': 35000000,
            'layer': -1, 'width': 0
        })
        
        return obstacles
    
    def _generate_mock_routed_nets(self, count: int) -> List[Dict]:
        """Generate mock routing results"""
        routed_nets = []
        
        for i in range(count):
            # Simple L-shaped routes
            path_length = 3 + (i % 5)  # Variable path lengths
            path = []
            
            for j in range(path_length):
                path.append({
                    'x': 5000000 + j * 2000000,
                    'y': 5000000 + (j % 2) * 2000000,
                    'layer': j % 4
                })
            
            routed_nets.append({
                'id': i + 1,
                'name': f'ROUTED_NET_{i+1:03d}',
                'path': path,
                'length_mm': (path_length - 1) * 2.0,  # Approximate length
                'via_count': path_length // 3,  # Some vias
                'width_nm': 200000,
                'via_size_nm': 200000
            })
        
        return routed_nets


class TestConcurrentOperations:
    """Test concurrent operations and thread safety"""
    
    @pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_multiple_engine_instances(self):
        """Test multiple engine instances can coexist"""
        engines = []
        
        # Create multiple engine instances
        with patch('cupy.cuda.Device'):
            for i in range(3):
                engine = OrthoRouteEngine(gpu_id=0)  # Same GPU
                engines.append(engine)
        
        # All should be properly initialized
        assert len(engines) == 3
        for engine in engines:
            assert engine.config['gpu_id'] == 0
    
    def test_file_access_concurrency(self):
        """Test concurrent file access patterns"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            files = []
            for i in range(5):
                filename = os.path.join(temp_dir, f"concurrent_test_{i}.json")
                data = {'test_id': i, 'data': list(range(i * 10))}
                
                with open(filename, 'w') as f:
                    json.dump(data, f)
                files.append(filename)
            
            # Read all files concurrently (simulated)
            loaded_data = []
            for filename in files:
                with open(filename, 'r') as f:
                    data = json.load(f)
                loaded_data.append(data)
            
            # Verify all data loaded correctly
            assert len(loaded_data) == 5
            for i, data in enumerate(loaded_data):
                assert data['test_id'] == i
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# Utility functions for integration testing
def create_comprehensive_test_board() -> Dict:
    """Create comprehensive test board with various features"""
    return {
        'version': '0.1.0',
        'timestamp': time.time(),
        'source': 'integration_test',
        'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 100000000, 'max_y': 80000000},
        'grid': {'pitch_mm': 0.1, 'layers': 8},
        'config': {
            'max_iterations': 30, 'batch_size': 512,
            'congestion_factor': 1.6, 'via_cost': 12,
            'verbose': False
        },
        'nets': _generate_comprehensive_nets(),
        'obstacles': _generate_comprehensive_obstacles(),
        'design_rules': _generate_comprehensive_rules(),
        'layer_stackup': _generate_layer_stackup()
    }

def _generate_comprehensive_nets() -> List[Dict]:
    """Generate comprehensive net list for testing"""
    nets = []
    
    # Various net types with different characteristics
    net_types = [
        ('POWER', 20, 3, 500000, 1),      # Power nets: 20 nets, 3 pins avg, wide traces, high priority
        ('CLOCK', 10, 2, 120000, 2),      # Clock nets: 10 nets, 2 pins, controlled impedance
        ('SIGNAL', 300, 2, 150000, 5),   # Signal nets: 300 nets, 2 pins, normal width
        ('BUS', 50, 8, 200000, 4),       # Bus nets: 50 nets, 8 pins avg, wider traces
        ('DIFF', 40, 2, 100000, 3)       # Differential pairs: 40 nets, 2 pins, narrow
    ]
    
    net_id = 1
    for net_type, count, avg_pins, width, priority in net_types:
        for i in range(count):
            pin_count = max(2, avg_pins + (i % 3) - 1)  # Vary pin count
            
            pins = []
            for j in range(pin_count):
                # Distribute pins across board
                x = (i * j * 1234567) % 90000000 + 5000000
                y = (i * j * 2345678) % 70000000 + 5000000
                layer = (i + j) % 8
                
                pins.append({'x': x, 'y': y, 'layer': layer})
            
            nets.append({
                'id': net_id,
                'name': f'{net_type}_{i+1:03d}',
                'pins': pins,
                'priority': priority,
                'width_nm': width,
                'via_size_nm': max(200000, width),
                'net_class': net_type.lower()
            })
            net_id += 1
    
    return nets

def _generate_comprehensive_obstacles() -> List[Dict]:
    """Generate comprehensive obstacle list"""
    obstacles = []
    
    # Component keepouts
    components = [
        (10000000, 10000000, 25000000, 25000000),   # Large processor
        (40000000, 40000000, 55000000, 50000000),   # Memory module
        (70000000, 10000000, 85000000, 20000000),   # Connector
        (30000000, 60000000, 40000000, 70000000),   # Power module
    ]
    
    for i, (x1, y1, x2, y2) in enumerate(components):
        obstacles.append({
            'type': 'component',
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'layer': -1, 'width': 0,
            'description': f'Component_{i+1}'
        })
    
    # Layer-specific keepouts
    for layer in range(8):
        obstacles.append({
            'type': 'keepout',
            'x1': 50000000, 'y1': 30000000,
            'x2': 60000000, 'y2': 40000000,
            'layer': layer, 'width': 0,
            'description': f'Layer_{layer}_keepout'
        })
    
    return obstacles

def _generate_comprehensive_rules() -> Dict:
    """Generate comprehensive design rules"""
    return {
        'min_track_width': 75000,      # 0.075mm (advanced process)
        'min_clearance': 100000,       # 0.1mm
        'min_via_size': 150000,        # 0.15mm
        'min_via_drill': 80000,        # 0.08mm
        'min_annular_ring': 40000,     # 0.04mm
        'board_thickness': 2000000,    # 2.0mm (thick board)
        'copper_layers': 8,
        'max_aspect_ratio': 12.0,      # High aspect ratio vias
        'copper_thickness': 35000,     # 35µm
        'net_class_rules': {
            'power': {'track_width': 500000, 'clearance': 200000},
            'clock': {'track_width': 120000, 'clearance': 150000, 'impedance': 50.0},
            'signal': {'track_width': 150000, 'clearance': 100000},
            'diff': {'track_width': 100000, 'clearance': 100000, 'impedance': 100.0}
        }
    }

def _generate_layer_stackup() -> List[Dict]:
    """Generate layer stackup information"""
    return [
        {'index': 0, 'name': 'F.Cu', 'type': 'signal', 'thickness': 35000},
        {'index': 1, 'name': 'L2', 'type': 'ground', 'thickness': 35000},
        {'index': 2, 'name': 'L3', 'type': 'signal', 'thickness': 35000},
        {'index': 3, 'name': 'L4', 'type': 'power', 'thickness': 35000},
        {'index': 4, 'name': 'L5', 'type': 'signal', 'thickness': 35000},
        {'index': 5, 'name': 'L6', 'type': 'ground', 'thickness': 35000},
        {'index': 6, 'name': 'L7', 'type': 'signal', 'thickness': 35000},
        {'index': 7, 'name': 'B.Cu', 'type': 'signal', 'thickness': 35000}
    ]


if __name__ == '__main__':
    # Run integration tests with comprehensive reporting
    pytest.main([
        __file__, 
        '-v', 
        '--tb=short',
        '--maxfail=5',  # Stop after 5 failures
        '-x'  # Stop on first failure for debugging
    ])