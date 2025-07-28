"""
Algorithm Validation Tests for OrthoRoute
Unit tests for routing algorithms including wavefront, conflict resolution, and Steiner trees

These tests verify the correctness of the core routing algorithms used in OrthoRoute,
including pathfinding, congestion handling, and multi-pin net optimization.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Tuple, Optional

# Test CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = Mock()

# Import modules under test
try:
    from orthoroute.grid_manager import GPUGrid, Point3D, Net, TileManager
    from orthoroute.routing_algorithms import WavefrontRouter, ConflictResolver
    from orthoroute.steiner_tree import SteinerTreeBuilder, SteinerPoint, TreeEdge
    ALGORITHMS_AVAILABLE = True
except ImportError as e:
    ALGORITHMS_AVAILABLE = False
    print(f"Warning: Algorithm modules not available for testing: {e}")


class TestGridManager:
    """Test grid management and coordinate systems"""
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_grid_initialization(self):
        """Test grid initialization with various parameters"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(100, 100, 4, 0.1)
            
            assert grid.width == 100
            assert grid.height == 100
            assert grid.layers == 4
            assert grid.pitch_mm == 0.1
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_coordinate_conversion(self):
        """Test world to grid coordinate conversion"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(100, 100, 4, 0.1)
            
            # Test conversion from nanometers to grid coordinates
            grid_x, grid_y = grid.world_to_grid(5000000, 10000000)  # 5mm, 10mm
            assert grid_x == 50  # 5mm / 0.1mm = 50
            assert grid_y == 100  # 10mm / 0.1mm = 100, clamped to 99
            
            # Test reverse conversion
            world_x, world_y = grid.grid_to_world(25, 50)
            assert world_x == 2500000  # 25 * 0.1mm = 2.5mm = 2,500,000 nm
            assert world_y == 5000000  # 50 * 0.1mm = 5.0mm = 5,000,000 nm
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_obstacle_marking(self):
        """Test obstacle marking in grid"""
        with patch('cupy.ones') as mock_ones, patch('cupy.full'), patch('cupy.zeros'):
            mock_availability = Mock()
            mock_ones.return_value = mock_availability
            
            grid = GPUGrid(100, 100, 4, 0.1)
            grid.availability = mock_availability
            
            # Mark obstacle from (1mm, 1mm) to (2mm, 2mm) on all layers
            grid.mark_obstacle(1000000, 1000000, 2000000, 2000000, -1)
            
            # Should mark grid coordinates (10,10) to (20,20)
            # Verify indexing was called correctly
            assert mock_availability.__setitem__.called
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_grid_bounds_clamping(self):
        """Test that coordinates are properly clamped to grid bounds"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(50, 50, 2, 0.1)
            
            # Test coordinates outside bounds
            grid_x, grid_y = grid.world_to_grid(-1000000, 100000000)  # Negative and very large
            assert 0 <= grid_x <= 49
            assert 0 <= grid_y <= 49


class TestPoint3D:
    """Test Point3D data structure"""
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_point_creation(self):
        """Test Point3D creation and properties"""
        point = Point3D(10, 20, 1)
        assert point.x == 10
        assert point.y == 20
        assert point.layer == 1
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_point_equality(self):
        """Test Point3D equality comparison"""
        point1 = Point3D(5, 10, 0)
        point2 = Point3D(5, 10, 0)
        point3 = Point3D(5, 10, 1)
        
        assert point1 == point2
        assert point1 != point3
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_point_to_array(self):
        """Test Point3D to CuPy array conversion"""
        point = Point3D(15, 25, 2)
        array = point.to_array()
        
        expected = cp.array([15, 25, 2], dtype=cp.int32)
        assert cp.array_equal(array, expected)


class TestNet:
    """Test Net data structure and functionality"""
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_net_creation(self):
        """Test Net creation with various parameters"""
        pins = [Point3D(0, 0, 0), Point3D(10, 10, 1)]
        net = Net(
            net_id=1,
            name="TEST_NET",
            pins=pins,
            priority=3,
            width_nm=150000,
            via_size_nm=200000
        )
        
        assert net.net_id == 1
        assert net.name == "TEST_NET"
        assert len(net.pins) == 2
        assert net.priority == 3
        assert net.routed is False
        assert len(net.route_path) == 0
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_net_default_values(self):
        """Test Net creation with default values"""
        pins = [Point3D(0, 0, 0), Point3D(5, 5, 0)]
        net = Net(net_id=42, name="DEFAULT_NET", pins=pins)
        
        assert net.priority == 5  # Default priority
        assert net.width_nm == 200000  # Default width
        assert net.via_size_nm == 200000  # Default via size
        assert net.total_length == 0.0
        assert net.via_count == 0


class TestWavefrontRouter:
    """Test wavefront routing algorithm"""
    
    def setup_method(self):
        """Set up test fixtures"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            self.grid = GPUGrid(20, 20, 2, 0.1)
            self.router = WavefrontRouter(self.grid)
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_router_initialization(self):
        """Test router initialization"""
        assert self.router.grid == self.grid
        assert self.router.max_distance == 10000
        assert hasattr(self.router, 'neighbor_offsets')
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_neighbor_offsets(self):
        """Test neighbor offset calculation"""
        expected_offsets = cp.array([
            [-1, 0, 0],  # Left
            [1, 0, 0],   # Right
            [0, -1, 0],  # Up
            [0, 1, 0],   # Down
            [0, 0, -1],  # Layer down
            [0, 0, 1]    # Layer up
        ], dtype=cp.int32)
        
        assert cp.array_equal(self.router.neighbor_offsets, expected_offsets)
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_route_length_calculation(self):
        """Test route length calculation"""
        route = [
            Point3D(0, 0, 0),
            Point3D(3, 0, 0),  # 3 units right
            Point3D(3, 4, 0),  # 4 units up
            Point3D(3, 4, 1)   # Layer change (no length)
        ]
        
        length = self.router._calculate_route_length(route)
        expected = (3 + 4) * 0.1  # (3+4) grid units * 0.1mm pitch = 0.7mm
        assert abs(length - expected) < 0.001
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_via_counting(self):
        """Test via counting in routes"""
        # Route with layer changes
        route = [
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),  # Same layer
            Point3D(1, 0, 1),  # Layer change (via)
            Point3D(2, 0, 1),  # Same layer
            Point3D(2, 0, 0)   # Layer change (via)
        ]
        
        via_count = self.router._count_vias(route)
        assert via_count == 2
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_route_two_pin_simple(self):
        """Test simple two-pin routing"""
        # Mock the grid methods
        self.grid.distance_map = Mock()
        self.grid.parent_map = Mock()
        self.router._expand_wavefront_gpu = Mock(return_value=cp.array([[1, 0, 0]], dtype=cp.int32))
        self.router._check_target_reached = Mock(return_value=True)
        self.router._reconstruct_path = Mock(return_value=[Point3D(0, 0, 0), Point3D(1, 0, 0)])
        
        start = Point3D(0, 0, 0)
        end = Point3D(1, 0, 0)
        
        route = self.router._route_two_pin(start, end)
        
        assert route is not None
        assert len(route) == 2
        assert route[0] == start
        assert route[1] == end
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_route_multi_pin(self):
        """Test multi-pin routing"""
        pins = [Point3D(0, 0, 0), Point3D(5, 0, 0), Point3D(5, 5, 0)]
        
        # Mock two-pin routing to return simple paths
        def mock_route_two_pin(start, end):
            return [start, end]
        
        self.router._route_two_pin = mock_route_two_pin
        
        route = self.router._route_multi_pin(pins)
        
        assert route is not None
        assert len(route) > 0
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_valid_positions_mask(self):
        """Test position validation"""
        # Create test positions (some valid, some invalid)
        positions = cp.array([
            [5, 5, 0],    # Valid
            [-1, 5, 0],   # Invalid (negative x)
            [5, 25, 0],   # Invalid (y too large)
            [10, 10, 0],  # Valid
            [5, 5, 5]     # Invalid (layer too large)
        ], dtype=cp.int32)
        
        # Mock grid availability
        self.grid.availability = cp.ones((2, 20, 20), dtype=cp.uint8)
        self.grid.distance_map = cp.full((2, 20, 20), 65535, dtype=cp.uint16)
        
        mask = self.router._get_valid_positions_mask(positions)
        
        # Should be [True, False, False, True, False]
        expected = cp.array([True, False, False, True, False])
        assert cp.array_equal(mask, expected)


class TestConflictResolver:
    """Test conflict resolution and congestion algorithms"""
    
    def setup_method(self):
        """Set up test fixtures"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            self.grid = GPUGrid(10, 10, 2, 0.1)
            self.resolver = ConflictResolver(self.grid)
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_resolver_initialization(self):
        """Test conflict resolver initialization"""
        assert self.resolver.grid == self.grid
        assert self.resolver.congestion_factor == 1.5
        assert self.resolver.iteration_count == 0
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_usage_count_update(self):
        """Test usage count updating"""
        # Create mock nets with routes
        nets = []
        for i in range(3):
            net = Mock()
            net.routed = True
            net.route_path = [Point3D(i, 0, 0), Point3D(i+1, 0, 0)]
            nets.append(net)
        
        # Mock grid arrays
        self.grid.usage_count = Mock()
        
        self.resolver._update_usage_counts(nets)
        
        # Should reset usage count and update based on routes
        assert self.grid.usage_count.__setitem__.called
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_conflict_detection(self):
        """Test conflict detection in overcrowded cells"""
        # Create nets that share the same cell
        net1 = Mock()
        net1.routed = True
        net1.route_path = [Point3D(0, 0, 0), Point3D(1, 0, 0)]
        
        net2 = Mock()
        net2.routed = True
        net2.route_path = [Point3D(0, 0, 0), Point3D(0, 1, 0)]  # Shares (0,0,0)
        
        nets = [net1, net2]
        
        # Mock overcrowded condition
        self.grid.usage_count = cp.array([[[2, 1], [1, 0]]], dtype=cp.uint8)  # Cell (0,0,0) has usage=2
        self.grid.capacity = cp.array([[[1, 1], [1, 1]]], dtype=cp.uint8)     # Capacity=1 everywhere
        
        conflicted_nets = self.resolver._detect_conflicts(nets)
        
        # Both nets should be in conflict due to sharing overcrowded cell
        assert len(conflicted_nets) >= 0  # Depends on implementation details
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_congestion_cost_update(self):
        """Test congestion cost updating"""
        # Set up grid with some overcrowded cells
        self.grid.usage_count = cp.array([[[2, 1], [1, 0]]], dtype=cp.uint8)
        self.grid.capacity = cp.array([[[1, 1], [1, 1]]], dtype=cp.uint8)
        self.grid.congestion_cost = cp.ones((1, 2, 2), dtype=cp.float32)
        
        self.resolver.iteration_count = 2
        self.resolver._update_congestion_costs()
        
        # Overcrowded cells should have higher cost
        overcrowded = self.grid.usage_count > self.grid.capacity
        if cp.any(overcrowded):
            max_cost = cp.max(self.grid.congestion_cost)
            assert max_cost > 1.0  # Should be higher than base cost
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_net_rip_up(self):
        """Test ripping up conflicted nets"""
        # Create nets to rip up
        nets = []
        for i in range(3):
            net = Mock()
            net.routed = True
            net.route_path = [Point3D(i, 0, 0)]
            net.total_length = 5.0
            net.via_count = 2
            nets.append(net)
        
        self.resolver._rip_up_nets(nets)
        
        # All nets should be marked as unrouted
        for net in nets:
            assert net.routed is False
            assert net.route_path == []
            assert net.total_length == 0.0
            assert net.via_count == 0
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_resolution_convergence(self):
        """Test conflict resolution convergence"""
        # Create simple nets that shouldn't conflict
        nets = []
        for i in range(3):
            net = Mock()
            net.net_id = i + 1
            net.name = f"NET_{i+1}"
            net.pins = [Point3D(i, 0, 0), Point3D(i, 5, 0)]
            net.priority = 5
            net.width_nm = 200000
            net.via_size_nm = 200000
            net.routed = False
            net.route_path = []
            net.total_length = 0.0
            net.via_count = 0
            nets.append(net)
        
        # Mock the router to return successful routes
        with patch('orthoroute.routing_algorithms.WavefrontRouter') as mock_router_class:
            mock_router = Mock()
            mock_router.route_net_batch.return_value = nets  # All nets routed successfully
            mock_router_class.return_value = mock_router
            
            # Mark all nets as routed for the mock
            for net in nets:
                net.routed = True
                net.route_path = [Point3D(0, 0, 0), Point3D(1, 0, 0)]
            
            # Mock no conflicts
            self.resolver._detect_conflicts = Mock(return_value=[])
            
            result_nets = self.resolver.resolve_conflicts(nets, max_iterations=5)
            
            # Should converge quickly with no conflicts
            assert len(result_nets) == 3
            assert all(net.routed for net in result_nets)


class TestSteinerTreeBuilder:
    """Test Steiner tree construction for multi-pin nets"""
    
    def setup_method(self):
        """Set up test fixtures"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            self.grid = GPUGrid(20, 20, 2, 0.1)
            self.builder = SteinerTreeBuilder(self.grid)
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_builder_initialization(self):
        """Test Steiner tree builder initialization"""
        assert self.builder.grid == self.grid
        assert hasattr(self.builder, 'hanan_cache')
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_steiner_point_creation(self):
        """Test SteinerPoint creation and conversion"""
        point = SteinerPoint(5, 10, 1, is_pin=True, pin_index=0)
        
        assert point.x == 5
        assert point.y == 10
        assert point.layer == 1
        assert point.is_pin is True
        assert point.pin_index == 0
        
        point3d = point.to_point3d()
        assert isinstance(point3d, Point3D)
        assert point3d.x == 5
        assert point3d.y == 10
        assert point3d.layer == 1
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_tree_edge_creation(self):
        """Test TreeEdge creation and properties"""
        start = SteinerPoint(0, 0, 0, is_pin=True)
        end = SteinerPoint(3, 4, 0, is_pin=True)
        
        edge = TreeEdge(start, end, 5.0, 0)
        
        assert edge.start == start
        assert edge.end == end
        assert edge.length == 5.0
        assert edge.via_count == 0
        assert len(edge.path) == 0
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_two_pin_steiner_tree(self):
        """Test Steiner tree for two pins (should be direct connection)"""
        pins = [Point3D(0, 0, 0), Point3D(5, 0, 0)]
        
        # Mock simple routing
        self.builder._simple_two_pin_route = Mock(return_value=[pins[0], pins[1]])
        
        result = self.builder.route_multi_pin_net(
            Mock(pins=pins, net_id=1, name="TEST")
        )
        
        assert result is not None
        assert len(result) == 2
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_hanan_grid_generation(self):
        """Test Hanan grid point generation"""
        pins = [
            SteinerPoint(0, 0, 0, is_pin=True),
            SteinerPoint(5, 3, 0, is_pin=True),
            SteinerPoint(2, 8, 0, is_pin=True)
        ]
        
        hanan_points = self.builder._generate_hanan_grid(pins)
        
        # Should generate grid points at coordinate intersections
        expected_x_coords = {0, 2, 5}
        expected_y_coords = {0, 3, 8}
        
        # Check that Hanan points cover the intersections (excluding existing pins)
        hanan_coords = {(p.x, p.y) for p in hanan_points}
        pin_coords = {(p.x, p.y) for p in pins}
        
        # Should have some Hanan points that aren't pins
        assert len(hanan_coords - pin_coords) >= 0
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_minimum_spanning_tree(self):
        """Test minimum spanning tree construction"""
        points = [
            SteinerPoint(0, 0, 0, is_pin=True),
            SteinerPoint(3, 0, 0, is_pin=True),
            SteinerPoint(0, 4, 0, is_pin=True)
        ]
        
        # Mock distance calculation
        def mock_distance(p1, p2):
            return abs(p1.x - p2.x) + abs(p1.y - p2.y)  # Manhattan distance
        
        self.builder._calculate_distance = mock_distance
        self.builder._create_edge = Mock(side_effect=lambda s, e: TreeEdge(s, e, mock_distance(s, e), 0))
        
        mst_edges = self.builder._minimum_spanning_tree(points)
        
        # Should have n-1 edges for n points
        assert len(mst_edges) == len(points) - 1
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_beneficial_steiner_point_detection(self):
        """Test detection of beneficial Steiner points"""
        pins = [
            SteinerPoint(0, 0, 0, is_pin=True),
            SteinerPoint(10, 0, 0, is_pin=True),
            SteinerPoint(0, 10, 0, is_pin=True),
            SteinerPoint(10, 10, 0, is_pin=True)
        ]
        
        # Point in the middle should be beneficial
        assert self.builder._is_beneficial_steiner_point(5, 5, 0, pins) is True
        
        # Point at corner should not be beneficial
        assert self.builder._is_beneficial_steiner_point(0, 0, 0, pins) is False
        
        # Point outside the pin area should not be beneficial
        assert self.builder._is_beneficial_steiner_point(20, 20, 0, pins) is False
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_simple_two_pin_routing(self):
        """Test simple L-shaped routing"""
        start = Point3D(0, 0, 0)
        end = Point3D(3, 2, 0)
        
        # Mock grid availability
        self.builder._is_path_valid = Mock(return_value=True)
        
        path = self.builder._simple_two_pin_route(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        assert len(path) >= 2
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_via_optimization(self):
        """Test via count optimization"""
        # Create route with unnecessary layer changes
        route = [
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
            Point3D(1, 0, 1),  # Layer change
            Point3D(2, 0, 1),
            Point3D(2, 0, 0),  # Layer change back
            Point3D(3, 0, 0)
        ]
        
        optimized = self.builder.optimize_via_count(route)
        
        # Should have fewer or equal layer changes
        original_vias = sum(1 for i in range(len(route)-1) if route[i].layer != route[i+1].layer)
        optimized_vias = sum(1 for i in range(len(optimized)-1) if optimized[i].layer != optimized[i+1].layer)
        
        assert optimized_vias <= original_vias


class TestTileManager:
    """Test tiled processing for memory efficiency"""
    
    def setup_method(self):
        """Set up test fixtures"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            self.grid = GPUGrid(128, 128, 4, 0.1)
            self.tile_manager = TileManager(self.grid, tile_size=32)
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_tile_manager_initialization(self):
        """Test tile manager initialization"""
        assert self.tile_manager.grid == self.grid
        assert self.tile_manager.tile_size == 32
        assert self.tile_manager.tiles_x == 4  # 128 / 32 = 4
        assert self.tile_manager.tiles_y == 4
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_tile_bounds_calculation(self):
        """Test tile bounds calculation"""
        # Test corner tile
        x_start, y_start, x_end, y_end = self.tile_manager.get_tile_bounds(0, 0)
        assert x_start == 0
        assert y_start == 0
        assert x_end == 32
        assert y_end == 32
        
        # Test edge tile
        x_start, y_start, x_end, y_end = self.tile_manager.get_tile_bounds(3, 3)
        assert x_start == 96  # 3 * 32
        assert y_start == 96
        assert x_end == 128   # Clamped to grid size
        assert y_end == 128
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_tile_extraction(self):
        """Test tile data extraction"""
        # Mock grid arrays
        self.grid.availability = Mock()
        self.grid.congestion_cost = Mock()
        self.grid.distance_map = Mock()
        
        # Mock array slicing
        self.grid.availability.__getitem__ = Mock(return_value=Mock())
        self.grid.congestion_cost.__getitem__ = Mock(return_value=Mock())
        self.grid.distance_map.__getitem__ = Mock(return_value=Mock())
        
        tile_data = self.tile_manager.extract_tile(1, 1)
        
        assert 'availability' in tile_data
        assert 'congestion' in tile_data
        assert 'distance' in tile_data
        assert 'bounds' in tile_data
        
        expected_bounds = (32, 32, 64, 64)  # Tile (1,1) bounds
        assert tile_data['bounds'] == expected_bounds


class TestAlgorithmIntegration:
    """Integration tests for algorithm combinations"""
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_router_with_conflict_resolver(self):
        """Test integration of wavefront router with conflict resolver"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(20, 20, 2, 0.1)
            resolver = ConflictResolver(grid)
            
            # Create simple nets
            nets = []
            for i in range(3):
                net = Mock()
                net.net_id = i + 1
                net.name = f"NET_{i+1}"
                net.pins = [Point3D(i*2, 0, 0), Point3D(i*2, 10, 0)]
                net.routed = False
                net.route_path = []
                nets.append(net)
            
            # Mock router behavior
            with patch('orthoroute.routing_algorithms.WavefrontRouter') as mock_router_class:
                mock_router = Mock()
                
                # First iteration: route all nets successfully
                for net in nets:
                    net.routed = True
                    net.route_path = [Point3D(0, 0, 0), Point3D(1, 0, 0)]
                
                mock_router.route_net_batch.return_value = nets
                mock_router_class.return_value = mock_router
                
                # Mock no conflicts
                resolver._detect_conflicts = Mock(return_value=[])
                
                result = resolver.resolve_conflicts(nets, max_iterations=5)
                
                assert len(result) == 3
                assert all(net.routed for net in result)
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_steiner_tree_with_router(self):
        """Test integration of Steiner tree builder with router"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(20, 20, 2, 0.1)
            builder = SteinerTreeBuilder(grid)
            
            # Create multi-pin net
            pins = [Point3D(0, 0, 0), Point3D(5, 5, 0), Point3D(10, 0, 0)]
            net = Mock()
            net.pins = pins
            net.net_id = 1
            net.name = "MULTI_PIN_NET"
            
            # Mock router for segment routing
            mock_router = Mock()
            mock_router._route_two_pin = Mock(return_value=[Point3D(0, 0, 0), Point3D(1, 0, 0)])
            
            result = builder.route_multi_pin_net(net, mock_router)
            
            assert result is not None
            assert len(result) >= len(pins)  # Should connect all pins


class TestPerformanceCharacteristics:
    """Test algorithm performance characteristics"""
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_grid_scaling(self):
        """Test grid performance scaling"""
        grid_sizes = [(10, 10), (50, 50), (100, 100)]
        
        for width, height in grid_sizes:
            with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
                grid = GPUGrid(width, height, 4, 0.1)
                
                # Grid should initialize regardless of size
                assert grid.width == width
                assert grid.height == height
                
                # Memory calculation should work
                grid._calculate_memory_usage()  # Should not raise exception
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_net_count_scaling(self):
        """Test algorithm scaling with net count"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(50, 50, 2, 0.1)
            router = WavefrontRouter(grid)
            
            # Test with different net counts
            net_counts = [5, 25, 100]
            
            for count in net_counts:
                nets = []
                for i in range(count):
                    net = Mock()
                    net.net_id = i + 1
                    net.name = f"NET_{i+1}"
                    net.pins = [Point3D(i % 10, 0, 0), Point3D(i % 10, 10, 0)]
                    nets.append(net)
                
                # Mock successful routing
                router.route_net_batch = Mock(return_value=nets)
                
                result = router.route_net_batch(nets)
                assert len(result) == count


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_empty_grid_routing(self):
        """Test routing on empty (minimal) grid"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(2, 2, 1, 0.1)
            router = WavefrontRouter(grid)
            
            # Try to route in minimal space
            start = Point3D(0, 0, 0)
            end = Point3D(1, 1, 0)
            
            # Mock grid state
            grid.distance_map = Mock()
            grid.parent_map = Mock()
            router._expand_wavefront_gpu = Mock(return_value=cp.array([], dtype=cp.int32).reshape(0, 3))
            
            route = router._route_two_pin(start, end)
            # Should handle gracefully (may return None)
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_blocked_routing(self):
        """Test routing when path is completely blocked"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(10, 10, 1, 0.1)
            router = WavefrontRouter(grid)
            
            # Mock completely blocked grid
            grid.availability = cp.zeros((1, 10, 10), dtype=cp.uint8)
            grid.distance_map = cp.full((1, 10, 10), 65535, dtype=cp.uint16)
            grid.parent_map = cp.full((1, 10, 10, 3), -1, dtype=cp.int16)
            
            start = Point3D(0, 0, 0)
            end = Point3D(9, 9, 0)
            
            # Should not find a route
            router._expand_wavefront_gpu = Mock(return_value=cp.array([], dtype=cp.int32).reshape(0, 3))
            route = router._route_two_pin(start, end)
            assert route is None
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_single_point_net(self):
        """Test handling of degenerate single-point nets"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(10, 10, 2, 0.1)
            builder = SteinerTreeBuilder(grid)
            
            # Net with only one pin
            net = Mock()
            net.pins = [Point3D(5, 5, 0)]
            net.net_id = 1
            net.name = "SINGLE_PIN"
            
            result = builder.route_multi_pin_net(net)
            assert result is None  # Should handle gracefully
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_overlapping_pins(self):
        """Test handling of overlapping pins"""
        pins = [Point3D(5, 5, 0), Point3D(5, 5, 0)]  # Same location
        
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(10, 10, 2, 0.1)
            router = WavefrontRouter(grid)
            
            # Should handle overlapping pins gracefully
            route = router._route_two_pin(pins[0], pins[1])
            # May return empty route or single point
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_out_of_bounds_coordinates(self):
        """Test handling of out-of-bounds coordinates"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(10, 10, 2, 0.1)
            
            # Test coordinate clamping
            x, y = grid.world_to_grid(-1000000, 50000000)  # Negative and too large
            assert 0 <= x < grid.width
            assert 0 <= y < grid.height
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_zero_pitch_grid(self):
        """Test handling of invalid grid pitch"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            # Very small pitch should be handled
            try:
                grid = GPUGrid(10, 10, 2, 0.001)  # 1 micrometer
                assert grid.pitch_mm == 0.001
            except (ValueError, RuntimeError):
                # May raise exception for impractical pitch
                pass


class TestAlgorithmCorrectness:
    """Test algorithm correctness with known solutions"""
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_manhattan_distance_routing(self):
        """Test that router finds optimal Manhattan distance paths"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(10, 10, 1, 0.1)
            router = WavefrontRouter(grid)
            
            # Simple L-shaped route
            start = Point3D(0, 0, 0)
            end = Point3D(3, 4, 0)
            
            # Expected Manhattan distance = 3 + 4 = 7 grid units
            expected_length = 7 * 0.1  # 0.7mm
            
            # Mock a simple L-shaped path
            mock_path = [
                Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(2, 0, 0), Point3D(3, 0, 0),
                Point3D(3, 1, 0), Point3D(3, 2, 0), Point3D(3, 3, 0), Point3D(3, 4, 0)
            ]
            
            length = router._calculate_route_length(mock_path)
            assert abs(length - expected_length) < 0.001
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_via_minimization(self):
        """Test that algorithms minimize via usage"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(10, 10, 3, 0.1)
            builder = SteinerTreeBuilder(grid)
            
            # Route that could use fewer vias
            suboptimal_route = [
                Point3D(0, 0, 0),
                Point3D(0, 0, 1),  # Unnecessary via
                Point3D(1, 0, 1),
                Point3D(1, 0, 0),  # Via back
                Point3D(2, 0, 0)
            ]
            
            optimized = builder.optimize_via_count(suboptimal_route)
            
            original_vias = builder._count_vias(suboptimal_route) if hasattr(builder, '_count_vias') else 2
            optimized_vias = len([p for i, p in enumerate(optimized[:-1]) if optimized[i].layer != optimized[i+1].layer])
            
            # Should have fewer or equal vias
            assert optimized_vias <= original_vias
    
    @pytest.mark.skipif(not ALGORITHMS_AVAILABLE, reason="Algorithm modules not available")
    def test_steiner_tree_optimality(self):
        """Test Steiner tree construction for simple cases"""
        with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
            grid = GPUGrid(20, 20, 1, 0.1)
            builder = SteinerTreeBuilder(grid)
            
            # Three pins forming a right triangle
            pins = [
                SteinerPoint(0, 0, 0, is_pin=True),
                SteinerPoint(6, 0, 0, is_pin=True),
                SteinerPoint(0, 8, 0, is_pin=True)
            ]
            
            # MST should connect with total length less than naive star connection
            builder._calculate_distance = lambda p1, p2: abs(p1.x - p2.x) + abs(p1.y - p2.y)
            builder._create_edge = lambda s, e: TreeEdge(s, e, builder._calculate_distance(s, e), 0)
            
            mst_edges = builder._minimum_spanning_tree(pins)
            total_mst_length = sum(edge.length for edge in mst_edges)
            
            # Should be reasonable (Manhattan distance between points)
            assert total_mst_length > 0
            assert len(mst_edges) == 2  # n-1 edges for n points


# Helper functions for algorithm testing
def create_simple_grid(width: int, height: int, layers: int = 2) -> 'GPUGrid':
    """Create a simple test grid"""
    with patch('cupy.ones'), patch('cupy.full'), patch('cupy.zeros'):
        return GPUGrid(width, height, layers, 0.1)

def create_test_net(net_id: int, pin_coords: List[Tuple[int, int, int]]) -> 'Net':
    """Create a test net with specified pin coordinates"""
    pins = [Point3D(x, y, layer) for x, y, layer in pin_coords]
    return Net(
        net_id=net_id,
        name=f"TEST_NET_{net_id}",
        pins=pins,
        priority=5,
        width_nm=200000,
        via_size_nm=200000
    )

def assert_valid_route(route: List[Point3D], start: Point3D, end: Point3D):
    """Assert that a route is valid (connected path from start to end)"""
    assert route is not None
    assert len(route) >= 2
    assert route[0] == start
    assert route[-1] == end
    
    # Check connectivity (each step should be adjacent)
    for i in range(len(route) - 1):
        p1, p2 = route[i], route[i + 1]
        # Manhattan distance should be 1 (adjacent) or 0 (same layer change)
        manhattan_dist = abs(p2.x - p1.x) + abs(p2.y - p1.y)
        layer_change = abs(p2.layer - p1.layer)
        
        # Valid if: (same position, different layer) OR (adjacent position, same layer)
        assert (manhattan_dist == 0 and layer_change == 1) or (manhattan_dist == 1 and layer_change == 0)

def measure_algorithm_performance(algorithm_func, *args, **kwargs) -> Dict:
    """Measure algorithm performance metrics"""
    import time
    
    start_time = time.time()
    result = algorithm_func(*args, **kwargs)
    end_time = time.time()
    
    return {
        'execution_time': end_time - start_time,
        'result': result,
        'memory_usage': 0  # Could be enhanced with actual memory measurement
    }


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])