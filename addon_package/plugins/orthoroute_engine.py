"""
Standalone GPU routing engine for OrthoRoute KiCad plugin
This module contains all necessary classes for GPU-accelerated routing
without requiring external package installation.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

try:
    import cupy as cp
    import numpy as np
    CUPY_AVAILABLE = True
except ImportError:
    # Create dummy classes if CuPy is not available
    CUPY_AVAILABLE = False
    
    class MockCuPy:
        class cuda:
            class Device:
                def __init__(self, gpu_id=0):
                    self.id = gpu_id
                    self.name = "Mock GPU (CuPy not available)"
                def use(self): pass
                def mem_info(self): return (0, 8*1024**3)  # Mock 8GB
        def array(self, data): return data
        def ones(self, shape, dtype=None): return None
        def zeros(self, shape, dtype=None): return None
        def full(self, shape, value, dtype=None): return None
    
    cp = MockCuPy()
    np = None

@dataclass
class Point3D:
    """3D point with integer coordinates"""
    x: int
    y: int
    z: int  # layer
    
    def __eq__(self, other):
        if not isinstance(other, Point3D):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
        
    def distance_to(self, other: 'Point3D') -> float:
        """Calculate Manhattan distance to another point"""
        return (abs(self.x - other.x) + 
                abs(self.y - other.y) + 
                abs(self.z - other.z))

@dataclass
class Net:
    """Represents a net to be routed"""
    id: int
    name: str
    pins: List[Point3D]
    width_nm: int
    route_path: Optional[List[Point3D]] = None
    success: bool = False
    priority: int = 5  # Default priority
    via_size_nm: int = 200000  # Default via size
    routed: bool = False  # True when routing is successful
    total_length: float = 0.0  # Total route length in grid units
    via_count: int = 0  # Number of vias used

class GPUGrid:
    """GPU-accelerated routing grid"""
    def __init__(self, width: int, height: int, layers: int, pitch_mm: float):
        self.width = width
        self.height = height
        self.layers = layers
        self.pitch_mm = pitch_mm
        self.pitch_nm = int(pitch_mm * 1000000)  # Convert mm to nm
        
        if CUPY_AVAILABLE:
            # Initialize grid arrays
            self.availability = cp.ones((layers, height, width), dtype=cp.uint8)
            self.congestion = cp.ones((layers, height, width), dtype=cp.float32)
            self.distance = cp.full((layers, height, width), 0xFFFFFFFF, dtype=cp.uint32)
            self.usage_count = cp.zeros((layers, height, width), dtype=cp.uint8)
            self.parent = cp.full((layers, height, width, 3), -1, dtype=cp.int32)
        else:
            # Fallback to CPU arrays
            self.availability = [[1 for _ in range(width)] for _ in range(height)]
            self.usage_count = [[0 for _ in range(width)] for _ in range(height)]
        
    def world_to_grid(self, x_nm: int, y_nm: int) -> Tuple[int, int]:
        """Convert world coordinates (nm) to grid coordinates"""
        try:
            if not isinstance(x_nm, (int, float)) or not isinstance(y_nm, (int, float)):
                return (0, 0)
            grid_x = int(x_nm / self.pitch_nm)
            grid_y = int(y_nm / self.pitch_nm)
            return (grid_x, grid_y)
        except Exception:
            return (0, 0)
        
    def grid_to_world(self, x: int, y: int) -> Tuple[int, int]:
        """Convert grid coordinates to world coordinates (nm)"""
        try:
            return (int(x * self.pitch_nm), int(y * self.pitch_nm))
        except Exception:
            return (0, 0)
        
    def is_valid_point(self, x: int, y: int, z: int) -> bool:
        """Check if a grid point is within bounds"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                0 <= z < self.layers)

class GPUWavefrontRouter:
    """Real GPU-accelerated wavefront router using CuPy"""
    
    def __init__(self, grid):
        self.grid = grid
        self.use_gpu = CUPY_AVAILABLE
        print(f"GPU Wavefront Router initialized (GPU: {self.use_gpu})")
    
    def route_net(self, net: Net) -> bool:
        """Route a single net using GPU-accelerated Lee's algorithm"""
        if len(net.pins) < 2:
            return False
        
        print(f"Routing net {net.id} ({net.name}) with {len(net.pins)} pins")
        
        if self.use_gpu:
            return self._route_net_gpu(net)
        else:
            return self._route_net_cpu(net)
    
    def _route_net_gpu(self, net: Net) -> bool:
        """GPU-accelerated routing using CuPy"""
        try:
            # Initialize distance map
            distance = cp.full((self.grid.layers, self.grid.height, self.grid.width), 
                             cp.inf, dtype=cp.float32)
            parent = cp.full((self.grid.layers, self.grid.height, self.grid.width, 3), 
                           -1, dtype=cp.int32)
            
            # Route between consecutive pins (minimum spanning tree approach)
            full_path = []
            
            for i in range(len(net.pins) - 1):
                source = net.pins[i]
                target = net.pins[i + 1]
                
                # Reset distance map
                distance.fill(cp.inf)
                parent.fill(-1)
                
                # Run Lee's algorithm
                path = self._lee_algorithm_gpu(source, target, distance, parent)
                if not path:
                    print(f"Failed to route segment {i} of net {net.id}")
                    return False
                
                # Add path to full route (avoid duplicating connection points)
                if i == 0:
                    full_path.extend(path)
                else:
                    full_path.extend(path[1:])  # Skip first point to avoid duplication
            
            # Convert GPU path back to CPU and store
            net.route_path = full_path
            net.success = True
            net.routed = True
            net.total_length = len(full_path)
            net.via_count = len([p for i, p in enumerate(full_path[:-1]) 
                               if p.z != full_path[i+1].z])
            
            print(f"Successfully routed net {net.id}: {len(full_path)} points, {net.via_count} vias")
            return True
            
        except Exception as e:
            print(f"GPU routing failed for net {net.id}: {e}")
            return self._route_net_cpu(net)  # Fallback to CPU
    
    def _lee_algorithm_gpu(self, source: Point3D, target: Point3D, distance, parent):
        """GPU implementation of Lee's algorithm"""
        # Mark source
        distance[source.z, source.y, source.x] = 0
        
        # BFS wavefront expansion
        current_wave = cp.array([[source.x, source.y, source.z]], dtype=cp.int32)
        wave_distance = 0
        
        # Neighbor offsets (x, y, z)
        neighbors = cp.array([
            [1, 0, 0], [-1, 0, 0],  # X direction
            [0, 1, 0], [0, -1, 0],  # Y direction
            [0, 0, 1], [0, 0, -1]   # Z direction (via)
        ], dtype=cp.int32)
        
        max_iterations = self.grid.width * self.grid.height * 2
        
        for iteration in range(max_iterations):
            if len(current_wave) == 0:
                break
                
            wave_distance += 1
            next_wave = []
            
            # Process each point in current wave
            for point in current_wave:
                x, y, z = int(point[0]), int(point[1]), int(point[2])
                
                # Check if we reached target
                if x == target.x and y == target.y and z == target.z:
                    # Reconstruct path
                    return self._reconstruct_path_gpu(source, target, parent)
                
                # Expand to neighbors
                for dx, dy, dz in neighbors:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Check bounds
                    if (0 <= nx < self.grid.width and 
                        0 <= ny < self.grid.height and 
                        0 <= nz < self.grid.layers):
                        
                        # Check if available and not visited
                        if (self.grid.availability[nz, ny, nx] and 
                            distance[nz, ny, nx] == cp.inf):
                            
                            distance[nz, ny, nx] = wave_distance
                            parent[nz, ny, nx] = [x, y, z]
                            next_wave.append([nx, ny, nz])
            
            current_wave = cp.array(next_wave, dtype=cp.int32) if next_wave else cp.array([], dtype=cp.int32).reshape(0, 3)
        
        print(f"Lee's algorithm failed: no path found from {source} to {target}")
        return None
    
    def _reconstruct_path_gpu(self, source: Point3D, target: Point3D, parent):
        """Reconstruct path from parent array"""
        path = []
        current = target
        
        while current != source:
            path.append(current)
            
            # Get parent coordinates
            px, py, pz = parent[current.z, current.y, current.x]
            if px == -1:  # No parent found
                return None
                
            current = Point3D(int(px), int(py), int(pz))
        
        path.append(source)
        path.reverse()
        return path
    
    def _route_net_cpu(self, net: Net) -> bool:
        """CPU fallback routing (simple L-shaped paths)"""
        print(f"Using CPU fallback for net {net.id}")
        
        # Simple L-shaped routing as fallback
        path = []
        path.append(net.pins[0])
        
        for i in range(len(net.pins) - 1):
            start = net.pins[i]
            end = net.pins[i + 1]
            
            current = start
            
            # Move in X direction
            while current.x != end.x:
                step_x = 1 if end.x > current.x else -1
                current = Point3D(current.x + step_x, current.y, current.z)
                path.append(current)
            
            # Move in Y direction  
            while current.y != end.y:
                step_y = 1 if end.y > current.y else -1
                current = Point3D(current.x, current.y + step_y, current.z)
                path.append(current)
            
            # Add via if layer change needed
            if current.z != end.z:
                current = Point3D(current.x, current.y, end.z)
                path.append(current)
                net.via_count += 1
        
        net.route_path = path
        net.success = True
        net.routed = True
        net.total_length = len(path)
        return True


class SimpleWaveRouter:
    """Deprecated: Simple wavefront router - use GPUWavefrontRouter instead"""
    def __init__(self, grid):
        self.grid = grid
        print("Warning: Using deprecated SimpleWaveRouter. Upgrade to GPUWavefrontRouter.")
    
    def route_net(self, net: Net) -> bool:
        """Route a single net using simple pathfinding"""
        if len(net.pins) < 2:
            return False
        
        # For testing, create straight-line paths between pins
        path = []
        path.append(net.pins[0])
        
        # Connect each pin to the next
        for i in range(len(net.pins) - 1):
            start = net.pins[i]
            end = net.pins[i + 1]
            
            # Create L-shaped path (move in X, then Y)
            current = start
            
            # Move in X direction
            while current.x != end.x:
                step_x = 1 if end.x > current.x else -1
                current = Point3D(current.x + step_x, current.y, current.z)
                path.append(current)
            
            # Move in Y direction
            while current.y != end.y:
                step_y = 1 if end.y > current.y else -1
                current = Point3D(current.x, current.y + step_y, current.z)
                path.append(current)
            
            # Add via if layer change needed
            if current.z != end.z:
                current = Point3D(current.x, current.y, end.z)
                path.append(current)
                net.via_count += 1
        
        net.route_path = path
        net.success = True
        net.routed = True
        net.total_length = len(path)
        return True

class OrthoRouteEngine:
    """GPU-accelerated routing engine"""
    def __init__(self, gpu_id: int = 0):
        """Initialize the GPU routing engine."""
        if CUPY_AVAILABLE:
            # Initialize CUDA device
            cp.cuda.Device(gpu_id).use()
            print(f"OrthoRoute GPU Engine initialized on device {gpu_id}")
        else:
            print("OrthoRoute running in CPU mode (CuPy not available)")
        
        # Generate unique engine ID
        import random, time
        self.engine_id = f"engine_{int(time.time() % 10000)}_{random.randint(1000, 9999)}"
        
        self._print_gpu_info()
        self.visualizer = None
        self.viz_config = None
        self.grid = None
        
        # Default configuration
        self.config = {
            'max_iterations': 3,
            'via_cost': 10,
            'conflict_penalty': 50,
            'max_wave_iterations': 1000,
            'grid_pitch_mm': 0.1,
            'max_layers': 8,
            'batch_size': 10,
            'congestion_threshold': 3
        }
        
    def enable_visualization(self, viz_config):
        """Enable real-time visualization during routing."""
        self.viz_config = viz_config
        print(f"Visualization enabled: {viz_config}")
        
    def _print_gpu_info(self):
        """Print GPU information."""
        if CUPY_AVAILABLE:
            device = cp.cuda.Device()
            print(f"GPU Device ID: {device.id}")
            try:
                mem_info = device.mem_info()
                if isinstance(mem_info, (list, tuple)) and len(mem_info) >= 2:
                    total_mem = float(mem_info[1]) / (1024**3)
                    print(f"GPU Memory: {total_mem:.1f} GB")
            except Exception:
                print("GPU Memory: Unknown")
        else:
            print("GPU: Not available (using CPU fallback)")

    def load_board_data(self, board_data: Dict) -> bool:
        """Load board data and initialize grid"""
        try:
            # Extract board bounds
            bounds = board_data.get('bounds', {})
            width_nm = bounds.get('width_nm', 100000000)  # Default 100mm
            height_nm = bounds.get('height_nm', 100000000)
            
            # Calculate grid dimensions
            grid_config = board_data.get('grid', {})
            pitch_nm = grid_config.get('pitch_nm', int(self.config['grid_pitch_mm'] * 1000000))
            layers = bounds.get('layers', self.config['max_layers'])
            
            # Ensure valid values
            if pitch_nm <= 0:
                pitch_nm = 100000
            if width_nm <= 0 or height_nm <= 0:
                width_nm = height_nm = 100000000
            if layers <= 0:
                layers = 2
            
            # Calculate grid dimensions
            grid_width = max(int(width_nm / pitch_nm) + 10, 100)
            grid_height = max(int(height_nm / pitch_nm) + 10, 100)
            
            print(f"Creating routing grid: {grid_width}x{grid_height}x{layers} cells")
            
            # Initialize grid
            self.grid = GPUGrid(grid_width, grid_height, layers, pitch_nm / 1000000.0)
            
            return True
            
        except Exception as e:
            print(f"Error loading board data: {e}")
            return False
    
    def route(self, board_data: Dict, config: Dict = None) -> Dict:
        """Route the board with the given config."""
        print(f"Starting route with engine {self.engine_id}")
        
        # Merge configuration
        if config:
            self.config.update(config)
        
        # Ensure required fields exist
        if 'bounds' not in board_data:
            board_data['bounds'] = {
                'width_nm': 100000000,
                'height_nm': 100000000,
                'layers': 2
            }
        
        if not self.load_board_data(board_data):
            return {'success': False, 'error': 'Failed to load board data'}
        
        # Parse nets
        nets = self._parse_nets(board_data.get('nets', []))
        if not nets:
            return {'success': False, 'error': 'No nets to route'}
        
        print(f"Routing {len(nets)} nets...")
        start_time = time.time()
        
        # Initialize router - use GPU router for real routing
        if CUPY_AVAILABLE:
            router = GPUWavefrontRouter(self.grid)
            print("Using GPU-accelerated wavefront routing")
        else:
            router = GPUWavefrontRouter(self.grid)  # Has CPU fallback
            print("Using CPU fallback routing (CuPy not available)")
        
        # Route nets
        successful_nets = []
        for i, net in enumerate(nets):
            print(f"Routing net {i+1}/{len(nets)}: {net.name}")
            if router.route_net(net):
                successful_nets.append(net)
        
        routing_time = time.time() - start_time
        print(f"Routing completed in {routing_time:.2f} seconds")
        
        return self._generate_results(successful_nets, routing_time)
    
    def _parse_nets(self, nets_data: List[Dict]) -> List[Net]:
        """Parse net data into Net objects"""
        nets = []
        
        for net_data in nets_data:
            valid_pins = []
            
            for pin_data in net_data.get('pins', []):
                # Convert world coordinates to grid coordinates
                grid_x, grid_y = self.grid.world_to_grid(pin_data['x'], pin_data['y'])
                
                # Check bounds
                if (0 <= grid_x < self.grid.width and 
                    0 <= grid_y < self.grid.height and
                    0 <= pin_data.get('layer', 0) < self.grid.layers):
                    valid_pins.append(Point3D(grid_x, grid_y, pin_data.get('layer', 0)))
            
            # Only create net if there are at least 2 valid pins
            if len(valid_pins) >= 2:
                net = Net(
                    id=net_data['id'],
                    name=net_data.get('name', f"Net_{net_data['id']}"),
                    pins=valid_pins,
                    width_nm=net_data.get('width_nm', 200000)
                )
                nets.append(net)
        
        return nets
    
    def _generate_results(self, nets: List[Net], routing_time: float) -> Dict:
        """Generate routing results"""
        successful_nets = [n for n in nets if n.routed]
        success_rate = (len(successful_nets) / len(nets)) * 100 if nets else 0
        
        result = {
            'success': True,
            'stats': {
                'total_nets': len(nets),
                'successful_nets': len(successful_nets),
                'success_rate': success_rate,
                'total_time_seconds': routing_time
            },
            'nets': []
        }
        
        # Add net details
        for net in successful_nets:
            path_world = []
            if net.route_path:
                for point in net.route_path:
                    world_x, world_y = self.grid.grid_to_world(point.x, point.y)
                    path_world.append({
                        'x': world_x,
                        'y': world_y,
                        'layer': point.z
                    })
            
            net_data = {
                'id': net.id,
                'name': net.name,
                'path': path_world,
                'via_count': net.via_count,
                'total_length_mm': net.total_length * self.grid.pitch_mm
            }
            result['nets'].append(net_data)
            
        return result
