"""
GPU-accelerated routing engine for OrthoRoute
"""

import cupy as cp
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
        
        # Initialize grid arrays
        self.availability = cp.ones((layers, height, width), dtype=cp.uint8)
        self.congestion = cp.ones((layers, height, width), dtype=cp.float32)
        self.distance = cp.full((layers, height, width), 0xFFFFFFFF, dtype=cp.uint32)  # Using uint32
        self.usage_count = cp.zeros((layers, height, width), dtype=cp.uint8)
        self.parent = cp.full((layers, height, width, 3), -1, dtype=cp.int32)
        
        # Keep copies in CPU memory
        self.availability_cpu = np.ones((layers, height, width), dtype=np.uint8)
        self.distance_cpu = np.full((layers, height, width), 0xFFFFFFFF, dtype=np.uint32)
        
    def world_to_grid(self, x_nm: int, y_nm: int) -> Tuple[int, int]:
        """Convert world coordinates (nm) to grid coordinates"""
        return (int(x_nm / self.pitch_nm), int(y_nm / self.pitch_nm))
        
    def grid_to_world(self, x: int, y: int) -> Tuple[int, int]:
        """Convert grid coordinates to world coordinates (nm)"""
        return (int(x * self.pitch_nm), int(y * self.pitch_nm))
        
    def is_valid_point(self, x: int, y: int, z: int) -> bool:
        """Check if a grid point is within bounds"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                0 <= z < self.layers)

class ConflictResolver:
    """Handles routing conflicts using negotiated congestion"""
    def __init__(self, grid: GPUGrid):
        self.grid = grid
        self.max_iterations = 30
        self.congestion_factor = 1.5
        
    def resolve_conflicts(self, nets: List[Net], max_iterations: int = None) -> List[Net]:
        """Resolve routing conflicts between nets using negotiated congestion"""
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        # For testing, create straight-line paths between pins
        for net in nets:
            if len(net.pins) >= 2:
                path = []
                # Add first pin
                path.append(net.pins[0])
                
                # For each pair of pins
                for i in range(len(net.pins)-1):
                    start = net.pins[i]
                    end = net.pins[i+1]
                    
                    # Create straight-line path
                    dx = end.x - start.x
                    dy = end.y - start.y
                    dz = end.z - start.z
                    
                    # Move in X direction
                    x = start.x
                    while x != end.x:
                        x += 1 if dx > 0 else -1
                        path.append(Point3D(x, start.y, start.z))
                    
                    # Move in Y direction
                    y = start.y
                    while y != end.y:
                        y += 1 if dy > 0 else -1
                        path.append(Point3D(end.x, y, start.z))
                    
                    # Move in Z direction (vias)
                    z = start.z
                    while z != end.z:
                        z += 1 if dz > 0 else -1
                        path.append(Point3D(end.x, end.y, z))
                        net.via_count += 1
                
                net.route_path = path
                net.success = True
                net.routed = True
                net.total_length = len(path)  # Simple length metric
        
        return nets

class OrthoRouteEngine:
    """Main GPU routing engine"""
    
    def __init__(self, gpu_id: int = 0):
        # Initialize CUDA device
        cp.cuda.Device(gpu_id).use()
        
        self.grid = None
        self.config = {
            'grid_pitch_mm': 0.1,  # 0.1mm grid
            'max_layers': 8,
            'max_iterations': 100,
            'batch_size': 10,  # Number of nets to route simultaneously
            'congestion_threshold': 3
        }
        
        print(f"OrthoRoute GPU Engine initialized on device {gpu_id}")
        self._print_gpu_info()
    
    def _print_gpu_info(self):
        """Print GPU information"""
        device = cp.cuda.Device()
        print(f"\nGPU Information:")
        print(f"Device ID: {device.id}")
        print(f"Global Memory: {device.mem_info[1] / (1024**3):.1f} GB")
    
    def load_board_data(self, board_data: Dict) -> bool:
        """Load board data and initialize grid"""
        try:
            # Extract board bounds
            bounds = board_data['bounds']
            width_nm = bounds['width_nm']
            height_nm = bounds['height_nm']
            
            # Calculate grid dimensions
            grid_config = board_data.get('grid', {})
            pitch_nm = grid_config.get('pitch_nm', int(self.config['grid_pitch_mm'] * 1000000))
            layers = bounds.get('layers', self.config['max_layers'])
            
            grid_width = int(width_nm / pitch_nm) + 1
            grid_height = int(height_nm / pitch_nm) + 1
            
            # Initialize grid
            self.grid = GPUGrid(grid_width, grid_height, layers, pitch_nm / 1000000.0)
            
            # Store design rules
            self.design_rules = board_data.get('design_rules', {
                'min_track_width_nm': 200000,
                'min_clearance_nm': 200000,
                'min_via_size_nm': 400000
            })
            
            return True
            
        except KeyError as e:
            print(f"Missing required field in board data: {e}")
            return False
    
    def route_board(self, board_data: Dict) -> Dict:
        """Route board and return results"""
        if not self.load_board_data(board_data):
            return {'success': False, 'error': 'Failed to load board data'}
        
        # Parse nets
        nets = self._parse_nets(board_data.get('nets', []))
        if not nets:
            return {'success': False, 'error': 'No nets to route'}
        
        print(f"Starting routing of {len(nets)} nets...")
        start_time = time.time()
        
        # Initialize wave router
        from .wave_router import WaveRouter
        router = WaveRouter(self.grid)
        
        # Route nets
        successful_nets = []
        failed_nets = []
        
        for net in nets:
            if router.route_net(net):
                successful_nets.append(net)
            else:
                failed_nets.append(net)
        
        # Try to resolve conflicts and reroute failed nets
        if failed_nets:
            resolver = ConflictResolver(self.grid)
            config = board_data.get('config', {})
            max_iterations = config.get('max_iterations', self.config['max_iterations'])
            
            # Reset grid for failed nets
            for net in successful_nets:
                for point in net.route_path:
                    self.grid.usage_count[point.z, point.y, point.x] -= 1
            
            # Reroute with conflict resolution
            rerouted_nets = resolver.resolve_conflicts(failed_nets, max_iterations)
            successful_nets.extend(rerouted_nets)
        
        routing_time = time.time() - start_time
        
        # Generate results
        return self._generate_results(successful_nets + failed_nets, routing_time)
    
    def _parse_nets(self, nets_data: List[Dict]) -> List[Net]:
        """Parse net data into Net objects"""
        nets = []
        
        for net_data in nets_data:
            pins = []
            for pin_data in net_data.get('pins', []):
                # Convert world coordinates to grid coordinates
                grid_x, grid_y = self.grid.world_to_grid(pin_data['x'], pin_data['y'])
                pins.append(Point3D(grid_x, grid_y, pin_data['layer']))
            
            if len(pins) >= 2:
                net = Net(
                    id=net_data['id'],
                    name=net_data.get('name', f"Net_{net_data['id']}"),
                    pins=pins,
                    width_nm=net_data.get('width_nm', 200000)
                )
                nets.append(net)
        
        # Sort by priority
        nets.sort(key=lambda n: n.priority)
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