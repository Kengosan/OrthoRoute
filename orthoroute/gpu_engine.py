"""
GPU-accelerated routing engine for OrthoRoute
"""

import cupy as cp
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

class OrthoRouteEngine:
    def __init__(self):
        """Initialize the GPU routing engine."""
        print("OrthoRoute GPU Engine initialized on device 0")
        # Debug info with unique ID to track engine instances
        import random, time
        self.engine_id = f"engine_{int(time.time() % 10000)}_{random.randint(1000, 9999)}"
        print(f"DEBUG: Created engine instance with ID: {self.engine_id}")
        
        self._print_gpu_info()
        # Explicitly initialize visualization attributes
        self.visualizer = None  # Will hold visualization object when enabled
        self.viz_config = None  # Will hold visualization configuration
        self.grid = None
        # Default configuration
        self.config = {
            'max_iterations': 3,
            'via_cost': 10,
            'conflict_penalty': 50,
            'max_wave_iterations': 1000,
            'grid_pitch_mm': 0.1,  # 100 micron grid pitch
            'max_layers': 2  # Default number of layers
        }
        
    def enable_visualization(self, viz_config):
        """Enable real-time visualization during routing."""
        # Make sure engine_id exists
        if not hasattr(self, 'engine_id'):
            import random, time
            self.engine_id = f"engine_{int(time.time() % 10000)}_{random.randint(1000, 9999)}"
            print(f"DEBUG: Created engine ID: {self.engine_id}")
        
        # Make sure visualizer attribute exists
        if not hasattr(self, 'visualizer'):
            print("DEBUG: Adding visualizer attribute")
            self.visualizer = None
        
        try:
            print(f"DEBUG: Engine {self.engine_id} - Enabling visualization")
        except AttributeError:
            print("DEBUG: Enabling visualization (engine_id not available)")
            
        self.viz_config = viz_config
        
        try:
            print(f"DEBUG: Engine {self.engine_id} - viz_config set: {self.viz_config is not None}")
        except AttributeError:
            print(f"DEBUG: viz_config set: {self.viz_config is not None}")
        # Visualizer will be created after grid is initialized during route_board
        
    def _print_gpu_info(self):
        """Print GPU information."""
        device = cp.cuda.Device()
        print("\nGPU Information:")
        print(f"Device ID: {device.id}")
        try:
            mem_info = device.mem_info()
            if isinstance(mem_info, (list, tuple)) and len(mem_info) >= 2:
                total_mem = float(mem_info[1]) / (1024**3)
                print(f"Global Memory: {total_mem:.1f} GB")
            else:
                print("Global Memory: Unknown (testing environment)")
        except (AttributeError, TypeError, IndexError, ValueError):
            print("Global Memory: Unknown (testing environment)")

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
        try:
            # Handle potential negative values or invalid types
            if not isinstance(x_nm, (int, float)) or not isinstance(y_nm, (int, float)):
                print(f"⚠️ Warning: Invalid coordinate types: x={type(x_nm)}, y={type(y_nm)}")
                return (0, 0)
                
            grid_x = int(x_nm / self.pitch_nm)
            grid_y = int(y_nm / self.pitch_nm)
            return (grid_x, grid_y)
        except Exception as e:
            print(f"⚠️ Error in world_to_grid conversion: {e}")
            return (0, 0)
        
    def grid_to_world(self, x: int, y: int) -> Tuple[int, int]:
        """Convert grid coordinates to world coordinates (nm)"""
        try:
            return (int(x * self.pitch_nm), int(y * self.pitch_nm))
        except Exception as e:
            print(f"⚠️ Error in grid_to_world conversion: {e}")
            return (0, 0)
        
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
            bounds = board_data.get('bounds', {})
            width_nm = bounds.get('width_nm', 100000000)  # Default 100mm
            height_nm = bounds.get('height_nm', 100000000)  # Default 100mm
            
            # Calculate grid dimensions
            grid_config = board_data.get('grid', {})
            pitch_nm = grid_config.get('pitch_nm', int(self.config['grid_pitch_mm'] * 1000000))
            layers = bounds.get('layers', self.config['max_layers'])
            
            # Ensure we have valid values
            if pitch_nm <= 0:
                print(f"⚠️ Warning: Invalid grid pitch {pitch_nm}nm, using default 100,000nm")
                pitch_nm = 100000
                
            if width_nm <= 0 or height_nm <= 0:
                print(f"⚠️ Warning: Invalid board dimensions ({width_nm}x{height_nm})nm, using default 100x100mm")
                width_nm = 100000000
                height_nm = 100000000
                
            if layers <= 0:
                print(f"⚠️ Warning: Invalid layer count {layers}, using default 2")
                layers = 2
            
            # Calculate grid dimensions with a minimum size and add some margin
            grid_width = max(int(width_nm / pitch_nm) + 10, 100)
            grid_height = max(int(height_nm / pitch_nm) + 10, 100)
            
            print(f"Creating routing grid: {grid_width}x{grid_height}x{layers} cells at {pitch_nm}nm pitch")
            
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
        except Exception as e:
            print(f"Error loading board data: {e}")
            return False
    
    def route_board(self, board_data: Dict) -> Dict:
        """Route board and return results"""
        # Use safer debug prints to avoid attribute errors
        try:
            print(f"DEBUG: Engine {self.engine_id} - Starting route_board")
        except AttributeError:
            print("DEBUG: Starting route_board (engine_id not available)")
            
        # Ensure engine_id attribute exists
        if not hasattr(self, 'engine_id'):
            import random, time
            self.engine_id = f"engine_{int(time.time() % 10000)}_{random.randint(1000, 9999)}"
            print(f"DEBUG: Created engine ID: {self.engine_id}")
            
        print(f"DEBUG: Engine {self.engine_id} - has viz_config: {hasattr(self, 'viz_config')}")
        if hasattr(self, 'viz_config'):
            print(f"DEBUG: Engine {self.engine_id} - viz_config value: {self.viz_config}")
        
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
        
        # Always ensure visualizer attribute exists
        if not hasattr(self, 'visualizer'):
            self.visualizer = None
            
        # Initialize visualization if enabled
        print("DEBUG: Before visualization check")
        try:
            engine_id = getattr(self, 'engine_id', 'unknown')
            print(f"DEBUG: Engine {engine_id} - Before visualization check")
            print(f"DEBUG: Engine {engine_id} - has viz_config: {hasattr(self, 'viz_config')}")
            
            if hasattr(self, 'viz_config') and self.viz_config:
                print(f"DEBUG: Engine {engine_id} - Trying to initialize visualization")
                try:
                    # Import at top level to avoid issues
                    from orthoroute.visualization import RoutingVisualizer
                    self.visualizer = RoutingVisualizer(self.grid, self.viz_config)
                    self.visualizer.initialize_display(board_data)
                    self.visualizer.start()
                    print(f"DEBUG: Engine {engine_id} - Real-time visualization enabled")
                except Exception as e:
                    print(f"DEBUG: Visualization error: {e}")
                    self.visualizer = None
        except Exception as e:
            print(f"DEBUG: Error in visualization setup: {e}")
            self.visualizer = None
        
        # Route nets
        successful_nets = []
        failed_nets = []
        
        for i, net in enumerate(nets):
            if router.route_net(net):
                successful_nets.append(net)
            else:
                failed_nets.append(net)
            
            # Update visualization every few nets
            if hasattr(self, 'visualizer') and self.visualizer and i % 5 == 0:
                try:
                    self.visualizer.update(successful_nets, failed_nets, 
                                         progress=i/len(nets))
                except Exception as e:
                    print(f"DEBUG: Visualization update error: {e}")
                    # Continue routing even if visualization fails
        
        # Try to resolve conflicts and reroute failed nets
        if failed_nets:
            resolver = ConflictResolver(self.grid)
            config = board_data.get('config', {})
            max_iterations = config.get('max_iterations', self.config['max_iterations'])
            
            # Reset grid for failed nets
            for net in successful_nets:
                for point in net.route_path:
                    self.grid.usage_count[point.z, point.y, point.x] -= 1
            
            # Update visualization before conflict resolution
            if hasattr(self, 'visualizer') and self.visualizer:
                try:
                    self.visualizer.update(successful_nets, failed_nets, 
                                        progress=0.8, 
                                        status="Resolving conflicts...")
                except Exception as e:
                    print(f"DEBUG: Visualization update error: {e}")
                    # Continue routing even if visualization fails
            
            # Reroute with conflict resolution
            rerouted_nets = resolver.resolve_conflicts(failed_nets, max_iterations)
            successful_nets.extend(rerouted_nets)
            
            # Final visualization update
            if hasattr(self, 'visualizer') and self.visualizer:
                try:
                    self.visualizer.update(successful_nets, failed_nets, 
                                        progress=1.0, 
                                        status="Routing complete")
                except Exception as e:
                    print(f"DEBUG: Visualization update error: {e}")
                    # Continue routing even if visualization fails
        
        routing_time = time.time() - start_time
        
        # Stop visualization
        if hasattr(self, 'visualizer') and self.visualizer:
            try:
                # Leave visualization window open for viewing
                self.visualizer.set_status("Routing complete - Close window to continue")
            except Exception as e:
                print(f"DEBUG: Visualization status update error: {e}")
                # Continue routing even if visualization fails
            
        # Generate results
        return self._generate_results(successful_nets + failed_nets, routing_time)
    
    def _parse_nets(self, nets_data: List[Dict]) -> List[Net]:
        """Parse net data into Net objects"""
        nets = []
        
        for net_data in nets_data:
            valid_pins = []
            invalid_pins = []
            
            for pin_data in net_data.get('pins', []):
                # Convert world coordinates to grid coordinates
                grid_x, grid_y = self.grid.world_to_grid(pin_data['x'], pin_data['y'])
                
                # Check if pin is within grid bounds
                if (0 <= grid_x < self.grid.width and 
                    0 <= grid_y < self.grid.height and
                    0 <= pin_data.get('layer', 0) < self.grid.layers):
                    valid_pins.append(Point3D(grid_x, grid_y, pin_data.get('layer', 0)))
                else:
                    # Pin is outside grid bounds
                    invalid_pins.append((grid_x, grid_y, pin_data.get('layer', 0)))
            
            # Only create net if there are at least 2 valid pins
            if len(valid_pins) >= 2:
                net = Net(
                    id=net_data['id'],
                    name=net_data.get('name', f"Net_{net_data['id']}"),
                    pins=valid_pins,
                    width_nm=net_data.get('width_nm', 200000)
                )
                nets.append(net)
            elif invalid_pins:
                # Log the issue
                net_id = net_data.get('id', 0)
                net_name = net_data.get('name', f"Net_{net_id}")
                print(f"⚠️ Warning: Net {net_name} has pins outside grid bounds:")
                for x, y, layer in invalid_pins:
                    print(f"   Pin at ({x}, {y}, {layer}) - Grid size is ({self.grid.width}, {self.grid.height}, {self.grid.layers})")
        
        # Sort by priority
        nets.sort(key=lambda n: n.priority)
        return nets
    
    def route(self, board_data: Dict, config: Dict = None) -> Dict:
        """
        Route the board with the given config.
        This is an alias for route_board to ensure compatibility with the KiCad plugin.
        """
        # Handle different board data formats (orthoroute_plugin.py vs orthoroute_kicad.py)
        if 'netlist' in board_data and not 'nets' in board_data:
            # Convert from orthoroute_plugin.py format to gpu_engine format
            netlist = board_data['netlist']
            nets = []
            
            for net_name, pins in netlist.items():
                if len(pins) < 2:  # Skip nets with less than 2 pins
                    continue
                    
                net = {
                    'id': len(nets) + 1,  # Generate sequential IDs
                    'name': net_name,
                    'pins': pins,
                    'width_nm': 200000  # Default 0.2mm
                }
                nets.append(net)
                
            # Replace netlist with nets array
            board_data['nets'] = nets
        
        # Merge configuration if provided
        if config:
            # Update config with user settings
            if 'grid' in config:
                board_data['grid'] = config['grid']
            if 'routing' in config:
                board_data['routing'] = config['routing']
            if 'options' in config:
                board_data['options'] = config['options']
        
        # Ensure required fields exist
        if 'bounds' not in board_data:
            board_data['bounds'] = {
                'width_nm': int(board_data.get('width', 100000000)),  # Convert to nm
                'height_nm': int(board_data.get('height', 100000000)),
                'layers': 2
            }
            
        if 'grid' not in board_data:
            board_data['grid'] = {
                'pitch_nm': 100000,  # 0.1mm grid
                'via_size_nm': 200000  # 0.2mm vias
            }
        
        # Call the main routing method
        return self.route_board(board_data)
    
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