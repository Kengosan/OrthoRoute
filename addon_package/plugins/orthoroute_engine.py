"""
Standalone GPU routing engine for OrthoRoute KiCad plugin
This module contains all necessary classes for GPU-accelerated routing
without requiring external package installation.

Now includes innovative grid-based routing for complex backplane designs.
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
                def use(self): pass
                def mem_info(self): return (0, 8*1024**3)  # Mock 8GB
                @property
                def name(self):
                    return "Mock GPU (CuPy not available)"
            
            class runtime:
                @staticmethod
                def getDeviceProperties(device_id):
                    return {'name': b'Mock GPU (CuPy not available)'}
                    
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
        """Convert world coordinates (nm) to grid coordinates using simple direct mapping"""
        try:
            if not isinstance(x_nm, (int, float)) or not isinstance(y_nm, (int, float)):
                return (0, 0)
            
            # Get simple coordinate system from grid
            min_x_nm = getattr(self, 'min_x_nm', 0)
            min_y_nm = getattr(self, 'min_y_nm', 0)
            margin_cells = getattr(self, 'margin_cells', 50)
            
            # Simple direct conversion: translate to origin, add margin, convert to grid units
            grid_x = int((x_nm - min_x_nm) / self.pitch_nm) + margin_cells // 2
            grid_y = int((y_nm - min_y_nm) / self.pitch_nm) + margin_cells // 2
            
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
    
    def cleanup(self):
        """Clean up GPU grid resources"""
        if CUPY_AVAILABLE and hasattr(self, 'availability'):
            try:
                # Delete GPU arrays to free memory
                del self.availability
                del self.congestion
                del self.distance
                del self.usage_count
                del self.parent
                print("🧹 GPU grid arrays cleaned up")
            except Exception as e:
                print(f"⚠️ Grid cleanup warning: {e}")

class GPUWavefrontRouter:
    """Real GPU-accelerated wavefront router using CuPy"""
    
    def __init__(self, grid):
        self.grid = grid
        self.use_gpu = CUPY_AVAILABLE
        self.should_cancel = lambda: False  # Default no-cancel callback
        print(f"GPU Wavefront Router initialized (GPU: {self.use_gpu})") 
    
    def set_cancel_callback(self, should_cancel_fn):
        """Set the cancellation callback function"""
        self.should_cancel = should_cancel_fn or (lambda: False)
    
    def route_net(self, net: Net) -> bool:
        """Route a single net using GPU-accelerated Lee's algorithm"""
        if len(net.pins) < 2:
            return False
        
        # Check for cancellation at start
        if self.should_cancel():
            print(f"🛑 Routing cancelled before starting net {net.id}")
            return False
        
        print(f"Routing net {net.id} ({net.name}) with {len(net.pins)} pins")
        
        if self.use_gpu:
            return self._route_net_gpu(net)
        else:
            return self._route_net_cpu(net)
    
    def _route_net_gpu(self, net: Net) -> bool:
        """GPU-accelerated routing using CuPy"""
        
        # If CuPy is not available, fall back to CPU
        if not CUPY_AVAILABLE:
            print(f"⚠️ CuPy not available, falling back to CPU routing for {net.name}")
            return self._route_net_cpu(net)
        
        try:
            print(f"🚀 Starting GPU routing for {net.name}")
            
            # Initialize arrays on CPU first to avoid CuPy mock object issues
            distance_cpu = np.full((self.grid.layers, self.grid.height, self.grid.width), 
                                 np.inf, dtype=np.float32)
            parent_cpu = np.full((self.grid.layers, self.grid.height, self.grid.width, 3), 
                               -1, dtype=np.int32)
            
            # Only transfer to GPU if CuPy is truly available (not mock)
            try:
                distance = cp.array(distance_cpu)
                parent = cp.array(parent_cpu)
                print("✅ Arrays successfully transferred to GPU")
            except Exception as e:
                print(f"❌ GPU transfer failed: {e}, falling back to CPU")
                return self._route_net_cpu(net)
            
            # Route between consecutive pins (minimum spanning tree approach)
            full_path = []
            
            for i in range(len(net.pins) - 1):
                # Check for cancellation between pin pairs
                if self.should_cancel():
                    print(f"🛑 Routing cancelled during segment {i} of net {net.id}")
                    return False
                    
                source = net.pins[i]
                target = net.pins[i + 1]
                
                print(f"🔍 Routing segment {i}: ({source.x}, {source.y}, {source.z}) → ({target.x}, {target.y}, {target.z})")
                
                # Reset distance map
                distance.fill(cp.inf)
                parent.fill(-1)
                
                # Run Lee's algorithm
                path = self._lee_algorithm_gpu(source, target, distance, parent)
                if not path:
                    print(f"❌ Failed to route segment {i} of net {net.id}")
                    return False
                
                print(f"✅ Found path with {len(path)} points")
                
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
        
        # Validate coordinates
        if not (0 <= source.x < self.grid.width and 0 <= source.y < self.grid.height and 0 <= source.z < self.grid.layers):
            print(f"❌ Source point ({source.x}, {source.y}, {source.z}) out of bounds")
            return None
            
        if not (0 <= target.x < self.grid.width and 0 <= target.y < self.grid.height and 0 <= target.z < self.grid.layers):
            print(f"❌ Target point ({target.x}, {target.y}, {target.z}) out of bounds")
            return None
        
        try:
            print(f"🔍 Starting Lee's algorithm: source ({source.x},{source.y},{source.z}) → target ({target.x},{target.y},{target.z})")
            
            # Mark source
            distance[source.z, source.y, source.x] = 0
            
            # Use CPU-based queue for better reliability
            from collections import deque
            queue = deque([(source.x, source.y, source.z, 0)])  # (x, y, z, dist)
            
            # Neighbor offsets (x, y, z) - only in-plane for now (no vias)
            neighbors = [
                (1, 0, 0), (-1, 0, 0),  # X direction
                (0, 1, 0), (0, -1, 0),  # Y direction
                # Temporarily disable vias: (0, 0, 1), (0, 0, -1)   # Z direction (via)
            ]
            
            max_iterations = min(10000, self.grid.width * self.grid.height)  # Reasonable limit
            iteration = 0
            
            while queue and iteration < max_iterations:
                iteration += 1
                x, y, z, dist = queue.popleft()
                
                # Check if we reached target
                if x == target.x and y == target.y and z == target.z:
                    print(f"✅ Found target at distance {dist}, iterations: {iteration}")
                    # Reconstruct path
                    return self._reconstruct_path_gpu(source, target, parent)
                
                # Expand to neighbors
                for dx, dy, dz in neighbors:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Check bounds
                    if (0 <= nx < self.grid.width and 
                        0 <= ny < self.grid.height and 
                        0 <= nz < self.grid.layers):
                        
                        try:
                            # Check if available and not visited
                            if (self.grid.availability[nz, ny, nx] and 
                                distance[nz, ny, nx] == cp.inf):
                                
                                distance[nz, ny, nx] = dist + 1
                                parent[nz, ny, nx] = [x, y, z]
                                queue.append((nx, ny, nz, dist + 1))
                        except Exception as e:
                            print(f"❌ Error accessing grid at ({nx},{ny},{nz}): {e}")
                            continue
                
                if iteration % 1000 == 0:
                    print(f"🔄 Lee's iteration {iteration}, queue size: {len(queue)}")
            
            print(f"❌ Target not reached after {iteration} iterations, queue size: {len(queue)}")
            return None
            
        except Exception as e:
            print(f"❌ Exception in Lee's algorithm: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _reconstruct_path_gpu(self, source: Point3D, target: Point3D, parent):
        """Reconstruct path from parent array"""
        try:
            path = []
            current = target
            max_path_length = self.grid.width + self.grid.height  # Prevent infinite loops
            
            for step in range(max_path_length):
                path.append(current)
                
                # Check if we reached the source
                if current.x == source.x and current.y == source.y and current.z == source.z:
                    path.reverse()
                    print(f"✅ Path reconstructed: {len(path)} points")
                    return path
                
                # Get parent coordinates
                try:
                    parent_coords = parent[current.z, current.y, current.x]
                    px, py, pz = int(parent_coords[0]), int(parent_coords[1]), int(parent_coords[2])
                    
                    if px == -1:  # No parent found
                        print(f"❌ No parent found for point ({current.x}, {current.y}, {current.z})")
                        return None
                        
                    current = Point3D(px, py, pz)
                    
                except Exception as e:
                    print(f"❌ Error accessing parent at ({current.x}, {current.y}, {current.z}): {e}")
                    return None
            
            print(f"❌ Path reconstruction exceeded maximum length ({max_path_length})")
            return None
            
        except Exception as e:
            print(f"❌ Exception in path reconstruction: {e}")
            return None
    
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
    
    def cleanup(self):
        """Clean up GPU router resources"""
        if CUPY_AVAILABLE:
            try:
                # Force GPU synchronization and cleanup
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                print("🧹 GPU router cleaned up")
            except Exception as e:
                print(f"⚠️ Router cleanup warning: {e}")


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
            'congestion_threshold': 3,
            'routing_algorithm': 'gpu_wavefront',  # 'gpu_wavefront' or 'grid_based'
            'grid_spacing': 2540000,  # Grid spacing for grid-based routing (nm)
            'prefer_grid_routing': False  # Prefer grid routing for complex designs
        }
        
    def enable_visualization(self, viz_config):
        """Enable real-time visualization during routing."""
        self.viz_config = viz_config
        print(f"Visualization enabled: {viz_config}")
        
    def _print_gpu_info(self):
        """Print GPU information."""
        if CUPY_AVAILABLE:
            device = cp.cuda.Device()
            
            # Get device name
            device_name = "Unknown GPU"
            try:
                device_props = cp.cuda.runtime.getDeviceProperties(device.id)
                device_name = device_props['name'].decode('utf-8')
            except:
                device_name = f"CUDA Device {device.id}"
            
            print(f"GPU Device: {device_name} (ID: {device.id})")
            
            # Get memory info
            try:
                mem_info = device.mem_info()
                if isinstance(mem_info, (list, tuple)) and len(mem_info) >= 2:
                    free_mem = float(mem_info[0]) / (1024**3)
                    total_mem = float(mem_info[1]) / (1024**3)
                    used_mem = total_mem - free_mem
                    print(f"GPU Memory: {used_mem:.1f}/{total_mem:.1f} GB used")
                else:
                    print("GPU Memory: Available")
            except Exception as e:
                print(f"GPU Memory: Unknown ({e})")
        else:
            print("GPU: Not available (using CPU fallback)")

    def load_board_data(self, board_data: Dict) -> bool:
        """Load board data and initialize grid"""
        debug_print = getattr(self, 'debug_print', print)
        
        try:
            # Extract board bounds
            bounds = board_data.get('bounds', {})
            width_nm = bounds.get('width_nm', 100000000)  # Default 100mm
            height_nm = bounds.get('height_nm', 100000000)
            
            # Get actual coordinate range (if available)
            min_x_nm = bounds.get('min_x_nm', 0)
            min_y_nm = bounds.get('min_y_nm', 0)
            max_x_nm = bounds.get('max_x_nm', width_nm)
            max_y_nm = bounds.get('max_y_nm', height_nm)
            
            debug_print(f"📐 Board bounds: {width_nm/1e6:.1f}mm x {height_nm/1e6:.1f}mm")
            debug_print(f"📍 Coordinate range: X({min_x_nm/1e6:.1f} to {max_x_nm/1e6:.1f}mm), Y({min_y_nm/1e6:.1f} to {max_y_nm/1e6:.1f}mm)")
            
            # Calculate grid dimensions
            grid_config = board_data.get('grid', {})
            pitch_nm = grid_config.get('pitch_nm', int(self.config['grid_pitch_mm'] * 1000000))
            layers = bounds.get('layers', self.config['max_layers'])
            
            debug_print(f"📏 Grid pitch: {pitch_nm/1e6:.2f}mm")
            
            # Ensure valid values
            if pitch_nm <= 0:
                pitch_nm = 100000  # 0.1mm default
            if width_nm <= 0 or height_nm <= 0:
                width_nm = height_nm = 100000000
            if layers <= 0:
                layers = 2
            
            # SIMPLE DIRECT APPROACH: Calculate grid from actual pins
            debug_print(f"🎯 Using simple direct coordinate mapping...")
            
            # Get all pin coordinates from all nets
            all_pin_coords = []
            for net_data in board_data.get('nets', []):
                for pin in net_data.get('pins', []):
                    all_pin_coords.append((pin['x'], pin['y']))
            
            if not all_pin_coords:
                debug_print("❌ No pins found for grid calculation")
                return False
            
            # Direct coordinate range calculation
            min_x = min(coord[0] for coord in all_pin_coords)
            max_x = max(coord[0] for coord in all_pin_coords)
            min_y = min(coord[1] for coord in all_pin_coords)
            max_y = max(coord[1] for coord in all_pin_coords)
            
            coord_width = max_x - min_x
            coord_height = max_y - min_y
            
            debug_print(f"📍 Pin coordinate range: X({min_x/1e6:.1f} to {max_x/1e6:.1f}mm) = {coord_width/1e6:.1f}mm")
            debug_print(f"📍 Pin coordinate range: Y({min_y/1e6:.1f} to {max_y/1e6:.1f}mm) = {coord_height/1e6:.1f}mm")
            
            # Simple grid sizing with fixed margin
            MARGIN_CELLS = 50  # Simple 50-cell margin (5mm at 0.1mm pitch)
            grid_width = int(coord_width / pitch_nm) + MARGIN_CELLS
            grid_height = int(coord_height / pitch_nm) + MARGIN_CELLS
            
            debug_print(f"🏗️ Creating routing grid: {grid_width}x{grid_height}x{layers} cells")
            debug_print(f"   Grid covers: {(grid_width * pitch_nm)/1e6:.1f}mm x {(grid_height * pitch_nm)/1e6:.1f}mm")
            
            # Initialize grid
            self.grid = GPUGrid(grid_width, grid_height, layers, pitch_nm / 1000000.0)
            
            # Store simple coordinate system in grid
            self.grid.min_x_nm = min_x
            self.grid.min_y_nm = min_y
            self.grid.coord_width = coord_width
            self.grid.coord_height = coord_height
            self.grid.margin_cells = MARGIN_CELLS
            
            debug_print(f"📍 Simple coordinate system: origin=({min_x/1e6:.1f}, {min_y/1e6:.1f})mm, margin={MARGIN_CELLS} cells")
            
            return True
            
        except Exception as e:
            debug_print(f"❌ Error loading board data: {e}")
            return False
    
    def route(self, board_data: Dict, config: Dict = None, board=None) -> Dict:
        """Route the board with the given config."""
        debug_print = getattr(self, 'debug_print', print)
        
        # Store board reference for track creation
        self.board = board
        
        debug_print(f"🚀 Starting route with engine {self.engine_id}")
        debug_print(f"📋 Board reference available: {board is not None}")
        
        # Debug board data received
        debug_print(f"📊 Board data summary:")
        debug_print(f"   - Bounds: {board_data.get('bounds', 'Missing')}")
        debug_print(f"   - Raw nets count: {len(board_data.get('nets', []))}")
        
        # Debug first few nets
        nets_data = board_data.get('nets', [])
        for i, net_data in enumerate(nets_data[:3]):
            pins = net_data.get('pins', [])
            debug_print(f"   - Net {i+1}: {net_data.get('name', 'Unknown')} ({len(pins)} pins)")
            for j, pin in enumerate(pins[:2]):
                debug_print(f"     Pin {j+1}: x={pin.get('x', 'Missing')}, y={pin.get('y', 'Missing')}, layer={pin.get('layer', 'Missing')}")
        
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
            debug_print("❌ CRITICAL: load_board_data returned False!")
            return {'success': False, 'error': 'Failed to load board data'}
        
        debug_print("✅ CHECKPOINT: Board data loaded successfully")
        
        # Parse nets with detailed logging
        debug_print("🔍 Parsing nets...")
        debug_print("📍 CHECKPOINT: About to call _parse_nets")
        nets = self._parse_nets(board_data.get('nets', []), debug_print)
        debug_print(f"✅ Parsed {len(nets)} valid nets for routing")
        
        if not nets:
            debug_print("❌ No valid nets found after parsing!")
            return {'success': False, 'error': 'No nets to route'}
        
        debug_print(f"Routing {len(nets)} nets...")
        start_time = time.time()
        
        # Extract progress callback
        progress_callback = config.get('progress_callback', None)
        should_cancel = config.get('should_cancel', lambda: False)
        
        # Choose routing algorithm based on config and board complexity
        routing_algorithm = self.config.get('routing_algorithm', 'gpu_wavefront')
        
        # Determine if grid routing should be used
        net_count = len(nets)
        average_pins_per_net = sum(len(net.pins) for net in nets) / max(net_count, 1)
        
        use_grid_routing = (
            routing_algorithm == 'grid_based' or
            (self.config.get('prefer_grid_routing', False) and net_count > 50) or
            (average_pins_per_net > 10)  # Complex nets benefit from grid routing
        )
        
        debug_print(f"📍 CHECKPOINT 4: Grid routing decision")
        debug_print(f"   Net count: {net_count}")
        debug_print(f"   Average pins per net: {average_pins_per_net:.1f}")
        debug_print(f"   Use grid routing: {use_grid_routing}")
        
        successful_nets = []
        
        if use_grid_routing:
            debug_print("📍 CHECKPOINT 5: Attempting grid routing")
            print("🏗️ Using innovative grid-based routing for complex design")
            try:
                # Import grid router (lazy import to avoid circular dependencies)
                from .grid_router import create_grid_router
                debug_print("📍 Grid router imported successfully")
                
                # Create board data in format expected by grid router
                grid_board_data = {
                    'board_width': board_data['bounds']['width_nm'],
                    'board_height': board_data['bounds']['height_nm'],
                    'layer_count': board_data['bounds']['layers'],
                    'obstacles': board_data.get('obstacles', {})
                }
                
                # Create and configure grid router
                debug_print("📍 Creating grid router...")
                grid_router = create_grid_router(grid_board_data, self.config)
                debug_print(f"📍 Grid router created: {grid_router is not None}")
                
                if grid_router:
                    debug_print("📍 Grid router available, converting nets...")
                    # Convert nets to grid router format
                    grid_nets = []
                    for net in nets:
                        grid_net = {
                            'net_code': net.id,
                            'net_name': net.name,
                            'pins': [{'x': pin.x * 1000000, 'y': pin.y * 1000000, 'layer': pin.z} 
                                   for pin in net.pins]  # Convert back to nm
                        }
                        grid_nets.append(grid_net)
                    
                    debug_print(f"📍 Converted {len(grid_nets)} nets for grid routing")
                    
                    # Route using grid algorithm
                    debug_print("📍 Starting grid routing...")
                    grid_results = grid_router.route_nets(grid_nets)
                    debug_print(f"📍 Grid routing completed: {grid_results}")
                    
                    # Convert successful nets back to our format
                    for net_code, route_data in grid_results.get('routed_nets', {}).items():
                        # Find original net
                        original_net = next((n for n in nets if n.id == net_code), None)
                        if original_net:
                            successful_nets.append(original_net)
                    
                    grid_router.cleanup()
                    print(f"✅ Grid routing completed: {len(successful_nets)}/{len(nets)} nets routed")
                else:
                    debug_print("❌ Grid router creation failed, falling back to GPU wavefront")
                    use_grid_routing = False
                    
            except ImportError as e:
                debug_print(f"⚠️ Grid router not available: {e}, using GPU wavefront")
                use_grid_routing = False
            except Exception as e:
                debug_print(f"⚠️ Grid routing failed: {e}, falling back to GPU wavefront")
                import traceback
                debug_print(f"⚠️ Grid routing traceback: {traceback.format_exc()}")
                use_grid_routing = False
        else:
            debug_print("📍 CHECKPOINT 5: Skipping grid routing")
        
        # Fall back to GPU wavefront routing if grid routing not used or failed
        if not use_grid_routing:
            debug_print("📍 CHECKPOINT 6: Starting GPU wavefront routing")
            print(f"🚀 Starting GPU wavefront routing for {len(nets)} nets...")
            
            # Initialize router - use GPU router for real routing
            if CUPY_AVAILABLE:
                debug_print("📍 CuPy available, creating GPU router")
                router = GPUWavefrontRouter(self.grid)
                print("🚀 Using GPU-accelerated wavefront routing")
            else:
                debug_print("📍 CuPy not available, using CPU fallback")
                router = GPUWavefrontRouter(self.grid)  # Has CPU fallback
                print("🖥️ Using CPU fallback routing (CuPy not available)")
            
            # Set cancellation callback
            debug_print("📍 Setting cancel callback")
            router.set_cancel_callback(should_cancel)
            
            debug_print(f"📍 Router initialized, starting route loop...")
            
            # Route nets with progress reporting
            try:
                debug_print(f"📍 CHECKPOINT 7: Entering routing loop for {len(nets)} nets")
                for i, net in enumerate(nets):
                    debug_print(f"📍 Loop iteration {i}: Processing net '{net.name}' with {len(net.pins)} pins")
                    
                    # Check for cancellation
                    if should_cancel():
                        print("🛑 Routing cancelled by user")
                        break
                    
                    net_name = net.name
                    net_progress = (i / len(nets)) * 100
                    
                    print(f"Routing net {i+1}/{len(nets)}: {net_name}")
                    
                    # Report routing start
                    if progress_callback:
                        debug_print(f"📍 Calling progress_callback for net {net_name}")
                        try:
                            progress_callback({
                                'current_net': net_name,
                                'progress': net_progress,
                                'stage': 'wavefront'
                            })
                        except Exception as e:
                            debug_print(f"❌ Progress callback error: {e}")
                    
                    # Route the net
                    debug_print(f"📍 Calling router.route_net for {net_name}...")
                    route_success = router.route_net(net)
                    debug_print(f"📍 Router returned: {route_success} for {net_name}")
                    
                    if route_success:
                        successful_nets.append(net)
                        debug_print(f"✅ Successfully routed {net_name}")
                        
                        # Report success
                        if progress_callback:
                            try:
                                progress_callback({
                                    'current_net': net_name,
                                    'progress': net_progress,
                                    'stage': 'complete',
                                    'success': True
                                })
                            except Exception as e:
                                debug_print(f"❌ Progress callback error: {e}")
                    else:
                        debug_print(f"❌ Failed to route {net_name}")
                        
                        # Report failure
                        if progress_callback:
                            try:
                                progress_callback({
                                    'current_net': net_name,
                                    'progress': net_progress,
                                    'stage': 'complete',
                                    'success': False
                                })
                            except Exception as e:
                                debug_print(f"❌ Progress callback error: {e}")
                                
                    debug_print(f"📍 Completed processing net {i+1}/{len(nets)}")
                    
                debug_print(f"📍 Route loop completed! Successfully routed: {len(successful_nets)}/{len(nets)} nets")
                    
            except Exception as routing_error:
                debug_print(f"❌ Exception in routing loop: {routing_error}")
                import traceback
                debug_print(f"❌ Full traceback: {traceback.format_exc()}")
            finally:
                # Always cleanup GPU resources
                try:
                    self._cleanup_gpu_resources()
                    debug_print("🧹 GPU resources cleaned up")
                except Exception as cleanup_error:
                    debug_print(f"❌ Cleanup error: {cleanup_error}")
        
        routing_time = time.time() - start_time
        print(f"Routing completed in {routing_time:.2f} seconds")
        
        return self._generate_results(successful_nets, routing_time)
    
    def _parse_nets(self, nets_data: List[Dict], debug_print=None) -> List[Net]:
        """Parse net data into Net objects"""
        if debug_print is None:
            debug_print = print
            
        nets = []
        
        debug_print(f"📝 Parsing {len(nets_data)} nets...")
        
        for i, net_data in enumerate(nets_data):
            # Show progress for every 5 nets
            if i > 0 and i % 5 == 0:
                debug_print(f"   ... processed {i}/{len(nets_data)} nets")
                
            valid_pins = []
            net_name = net_data.get('name', f"Net_{net_data.get('id', i)}")
            
            # Only show detailed processing for first 3 nets
            if i < 3:
                debug_print(f"   Processing net {i+1}: {net_name}")
            
            for j, pin_data in enumerate(net_data.get('pins', [])):
                try:
                    # Check if pin data has required fields
                    if 'x' not in pin_data or 'y' not in pin_data:
                        if i < 3:  # Only show errors for first few nets
                            debug_print(f"     ❌ Pin {j+1}: Missing x or y coordinate")
                        continue
                    
                    # Convert world coordinates to grid coordinates
                    x_nm = pin_data['x']
                    y_nm = pin_data['y']
                    layer = pin_data.get('layer', 0)
                    
                    grid_x, grid_y = self.grid.world_to_grid(x_nm, y_nm)
                    
                    # Only show detailed coordinate conversion for first net
                    if i == 0 and j < 3:
                        debug_print(f"     Pin {j+1}: ({x_nm/1e6:.2f}, {y_nm/1e6:.2f})mm → grid({grid_x}, {grid_y}) layer {layer}")
                    
                    # Check bounds
                    if (0 <= grid_x < self.grid.width and 
                        0 <= grid_y < self.grid.height and
                        0 <= layer < self.grid.layers):
                        valid_pins.append(Point3D(grid_x, grid_y, layer))
                        if i == 0 and j < 3:  # Only for first net
                            debug_print(f"     ✅ Pin {j+1} added to valid pins")
                    else:
                        if i < 3:  # Show bounds errors for first few nets
                            debug_print(f"     ❌ Pin {j+1} out of bounds: grid({grid_x}, {grid_y}) layer {layer}")
                            if i == 0:  # Show bounds info only once
                                debug_print(f"        Grid bounds: {self.grid.width}x{self.grid.height}, layers: {self.grid.layers}")
                        
                except Exception as e:
                    if i < 3:  # Only show parsing errors for first few nets
                        debug_print(f"     ❌ Pin {j+1} parsing error: {e}")
                    continue
            
            # Only create net if there are at least 2 valid pins
            if len(valid_pins) >= 2:
                net = Net(
                    id=net_data.get('id', i),
                    name=net_name,
                    pins=valid_pins,
                    width_nm=net_data.get('width_nm', 200000)
                )
                
                # Store KiCad net reference for track creation
                net.kicad_net = net_data.get('kicad_net', None)
                
                nets.append(net)
                if i < 5:  # Show creation for first 5 nets
                    debug_print(f"   ✅ Net '{net_name}' created with {len(valid_pins)} valid pins")
            else:
                if i < 5:  # Show skip message for first 5 nets
                    debug_print(f"   ⏭️ Net '{net_name}' skipped: only {len(valid_pins)} valid pins")
        
        debug_print(f"📊 Net parsing complete: {len(nets)} nets ready for routing")
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
            'nets': [],
            'tracks': []  # Add tracks to results for KiCad integration
        }
        
        # Add net details AND create tracks
        for net in successful_nets:
            path_world = []
            tracks_created = []
            
            if net.route_path:
                # Convert path to world coordinates
                for point in net.route_path:
                    world_x, world_y = self.grid.grid_to_world(point.x, point.y)
                    path_world.append({
                        'x': world_x,
                        'y': world_y,
                        'layer': point.z
                    })
                
                # CREATE ACTUAL KICAD TRACKS FROM PATH
                tracks_created = self._create_tracks_from_path(net, path_world)
            
            net_data = {
                'id': net.id,
                'name': net.name,
                'path': path_world,
                'via_count': net.via_count,
                'total_length_mm': net.total_length * self.grid.pitch_mm,
                'tracks_created': len(tracks_created)
            }
            result['nets'].append(net_data)
            result['tracks'].extend(tracks_created)
            
        return result
    
    def _create_tracks_from_path(self, net: Net, path_world: List[Dict]) -> List[Dict]:
        """Convert route path to KiCad tracks - THIS WAS THE MISSING PIECE!"""
        if not path_world or len(path_world) < 2:
            return []
        
        debug_print = getattr(self, 'debug_print', print)
        tracks_created = []
        
        try:
            # Import KiCad modules
            import pcbnew
            
            # Get board if available
            board = getattr(self, 'board', None)
            if not board:
                debug_print("❌ No board available for track creation")
                return []
            
            debug_print(f"🔨 Creating tracks for net {net.name} with {len(path_world)} points")
            
            # Create track segments between consecutive points
            for i in range(len(path_world) - 1):
                start_point = path_world[i]
                end_point = path_world[i + 1]
                
                # Skip if same point (shouldn't happen but be safe)
                if (start_point['x'] == end_point['x'] and 
                    start_point['y'] == end_point['y'] and 
                    start_point['layer'] == end_point['layer']):
                    continue
                
                # Create track segment
                track = pcbnew.PCB_TRACK(board)
                
                # Set start and end points (KiCad uses nanometers)
                track.SetStart(pcbnew.VECTOR2I(int(start_point['x']), int(start_point['y'])))
                track.SetEnd(pcbnew.VECTOR2I(int(end_point['x']), int(end_point['y'])))
                
                # Set layer
                layer_id = self._get_kicad_layer_id(start_point['layer'])
                track.SetLayer(layer_id)
                
                # Set net
                if hasattr(net, 'kicad_net') and net.kicad_net:
                    track.SetNet(net.kicad_net)
                
                # Set track width (default to net width or board default)
                track_width = getattr(net, 'width', 200000)  # 0.2mm default
                track.SetWidth(track_width)
                
                # Add to board
                board.Add(track)
                
                tracks_created.append({
                    'start': {'x': start_point['x'], 'y': start_point['y']},
                    'end': {'x': end_point['x'], 'y': end_point['y']},
                    'layer': start_point['layer'],
                    'net_id': net.id,
                    'net_name': net.name
                })
                
                debug_print(f"  ✅ Track segment {i+1}: ({start_point['x']/1e6:.2f}, {start_point['y']/1e6:.2f}) → ({end_point['x']/1e6:.2f}, {end_point['y']/1e6:.2f})")
            
            # Handle layer changes (vias)
            for i in range(len(path_world) - 1):
                start_point = path_world[i]
                end_point = path_world[i + 1]
                
                if start_point['layer'] != end_point['layer']:
                    # Create via
                    via = pcbnew.PCB_VIA(board)
                    via.SetPosition(pcbnew.VECTOR2I(int(end_point['x']), int(end_point['y'])))
                    
                    # Set via properties
                    via.SetViaType(pcbnew.VIATYPE_THROUGH)  # Through hole via
                    if hasattr(net, 'kicad_net') and net.kicad_net:
                        via.SetNet(net.kicad_net)
                    
                    board.Add(via)
                    debug_print(f"  🔗 Via added at ({end_point['x']/1e6:.2f}, {end_point['y']/1e6:.2f}) layer {start_point['layer']}→{end_point['layer']}")
            
            debug_print(f"✅ Created {len(tracks_created)} track segments for net {net.name}")
            return tracks_created
            
        except Exception as e:
            debug_print(f"❌ Error creating tracks for net {net.name}: {e}")
            import traceback
            debug_print(f"❌ Traceback: {traceback.format_exc()}")
            return []
    
    def _get_kicad_layer_id(self, internal_layer: int) -> int:
        """Convert internal layer number to KiCad layer ID"""
        try:
            import pcbnew
            
            # Map internal layers to KiCad standard layers
            layer_map = {
                0: pcbnew.F_Cu,     # Front copper
                1: pcbnew.B_Cu,     # Back copper
                2: pcbnew.In1_Cu,   # Inner layer 1
                3: pcbnew.In2_Cu,   # Inner layer 2
                4: pcbnew.In3_Cu,   # Inner layer 3
                5: pcbnew.In4_Cu,   # Inner layer 4
                # Add more layers as needed
            }
            
            return layer_map.get(internal_layer, pcbnew.F_Cu)  # Default to front copper
            
        except Exception:
            return 0  # Fallback to layer 0

    def _cleanup_gpu_resources(self):
        """Clean up GPU memory and resources"""
        if CUPY_AVAILABLE:
            try:
                # Clear GPU memory pool
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                
                # Synchronize to ensure all operations complete
                cp.cuda.Stream.null.synchronize()
                
                print("✅ GPU memory cleaned up successfully")
            except Exception as e:
                print(f"⚠️ GPU cleanup warning: {e}")
        
        # Clean up grid resources
        if hasattr(self, 'grid') and self.grid:
            try:
                if hasattr(self.grid, 'cleanup'):
                    self.grid.cleanup()
            except Exception as e:
                print(f"⚠️ Grid cleanup warning: {e}")
