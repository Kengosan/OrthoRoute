"""
GPU-accelerated routing algorithms for OrthoRoute
"""
import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from .gpu_engine import Point3D, Net, GPUGrid

def init_wavefront_kernel():
    """Initialize CUDA kernel for wavefront propagation"""
    return cp.RawKernel(r'''
    extern "C" __global__
    void wavefront_propagate(const unsigned char* availability,
                           unsigned int* distance,
                           int* parent,
                           const int width, const int height, const int layers,
                           const int* active_points,
                           const int active_count,
                           const int* target,
                           const float congestion_factor,
                           bool* found_target) {
        // Get thread index
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= active_count)
            return;
            
        // Get current point coordinates
        const int x = active_points[idx * 3];
        const int y = active_points[idx * 3 + 1];
        const int z = active_points[idx * 3 + 2];
        
        // Check bounds
        if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= layers)
            return;
            
        // Get current index and distance
        const int curr_idx = z * width * height + y * width + x;
        const unsigned int current_dist = distance[curr_idx];
        
            // Target reached?
        const int target_idx = target[2] * width * height + target[1] * width + target[0];
        if (curr_idx == target_idx) {
            atomicExch((int*)found_target, 1);
            return;
        }
        
        // Neighbor offsets (6-way connectivity)
        const int dx[] = {1, -1, 0, 0, 0, 0};
        const int dy[] = {0, 0, 1, -1, 0, 0};
        const int dz[] = {0, 0, 0, 0, 1, -1};
        const float base_costs[] = {1.0f, 1.0f, 1.0f, 1.0f, 3.0f, 3.0f};  // Higher cost for vias        // Check all neighbors
        for (int dir = 0; dir < 6; dir++) {
            const int nx = x + dx[dir];
            const int ny = y + dy[dir];
            const int nz = z + dz[dir];
            
            // Check bounds
            if (nx < 0 || nx >= width || 
                ny < 0 || ny >= height || 
                nz < 0 || nz >= layers)
                continue;
                
            // Get neighbor index
            const int n_idx = nz * width * height + ny * width + nx;
            
            // Check if neighbor is available
            if (availability[n_idx] == 0)
                continue;
                
            // Calculate cost to move to neighbor
            // Base cost + congestion penalty
            const float move_cost = base_costs[dir] * 
                (1.0f + (congestion_factor * __uint2float_rn(availability[n_idx])));
            
            // Total cost to reach neighbor
            const unsigned int total_cost = current_dist + (unsigned int)move_cost;
            
            // Update distance if new path is shorter
            if (total_cost < distance[n_idx]) {
                atomicMin(&distance[n_idx], total_cost);
                
                // If we won the race to update this cell, set parent
                if (distance[n_idx] == total_cost) {
                    parent[n_idx * 3] = x;
                    parent[n_idx * 3 + 1] = y;
                    parent[n_idx * 3 + 2] = z;
                }
            }
        }
    }
    ''', 'wavefront_propagate')

class WaveRouter:
    """GPU-accelerated wave propagation router"""
    def __init__(self, grid: GPUGrid):
        self.grid = grid
        self.wave_kernel = init_wavefront_kernel()
        self.max_iterations = 1000
        self.congestion_factor = 0.5
        
        # Work arrays are now shared with grid
        self.distance = grid.distance
        self.parent = grid.parent
        
    def route_net(self, net: Net) -> bool:
        """Route a single net using wave propagation"""
        try:
            if len(net.pins) < 2:
                return False
                
            # Validate pin coordinates before routing
            width, height, layers = self.grid.width, self.grid.height, self.grid.layers
            valid_pins = []
            
            for pin in net.pins:
                if (0 <= pin.x < width and 0 <= pin.y < height and 0 <= pin.z < layers):
                    valid_pins.append(pin)
                else:
                    print(f"⚠️ Warning: Pin at ({pin.x}, {pin.y}, {pin.z}) is outside grid bounds ({width}, {height}, {layers})")
            
            if len(valid_pins) < 2:
                print(f"⚠️ Warning: Net {net.name} has fewer than 2 valid pins after bounds check")
                return False
                
            # For multi-pin nets, route pin-by-pin
            route_points = []
            remaining_pins = valid_pins[1:]
            current = valid_pins[0]
            
            while remaining_pins:
                # Find nearest unrouted pin
                nearest = min(remaining_pins, 
                            key=lambda p: p.distance_to(current))
                
                # Route to nearest pin
                path = self._route_two_points(current, nearest)
                if not path:
                    return False
                    
                # Add path to route (skip first point if not first segment)
                if route_points:
                    path = path[1:]  # Skip first point to avoid duplicates
                route_points.extend(path)
                
                # Update for next iteration
                current = nearest
                remaining_pins.remove(nearest)
            
            # Store route
            net.route_path = route_points
            net.success = True
            net.routed = True
            net.via_count = sum(1 for i in range(len(route_points)-1) 
                               if route_points[i].z != route_points[i+1].z)
            net.total_length = len(route_points)
            
            # Update grid usage with bounds checking
            for point in route_points:
                if (0 <= point.x < width and 0 <= point.y < height and 0 <= point.z < layers):
                    self.grid.usage_count[point.z, point.y, point.x] += 1
                    if self.grid.usage_count[point.z, point.y, point.x] > 3:  # Congestion threshold
                        self.grid.availability[point.z, point.y, point.x] = 0
            
            return True
            
        except Exception as e:
            print(f"❌ Error routing net {net.name}: {e}")
            return False
    
    def _route_two_points(self, start: Point3D, end: Point3D) -> Optional[List[Point3D]]:
        """Route between two points using wave propagation"""
        # Validate points are within bounds
        width, height, layers = self.grid.width, self.grid.height, self.grid.layers
        
        if (not (0 <= start.x < width and 0 <= start.y < height and 0 <= start.z < layers) or
            not (0 <= end.x < width and 0 <= end.y < height and 0 <= end.z < layers)):
            print(f"⚠️ Error: Route points out of bounds: Start({start.x}, {start.y}, {start.z}), End({end.x}, {end.y}, {end.z})")
            print(f"Grid dimensions: ({width}, {height}, {layers})")
            return None
        
        # Reset arrays
        self.distance.fill(0xFFFFFFFF)
        self.parent.fill(-1)
        
        # Set start point
        self.distance[start.z, start.y, start.x] = 0
        
        # Initialize active points with start
        active_points = cp.array([[start.x, start.y, start.z]], dtype=cp.int32)
        target = cp.array([end.x, end.y, end.z], dtype=cp.int32)
        found_target = cp.array([False], dtype=cp.bool_)
        
        # CUDA grid configuration
        block_size = 256
        
        # Main routing loop
        for _ in range(self.max_iterations):
            if len(active_points) == 0:
                break
                
            grid_size = (len(active_points) + block_size - 1) // block_size
            
            # Run wave propagation kernel
            try:
                self.wave_kernel(
                    (grid_size,),
                    (block_size,),
                    (self.grid.availability,
                     self.distance,
                     self.parent,
                     self.grid.width,
                     self.grid.height,
                     self.grid.layers,
                     active_points,
                     len(active_points),
                     target,
                     self.congestion_factor,
                     found_target)
                )
            except Exception as e:
                print(f"❌ Error running wave propagation kernel: {e}")
                print(f"Grid size: {self.grid.width}x{self.grid.height}x{self.grid.layers}")
                print(f"Active points: {len(active_points)}")
                print(f"Target: ({target[0]}, {target[1]}, {target[2]})")
                return None
            
            # Check if target found
            if found_target.get():
                return self._reconstruct_path(end)
            
            # Get points for next iteration
            active_points = self._get_active_points()
        
        return None  # No path found
    
    def _get_active_points(self) -> cp.ndarray:
        """Get points that were updated in the last iteration"""
        # Find points with finite distances
        valid_points = cp.where(self.distance < 0xFFFFFFFF)
        
        if len(valid_points[0]) == 0:
            return cp.array([], dtype=cp.int32)
        
        # Stack coordinates into (N, 3) array
        points = cp.stack([
            valid_points[2],  # x coordinates
            valid_points[1],  # y coordinates
            valid_points[0]   # z coordinates
        ], axis=1)
        
        return points
    
    def _reconstruct_path(self, end: Point3D) -> List[Point3D]:
        """Reconstruct path from parent pointers"""
        path = [end]
        current = end
        
        while True:
            parent_x = self.parent[current.z, current.y, current.x, 0]
            if parent_x < 0:  # No parent (reached start)
                break
                
            current = Point3D(
                x=int(parent_x),
                y=int(self.parent[current.z, current.y, current.x, 1]),
                z=int(self.parent[current.z, current.y, current.x, 2])
            )
            path.append(current)
        
        return list(reversed(path))  # Return path from start to end
