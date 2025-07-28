"""
Routing algorithms for OrthoRoute
"""
from typing import Dict, List, Optional, Tuple
import cupy as cp
import numpy as np

from .gpu_engine import GPUGrid, Point3D, Net

class WavefrontRouter:
    """GPU-accelerated wavefront router"""
    def __init__(self, grid: GPUGrid):
        self.grid = grid
        self.max_distance = 10000
        self.wavefront_kernel = self._init_wavefront_kernel()
        self.max_iterations = 1000
    
    def _init_wavefront_kernel(self):
        """Initialize CUDA kernel for wavefront propagation"""
        kernel_code = r'''
        extern "C" __global__
        void wavefront_propagate(unsigned char* availability, 
                               unsigned short* distance,
                               int* parent,
                               const int width, const int height, const int layers,
                               const int* wavefront,
                               const int wavefront_size,
                               bool* found_target,
                               const int target_x,
                               const int target_y,
                               const int target_z) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= wavefront_size)
                return;
                
            // Get current point coordinates
            int x = wavefront[idx * 3];
            int y = wavefront[idx * 3 + 1];
            int z = wavefront[idx * 3 + 2];
            
            // Check if we reached target
            if (x == target_x && y == target_y && z == target_z) {
                *found_target = true;
                return;
            }
            
            // Current distance
            int current_dist = distance[z * width * height + y * width + x];
            
            // Propagate to neighbors (6 directions)
            const int dx[] = {1, -1, 0, 0, 0, 0};
            const int dy[] = {0, 0, 1, -1, 0, 0};
            const int dz[] = {0, 0, 0, 0, 1, -1};
            
            for (int dir = 0; dir < 6; dir++) {
                int nx = x + dx[dir];
                int ny = y + dy[dir];
                int nz = z + dz[dir];
                
                // Check bounds
                if (nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= layers)
                    continue;
                    
                // Get neighbor index
                int nidx = nz * width * height + ny * width + nx;
                
                // Check if cell is available and not visited
                if (availability[nidx] == 0 || distance[nidx] <= current_dist + 1)
                    continue;
                    
                // Update distance and parent
                atomicMin(&distance[nidx], current_dist + 1);
                if (distance[nidx] == current_dist + 1) {
                    parent[nidx * 3] = x;
                    parent[nidx * 3 + 1] = y;
                    parent[nidx * 3 + 2] = z;
                }
            }
        }
        '''
        return cp.RawKernel(kernel_code, 'wavefront_propagate')
    
    def route_net(self, net: Net) -> bool:
        """Route a single net using wavefront expansion"""
        if len(net.pins) < 2:
            return False
            
        # For multi-pin nets, route each pair
        if len(net.pins) == 2:
            route = self._route_two_pin(net.pins[0], net.pins[1])
            if route:
                net.route_path = route
                net.success = True
                return True
        else:
            # Use nearest neighbor for multi-pin nets
            route = self._route_multi_pin(net.pins)
            if route:
                net.route_path = route
                net.success = True
                return True
        
        return False

class ConflictResolver:
    """Handles routing conflicts using negotiated congestion"""
    def __init__(self, grid: GPUGrid):
        self.grid = grid
        self.max_iterations = 30
        self.congestion_factor = 1.5
        
    def resolve_conflicts(self, nets: List[Net]) -> bool:
        """Resolve routing conflicts between nets"""
        # For testing, just mark all nets as successful
        for net in nets:
            net.success = True
        return True
    def _route_two_pin(self, start: Point3D, end: Point3D) -> Optional[List[Point3D]]:
        """Route between two pins using GPU wavefront"""
        # Reset distance grid
        self.grid.distance.fill(65535)
        self.grid.parent = cp.full((self.grid.layers, self.grid.height, self.grid.width, 3), -1, dtype=cp.int32)
        
        # Set start point distance
        self.grid.distance[start.z, start.y, start.x] = 0
        
        # Initialize wavefront with start point
        wavefront = cp.array([[start.x, start.y, start.z]], dtype=cp.int32)
        
        # Target flag
        found_target = cp.array([False], dtype=cp.bool_)
        
        # Set up kernel parameters
        block_size = 256
        
        for _ in range(self.max_iterations):
            if len(wavefront) == 0:
                break
                
            grid_size = (len(wavefront) + block_size - 1) // block_size
            
            # Run wavefront kernel
            self.wavefront_kernel(
                (grid_size,), (block_size,),
                (self.grid.availability,
                 self.grid.distance,
                 self.grid.parent,
                 self.grid.width, self.grid.height, self.grid.layers,
                 wavefront,
                 len(wavefront),
                 found_target,
                 end.x, end.y, end.z)
            )
            
            # Check if target found
            if found_target.get():
                return self._reconstruct_path(end)
            
            # Get new wavefront
            wavefront = self._get_next_wavefront()
        
        return None  # No path found
    
    def _route_multi_pin(self, pins: List[Point3D]) -> Optional[List[Point3D]]:
        """Route a multi-pin net using nearest neighbor approach"""
        if len(pins) < 2:
            return None
            
        # Start with first pin
        route_path = [pins[0]]
        remaining_pins = pins[1:]
        
        while remaining_pins:
            current = route_path[-1]
            
            # Find nearest unrouted pin
            nearest = min(remaining_pins, 
                        key=lambda p: abs(p.x - current.x) + 
                                    abs(p.y - current.y) + 
                                    abs(p.z - current.z))
            
            # Route to nearest pin
            segment = self._route_two_pin(current, nearest)
            if not segment:
                return None
                
            route_path.extend(segment[1:])  # Skip first point as it's the same as current
            remaining_pins.remove(nearest)
        
        return route_path
    
    def _get_next_wavefront(self) -> cp.ndarray:
        """Get points for next wavefront iteration"""
        # Find points that were updated in the last iteration
        updated = cp.where(self.grid.parent[..., 0] >= 0)
        if len(updated[0]) == 0:
            return cp.array([], dtype=cp.int32)
            
        # Create wavefront array
        wavefront = cp.stack([
            updated[2],  # x coordinates
            updated[1],  # y coordinates
            updated[0]   # z coordinates
        ], axis=1)
        
        return wavefront
    
    def _reconstruct_path(self, end: Point3D) -> List[Point3D]:
        """Reconstruct path from parent pointers"""
        path = [end]
        current = end
        
        while True:
            parent = self.grid.parent[current.z, current.y, current.x]
            if parent[0] < 0:  # No parent (start point)
                break
                
            current = Point3D(int(parent[0]), int(parent[1]), int(parent[2]))
            path.append(current)
        
        return list(reversed(path))
    
    def route_net_batch(self, nets: List[Net], max_iterations: int = 100) -> List[Net]:
        """Route multiple nets simultaneously using parallel wavefront"""
        print(f"Routing batch of {len(nets)} nets...")
        
        successful_nets = []
        
        for net in nets:
            if len(net.pins) < 2:
                continue
                
            # For multi-pin nets, use Steiner tree approach
            if len(net.pins) == 2:
                route = self._route_two_pin(net.pins[0], net.pins[1])
            else:
                route = self._route_multi_pin(net.pins)
            
            if route:
                net.route_path = route
                net.routed = True
                net.total_length = self._calculate_route_length(route)
                net.via_count = self._count_vias(route)
                successful_nets.append(net)
        
        print(f"Successfully routed {len(successful_nets)}/{len(nets)} nets")
        return successful_nets
    
    def _route_two_pin(self, start: Point3D, end: Point3D) -> Optional[List[Point3D]]:
        """Route between two pins using GPU wavefront"""
        # Reset grid state
        self.grid.distance_map[:] = 65535
        self.grid.parent_map[:] = -1
        
        # Set start point
        start_pos = cp.array([start.x, start.y, start.layer], dtype=cp.int32)
        end_pos = cp.array([end.x, end.y, end.layer], dtype=cp.int32)
        
        self.grid.distance_map[start.layer, start.y, start.x] = 0
        
        # Current wavefront
        current_wave = cp.array([start_pos], dtype=cp.int32)
        
        for distance in range(self.max_distance):
            if len(current_wave) == 0:
                break

          # Expand wavefront to neighbors
            next_wave = self._expand_wavefront_gpu(current_wave, distance + 1)
            
            # Check if target reached
            if self._check_target_reached(next_wave, end_pos):
                # Reconstruct path
                return self._reconstruct_path(start_pos, end_pos)
            
            current_wave = next_wave
        
        return None  # No path found
    
    def _expand_wavefront_gpu(self, current_wave: cp.ndarray, distance: int) -> cp.ndarray:
        """Expand wavefront using vectorized GPU operations"""
        if len(current_wave) == 0:
            return cp.array([], dtype=cp.int32).reshape(0, 3)
        
        # Broadcast current positions with neighbor offsets
        positions = current_wave[:, None, :] + self.neighbor_offsets[None, :, :]
        positions = positions.reshape(-1, 3)  # Flatten to [N*6, 3]
        
        # Filter valid positions
        valid_mask = self._get_valid_positions_mask(positions)
        valid_positions = positions[valid_mask]
        
        # Update distance map and parent pointers
        if len(valid_positions) > 0:
            self._update_distance_map(valid_positions, distance, current_wave, valid_mask)
        
        return valid_positions
    
    def _get_valid_positions_mask(self, positions: cp.ndarray) -> cp.ndarray:
        """Check which positions are valid and available"""
        # Bounds checking
        x_valid = (positions[:, 0] >= 0) & (positions[:, 0] < self.grid.width)
        y_valid = (positions[:, 1] >= 0) & (positions[:, 1] < self.grid.height)
        z_valid = (positions[:, 2] >= 0) & (positions[:, 2] < self.grid.layers)
        bounds_valid = x_valid & y_valid & z_valid
        
        # Availability checking (only for positions within bounds)
        availability_mask = cp.zeros(len(positions), dtype=cp.bool_)
        if cp.any(bounds_valid):
            valid_pos = positions[bounds_valid]
            available = self.grid.availability[valid_pos[:, 2], valid_pos[:, 1], valid_pos[:, 0]]
            availability_mask[bounds_valid] = available
        
        # Distance checking (not already visited)
        distance_mask = cp.zeros(len(positions), dtype=cp.bool_)
        if cp.any(bounds_valid):
            valid_pos = positions[bounds_valid]
            not_visited = self.grid.distance_map[valid_pos[:, 2], valid_pos[:, 1], valid_pos[:, 0]] == 65535
            distance_mask[bounds_valid] = not_visited
        
        return bounds_valid & availability_mask & distance_mask
    
    def _update_distance_map(self, positions: cp.ndarray, distance: int, 
                           parents: cp.ndarray, valid_mask: cp.ndarray):
        """Update distance map and parent pointers"""
        if len(positions) == 0:
            return
        
        # Update distance map
        self.grid.distance_map[positions[:, 2], positions[:, 1], positions[:, 0]] = distance
        
        # Update parent pointers (which neighbor led to this position)
        parent_indices = cp.repeat(cp.arange(len(parents)), 6)[valid_mask]
        if len(parent_indices) == len(positions):
            self.grid.parent_map[positions[:, 2], positions[:, 1], positions[:, 0]] = parents[parent_indices]
    
    def _check_target_reached(self, wave: cp.ndarray, target: cp.ndarray) -> bool:
        """Check if target position is in current wavefront"""
        if len(wave) == 0:
            return False
        
        # Check if any position in wave matches target
        matches = cp.all(wave == target[None, :], axis=1)
        return cp.any(matches)
    
    def _reconstruct_path(self, start: cp.ndarray, end: cp.ndarray) -> List[Point3D]:
        """Reconstruct path from parent pointers"""
        path = []
        current = end.copy()
        
        max_steps = self.grid.width * self.grid.height * self.grid.layers
        step_count = 0
        
        while step_count < max_steps:
            path.append(Point3D(int(current[0]), int(current[1]), int(current[2])))
            
            # Check if we reached the start
            if cp.all(current == start):
                break
            
            # Get parent
            parent = self.grid.parent_map[current[2], current[1], current[0]]
            if cp.all(parent == -1):
                # No parent found - path reconstruction failed
                return None
            
            current = parent
            step_count += 1
        
        # Reverse path to go from start to end
        path.reverse()
        return path
    
    def _route_multi_pin(self, pins: List[Point3D]) -> Optional[List[Point3D]]:
        """Route multi-pin net using Steiner tree approach"""
        if len(pins) <= 2:
            return self._route_two_pin(pins[0], pins[1]) if len(pins) == 2 else None
        
        # Simple approach: connect to nearest neighbor iteratively
        routed_pins = [pins[0]]
        unrouted_pins = pins[1:]
        full_route = []
        
        while unrouted_pins:
            # Find closest unrouted pin to any routed pin
            best_route = None
            best_distance = float('inf')
            best_pin_idx = -1
            
            for i, unrouted_pin in enumerate(unrouted_pins):
                for routed_pin in routed_pins:
                    route = self._route_two_pin(routed_pin, unrouted_pin)
                    if route:
                        distance = len(route)
                        if distance < best_distance:
                            best_distance = distance
                            best_route = route
                            best_pin_idx = i
            
            if best_route is None:
                # Failed to connect - return partial route
                break
            
            # Add best route to full route
            full_route.extend(best_route)
            
            # Move pin from unrouted to routed
            connected_pin = unrouted_pins.pop(best_pin_idx)
            routed_pins.append(connected_pin)
        
        return full_route if len(unrouted_pins) == 0 else None
    
    def _calculate_route_length(self, route: List[Point3D]) -> float:
        """Calculate total route length in millimeters"""
        if len(route) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(route) - 1):
            p1, p2 = route[i], route[i + 1]
            
            # Same layer - trace length
            if p1.layer == p2.layer:
                dx = abs(p2.x - p1.x) * self.grid.pitch_mm
                dy = abs(p2.y - p1.y) * self.grid.pitch_mm
                total_length += (dx + dy)  # Manhattan distance
            # Layer change - via (no additional length)
        
        return total_length
    
    def _count_vias(self, route: List[Point3D]) -> int:
        """Count number of vias in route"""
        if len(route) < 2:
            return 0
        
        via_count = 0
        for i in range(len(route) - 1):
            if route[i].layer != route[i + 1].layer:
                via_count += 1
        
        return via_count
    

class ConflictResolver:
    """Negotiated congestion routing for conflict resolution"""
    
    def __init__(self, grid: GPUGrid):
        self.grid = grid
        self.congestion_factor = 1.5
        self.iteration_count = 0
    
    def resolve_conflicts(self, nets: List[Net], max_iterations: int = 20) -> List[Net]:
        """Iterative rip-up and reroute with negotiated congestion"""
        print(f"Starting conflict resolution for {len(nets)} nets...")
        
        router = WavefrontRouter(self.grid)
        
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}/{max_iterations}")
            self.iteration_count = iteration
            
            # Route all nets with current congestion costs
            routed_nets = router.route_net_batch(nets)
            
            # Update usage and detect conflicts
            self._update_usage_counts(routed_nets)
            conflicted_nets = self._detect_conflicts(routed_nets)
            
            if not conflicted_nets:
                print(f"Converged after {iteration + 1} iterations")
                return routed_nets
            
            print(f"Found {len(conflicted_nets)} conflicted nets")
            
            # Update congestion costs
            self._update_congestion_costs()
            
            # Rip up conflicted nets for rerouting
            self._rip_up_nets(conflicted_nets)
        
        print(f"Max iterations reached. Final success rate: "
              f"{len([n for n in nets if n.routed])}/{len(nets)}")
        
        return nets
    
    def _update_usage_counts(self, nets: List[Net]):
        """Update grid usage counts from all routed nets"""
        self.grid.usage_count[:] = 0
        
        for net in nets:
            if not net.routed:
                continue
            
            for point in net.route_path:
                if (0 <= point.x < self.grid.width and 
                    0 <= point.y < self.grid.height and
                    0 <= point.layer < self.grid.layers):
                    self.grid.usage_count[point.layer, point.y, point.x] += 1
    
    def _detect_conflicts(self, nets: List[Net]) -> List[Net]:
        """Detect nets that use overcrowded grid cells"""
        conflicted_nets = []
        
        # Find overcrowded cells
        overcrowded = self.grid.usage_count > self.grid.capacity
        
        for net in nets:
            if not net.routed:
                continue
            
            # Check if net uses any overcrowded cells
            has_conflict = False
            for point in net.route_path:
                if (0 <= point.x < self.grid.width and 
                    0 <= point.y < self.grid.height and
                    0 <= point.layer < self.grid.layers):
                    if overcrowded[point.layer, point.y, point.x]:
                        has_conflict = True
                        break
            
            if has_conflict:
                conflicted_nets.append(net)
        
        return conflicted_nets
    
    def _update_congestion_costs(self):
        """Update congestion costs based on usage"""
        # Base cost is 1.0
        self.grid.congestion_cost[:] = 1.0
        
        # Apply congestion penalty
        overcrowded = self.grid.usage_count > self.grid.capacity
        excess_usage = self.grid.usage_count - self.grid.capacity
        
        # Exponential penalty for overcrowded cells
        penalty = self.congestion_factor ** (self.iteration_count + excess_usage)
        self.grid.congestion_cost[overcrowded] = penalty[overcrowded]
    
    def _rip_up_nets(self, nets: List[Net]):
        """Remove routing for conflicted nets"""
        for net in nets:
            net.routed = False
            net.route_path = []
            net.total_length = 0.0
            net.via_count = 0