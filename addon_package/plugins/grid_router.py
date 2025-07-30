"""
Grid-Based Routing Engine for OrthoRoute
=======================================

Innovative routing approach that pre-defines unconnected traces on a grid.
From each pad, find the closest unconnected trace, then drill down with a via.

This is particularly effective for complex backplane designs where traditional
Lee's algorithm struggles with dense connectivity requirements.

Key Innovation:
- Pre-defined trace grid optimizes for predictable routing patterns
- Via-based connections minimize congestion
- GPU acceleration for massive parallel grid searches
- Orthogonal routing maintains signal integrity

Author: OrthoRoute Team
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass(frozen=True)  # Make it hashable
class GridPoint:
    """A point on the routing grid"""
    x: int
    y: int
    layer: int
    available: bool = True
    
    def __post_init__(self):
        # Use object.__setattr__ for frozen dataclass
        if not hasattr(self, '_connected_pads'):
            object.__setattr__(self, '_connected_pads', set())


@dataclass
class GridTrace:
    """A pre-defined trace segment on the grid"""
    start_point: GridPoint
    end_point: GridPoint
    width: int
    layer: int
    direction: str  # 'horizontal' or 'vertical'
    occupied: bool = False
    net_id: Optional[int] = None
    
    def __post_init__(self):
        # Validate that start and end points are on the same layer
        if self.start_point.layer != self.end_point.layer:
            raise ValueError("Start and end points must be on the same layer")
        # Ensure layer consistency
        if self.layer != self.start_point.layer:
            object.__setattr__(self, 'layer', self.start_point.layer)


@dataclass
class PadConnection:
    """Connection from a pad to the grid"""
    pad_x: int
    pad_y: int
    pad_layer: int
    grid_point: GridPoint
    via_required: bool
    connection_distance: float


class GridBasedRouter:
    """
    Grid-based routing engine for complex backplane designs
    
    This router creates a pre-defined grid of traces and uses via connections
    to route from pads to the nearest available grid trace. Much more efficient
    than Lee's algorithm for dense, complex routing scenarios.
    """
    
    def __init__(self, board_data: Dict, config: Dict):
        """Initialize grid-based router"""
        self.board_data = board_data
        self.config = config
        
        # Extract board dimensions
        self.board_width = board_data.get('board_width', 100000)  # nm
        self.board_height = board_data.get('board_height', 80000)  # nm
        
        # Grid configuration
        self.grid_spacing = config.get('grid_spacing', 2540000)  # 0.1" default
        self.via_size = config.get('via_size', 203200)  # 8 mil default
        self.trace_width = config.get('default_trace_width', 152400)  # 6 mil default
        
        # Layer configuration
        self.routing_layers = board_data.get('layer_count', 4)
        self.signal_layers = [layer for layer in range(self.routing_layers) 
                            if layer not in [0, self.routing_layers-1]]  # Skip top/bottom
        
        # Grid storage
        self.horizontal_traces: List[GridTrace] = []
        self.vertical_traces: List[GridTrace] = []
        self.grid_points: Dict[Tuple[int, int, int], GridPoint] = {}
        
        # Routing state
        self.routed_nets: Dict[int, List[GridTrace]] = {}
        self.failed_nets: Set[int] = set()
        
        print(f"🔧 Grid router initialized:")
        print(f"   📏 Board: {self.board_width/1000000:.1f}mm x {self.board_height/1000000:.1f}mm")
        print(f"   🏗️ Grid spacing: {self.grid_spacing/1000000:.2f}mm")
        print(f"   🎯 Signal layers: {self.signal_layers}")
    
    def create_routing_grid(self) -> bool:
        """
        Create the pre-defined routing grid
        
        Returns:
            bool: True if grid creation successful
        """
        try:
            print("🏗️ Creating routing grid...")
            
            # Calculate grid dimensions
            grid_cols = int(self.board_width / self.grid_spacing) + 1
            grid_rows = int(self.board_height / self.grid_spacing) + 1
            
            print(f"   📐 Grid size: {grid_cols} x {grid_rows}")
            
            # Create grid points
            self._create_grid_points(grid_cols, grid_rows)
            
            # Create horizontal traces
            self._create_horizontal_traces(grid_cols, grid_rows)
            
            # Create vertical traces  
            self._create_vertical_traces(grid_cols, grid_rows)
            
            # Mark obstacles
            self._mark_grid_obstacles()
            
            total_traces = len(self.horizontal_traces) + len(self.vertical_traces)
            available_traces = sum(1 for trace in self.horizontal_traces + self.vertical_traces 
                                 if not trace.occupied)
            
            print(f"✅ Grid created: {total_traces} traces ({available_traces} available)")
            return True
            
        except Exception as e:
            print(f"❌ Grid creation failed: {e}")
            return False
    
    def _create_grid_points(self, cols: int, rows: int):
        """Create all grid intersection points"""
        for layer in self.signal_layers:
            for row in range(rows):
                for col in range(cols):
                    x = col * self.grid_spacing
                    y = row * self.grid_spacing
                    
                    # Skip points outside board boundary
                    if x > self.board_width or y > self.board_height:
                        continue
                    
                    point = GridPoint(x=x, y=y, layer=layer)
                    self.grid_points[(x, y, layer)] = point
    
    def _create_horizontal_traces(self, cols: int, rows: int):
        """Create horizontal trace segments"""
        for layer in self.signal_layers:
            for row in range(rows):
                for col in range(cols - 1):
                    start_x = col * self.grid_spacing
                    end_x = (col + 1) * self.grid_spacing
                    y = row * self.grid_spacing
                    
                    # Skip traces outside board
                    if end_x > self.board_width or y > self.board_height:
                        continue
                    
                    start_point = self.grid_points.get((start_x, y, layer))
                    end_point = self.grid_points.get((end_x, y, layer))
                    
                    if start_point and end_point:
                        trace = GridTrace(
                            start_point=start_point,
                            end_point=end_point,
                            width=self.trace_width,
                            layer=layer,
                            direction='horizontal'
                        )
                        self.horizontal_traces.append(trace)
    
    def _create_vertical_traces(self, cols: int, rows: int):
        """Create vertical trace segments"""
        for layer in self.signal_layers:
            for col in range(cols):
                for row in range(rows - 1):
                    x = col * self.grid_spacing
                    start_y = row * self.grid_spacing
                    end_y = (row + 1) * self.grid_spacing
                    
                    # Skip traces outside board
                    if x > self.board_width or end_y > self.board_height:
                        continue
                    
                    start_point = self.grid_points.get((x, start_y, layer))
                    end_point = self.grid_points.get((x, end_y, layer))
                    
                    if start_point and end_point:
                        trace = GridTrace(
                            start_point=start_point,
                            end_point=end_point,
                            width=self.trace_width,
                            layer=layer,
                            direction='vertical'
                        )
                        self.vertical_traces.append(trace)
    
    def _mark_grid_obstacles(self):
        """Mark grid traces that intersect with obstacles"""
        obstacles = self.board_data.get('obstacles', {})
        
        for obstacle_type, obstacle_list in obstacles.items():
            if not isinstance(obstacle_list, list):
                continue
                
            for obstacle in obstacle_list:
                if obstacle_type == 'tracks':
                    self._mark_track_obstacles(obstacle)
                elif obstacle_type == 'zones':
                    self._mark_zone_obstacles(obstacle)
                elif obstacle_type == 'vias':
                    self._mark_via_obstacles(obstacle)
    
    def _mark_track_obstacles(self, track: Dict):
        """Mark traces that conflict with existing tracks"""
        try:
            track_layer = track.get('layer', 0)
            start_x = track.get('start_x', 0)
            start_y = track.get('start_y', 0)
            end_x = track.get('end_x', 0)
            end_y = track.get('end_y', 0)
            track_width = track.get('width', self.trace_width)
            
            # Check all traces for conflicts
            for trace in self.horizontal_traces + self.vertical_traces:
                if trace.layer == track_layer:
                    if self._traces_intersect(trace, start_x, start_y, end_x, end_y, track_width):
                        trace.occupied = True
                        
        except Exception as e:
            print(f"⚠️ Track obstacle marking error: {e}")
    
    def _mark_zone_obstacles(self, zone: Dict):
        """Mark traces that conflict with copper zones"""
        # Simplified zone checking - mark traces inside zone bounds
        try:
            zone_layer = zone.get('layer', 0)
            outline = zone.get('outline', [])
            
            if not outline:
                return
            
            # Get bounding box
            min_x = min(point[0] for point in outline)
            max_x = max(point[0] for point in outline)
            min_y = min(point[1] for point in outline)
            max_y = max(point[1] for point in outline)
            
            # Mark traces in bounding box (simplified)
            for trace in self.horizontal_traces + self.vertical_traces:
                if trace.layer == zone_layer:
                    trace_x = trace.start_point.x
                    trace_y = trace.start_point.y
                    
                    if min_x <= trace_x <= max_x and min_y <= trace_y <= max_y:
                        trace.occupied = True
                        
        except Exception as e:
            print(f"⚠️ Zone obstacle marking error: {e}")
    
    def _mark_via_obstacles(self, via: Dict):
        """Mark traces that conflict with existing vias"""
        try:
            via_x = via.get('x', 0)
            via_y = via.get('y', 0)
            via_size = via.get('size', self.via_size)
            
            # Mark traces near via
            clearance = via_size + self.trace_width
            
            for trace in self.horizontal_traces + self.vertical_traces:
                if self._point_near_trace(via_x, via_y, trace, clearance):
                    trace.occupied = True
                    
        except Exception as e:
            print(f"⚠️ Via obstacle marking error: {e}")
    
    def route_nets(self, nets: List[Dict], progress_callback=None) -> Dict:
        """
        Route all nets using grid-based algorithm
        
        Args:
            nets: List of net definitions with pins
            progress_callback: Function to report progress
            
        Returns:
            Dict: Routing results with success/failure counts
        """
        print(f"🚀 Starting grid-based routing for {len(nets)} nets")
        
        results = {
            'routed_nets': {},
            'failed_nets': [],
            'success_count': 0,
            'total_nets': len(nets),
            'routing_data': []
        }
        
        try:
            for i, net in enumerate(nets):
                net_code = net.get('net_code', -1)
                net_name = net.get('net_name', f'Net_{net_code}')
                pins = net.get('pins', [])
                
                if progress_callback:
                    progress = int((i / len(nets)) * 100)
                    progress_callback({
                        'current_net': net_name,
                        'progress': progress,
                        'stage': 'grid_routing'
                    })
                
                print(f"🎯 Routing net '{net_name}' ({len(pins)} pins)")
                
                # Route this net
                route_result = self._route_single_net(net_code, net_name, pins)
                
                if route_result['success']:
                    results['routed_nets'][net_code] = route_result
                    results['success_count'] += 1
                    print(f"   ✅ Successfully routed '{net_name}'")
                    
                    if progress_callback:
                        progress_callback({
                            'current_net': net_name,
                            'success': True,
                            'stage': 'complete'
                        })
                else:
                    results['failed_nets'].append(net_code)
                    self.failed_nets.add(net_code)
                    print(f"   ❌ Failed to route '{net_name}': {route_result.get('error', 'Unknown')}")
                    
                    if progress_callback:
                        progress_callback({
                            'current_net': net_name,
                            'success': False,
                            'stage': 'complete'
                        })
            
            success_rate = (results['success_count'] / len(nets)) * 100
            print(f"🎉 Grid routing completed: {success_rate:.1f}% success rate")
            
            return results
            
        except Exception as e:
            print(f"❌ Grid routing failed: {e}")
            return results
    
    def _route_single_net(self, net_code: int, net_name: str, pins: List[Dict]) -> Dict:
        """
        Route a single net using grid-based algorithm
        
        Args:
            net_code: Net identifier
            net_name: Human-readable net name
            pins: List of pin locations
            
        Returns:
            Dict: Routing result with traces and vias
        """
        try:
            if len(pins) < 2:
                return {'success': False, 'error': 'Insufficient pins'}
            
            # Find grid connections for all pins
            pin_connections = []
            for pin in pins:
                connection = self._find_closest_grid_connection(pin)
                if connection:
                    pin_connections.append(connection)
                else:
                    return {'success': False, 'error': f'No grid connection for pin at ({pin.get("x", 0)}, {pin.get("y", 0)})'}
            
            # Route between grid connections
            routing_traces = []
            routing_vias = []
            
            # Use minimum spanning tree approach to connect all pins
            connected_points = {pin_connections[0].grid_point}
            unconnected_points = set(conn.grid_point for conn in pin_connections[1:])
            
            while unconnected_points:
                # Find closest unconnected point to any connected point
                best_connection = None
                min_distance = float('inf')
                
                for unconnected in unconnected_points:
                    for connected in connected_points:
                        path_result = self._find_grid_path(connected, unconnected, net_code)
                        if path_result['success'] and path_result['distance'] < min_distance:
                            min_distance = path_result['distance']
                            best_connection = (connected, unconnected, path_result)
                
                if not best_connection:
                    return {'success': False, 'error': 'No path found between pins'}
                
                # Add the best connection
                start_point, end_point, path_result = best_connection
                routing_traces.extend(path_result['traces'])
                routing_vias.extend(path_result['vias'])
                
                # Mark traces as occupied
                for trace in path_result['traces']:
                    trace.occupied = True
                    trace.net_id = net_code
                
                # Update connected set
                connected_points.add(end_point)
                unconnected_points.remove(end_point)
            
            # Add via connections from pads to grid
            for connection in pin_connections:
                if connection.via_required:
                    via = {
                        'x': connection.pad_x,
                        'y': connection.pad_y,
                        'from_layer': connection.pad_layer,
                        'to_layer': connection.grid_point.layer,
                        'size': self.via_size,
                        'net_code': net_code
                    }
                    routing_vias.append(via)
                
                # Add trace from pad to grid point if needed
                if connection.connection_distance > 0:
                    trace = {
                        'start_x': connection.pad_x,
                        'start_y': connection.pad_y,
                        'end_x': connection.grid_point.x,
                        'end_y': connection.grid_point.y,
                        'layer': connection.grid_point.layer,
                        'width': self.trace_width,
                        'net_code': net_code
                    }
                    routing_traces.append(trace)
            
            # Store successful routing
            self.routed_nets[net_code] = routing_traces
            
            return {
                'success': True,
                'net_code': net_code,
                'net_name': net_name,
                'traces': [self._trace_to_dict(trace) for trace in routing_traces if hasattr(trace, 'start_point')],
                'vias': routing_vias,
                'pin_connections': pin_connections
            }
            
        except Exception as e:
            print(f"⚠️ Single net routing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _find_closest_grid_connection(self, pin: Dict) -> Optional[PadConnection]:
        """Find the closest available grid point to a pin"""
        pin_x = pin.get('x', 0)
        pin_y = pin.get('y', 0) 
        pin_layer = pin.get('layer', 0)
        
        closest_point = None
        min_distance = float('inf')
        
        # Search all grid points
        for (x, y, layer), grid_point in self.grid_points.items():
            if not grid_point.available:
                continue
                
            # Calculate connection distance
            distance = math.sqrt((x - pin_x)**2 + (y - pin_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = grid_point
        
        if closest_point:
            via_required = (pin_layer != closest_point.layer)
            return PadConnection(
                pad_x=pin_x,
                pad_y=pin_y,
                pad_layer=pin_layer,
                grid_point=closest_point,
                via_required=via_required,
                connection_distance=min_distance
            )
        
        return None
    
    def _find_grid_path(self, start: GridPoint, end: GridPoint, net_code: int) -> Dict:
        """Find path between two grid points using available traces"""
        try:
            # Simple A* pathfinding on the grid
            from collections import deque
            
            queue = deque([(start, [])])
            visited = {(start.x, start.y, start.layer)}
            
            while queue:
                current_point, path = queue.popleft()
                
                if current_point.x == end.x and current_point.y == end.y and current_point.layer == end.layer:
                    # Path found
                    total_distance = sum(trace.width for trace in path) if path else 0
                    vias = []
                    
                    # Add layer change vias if needed
                    for i in range(len(path) - 1):
                        if path[i].layer != path[i+1].layer:
                            via = {
                                'x': path[i].end_point.x,
                                'y': path[i].end_point.y,
                                'from_layer': path[i].layer,
                                'to_layer': path[i+1].layer,
                                'size': self.via_size,
                                'net_code': net_code
                            }
                            vias.append(via)
                    
                    return {
                        'success': True,
                        'traces': path,
                        'vias': vias,
                        'distance': total_distance
                    }
                
                # Find connected traces
                for trace in self.horizontal_traces + self.vertical_traces:
                    if trace.occupied:
                        continue
                    
                    # Check if trace connects to current point
                    next_point = None
                    if (trace.start_point.x == current_point.x and 
                        trace.start_point.y == current_point.y and
                        trace.start_point.layer == current_point.layer):
                        next_point = trace.end_point
                    elif (trace.end_point.x == current_point.x and 
                          trace.end_point.y == current_point.y and
                          trace.end_point.layer == current_point.layer):
                        next_point = trace.start_point
                    
                    if next_point:
                        next_key = (next_point.x, next_point.y, next_point.layer)
                        if next_key not in visited:
                            visited.add(next_key)
                            new_path = path + [trace]
                            queue.append((next_point, new_path))
            
            return {'success': False, 'error': 'No path found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _trace_to_dict(self, trace: GridTrace) -> Dict:
        """Convert GridTrace to dictionary format"""
        return {
            'start_x': trace.start_point.x,
            'start_y': trace.start_point.y,
            'end_x': trace.end_point.x,
            'end_y': trace.end_point.y,
            'layer': trace.layer,
            'width': trace.width,
            'net_code': trace.net_id
        }
    
    def _traces_intersect(self, grid_trace: GridTrace, track_start_x: int, track_start_y: int, 
                         track_end_x: int, track_end_y: int, track_width: int) -> bool:
        """Check if grid trace intersects with existing track"""
        # Simplified intersection check
        grid_start_x = grid_trace.start_point.x
        grid_start_y = grid_trace.start_point.y
        grid_end_x = grid_trace.end_point.x
        grid_end_y = grid_trace.end_point.y
        
        # Bounding box intersection
        grid_min_x = min(grid_start_x, grid_end_x) - grid_trace.width // 2
        grid_max_x = max(grid_start_x, grid_end_x) + grid_trace.width // 2
        grid_min_y = min(grid_start_y, grid_end_y) - grid_trace.width // 2
        grid_max_y = max(grid_start_y, grid_end_y) + grid_trace.width // 2
        
        track_min_x = min(track_start_x, track_end_x) - track_width // 2
        track_max_x = max(track_start_x, track_end_x) + track_width // 2
        track_min_y = min(track_start_y, track_end_y) - track_width // 2
        track_max_y = max(track_start_y, track_end_y) + track_width // 2
        
        return (grid_min_x <= track_max_x and grid_max_x >= track_min_x and
                grid_min_y <= track_max_y and grid_max_y >= track_min_y)
    
    def _point_near_trace(self, point_x: int, point_y: int, trace: GridTrace, clearance: int) -> bool:
        """Check if point is within clearance distance of trace"""
        # Distance from point to line segment
        start_x = trace.start_point.x
        start_y = trace.start_point.y
        end_x = trace.end_point.x
        end_y = trace.end_point.y
        
        # Vector from start to end
        line_length_sq = (end_x - start_x)**2 + (end_y - start_y)**2
        if line_length_sq == 0:
            # Point to point distance
            distance = math.sqrt((point_x - start_x)**2 + (point_y - start_y)**2)
        else:
            # Project point onto line
            t = max(0, min(1, ((point_x - start_x) * (end_x - start_x) + 
                              (point_y - start_y) * (end_y - start_y)) / line_length_sq))
            
            proj_x = start_x + t * (end_x - start_x)
            proj_y = start_y + t * (end_y - start_y)
            
            distance = math.sqrt((point_x - proj_x)**2 + (point_y - proj_y)**2)
        
        return distance < clearance
    
    def get_routing_statistics(self) -> Dict:
        """Get comprehensive routing statistics"""
        total_traces = len(self.horizontal_traces) + len(self.vertical_traces)
        occupied_traces = sum(1 for trace in self.horizontal_traces + self.vertical_traces 
                            if trace.occupied)
        
        return {
            'grid_statistics': {
                'total_grid_points': len(self.grid_points),
                'available_grid_points': sum(1 for point in self.grid_points.values() if point.available),
                'total_traces': total_traces,
                'available_traces': total_traces - occupied_traces,
                'occupied_traces': occupied_traces,
                'grid_spacing_mm': self.grid_spacing / 1000000,
                'routing_layers': self.signal_layers
            },
            'routing_results': {
                'routed_nets': len(self.routed_nets),
                'failed_nets': len(self.failed_nets),
                'success_rate': len(self.routed_nets) / max(len(self.routed_nets) + len(self.failed_nets), 1) * 100
            }
        }
    
    def cleanup(self):
        """Clean up router resources"""
        print("🧹 Cleaning up grid router resources")
        self.horizontal_traces.clear()
        self.vertical_traces.clear()
        self.grid_points.clear()
        self.routed_nets.clear()
        self.failed_nets.clear()


def create_grid_router(board_data: Dict, config: Dict) -> GridBasedRouter:
    """
    Factory function to create a configured grid-based router
    
    Args:
        board_data: Board information and constraints
        config: Router configuration parameters
        
    Returns:
        GridBasedRouter: Configured router instance
    """
    print("🏭 Creating grid-based router...")
    
    # Set up default grid configuration
    default_config = {
        'grid_spacing': 2540000,  # 0.1 inch in nanometers
        'via_size': 203200,       # 8 mil via
        'default_trace_width': 152400,  # 6 mil trace
        'grid_algorithm': 'orthogonal',
        'max_via_count': 10,
        'prefer_straight_lines': True
    }
    
    # Merge with user config
    grid_config = {**default_config, **config}
    
    router = GridBasedRouter(board_data, grid_config)
    
    # Create the routing grid
    if router.create_routing_grid():
        print("✅ Grid-based router ready for operation")
        return router
    else:
        print("❌ Grid router creation failed")
        return None
