"""
Steiner Tree Builder for OrthoRoute
Multi-pin net routing using GPU-accelerated algorithms

This module handles routing of nets with more than 2 pins by constructing
Steiner trees to minimize total wire length and via count.
"""

import cupy as cp
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from .grid_manager import Point3D, Net, GPUGrid
import heapq
import itertools

@dataclass
class SteinerPoint:
    """A point in a Steiner tree (pin or Steiner node)"""
    x: int
    y: int
    layer: int
    is_pin: bool = False
    pin_index: int = -1  # Index in original pin list
    
    def to_point3d(self) -> Point3D:
        return Point3D(self.x, self.y, self.layer)

@dataclass  
class TreeEdge:
    """Edge in Steiner tree"""
    start: SteinerPoint
    end: SteinerPoint
    length: float
    via_count: int
    path: List[Point3D] = None
    
    def __post_init__(self):
        if self.path is None:
            self.path = []

class SteinerTreeBuilder:
    """Build Steiner trees for multi-pin nets using GPU acceleration"""
    
    def __init__(self, grid: GPUGrid):
        self.grid = grid
        self.hanan_cache = {}  # Cache Hanan grids for performance
        
    def route_multi_pin_net(self, net: Net, router=None) -> Optional[List[Point3D]]:
        """
        Route a multi-pin net using Steiner tree approach.
        
        Args:
            net: Net with multiple pins to route
            router: Optional WavefrontRouter for segment routing
            
        Returns:
            Complete route path connecting all pins, or None if routing failed
        """
        if len(net.pins) < 2:
            return None
        elif len(net.pins) == 2:
            # Use direct two-pin routing
            if router:
                return router._route_two_pin(net.pins[0], net.pins[1])
            else:
                return self._simple_two_pin_route(net.pins[0], net.pins[1])
        
        # Multi-pin routing with Steiner tree
        steiner_tree = self._build_steiner_tree(net.pins)
        if not steiner_tree:
            return None
        
        # Route each edge of the Steiner tree
        full_path = []
        for edge in steiner_tree:
            if router:
                segment_path = router._route_two_pin(
                    edge.start.to_point3d(), 
                    edge.end.to_point3d()
                )
            else:
                segment_path = self._simple_two_pin_route(
                    edge.start.to_point3d(),
                    edge.end.to_point3d()
                )
            
            if not segment_path:
                # Failed to route this segment
                return None
            
            # Add segment to full path (avoiding duplicates at connection points)
            if not full_path:
                full_path.extend(segment_path)
            else:
                # Skip first point if it's the same as the last point in full_path
                if segment_path and full_path and segment_path[0] == full_path[-1]:
                    full_path.extend(segment_path[1:])
                else:
                    full_path.extend(segment_path)
        
        return full_path
    
    def _build_steiner_tree(self, pins: List[Point3D]) -> Optional[List[TreeEdge]]:
        """
        Build Steiner tree connecting all pins.
        
        Uses a combination of approaches:
        1. Hanan grid generation for candidate Steiner points
        2. Minimum spanning tree construction
        3. Local optimization to reduce via count
        
        Args:
            pins: List of pins to connect
            
        Returns:
            List of edges forming the Steiner tree
        """
        if len(pins) <= 1:
            return []
        
        # Convert pins to SteinerPoints
        steiner_pins = []
        for i, pin in enumerate(pins):
            steiner_pins.append(SteinerPoint(
                x=pin.x, y=pin.y, layer=pin.layer,
                is_pin=True, pin_index=i
            ))
        
        # Strategy selection based on pin count
        if len(pins) <= 4:
            # Small nets: Try optimal approaches
            return self._build_optimal_steiner_tree(steiner_pins)
        elif len(pins) <= 20:
            # Medium nets: Hanan grid + MST
            return self._build_hanan_mst(steiner_pins)
        else:
            # Large nets: Fast approximation
            return self._build_approximate_tree(steiner_pins)
    
    def _build_optimal_steiner_tree(self, pins: List[SteinerPoint]) -> List[TreeEdge]:
        """Build optimal Steiner tree for small number of pins"""
        if len(pins) == 2:
            return [self._create_edge(pins[0], pins[1])]
        
        # For 3-4 pins, try different connection strategies
        best_tree = None
        best_cost = float('inf')
        
        # Strategy 1: Direct minimum spanning tree
        mst_tree = self._minimum_spanning_tree(pins)
        mst_cost = self._calculate_tree_cost(mst_tree)
        
        if mst_cost < best_cost:
            best_cost = mst_cost
            best_tree = mst_tree
        
        # Strategy 2: Add Steiner points at pin intersections
        if len(pins) <= 4:
            hanan_tree = self._build_hanan_mst(pins)
            if hanan_tree:
                hanan_cost = self._calculate_tree_cost(hanan_tree)
                if hanan_cost < best_cost:
                    best_cost = hanan_cost
                    best_tree = hanan_tree
        
        return best_tree if best_tree else mst_tree
    
    def _build_hanan_mst(self, pins: List[SteinerPoint]) -> List[TreeEdge]:
        """Build Steiner tree using Hanan grid and MST"""
        # Generate Hanan grid points
        hanan_points = self._generate_hanan_grid(pins)
        
        # Add original pins to candidate points
        all_points = pins + hanan_points
        
        # Build minimum spanning tree on all candidate points
        mst_edges = self._minimum_spanning_tree(all_points)
        
        # Remove unnecessary Steiner points (degree 2 nodes that don't reduce cost)
        optimized_edges = self._optimize_steiner_points(mst_edges, pins)
        
        return optimized_edges
    
    def _build_approximate_tree(self, pins: List[SteinerPoint]) -> List[TreeEdge]:
        """Fast approximation for large pin counts"""
        # Use nearest neighbor heuristic with some optimization
        
        if not pins:
            return []
        
        # Start with arbitrary pin
        tree_edges = []
        connected_pins = {pins[0]}
        remaining_pins = set(pins[1:])
        
        while remaining_pins:
            # Find closest unconnected pin to any connected pin
            best_distance = float('inf')
            best_connection = None
            
            for connected_pin in connected_pins:
                for remaining_pin in remaining_pins:
                    distance = self._calculate_distance(connected_pin, remaining_pin)
                    if distance < best_distance:
                        best_distance = distance
                        best_connection = (connected_pin, remaining_pin)
            
            if best_connection:
                edge = self._create_edge(best_connection[0], best_connection[1])
                tree_edges.append(edge)
                connected_pins.add(best_connection[1])
                remaining_pins.remove(best_connection[1])
        
        return tree_edges
    
    def _generate_hanan_grid(self, pins: List[SteinerPoint]) -> List[SteinerPoint]:
        """Generate Hanan grid points for Steiner tree construction"""
        # Extract unique coordinates
        x_coords = sorted(set(pin.x for pin in pins))
        y_coords = sorted(set(pin.y for pin in pins))
        layers = sorted(set(pin.layer for pin in pins))
        
        # Create cache key
        cache_key = (tuple(x_coords), tuple(y_coords), tuple(layers))
        if cache_key in self.hanan_cache:
            return self.hanan_cache[cache_key]
        
        hanan_points = []
        
        # Generate grid points at coordinate intersections
        for x in x_coords:
            for y in y_coords:
                for layer in layers:
                    # Check if this point is not already a pin
                    is_existing_pin = any(
                        pin.x == x and pin.y == y and pin.layer == layer 
                        for pin in pins
                    )
                    
                    if not is_existing_pin:
                        # Only add if point could potentially improve routing
                        if self._is_beneficial_steiner_point(x, y, layer, pins):
                            hanan_points.append(SteinerPoint(x, y, layer, is_pin=False))
        
        # Cache result
        self.hanan_cache[cache_key] = hanan_points
        return hanan_points
    
    def _is_beneficial_steiner_point(self, x: int, y: int, layer: int, 
                                   pins: List[SteinerPoint]) -> bool:
        """Check if a Steiner point could improve routing"""
        # Simple heuristic: point is beneficial if it's "between" at least 2 pins
        
        # Count pins in each direction
        pins_left = sum(1 for pin in pins if pin.x < x)
        pins_right = sum(1 for pin in pins if pin.x > x)
        pins_below = sum(1 for pin in pins if pin.y < y)
        pins_above = sum(1 for pin in pins if pin.y > y)
        
        # Point is beneficial if it has pins on multiple sides
        horizontal_split = pins_left > 0 and pins_right > 0
        vertical_split = pins_below > 0 and pins_above > 0
        
        return horizontal_split or vertical_split
    
    def _minimum_spanning_tree(self, points: List[SteinerPoint]) -> List[TreeEdge]:
        """Build minimum spanning tree using Prim's algorithm"""
        if len(points) <= 1:
            return []
        
        # Start with arbitrary point
        mst_edges = []
        in_tree = {points[0]}
        candidate_edges = []
        
        # Add all edges from starting point
        for point in points[1:]:
            edge = self._create_edge(points[0], point)
            heapq.heappush(candidate_edges, (edge.length, len(candidate_edges), edge))
        
        while candidate_edges and len(in_tree) < len(points):
            cost, _, edge = heapq.heappop(candidate_edges)
            
            # Check if edge connects tree to new point
            start_in_tree = edge.start in in_tree
            end_in_tree = edge.end in in_tree
            
            if start_in_tree and not end_in_tree:
                # Add edge and new point to tree
                mst_edges.append(edge)
                in_tree.add(edge.end)
                new_point = edge.end
            elif end_in_tree and not start_in_tree:
                # Add edge and new point to tree
                mst_edges.append(edge)
                in_tree.add(edge.start)
                new_point = edge.start
            else:
                # Edge doesn't extend tree, skip
                continue
            
            # Add edges from new point to remaining points
            for point in points:
                if point not in in_tree:
                    new_edge = self._create_edge(new_point, point)
                    heapq.heappush(candidate_edges, 
                                 (new_edge.length, len(candidate_edges), new_edge))
        
        return mst_edges
    
    def _optimize_steiner_points(self, edges: List[TreeEdge], 
                               original_pins: List[SteinerPoint]) -> List[TreeEdge]:
        """Remove unnecessary Steiner points and optimize tree"""
        # Build adjacency list
        adjacency = {}
        for edge in edges:
            if edge.start not in adjacency:
                adjacency[edge.start] = []
            if edge.end not in adjacency:
                adjacency[edge.end] = []
            adjacency[edge.start].append((edge.end, edge))
            adjacency[edge.end].append((edge.start, edge))
        
        # Find Steiner points that can be removed (degree 2, not reducing cost)
        points_to_remove = []
        for point in adjacency:
            if not point.is_pin and len(adjacency[point]) == 2:
                # This is a degree-2 Steiner point - check if removing improves cost
                neighbors = [neighbor for neighbor, _ in adjacency[point]]
                
                # Cost with Steiner point
                current_cost = sum(edge.length for _, edge in adjacency[point])
                
                # Cost without Steiner point (direct connection)
                direct_edge = self._create_edge(neighbors[0], neighbors[1])
                direct_cost = direct_edge.length
                
                if direct_cost <= current_cost:
                    points_to_remove.append(point)
        
        # Remove unnecessary points and rebuild edge list
        if points_to_remove:
            return self._rebuild_tree_without_points(edges, points_to_remove)
        
        return edges
    
    def _rebuild_tree_without_points(self, edges: List[TreeEdge], 
                                   points_to_remove: List[SteinerPoint]) -> List[TreeEdge]:
        """Rebuild tree after removing unnecessary Steiner points"""
        # Create mapping of connections through removed points
        new_edges = []
        remove_set = set(points_to_remove)
        
        # Build connectivity through removed points
        connections_through_removed = {}
        for point in points_to_remove:
            neighbors = []
            for edge in edges:
                if edge.start == point:
                    neighbors.append(edge.end)
                elif edge.end == point:
                    neighbors.append(edge.start)
            
            # Connect neighbors directly
            if len(neighbors) == 2:
                connections_through_removed[point] = neighbors
        
        # Add edges that don't involve removed points
        for edge in edges:
            if edge.start not in remove_set and edge.end not in remove_set:
                new_edges.append(edge)
        
        # Add direct connections for removed points
        for removed_point, neighbors in connections_through_removed.items():
            if len(neighbors) == 2:
                new_edge = self._create_edge(neighbors[0], neighbors[1])
                new_edges.append(new_edge)
        
        return new_edges
    
    def _create_edge(self, start: SteinerPoint, end: SteinerPoint) -> TreeEdge:
        """Create edge between two points with cost calculation"""
        # Calculate Manhattan distance
        dx = abs(end.x - start.x)
        dy = abs(end.y - start.y)
        
        # Layer distance (via cost)
        dz = abs(end.layer - start.layer)
        via_count = 1 if dz > 0 else 0
        
        # Total length in grid units
        manhattan_distance = dx + dy
        length = manhattan_distance * self.grid.pitch_mm
        
        # Add via penalty
        if via_count > 0:
            length += via_count * 0.1  # 0.1mm penalty per via
        
        return TreeEdge(start, end, length, via_count)
    
    def _calculate_distance(self, p1: SteinerPoint, p2: SteinerPoint) -> float:
        """Calculate routing distance between two points"""
        dx = abs(p2.x - p1.x)
        dy = abs(p2.y - p1.y)
        dz = abs(p2.layer - p1.layer)
        
        # Manhattan distance + via penalty
        distance = (dx + dy) * self.grid.pitch_mm
        if dz > 0:
            distance += dz * 0.1  # Via penalty
        
        return distance
    
    def _calculate_tree_cost(self, edges: List[TreeEdge]) -> float:
        """Calculate total cost of tree"""
        if not edges:
            return float('inf')
        
        total_length = sum(edge.length for edge in edges)
        total_vias = sum(edge.via_count for edge in edges)
        
        # Cost function: length + via penalty
        return total_length + total_vias * 0.1
    
    def _simple_two_pin_route(self, start: Point3D, end: Point3D) -> Optional[List[Point3D]]:
        """
        Simple two-pin routing for fallback when WavefrontRouter not available.
        Uses L-shaped routing (horizontal then vertical, or vice versa).
        """
        # Try horizontal-first routing
        path1 = []
        
        # Horizontal movement
        current_x = start.x
        while current_x != end.x:
            if end.x > current_x:
                current_x += 1
            else:
                current_x -= 1
            path1.append(Point3D(current_x, start.y, start.layer))
        
        # Vertical movement
        current_y = start.y
        while current_y != end.y:
            if end.y > current_y:
                current_y += 1
            else:
                current_y -= 1
            path1.append(Point3D(end.x, current_y, start.layer))
        
        # Layer change if needed
        if start.layer != end.layer:
            path1.append(Point3D(end.x, end.y, end.layer))
        
        # Check if path is valid (no obstacles)
        if self._is_path_valid(path1):
            return [start] + path1
        
        # Try vertical-first routing
        path2 = []
        
        # Vertical movement first
        current_y = start.y
        while current_y != end.y:
            if end.y > current_y:
                current_y += 1
            else:
                current_y -= 1
            path2.append(Point3D(start.x, current_y, start.layer))
        
        # Horizontal movement
        current_x = start.x
        while current_x != end.x:
            if end.x > current_x:
                current_x += 1
            else:
                current_x -= 1
            path2.append(Point3D(current_x, end.y, start.layer))
        
        # Layer change if needed
        if start.layer != end.layer:
            path2.append(Point3D(end.x, end.y, end.layer))
        
        if self._is_path_valid(path2):
            return [start] + path2
        
        # Both paths blocked
        return None
    
    def _is_path_valid(self, path: List[Point3D]) -> bool:
        """Check if path is valid (no obstacles)"""
        for point in path:
            if (0 <= point.x < self.grid.width and 
                0 <= point.y < self.grid.height and 
                0 <= point.layer < self.grid.layers):
                
                if not self.grid.availability[point.layer, point.y, point.x]:
                    return False
            else:
                return False  # Out of bounds
        
        return True
    
    def optimize_via_count(self, route_path: List[Point3D]) -> List[Point3D]:
        """
        Optimize route to minimize via count while maintaining connectivity.
        
        Args:
            route_path: Original route path
            
        Returns:
            Optimized path with fewer layer changes
        """
        if len(route_path) <= 2:
            return route_path
        
        # Group consecutive segments by layer
        segments = []
        current_segment = [route_path[0]]
        
        for i in range(1, len(route_path)):
            if route_path[i].layer == current_segment[-1].layer:
                current_segment.append(route_path[i])
            else:
                # Layer change - start new segment
                segments.append(current_segment)
                current_segment = [route_path[i]]
        
        if current_segment:
            segments.append(current_segment)
        
        # Try to merge adjacent segments on same layer
        optimized_segments = []
        i = 0
        while i < len(segments):
            current_seg = segments[i]
            
            # Look for segments on same layer that can be merged
            merged = False
            for j in range(i + 2, len(segments)):  # Skip adjacent segment
                if segments[j][0].layer == current_seg[0].layer:
                    # Try to merge segments i and j
                    merged_path = self._try_merge_segments(current_seg, segments[i+1:j], segments[j])
                    if merged_path:
                        optimized_segments.append(merged_path)
                        i = j + 1
                        merged = True
                        break
            
            if not merged:
                optimized_segments.append(current_seg)
                i += 1
        
        # Reconstruct path
        optimized_path = []
        for segment in optimized_segments:
            if optimized_path and segment and segment[0] == optimized_path[-1]:
                optimized_path.extend(segment[1:])
            else:
                optimized_path.extend(segment)
        
        return optimized_path
    
    def _try_merge_segments(self, seg1: List[Point3D], middle_segs: List[List[Point3D]], 
                           seg3: List[Point3D]) -> Optional[List[Point3D]]:
        """Try to merge two segments on same layer, bypassing middle segments"""
        if not seg1 or not seg3:
            return None
        
        start = seg1[-1]  # End of first segment
        end = seg3[0]     # Start of third segment
        
        # Try direct connection on the shared layer
        direct_path = self._simple_two_pin_route(
            Point3D(start.x, start.y, seg1[0].layer),
            Point3D(end.x, end.y, seg1[0].layer)
        )
        
        if direct_path:
            # Successful merge
            merged = seg1[:-1] + direct_path + seg3[1:]
            return merged
        
        return None

# Utility functions for Steiner tree algorithms
def calculate_steiner_ratio(tree_length: float, mst_length: float) -> float:
    """Calculate Steiner ratio (should be <= 1.0)"""
    if mst_length == 0:
        return 1.0
    return tree_length / mst_length

def estimate_steiner_improvement(pin_count: int) -> float:
    """Estimate potential improvement from Steiner tree vs MST"""
    if pin_count <= 2:
        return 0.0
    elif pin_count <= 4:
        return 0.1  # Up to 10% improvement
    elif pin_count <= 10:
        return 0.15  # Up to 15% improvement
    else:
        return 0.2   # Up to 20% improvement for large nets