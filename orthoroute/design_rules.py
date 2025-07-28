"""
Design Rules and Validation for OrthoRoute
DRC (Design Rule Checking) and constraint management

This module handles all design rule validation, constraint checking,
and manufacturing compliance for routed PCBs.
"""

import cupy as cp
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from .grid_manager import Point3D, Net, GPUGrid

class ViolationType(Enum):
    """Types of design rule violations"""
    CLEARANCE = "clearance"
    TRACK_WIDTH = "track_width"
    VIA_SIZE = "via_size"
    VIA_DRILL = "via_drill"
    ANNULAR_RING = "annular_ring"
    THERMAL_RELIEF = "thermal_relief"
    COPPER_POUR = "copper_pour"
    SOLDER_MASK = "solder_mask"
    HOLE_TO_HOLE = "hole_to_hole"
    EDGE_CLEARANCE = "edge_clearance"
    DIFFERENTIAL_PAIR = "differential_pair"
    IMPEDANCE = "impedance"
    CURRENT_CAPACITY = "current_capacity"

@dataclass
class DesignRuleViolation:
    """A specific design rule violation"""
    violation_type: ViolationType
    severity: str  # "error", "warning", "info"
    net_id: Optional[int]
    net_name: Optional[str]
    location: Point3D
    measured_value: float
    required_value: float
    description: str
    suggestion: Optional[str] = None

@dataclass
class DesignRules:
    """Complete set of design rules for PCB manufacturing"""
    
    # Basic geometric rules (in nanometers)
    min_track_width: int = 100000          # 0.1mm
    max_track_width: int = 10000000        # 10mm
    min_clearance: int = 150000            # 0.15mm
    min_via_size: int = 200000             # 0.2mm
    min_via_drill: int = 100000            # 0.1mm
    max_via_drill: int = 6000000           # 6mm
    min_annular_ring: int = 50000          # 0.05mm
    
    # Advanced rules
    min_hole_to_hole: int = 250000         # 0.25mm
    min_edge_clearance: int = 300000       # 0.3mm
    max_aspect_ratio: float = 10.0         # Via depth/drill ratio
    
    # Layer specific rules
    layer_thickness: int = 1600000         # 1.6mm board thickness
    copper_thickness: int = 35000          # 35µm copper
    prepreg_thickness: int = 100000        # 0.1mm prepreg
    
    # Electrical rules
    max_current_density: float = 10.0      # A/mm² for external traces
    impedance_tolerance: float = 10.0      # ±10% impedance tolerance
    
    # Manufacturing tolerances
    drill_tolerance: float = 0.05          # ±0.05mm drill tolerance
    track_tolerance: float = 0.025         # ±0.025mm track tolerance
    
    # Thermal rules
    thermal_relief_spoke_width: int = 100000    # 0.1mm
    thermal_relief_clearance: int = 200000      # 0.2mm
    
    # Net class specific rules
    net_class_rules: Dict[str, Dict] = field(default_factory=dict)
    
    # Via rules by size
    via_rules: List[Dict] = field(default_factory=list)
    
    # Differential pair rules
    diff_pair_width: int = 150000          # 0.15mm
    diff_pair_gap: int = 150000            # 0.15mm
    diff_pair_tolerance: float = 5.0       # ±5% matching
    
    def __post_init__(self):
        """Initialize default rules if not provided"""
        if not self.net_class_rules:
            self.net_class_rules = {
                'default': {
                    'track_width': self.min_track_width,
                    'clearance': self.min_clearance,
                    'via_size': self.min_via_size
                },
                'power': {
                    'track_width': 500000,  # 0.5mm
                    'clearance': 200000,    # 0.2mm
                    'via_size': 400000      # 0.4mm
                },
                'high_speed': {
                    'track_width': 100000,  # 0.1mm
                    'clearance': 200000,    # 0.2mm
                    'via_size': 200000,     # 0.2mm
                    'impedance': 50.0       # 50Ω ±10%
                }
            }
        
        if not self.via_rules:
            self.via_rules = [
                {'diameter': 200000, 'drill': 100000, 'annular_ring': 50000},   # 0.2/0.1
                {'diameter': 300000, 'drill': 150000, 'annular_ring': 75000},   # 0.3/0.15
                {'diameter': 400000, 'drill': 200000, 'annular_ring': 100000},  # 0.4/0.2
                {'diameter': 600000, 'drill': 300000, 'annular_ring': 150000}   # 0.6/0.3
            ]

class DesignRuleChecker:
    """GPU-accelerated design rule checking"""
    
    def __init__(self, grid: GPUGrid, rules: DesignRules):
        self.grid = grid
        self.rules = rules
        self.violations = []
        
        # GPU arrays for efficient checking
        self._setup_gpu_arrays()
    
    def _setup_gpu_arrays(self):
        """Initialize GPU arrays for DRC"""
        grid_shape = (self.grid.layers, self.grid.height, self.grid.width)
        
        # Track occupancy maps
        self.track_map = cp.zeros(grid_shape, dtype=cp.uint8)
        self.via_map = cp.zeros((self.grid.height, self.grid.width), dtype=cp.uint8)
        
        # Clearance requirement maps
        self.clearance_map = cp.full(grid_shape, self.rules.min_clearance, dtype=cp.int32)
        
        # Net assignment maps
        self.net_map = cp.zeros(grid_shape, dtype=cp.int32)
    
    def check_all_rules(self, nets: List[Net]) -> List[DesignRuleViolation]:
        """
        Perform comprehensive design rule checking on all nets.
        
        Args:
            nets: List of routed nets to check
            
        Returns:
            List of design rule violations found
        """
        print(f"🔍 Starting DRC on {len(nets)} nets...")
        
        self.violations = []
        
        # Build occupancy maps
        self._build_occupancy_maps(nets)
        
        # Run all DRC checks
        self._check_track_widths(nets)
        self._check_clearances(nets)
        self._check_via_rules(nets)
        self._check_edge_clearances(nets)
        self._check_manufacturing_rules(nets)
        self._check_electrical_rules(nets)
        
        # Sort violations by severity
        self.violations.sort(key=lambda v: {'error': 0, 'warning': 1, 'info': 2}[v.severity])
        
        print(f"📋 DRC complete: {len(self.violations)} violations found")
        return self.violations
    
    def _build_occupancy_maps(self, nets: List[Net]):
        """Build GPU maps of track and via occupancy"""
        # Reset maps
        self.track_map[:] = 0
        self.via_map[:] = 0
        self.net_map[:] = 0
        
        for net in nets:
            if not net.routed or not net.route_path:
                continue
            
            # Mark track occupancy
            for i, point in enumerate(net.route_path):
                if (0 <= point.x < self.grid.width and 
                    0 <= point.y < self.grid.height and 
                    0 <= point.layer < self.grid.layers):
                    
                    self.track_map[point.layer, point.y, point.x] = 1
                    self.net_map[point.layer, point.y, point.x] = net.net_id
                    
                    # Check for vias (layer changes)
                    if i > 0:
                        prev_point = net.route_path[i-1]
                        if point.layer != prev_point.layer:
                            self.via_map[point.y, point.x] = 1
    
    def _check_track_widths(self, nets: List[Net]):
        """Check track width compliance"""
        for net in nets:
            if not net.routed:
                continue
            
            # Get required width for this net
            net_class = self._get_net_class(net.name)
            required_width = self.rules.net_class_rules[net_class]['track_width']
            
            # Check if net width meets requirements
            if net.width_nm < self.rules.min_track_width:
                self.violations.append(DesignRuleViolation(
                    violation_type=ViolationType.TRACK_WIDTH,
                    severity="error",
                    net_id=net.net_id,
                    net_name=net.name,
                    location=net.route_path[0] if net.route_path else Point3D(0, 0, 0),
                    measured_value=net.width_nm / 1000000.0,  # Convert to mm
                    required_value=self.rules.min_track_width / 1000000.0,
                    description=f"Track width {net.width_nm/1000000:.3f}mm below minimum {self.rules.min_track_width/1000000:.3f}mm",
                    suggestion=f"Increase track width to at least {self.rules.min_track_width/1000000:.3f}mm"
                ))
            
            if net.width_nm < required_width:
                self.violations.append(DesignRuleViolation(
                    violation_type=ViolationType.TRACK_WIDTH,
                    severity="warning",
                    net_id=net.net_id,
                    net_name=net.name,
                    location=net.route_path[0] if net.route_path else Point3D(0, 0, 0),
                    measured_value=net.width_nm / 1000000.0,
                    required_value=required_width / 1000000.0,
                    description=f"Track width {net.width_nm/1000000:.3f}mm below recommended {required_width/1000000:.3f}mm for net class",
                    suggestion=f"Consider increasing track width to {required_width/1000000:.3f}mm"
                ))
    
    def _check_clearances(self, nets: List[Net]):
        """Check clearance violations using GPU acceleration"""
        print("  🔍 Checking clearances...")
        
        # Create expanded occupancy map for clearance checking
        clearance_cells = int(self.rules.min_clearance / (self.grid.pitch_mm * 1000000))
        
        # Use morphological operations to check clearances
        for layer in range(self.grid.layers):
            layer_occupancy = self.track_map[layer]
            
            # Create structuring element for clearance
            if clearance_cells > 0:
                struct_element = cp.ones((clearance_cells*2+1, clearance_cells*2+1))
                
                # Dilate occupancy map
                expanded = self._morphological_dilation(layer_occupancy, struct_element)
                
                # Find overlaps
                overlaps = expanded & layer_occupancy
                
                # Check for violations
                violation_coords = cp.where(overlaps)
                
                if len(violation_coords[0]) > 0:
                    # Convert back to CPU for detailed analysis
                    y_coords = cp.asnumpy(violation_coords[0])
                    x_coords = cp.asnumpy(violation_coords[1])
                    
                    for y, x in zip(y_coords, x_coords):
                        # Find the nets involved
                        net1_id = int(self.net_map[layer, y, x])
                        
                        # Look for nearby different nets
                        for dy in range(-clearance_cells, clearance_cells+1):
                            for dx in range(-clearance_cells, clearance_cells+1):
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < self.grid.height and 
                                    0 <= nx < self.grid.width):
                                    net2_id = int(self.net_map[layer, ny, nx])
                                    
                                    if net2_id != 0 and net2_id != net1_id:
                                        # Found clearance violation
                                        distance = np.sqrt(dx*dx + dy*dy) * self.grid.pitch_mm
                                        
                                        self.violations.append(DesignRuleViolation(
                                            violation_type=ViolationType.CLEARANCE,
                                            severity="error",
                                            net_id=net1_id,
                                            net_name=f"Net {net1_id}",
                                            location=Point3D(int(x), int(y), layer),
                                            measured_value=distance,
                                            required_value=self.rules.min_clearance / 1000000.0,
                                            description=f"Clearance violation: {distance:.3f}mm between nets {net1_id} and {net2_id}",
                                            suggestion="Increase spacing between traces or reduce track width"
                                        ))
                                        break
    
    def _check_via_rules(self, nets: List[Net]):
        """Check via size and drill rules"""
        for net in nets:
            if not net.routed or net.via_count == 0:
                continue
            
            # Check via size
            if net.via_size_nm < self.rules.min_via_size:
                self.violations.append(DesignRuleViolation(
                    violation_type=ViolationType.VIA_SIZE,
                    severity="error",
                    net_id=net.net_id,
                    net_name=net.name,
                    location=self._find_first_via_location(net),
                    measured_value=net.via_size_nm / 1000000.0,
                    required_value=self.rules.min_via_size / 1000000.0,
                    description=f"Via size {net.via_size_nm/1000000:.3f}mm below minimum {self.rules.min_via_size/1000000:.3f}mm",
                    suggestion=f"Increase via size to at least {self.rules.min_via_size/1000000:.3f}mm"
                ))
            
            # Check drill size (assuming 60% of via size)
            drill_size = int(net.via_size_nm * 0.6)
            if drill_size < self.rules.min_via_drill:
                self.violations.append(DesignRuleViolation(
                    violation_type=ViolationType.VIA_DRILL,
                    severity="error",
                    net_id=net.net_id,
                    net_name=net.name,
                    location=self._find_first_via_location(net),
                    measured_value=drill_size / 1000000.0,
                    required_value=self.rules.min_via_drill / 1000000.0,
                    description=f"Via drill {drill_size/1000000:.3f}mm below minimum {self.rules.min_via_drill/1000000:.3f}mm",
                    suggestion="Increase via size or check via rules"
                ))
            
            # Check annular ring
            annular_ring = (net.via_size_nm - drill_size) / 2
            if annular_ring < self.rules.min_annular_ring:
                self.violations.append(DesignRuleViolation(
                    violation_type=ViolationType.ANNULAR_RING,
                    severity="warning",
                    net_id=net.net_id,
                    net_name=net.name,
                    location=self._find_first_via_location(net),
                    measured_value=annular_ring / 1000000.0,
                    required_value=self.rules.min_annular_ring / 1000000.0,
                    description=f"Annular ring {annular_ring/1000000:.3f}mm below recommended {self.rules.min_annular_ring/1000000:.3f}mm",
                    suggestion="Increase via size or reduce drill size"
                ))
    
    def _check_edge_clearances(self, nets: List[Net]):
        """Check clearance to board edges"""
        for net in nets:
            if not net.routed or not net.route_path:
                continue
            
            for point in net.route_path:
                # Check distance to edges
                edge_distances = [
                    point.x * self.grid.pitch_mm * 1000000,  # Distance to left edge
                    point.y * self.grid.pitch_mm * 1000000,  # Distance to top edge
                    (self.grid.width - point.x) * self.grid.pitch_mm * 1000000,  # Distance to right edge
                    (self.grid.height - point.y) * self.grid.pitch_mm * 1000000   # Distance to bottom edge
                ]
                
                min_edge_distance = min(edge_distances)
                
                if min_edge_distance < self.rules.min_edge_clearance:
                    self.violations.append(DesignRuleViolation(
                        violation_type=ViolationType.EDGE_CLEARANCE,
                        severity="warning",
                        net_id=net.net_id,
                        net_name=net.name,
                        location=point,
                        measured_value=min_edge_distance / 1000000.0,
                        required_value=self.rules.min_edge_clearance / 1000000.0,
                        description=f"Track too close to board edge: {min_edge_distance/1000000:.3f}mm clearance",
                        suggestion=f"Maintain at least {self.rules.min_edge_clearance/1000000:.3f}mm from board edges"
                    ))
    
    def _check_manufacturing_rules(self, nets: List[Net]):
        """Check manufacturing constraints"""
        # Check aspect ratios for vias
        board_thickness = self.rules.layer_thickness
        
        for net in nets:
            if not net.routed or net.via_count == 0:
                continue
            
            drill_size = int(net.via_size_nm * 0.6)
            aspect_ratio = board_thickness / drill_size
            
            if aspect_ratio > self.rules.max_aspect_ratio:
                self.violations.append(DesignRuleViolation(
                    violation_type=ViolationType.VIA_DRILL,
                    severity="warning",
                    net_id=net.net_id,
                    net_name=net.name,
                    location=self._find_first_via_location(net),
                    measured_value=aspect_ratio,
                    required_value=self.rules.max_aspect_ratio,
                    description=f"Via aspect ratio {aspect_ratio:.1f} exceeds recommended {self.rules.max_aspect_ratio}",
                    suggestion="Increase drill size or use blind/buried vias"
                ))
    
    def _check_electrical_rules(self, nets: List[Net]):
        """Check electrical constraints"""
        for net in nets:
            if not net.routed:
                continue
            
            # Check current capacity (simplified)
            net_class = self._get_net_class(net.name)
            if net_class == 'power':
                # Power nets need higher current capacity
                track_area = (net.width_nm / 1000000.0) * (self.rules.copper_thickness / 1000000.0)  # mm²
                max_current = track_area * self.rules.max_current_density
                
                # This is just an example - real current requirements would come from netlist
                if max_current < 1.0:  # Assume power nets need >1A capacity
                    self.violations.append(DesignRuleViolation(
                        violation_type=ViolationType.CURRENT_CAPACITY,
                        severity="warning",
                        net_id=net.net_id,
                        net_name=net.name,
                        location=net.route_path[0] if net.route_path else Point3D(0, 0, 0),
                        measured_value=max_current,
                        required_value=1.0,
                        description=f"Power track current capacity {max_current:.2f}A may be insufficient",
                        suggestion="Increase track width for higher current capacity"
                    ))
    
    def _get_net_class(self, net_name: str) -> str:
        """Determine net class from net name"""
        net_upper = net_name.upper()
        
        if any(keyword in net_upper for keyword in ['VCC', 'VDD', 'GND', 'POWER', 'PWR']):
            return 'power'
        elif any(keyword in net_upper for keyword in ['CLK', 'CLOCK', 'USB', 'PCIE', 'HDMI']):
            return 'high_speed'
        else:
            return 'default'
    
    def _find_first_via_location(self, net: Net) -> Point3D:
        """Find location of first via in net"""
        if not net.route_path or len(net.route_path) < 2:
            return Point3D(0, 0, 0)
        
        for i in range(len(net.route_path) - 1):
            if net.route_path[i].layer != net.route_path[i+1].layer:
                return net.route_path[i]
        
        return net.route_path[0]
    
    def _morphological_dilation(self, image: cp.ndarray, struct_element: cp.ndarray) -> cp.ndarray:
        """Simple morphological dilation for clearance checking"""
        # This is a simplified implementation
        # A full implementation would use optimized convolution
        
        if struct_element.size == 1:
            return image
        
        # Use convolution for dilation
        result = cp.zeros_like(image)
        sh, sw = struct_element.shape
        h, w = image.shape
        
        for i in range(h):
            for j in range(w):
                if image[i, j]:
                    # Place structuring element
                    start_i = max(0, i - sh//2)
                    end_i = min(h, i + sh//2 + 1)
                    start_j = max(0, j - sw//2)
                    end_j = min(w, j + sw//2 + 1)
                    
                    result[start_i:end_i, start_j:end_j] = 1
        
        return result

def generate_drc_report(violations: List[DesignRuleViolation]) -> str:
    """Generate comprehensive DRC report"""
    if not violations:
        return "✅ No design rule violations found!"
    
    # Count violations by type and severity
    error_count = sum(1 for v in violations if v.severity == "error")
    warning_count = sum(1 for v in violations if v.severity == "warning")
    info_count = sum(1 for v in violations if v.severity == "info")
    
    # Group by violation type
    violations_by_type = {}
    for violation in violations:
        vtype = violation.violation_type.value
        if vtype not in violations_by_type:
            violations_by_type[vtype] = []
        violations_by_type[vtype].append(violation)
    
    report = f"""
Design Rule Check Report
========================

Summary:
  Total violations: {len(violations)}
  Errors: {error_count}
  Warnings: {warning_count}
  Info: {info_count}

Violations by Type:
"""
    
    for vtype, vlist in violations_by_type.items():
        report += f"\n{vtype.upper()}:\n"
        
        # Show first few violations of each type
        for i, violation in enumerate(vlist[:5]):
            severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[violation.severity]
            report += f"  {severity_icon} Net {violation.net_name}: {violation.description}\n"
            if violation.suggestion:
                report += f"     💡 {violation.suggestion}\n"
        
        if len(vlist) > 5:
            report += f"     ... and {len(vlist) - 5} more\n"
    
    report += f"\nRecommendations:\n"
    
    if error_count > 0:
        report += "  🔴 Fix all errors before manufacturing\n"
    if warning_count > 0:
        report += "  🟡 Review warnings for potential issues\n"
    if error_count == 0 and warning_count == 0:
        report += "  ✅ Design ready for manufacturing\n"
    
    return report

def create_default_rules() -> DesignRules:
    """Create default design rules for typical PCB manufacturing"""
    return DesignRules(
        min_track_width=100000,      # 0.1mm (4 mil)
        min_clearance=150000,        # 0.15mm (6 mil)
        min_via_size=200000,         # 0.2mm (8 mil)
        min_via_drill=100000,        # 0.1mm (4 mil)
        min_annular_ring=50000,      # 0.05mm (2 mil)
        min_hole_to_hole=250000,     # 0.25mm (10 mil)
        min_edge_clearance=300000,   # 0.3mm (12 mil)
        max_aspect_ratio=8.0         # 8:1 drill aspect ratio
    )