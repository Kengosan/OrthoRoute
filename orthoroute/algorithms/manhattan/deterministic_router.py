"""
Deterministic Manhattan Router
Advanced PCB routing engine with grid-based pathfinding, escape routing, and rip-up/repair
"""

import numpy as np
import heapq
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import math
import time
import sys
import os

# Add debug logging capability
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
try:
    from debug_logger import get_debug_logger
    DEBUG_LOGGING_AVAILABLE = True
except ImportError:
    DEBUG_LOGGING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Occupancy kind constants
FREE = 0
PAD = 1
TRACE = 2
KEEPOUT = 3

# Unit conversion constant
IU_PER_MM = 1_000_000.0

@dataclass
class Pad:
    """Normalized pad representation"""
    net_name: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    layer_set: Union[Set[str], str]  # Set of layer names or "THRU"
    is_through_hole: bool

@dataclass
class Via:
    """Via structure"""
    x: float
    y: float
    from_layer: int
    to_layer: int
    size: float
    drill: float
    net_id: int

@dataclass
class Track:
    """Track/trace structure"""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    layer: int
    width: float
    net_id: int

@dataclass
class RouteResult:
    """Routing result"""
    success: bool
    tracks: List[Track]
    vias: List[Via]
    routed_nets: int
    failed_nets: int
    stats: Dict[str, Any]

@dataclass
class RoutingConfig:
    """Routing configuration"""
    grid_pitch: float = 0.40  # mm
    track_width: float = 0.2  # mm - KiCad default
    clearance: float = 0.2  # mm - KiCad default
    via_drill: float = 0.4  # mm - typical drill for 0.8mm via
    via_diameter: float = 0.25  # mm - Original working size
    bend_penalty: int = 2
    via_cost: int = 10
    expansion_margin: float = 3.0  # mm
    max_ripups_per_net: int = 2  # Reduced from 3 to 2
    max_global_failures: int = 100
    timeout_per_net: float = 0.1  # AGGRESSIVE: 0.1 seconds for 2 nets/second target
    
    @property
    def via_size(self) -> float:
        """Alias for via_diameter for compatibility"""
        return self.via_diameter

@dataclass 
class FabricSegment:
    """A segment of routing fabric - straight trace between two nodes"""
    segment_id: str
    layer_name: str 
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    is_horizontal: bool
    status: str = "AVAILABLE"  # AVAILABLE | CLAIMED | RESERVED
    claimed_by_net: Optional[str] = None
    
    @property
    def length(self) -> float:
        return ((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)**0.5

@dataclass
class FabricNode:
    """Intersection point where fabric segments meet - via location"""
    node_id: str
    x: float
    y: float
    layers: List[str]  # Which layers this node connects
    connected_segments: List[str]  # Segment IDs connected to this node
    
@dataclass
class FabricNetwork:
    """Complete routing fabric for internal layers"""
    segments: Dict[str, FabricSegment]  # segment_id -> FabricSegment
    nodes: Dict[str, FabricNode]  # node_id -> FabricNode
    layer_segments: Dict[str, List[str]]  # layer_name -> [segment_ids]
    
    def get_available_segments(self, layer_name: str) -> List[FabricSegment]:
        """Get all available segments on a layer"""
        layer_seg_ids = self.layer_segments.get(layer_name, [])
        return [self.segments[seg_id] for seg_id in layer_seg_ids 
                if self.segments[seg_id].status == "AVAILABLE"]
    
    def claim_segment(self, segment_id: str, net_name: str) -> bool:
        """Claim a segment for a net"""
        if segment_id in self.segments and self.segments[segment_id].status == "AVAILABLE":
            self.segments[segment_id].status = "CLAIMED"
            self.segments[segment_id].claimed_by_net = net_name
            return True
        return False
    
    def release_segments(self, net_name: str) -> int:
        """Release all segments claimed by a net"""
        released = 0
        for segment in self.segments.values():
            if segment.claimed_by_net == net_name:
                segment.status = "AVAILABLE"
                segment.claimed_by_net = None
                released += 1
        return released

def get_pos_mm(pad_data: Dict) -> Tuple[float, float]:
    """Extract position in mm from pad data (already converted by KiCad interface)"""
    return float(pad_data.get('x', 0.0)), float(pad_data.get('y', 0.0))

def get_net_name(pad_data: Dict) -> Optional[str]:
    """Extract net name from pad data"""
    # Try both 'net' and 'net_name' fields (KiCad interface uses 'net_name')
    net_name = pad_data.get('net', '')
    if not net_name or net_name == 'MISSING' or not net_name.strip():
        net_name = pad_data.get('net_name', '')
    return net_name if net_name and net_name.strip() else None

def is_through_hole(pad_data: Dict) -> bool:
    """Check if pad is through hole"""
    pad_type = pad_data.get('type', 'smd')
    return pad_type == 'through_hole'

def get_layer_set(pad_data: Dict) -> Union[Set[str], str]:
    """Get pad layer set"""
    if is_through_hole(pad_data):
        return "THRU"
    layers = pad_data.get('layers', ['F.Cu'])
    return set(layers) if isinstance(layers, list) else {layers}

def snapshot_board(board_data: Dict[str, Any]) -> Tuple[List[Pad], Dict[str, List[Pad]]]:
    """Create unified board snapshot from board data"""
    logger.info("Creating unified board snapshot...")
    
    pads = []
    nets = defaultdict(list)
    
    raw_pads = board_data.get('pads', [])
    logger.info(f"Raw pads from board_data: {len(raw_pads)}")
    
    # Debug: Check first few pads
    sample_count = min(5, len(raw_pads))
    for i in range(sample_count):
        pad = raw_pads[i]
        net_field = pad.get('net', 'MISSING')
        net_name_field = pad.get('net_name', 'MISSING')
        all_keys = list(pad.keys())
        logger.info(f"Sample pad {i}: net='{net_field}', net_name='{net_name_field}', x={pad.get('x', 'MISSING')}, layers={pad.get('layers', 'MISSING')}")
        logger.info(f"Sample pad {i} all keys: {all_keys}")
    
    skipped_no_net = 0
    skipped_errors = 0
    
    for pad_data in raw_pads:
        try:
            net_name = get_net_name(pad_data)
            if not net_name:  # Skip net 0 / no net
                skipped_no_net += 1
                continue
                
            x_mm, y_mm = get_pos_mm(pad_data)
            width_mm = float(pad_data.get('width', 1.0))
            height_mm = float(pad_data.get('height', 1.0))
            layer_set = get_layer_set(pad_data)
            is_thru = is_through_hole(pad_data)
            
            pad = Pad(
                net_name=net_name,
                x_mm=x_mm,
                y_mm=y_mm,
                width_mm=width_mm,
                height_mm=height_mm,
                layer_set=layer_set,
                is_through_hole=is_thru
            )
            
            pads.append(pad)
            nets[net_name].append(pad)
            
        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Skipping invalid pad: {e}")
            skipped_errors += 1
            continue
    
    # Filter to routable nets (>=2 pads)
    routable_nets = {net_name: pad_list for net_name, pad_list in nets.items() if len(pad_list) >= 2}
    
    logger.info(f"Snapshot: pads={len(pads)}, nets total={len(nets)}, nets >=2 pads={len(routable_nets)}")
    logger.info(f"Skipped: {skipped_no_net} pads with no net, {skipped_errors} pads with errors")
    
    # Sample pad logging for validation  
    if pads:
        sample_pad = pads[0]
        logger.info(f"Sample pad: net={sample_pad.net_name} at=({sample_pad.x_mm:.2f},{sample_pad.y_mm:.2f}) thru={sample_pad.is_through_hole} layers={sample_pad.layer_set}")
    else:
        logger.error("CRITICAL: No pads extracted in snapshot!")
    
    return pads, routable_nets

class DeterministicManhattanRouter:
    """
    Deterministic Manhattan router with grid-based pathfinding
    """
    
    def __init__(self, config: RoutingConfig = None, gpu_provider=None, drc_constraints=None):
        self.config = config or RoutingConfig()
        self.gpu_provider = gpu_provider
        self.drc_constraints = drc_constraints
        
        # If no DRC constraints provided, create defaults
        if self.drc_constraints is None:
            try:
                from ...domain.models.constraints import DRCConstraints
                self.drc_constraints = DRCConstraints()
                logger.info("Created default DRC constraints for router")
            except ImportError:
                logger.warning("Could not import DRC constraints")
                self.drc_constraints = None
        
        # Log GPU availability
        if self.gpu_provider and self.gpu_provider.is_available():
            try:
                device_info = self.gpu_provider.get_device_info()
                logger.info(f"Using GPU acceleration: {device_info.get('name', 'Unknown GPU')}")
                logger.info(f"GPU Memory: {device_info.get('total_memory', 0) / 1024**3:.1f}GB total")
            except Exception as e:
                logger.warning(f"Error getting GPU info: {e}")
        else:
            logger.info("GPU acceleration not available, using CPU fallback")
        # Grid data structures
        self.grid_width = 0
        self.grid_height = 0
        self.num_layers = 0
        self.layer_directions = {}  # layer_idx -> 'H' or 'V'
        
        # 3D grids [layer, y, x]
        self.occupancy = None    # uint8: 0=free, 1=obstacle, 2=routed
        self.net_id_grid = None  # int32: 0=none, >0=net_id
        self.cost_grid = None    # uint16: congestion penalty
        
        # Routing bounds
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        
        # Results
        self.tracks = []
        self.vias = []
        self.layer_mapping = {}  # KiCad layer name -> grid layer index
        self.reverse_layer_mapping = {}  # grid layer index -> KiCad layer name
        
        # Board snapshot
        self.board_pads = []  # List[Pad]
        self.routable_nets = {}  # Dict[str, List[Pad]]
        
        # Statistics
        self.routed_count = 0
        self.failed_count = 0
        self.ripup_stats = {}
        
        # Fabric-first routing system
        self.fabric_network: Optional[FabricNetwork] = None
        
    def route_board(self, board_data: Dict[str, Any], progress_callback=None) -> RouteResult:
        """
        Main routing entry point
        """
        logger.info("🚀 Starting deterministic Manhattan routing")
        self._progress_callback = progress_callback  # Store for use in sub-methods
        start_time = time.time()
        
        try:
            # Create unified board snapshot
            self.board_pads, self.routable_nets = snapshot_board(board_data)
            
            # Initialize routing grid using snapshot
            self._initialize_grid(board_data)
            
            # Generate fabric network for internal layers
            self.fabric_network = self._generate_fabric_network()
            
            # Prepare nets for routing (now using snapshot)
            nets = self._prepare_nets_from_snapshot()
            logger.info(f"Routing {len(nets)} nets on {self.grid_width}x{self.grid_height} grid")
            
            # Critical assertions and instrumentation before routing
            total_pads_in_nets = sum(len(net_data['pads']) for _, net_data in nets)
            assert total_pads_in_nets > 0, "No pads in routable nets"
            
            # Bounds checking instrumentation
            pads_in_bounds = sum(1 for pad in self.board_pads 
                               if self.min_x <= pad.x_mm <= self.max_x and self.min_y <= pad.y_mm <= self.max_y)
            logger.info(f"Pads in-bounds: {pads_in_bounds} / {len(self.board_pads)}")
            
            # Grid coordinate validation for sample pad
            if self.board_pads:
                sample_pad = self.board_pads[0]
                assert self.min_x <= sample_pad.x_mm <= self.max_x, f"Sample pad X out of bounds: {sample_pad.x_mm} not in [{self.min_x}, {self.max_x}]"
                assert self.min_y <= sample_pad.y_mm <= self.max_y, f"Sample pad Y out of bounds: {sample_pad.y_mm} not in [{self.min_y}, {self.max_y}]"
                
                gx, gy = self.world_to_grid(sample_pad.x_mm, sample_pad.y_mm)
                assert 0 <= gx < self.grid_width, f"Sample pad grid X out of bounds: {gx} not in [0, {self.grid_width})"
                assert 0 <= gy < self.grid_height, f"Sample pad grid Y out of bounds: {gy} not in [0, {self.grid_height})"
                logger.info(f"Grid coordinate validation passed: ({sample_pad.x_mm:.2f},{sample_pad.y_mm:.2f}) -> ({gx},{gy})")
            
            # Check obstacle count (0 is valid with net-aware rasterization)
            total_obstacles = np.sum(self.occ_kind == PAD)
            logger.info(f"Ready to route: {total_obstacles} obstacle cells across {self.num_layers} layers")
            
            # Route each net
            for i, (net_name, net_data) in enumerate(nets):
                if progress_callback and i % 10 == 0:
                    progress_callback(i, len(nets), f"Routing {net_name}")
                
                logger.info(f"Routing net {i+1}/{len(nets)}: {net_name}")
                
                success = self._route_net(net_name, net_data, i+1)
                if success:
                    self.routed_count += 1
                    logger.info(f"✅ Net {net_name} routed successfully")
                else:
                    self.failed_count += 1
                    logger.warning(f"❌ Failed to route net {net_name}")
                    
                    if self.failed_count >= self.config.max_global_failures:
                        logger.error(f"Stopping after {self.failed_count} failures")
                        break
                        
                # Progress update every 10 nets
                if i % 10 == 0:
                    logger.info(f"Progress: {self.routed_count} routed, {self.failed_count} failed")
            
            elapsed_time = time.time() - start_time
            logger.info(f"🏁 Routing completed in {elapsed_time:.2f}s")
            logger.info(f"📊 Results: {self.routed_count} routed, {self.failed_count} failed")
            
            # Clean up fabric network and convert claimed segments to tracks
            fabric_cleanup_stats = self._cleanup_fabric_network()
            
            return RouteResult(
                success=self.failed_count == 0,
                tracks=self.tracks,
                vias=self.vias,
                routed_nets=self.routed_count,
                failed_nets=self.failed_count,
                stats={
                    'elapsed_time': elapsed_time,
                    'total_nets': len(nets),
                    'grid_size': (self.grid_width, self.grid_height, self.num_layers),
                    'ripup_stats': self.ripup_stats,
                    'fabric_cleanup': fabric_cleanup_stats
                }
            )
            
        except Exception as e:
            logger.error(f"Manhattan routing failed: {e}")
            return RouteResult(
                success=False,
                tracks=[],
                vias=[],
                routed_nets=0,
                failed_nets=0,
                stats={'error': str(e)}
            )
    
    def _initialize_grid(self, board_data: Dict[str, Any]):
        """Initialize routing grid and data structures"""
        logger.info("Initializing routing grid...")
        
        # Calculate routing bounds from airwires
        airwires = board_data.get('airwires', [])
        if not airwires:
            raise ValueError("No airwires found for routing")
        
        # Find bounding box of all airwires
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for airwire in airwires:
            min_x = min(min_x, airwire['start_x'], airwire['end_x'])
            max_x = max(max_x, airwire['start_x'], airwire['end_x'])
            min_y = min(min_y, airwire['start_y'], airwire['end_y'])
            max_y = max(max_y, airwire['start_y'], airwire['end_y'])
        
        # Expand bounds by margin
        self.min_x = min_x - self.config.expansion_margin
        self.max_x = max_x + self.config.expansion_margin
        self.min_y = min_y - self.config.expansion_margin
        self.max_y = max_y + self.config.expansion_margin
        
        # Calculate grid dimensions
        self.grid_width = int(math.ceil((self.max_x - self.min_x) / self.config.grid_pitch))
        self.grid_height = int(math.ceil((self.max_y - self.min_y) / self.config.grid_pitch))
        
        # Set up layer mapping (include F.Cu for escape routing)
        board_layers = board_data.get('layers', 2)
        layer_names = []
        
        # Add front copper for escape routing
        layer_names.append('F.Cu')
        
        # Add internal layers
        for i in range(1, board_layers - 1):
            layer_names.append(f'In{i}.Cu')
        
        # Add back copper
        layer_names.append('B.Cu')
        
        self.num_layers = len(layer_names)
        
        # Create layer mappings
        for idx, layer_name in enumerate(layer_names):
            self.layer_mapping[layer_name] = idx
            self.reverse_layer_mapping[idx] = layer_name
        
        # Set layer directions using proper name-based mapping
        # Spec: F.Cu = V (vertical), odd internals = H, even internals + B.Cu = V
        def is_horizontal(layer_name: str) -> bool:
            if layer_name == "F.Cu":
                return False  # F.Cu is vertical for proper DRC compliance
            if layer_name == "B.Cu":
                return False  # B.Cu is vertical
            if layer_name.startswith("In"):
                try:
                    n = int(layer_name[2:-3])  # "In5.Cu" -> 5
                    return (n % 2 == 1)        # 1,3,5,7,9 -> horizontal
                except ValueError:
                    return False  # Fallback for invalid names
            return False  # Default to vertical
        
        for idx, layer_name in enumerate(layer_names):
            self.layer_directions[idx] = 'H' if is_horizontal(layer_name) else 'V'
        
        logger.info(f"Grid: {self.grid_width}x{self.grid_height}x{self.num_layers}")
        logger.info(f"Bounds: ({self.min_x:.2f}, {self.min_y:.2f}) to ({self.max_x:.2f}, {self.max_y:.2f})")
        logger.info(f"Layers: {list(self.layer_mapping.keys())}")
        
        # Log directions in the correct format
        direction_strs = []
        for idx in range(self.num_layers):
            layer_name = self.reverse_layer_mapping[idx]
            direction = self.layer_directions[idx]
            direction_strs.append(f"{layer_name}:{direction}")
        logger.info(f"Directions: {', '.join(direction_strs)}")
        
        # Critical assertions
        assert self.grid_width > 0 and self.grid_height > 0, f"Invalid grid size: {self.grid_width}x{self.grid_height}"
        assert len(layer_names) == self.num_layers, f"Layer count mismatch: {len(layer_names)} vs {self.num_layers}"
        assert self.num_layers > 0, "No routing layers available"
        
        # Initialize 3D grids with net-aware occupancy tracking
        if self.gpu_provider and self.gpu_provider.is_available():
            try:
                logger.info("Using GPU for grid initialization and storage")
                # Initialize the GPU provider if needed
                if not hasattr(self.gpu_provider, '_initialized') or not self.gpu_provider._initialized:
                    self.gpu_provider.initialize()
                
                # Create arrays on GPU
                self.occ_kind = self.gpu_provider.create_array(
                    (self.num_layers, self.grid_height, self.grid_width), 
                    dtype=np.uint8, 
                    fill_value=0
                )  # 0=free, 1=pad, 2=trace, 3=keepout
                
                self.occ_owner = self.gpu_provider.create_array(
                    (self.num_layers, self.grid_height, self.grid_width), 
                    dtype=np.int32, 
                    fill_value=0
                )  # 0=none, >0=net_id
                
                self.cost_grid = self.gpu_provider.create_array(
                    (self.num_layers, self.grid_height, self.grid_width), 
                    dtype=np.uint16, 
                    fill_value=0
                )
                
                # Log GPU memory usage
                mem_info = self.gpu_provider.get_memory_info()
                logger.info(f"GPU memory used: {mem_info.get('used_memory', 0) / 1024**2:.1f}MB")
                
            except Exception as e:
                logger.warning(f"Failed to use GPU for grid initialization: {e}")
                # Fallback to CPU arrays
                self.occ_kind = np.zeros((self.num_layers, self.grid_height, self.grid_width), dtype=np.uint8)
                self.occ_owner = np.zeros((self.num_layers, self.grid_height, self.grid_width), dtype=np.int32)
                self.cost_grid = np.zeros((self.num_layers, self.grid_height, self.grid_width), dtype=np.uint16)
        else:
            # Use CPU arrays
            self.occ_kind = np.zeros((self.num_layers, self.grid_height, self.grid_width), dtype=np.uint8)
            self.occ_owner = np.zeros((self.num_layers, self.grid_height, self.grid_width), dtype=np.int32)
            self.cost_grid = np.zeros((self.num_layers, self.grid_height, self.grid_width), dtype=np.uint16)
        
        # Legacy compatibility
        self.occupancy = self.occ_kind  # For backward compatibility
        self.net_id_grid = self.occ_owner  # For backward compatibility
        
        # Rasterize obstacles using board snapshot
        self._rasterize_obstacles_from_snapshot()
    
    def _rasterize_obstacles_from_snapshot(self):
        """Rasterize obstacles using unified board snapshot with DRC-compliant clearances"""
        logger.info("Rasterizing obstacles from board snapshot with DRC compliance...")
        
        # Get DRC constraints from board data if available
        drc_constraints = None
        if hasattr(self, 'drc_constraints'):
            drc_constraints = self.drc_constraints
        else:
            # Import and create default DRC constraints
            try:
                from ...domain.models.constraints import DRCConstraints
                drc_constraints = DRCConstraints()
                logger.warning("No DRC constraints provided, using defaults")
            except ImportError:
                logger.warning("Could not import DRC constraints, using basic clearances")
        
        # Calculate proper DRC clearance instead of minimal halo
        if drc_constraints:
            # Use DRC-compliant clearance: track width/2 + netclass clearance
            default_netclass = drc_constraints.get_netclass('Default')
            track_half_width = default_netclass.track_width / 2.0
            required_clearance = default_netclass.clearance
            total_clearance = track_half_width + required_clearance
            
            halo_cells = max(2, math.ceil(total_clearance / self.config.grid_pitch))
            logger.info(f"Using DRC clearance: track_width={default_netclass.track_width:.3f}mm, clearance={required_clearance:.3f}mm")
            logger.info(f"Total clearance: {total_clearance:.3f}mm = {halo_cells} cells")
        else:
            # Fallback to old method but with larger minimum
            halo_cells = max(2, math.ceil((self.config.track_width/2 + self.config.clearance) / self.config.grid_pitch))
            logger.info(f"Using basic clearance calculation: {halo_cells} cells")
        
        obstacle_count = 0
        pads_in_bounds = 0
        pads_out_bounds = 0
        layer_obstacle_counts = {}
        
        # Initialize layer counts
        for layer_idx in range(self.num_layers):
            layer_obstacle_counts[layer_idx] = 0
        
        for pad in self.board_pads:
            try:
                # Sanity check: ensure pad is within reasonable bounds
                if not (self.min_x - 5 <= pad.x_mm <= self.max_x + 5 and 
                        self.min_y - 5 <= pad.y_mm <= self.max_y + 5):
                    pads_out_bounds += 1
                    if pads_out_bounds <= 3:  # Log first few out-of-bounds pads
                        logger.debug(f"Pad out of bounds: ({pad.x_mm:.2f}, {pad.y_mm:.2f}) vs bounds ({self.min_x:.2f}-{self.max_x:.2f}, {self.min_y:.2f}-{self.max_y:.2f})")
                    continue
                
                pads_in_bounds += 1
                
                # Net-aware pad marking: only mark layers the pad actually exists on
                cells_marked = self._mark_pad_net_aware(pad, halo_cells, layer_obstacle_counts)
                obstacle_count += cells_marked
                                
            except Exception as e:
                logger.warning(f"Skipping invalid pad: {e}")
                continue
        
        # Instrumentation
        logger.info(f"Pads processed: {pads_in_bounds} in-bounds, {pads_out_bounds} out-of-bounds")
        logger.info(f"Rasterized {obstacle_count} obstacle cells from {len(self.board_pads)} pads")
        
        # Log obstacles per layer
        per_layer = [layer_obstacle_counts[L] for L in range(self.num_layers)]
        logger.info(f"Obstacle cells per layer: {per_layer}")
        
        for layer_idx, count in layer_obstacle_counts.items():
            layer_name = self.reverse_layer_mapping.get(layer_idx, f"Layer{layer_idx}")
            logger.info(f"  {layer_name}: {count} obstacle cells")
        
        # With net-aware rasterization, obstacles are only on layers where pads exist
        if obstacle_count == 0:
            logger.info("Net-aware rasterization: 0 obstacles - all pads will allow same-net routing")
        else:
            logger.info(f"Successfully rasterized {obstacle_count} obstacles across {len(layer_obstacle_counts)} routing layers")
        
        # Debug logging for board analysis
        if DEBUG_LOGGING_AVAILABLE:
            try:
                debug_logger = get_debug_logger()
                # Just log grid analysis information
                debug_logger.log_grid_analysis({
                    'grid_width': self.grid_width,
                    'grid_height': self.grid_height, 
                    'num_layers': self.num_layers,
                    'grid_pitch': self.config.grid_pitch,
                    'bounds': (self.min_x, self.min_y, self.max_x, self.max_y),
                    'obstacle_counts': dict(layer_obstacle_counts),
                    'layer_directions': {self.reverse_layer_mapping.get(i, f'Layer{i}'): dir 
                                        for i, dir in self.layer_directions.items()}
                })
            except Exception as e:
                logger.warning(f"Debug logging failed: {e}")
    
    def _mark_pad_net_aware(self, pad, halo_cells: int, counts: Dict[int, int]) -> int:
        """Mark pad obstacles with net-aware and layer-appropriate logic"""
        gx, gy = self.world_to_grid(pad.x_mm, pad.y_mm)
        
        # Bounds check
        if not (0 <= gx < self.grid_width and 0 <= gy < self.grid_height):
            return 0
        
        total_marked = 0
        
        # Get net ID for this pad
        net_id = self._get_net_id_for_pad(pad)
        
        # Determine which layers to block
        if pad.is_through_hole:
            # Through-hole: block all routing layers
            layers_to_block = range(self.num_layers)
        else:
            # SMD: block only layers that pad actually exists on AND are in routing grid
            layers_to_block = []
            for layer_name in pad.layer_set:
                if layer_name in self.layer_mapping:
                    layers_to_block.append(self.layer_mapping[layer_name])
        
        # Mark pad area with halo on appropriate layers
        for layer_idx in layers_to_block:
            for dy in range(-halo_cells, halo_cells + 1):
                for dx in range(-halo_cells, halo_cells + 1):
                    new_x, new_y = gx + dx, gy + dy
                    
                    # Bounds check for new coordinates
                    if (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height):
                        # Mark as pad owned by this net
                        if self.occ_kind[layer_idx, new_y, new_x] == FREE:  # Only mark if free
                            self.occ_kind[layer_idx, new_y, new_x] = PAD
                            self.occ_owner[layer_idx, new_y, new_x] = net_id
                            counts[layer_idx] += 1
                            total_marked += 1
        
        return total_marked
    
    def _get_net_id_for_pad(self, pad) -> int:
        """Get numeric net ID for a pad"""
        if not hasattr(self, '_net_name_to_id_cache'):
            self._net_name_to_id_cache = {}
            self._next_net_id = 1
        
        net_name = pad.net_name
        if net_name not in self._net_name_to_id_cache:
            self._net_name_to_id_cache[net_name] = self._next_net_id
            self._next_net_id += 1
        
        return self._net_name_to_id_cache[net_name]
    
    def _seed_from_pad(self, pad: Dict, layer_idx: int, net_id: int, max_r: int = 4) -> Optional[Tuple[int, int]]:
        """Find an escape seed near a pad on the specified layer"""
        gx, gy = self.world_to_grid(pad['x'], pad['y'])
        
        # Get preferred directions for this layer
        layer_direction = self.layer_directions.get(layer_idx, 'H')
        if layer_direction == 'H':
            # Horizontal layer: prefer horizontal escapes first
            preferred_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            # Vertical layer: prefer vertical escapes first
            preferred_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Search in expanding rings
        for r in range(1, max_r + 1):
            # Try preferred directions first
            for dx, dy in preferred_dirs:
                if abs(dx) <= r and abs(dy) <= r and (abs(dx) == r or abs(dy) == r):
                    x, y = gx + dx, gy + dy
                    if self._is_cell_free(layer_idx, y, x, net_id):
                        return (x, y)
            
            # Try all other positions in ring
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:  # On ring perimeter
                        if (dx, dy) not in preferred_dirs:  # Not already tried
                            x, y = gx + dx, gy + dy
                            if self._is_cell_free(layer_idx, y, x, net_id):
                                return (x, y)
        
        return None
    
    def _prepare_nets_from_snapshot(self) -> List[Tuple[str, Dict]]:
        """Prepare nets for routing from unified board snapshot"""
        logger.info("Preparing nets from board snapshot...")
        
        # Get airwires for distance calculation if available (optional)
        airwires = []  # Could get from board_data if needed
        
        # Sample logging for validation
        if self.board_pads:
            sample_pad = self.board_pads[0]
            logger.info(f"Sample pad: net={sample_pad.net_name} at=({sample_pad.x_mm:.2f},{sample_pad.y_mm:.2f}) thru={sample_pad.is_through_hole}")
        
        # Convert routable nets to routing format
        routable_nets = []
        for net_name, net_pads in self.routable_nets.items():
            # Calculate shortest distance for ordering
            if len(net_pads) >= 2:
                pad1, pad2 = net_pads[0], net_pads[1]
                dx = pad2.x_mm - pad1.x_mm
                dy = pad2.y_mm - pad1.y_mm
                min_dist = math.sqrt(dx*dx + dy*dy)
            else:
                min_dist = 0.0
            
            # Convert pads to routing format 
            routing_pads = []
            for pad in net_pads:
                routing_pad = {
                    'x': pad.x_mm,
                    'y': pad.y_mm,
                    'width': pad.width_mm,
                    'height': pad.height_mm,
                    'layers': list(pad.layer_set) if pad.layer_set != "THRU" else ['F.Cu', 'B.Cu'],
                    'type': 'through_hole' if pad.is_through_hole else 'smd',
                    'net': pad.net_name
                }
                routing_pads.append(routing_pad)
            
            # Create net data structure
            net_data = {
                'pads': routing_pads,
                'airwires': []  # Could add airwires if needed
            }
            
            routable_nets.append((min_dist, net_name, net_data))
        
        # Sort by distance (shortest first), then alphabetically for determinism
        routable_nets.sort(key=lambda x: (x[0], x[1]))
        
        logger.info(f"Prepared {len(routable_nets)} routable nets from snapshot")
        
        # Return without distance
        return [(name, data) for _, name, data in routable_nets]
    
    def _route_net(self, net_name: str, net_data: Dict, net_id: int) -> bool:
        """Route a single net using A* pathfinding"""
        pads = net_data.get('pads', [])
        if len(pads) < 2:
            return False
        
        # Initialize ripup counter for this net
        if net_name not in self.ripup_stats:
            self.ripup_stats[net_name] = 0
        
        # Try routing with increasing ripup attempts
        for attempt in range(self.config.max_ripups_per_net + 1):
            try:
                success = self._route_net_attempt(net_name, net_data, net_id)
                if success:
                    # Debug log successful routing
                    if DEBUG_LOGGING_AVAILABLE:
                        debug_logger = get_debug_logger()
                        # Note: detailed routing info logged within _route_net_attempt
                    return True
                    
                # If failed and have attempts left, try ripup (only if something exists to rip)
                if attempt < self.config.max_ripups_per_net:
                    # Count existing routed segments
                    routed_segments = np.sum(self.occ_kind == TRACE)
                    if routed_segments > 0:
                        logger.info(f"Attempting ripup for {net_name} (attempt {attempt + 1}) - {routed_segments} segments exist")
                        ripped_net = self._ripup_blocking_net(pads)
                        if ripped_net:
                            self.ripup_stats[net_name] += 1
                            logger.info(f"Ripped up net {ripped_net} for {net_name}")
                    else:
                        logger.info(f"Skipping ripup for {net_name} - no routed segments exist yet")
                        
            except Exception as e:
                logger.error(f"Error routing {net_name}: {e}")
                
                # Debug log routing error
                if DEBUG_LOGGING_AVAILABLE:
                    debug_logger = get_debug_logger()
                    pads = net_data.get('pads', [])
                    if len(pads) >= 2:
                        debug_logger.log_routing_attempt(net_name, pads[0], pads[1], None, str(e))
                break
        
        return False
    
    def _route_net_attempt(self, net_name: str, net_data: Dict, net_id: int) -> bool:
        """Single attempt to route a net"""
        pads = net_data.get('pads', [])
        
        # Start with first pad, route to all others
        routed_pads = set([0])  # indices of routed pads
        net_tracks = []
        net_vias = []
        
        while len(routed_pads) < len(pads):
            best_connection = None
            best_cost = float('inf')
            
            # Find closest unrouted pad to any routed pad
            for routed_idx in routed_pads:
                for unrouted_idx in range(len(pads)):
                    if unrouted_idx in routed_pads:
                        continue
                    
                    # Try to route between these pads
                    path = self._find_path(pads[routed_idx], pads[unrouted_idx], net_id)
                    if path and path['cost'] < best_cost:
                        best_cost = path['cost']
                        best_connection = (routed_idx, unrouted_idx, path)
            
            if not best_connection:
                # Can't route remaining pads
                return False
            
            # Apply best connection
            routed_idx, unrouted_idx, path = best_connection
            routed_pads.add(unrouted_idx)
            
            # Add tracks and vias to result
            net_tracks.extend(path.get('tracks', []))
            net_vias.extend(path.get('vias', []))
            
            # Mark path in grids
            self._mark_path_in_grid(path, net_id)
        
        # All pads connected successfully
        self.tracks.extend(net_tracks)
        self.vias.extend(net_vias)
        
        # Notify progress callback with new geometry (if available)  
        if hasattr(self, '_progress_callback') and self._progress_callback and net_tracks:
            self._progress_callback(0, 1, f"Routed {net_name}", tracks=net_tracks, vias=net_vias)
            
        return True
    
    def _find_path(self, start_pad: Dict, end_pad: Dict, net_id: int) -> Optional[Dict]:
        """Find path between two pads with F.Cu escape routing and blind/buried vias"""
        # Convert pads to grid coordinates
        start_gx = int((start_pad['x'] - self.min_x) / self.config.grid_pitch)
        start_gy = int((start_pad['y'] - self.min_y) / self.config.grid_pitch)
        end_gx = int((end_pad['x'] - self.min_x) / self.config.grid_pitch)
        end_gy = int((end_pad['y'] - self.min_y) / self.config.grid_pitch)
        
        # Clamp to grid bounds
        start_gx = max(0, min(self.grid_width - 1, start_gx))
        start_gy = max(0, min(self.grid_height - 1, start_gy))
        end_gx = max(0, min(self.grid_width - 1, end_gx))
        end_gy = max(0, min(self.grid_height - 1, end_gy))
        
        logger.info(f"Finding path: start_pad layers={start_pad.get('layers', [])}, end_pad layers={end_pad.get('layers', [])}")
        logger.debug(f"Grid coords: start=({start_gx},{start_gy}), end=({end_gx},{end_gy})")
        
        # Handle F.Cu pads specially - they need escape routing to internal layers
        start_layers = start_pad.get('layers', [])
        end_layers = end_pad.get('layers', [])
        
        # Check if both pads are F.Cu only
        if 'F.Cu' in start_layers and 'F.Cu' in end_layers and len(start_layers) == 1 and len(end_layers) == 1:
            logger.info("Both pads are F.Cu only - creating escape routing with blind vias")
            return self._route_fcu_to_fcu(start_pad, end_pad, net_id)
        
        # Try routing on internal layers first (excluding F.Cu for through-grid routing)
        for layer_idx in range(1, self.num_layers):  # Start from index 1 to skip F.Cu
            layer_name = self.reverse_layer_mapping.get(layer_idx, f'Layer{layer_idx}')
            logger.info(f"Trying A* search on {layer_name} (idx {layer_idx})")
            
            # Find escape seeds for pads
            start_seed = self._get_escape_seed(start_pad, layer_idx, net_id)
            end_seed = self._get_escape_seed(end_pad, layer_idx, net_id)
            
            logger.info(f"Start seed on {layer_name}: {start_seed}")
            logger.info(f"End seed on {layer_name}: {end_seed}")
            
            if start_seed is None or end_seed is None:
                logger.info(f"Skipping {layer_name} - no valid seeds found")
                continue
                
            start_sx, start_sy = start_seed
            end_sx, end_sy = end_seed
            
            path = self._astar_search(
                (layer_idx, start_sy, start_sx),
                (layer_idx, end_sy, end_sx),
                net_id
            )
            if path:
                logger.info(f"Found path on {layer_name} with cost {path['cost']}")
                # Add escape routing for F.Cu pads
                path = self._add_escape_routing(path, start_pad, end_pad, net_id)
                
                # Debug log successful routing
                if DEBUG_LOGGING_AVAILABLE:
                    debug_logger = get_debug_logger() 
                    debug_logger.log_routing_attempt("internal_route", start_pad, end_pad, path)
                
                return path
            else:
                logger.info(f"No path found on {layer_name}")
                
                # Debug log failed path attempt
                if DEBUG_LOGGING_AVAILABLE:
                    debug_logger = get_debug_logger()
                    debug_logger.log_routing_attempt(f"failed_{layer_name}", start_pad, end_pad, None, f"No path on {layer_name}")
        
        logger.info("No path found on any internal layer")
        return {
            'cost': float('inf'),
            'tracks': [],
            'vias': [],
            'path_points': []
        }
    
    def _route_fcu_to_fcu(self, start_pad: Dict, end_pad: Dict, net_id: int) -> Optional[Dict]:
        """FABRIC-FIRST: Route between F.Cu pads via fabric network"""
        logger.info("🏗️ FABRIC-FIRST: Routing F.Cu to F.Cu via fabric network")
        
        if not self.fabric_network:
            logger.error("No fabric network available - falling back to old routing")
            return self._route_fcu_to_fcu_legacy(start_pad, end_pad, net_id)
        
        # Get pad positions
        start_x, start_y = get_pos_mm(start_pad)
        end_x, end_y = get_pos_mm(end_pad)
        net_name = f"net_{net_id}"
        
        # Find nearest fabric nodes for F.Cu escapes
        start_node_id = self._find_nearest_fabric_node(start_x, start_y)
        end_node_id = self._find_nearest_fabric_node(end_x, end_y)
        
        if not start_node_id or not end_node_id:
            logger.error("Could not find fabric nodes for pads")
            return None
        
        start_node = self.fabric_network.nodes[start_node_id]
        end_node = self.fabric_network.nodes[end_node_id]
        
        logger.info(f"Start pad ({start_x:.3f}, {start_y:.3f}) -> fabric node {start_node_id} ({start_node.x:.3f}, {start_node.y:.3f})")
        logger.info(f"End pad ({end_x:.3f}, {end_y:.3f}) -> fabric node {end_node_id} ({end_node.x:.3f}, {end_node.y:.3f})")
        
        # Find path through fabric
        fabric_segment_path = self._find_fabric_path(start_node_id, end_node_id, net_name)
        
        if not fabric_segment_path:
            logger.info("No fabric path found between nodes")
            return None
        
        logger.info(f"Found fabric path with {len(fabric_segment_path)} segments")
        
        # Claim fabric segments for this net
        claimed_segments = []
        for seg_id in fabric_segment_path:
            if self.fabric_network.claim_segment(seg_id, net_name):
                claimed_segments.append(seg_id)
                logger.debug(f"Claimed segment {seg_id} for {net_name}")
            else:
                logger.warning(f"Could not claim segment {seg_id} - may already be claimed")
        
        # Generate route structure
        tracks = []
        vias = []
        path_points = []
        
        # 1. F.Cu escape from start pad to fabric
        escape_track = {
            'start_x': start_x,
            'start_y': start_y,
            'end_x': start_node.x,
            'end_y': start_node.y,
            'layer': 'F.Cu',
            'width': self.config.track_width,
            'net_id': net_id
        }
        tracks.append(escape_track)
        path_points.append((start_x, start_y))
        path_points.append((start_node.x, start_node.y))
        
        # 2. Blind via F.Cu -> In1.Cu at start
        start_via = {
            'x': start_node.x,
            'y': start_node.y,
            'drill': self.config.via_drill,
            'diameter': self.config.via_diameter,
            'from_layer': 'F.Cu',
            'to_layer': 'In1.Cu',  # First internal layer
            'net_id': net_id,
            'via_type': 'blind'
        }
        vias.append(start_via)
        
        # 3. Convert fabric segments to tracks
        current_x, current_y = start_node.x, start_node.y
        
        for seg_id in fabric_segment_path:
            segment = self.fabric_network.segments[seg_id]
            
            # Create track for this fabric segment
            fabric_track = {
                'start_x': segment.start_x,
                'start_y': segment.start_y,
                'end_x': segment.end_x,
                'end_y': segment.end_y,
                'layer': segment.layer_name,
                'width': self.config.track_width,
                'net_id': net_id
            }
            tracks.append(fabric_track)
            path_points.extend([(segment.start_x, segment.start_y), (segment.end_x, segment.end_y)])
            
            current_x, current_y = segment.end_x, segment.end_y
        
        # 4. Entry from fabric to end pad  
        entry_track = {
            'start_x': end_node.x,
            'start_y': end_node.y,
            'end_x': end_x,
            'end_y': end_y,
            'layer': 'F.Cu',
            'width': self.config.track_width,
            'net_id': net_id
        }
        tracks.append(entry_track)
        path_points.append((end_x, end_y))
        
        # 5. Blind via In1.Cu -> F.Cu at end
        end_via = {
            'x': end_node.x,
            'y': end_node.y,
            'drill': self.config.via_drill,
            'diameter': self.config.via_diameter,
            'from_layer': 'In1.Cu',
            'to_layer': 'F.Cu',
            'net_id': net_id,
            'via_type': 'blind'
        }
        vias.append(end_via)
        
        route_length = sum(((t['end_x']-t['start_x'])**2 + (t['end_y']-t['start_y'])**2)**0.5 for t in tracks)
        
        return {
            'cost': route_length + len(vias) * self.config.via_cost,
            'tracks': tracks,
            'vias': vias,
            'path_points': path_points
        }

    def _route_fcu_to_fcu_legacy(self, start_pad: Dict, end_pad: Dict, net_id: int) -> Optional[Dict]:
        """Legacy F.Cu routing (fallback only)"""
        # Find F.Cu layer index 
        fcu_idx = self.layer_mapping.get('F.Cu', 0)
        
        # Create escape route from start pad and entry route to end pad
        start_escape = self._create_fcu_escape(start_pad, net_id)
        end_entry = self._create_fcu_entry(end_pad, net_id)
        
        if not start_escape or not end_entry:
            logger.info("Failed to create F.Cu escape/entry routes")
            return {
                'cost': float('inf'),
                'tracks': [],
                'vias': [],
                'path_points': []
            }
        
        # Get grid points from escape and entry
        start_grid_point = start_escape['grid_entry']
        end_grid_point = end_entry['grid_entry']
        
        # Find a suitable internal layer for main routing based on required movement
        dx = end_grid_point[0] - start_grid_point[0]
        dy = end_grid_point[1] - start_grid_point[1]
        
        # Determine required first direction
        required_dir = 'V' if abs(dy) > abs(dx) else 'H'
        
        # Find first internal layer with matching direction
        main_layer_idx = 1  # Default fallback
        for layer_idx in range(1, self.num_layers):  # Skip F.Cu (idx 0)
            if self.layer_directions.get(layer_idx, 'H') == required_dir:
                main_layer_idx = layer_idx
                layer_name = self.reverse_layer_mapping.get(layer_idx, f'Layer{layer_idx}')
                logger.debug(f"Selected {layer_name} (direction {required_dir}) for routing dx={dx}, dy={dy}")
                break
        
        if main_layer_idx >= self.num_layers:
            main_layer_idx = self.num_layers - 1  # Fallback to last layer
        
        # Route on internal layer between escape points
        
        main_path = self._astar_search(
            (main_layer_idx, start_grid_point[1], start_grid_point[0]),
            (main_layer_idx, end_grid_point[1], end_grid_point[0]),
            net_id
        )
        
        if not main_path:
            logger.info("Failed to route on internal layer")
            return {
                'cost': float('inf'),
                'tracks': [],
                'vias': [],
                'path_points': []
            }
        
        # Combine escape route, main path, and entry route
        return self._combine_escape_and_main_path(start_escape, end_entry, main_path, net_id, main_layer_idx)
    
    def _create_fcu_escape(self, pad: Dict, net_id: int) -> Optional[Dict]:
        """Create F.Cu escape route from pad to grid entry point"""        
        pad_x, pad_y = pad['x'], pad['y']
        gx, gy = self.world_to_grid(pad_x, pad_y)
        fcu_idx = self.layer_mapping.get('F.Cu', 0)
        
        # Find escape direction on F.Cu (vertical preferred since F.Cu is typically vertical)
        escape_points = []
        
        # Get F.Cu layer direction
        fcu_direction = self.layer_directions.get(fcu_idx, 'V')  # Default to vertical
        
        # Try taxicab ring order for escape points
        for step in range(1, 6):  # Try distances 1-5 grid cells
            # Prioritize escapes in the layer's preferred direction
            if fcu_direction == 'V':  # Vertical preferred - try vertical escapes first
                direction_order = [(0, step), (0, -step), (step, 0), (-step, 0)]
            else:  # Horizontal preferred - try horizontal escapes first
                direction_order = [(step, 0), (-step, 0), (0, step), (0, -step)]
            
            for dx, dy in direction_order:
                escape_x = gx + dx
                escape_y = gy + dy
                
                if (0 <= escape_x < self.grid_width and 
                    0 <= escape_y < self.grid_height):
                    
                    # Check if F.Cu cell is free for escape trace
                    if self._is_cell_free(fcu_idx, escape_y, escape_x, net_id):
                        # Check if we can place a blind via here (target layer must be passable)
                        target_layer_idx = 1  # In1.Cu
                        if target_layer_idx < self.num_layers:
                            # Temporarily mark the via site as passable for this net
                            self._mark_via_site_passable(escape_x, escape_y, target_layer_idx, net_id)
                            
                            # Verify the via site is now usable
                            if self._is_cell_free(target_layer_idx, escape_y, escape_x, net_id):
                                escape_points.append((escape_x, escape_y, step))
                                logger.debug(f"Found viable escape point at ({escape_x},{escape_y}) step {step}")
                                break
        
        if not escape_points:
            logger.info(f"No escape points found for F.Cu pad at ({gx},{gy})")
            return None
            
        # Choose closest escape point
        escape_x, escape_y, distance = min(escape_points, key=lambda p: p[2])
        
        # Create F.Cu trace from pad to escape point with GRID-ALIGNED via placement
        world_escape_x, world_escape_y = self.grid_to_world(escape_x, escape_y)
        
        # CRITICAL: Force via to exact grid alignment for DRC compliance
        # Snap to nearest 0.4mm grid position
        grid_snap = self.config.grid_pitch
        world_escape_x = round(world_escape_x / grid_snap) * grid_snap
        world_escape_y = round(world_escape_y / grid_snap) * grid_snap
        
        # Ensure F.Cu trace is perfectly vertical (DRC requirement)
        if abs(world_escape_x - pad_x) > abs(world_escape_y - pad_y):
            # Horizontal escape - force to vertical
            world_escape_x = pad_x  # Keep X same, only change Y
        
        escape_track = {
            'start_x': pad_x,
            'start_y': pad_y,
            'end_x': world_escape_x,
            'end_y': world_escape_y,
            'layer': 'F.Cu',
            'width': self.config.track_width
        }
        
        logger.debug(f"Created F.Cu escape from ({pad_x:.2f},{pad_y:.2f}) to ({world_escape_x:.2f},{world_escape_y:.2f})")
        
        escape_result = {
            'tracks': [escape_track],
            'grid_entry': (escape_x, escape_y),
            'pad_position': (gx, gy)
        }
        
        # Debug log escape attempt
        if DEBUG_LOGGING_AVAILABLE:
            debug_logger = get_debug_logger()
            debug_logger.log_escape_attempt(pad, net_id, escape_result, "escape")
        
        return escape_result
    
    def _mark_via_site_passable(self, gx: int, gy: int, layer_idx: int, net_id: int):
        """Mark a via site as passable for a specific net (override keepout)"""
        if (0 <= gx < self.grid_width and 0 <= gy < self.grid_height and 
            0 <= layer_idx < self.num_layers):
            
            # Check current state
            current_kind = self.occ_kind[layer_idx, gy, gx]
            current_owner = self.occ_owner[layer_idx, gy, gx]
            
            # If it's a pad/keepout owned by a different net, temporarily allow this net
            if current_kind == PAD and current_owner != net_id:
                logger.debug(f"Overriding keepout at ({gx},{gy}) layer {layer_idx} for net {net_id}")
                # Mark as free for pathfinding purposes
                self.occ_kind[layer_idx, gy, gx] = FREE
                self.occ_owner[layer_idx, gy, gx] = 0
    
    def _add_via(self, route: Dict, x: float, y: float, from_layer_name: str, to_layer_name: str, net_id: int, via_type: str = 'through'):
        """Add a via to route with proper tracking and callbacks"""
        via = {
            "x": x, 
            "y": y,
            "size": self.config.via_diameter,
            "drill": self.config.via_drill,
            "from_layer": from_layer_name,
            "to_layer": to_layer_name,
            "net": net_id, 
            "via_type": via_type,
        }
        
        # Ensure route has vias list
        if 'vias' not in route:
            route['vias'] = []
        
        route["vias"].append(via)
        
        # Call via callback if available
        if hasattr(self, 'via_callback') and self.via_callback:
            self.via_callback(via)
        
        logger.debug(f"Added {via_type} via at ({x:.2f},{y:.2f}) from {from_layer_name} to {to_layer_name}")
        
        return via
    
    def _create_fcu_entry(self, pad: Dict, net_id: int) -> Optional[Dict]:
        """Create F.Cu entry route from grid entry point to pad"""
        pad_x, pad_y = pad['x'], pad['y']
        gx, gy = self.world_to_grid(pad_x, pad_y)
        fcu_idx = self.layer_mapping.get('F.Cu', 0)
        
        # Find entry direction on F.Cu (vertical preferred since F.Cu is typically vertical)
        entry_points = []
        
        # Get F.Cu layer direction
        fcu_direction = self.layer_directions.get(fcu_idx, 'V')  # Default to vertical
        
        # Try taxicab ring order for entry points  
        for step in range(1, 6):  # Try distances 1-5 grid cells
            # Prioritize entries in the layer's preferred direction
            if fcu_direction == 'V':  # Vertical preferred - try vertical entries first
                direction_order = [(0, step), (0, -step), (step, 0), (-step, 0)]
            else:  # Horizontal preferred - try horizontal entries first
                direction_order = [(step, 0), (-step, 0), (0, step), (0, -step)]
            
            for dx, dy in direction_order:
                entry_x = gx + dx
                entry_y = gy + dy
                
                if (0 <= entry_x < self.grid_width and 
                    0 <= entry_y < self.grid_height):
                    
                    # Check if F.Cu cell is free for entry trace
                    if self._is_cell_free(fcu_idx, entry_y, entry_x, net_id):
                        # Check if we can place a blind via here (target layer must be passable)
                        target_layer_idx = 1  # In1.Cu
                        if target_layer_idx < self.num_layers:
                            # Temporarily mark the via site as passable for this net
                            self._mark_via_site_passable(entry_x, entry_y, target_layer_idx, net_id)
                            
                            # Verify the via site is now usable
                            if self._is_cell_free(target_layer_idx, entry_y, entry_x, net_id):
                                entry_points.append((entry_x, entry_y, step))
                                logger.debug(f"Found viable entry point at ({entry_x},{entry_y}) step {step}")
                                break
        
        if not entry_points:
            logger.info(f"No entry points found for F.Cu pad at ({gx},{gy})")
            return None
            
        # Choose closest entry point
        entry_x, entry_y, distance = min(entry_points, key=lambda p: p[2])
        
        # Create F.Cu trace from entry point to pad with GRID-ALIGNED via placement
        world_entry_x, world_entry_y = self.grid_to_world(entry_x, entry_y)
        
        # CRITICAL: Force via to exact grid alignment for DRC compliance
        # Snap to nearest 0.4mm grid position
        grid_snap = self.config.grid_pitch
        world_entry_x = round(world_entry_x / grid_snap) * grid_snap
        world_entry_y = round(world_entry_y / grid_snap) * grid_snap
        
        # Ensure F.Cu trace is perfectly vertical (DRC requirement)
        if abs(world_entry_x - pad_x) > abs(world_entry_y - pad_y):
            # Horizontal entry - force to vertical
            world_entry_x = pad_x  # Keep X same, only change Y
        
        entry_track = {
            'start_x': world_entry_x,
            'start_y': world_entry_y,
            'end_x': pad_x,
            'end_y': pad_y,
            'layer': 'F.Cu',
            'width': self.config.track_width
        }
        
        logger.debug(f"Created F.Cu entry from ({world_entry_x:.2f},{world_entry_y:.2f}) to ({pad_x:.2f},{pad_y:.2f})")
        
        entry_result = {
            'tracks': [entry_track],
            'grid_entry': (entry_x, entry_y),
            'pad_position': (gx, gy)
        }
        
        # Debug log entry attempt
        if DEBUG_LOGGING_AVAILABLE:
            debug_logger = get_debug_logger()
            debug_logger.log_escape_attempt(pad, net_id, entry_result, "entry")
        
        return entry_result
    
    def _get_escape_seed(self, pad: Dict, layer_idx: int, net_id: int, max_r: int = 4) -> Optional[Tuple[int, int]]:
        """Get escape seed for a pad, handling F.Cu pads specially"""
        pad_layers = pad.get('layers', [])
        
        # If this is an F.Cu pad and we're routing on an internal layer,
        # create an escape route first
        if 'F.Cu' in pad_layers and layer_idx > 0:  # Internal layer
            escape_route = self._create_fcu_escape(pad, net_id)
            if escape_route:
                return escape_route['grid_entry']
        
        # For other cases, use the original seed finding logic
        return self._seed_from_pad(pad, layer_idx, net_id, max_r)
    
    def _add_escape_routing(self, path: Dict, start_pad: Dict, end_pad: Dict, net_id: int) -> Dict:
        """Add F.Cu escape/entry routing to an existing path"""
        routing_tracks = []
        
        # Create result structure for via tracking
        result = {'tracks': [], 'vias': [], 'cost': path.get('cost', 0), 'path_points': path.get('path_points', [])}
        
        # Add escape routing for start pad if F.Cu
        if 'F.Cu' in start_pad.get('layers', []):
            escape = self._create_fcu_escape(start_pad, net_id)
            if escape:
                routing_tracks.extend(escape['tracks'])
                # Add blind via from F.Cu to first internal layer
                world_x, world_y = self.grid_to_world(*escape['grid_entry'])
                self._add_via(result, world_x, world_y, 'F.Cu', 
                            self.reverse_layer_mapping.get(1, 'In1.Cu'), net_id, 'blind')
        
        # Add entry routing for end pad if F.Cu
        if 'F.Cu' in end_pad.get('layers', []):
            entry = self._create_fcu_entry(end_pad, net_id)
            if entry:
                routing_tracks.extend(entry['tracks'])
                # Add blind via from first internal layer to F.Cu
                world_x, world_y = self.grid_to_world(*entry['grid_entry'])
                self._add_via(result, world_x, world_y, 
                            self.reverse_layer_mapping.get(1, 'In1.Cu'), 'F.Cu', net_id, 'blind')
        
        # Combine with original path
        combined_tracks = routing_tracks + path.get('tracks', [])
        combined_vias = result.get('vias', []) + path.get('vias', [])
        
        return {
            'cost': path.get('cost', 0),
            'tracks': combined_tracks,
            'vias': combined_vias,
            'path_points': path.get('path_points', [])
        }
    
    def _combine_escape_and_main_path(self, start_escape: Dict, end_entry: Dict, main_path: Dict, net_id: int, main_layer_idx: int) -> Dict:
        """Combine F.Cu escape, main internal routing, and F.Cu entry"""
        all_tracks = []
        all_vias = []
        
        # Add start escape
        all_tracks.extend(start_escape['tracks'])
        start_world_x, start_world_y = self.grid_to_world(*start_escape['grid_entry'])
        
        # Create result structure to track vias
        result = {'tracks': all_tracks, 'vias': [], 'cost': main_path.get('cost', 0), 'path_points': main_path.get('path_points', [])}
        
        # Add start blind via
        start_via = self._add_via(
            result, start_world_x, start_world_y, 
            'F.Cu', self.reverse_layer_mapping.get(main_layer_idx, 'In1.Cu'),
            net_id, 'blind'
        )
        
        # Add main path
        all_tracks.extend(main_path.get('tracks', []))
        all_vias.extend(main_path.get('vias', []))
        
        # Add end entry routing
        all_tracks.extend(end_entry['tracks'])
        end_world_x, end_world_y = self.grid_to_world(*end_entry['grid_entry'])
        
        # Add end blind via (from internal layer to F.Cu entry point)
        end_via = self._add_via(
            result, end_world_x, end_world_y,
            self.reverse_layer_mapping.get(main_layer_idx, 'In1.Cu'), 'F.Cu',
            net_id, 'blind'
        )
        
        # Update tracks and vias
        result['tracks'] = all_tracks
        result['vias'].extend(all_vias)
        
        return result
    
    def _astar_search(self, start: Tuple[int, int, int], goal: Tuple[int, int, int], net_id: int) -> Optional[Dict]:
        """A* pathfinding implementation"""
        start_layer, start_y, start_x = start
        goal_layer, goal_y, goal_x = goal
        
        # Priority queue: (f_cost, g_cost, position)
        open_set = [(0, 0, start)]
        closed_set = set()
        came_from = {}  # Track parent relationships
        g_scores = {start: 0}
        
        timeout_start = time.time()
        max_iterations = 10000  # FAST-FAIL: Limit search iterations for 0.1s timeout
        iteration_count = 0
        
        while open_set:
            iteration_count += 1
            
            # AGGRESSIVE FAST-FAIL: Multiple exit conditions
            if (time.time() - timeout_start > self.config.timeout_per_net or
                iteration_count > max_iterations or
                len(closed_set) > 5000):  # Limit explored nodes
                return None  # Fast timeout
            
            f_cost, g_cost, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check if reached goal
            if current == goal:
                return self._reconstruct_path(current, came_from, g_scores)
            
            # Get the direction from parent (if any)
            last_direction = None
            if current in came_from:
                parent = came_from[current]
                if parent[0] != current[0]:  # Layer change
                    last_direction = 'V'  # Via
                elif parent[1] != current[1]:  # Y change
                    last_direction = 'V'  # Vertical
                else:  # X change
                    last_direction = 'H'  # Horizontal
            
            # Explore neighbors
            for neighbor, move_cost, direction in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Check if cell is free
                layer, y, x = neighbor
                if not self._is_cell_free(layer, y, x, net_id):
                    continue
                
                # Calculate costs
                tentative_g = g_cost + move_cost
                
                # Add bend penalty
                if last_direction and direction != last_direction:
                    tentative_g += self.config.bend_penalty
                
                # Add congestion cost
                tentative_g += self.cost_grid[layer, y, x]
                
                # Add via cost for layer changes
                if layer != current[0]:
                    tentative_g += self.config.via_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current  # Track parent
                    g_scores[neighbor] = tentative_g
                    h_cost = self._manhattan_heuristic(neighbor, goal)
                    f_cost = tentative_g + h_cost
                    
                    heapq.heappush(open_set, (f_cost, tentative_g, neighbor))
        
        return None  # No path found
    
    def _get_neighbors(self, position: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], int, str]]:
        """Get valid neighbors for A* search with blind/buried via support"""
        layer, y, x = position
        neighbors = []
        
        # Same layer moves (based on layer direction)
        direction = self.layer_directions[layer]
        if direction == 'H':  # Horizontal moves
            if x > 0:
                neighbors.append(((layer, y, x-1), 1, 'H'))
            if x < self.grid_width - 1:
                neighbors.append(((layer, y, x+1), 1, 'H'))
        else:  # Vertical moves
            if y > 0:
                neighbors.append(((layer, y-1, x), 1, 'V'))
            if y < self.grid_height - 1:
                neighbors.append(((layer, y+1, x), 1, 'V'))
        
        # Enhanced via transitions: support blind and buried vias
        # Adjacent layer transitions (standard vias)
        if layer > 0:  # Can go to layer below
            neighbors.append(((layer-1, y, x), self.config.via_cost, 'Via'))
        if layer < self.num_layers - 1:  # Can go to layer above
            neighbors.append(((layer+1, y, x), self.config.via_cost, 'Via'))
        
        # Blind vias: from F.Cu (layer 0) to internal layers
        if layer == 0:  # F.Cu layer
            for target_layer in range(1, min(4, self.num_layers)):  # Blind vias to first few internals
                via_cost = self.config.via_cost + (target_layer * 2)  # Higher cost for deeper blind vias
                neighbors.append(((target_layer, y, x), via_cost, 'BlindVia'))
        
        # Buried vias: between internal layers (skip multiple layers)
        if 0 < layer < self.num_layers - 1:  # Internal layers only
            # Allow buried vias that skip 1-2 layers
            for target_layer in range(max(1, layer - 3), min(self.num_layers - 1, layer + 4)):
                if target_layer != layer and abs(target_layer - layer) > 1:  # Skip adjacent (already handled)
                    buried_cost = self.config.via_cost + (abs(target_layer - layer) * 3)  # Cost based on layer span
                    neighbors.append(((target_layer, y, x), buried_cost, 'BuriedVia'))
        
        return neighbors
    
    def _is_cell_free(self, layer: int, y: int, x: int, net_id: int) -> bool:
        """Check if a grid cell is free for routing with net-aware logic"""
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return False
        
        kind = self.occ_kind[layer, y, x]
        owner = self.occ_owner[layer, y, x]
        
        # Free cell
        if kind == FREE:
            return True
            
        # Same-net pad (can be used as escape origin/destination)
        if kind == PAD and owner == net_id:
            return True
            
        # Same-net trace (can route through own traces)  
        if kind == TRACE and owner == net_id:
            return True
        
        # Everything else is blocked
        return False
    
    def _manhattan_heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> int:
        """Manhattan distance heuristic"""
        l1, y1, x1 = pos1
        l2, y2, x2 = pos2
        return abs(x2 - x1) + abs(y2 - y1) + abs(l2 - l1) * self.config.via_cost
    
    def _reconstruct_path(self, goal, came_from, g_scores) -> Dict:
        """Reconstruct path from A* search"""
        path_points = []
        current = goal
        
        # Build path from goal back to start
        while current is not None:
            path_points.append(current)
            current = came_from.get(current)
        
        path_points.reverse()  # Start to goal order
        
        if len(path_points) < 2:
            return {
                'cost': g_scores.get(goal, float('inf')),
                'tracks': [],
                'vias': [],
                'path_points': path_points
            }
        
        tracks = []
        vias = []
        
        # Convert path points to tracks and vias
        for i in range(len(path_points) - 1):
            current_point = path_points[i]
            next_point = path_points[i + 1]
            
            curr_layer, curr_y, curr_x = current_point
            next_layer, next_y, next_x = next_point
            
            # Convert grid coordinates to world coordinates
            curr_world_x = self.min_x + curr_x * self.config.grid_pitch
            curr_world_y = self.min_y + curr_y * self.config.grid_pitch
            next_world_x = self.min_x + next_x * self.config.grid_pitch
            next_world_y = self.min_y + next_y * self.config.grid_pitch
            
            # Layer change - create via with proper type
            if curr_layer != next_layer:
                from_layer = self._get_layer_name(curr_layer)
                to_layer = self._get_layer_name(next_layer)
                
                # Determine via type
                via_type = 'through'  # Default
                if from_layer == 'F.Cu' or to_layer == 'F.Cu':
                    if from_layer != 'B.Cu' and to_layer != 'B.Cu':
                        via_type = 'blind'
                elif from_layer != 'F.Cu' and to_layer != 'F.Cu' and from_layer != 'B.Cu' and to_layer != 'B.Cu':
                    via_type = 'buried'
                
                via = {
                    'x': curr_world_x,
                    'y': curr_world_y,
                    'from_layer': from_layer,
                    'to_layer': to_layer,
                    'size': self.config.via_size,
                    'drill': self.config.via_drill,
                    'via_type': via_type
                }
                vias.append(via)
            
            # Same layer - create track segment
            if curr_layer == next_layer and (curr_x != next_x or curr_y != next_y):
                track = {
                    'start_x': curr_world_x,
                    'start_y': curr_world_y,
                    'end_x': next_world_x,
                    'end_y': next_world_y,
                    'layer': self._get_layer_name(curr_layer),
                    'width': self.config.track_width
                }
                tracks.append(track)
        
        return {
            'cost': g_scores.get(goal, 0),
            'tracks': tracks,
            'vias': vias,
            'path_points': path_points
        }
    
    def _get_layer_name(self, layer_idx: int) -> str:
        """Get layer name from layer index"""
        return self.reverse_layer_mapping.get(layer_idx, f'Layer{layer_idx}')
    
    def _mark_path_in_grid(self, path: Dict, net_id: int):
        """Mark the routed path in the occupancy grid"""
        if 'path_points' not in path:
            logger.warning(f"No path points found for net {net_id}")
            return
            
        path_points = path['path_points']
        cells_marked = 0
        
        for point in path_points:
            layer, y, x = point
            
            # Bounds check
            if (0 <= layer < self.num_layers and 
                0 <= y < self.grid_height and 
                0 <= x < self.grid_width):
                
                # Mark cell as occupied by this net
                self.occupancy[layer, y, x] = 1
                self.net_id_grid[layer, y, x] = net_id
                cells_marked += 1
        
        logger.debug(f"Marked {cells_marked} cells for net {net_id}")
        
        # Also mark via locations if any
        for via in path.get('vias', []):
            via_x = via['x']
            via_y = via['y']
            grid_x, grid_y = self.world_to_grid(via_x, via_y)
            
            # Mark via footprint on both layers
            from_layer_idx = None
            to_layer_idx = None
            
            for idx, layer_name in self.reverse_layer_mapping.items():
                if layer_name == via['from_layer']:
                    from_layer_idx = idx
                if layer_name == via['to_layer']:
                    to_layer_idx = idx
            
            # Mark cells with via clearance
            halo = max(1, math.ceil(self.config.via_size / 2 / self.config.grid_pitch))
            
            for dy in range(-halo, halo + 1):
                for dx in range(-halo, halo + 1):
                    new_y, new_x = grid_y + dy, grid_x + dx
                    
                    if (0 <= new_y < self.grid_height and 0 <= new_x < self.grid_width):
                        if from_layer_idx is not None:
                            self.occupancy[from_layer_idx, new_y, new_x] = 1
                            self.net_id_grid[from_layer_idx, new_y, new_x] = net_id
                        if to_layer_idx is not None:
                            self.occupancy[to_layer_idx, new_y, new_x] = 1  
                            self.net_id_grid[to_layer_idx, new_y, new_x] = net_id
    
    def _ripup_blocking_net(self, target_pads: List[Dict]) -> Optional[str]:
        """Find and rip up a blocking net for rip-up/repair"""
        if len(target_pads) < 2:
            return None
            
        # Find the bounding box of target pads
        pad_positions = []
        for pad in target_pads:
            grid_x, grid_y = self.world_to_grid(pad['x'], pad['y'])
            pad_positions.append((grid_x, grid_y))
        
        if not pad_positions:
            return None
            
        min_x = min(pos[0] for pos in pad_positions)
        max_x = max(pos[0] for pos in pad_positions)
        min_y = min(pos[1] for pos in pad_positions)
        max_y = max(pos[1] for pos in pad_positions)
        
        # Expand search area slightly
        margin = max(3, (max_x - min_x + max_y - min_y) // 4)
        min_x = max(0, min_x - margin)
        max_x = min(self.grid_width - 1, max_x + margin)
        min_y = max(0, min_y - margin)
        max_y = min(self.grid_height - 1, max_y + margin)
        
        # Count net usage in the target area
        net_usage_count = {}
        
        for layer in range(self.num_layers):
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if self.occupancy[layer, y, x] == 1:
                        net_id = self.net_id_grid[layer, y, x]
                        if net_id != 0:  # 0 means obstacle, not a net
                            net_usage_count[net_id] = net_usage_count.get(net_id, 0) + 1
        
        if not net_usage_count:
            logger.debug("No nets found to rip up in target area")
            return None
        
        # AGGRESSIVE RIPUP: Select net with most usage in the area (most blocking)
        # But prioritize recently routed nets (easier to re-route)
        sorted_nets = sorted(net_usage_count.keys(), 
                           key=lambda k: (net_usage_count[k], -k),  # Higher usage first, recent nets second
                           reverse=True)
        
        best_net = sorted_nets[0]
        
        # FAST RIPUP: Only clear cells in the blocking area + small margin (not entire board)
        ripup_margin = 2
        clear_min_x = max(0, min_x - ripup_margin)
        clear_max_x = min(self.grid_width - 1, max_x + ripup_margin)
        clear_min_y = max(0, min_y - ripup_margin)
        clear_max_y = min(self.grid_height - 1, max_y + ripup_margin)
        
        cells_cleared = 0
        for layer in range(self.num_layers):
            for y in range(clear_min_y, clear_max_y + 1):
                for x in range(clear_min_x, clear_max_x + 1):
                    if self.net_id_grid[layer, y, x] == best_net:
                        self.occupancy[layer, y, x] = 0
                        self.net_id_grid[layer, y, x] = 0
                        cells_cleared += 1
        
        # Remove from routed_nets if it exists
        if hasattr(self, 'routed_nets') and best_net in self.routed_nets:
            del self.routed_nets[best_net]
        
        logger.debug(f"Ripped up net {best_net} (cleared {cells_cleared} cells, usage in area: {net_usage_count[best_net]})")
        
        # Return the net name for re-routing
        for net_name, net_id in getattr(self, 'net_name_to_id', {}).items():
            if net_id == best_net:
                return net_name
                
        return f"net_{best_net}"  # Fallback

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.min_x) / self.config.grid_pitch)
        grid_y = int((y - self.min_y) / self.config.grid_pitch)
        return max(0, min(self.grid_width - 1, grid_x)), max(0, min(self.grid_height - 1, grid_y))
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x = self.min_x + grid_x * self.config.grid_pitch
        y = self.min_y + grid_y * self.config.grid_pitch
        return x, y

    def is_horizontal(self, layer_name: str) -> bool:
        """Determine if a layer has horizontal traces"""
        if layer_name == "F.Cu":
            return False  # F.Cu is vertical for proper DRC compliance
        if layer_name == "B.Cu":
            return False  # B.Cu is vertical
        if layer_name.startswith("In"):
            try:
                n = int(layer_name[2:-3])  # "In5.Cu" -> 5
                return (n % 2 == 1)        # 1,3,5,7,9 -> horizontal
            except ValueError:
                return False  # Fallback for invalid names
        return False  # Default to vertical

    def _generate_fabric_network(self) -> FabricNetwork:
        """Generate complete routing fabric on internal layers"""
        logger.info("🏗️ Generating routing fabric on internal layers")
        
        segments = {}
        nodes = {}
        layer_segments = {}
        
        # Define fabric spacing (2x grid pitch for DRC clearance)
        fabric_spacing = self.config.grid_pitch * 2.0  # 0.8mm spacing
        
        # Get internal layer names (exclude F.Cu and B.Cu)
        internal_layers = []
        for layer_name, layer_idx in self.layer_mapping.items():
            if layer_name not in ['F.Cu', 'B.Cu']:
                internal_layers.append((layer_name, layer_idx))
        
        # Add B.Cu to the fabric system
        if 'B.Cu' in self.layer_mapping:
            internal_layers.append(('B.Cu', self.layer_mapping['B.Cu']))
        
        logger.info(f"Creating fabric on {len(internal_layers)} layers: {[name for name, _ in internal_layers]}")
        
        # Generate fabric segments for each layer
        for layer_name, layer_idx in internal_layers:
            layer_segments[layer_name] = []
            is_horizontal = self.is_horizontal(layer_name)
            
            if is_horizontal:
                # Horizontal traces: one continuous trace per row
                y = self.min_y + fabric_spacing
                row = 0
                while y < self.max_y - fabric_spacing:
                    x_start = self.min_x + fabric_spacing
                    x_end = self.max_x - fabric_spacing
                    
                    # Create ONE continuous horizontal trace across entire width
                    seg_id = f"{layer_name}_H_{row}"
                    
                    segment = FabricSegment(
                        segment_id=seg_id,
                        layer_name=layer_name,
                        start_x=x_start,
                        start_y=y,
                        end_x=x_end,
                        end_y=y,
                        is_horizontal=True
                    )
                    segments[seg_id] = segment
                    layer_segments[layer_name].append(seg_id)
                    
                    y += fabric_spacing
                    row += 1
            else:
                # Vertical traces: one continuous trace per column
                x = self.min_x + fabric_spacing
                col = 0
                while x < self.max_x - fabric_spacing:
                    y_start = self.min_y + fabric_spacing
                    y_end = self.max_y - fabric_spacing
                    
                    # Create ONE continuous vertical trace across entire height
                    seg_id = f"{layer_name}_V_{col}"
                    
                    segment = FabricSegment(
                        segment_id=seg_id,
                        layer_name=layer_name,
                        start_x=x,
                        start_y=y_start,
                        end_x=x,
                        end_y=y_end,
                        is_horizontal=False
                    )
                    segments[seg_id] = segment
                    layer_segments[layer_name].append(seg_id)
                    
                    x += fabric_spacing
                    col += 1
        
        # Generate via nodes at fabric intersections
        node_id_counter = 0
        for x in [self.min_x + fabric_spacing + i * fabric_spacing 
                  for i in range(int((self.max_x - self.min_x - 2*fabric_spacing) / fabric_spacing))]:
            for y in [self.min_y + fabric_spacing + i * fabric_spacing 
                      for i in range(int((self.max_y - self.min_y - 2*fabric_spacing) / fabric_spacing))]:
                
                node_id = f"node_{node_id_counter}"
                node_layers = [name for name, _ in internal_layers]
                connected_segs = []
                
                # Find segments that connect to this node
                for seg_id, segment in segments.items():
                    if ((abs(segment.start_x - x) < 0.01 and abs(segment.start_y - y) < 0.01) or
                        (abs(segment.end_x - x) < 0.01 and abs(segment.end_y - y) < 0.01)):
                        connected_segs.append(seg_id)
                
                if connected_segs:  # Only create nodes that connect segments
                    nodes[node_id] = FabricNode(
                        node_id=node_id,
                        x=x,
                        y=y,
                        layers=node_layers,
                        connected_segments=connected_segs
                    )
                    node_id_counter += 1
        
        fabric = FabricNetwork(
            segments=segments,
            nodes=nodes,
            layer_segments=layer_segments
        )
        
        total_segments = len(segments)
        total_nodes = len(nodes)
        logger.info(f"✅ Generated fabric: {total_segments} segments, {total_nodes} nodes")
        
        return fabric

    def _find_fabric_taps(self, start_x: float, start_y: float, end_x: float, end_y: float) -> List[Dict]:
        """Find tap points where a net connects to fabric traces"""
        if not self.fabric_network:
            return []
        
        taps = []
        
        # Find horizontal fabric traces that can carry this net
        for layer_name, segment_ids in self.fabric_network.layer_segments.items():
            if not self.is_horizontal(layer_name):
                continue  # Skip vertical layers for now, handle in next step
                
            for seg_id in segment_ids:
                segment = self.fabric_network.segments[seg_id]
                
                # Check if this horizontal trace is between start and end Y coordinates
                trace_y = segment.start_y
                if min(start_y, end_y) <= trace_y <= max(start_y, end_y):
                    # This trace can be tapped
                    tap_x = (start_x + end_x) / 2  # Tap at midpoint X
                    taps.append({
                        'layer': layer_name,
                        'trace_id': seg_id,
                        'tap_x': tap_x,
                        'tap_y': trace_y,
                        'trace_start': (segment.start_x, segment.start_y),
                        'trace_end': (segment.end_x, segment.end_y)
                    })
        
        # Find vertical fabric traces
        for layer_name, segment_ids in self.fabric_network.layer_segments.items():
            if self.is_horizontal(layer_name):
                continue  # Skip horizontal layers
                
            for seg_id in segment_ids:
                segment = self.fabric_network.segments[seg_id]
                
                # Check if this vertical trace is between start and end X coordinates  
                trace_x = segment.start_x
                if min(start_x, end_x) <= trace_x <= max(start_x, end_x):
                    # This trace can be tapped
                    tap_y = (start_y + end_y) / 2  # Tap at midpoint Y
                    taps.append({
                        'layer': layer_name,
                        'trace_id': seg_id,
                        'tap_x': trace_x,
                        'tap_y': tap_y,
                        'trace_start': (segment.start_x, segment.start_y),
                        'trace_end': (segment.end_x, segment.end_y)
                    })
        
        return taps
    
    def _find_nearest_fabric_node(self, x: float, y: float) -> Optional[str]:
        """Find nearest fabric node to given coordinates"""
        if not self.fabric_network or not self.fabric_network.nodes:
            return None
        
        min_distance = float('inf')
        nearest_node_id = None
        
        for node_id, node in self.fabric_network.nodes.items():
            dx = node.x - x
            dy = node.y - y
            distance = dx * dx + dy * dy  # Squared distance for efficiency
            
            if distance < min_distance:
                min_distance = distance
                nearest_node_id = node_id
        
        return nearest_node_id

    def _cleanup_fabric_network(self) -> Dict[str, int]:
        """Remove unused fabric segments and convert claimed segments to actual tracks"""
        if not self.fabric_network:
            return {'removed_segments': 0, 'converted_tracks': 0, 'added_vias': 0}
        
        logger.info("🧹 Cleaning up fabric network")
        
        removed_count = 0
        converted_tracks = 0
        added_vias = 0
        
        # Convert claimed fabric segments to actual routing tracks
        for seg_id, segment in self.fabric_network.segments.items():
            if segment.status == "CLAIMED" and segment.claimed_by_net:
                # Convert fabric segment to track
                track = Track(
                    start_x=segment.start_x,
                    start_y=segment.start_y,
                    end_x=segment.end_x,
                    end_y=segment.end_y,
                    layer=segment.layer_name,
                    width=self.config.track_width,
                    net_id=int(segment.claimed_by_net.split('_')[-1]) if segment.claimed_by_net.startswith('net_') else 0
                )
                self.tracks.append(track)
                converted_tracks += 1
                logger.debug(f"Converted fabric segment {seg_id} to track on {segment.layer_name}")
        
        # Add inter-layer vias at fabric nodes where multiple layers are used by the same net
        net_usage_per_node = {}  # node_id -> set of net_names using this node
        
        for seg_id, segment in self.fabric_network.segments.items():
            if segment.status == "CLAIMED" and segment.claimed_by_net:
                # Find nodes connected to this segment
                for node_id, node in self.fabric_network.nodes.items():
                    if seg_id in node.connected_segments:
                        if node_id not in net_usage_per_node:
                            net_usage_per_node[node_id] = set()
                        net_usage_per_node[node_id].add(segment.claimed_by_net)
        
        # Add vias between fabric layers where nets transition
        for node_id, nets_using_node in net_usage_per_node.items():
            if len(nets_using_node) == 1:  # Only one net uses this node
                net_name = list(nets_using_node)[0]
                node = self.fabric_network.nodes[node_id]
                
                # Check if this net uses multiple layers at this node
                layers_used = set()
                for seg_id in node.connected_segments:
                    segment = self.fabric_network.segments[seg_id]
                    if segment.status == "CLAIMED" and segment.claimed_by_net == net_name:
                        layers_used.add(segment.layer_name)
                
                # Add vias between adjacent layers
                if len(layers_used) > 1:
                    sorted_layers = sorted(layers_used, key=lambda x: self.layer_mapping.get(x, 0))
                    for i in range(len(sorted_layers) - 1):
                        from_layer = sorted_layers[i]
                        to_layer = sorted_layers[i + 1]
                        
                        via = Via(
                            x=node.x,
                            y=node.y,
                            drill=self.config.via_drill,
                            diameter=self.config.via_diameter,
                            from_layer=from_layer,
                            to_layer=to_layer,
                            net_id=int(net_name.split('_')[-1]) if net_name.startswith('net_') else 0
                        )
                        self.vias.append(via)
                        added_vias += 1
                        logger.debug(f"Added fabric via {from_layer}->{to_layer} at ({node.x:.3f}, {node.y:.3f})")
        
        # Remove unclaimed fabric segments (they're not needed)
        segments_to_remove = []
        for seg_id, segment in self.fabric_network.segments.items():
            if segment.status == "AVAILABLE":  # Unused fabric
                segments_to_remove.append(seg_id)
        
        for seg_id in segments_to_remove:
            del self.fabric_network.segments[seg_id]
            removed_count += 1
        
        # Clean up layer_segments mapping
        for layer_name in self.fabric_network.layer_segments:
            self.fabric_network.layer_segments[layer_name] = [
                seg_id for seg_id in self.fabric_network.layer_segments[layer_name]
                if seg_id in self.fabric_network.segments
            ]
        
        logger.info(f"✅ Fabric cleanup: {removed_count} unused segments removed, {converted_tracks} fabric tracks converted, {added_vias} vias added")
        
        return {
            'removed_segments': removed_count, 
            'converted_tracks': converted_tracks,
            'added_vias': added_vias
        }