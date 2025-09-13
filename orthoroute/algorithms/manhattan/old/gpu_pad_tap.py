"""
GPU-Accelerated PadTap Builder for Vertical Pad Escapes
Generates DRC-aware vertical tap candidates for F.Cu pads to connect to Manhattan fabric
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CuPy available - GPU PadTap generation enabled")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    logger.warning("CuPy not available - CPU PadTap fallback")

@dataclass
class TapCandidate:
    """Represents a potential tap point for pad connection"""
    tap_x: float              # Grid-aligned X position
    tap_y: float              # Y position (same as pad)  
    via_layers: Tuple[int, int]  # (start_layer, end_layer) for via
    escape_length: float      # Horizontal distance from pad to tap
    total_cost: float         # Routing cost including penalties
    blocked_layers: Set[int]  # Layers blocked by this via

@dataclass
class PadTapConfig:
    """Configuration for PadTap generation"""
    # Cost parameters
    k_fc_len: float = 0.5         # F.Cu trace length penalty
    k_fc_horiz: float = 1.8       # Horizontal escape penalty  
    k_bend: float = 1.0           # Corner/bend penalty
    k_via: float = 10.0           # Via penalty (high for reliability)
    
    # Layer penalties (encourage shallower routing)
    band_penalty: List[float] = None  # [0, 0.3, 0.6, 1.0, ...]
    
    # Search parameters
    vertical_reach: int = 15      # Grid cells to search around pad (2.5mm / 0.2mm = 12.5, rounded up)
    tap_stride: int = 1           # Grid cell spacing for tap candidates
    max_taps_per_pad: int = 8     # Limit candidates per pad (reduced for performance)
    
    # DRC parameters  
    pad_clearance: float = 0.1    # mm clearance from pad edges
    trace_width: float = 0.089    # Default trace width
    via_clearance: float = 0.1    # mm clearance around vias
    
    def __post_init__(self):
        if self.band_penalty is None:
            # Default layer penalties - encourage shallower routing
            self.band_penalty = [0.0, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0, 2.3, 2.6, 3.0, 3.3, 3.6]

class GPUPadTapBuilder:
    """GPU-accelerated vertical pad tap generation for backplane routing"""
    
    def __init__(self, config: PadTapConfig):
        self.config = config
        self.use_gpu = GPU_AVAILABLE
        
        # GPU arrays for computation
        self.pad_positions = None      # [num_pads, 2] - (x, y) coordinates
        self.grid_rails = None         # [num_rails] - sorted vertical rail X positions
        self.tap_candidates = None     # [num_pads, max_taps, tap_data]
        self.tap_costs = None          # [num_pads, max_taps] - routing costs
        
        # DRC collision detection
        self.pad_bounds = None         # [num_pads, 4] - (min_x, min_y, max_x, max_y)
        
        logger.info(f"GPU PadTap builder initialized (GPU: {self.use_gpu})")
        
    def build_tap_candidates_for_net(self, net_pads: List[Dict], grid_x_positions: List[float],
                                   num_layers: int = 12) -> List[TapCandidate]:
        """Generate tap candidates for pads belonging to a specific net"""
        
        logger.info(f"Generating tap candidates for {len(net_pads)} pads in current net")
        start_time = time.time()
        
        # Convert input data to GPU arrays for this net only
        self._prepare_gpu_data(net_pads, grid_x_positions, num_layers)
        
        # Generate tap candidates
        if self.use_gpu:
            candidates = self._gpu_generate_taps_optimized()
        else:
            candidates = self._cpu_generate_taps_optimized()
        
        # Convert to flat list of TapCandidate objects
        tap_candidates = []
        for pad_idx, pad in enumerate(net_pads):
            pad_taps = candidates[pad_idx]
            
            for tap_idx in range(self.config.max_taps_per_pad):
                tap_info = pad_taps[tap_idx]
                
                # Skip empty entries
                if float(tap_info[5]) == 0.0:
                    break
                
                tap_x, tap_y = float(tap_info[0]), float(tap_info[1])
                start_layer, end_layer = int(tap_info[2]), int(tap_info[3])
                escape_length, cost = float(tap_info[4]), float(tap_info[5])
                
                blocked_layers = set(range(start_layer, end_layer + 1))
                
                candidate = TapCandidate(
                    tap_x=tap_x,
                    tap_y=tap_y,
                    via_layers=(start_layer, end_layer),
                    escape_length=escape_length,
                    total_cost=cost,
                    blocked_layers=blocked_layers
                )
                
                tap_candidates.append(candidate)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated {len(tap_candidates)} tap candidates for net in {generation_time:.3f}s")
        
        return tap_candidates

    def build_tap_candidates(self, pads: List[Dict], grid_x_positions: List[float], 
                           num_layers: int = 12) -> Dict[str, List[TapCandidate]]:
        """Generate vertical tap candidates for all pads"""
        
        logger.info(f"Generating tap candidates for {len(pads)} pads")
        start_time = time.time()
        
        # Convert input data to GPU arrays
        self._prepare_gpu_data(pads, grid_x_positions, num_layers)
        
        # Generate tap candidates (optimized for large pad counts)
        if self.use_gpu:
            candidates = self._gpu_generate_taps_optimized()
        else:
            candidates = self._cpu_generate_taps_optimized()
        
        # Convert back to structured format
        results = self._format_results(pads, candidates)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated {sum(len(taps) for taps in results.values())} tap candidates in {generation_time:.2f}s")
        
        return results
    
    def _prepare_gpu_data(self, pads: List[Dict], grid_x_positions: List[float], num_layers: int):
        """Convert input data to GPU arrays"""
        
        # Extract pad positions and create bounds for DRC
        positions = []
        bounds = []
        
        for pad in pads:
            x, y = pad['x'], pad['y'] 
            width = pad.get('width', 1.0)
            height = pad.get('height', 1.0)
            
            positions.append([x, y])
            
            # Pad bounds with clearance for DRC checking
            clearance = self.config.pad_clearance
            bounds.append([
                x - width/2 - clearance,   # min_x  
                y - height/2 - clearance,  # min_y
                x + width/2 + clearance,   # max_x
                y + height/2 + clearance   # max_y
            ])
        
        # Convert to GPU arrays
        if self.use_gpu:
            self.pad_positions = cp.array(positions, dtype=cp.float32)
            self.pad_bounds = cp.array(bounds, dtype=cp.float32)
            self.grid_rails = cp.array(sorted(grid_x_positions), dtype=cp.float32)
        else:
            self.pad_positions = np.array(positions, dtype=np.float32)
            self.pad_bounds = np.array(bounds, dtype=np.float32)  
            self.grid_rails = np.array(sorted(grid_x_positions), dtype=np.float32)
        
        self.num_layers = num_layers
        logger.info(f"Prepared GPU data: {len(pads)} pads, {len(grid_x_positions)} rail positions")
    
    def _gpu_generate_taps_optimized(self) -> cp.ndarray:
        """Generate tap candidates using GPU acceleration"""
        
        num_pads = len(self.pad_positions)
        max_taps = self.config.max_taps_per_pad
        
        # Allocate output arrays
        # Each tap: [tap_x, tap_y, start_layer, end_layer, escape_len, cost]
        tap_data = cp.zeros((num_pads, max_taps, 6), dtype=cp.float32)
        
        # Optimized batch processing for large pad counts
        logger.info(f"Processing {num_pads} pads in batches...")
        
        batch_size = 1000  # Process pads in smaller batches
        num_batches = (num_pads + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_pads)
            
            # Process this batch of pads
            for pad_idx in range(start_idx, end_idx):
                candidates = self._generate_pad_taps_gpu_fast(pad_idx)
                
                # Store top candidates (sorted by cost)
                num_candidates = min(len(candidates), max_taps)
                for i in range(num_candidates):
                    tap_data[pad_idx, i] = candidates[i]
            
            # Progress logging
            if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                progress = (end_idx / num_pads) * 100
                logger.info(f"Tap generation progress: {progress:.1f}% ({end_idx}/{num_pads} pads)")
        
        return tap_data
    
    def _generate_pad_taps_gpu_fast(self, pad_idx: int) -> List[cp.ndarray]:
        """Generate tap candidates for a single pad (GPU kernel simulation)"""
        
        pad_x, pad_y = float(self.pad_positions[pad_idx, 0]), float(self.pad_positions[pad_idx, 1])
        candidates = []
        
        # Optimized rail search - only consider nearest rails
        reach = self.config.vertical_reach * 0.2  # Convert grid cells to mm (0.2mm grid)
        
        # Find 3 closest rails (left, exact, right) for efficiency
        rail_distances = cp.abs(self.grid_rails - pad_x)
        closest_indices = cp.argsort(rail_distances)[:3]  # Top 3 closest rails
        nearby_rails = self.grid_rails[closest_indices]
        
        # Only use rails within reach
        nearby_rails = nearby_rails[cp.abs(nearby_rails - pad_x) <= reach]
        
        # Generate tap candidates at each nearby rail (use full layer stack for proper negotiation)
        max_layer_depth = self.num_layers  # Use all available layers for proper PathFinder negotiation
        
        for rail_x in nearby_rails:
            rail_x = float(rail_x)
            escape_length = abs(rail_x - pad_x)
            
            # Skip if escape is too long
            if escape_length > reach:
                continue
            
            # Generate via options (fewer layer options for speed)
            for end_layer in range(1, max_layer_depth):  # In1.Cu through In3.Cu initially
                start_layer = 0  # F.Cu
                
                # Calculate costs (simplified for speed)
                via_cost = self.config.k_via
                if end_layer < len(self.config.band_penalty):
                    via_cost += self.config.band_penalty[end_layer]
                
                escape_cost = escape_length * (self.config.k_fc_len + self.config.k_fc_horiz)
                total_cost = via_cost + escape_cost
                
                # Create candidate
                candidate = cp.array([
                    rail_x, pad_y, start_layer, end_layer, escape_length, total_cost
                ], dtype=cp.float32)
                
                candidates.append(candidate)
                
                # Limit candidates per pad for performance
                if len(candidates) >= 8:  # Reduced from max_taps_per_pad
                    break
            
            if len(candidates) >= 8:
                break
        
        # Sort candidates by cost (lowest first)
        if candidates:
            candidates.sort(key=lambda c: float(c[5]))
        
        return candidates[:self.config.max_taps_per_pad]
    
    def _cpu_generate_taps_optimized(self) -> np.ndarray:
        """CPU fallback for tap generation"""
        
        num_pads = len(self.pad_positions)
        max_taps = self.config.max_taps_per_pad
        
        # Same logic as GPU version but using NumPy
        tap_data = np.zeros((num_pads, max_taps, 6), dtype=np.float32)
        
        for pad_idx in range(num_pads):
            candidates = self._generate_pad_taps_cpu(pad_idx)
            
            num_candidates = min(len(candidates), max_taps)
            for i in range(num_candidates):
                tap_data[pad_idx, i] = candidates[i]
        
        return tap_data
    
    def _generate_pad_taps_cpu(self, pad_idx: int) -> List[np.ndarray]:
        """CPU version of single pad tap generation"""
        
        pad_x, pad_y = float(self.pad_positions[pad_idx, 0]), float(self.pad_positions[pad_idx, 1])
        candidates = []
        
        # Find nearby vertical rails
        reach = self.config.vertical_reach * 0.2  # mm
        rail_mask = np.abs(self.grid_rails - pad_x) <= reach
        nearby_rails = self.grid_rails[rail_mask]
        
        for rail_x in nearby_rails:
            rail_x = float(rail_x)
            escape_length = abs(rail_x - pad_x)
            
            if escape_length > reach:
                continue
            
            # Generate via depth options
            for end_layer in range(1, self.num_layers):
                start_layer = 0
                
                # Calculate costs
                via_cost = self.config.k_via
                if end_layer < len(self.config.band_penalty):
                    via_cost += self.config.band_penalty[end_layer]
                
                escape_cost = escape_length * self.config.k_fc_len
                if escape_length > 0:
                    escape_cost += escape_length * self.config.k_fc_horiz
                
                total_cost = via_cost + escape_cost
                
                candidate = np.array([
                    rail_x, pad_y, start_layer, end_layer, escape_length, total_cost
                ], dtype=np.float32)
                
                candidates.append(candidate)
        
        # Sort by cost
        if candidates:
            candidates.sort(key=lambda c: float(c[5]))
        
        return candidates[:self.config.max_taps_per_pad]
    
    def _format_results(self, pads: List[Dict], tap_data) -> Dict[str, List[TapCandidate]]:
        """Convert GPU arrays back to structured tap candidates"""
        
        results = {}
        
        for pad_idx, pad in enumerate(pads):
            pad_name = pad.get('name', f"pad_{pad_idx}")
            net_name = pad.get('net', f"net_{pad_idx}")
            
            # Use net name as key for grouping
            if net_name not in results:
                results[net_name] = []
            
            # Extract tap candidates for this pad
            pad_taps = tap_data[pad_idx]
            
            for tap_idx in range(self.config.max_taps_per_pad):
                tap_info = pad_taps[tap_idx]
                
                # Skip empty entries (cost = 0 means no candidate)
                if float(tap_info[5]) == 0.0:
                    break
                
                # Create structured tap candidate
                tap_x, tap_y = float(tap_info[0]), float(tap_info[1])
                start_layer, end_layer = int(tap_info[2]), int(tap_info[3])
                escape_length, cost = float(tap_info[4]), float(tap_info[5])
                
                # Calculate blocked layers for this via
                blocked_layers = set(range(start_layer, end_layer + 1))
                
                candidate = TapCandidate(
                    tap_x=tap_x,
                    tap_y=tap_y,
                    via_layers=(start_layer, end_layer),
                    escape_length=escape_length,
                    total_cost=cost,
                    blocked_layers=blocked_layers
                )
                
                results[net_name].append(candidate)
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU/CPU memory usage for tap generation"""
        
        if not self.pad_positions is None:
            num_pads = len(self.pad_positions)
            max_taps = self.config.max_taps_per_pad
            
            # Estimate memory usage
            pad_data_mb = (num_pads * 2 * 4) / (1024**2)  # positions
            tap_data_mb = (num_pads * max_taps * 6 * 4) / (1024**2)  # tap candidates
            grid_data_mb = (len(self.grid_rails) * 4) / (1024**2) if self.grid_rails is not None else 0
            
            total_mb = pad_data_mb + tap_data_mb + grid_data_mb
            
            return {
                'pad_data_mb': pad_data_mb,
                'tap_data_mb': tap_data_mb, 
                'grid_data_mb': grid_data_mb,
                'total_mb': total_mb,
                'device': 'GPU' if self.use_gpu else 'CPU'
            }
        
        return {'total_mb': 0, 'device': 'GPU' if self.use_gpu else 'CPU'}
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.use_gpu:
            try:
                if self.pad_positions is not None:
                    del self.pad_positions
                if self.grid_rails is not None:
                    del self.grid_rails
                if self.tap_candidates is not None:
                    del self.tap_candidates
                if self.tap_costs is not None:
                    del self.tap_costs
                if self.pad_bounds is not None:
                    del self.pad_bounds
                
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("GPU PadTap memory cleaned up")
            except:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()