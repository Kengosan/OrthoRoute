"""
Ripup and repair logic for Manhattan routing
"""

import logging
import heapq
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from .types import Pad, Track, Via, RoutingConfig
from .grid_manager import GridManager

logger = logging.getLogger(__name__)

class RipupRepairManager:
    """Manages ripup and repair operations with intelligent prioritization"""
    
    def __init__(self, config: RoutingConfig, grid_manager: GridManager):
        self.config = config
        self.grid = grid_manager
        
        # Ripup tracking
        self.net_ripup_counts = defaultdict(int)
        self.net_path_lengths = {}  # net_id -> path length
        self.failed_attempts = defaultdict(int)
    
    def find_conflicting_nets(self, target_bounds: Tuple[float, float, float, float], target_net_id: int) -> Set[int]:
        """
        Find nets that are blocking the target routing area
        
        Per spec: "If you can't route a net, I'd like you to find the trace that's blocking it"
        
        Args:
            target_bounds: (min_x, min_y, max_x, max_y) in mm
            target_net_id: Net ID that needs routing space
            
        Returns:
            Set of conflicting net IDs
        """
        min_x, min_y, max_x, max_y = target_bounds
        
        # Convert bounds to grid coordinates
        start_gx, start_gy = self.grid.world_to_grid(min_x, min_y)
        end_gx, end_gy = self.grid.world_to_grid(max_x, max_y)
        
        # Add search margin
        margin = 2
        start_gx = max(0, start_gx - margin)
        start_gy = max(0, start_gy - margin)
        end_gx = min(self.grid.grid_width - 1, end_gx + margin)
        end_gy = min(self.grid.grid_height - 1, end_gy + margin)
        
        conflicting_nets = set()
        
        # Search all layers in the target region
        for layer in range(self.grid.num_layers):
            for gy in range(start_gy, end_gy + 1):
                for gx in range(start_gx, end_gx + 1):
                    cell_net = self.grid.net_id_grid[layer, gy, gx]
                    if cell_net > 0 and cell_net != target_net_id:
                        conflicting_nets.add(cell_net)
        
        logger.debug(f"Found {len(conflicting_nets)} conflicting nets in target area")
        return conflicting_nets
    
    def prioritize_ripup_candidates(self, conflicting_nets: Set[int]) -> List[int]:
        """
        Prioritize nets for ripup based on intelligent criteria
        
        Per spec: "For rip-up, prioritize removing nets with longer paths first, 
        and avoid ripping up nets that have already been ripped up multiple times"
        
        Args:
            conflicting_nets: Set of net IDs that could be ripped up
            
        Returns:
            List of net IDs in ripup priority order (highest priority first)
        """
        candidates = []
        
        for net_id in conflicting_nets:
            # Calculate priority score (higher = more likely to ripup)
            path_length = self.get_net_path_length(net_id)
            ripup_count = self.net_ripup_counts[net_id]
            failed_count = self.failed_attempts[net_id]
            
            # Priority factors:
            # 1. Longer paths get higher ripup priority
            # 2. Nets with fewer previous ripups get higher priority  
            # 3. Nets with recent failures get lower priority
            
            length_score = path_length / 10.0  # Normalize to reasonable range
            ripup_penalty = ripup_count * 2.0  # Heavy penalty for repeated ripups
            failure_penalty = failed_count * 1.5  # Moderate penalty for failures
            
            priority_score = length_score - ripup_penalty - failure_penalty
            
            candidates.append((priority_score, net_id, path_length, ripup_count))
        
        # Sort by priority score (descending)
        candidates.sort(reverse=True)
        
        # Log ripup prioritization
        if candidates:
            logger.debug("Ripup priority order:")
            for i, (score, net_id, length, ripups) in enumerate(candidates[:5]):  # Show top 5
                logger.debug(f"  {i+1}. Net {net_id}: score={score:.2f}, length={length:.2f}, ripups={ripups}")
        
        return [net_id for _, net_id, _, _ in candidates]
    
    def ripup_net(self, net_id: int) -> Dict:
        """
        Ripup (remove) all traces of a net from the grid
        
        Args:
            net_id: Net ID to ripup
            
        Returns:
            Dict with ripup statistics
        """
        logger.info(f"Ripping up net {net_id}")
        
        # Track what we're removing
        cleared_cells = 0
        removed_tracks = []
        removed_vias = []
        
        # Clear from grid
        cleared_cells = self.grid.clear_net_from_grid(net_id)
        
        # Update ripup statistics
        self.net_ripup_counts[net_id] += 1
        
        ripup_stats = {
            'net_id': net_id,
            'cleared_cells': cleared_cells,
            'ripup_count': self.net_ripup_counts[net_id],
            'removed_tracks': len(removed_tracks),
            'removed_vias': len(removed_vias)
        }
        
        logger.debug(f"Ripped up net {net_id}: {cleared_cells} cells cleared, "
                    f"ripup count now {self.net_ripup_counts[net_id]}")
        
        return ripup_stats
    
    def should_attempt_ripup(self, net_id: int, conflicting_nets: Set[int]) -> bool:
        """
        Decide whether to attempt ripup for a failed net routing
        
        Args:
            net_id: Net that failed to route
            conflicting_nets: Nets that are blocking it
            
        Returns:
            True if ripup should be attempted
        """
        # Don't ripup if we've exceeded max ripup attempts for this net
        if self.net_ripup_counts[net_id] >= self.config.max_ripups_per_net:
            logger.debug(f"Net {net_id} exceeded max ripup attempts ({self.config.max_ripups_per_net})")
            return False
        
        # Don't ripup if no conflicting nets found
        if not conflicting_nets:
            logger.debug(f"Net {net_id} has no conflicting nets to ripup")
            return False
        
        # Check if any conflicting nets are good ripup candidates
        candidates = self.prioritize_ripup_candidates(conflicting_nets)
        good_candidates = [net for net in candidates 
                          if self.net_ripup_counts[net] < self.config.max_ripups_per_net]
        
        if not good_candidates:
            logger.debug(f"Net {net_id} has no good ripup candidates")
            return False
        
        return True
    
    def execute_ripup_and_repair(self, failed_net_id: int, failed_net_data: Dict) -> bool:
        """
        Execute ripup and repair sequence
        
        Args:
            failed_net_id: Net that failed to route
            failed_net_data: Data for the failed net (bounds, etc.)
            
        Returns:
            True if ripup created space for routing
        """
        bounds = failed_net_data.get('bounds', (0, 0, 0, 0))
        
        # Find conflicting nets
        conflicting_nets = self.find_conflicting_nets(bounds, failed_net_id)
        
        # Check if ripup is worthwhile
        if not self.should_attempt_ripup(failed_net_id, conflicting_nets):
            return False
        
        # Get prioritized ripup candidates
        ripup_candidates = self.prioritize_ripup_candidates(conflicting_nets)
        
        # Ripup nets in priority order (start with best candidates)
        nets_ripped_up = 0
        max_ripups_per_attempt = min(3, len(ripup_candidates))  # Limit ripups per attempt
        
        for candidate_net in ripup_candidates[:max_ripups_per_attempt]:
            # Skip nets that have been ripped up too many times
            if self.net_ripup_counts[candidate_net] >= self.config.max_ripups_per_net:
                continue
            
            ripup_stats = self.ripup_net(candidate_net)
            nets_ripped_up += 1
            
            logger.info(f"Ripup {nets_ripped_up}: cleared {ripup_stats['cleared_cells']} cells from net {candidate_net}")
        
        if nets_ripped_up > 0:
            logger.info(f"Ripped up {nets_ripped_up} nets to make space for net {failed_net_id}")
            return True
        
        return False
    
    def get_net_path_length(self, net_id: int) -> float:
        """
        Calculate the total path length of a routed net
        
        Args:
            net_id: Net ID
            
        Returns:
            Total path length in mm
        """
        if net_id in self.net_path_lengths:
            return self.net_path_lengths[net_id]
        
        # Calculate path length by counting grid cells used by this net
        total_cells = 0
        
        for layer in range(self.grid.num_layers):
            layer_cells = (self.grid.net_id_grid[layer] == net_id).sum()
            total_cells += layer_cells
        
        # Convert to mm (approximate)
        path_length = total_cells * self.config.grid_pitch
        self.net_path_lengths[net_id] = path_length
        
        return path_length
    
    def record_routing_failure(self, net_id: int):
        """Record that a net failed to route"""
        self.failed_attempts[net_id] += 1
        logger.debug(f"Net {net_id} failed to route (failure count: {self.failed_attempts[net_id]})")
    
    def record_routing_success(self, net_id: int, path_length: float):
        """Record successful routing of a net"""
        self.net_path_lengths[net_id] = path_length
        # Reset failure count on success
        if net_id in self.failed_attempts:
            del self.failed_attempts[net_id]
    
    def get_ripup_statistics(self) -> Dict:
        """Get overall ripup statistics"""
        total_ripups = sum(self.net_ripup_counts.values())
        nets_with_ripups = len(self.net_ripup_counts)
        
        return {
            'total_ripups': total_ripups,
            'nets_with_ripups': nets_with_ripups,
            'average_ripups_per_net': total_ripups / max(1, nets_with_ripups),
            'max_ripups_single_net': max(self.net_ripup_counts.values()) if self.net_ripup_counts else 0,
            'nets_with_failures': len(self.failed_attempts),
            'total_failures': sum(self.failed_attempts.values())
        }