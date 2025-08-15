#!/usr/bin/env python3
"""
Code refactoring example: Extracting smaller functions from large routing methods

This demonstrates the "extract smaller, focused methods" improvement 
identified in the code analysis.
"""

from typing import Dict, List, Tuple, Optional
import time
import logging

logger = logging.getLogger(__name__)

class RoutingMethodRefactor:
    """Example of how to refactor large routing methods into smaller, focused functions"""
    
    def __init__(self):
        self.obstacle_grids = {}
        self.routing_stats = {'tracks_added': 0, 'vias_added': 0}
    
    # BEFORE: Large monolithic function (example of current pattern)
    def _route_single_net_monolithic(self, net_name: str, net_data: Dict) -> bool:
        """Example of large monolithic function that mixes concerns"""
        start_time = time.time()
        
        # Grid preparation (concern 1)
        net_obstacle_grids = {}
        for layer in ['F.Cu', 'B.Cu']:
            net_obstacle_grids[layer] = self._create_net_specific_obstacle_grid(layer, net_name)
            self._exclude_net_pads_from_obstacles(net_obstacle_grids[layer], layer, net_name)
        
        # Strategy selection (concern 2)
        pads = net_data.get('pads', [])
        constraints = self._get_net_constraints(net_name)
        timeout = 5.0
        
        # Routing attempts (concern 3)
        if len(pads) == 2:
            # Try single layer first
            best_layer = self._select_best_layer_for_connection_with_grids(
                pads[0], pads[1], net_name, net_obstacle_grids
            )
            if self._route_between_pads_with_timeout_and_grids(
                pads[0], pads[1], best_layer, net_name, constraints, 
                net_obstacle_grids, timeout * 0.4, start_time
            ):
                return True
            
            # Try via routing
            if time.time() - start_time < timeout * 0.8:
                if self._route_two_pads_with_vias_and_grids_timeout(
                    pads[0], pads[1], net_name, constraints, 
                    net_obstacle_grids, timeout * 0.4, start_time
                ):
                    return True
            
            # Try other layer
            other_layer = 'B.Cu' if best_layer == 'F.Cu' else 'F.Cu'
            if time.time() - start_time < timeout * 0.9:
                if self._route_between_pads_with_timeout_and_grids(
                    pads[0], pads[1], other_layer, net_name, constraints,
                    net_obstacle_grids, timeout * 0.2, start_time
                ):
                    return True
        
        # Solution application (concern 4)
        if hasattr(self, 'current_solution'):
            self._apply_routing_solution(self.current_solution, constraints)
            self._update_obstacle_grids_with_solution(self.current_solution)
        
        return False
    
    # AFTER: Extracted smaller, focused methods
    def _route_single_net_refactored(self, net_name: str, net_data: Dict) -> bool:
        """Refactored version with extracted, focused methods"""
        start_time = time.time()
        
        # Each method has a single, clear responsibility
        net_obstacle_grids = self._prepare_obstacle_grids(net_name)
        pads = net_data.get('pads', [])
        constraints = self._get_net_constraints(net_name)
        
        routing_result = self._attempt_routing_strategies(
            pads, net_name, constraints, net_obstacle_grids, start_time
        )
        
        if routing_result.success:
            self._apply_routing_solution(routing_result.solution, constraints)
            return True
        
        return False
    
    def _prepare_obstacle_grids(self, net_name: str) -> Dict:
        """Extract grid preparation logic - single responsibility"""
        logger.debug(f"🗺️ Preparing obstacle grids for {net_name}")
        
        net_obstacle_grids = {}
        for layer in ['F.Cu', 'B.Cu']:
            # Use optimized incremental approach
            net_obstacle_grids[layer] = self._copy_current_obstacle_grid(layer)
            self._exclude_net_pads_from_obstacles(net_obstacle_grids[layer], layer, net_name)
        
        logger.debug(f"⚡ Incremental obstacle grids prepared for {net_name}")
        return net_obstacle_grids
    
    def _attempt_routing_strategies(self, pads: List, net_name: str, 
                                   constraints: Dict, net_obstacle_grids: Dict, 
                                   start_time: float) -> 'RoutingResult':
        """Extract routing strategy logic - single responsibility"""
        timeout = constraints.get('timeout', 5.0)
        
        if len(pads) == 2:
            return self._route_two_pad_net(
                pads[0], pads[1], net_name, constraints, 
                net_obstacle_grids, timeout, start_time
            )
        else:
            return self._route_multi_pad_net(
                pads, net_name, constraints, 
                net_obstacle_grids, timeout, start_time
            )
    
    def _route_two_pad_net(self, pad_a: Dict, pad_b: Dict, net_name: str,
                          constraints: Dict, net_obstacle_grids: Dict,
                          timeout: float, start_time: float) -> 'RoutingResult':
        """Extract two-pad routing logic - single responsibility"""
        
        # Strategy 1: Single-layer on best layer (40% time)
        strategy_1_result = self._try_single_layer_strategy(
            pad_a, pad_b, net_name, constraints, net_obstacle_grids,
            timeout * 0.4, start_time
        )
        if strategy_1_result.success:
            return strategy_1_result
        
        # Strategy 2: Multi-layer with vias (40% time)
        if time.time() - start_time < timeout * 0.8:
            strategy_2_result = self._try_via_strategy(
                pad_a, pad_b, net_name, constraints, net_obstacle_grids,
                timeout * 0.4, start_time
            )
            if strategy_2_result.success:
                return strategy_2_result
        
        # Strategy 3: Fallback single-layer (20% time)
        if time.time() - start_time < timeout * 0.9:
            return self._try_fallback_strategy(
                pad_a, pad_b, net_name, constraints, net_obstacle_grids,
                timeout * 0.2, start_time
            )
        
        return RoutingResult(success=False, reason="All strategies failed")
    
    def _try_single_layer_strategy(self, pad_a: Dict, pad_b: Dict, net_name: str,
                                  constraints: Dict, net_obstacle_grids: Dict,
                                  timeout: float, start_time: float) -> 'RoutingResult':
        """Extract single-layer strategy - single responsibility"""
        logger.debug(f"🎯 Strategy 1: Single-layer routing for {net_name}")
        
        best_layer = self._select_best_layer_for_connection_with_grids(
            pad_a, pad_b, net_name, net_obstacle_grids
        )
        
        success = self._route_between_pads_with_timeout_and_grids(
            pad_a, pad_b, best_layer, net_name, constraints, 
            net_obstacle_grids, timeout, start_time
        )
        
        if success:
            logger.debug(f"✅ Strategy 1 succeeded for {net_name} on {best_layer}")
            return RoutingResult(success=True, strategy="single_layer", layer=best_layer)
        
        logger.debug(f"❌ Strategy 1 failed for {net_name}")
        return RoutingResult(success=False, strategy="single_layer")
    
    def _try_via_strategy(self, pad_a: Dict, pad_b: Dict, net_name: str,
                         constraints: Dict, net_obstacle_grids: Dict,
                         timeout: float, start_time: float) -> 'RoutingResult':
        """Extract via strategy - single responsibility"""
        logger.debug(f"🔗 Strategy 2: Via routing for {net_name}")
        
        success = self._route_two_pads_with_vias_and_grids_timeout(
            pad_a, pad_b, net_name, constraints, 
            net_obstacle_grids, timeout, start_time
        )
        
        if success:
            logger.debug(f"✅ Strategy 2 succeeded for {net_name} with vias")
            return RoutingResult(success=True, strategy="via_routing", has_vias=True)
        
        logger.debug(f"❌ Strategy 2 failed for {net_name}")
        return RoutingResult(success=False, strategy="via_routing")
    
    def _try_fallback_strategy(self, pad_a: Dict, pad_b: Dict, net_name: str,
                              constraints: Dict, net_obstacle_grids: Dict,
                              timeout: float, start_time: float) -> 'RoutingResult':
        """Extract fallback strategy - single responsibility"""
        logger.debug(f"🔄 Strategy 3: Fallback routing for {net_name}")
        
        # Try the other layer as fallback
        best_layer = self._select_best_layer_for_connection_with_grids(
            pad_a, pad_b, net_name, net_obstacle_grids
        )
        other_layer = 'B.Cu' if best_layer == 'F.Cu' else 'F.Cu'
        
        success = self._route_between_pads_with_timeout_and_grids(
            pad_a, pad_b, other_layer, net_name, constraints,
            net_obstacle_grids, timeout, start_time
        )
        
        if success:
            logger.debug(f"✅ Strategy 3 succeeded for {net_name} on {other_layer}")
            return RoutingResult(success=True, strategy="fallback", layer=other_layer)
        
        logger.debug(f"❌ Strategy 3 failed for {net_name}")
        return RoutingResult(success=False, strategy="fallback")
    
    def _apply_routing_solution(self, solution: 'RoutingSolution', constraints: Dict) -> None:
        """Extract solution application - single responsibility"""
        logger.debug(f"📝 Applying routing solution: {len(solution.tracks)} tracks, {len(solution.vias)} vias")
        
        # Add tracks to board
        for track in solution.tracks:
            self._add_track_to_board(track)
            self._add_track_to_obstacle_grids(track)
            self.routing_stats['tracks_added'] += 1
        
        # Add vias to board
        for via in solution.vias:
            self._add_via_to_board(via)
            self._mark_via_as_obstacle(via['x'], via['y'], via['diameter'], constraints['clearance'])
            self.routing_stats['vias_added'] += 1
        
        logger.debug(f"✅ Solution applied: {self.routing_stats}")
    
    # Placeholder methods for demonstration
    def _copy_current_obstacle_grid(self, layer: str): pass
    def _exclude_net_pads_from_obstacles(self, grid, layer: str, net_name: str): pass
    def _get_net_constraints(self, net_name: str) -> Dict: return {'timeout': 5.0}
    def _select_best_layer_for_connection_with_grids(self, pad_a, pad_b, net_name, grids): return 'F.Cu'
    def _route_between_pads_with_timeout_and_grids(self, *args): return False
    def _route_two_pads_with_vias_and_grids_timeout(self, *args): return False
    def _route_multi_pad_net(self, *args): return RoutingResult(success=False)
    def _add_track_to_board(self, track): pass
    def _add_track_to_obstacle_grids(self, track): pass
    def _add_via_to_board(self, via): pass
    def _mark_via_as_obstacle(self, x, y, diameter, clearance): pass
    def _create_net_specific_obstacle_grid(self, layer, net_name): pass

class RoutingResult:
    """Result object for routing operations"""
    def __init__(self, success: bool, strategy: str = None, layer: str = None, 
                 has_vias: bool = False, reason: str = None):
        self.success = success
        self.strategy = strategy
        self.layer = layer
        self.has_vias = has_vias
        self.reason = reason
        self.solution = RoutingSolution() if success else None

class RoutingSolution:
    """Container for routing solution data"""
    def __init__(self):
        self.tracks = []
        self.vias = []

if __name__ == "__main__":
    print("🔧 Code Refactoring Example: Extract Smaller Functions")
    print("=" * 60)
    print()
    print("✅ Benefits of extracted methods:")
    print("  • Single responsibility per function")
    print("  • Easier to test individual components") 
    print("  • Better error handling granularity")
    print("  • Clearer code organization")
    print("  • Reusable strategy components")
    print()
    print("📋 Method extraction completed:")
    print("  • _prepare_obstacle_grids() - Grid preparation")
    print("  • _attempt_routing_strategies() - Strategy coordination")
    print("  • _try_single_layer_strategy() - Single-layer routing")
    print("  • _try_via_strategy() - Via-aware routing")
    print("  • _try_fallback_strategy() - Fallback routing")
    print("  • _apply_routing_solution() - Solution application")
