#!/usr/bin/env python3
"""
Quick fix for the simplified through-hole routing function
"""

def _route_two_pads_multilayer_with_timeout_and_grids(self, source_pad, target_pad, net_name, net_constraints, net_obstacle_grids, timeout, start_time):
    """Route between two pads using SIMPLE through-hole aware strategy"""
    import time
    
    # Check timeout
    if time.time() - start_time > timeout:
        raise TimeoutError(f"Routing timeout for {net_name}")
    
    src_x, src_y = source_pad.get('x', 0), source_pad.get('y', 0)
    tgt_x, tgt_y = target_pad.get('x', 0), target_pad.get('y', 0)
    connection_distance = ((tgt_x - src_x)**2 + (tgt_y - src_y)**2)**0.5
    
    # For 2-layer boards with through-hole pads: SIMPLE STRATEGY
    # 1. Try F.Cu first (component side)
    # 2. If blocked, try B.Cu (solder side) 
    # 3. Through-hole pads are automatically connected to both layers
    # 4. No complex via logic needed!
    
    print(f"🔗 {net_name}: distance={connection_distance:.1f}mm - trying simple layer switching")
    
    # STRATEGY 1: Try F.Cu first (most components are on top)
    if time.time() - start_time < timeout * 0.6:
        if self._route_between_pads_with_timeout_and_grids(source_pad, target_pad, 'F.Cu', net_name, net_constraints, net_obstacle_grids, timeout * 0.5, start_time):
            print(f"✅ Successfully routed {net_name} on F.Cu")
            return True
        else:
            print(f"❌ F.Cu blocked for {net_name}")
    
    # STRATEGY 2: Try B.Cu (more space usually available)
    if time.time() - start_time < timeout * 0.9:
        if self._route_between_pads_with_timeout_and_grids(source_pad, target_pad, 'B.Cu', net_name, net_constraints, net_obstacle_grids, timeout * 0.4, start_time):
            print(f"✅ Successfully routed {net_name} on B.Cu")
            return True
        else:
            print(f"❌ B.Cu also blocked for {net_name}")
    
    # STRATEGY 3: Use vias ONLY as last resort for very complex routing
    if time.time() - start_time < timeout * 0.95:
        print(f"❌ Both layers failed, trying vias as last resort for {net_name}")
        if hasattr(self, '_route_two_pads_with_vias_and_grids_timeout'):
            if self._route_two_pads_with_vias_and_grids_timeout(source_pad, target_pad, net_name, net_constraints, net_obstacle_grids, timeout * 0.05, start_time):
                print(f"✅ Successfully routed {net_name} using vias (last resort)")
                return True
    
    print(f"❌ All routing strategies failed for {net_name}")
    return False

print("✅ Simplified through-hole routing strategy ready!")
print("📋 Function implements:")
print("   1. Try F.Cu first")  
print("   2. Try B.Cu if F.Cu blocked")
print("   3. Use vias only as last resort")
print("   4. Leverages through-hole connectivity")
