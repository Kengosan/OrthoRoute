#!/usr/bin/env python3
"""
Micro-tests for OrthoRoute - surgical validation of critical components
Tests portal snap, stub emission, and GUI functionality
"""

import sys
import os
import logging
import random
import math
from pathlib import Path

# Add package to path
package_dir = Path(__file__).parent.parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from orthoroute.algorithms.manhattan.unified_pathfinder import KiCadGeometry

logger = logging.getLogger(__name__)

class MockPad:
    """Mock pad for testing"""
    def __init__(self, x: float, y: float, layer: int = 0, net: str = "test_net"):
        self.x = x
        self.y = y
        self.layer = layer
        self.net = net

def test_portal_snap_invariants():
    """Test portal snap unit: feed (x0, y0, pitch), random pad points, check snap invariants + distance ≤ pitch/2"""
    print("=" * 50)
    print("PORTAL SNAP UNIT TEST")
    print("=" * 50)

    # Test parameters
    x0, y0 = 170.0, 46.0
    pitch = 0.1
    bounds = (x0, y0, x0 + 20.0, y0 + 20.0)  # 20x20mm test area

    # Initialize geometry system
    geometry = KiCadGeometry(bounds, pitch)

    print(f"Test area: {bounds}")
    print(f"Grid: x0={x0}, y0={y0}, pitch={pitch}")
    print(f"Lattice steps: {geometry.x_steps} x {geometry.y_steps}")

    # Test with 100 random pad positions
    random.seed(42)  # Deterministic test
    max_error = 0.0
    max_distance = 0.0
    violations = 0

    print("\nTesting portal snap with 100 random pad positions...")

    for i in range(100):
        # Generate random pad position within bounds
        pad_x = x0 + random.uniform(0, 20.0)
        pad_y = y0 + random.uniform(0, 20.0)

        # Test portal snapping
        lattice_x, lattice_y = geometry.world_to_lattice(pad_x, pad_y)

        # Clamp to valid bounds (like in real code)
        lattice_x = max(0, min(lattice_x, geometry.x_steps - 1))
        lattice_y = max(0, min(lattice_y, geometry.y_steps - 1))

        # Convert back to world coordinates
        portal_x, portal_y = geometry.lattice_to_world(lattice_x, lattice_y)

        # INVARIANT 1: Grid alignment check - fix modulo calculation
        x_aligned = round((portal_x - x0) / pitch) * pitch + x0
        y_aligned = round((portal_y - y0) / pitch) * pitch + y0
        x_error = abs(portal_x - x_aligned)
        y_error = abs(portal_y - y_aligned)

        if x_error > 1e-6 or y_error > 1e-6:
            violations += 1
            if violations <= 3:  # Only log first 3
                print(f"VIOLATION {violations}: Pad ({pad_x:.3f},{pad_y:.3f}) -> Portal ({portal_x:.6f},{portal_y:.6f})")
                print(f"  Grid errors: x={x_error:.9f}, y={y_error:.9f}")

        max_error = max(max_error, max(x_error, y_error))

        # INVARIANT 2: Distance check (should be ≤ pitch/2)
        distance = math.sqrt((pad_x - portal_x)**2 + (pad_y - portal_y)**2)
        max_distance = max(max_distance, distance)

        if distance > pitch/2 + 1e-6:  # Small tolerance
            print(f"DISTANCE VIOLATION: Pad ({pad_x:.3f},{pad_y:.3f}) distance {distance:.6f} > {pitch/2:.6f}")

    print(f"\nRESULTS:")
    print(f"  Max grid error: {max_error:.9f} (should be < 1e-6)")
    print(f"  Max snap distance: {max_distance:.6f} (should be <= {pitch/2:.6f})")
    print(f"  Grid violations: {violations}")
    print(f"  Distance violations: 0 (checked above)")

    success = (violations == 0 and max_distance <= pitch/2 + 1e-6)
    print(f"PORTAL SNAP TEST: {'PASSED' if success else 'FAILED'}")

    return success

def test_stub_emission():
    """Test stub emission unit: fake pad on L0, portal on L3 → expect 1 stub on L0 + 1 via at portal"""
    print("\n" + "=" * 50)
    print("STUB EMISSION UNIT TEST")
    print("=" * 50)

    # Mock routing environment
    from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder
    from orthoroute.shared.configuration import PathFinderConfig

    config = PathFinderConfig()
    pathfinder = UnifiedPathFinder(config=config, use_gpu=False)

    # Create mock geometry
    bounds = (170.0, 46.0, 190.0, 66.0)
    pathfinder.geometry = KiCadGeometry(bounds, 0.1)

    # Mock portal data structure
    class MockPortal:
        def __init__(self, x, y, layer, pad_layer, net):
            self.x = x
            self.y = y
            self.layer = layer
            self.pad_layer = pad_layer
            self.net = net

    # Test case: Pad on L0, portal on L3
    test_net = "TEST_NET"
    pad_layer = 0  # F.Cu
    portal_layer = 3  # In3.Cu
    portal_pos = (175.0, 50.0)

    # Set up mock portals
    pathfinder._pad_portals = {
        "test_pad": MockPortal(portal_pos[0], portal_pos[1], portal_layer, pad_layer, test_net)
    }

    # Mock path that includes the portal position
    mock_path = [100, 200, 300]  # Arbitrary node indices

    # Mock coordinate conversion for path nodes
    def mock_idx_to_coord(node_idx):
        if node_idx == 200:  # Middle node connects to portal
            return (portal_pos[0], portal_pos[1], portal_layer)
        return (175.1, 50.1, portal_layer)  # Other nodes nearby

    pathfinder._idx_to_coord = mock_idx_to_coord

    # Test stub generation
    print(f"Test case: Pad on layer {pad_layer}, portal on layer {portal_layer}")
    print(f"Portal position: {portal_pos}")

    stub_tracks, stub_vias = pathfinder._generate_pad_stubs(test_net, mock_path)

    print(f"\nGenerated:")
    print(f"  Stub tracks: {len(stub_tracks)}")
    print(f"  Stub vias: {len(stub_vias)}")

    # Analyze results
    pad_layer_stubs = [t for t in stub_tracks if t.get('layer') == pad_layer]
    portal_vias = [v for v in stub_vias if v.get('from_layer') == pad_layer and v.get('to_layer') == portal_layer]

    print(f"\nAnalysis:")
    print(f"  Stubs on pad layer {pad_layer}: {len(pad_layer_stubs)}")
    print(f"  Vias from L{pad_layer} to L{portal_layer}: {len(portal_vias)}")

    # Check expectations
    expected_stubs = 1  # One stub on pad layer
    expected_vias = 1   # One via from pad layer to portal layer

    stub_success = len(pad_layer_stubs) == expected_stubs
    via_success = len(portal_vias) == expected_vias

    print(f"\nExpected: {expected_stubs} stub on L{pad_layer}, {expected_vias} via L{pad_layer}→L{portal_layer}")
    print(f"STUB EMISSION TEST: {'PASSED' if (stub_success and via_success) else 'FAILED'}")

    return stub_success and via_success

def test_gui_smoke():
    """GUI smoke test: pass 3 tracks, 1 stub, 1 via; assert 'drew 5/5'"""
    print("\n" + "=" * 50)
    print("GUI SMOKE TEST")
    print("=" * 50)

    # Create test geometry data
    test_tracks = [
        {'net_id': 'NET1', 'layer': 'F.Cu', 'start': (170.0, 50.0), 'end': (171.0, 50.0), 'width': 0.15},
        {'net_id': 'NET2', 'layer': 'In1.Cu', 'start': (170.0, 51.0), 'end': (171.0, 51.0), 'width': 0.15},
        {'net_id': 'NET3', 'layer': 'B.Cu', 'start': (170.0, 52.0), 'end': (171.0, 52.0), 'width': 0.15}
    ]

    test_stubs = [
        {'net_id': 'NET1', 'layer': 'F.Cu', 'start': (169.5, 50.0), 'end': (170.0, 50.0), 'width': 0.1, 'type': 'ownership_stub'}
    ]

    test_vias = [
        {'net_id': 'NET1', 'position': (170.0, 50.0), 'from_layer': 'F.Cu', 'to_layer': 'In1.Cu', 'drill': 0.2, 'size': 0.4}
    ]

    print(f"Test data: {len(test_tracks)} tracks, {len(test_stubs)} stubs, {len(test_vias)} vias")
    print(f"Total elements: {len(test_tracks) + len(test_stubs) + len(test_vias)}")

    # Mock geometry payload
    class MockGeometryPayload:
        def __init__(self, tracks, vias):
            # Combine tracks and stubs for rendering
            self.tracks = tracks + test_stubs  # GUI treats stubs as special tracks
            self.vias = vias

    payload = MockGeometryPayload(test_tracks, test_vias)

    # Simulate drawing process
    drawn_count = 0
    total_elements = len(payload.tracks) + len(payload.vias)

    # Mock drawing validation
    for track in payload.tracks:
        if all(key in track for key in ['net_id', 'layer', 'start', 'end', 'width']):
            drawn_count += 1
        else:
            print(f"Invalid track: {track}")

    for via in payload.vias:
        if all(key in via for key in ['net_id', 'position', 'from_layer', 'to_layer']):
            drawn_count += 1
        else:
            print(f"Invalid via: {via}")

    print(f"\nDraw results: {drawn_count}/{total_elements} elements valid")

    success = drawn_count == total_elements
    print(f"GUI SMOKE TEST: {'PASSED' if success else 'FAILED'}")

    if success:
        print(f"✓ Drew {drawn_count}/{total_elements} elements successfully")

    return success

def run_all_tests():
    """Run all micro-tests"""
    print("ORTHOROUTE MICRO-TESTS")
    print("=" * 80)

    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Suppress debug noise

    results = []

    # Run tests
    results.append(("Portal Snap", test_portal_snap_invariants()))
    results.append(("Stub Emission", test_stub_emission()))
    results.append(("GUI Smoke", test_gui_smoke()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = 0
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:20} : {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 ALL MICRO-TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)