#!/usr/bin/env python3
"""
Simple router architecture test - direct coordinate mapping
"""

def simple_router_architecture():
    """Test a much simpler routing architecture"""
    
    # Step 1: Collect ALL pins with simple data structure
    print("=== SIMPLE ROUTER ARCHITECTURE TEST ===")
    
    # Simulate pin data (replace with actual KiCad extraction)
    all_pins = [
        {'x': 69850000, 'y': 27980000, 'net': 'GND', 'layer': 0},
        {'x': 52070000, 'y': 68580000, 'net': 'GND', 'layer': 0},
        {'x': 69850000, 'y': 30480000, 'net': 'Net-(C1-Pad1)', 'layer': 0},
        {'x': 55880000, 'y': 33020000, 'net': 'Net-(C1-Pad1)', 'layer': 0},
        {'x': 49530000, 'y': 71120000, 'net': '+5V', 'layer': 0},
        {'x': 63500000, 'y': 53340000, 'net': '+5V', 'layer': 0},
        {'x': 38100000, 'y': 48260000, 'net': 'GND', 'layer': 0},
        {'x': 78740000, 'y': 33020000, 'net': 'GND', 'layer': 0},
    ]
    
    print(f"Collected {len(all_pins)} pins")
    
    # Step 2: Direct coordinate range calculation
    min_x = min(pin['x'] for pin in all_pins)
    max_x = max(pin['x'] for pin in all_pins)
    min_y = min(pin['y'] for pin in all_pins)
    max_y = max(pin['y'] for pin in all_pins)
    
    coord_width = max_x - min_x
    coord_height = max_y - min_y
    
    print(f"Coordinate range: X({min_x/1e6:.1f} to {max_x/1e6:.1f}mm) = {coord_width/1e6:.1f}mm")
    print(f"Coordinate range: Y({min_y/1e6:.1f} to {max_y/1e6:.1f}mm) = {coord_height/1e6:.1f}mm")
    
    # Step 3: Direct grid sizing (no complex margins)
    GRID_PITCH_NM = 100000  # 0.1mm
    MARGIN_CELLS = 20  # Simple 20-cell margin
    
    grid_width = int(coord_width / GRID_PITCH_NM) + MARGIN_CELLS
    grid_height = int(coord_height / GRID_PITCH_NM) + MARGIN_CELLS
    
    print(f"Grid size: {grid_width}x{grid_height} cells")
    print(f"Grid covers: {grid_width * GRID_PITCH_NM/1e6:.1f}mm x {grid_height * GRID_PITCH_NM/1e6:.1f}mm")
    
    # Step 4: Simple coordinate conversion
    def world_to_grid(x, y):
        grid_x = int((x - min_x) / GRID_PITCH_NM) + MARGIN_CELLS // 2
        grid_y = int((y - min_y) / GRID_PITCH_NM) + MARGIN_CELLS // 2
        return grid_x, grid_y
    
    # Step 5: Test all pins
    print("\n=== PIN CONVERSION TEST ===")
    all_in_bounds = True
    for i, pin in enumerate(all_pins):
        gx, gy = world_to_grid(pin['x'], pin['y'])
        in_bounds = (0 <= gx < grid_width and 0 <= gy < grid_height)
        status = "✅" if in_bounds else "❌"
        print(f"Pin {i+1}: ({pin['x']/1e6:.2f}, {pin['y']/1e6:.2f})mm → grid({gx}, {gy}) {status}")
        if not in_bounds:
            all_in_bounds = False
    
    print(f"\n=== RESULT ===")
    print(f"All pins in bounds: {'✅ YES' if all_in_bounds else '❌ NO'}")
    
    if all_in_bounds:
        print("✅ This architecture should work!")
        print("Next steps:")
        print("1. Replace complex coordinate system with this simple approach")
        print("2. Embed visualization updates directly in routing loop")
        print("3. Remove unnecessary coordinate transformation layers")
    else:
        print("❌ Even simple approach has issues - need to debug coordinate extraction")
    
    return all_in_bounds

if __name__ == "__main__":
    simple_router_architecture()
