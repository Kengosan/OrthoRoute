"""Simple test to check grid router functionality"""
import sys
import os

# Add the plugin directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'addon_package', 'plugins'))

def main():
    print("🧪 Grid Router Quick Test")
    
    try:
        from grid_router import GridPoint, create_grid_router
        print("✅ Import successful")
        
        # Test GridPoint
        point = GridPoint(x=100, y=200, layer=1)
        print(f"✅ GridPoint created: ({point.x}, {point.y}, {point.layer})")
        
        # Test hashability
        point_set = {point}
        print(f"✅ GridPoint is hashable")
        
        # Test router creation
        board_data = {
            'board_width': 10160000,   # 10.16mm
            'board_height': 10160000,
            'layer_count': 4,
            'obstacles': {}
        }
        
        config = {'grid_spacing': 2540000}  # 2.54mm
        print("Creating router...")
        
        router = create_grid_router(board_data, config)
        if router:
            print("✅ Router created")
            stats = router.get_routing_statistics()
            print(f"Grid points: {stats['grid_statistics']['total_grid_points']}")
            router.cleanup()
        else:
            print("❌ Router creation failed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
