"""
Basic test of the OrthoRoute GPU autorouter
"""
import json
import numpy as np
import cupy as cp

def create_test_board(size_mm: float = 50, net_count: int = 10) -> dict:
    """Create a test board with random nets"""
    size_nm = int(size_mm * 1000000)  # Convert mm to nm
    nets = []
    
    # Create random 2-pin nets
    np.random.seed(42)  # For reproducible results
    for i in range(net_count):
        # Generate random pin positions
        x1 = np.random.randint(size_nm // 10, 9 * size_nm // 10)
        y1 = np.random.randint(size_nm // 10, 9 * size_nm // 10)
        x2 = np.random.randint(size_nm // 10, 9 * size_nm // 10)
        y2 = np.random.randint(size_nm // 10, 9 * size_nm // 10)
        
        net = {
            "id": i + 1,
            "name": f"NET_{i+1}",
            "pins": [
                {"x": x1, "y": y1, "layer": 0},  # Start pin on top layer
                {"x": x2, "y": y2, "layer": 1}   # End pin on bottom layer
            ],
            "width_nm": 200000  # 0.2mm trace width
        }
        nets.append(net)
    
    # Create board data
    board_data = {
        "bounds": {
            "width_nm": size_nm,
            "height_nm": size_nm,
            "layers": 2
        },
        "grid": {
            "pitch_nm": 100000,  # 0.1mm grid
            "via_size_nm": 200000  # 0.2mm vias
        },
        "nets": nets,
        "design_rules": {
            "min_track_width_nm": 200000,
            "min_clearance_nm": 200000,
            "min_via_size_nm": 400000
        }
    }
    
    return board_data

def run_test():
    """Run a basic autorouting test"""
    try:
        from orthoroute.gpu_engine import OrthoRouteEngine
        
        print("Creating test board...")
        board_data = create_test_board(size_mm=50, net_count=10)
        
        print("\nInitializing GPU engine...")
        engine = OrthoRouteEngine()
        
        print("\nRouting board...")
        results = engine.route_board(board_data)
        
        if results['success']:
            print("\nRouting successful!")
            stats = results['stats']
            print(f"Total nets: {stats['total_nets']}")
            print(f"Successfully routed: {stats['successful_nets']}")
            print(f"Success rate: {stats['success_rate']:.1f}%")
            print(f"Total time: {stats['total_time_seconds']:.2f} seconds")
        else:
            print("\nRouting failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        raise

if __name__ == "__main__":
    run_test()
