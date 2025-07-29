import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

print("Testing OrthoRoute Plugin Data Handling")
print("======================================")

# Import our module
try:
    from orthoroute.gpu_engine import OrthoRouteEngine
    
    # Create test data in 'netlist' format (from orthoroute_plugin.py)
    plugin_data = {
        'netlist': {
            'Net1': [
                {'x': 10000000, 'y': 10000000, 'layer': 0},
                {'x': 30000000, 'y': 30000000, 'layer': 0}
            ],
            'Net2': [
                {'x': 15000000, 'y': 15000000, 'layer': 0},
                {'x': 25000000, 'y': 25000000, 'layer': 0}
            ],
            'SinglePadNet': [
                {'x': 20000000, 'y': 20000000, 'layer': 0}
            ]
        },
        'width': 100000000,
        'height': 80000000
    }
    
    # Initialize engine
    print("Initializing GPU engine...")
    engine = OrthoRouteEngine()
    
    # Test route method with plugin data
    print("Testing route method with plugin_data format...")
    try:
        result = engine.route(plugin_data)
        print("✅ route() method handled 'netlist' format correctly!")
        print(f"Nets processed: {len(result.get('nets', []))}")
        print(f"Success: {result.get('success', False)}")
    except KeyError as e:
        print(f"❌ KeyError: {e}")
    except Exception as e:
        print(f"❌ Other error: {str(e)}")
    
    # Now test with standard format
    standard_data = {
        'nets': [
            {
                'id': 1,
                'name': 'Net1',
                'pins': [
                    {'x': 10000000, 'y': 10000000, 'layer': 0},
                    {'x': 30000000, 'y': 30000000, 'layer': 0}
                ]
            }
        ],
        'bounds': {
            'width_nm': 100000000,
            'height_nm': 80000000,
            'layers': 2
        }
    }
    
    print("\nTesting route method with standard format...")
    try:
        result = engine.route(standard_data)
        print("✅ route() method handled standard format correctly!")
        print(f"Nets processed: {len(result.get('nets', []))}")
        print(f"Success: {result.get('success', False)}")
    except KeyError as e:
        print(f"❌ KeyError: {e}")
    except Exception as e:
        print(f"❌ Other error: {str(e)}")
    
except ImportError as e:
    print(f"Error importing OrthoRouteEngine: {e}")
except Exception as e:
    print(f"General error: {e}")
