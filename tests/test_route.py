import json
from orthoroute.gpu_engine import OrthoRouteEngine

# Load test board data
with open('test_board.json') as f:
    board_data = json.load(f)

# Initialize engine
print("Initializing OrthoRoute engine...")
engine = OrthoRouteEngine()

# Route board
print("\nRouting board...")
results = engine.route_board(board_data)

# Check results
if results['success']:
    print(f"\nSuccess!")
    print(f"Routed {results['stats']['successful_nets']} nets")
    print(f"Success rate: {results['stats']['success_rate']:.1f}%")
    print(f"Total routing time: {results['stats']['total_time_seconds']:.2f} seconds")
else:
    print("\nRouting failed:")
    print(results['error'])
