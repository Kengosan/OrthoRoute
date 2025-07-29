"""
OrthoRoute Command Line Interface
GPU-accelerated PCB autorouter using CuPy

Usage:
    orthoroute_cli.py input_board.json [options]
    
Examples:
    # Basic routing
    python orthoroute_cli.py test_board.json
    
    # Custom settings
    python orthoroute_cli.py board.json --pitch 0.05 --layers 8 --iterations 30
    
    # Performance test
    python orthoroute_cli.py --generate-test 1000 --size 100
"""

import argparse
import json
import sys
import time
import random
import os
from typing import Dict, List, Optional

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import cupy as cp
        import numpy as np
        return True, None
    except ImportError as e:
        return False, str(e)

def generate_test_board(net_count: int, board_size_mm: float, output_file: str = None) -> Dict:
    """Generate test board with random nets for benchmarking"""
    print(f"Generating test board: {net_count} nets, {board_size_mm}mm × {board_size_mm}mm")
    
    board_size_nm = int(board_size_mm * 1000000)
    
    nets = []
    for i in range(net_count):
        # Generate random two-pin net with some realistic constraints
        x1 = random.randint(int(board_size_nm * 0.1), int(board_size_nm * 0.9))
        y1 = random.randint(int(board_size_nm * 0.1), int(board_size_nm * 0.9))
        x2 = random.randint(int(board_size_nm * 0.1), int(board_size_nm * 0.9))
        y2 = random.randint(int(board_size_nm * 0.1), int(board_size_nm * 0.9))
        
        # Ensure minimum separation
        while abs(x2 - x1) + abs(y2 - y1) < board_size_nm * 0.1:
            x2 = random.randint(int(board_size_nm * 0.1), int(board_size_nm * 0.9))
            y2 = random.randint(int(board_size_nm * 0.1), int(board_size_nm * 0.9))
        
        # Random layer assignment
        layer1 = random.randint(0, 3)
        layer2 = random.randint(0, 3)
        
        nets.append({
            'id': i + 1,
            'name': f'NET_{i+1:04d}',
            'pins': [
                {'x': x1, 'y': y1, 'layer': layer1},
                {'x': x2, 'y': y2, 'layer': layer2}
            ],
            'priority': random.randint(1, 5),
            'width_nm': random.choice([150000, 200000, 250000]),  # 0.15, 0.2, 0.25mm
            'via_size_nm': 200000
        })
    
    # Add some obstacles for realism
    obstacles = []
    num_obstacles = max(5, net_count // 50)  # Scale obstacles with board complexity
    
    for i in range(num_obstacles):
        # Random rectangular obstacles (components, keepouts)
        obs_size = random.randint(int(board_size_nm * 0.02), int(board_size_nm * 0.08))
        x1 = random.randint(0, board_size_nm - obs_size)
        y1 = random.randint(0, board_size_nm - obs_size)
        
        obstacles.append({
            'type': 'keepout',
            'x1': x1,
            'y1': y1,
            'x2': x1 + obs_size,
            'y2': y1 + obs_size,
            'layer': -1  # All layers
        })
    
    board_data = {
        'version': '0.1.0',
        'bounds': {
            'min_x': 0,
            'min_y': 0,
            'max_x': board_size_nm,
            'max_y': board_size_nm
        },
        'grid': {
            'pitch_mm': 0.1,
            'layers': 4
        },
        'config': {
            'max_iterations': 20,
            'batch_size': 256,
            'verbose': False
        },
        'nets': nets,
        'obstacles': obstacles,
        'design_rules': {
            'min_track_width': 100000,     # 0.1mm
            'min_clearance': 150000,       # 0.15mm
            'min_via_size': 200000,        # 0.2mm
            'min_via_drill': 120000        # 0.12mm
        }
    }
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(board_data, f, indent=2)
        print(f"Test board saved to: {output_file}")
    
    return board_data

def print_gpu_info():
    """Print GPU information if available"""
    try:
        import cupy as cp
        
        device = cp.cuda.Device()
        attrs = device.attributes
        mem_info = device.mem_info
        
        print(f"GPU Information:")
        print(f"  Device: {device.name}")
        print(f"  Compute Capability: {attrs['major']}.{attrs['minor']}")
        print(f"  Multiprocessors: {attrs['multiProcessorCount']}")
        print(f"  CUDA Cores: ~{attrs['multiProcessorCount'] * 128}")  # Rough estimate
        print(f"  Global Memory: {mem_info[1] / (1024**3):.1f} GB")
        print(f"  Available Memory: {mem_info[0] / (1024**3):.1f} GB")
        print(f"  Memory Usage: {(mem_info[1] - mem_info[0]) / (1024**3):.1f} GB")
        
    except Exception as e:
        print(f"Could not get GPU info: {e}")

def print_routing_summary(results: Dict):
    """Print detailed routing results summary"""
    if not results.get('success', False):
        print(f"❌ Routing FAILED: {results.get('error', 'Unknown error')}")
        return
    
    stats = results['stats']
    
    print(f"\n{'='*60}")
    print(f"🎯 ORTHOROUTE ROUTING RESULTS")
    print(f"{'='*60}")
    
    # Success metrics
    print(f"\n📊 SUCCESS METRICS:")
    print(f"  Total nets processed: {stats['total_nets']}")
    print(f"  Successfully routed:  {stats['successful_nets']}")
    print(f"  Failed to route:      {stats['failed_nets']}")
    print(f"  Success rate:         {stats['success_rate']:.1f}%")
    
    # Performance metrics
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"  Routing time:         {stats['routing_time_seconds']:.2f} seconds")
    print(f"  Processing rate:      {stats['nets_per_second']:.1f} nets/second")
    
    # Quality metrics
    print(f"\n📏 QUALITY METRICS:")
    print(f"  Total wire length:    {stats['total_length_mm']:.1f} mm")
    print(f"  Total vias:           {stats['total_vias']}")
    if stats['successful_nets'] > 0:
        print(f"  Avg length per net:   {stats['total_length_mm'] / stats['successful_nets']:.2f} mm")
        print(f"  Avg vias per net:     {stats['total_vias'] / stats['successful_nets']:.1f}")
    
    # Memory usage (if available)
    if 'gpu_memory_mb' in stats:
        print(f"\n💾 MEMORY USAGE:")
        print(f"  GPU memory used:      {stats['gpu_memory_mb']:.1f} MB")
    
    # Performance rating
    nets_per_sec = stats['nets_per_second']
    if nets_per_sec > 100:
        rating = "🔥 EXCELLENT"
    elif nets_per_sec > 50:
        rating = "✅ GOOD"
    elif nets_per_sec > 20:
        rating = "⚠️  ACCEPTABLE"
    else:
        rating = "🐌 SLOW"
    
    print(f"\n🏆 PERFORMANCE RATING: {rating}")
    
    print(f"{'='*60}")

def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    try:
        from orthoroute.gpu_engine import OrthoRouteEngine
    except ImportError:
        print("❌ Error: OrthoRoute package not found. Please install OrthoRoute first.")
        return False
    
    test_cases = [
        {'nets': 100, 'size_mm': 25, 'name': 'Small Board'},
        {'nets': 500, 'size_mm': 50, 'name': 'Medium Board'},
        {'nets': 2000, 'size_mm': 100, 'name': 'Large Board'},
        {'nets': 8000, 'size_mm': 200, 'name': 'Extreme Board'}
    ]
    
    print(f"\n🚀 ORTHOROUTE PERFORMANCE BENCHMARK")
    print(f"{'='*60}")
    
    print_gpu_info()
    
    engine = OrthoRouteEngine()
    
    benchmark_results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test {i+1}/4: {test_case['name']}")
        print(f"   Nets: {test_case['nets']}")
        print(f"   Size: {test_case['size_mm']}mm × {test_case['size_mm']}mm")
        
        # Generate test board
        board_data = generate_test_board(test_case['nets'], test_case['size_mm'])
        
        # Route board
        print(f"   🔄 Routing...")
        start_time = time.time()
        
        try:
            results = engine.route_board(board_data)
            end_time = time.time()
            
            if results['success']:
                stats = results['stats']
                test_result = {
                    'name': test_case['name'],
                    'nets': test_case['nets'],
                    'size_mm': test_case['size_mm'],
                    'success_rate': stats['success_rate'],
                    'routing_time': stats['routing_time_seconds'],
                    'nets_per_second': stats['nets_per_second'],
                    'total_time': end_time - start_time
                }
                benchmark_results.append(test_result)
                
                print(f"   ✅ Success: {stats['success_rate']:.1f}% in {stats['routing_time_seconds']:.2f}s")
                print(f"   ⚡ Rate: {stats['nets_per_second']:.1f} nets/second")
            else:
                print(f"   ❌ Failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
            # Continue with next test
    
    # Print benchmark summary
    if benchmark_results:
        print(f"\n📈 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"{'Test':<15} {'Nets':<6} {'Size':<8} {'Success':<8} {'Time':<8} {'Rate':<10}")
        print(f"{'-'*60}")
        
        for result in benchmark_results:
            print(f"{result['name']:<15} {result['nets']:<6} {result['size_mm']:<4}mm "
                  f"{result['success_rate']:<6.1f}% {result['routing_time']:<6.1f}s "
                  f"{result['nets_per_second']:<8.1f}")
    
    return True

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="OrthoRoute: GPU-accelerated PCB autorouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Route a board from JSON file
  python orthoroute_cli.py board.json
  
  # Custom grid settings
  python orthoroute_cli.py board.json --pitch 0.05 --layers 8
  
  # Generate and route test board
  python orthoroute_cli.py --generate-test 1000 --size 100
  
  # Run performance benchmark
  python orthoroute_cli.py --benchmark
  
  # Check GPU status
  python orthoroute_cli.py --gpu-info
        """
    )
    
    # Input options
    parser.add_argument('input_file', nargs='?', help='Input board data JSON file')
    parser.add_argument('-o', '--output', default='routing_results.json', 
                       help='Output results file (default: routing_results.json)')
    
    # GPU options
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--gpu-info', action='store_true', help='Show GPU information and exit')
    
    # Grid options
    parser.add_argument('--pitch', type=float, help='Grid pitch in mm (default: from input file or 0.1)')
    parser.add_argument('--layers', type=int, help='Number of routing layers (default: from input file or 4)')
    
    # Routing options
    parser.add_argument('--iterations', type=int, help='Max routing iterations (default: 20)')
    parser.add_argument('--batch-size', type=int, help='Nets per batch (default: 256)')
    
    # Test generation
    parser.add_argument('--generate-test', type=int, metavar='NETS', 
                       help='Generate test board with specified number of nets')
    parser.add_argument('--size', type=float, default=50.0, 
                       help='Test board size in mm (default: 50.0)')
    
    # Utilities
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmark')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate input file format')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--version', action='version', version='OrthoRoute 0.1.0')
    
    args = parser.parse_args()
    
    # Check dependencies first
    deps_ok, deps_error = check_dependencies()
    if not deps_ok:
        print(f"❌ Missing dependencies: {deps_error}")
        print("\nInstallation required:")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")
        print("  pip install numpy")
        return 1
    
    # Handle special commands
    if args.gpu_info:
        print_gpu_info()
        return 0
    
    if args.benchmark:
        success = run_performance_benchmark()
        return 0 if success else 1
    
    if args.generate_test:
        output_file = args.output if args.output != 'routing_results.json' else f'test_board_{args.generate_test}nets.json'
        board_data = generate_test_board(args.generate_test, args.size, output_file)
        
        if not args.input_file:
            # Just generate, don't route
            print(f"✅ Test board generated: {output_file}")
            return 0
        else:
            # Generate and route
            print(f"🔄 Routing generated test board...")
    
    # Validate input file
    if not args.input_file and not args.generate_test:
        print("❌ Error: Input file required (or use --generate-test)")
        parser.print_help()
        return 1
    
    try:
        # Load input data
        if args.generate_test and not args.input_file:
            # Already handled above
            return 0
        elif args.generate_test:
            # Use generated data
            board_data = generate_test_board(args.generate_test, args.size)
        else:
            # Load from file
            if not os.path.exists(args.input_file):
                print(f"❌ Error: Input file '{args.input_file}' not found")
                return 1
            
            try:
                with open(args.input_file, 'r') as f:
                    board_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ Error: Invalid JSON in '{args.input_file}': {e}")
                return 1
        
        # Validate input format
        if args.validate:
            required_keys = ['bounds', 'nets']
            missing_keys = [key for key in required_keys if key not in board_data]
            if missing_keys:
                print(f"❌ Invalid board data: missing keys {missing_keys}")
                return 1
            print("✅ Board data format is valid")
            if args.validate and not args.generate_test:
                return 0
        
        # Override configuration from command line
        if 'grid' not in board_data:
            board_data['grid'] = {}
        if 'config' not in board_data:
            board_data['config'] = {}
        
        if args.pitch:
            board_data['grid']['pitch_mm'] = args.pitch
        if args.layers:
            board_data['grid']['layers'] = args.layers
        if args.iterations:
            board_data['config']['max_iterations'] = args.iterations
        if args.batch_size:
            board_data['config']['batch_size'] = args.batch_size
        if args.verbose:
            board_data['config']['verbose'] = True
        
        # Import and initialize engine
        try:
            from orthoroute.gpu_engine import OrthoRouteEngine
        except ImportError:
            print("❌ Error: OrthoRoute package not found")
            print("Please install OrthoRoute:")
            print("  pip install orthoroute")
            print("  # OR install from source:")
            print("  git clone https://github.com/username/OrthoRoute.git")
            print("  cd OrthoRoute && pip install -e .")
            return 1
        
        if args.verbose:
            print(f"🔧 Initializing GPU engine (device {args.gpu_id})...")
            print_gpu_info()
        
        engine = OrthoRouteEngine(gpu_id=args.gpu_id)
        
        # Route board
        if args.verbose:
            print(f"🚀 Starting routing of {len(board_data['nets'])} nets...")
        
        start_time = time.time()
        results = engine.route_board(board_data)
        total_time = time.time() - start_time
        
        # Add total execution time
        if results.get('success', False):
            results['stats']['total_execution_time'] = total_time
        
        # Save results
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            if args.verbose:
                print(f"💾 Results saved to: {args.output}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save results to {args.output}: {e}")
        
        # Print summary
        print_routing_summary(results)
        
        # Return appropriate exit code
        if results.get('success', False):
            success_rate = results['stats']['success_rate']
            if success_rate >= 95:
                return 0  # Excellent
            elif success_rate >= 80:
                return 0  # Good enough
            else:
                return 2  # Poor results
        else:
            return 1  # Failed
            
    except KeyboardInterrupt:
        print("\n⏹️  Routing interrupted by user")
        return 130
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)