<img src="/Assets/icon200.png" width="200" align="left"/>

# OrthoRoute: GPU-Accelerated PCB Autorouter <nbsp><nbsp>


**OrthoRoute** is a GPU-accelerated PCB autorouter, designed for massively parallel routing of complex circuit boards. Unlike traditional CPU-based autorouters that process nets sequentially, OrthoRoute leverages thousands of CUDA cores to route nets simultaneously using modern GPU compute.

**Key Innovation:** Pure Python + CuPy implementation for maximum portability - no compilation required, just install CuPy and run!

## Technical Philosophy

### Core Principles
1. **Portability First** - Built on CuPy for zero compilation
2. **Scalability** - Designed to handle 8K+ nets on modern GPUs
3. **Memory Efficiency** - Tiled processing and compressed state management
4. **Conflict Resolution** - Negotiated congestion as primary algorithm, not afterthought
5. **Real-world Ready** - Multi-pin nets, design rules, production quality output

### Why CuPy Over Raw CUDA?
- **Zero compilation** - `pip install cupy-cuda12x` and you're ready
- **Cross-platform** - Works on Windows, Linux, macOS (with CUDA)
- **Rapid development** - NumPy-like syntax for GPU operations
- **Memory management** - Automatic GPU memory handling
- **JIT kernels** - Custom CUDA kernels when needed via RawKernel
- **Deployment simplicity** - Single Python script, no build system

## System Architecture

### High-Level Design

```
┌─────────────────┐    JSON    ┌──────────────────┐    Tracks    ┌─────────────┐
│   KiCad Board   │ ---------> │  OrthoRoute GPU  │ -----------> │ Routed PCB  │
│                 │  (export)  │     Engine       │  (import)    │             │
└─────────────────┘            └──────────────────┘              └─────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │ GPU Computation │
                               │ (CuPy + CUDA)   │
                               │ • Parallel A*   │
                               │ • Tiled routing │
                               │ • Conflict res. │
                               └─────────────────┘
```

### Installation

1. **Install OrthoRoute Package:**
```bash
# Clone repository
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute

# Install package
pip install .
```

2. **Install GPU Dependencies:**
```bash
# For CUDA 12.x:
pip install cupy-cuda12x

# For CUDA 11.x:
pip install cupy-cuda11x
```

3. **Install KiCad Plugin:**

Windows:
```powershell
$PLUGIN_DIR="$env:APPDATA\kicad\7.0\3rdparty\plugins\OrthoRoute"
mkdir -p $PLUGIN_DIR
cp -r kicad_plugin\* $PLUGIN_DIR
```

Linux:
```bash
PLUGIN_DIR="~/.local/share/kicad/7.0/3rdparty/plugins/OrthoRoute"
mkdir -p $PLUGIN_DIR
cp -r kicad_plugin/* $PLUGIN_DIR
```

macOS:
```bash
PLUGIN_DIR="~/Library/Application Support/kicad/7.0/3rdparty/plugins/OrthoRoute"
mkdir -p "$PLUGIN_DIR"
cp -r kicad_plugin/* "$PLUGIN_DIR"
```

4. **Restart KiCad**

### Component Architecture

1. **KiCad Integration** (`kicad_plugin/`)
   - Board data extraction
   - Net filtering and prioritization
   - Plugin UI and configuration
   - Route import and validation
   - User interface

2. **GPU Routing Engine** (`gpu_engine.py`)
   - CuPy-based OrthoRouteEngine class
   - GPU device management
   - Memory optimization
   - Parallel routing algorithms

3. **Grid Management** (`grid_manager.py`)
   - GPUGrid class for grid operations
   - TileManager for memory efficiency
   - Obstacle and via handling
   - Unit conversion and grid mapping

4. **Steiner Tree Builder** (`orthoroute_steiner.py`)
   - Multi-pin net handling
   - MST construction on Hanan grid
   - Via minimization

## Core Algorithms

### 1. Parallel Wavefront Routing (Lee's Algorithm)

**Traditional Approach:**
- Sequential processing, one net at a time
- Global priority queue (poor ((none)) GPU parallelism)
- Full grid state per net (memory explosion)

**OrthoRoute Approach:**
```python
# Parallel wavefront expansion using CuPy
def expand_wavefront_parallel(grid, current_wave, distance):
    # All nets expand simultaneously
    next_wave = cp.zeros_like(current_wave)
    
    # Vectorized neighbor exploration
    neighbors = get_all_neighbors(current_wave)  # GPU kernel
    valid_neighbors = filter_available(grid, neighbors)  # GPU kernel
    
    # Update distance map atomically
    update_distances(grid.distance_map, valid_neighbors, distance + 1)
    
    return next_wave
```

**Key Features:**
- **Batch processing** - Route 256-1024 nets simultaneously
- **Wavefront parallelism** - All threads process same distance level
- **Memory efficiency** - Shared distance maps, compressed state
- **Conflict detection** - Real-time congestion tracking

### 2. Tiled Memory Management

**Problem:** 8K nets × full board grid = impossible memory requirements

**Solution:** Process board in 64×64 tiles that fit in GPU shared memory

```python
class TileManager:
    def __init__(self, grid: GPUGrid, tile_size: int = 64):
        self.grid = grid
        self.tile_size = tile_size
        self.tiles_x = (grid.width + tile_size - 1) // tile_size
        self.tiles_y = (grid.height + tile_size - 1) // tile_size
        
        print(f"Tile configuration: {self.tiles_x}×{self.tiles_y} tiles "
              f"of {tile_size}×{tile_size}")
    
    def extract_tile(self, tile_x: int, tile_y: int) -> Dict[str, cp.ndarray]:
        """Extract tile data to GPU shared memory equivalent"""
        x_start, y_start, x_end, y_end = self.get_tile_bounds(tile_x, tile_y)
        
        tile_data = {
            'availability': self.grid.availability[:, y_start:y_end, x_start:x_end].copy(),
            'congestion': self.grid.congestion_cost[:, y_start:y_end, x_start:x_end].copy(),
            'distance': self.grid.distance_map[:, y_start:y_end, x_start:x_end].copy(),
            'bounds': (x_start, y_start, x_end, y_end)
        }
        return tile_data
```

### 3. Negotiated Congestion Routing

**Core Principle:** Cost of contested cells increases across iterations

```python
def update_congestion_costs(grid, routed_nets, iteration):
    # Reset usage counts
    grid.usage_count[:] = 0
    
    # Count usage per cell from all routes
    for net in routed_nets:
        for point in net.route_path:
            grid.usage_count[point.layer, point.y, point.x] += 1
    
    # Apply congestion penalty
    congestion_factor = 1.5 ** iteration
    overcrowded = grid.usage_count > grid.capacity
    
    grid.congestion_cost[overcrowded] *= congestion_factor
```

**Iteration Flow:**
1. Route all nets with current costs
2. Detect conflicts (cells with usage > capacity)
3. Increase cost of contested cells
4. Rip-up conflicted nets
5. Re-route with updated costs
6. Repeat until convergence

### 4. Multi-Pin Net Handling (Steiner Trees)

**Challenge:** Real nets aren't just point-to-point

**Solution:** Build minimum spanning tree on Hanan grid

```python
def build_steiner_tree(pins):
    if len(pins) == 2:
        return route_two_pins(pins[0], pins[1])
    
    # Generate Hanan grid points
    x_coords = {pin.x for pin in pins}
    y_coords = {pin.y for pin in pins}
    hanan_points = [(x, y) for x in x_coords for y in y_coords]
    
    # Build MST connecting all pins
    mst = minimum_spanning_tree(hanan_points, pins)
    
    # Route each MST edge
    tree_routes = []
    for edge in mst:
        route = route_segment(edge.start, edge.end)
        tree_routes.extend(route)
    
    return optimize_via_count(tree_routes)
```

## Memory Architecture

### GPU Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Global Memory                        │
├─────────────────────────────────────────────────────────────┤
│ Grid Arrays (Primary):                                      │
│ • availability[layers, height, width]      - uint8          │
│ • congestion_cost[layers, height, width]   - float32        │
│ • distance_map[layers, height, width]      - uint16         │
│ • usage_count[layers, height, width]       - uint8          │
├─────────────────────────────────────────────────────────────┤
│ Net State (Compressed):                                     │
│ • net_positions[batch_size, max_pins, 3]   - int32          │
│ • net_routes[batch_size, max_route_len, 3] - int32          │
│ • route_lengths[batch_size]                - int32          │
├─────────────────────────────────────────────────────────────┤
│ Working Memory:                                             │
│ • wavefront_current[batch_size, max_wave]  - int32          │
│ • wavefront_next[batch_size, max_wave]     - int32          │
│ • conflict_flags[batch_size]               - bool           │
└─────────────────────────────────────────────────────────────┘
```

### Memory Scaling

| Board Size | Grid Cells | Memory Usage | Recommended GPU |
|------------|------------|--------------|-----------------|
| 50mm × 50mm | 500×500×4 | 200 MB | RTX 3060 (8GB) |
| 100mm × 100mm | 1000×1000×6 | 800 MB | RTX 4070 (12GB) |
| 200mm × 200mm | 2000×2000×8 | 3.2 GB | RTX 4080 (16GB) |
| 300mm × 300mm | 3000×3000×12 | 10.8 GB | RTX 5080 (16GB) |

## Performance Architecture

### Parallelization Strategy

**Level 1: Net-Level Parallelism**
- Process 256-1024 nets simultaneously
- Each net gets dedicated GPU threads
- Shared grid state with atomic updates

**Level 2: Wavefront Parallelism**
- All threads at same distance level
- Vectorized neighbor exploration
- Coalesced memory access patterns

**Level 3: Tile Parallelism**
- Multiple tiles processed concurrently
- Shared memory optimization
- Reduced global memory bandwidth

### Expected Performance

**Routing Speed Targets:**
```python
performance_targets = {
    "simple_board": {
        "nets": 100,
        "time_seconds": 1,
        "rate_nets_per_second": 100
    },
    "medium_board": {
        "nets": 1000, 
        "time_seconds": 10,
        "rate_nets_per_second": 100
    },
    "complex_board": {
        "nets": 8000,
        "time_seconds": 120,  # 2 minutes
        "rate_nets_per_second": 67
    }
}
```

**Scaling Factors:**
- **Net count:** Linear scaling up to memory limits
- **Grid resolution:** Quadratic impact on memory
- **Iteration count:** Linear impact on time
- **Layer count:** Linear impact on memory

## Installation & Quick Start

### System Requirements

#### Minimum Requirements
- Python 3.8 or higher
- NVIDIA GPU with Compute Capability 7.5+
- CUDA Toolkit 11.8+ or 12.x
- 8GB GPU RAM for basic boards
- 16GB System RAM

#### Python Dependencies
```bash
# Core dependencies
cupy-cuda12x>=12.0.0  # For CUDA 12.x
numpy>=1.21.0
scipy>=1.7.0
networkx>=2.6.0

# Optional dependencies
matplotlib>=3.4.0  # For visualization
pytest>=6.0.0  # For testing
```

### Installation

**Windows Quick Install (Recommended)**

Method 1 - Using File Explorer:
1. Download and extract OrthoRoute
2. Right-click on `install_windows.ps1`
3. Select "Run with PowerShell as administrator"

Method 2 - Using PowerShell:
1. Press `Windows + X` key combination
2. Click "Windows PowerShell (Admin)" or "Terminal (Admin)"
3. Navigate to the OrthoRoute directory:
```powershell
cd "path\to\OrthoRoute"  # Replace with your actual path
.\install_windows.ps1
```

**Manual Installation (Advanced Users)**
```bash
# Install CuPy (choose based on CUDA version)
pip install cupy-cuda12x  # For CUDA 12.x
pip install cupy-cuda11x  # For CUDA 11.x

# Install OrthoRoute
pip install orthoroute  # When released on PyPI

# OR install from source
git clone https://github.com/username/OrthoRoute.git
cd OrthoRoute
pip install -e .
```

### Quick Test
```python
import cupy as cp
from orthoroute.gpu_engine import OrthoRouteEngine
from orthoroute.grid_manager import GPUGrid

# Initialize engine
engine = OrthoRouteEngine()

# Create test grid (50mm x 50mm board)
grid = GPUGrid(width=500, height=500, layers=4)  # 0.1mm pitch

print("OrthoRoute ready!")
print(f"Grid dimensions: {grid.width}×{grid.height}×{grid.layers}")
```

### Running in KiCad

#### Plugin Installation
1. First install OrthoRoute and its dependencies as described above
2. Install the plugin in your KiCad plugins directory:

Windows:
```powershell
$PLUGIN_DIR="$env:APPDATA\kicad\7.0\3rdparty\plugins\OrthoRoute"
mkdir -p $PLUGIN_DIR
cp -r kicad_plugin\* $PLUGIN_DIR
```

Linux:
```bash
PLUGIN_DIR="~/.local/share/kicad/7.0/3rdparty/plugins/OrthoRoute"
mkdir -p $PLUGIN_DIR
cp -r kicad_plugin/* $PLUGIN_DIR
```

macOS:
```bash
PLUGIN_DIR="~/Library/Application Support/kicad/7.0/3rdparty/plugins/OrthoRoute"
mkdir -p "$PLUGIN_DIR"
cp -r kicad_plugin/* "$PLUGIN_DIR"
```

#### Using OrthoRoute in KiCad
1. Open your PCB in KiCad PCB Editor
2. Go to **Tools > External Plugins > OrthoRoute GPU Autorouter**
3. In the plugin dialog:
   - Set your desired routing grid (default 0.1mm)
   - Choose number of layers to route on
   - Select nets to route (or use "Route All")
   - Configure design rules (trace width, clearance)
4. Click "Start Routing"
5. Monitor progress in the status bar
6. Review results - routed traces will be added to your board

#### Tips for Best Results
- Clear any existing routes first (**Tools > Global Delete > Tracks**)
- Set board outline and keepout zones before routing
- Place components with adequate spacing
- Consider using routing regions to guide difficult areas
- Start with larger grid sizes for faster results

#### Troubleshooting
- If the plugin doesn't appear, check KiCad's **Plugin and Content Manager**
- Verify CUDA installation with `nvidia-smi` command
- Check KiCad console for any Python errors
- Try reducing batch size if you encounter memory errors

## Usage Examples

### Command Line Interface
```bash
# Route a board from JSON data
orthoroute input_board.json -o results.json --gpu-id 0

# With custom settings
orthoroute board.json --pitch 0.05 --layers 8 --iterations 30
```

### Python API
```python
from orthoroute.gpu_engine import OrthoRouteEngine

# Load board data
with open('board_data.json') as f:
    board_data = json.load(f)

# Route on GPU
engine = OrthoRouteEngine()
results = engine.route_board(board_data)

# Check results
if results['success']:
    print(f"Routed {results['stats']['successful_nets']} nets")
    print(f"Success rate: {results['stats']['success_rate']:.1f}%")
```

### KiCad Integration
1. Open your PCB in KiCad
2. Click the OrthoRoute toolbar button
3. Configure grid settings (pitch, layers, iterations)
4. Click "Start GPU Routing"
5. Routes are automatically applied to your board

## Error Handling

### Common Issues and Solutions

1. **GPU Memory Errors**
```python
try:
    engine = OrthoRouteEngine()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU memory exceeded - try reducing batch size")
    else:
        print(f"GPU error: {e}")
```

2. **Invalid Board Data**
```python
def validate_board_data(board_data):
    required_fields = ['bounds', 'grid', 'nets']
    missing = [f for f in required_fields if f not in board_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
```

3. **Network Errors**
```python
try:
    with open(input_file, 'r') as f:
        board_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading board data: {e}")
```

## Configuration Management

### Environment Variables
```bash
# GPU Selection
ORTHOROUTE_GPU_ID=0  # Select specific GPU
ORTHOROUTE_MEMORY_LIMIT=8192  # Limit GPU memory usage (MB)

# Performance Tuning
ORTHOROUTE_BATCH_SIZE=256  # Routing batch size
ORTHOROUTE_TILE_SIZE=64   # Processing tile size
ORTHOROUTE_MAX_ITERATIONS=30  # Maximum routing iterations

# Debug Options
ORTHOROUTE_DEBUG=1        # Enable debug logging
ORTHOROUTE_PROFILE=1      # Enable performance profiling
```

### Configuration File
```json
{
    "gpu": {
        "device_id": 0,
        "memory_limit_mb": 8192,
        "compute_mode": "default"
    },
    "routing": {
        "batch_size": 256,
        "tile_size": 64,
        "max_iterations": 30,
        "via_cost": 10,
        "congestion_factor": 1.5
    },
    "design_rules": {
        "min_track_width_nm": 100000,
        "min_clearance_nm": 150000,
        "min_via_size_nm": 200000
    },
    "debug": {
        "logging_level": "INFO",
        "profile_memory": true,
        "save_iterations": false
    }
}
```

## Troubleshooting Guide

### Common Issues

1. **GPU Memory Errors**
   - Reduce batch size
   - Enable tiled processing
   - Monitor with nvidia-smi
   - Check for memory leaks

2. **Performance Issues**
   - Optimize grid pitch
   - Tune batch size
   - Enable profiling
   - Check GPU utilization

3. **DRC Violations**
   - Verify design rules
   - Check clearance settings
   - Validate via parameters
   - Review layer constraints

### Error Recovery

```python
try:
    engine = OrthoRouteEngine()
    engine.load_board(board_data)
except GPUMemoryError:
    # Try with reduced memory usage
    config = engine.config
    config['batch_size'] //= 2
    config['tile_size'] = 32
    engine = OrthoRouteEngine(config)
except CUDADriverError:
    # Fall back to CPU mode
    engine = OrthoRouteEngine(mode='cpu')
finally:
    # Clean up resources
    engine.cleanup()
```

## Implementation Details

### CuPy Kernel Integration

**Built-in Operations (90% of functionality):**
```python
# Vectorized operations using CuPy
neighbors = cp.roll(current_positions, 1, axis=1)  # Shift coordinates
valid_mask = grid.availability[neighbors[:, 2], neighbors[:, 1], neighbors[:, 0]]
new_positions = neighbors[valid_mask]
```

**Custom CUDA Kernels (10% for performance-critical operations):**
```python
# Custom kernel for complex operations
wavefront_kernel = cp.RawKernel(r'''
extern "C" __global__
void expand_wavefront(int* positions, bool* grid, int* new_positions, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    // Custom wavefront expansion logic
    // ... optimized CUDA code ...
}
''', 'expand_wavefront')
```

### Data Structure Design

**Grid Representation:**
```python
class GPUGrid:
    def __init__(self, width, height, layers):
        # All arrays in CuPy for GPU processing
        self.availability = cp.ones((layers, height, width), dtype=cp.uint8)
        self.congestion = cp.ones((layers, height, width), dtype=cp.float32) 
        self.distance = cp.full((layers, height, width), 65535, dtype=cp.uint16)
        
    def mark_obstacle(self, x1, y1, x2, y2, layer=-1):
        """Mark rectangular obstacle region"""
        if layer == -1:
            self.availability[:, y1:y2, x1:x2] = 0
        else:
            self.availability[layer, y1:y2, x1:x2] = 0
```

## Performance Optimization

### Hardware Recommendations

| Board Complexity | Nets | GPU Recommendation | Memory |
|------------------|------|--------------------|---------|
| Simple | <500 | RTX 3060 (3,584 cores) | 8GB |
| Medium | 500-2K | RTX 4070 (5,888 cores) | 12GB |
| Complex | 2K-8K | RTX 4080 (9,728 cores) | 16GB |
| Extreme | 8K+ | RTX 5080+ (10,752+ cores) | 16GB+ |

### Optimization Tips
- **Grid pitch:** Smaller = better quality, larger = faster routing
- **Batch size:** Larger batches = better GPU utilization
- **Memory monitoring:** Watch GPU memory usage with `nvidia-smi`
- **Layer count:** Use minimum layers needed for your design

## Quality Metrics & Validation

### Route Quality Assessment
```python
def evaluate_route_quality(routes, design_rules):
    metrics = {
        'total_length': sum(route.length for route in routes),
        'via_count': sum(route.via_count for route in routes),
        'drc_violations': count_drc_violations(routes, design_rules),
        'congestion_score': calculate_congestion_score(routes),
        'completion_rate': len([r for r in routes if r.success]) / len(routes)
    }
    return metrics
```

### Design Rule Checking
- Minimum trace width compliance
- Clearance verification
- Via size validation
- Layer stack compliance
- Manufacturing constraints

## Development & Contributing

## Project Structure
```
OrthoRoute/
├── orthoroute/                    # Main package
│   ├── __init__.py               # Package initialization
│   ├── gpu_engine.py             # Core GPU routing engine (CuPy)
│   ├── grid_manager.py           # Grid data structures and tiling
│   ├── routing_algorithms.py     # Wavefront, A*, conflict resolution
│   ├── steiner_tree.py           # Multi-pin net handling
│   ├── design_rules.py           # DRC and validation
│   └── visualization.py          # Real-time routing display
├── kicad_plugin/                  # KiCad integration
│   ├── __init__.py               # Plugin registration
│   ├── orthoroute_kicad.py       # Main KiCad plugin
│   ├── board_export.py           # KiCad → JSON conversion
│   ├── route_import.py           # JSON → KiCad tracks/vias
│   └── ui_dialogs.py             # Configuration GUI
├── tests/                         # Test suite
│   ├── test_utils.py            # Test utilities and fixtures
│   ├── test_gpu_engine_mock.py  # GPU engine mock tests
│   ├── benchmark_boards/        # Performance test cases
│   └── integration_tests.py     # End-to-end testing
├── examples/                      # Example usage
│   ├── simple_board.py           # Basic routing example
│   ├── hypercube_backplane.py    # Complex board example
│   └── performance_test.py       # Benchmark script
├── docs/                          # Documentation
│   ├── README.md                 # This design document
│   ├── installation.md           # Setup instructions
│   ├── api_reference.md          # API documentation
│   └── performance_guide.md      # Optimization tips
├── requirements.txt               # Python dependencies
├── setup.py                      # Package installation
└── orthoroute_cli.py             # Command-line interface
```

### Development Setup
```bash
# Clone repository
git clone https://github.com/username/OrthoRoute.git
cd OrthoRoute

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black orthoroute/ kicad_plugin/
```

### Contributing Guidelines
1. **Pure Python** - Easy to read and modify
2. **Comprehensive tests** - Automated validation
3. **Clear documentation** - Document all public APIs
4. **Performance focus** - Profile GPU usage
5. **Community first** - Responsive to issues and PRs

## Future Roadmap

### Phase 1: Core Implementation ✓
- ✓ Basic CuPy routing engine
- ✓ GPU grid management
- ✓ Mock testing framework
- ✓ Memory optimization

### Phase 2: Current Work
- KiCad plugin integration
- Board import/export
- Design rule implementation
- Basic routing algorithms

### Phase 3: Future Features
- Advanced routing strategies
- Interactive visualization
- Performance optimization

## Support & Community

### Getting Help
- **GitHub Issues:** Bug reports and feature requests
- **License???** Any license request will be immediately closed

### Development Status
- **Core Engine:** Basic GPU infrastructure complete
- **Grid Management:** Tiled processing implemented
- **Testing:** Mock framework and unit tests in place
- **Next Steps:** KiCad integration and routing algorithms

**Acknowledgments:**
- Inspired by the original Connection Machine architecture
- Built on the shoulders of the CuPy and CUDA ecosystem
- Thanks to the KiCad community for extensible design tools


