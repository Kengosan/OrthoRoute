# OrthoRoute Modular Architecture

This document describes the new modular architecture of OrthoRoute, designed to support multiple routing algorithms and clean separation of concerns.

## Architecture Overview

The refactored OrthoRoute uses a modular design with clear separation between:

1. **Core Infrastructure** - Shared functionality across all routing algorithms
2. **Routing Engines** - Pluggable routing algorithm implementations  
3. **Data Structures** - Common data structures and utilities
4. **Factory Interface** - Clean API for creating and configuring routers

## Directory Structure

```
src/
├── core/                          # Core infrastructure components
│   ├── __init__.py
│   ├── drc_rules.py              # KiCad DRC rules extraction & management
│   ├── gpu_manager.py            # GPU resource management & acceleration
│   └── board_interface.py        # Board data abstraction & access patterns
│
├── data_structures/              # Common data structures
│   ├── __init__.py
│   └── grid_config.py           # Grid configuration & coordinate conversion
│
├── routing_engines/              # Pluggable routing algorithms
│   ├── __init__.py
│   ├── base_router.py           # Abstract base class for all routers
│   └── lees_router.py           # Lee's wavefront expansion implementation
│
├── autorouter_factory.py         # Main factory & interface
├── orthoroute_plugin.py          # Main plugin entry point
└── orthoroute_window.py          # UI components (unchanged)
```

## Key Components

### Core Infrastructure (`src/core/`)

#### `DRCRules` - Design Rule Management
- Extracts DRC rules from KiCad using priority hierarchy
- Manages netclass-specific constraints
- Provides clearance calculations following KiCad methodology
- **Used by**: All routing algorithms for constraint checking

#### `GPUManager` - GPU Resource Management  
- Handles GPU initialization and memory management
- Provides GPU/CPU abstraction layer
- Manages CuPy arrays and memory pools
- **Used by**: Routing algorithms that support GPU acceleration

#### `BoardInterface` - Board Data Access
- Abstracts PCB geometry data access
- Caches frequently accessed data for performance
- Provides routing-specific access patterns
- **Used by**: All routing algorithms for board data

### Routing Engines (`src/routing_engines/`)

#### `BaseRouter` - Abstract Router Interface
- Defines common interface for all routing algorithms
- Handles obstacle grid initialization  
- Provides routing statistics and progress callbacks
- **Extended by**: All specific routing algorithm implementations

#### `LeeRouter` - Lee's Algorithm Implementation
- Implements Lee's wavefront expansion with GPU acceleration
- Supports multi-layer routing with strategic via placement
- Uses minimum spanning tree for multi-pad nets
- **Extends**: `BaseRouter`

### Factory Interface (`autorouter_factory.py`)

#### `AutorouterEngine` - Main Factory Class
- Creates and manages routing engine instances
- Provides algorithm switching capability
- Maintains backward compatibility with existing code
- **Used by**: Main plugin and UI code

## Usage Examples

### Basic Usage

```python
from autorouter_factory import create_autorouter, RoutingAlgorithm

# Create autorouter with Lee's algorithm
autorouter = create_autorouter(
    board_data=board_data,
    kicad_interface=kicad_interface,
    use_gpu=True,
    algorithm=RoutingAlgorithm.LEE_WAVEFRONT
)

# Route all nets
stats = autorouter.route_all_nets(timeout_per_net=5.0)
print(f"Routed {stats.nets_routed}/{stats.nets_attempted} nets")

# Get results
tracks = autorouter.get_routed_tracks()
vias = autorouter.get_routed_vias()
```

### Algorithm Switching

```python
# Switch to different algorithm (when available)
autorouter.set_routing_algorithm(RoutingAlgorithm.MANHATTAN)

# Route with new algorithm
result = autorouter.route_single_net("VCC")
```

### Custom Router Implementation

```python
from routing_engines.base_router import BaseRouter, RoutingResult

class MyCustomRouter(BaseRouter):
    def route_net(self, net_name: str, timeout: float) -> RoutingResult:
        # Implement your routing algorithm
        pass
    
    def route_two_pads(self, pad_a, pad_b, net_name, timeout):
        # Implement two-pad routing
        pass
```

## Migration from Legacy Code

The new architecture maintains backward compatibility through the `AutorouterEngine` class:

### Legacy Code (Still Works)
```python
from autorouter import AutorouterEngine

engine = AutorouterEngine(board_data, kicad_interface)
success = engine._route_single_net("VCC")
```

### New Modular Code (Recommended)
```python
from autorouter_factory import create_autorouter

engine = create_autorouter(board_data, kicad_interface)  
success = engine.route_single_net("VCC")
```

## Available Routing Algorithms

### Currently Implemented

1. **Lee's Wavefront (`LEE_WAVEFRONT`)**
   - GPU-accelerated wavefront expansion
   - Multi-layer support with strategic via placement
   - Minimum spanning tree for multi-pad nets
   - DRC-aware obstacle avoidance

### Future Implementations

2. **Manhattan Routing (`MANHATTAN`)**
   - H/V grid layers (horizontal on F.Cu, vertical on B.Cu)
   - Blind and buried vias for layer switching
   - Orthogonal routing patterns

3. **A* Pathfinding (`ASTAR`)**
   - Heuristic-guided pathfinding  
   - Faster than Lee's for long connections
   - Memory efficient for large boards

## Benefits of the New Architecture

### 🎯 **Modularity**
- Clean separation of concerns
- Easy to add new routing algorithms
- Independent testing of components

### 🚀 **Performance** 
- Shared GPU infrastructure
- Optimized data access patterns
- Reusable obstacle grids

### 🛡️ **Maintainability**
- Clear interfaces and abstractions
- Reduced code duplication
- Better error handling

### 🔧 **Extensibility**
- Simple to add new algorithms
- Plugin architecture for routing engines
- Configurable algorithm parameters

### 🔄 **Backward Compatibility**
- Existing code continues to work
- Gradual migration path
- Legacy API preserved

## Development Guidelines

### Adding a New Routing Algorithm

1. **Create the router class**:
   ```python
   # src/routing_engines/my_router.py
   class MyRouter(BaseRouter):
       def route_net(self, net_name, timeout):
           # Implementation
       
       def route_two_pads(self, pad_a, pad_b, net_name, timeout):
           # Implementation
   ```

2. **Add to the enum**:
   ```python
   # autorouter_factory.py
   class RoutingAlgorithm(Enum):
       MY_ALGORITHM = "my_algorithm"
   ```

3. **Register in factory**:
   ```python
   # autorouter_factory.py  
   def _initialize_routing_engines(self):
       self._routing_engines[RoutingAlgorithm.MY_ALGORITHM] = MyRouter(...)
   ```

### Core Infrastructure Guidelines

- **DRC Rules**: Add methods to `DRCRules` for new constraint types
- **GPU Manager**: Extend for new GPU operations or data types  
- **Board Interface**: Add methods for new board data access patterns
- **Base Router**: Add common functionality that all algorithms can use

## Testing

The modular architecture enables better testing:

```python
# Test individual components
def test_drc_rules():
    drc = DRCRules(board_data)
    assert drc.get_clearance_for_net("VCC") > 0

def test_lee_router():
    router = LeeRouter(board_interface, drc_rules, gpu_manager, grid_config)
    result = router.route_two_pads(pad_a, pad_b, "TEST_NET")
    assert result is not None
```

## Performance Considerations

The new architecture maintains the performance optimizations from the previous system:

- **Incremental Obstacle Grids**: Reuse grids between routing attempts
- **GPU Acceleration**: Shared GPU resources across algorithms  
- **Cached Data Access**: Board interface caches frequently accessed data
- **Parallel Processing**: GPU manager enables batch operations

## Future Enhancements

The modular architecture enables these future features:

- **Multi-Algorithm Routing**: Try multiple algorithms per net
- **Algorithm Parameters**: Configurable algorithm-specific settings
- **Routing Analytics**: Per-algorithm performance metrics
- **Custom DRC Rules**: User-defined design rule extensions
- **Distributed Routing**: Multi-GPU or cloud-based routing

---

This new architecture provides a solid foundation for OrthoRoute's evolution into a world-class PCB autorouting system while maintaining the performance and reliability of the existing Lee's algorithm implementation.
