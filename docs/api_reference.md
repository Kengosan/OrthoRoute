# OrthoRoute API Reference

## Core Classes and Methods

### OrthoRouteEngine

Main routing engine class that coordinates the entire routing process.

```python
class OrthoRouteEngine:
    def __init__(self, gpu_id: int = 0)
    def route(self, board_data: Dict, config: Dict = None, board=None) -> Dict
    def load_board_data(self, board_data: Dict) -> bool
    def _parse_nets(self, nets_data: List[Dict]) -> List[Net]
    def _create_tracks_from_path(self, net: Net, path_world: List[Dict]) -> List[Dict]
```

#### Key Methods

##### `route(board_data, config, board=None)`
**Purpose**: Main routing entry point
**Parameters**:
- `board_data`: Dictionary containing PCB data structure
- `config`: Routing configuration parameters
- `board`: KiCad board object reference (required for track creation)

**Returns**: Results dictionary with routing statistics and track information

**Recent Fixes Applied**:
- Added `board` parameter for track creation functionality
- Fixed method signature compatibility issues
- Enhanced error handling and logging

##### `_parse_nets(nets_data)`
**Purpose**: Convert raw net data into routing-ready Net objects
**Critical Fix**: Changed net-pad matching from object comparison to netcode comparison

**Original (Broken) Logic**:
```python
if pad.GetNet() == kicad_net:  # Object comparison failed
    valid_pins.append(pin)
```

**Fixed Logic**:
```python
pad_net = pad.GetNet()
if pad_net.GetNetCode() == netcode:  # Netcode comparison works
    valid_pins.append(pin)
```

##### `_create_tracks_from_path(net, path_world)`
**Purpose**: Convert routing paths to actual KiCad PCB_TRACK objects
**Status**: Newly implemented to address missing track creation

**Key Features**:
- Creates PCB_TRACK objects for each path segment
- Handles via creation for layer changes
- Integrates with KiCad board structure
- Sets proper net associations

### Net Class

Represents a routable net with pins and routing information.

```python
@dataclass
class Net:
    id: int
    name: str
    pins: List[Point3D]
    width_nm: int = 200000
    route_path: List[Point3D] = None
    routed: bool = False
    via_count: int = 0
    total_length: float = 0.0
    kicad_net: object = None  # KiCad net object reference
```

### Point3D Class

3D coordinate representation for routing grid.

```python
@dataclass
class Point3D:
    x: int  # Grid X coordinate
    y: int  # Grid Y coordinate  
    z: int  # Layer number
```

## KiCad Integration

### Board Data Extraction

The plugin extracts data from KiCad's PCB representation:

```python
def extract_board_data(self, board):
    """Extract routing data from KiCad board"""
    # Get board boundaries
    bounds = self._get_board_bounds(board)
    
    # Extract nets with proper netcode matching
    nets = []
    for kicad_net in board.GetNetInfo():
        netcode = kicad_net.GetNetCode()
        if netcode == 0:  # Skip unconnected
            continue
            
        # Find pads belonging to this net
        net_pads = []
        for footprint in board.GetFootprints():
            for pad in footprint.Pads():
                pad_net = pad.GetNet()
                if pad_net.GetNetCode() == netcode:  # Fixed comparison
                    net_pads.append(pad)
```

### Recent API Compatibility Fixes

#### Net-Pad Relationship Detection
**Problem**: Plugin couldn't detect which pads belong to which nets
**Root Cause**: Used object equality instead of netcode comparison
**Solution**: Implemented proper KiCad API usage pattern

#### Track Creation Integration
**Problem**: Routing completed but no tracks appeared on PCB
**Solution**: Added KiCad PCB_TRACK object creation and board integration

#### wxPython Dialog Compatibility
**Problem**: UI dialogs failed to display in KiCad 8.0+
**Solution**: Updated dialog constructors for current wxPython version

## Configuration Parameters

### Routing Configuration
```python
default_config = {
    'grid_pitch_mm': 0.1,           # Grid resolution
    'max_iterations': 5,            # Rerouting attempts
    'via_cost': 10,                 # Layer change penalty
    'batch_size': 20,               # Parallel processing
    'show_progress': True,          # UI progress display
    'debug_output': False,          # Verbose logging
    'routing_algorithm': 'gpu_wavefront'  # Algorithm selection
}
```

### Board Data Structure
```python
board_data = {
    'bounds': {
        'width_nm': int,    # Board width in nanometers
        'height_nm': int,   # Board height in nanometers
        'layers': int       # Number of routing layers
    },
    'nets': [
        {
            'id': int,              # Unique net identifier
            'name': str,            # Net name
            'pins': [               # Pin locations
                {
                    'x': int,       # X coordinate (nanometers)
                    'y': int,       # Y coordinate (nanometers)
                    'layer': int    # Layer number
                }
            ],
            'width_nm': int,        # Track width
            'kicad_net': object     # KiCad net reference
        }
    ],
    'obstacles': {}                 # Existing tracks/components
}
```

## Error Handling and Debugging

### Recent Debugging Framework
The plugin now includes comprehensive debugging capabilities:

```python
# Enhanced error reporting
def debug_print(self, message):
    """Centralized debug output"""
    if self.config.get('debug_output', False):
        print(f"OrthoRoute: {message}")

# API investigation tools
def investigate_board_api(self, board):
    """Analyze KiCad board structure for debugging"""
    # Created during debugging session
    # Reveals actual board state vs expected state
```

### Common Integration Issues

#### Issue: "No nets found to route"
**Diagnosis**: Net detection logic fails
**Investigation**: Use API investigation tools to verify board has nets
**Solution**: Check net-pad matching logic and KiCad API usage

#### Issue: "Plugin runs but no tracks created"
**Diagnosis**: Routing completes but track creation fails
**Investigation**: Verify board reference is passed to routing engine
**Solution**: Ensure `_create_tracks_from_path()` receives valid board object

#### Issue: "Plugin crashes on startup"
**Diagnosis**: Import or API compatibility problems
**Investigation**: Check KiCad Python console for error details
**Solution**: Verify KiCad version compatibility and API usage patterns

## Version History

### v1.0.0 (July 2025) - Major Debugging Release
- ✅ Fixed plugin crashes and import errors
- ✅ Added missing track creation functionality
- ✅ Fixed wxPython UI compatibility
- ✅ Corrected critical net-pad matching logic
- 🔄 **Current**: Investigating remaining net detection issues

### Development Notes
This release represents a comprehensive debugging effort to resolve the core issue: "plugin doesn't actually route." While significant progress has been made in eliminating errors and implementing missing functionality, further refinement of net detection logic may be needed for complete compatibility across all KiCad board configurations.
