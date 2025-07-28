# OrthoRoute API Reference

## Core Components

### GPU Engine (`gpu_engine.py`)

The GPU engine module handles all CUDA-accelerated computations for the routing process.

#### Classes

##### `GPUEngine`
Main class for GPU-accelerated routing operations.

```python
class GPUEngine:
    def __init__(self, grid_size: tuple, memory_limit: int = None):
        """
        Initialize GPU engine with grid dimensions.
        
        Args:
            grid_size (tuple): (width, height) of routing grid
            memory_limit (int, optional): GPU memory limit in MB
        """
        
    def route_net(self, start_points: list, end_points: list) -> List[Path]:
        """
        Route a single net between multiple start and end points.
        
        Args:
            start_points (list): List of starting coordinates [(x1,y1), ...]
            end_points (list): List of target coordinates [(x1,y1), ...]
            
        Returns:
            List[Path]: List of path objects representing the routed traces
        """
```

### Grid Manager (`grid_manager.py`)

Handles the routing grid representation and management.

#### Classes

##### `GridManager`
Manages the routing grid and layer stack-up.

```python
class GridManager:
    def __init__(self, board_dims: tuple, resolution: float = 0.1):
        """
        Initialize routing grid.
        
        Args:
            board_dims (tuple): Physical board dimensions (width, height)
            resolution (float): Grid resolution in mm
        """
        
    def add_obstacle(self, coords: list, layer: int):
        """
        Add obstacle to routing grid.
        
        Args:
            coords (list): List of (x,y) coordinates defining obstacle
            layer (int): Layer number for obstacle
        """
```

### Routing Algorithms (`routing_algorithms.py`)

Implementation of various routing algorithms.

#### Functions

```python
def astar_route(start: Point, end: Point, grid: np.ndarray) -> Path:
    """
    A* routing algorithm implementation.
    
    Args:
        start (Point): Starting point
        end (Point): Target point
        grid (np.ndarray): Routing grid
        
    Returns:
        Path: Routed path object
    """

def steiner_tree(points: List[Point], grid: np.ndarray) -> List[Path]:
    """
    Create Steiner tree for multi-point net.
    
    Args:
        points (List[Point]): List of points to connect
        grid (np.ndarray): Routing grid
        
    Returns:
        List[Path]: List of paths forming Steiner tree
    """
```

### Design Rules (`design_rules.py`)

Handles DRC constraints and rule checking.

#### Classes

##### `DesignRules`
Manages and enforces design rules.

```python
class DesignRules:
    def __init__(self):
        """Initialize design rules with default values"""
        
    def set_clearance(self, value: float, layers: List[int] = None):
        """
        Set clearance rule.
        
        Args:
            value (float): Clearance value in mm
            layers (List[int], optional): Specific layers for rule
        """
        
    def check_violations(self, path: Path) -> List[Violation]:
        """
        Check path for design rule violations.
        
        Args:
            path (Path): Path to check
            
        Returns:
            List[Violation]: List of found violations
        """
```

## KiCad Plugin Integration

### Board Export (`board_export.py`)

Handles KiCad PCB data export.

#### Functions

```python
def export_board(board: pcbnew.BOARD) -> Dict:
    """
    Export KiCad board data.
    
    Args:
        board (pcbnew.BOARD): KiCad board object
        
    Returns:
        Dict: Exported board data
    """

def import_routes(board: pcbnew.BOARD, routes: List[Path]):
    """
    Import routes back to KiCad.
    
    Args:
        board (pcbnew.BOARD): KiCad board object
        routes (List[Path]): Routes to import
    """
```

## Data Structures

### Point
```python
class Point:
    def __init__(self, x: int, y: int, layer: int = 0):
        self.x = x
        self.y = y
        self.layer = layer
```

### Path
```python
class Path:
    def __init__(self, points: List[Point], width: float):
        self.points = points
        self.width = width
```

### Violation
```python
class Violation:
    def __init__(self, type: str, location: Point, details: str):
        self.type = type
        self.location = location
        self.details = details
```

## Examples

### Basic Usage

```python
from orthoroute import GPUEngine, GridManager
from orthoroute.design_rules import DesignRules

# Initialize components
grid_mgr = GridManager(board_dims=(100, 100), resolution=0.1)
gpu_engine = GPUEngine(grid_size=grid_mgr.grid_size)
rules = DesignRules()

# Set design rules
rules.set_clearance(0.2)  # 0.2mm clearance

# Add obstacles
grid_mgr.add_obstacle([(10,10), (20,20)], layer=0)

# Route a net
start_points = [(0,0)]
end_points = [(50,50)]
routes = gpu_engine.route_net(start_points, end_points)

# Check violations
violations = rules.check_violations(routes[0])
```

### KiCad Plugin Usage

```python
import pcbnew
from orthoroute.kicad_plugin import export_board, import_routes

# Export board data
board = pcbnew.GetBoard()
board_data = export_board(board)

# Process with OrthoRoute
# ... routing operations ...

# Import routes back to KiCad
import_routes(board, routes)
```
