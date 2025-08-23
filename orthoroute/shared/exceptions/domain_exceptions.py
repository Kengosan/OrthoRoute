"""Domain-specific exceptions."""
from .base_exceptions import OrthoRouteException, RoutingError


class BoardLoadError(OrthoRouteException):
    """Exception raised when board loading fails."""
    
    def __init__(self, message: str, file_path: str = None, **kwargs):
        """Initialize board load error.
        
        Args:
            message: Error message
            file_path: Path to board file that failed to load
        """
        super().__init__(message, **kwargs)
        self.file_path = file_path


class NetRoutingError(RoutingError):
    """Exception raised when net routing fails."""
    
    def __init__(self, message: str, net_id: str, reason: str = None, **kwargs):
        """Initialize net routing error.
        
        Args:
            message: Error message
            net_id: ID of net that failed routing
            reason: Specific reason for routing failure
        """
        super().__init__(message, net_id=net_id, **kwargs)
        self.reason = reason


class DRCViolationError(OrthoRouteException):
    """Exception raised for design rule check violations."""
    
    def __init__(self, message: str, violation_type: str = None, 
                 location: tuple = None, **kwargs):
        """Initialize DRC violation error.
        
        Args:
            message: Error message
            violation_type: Type of DRC violation
            location: Location of violation (x, y, layer)
        """
        super().__init__(message, **kwargs)
        self.violation_type = violation_type
        self.location = location


class AlgorithmError(RoutingError):
    """Exception raised for routing algorithm errors."""
    
    def __init__(self, message: str, algorithm_name: str = None, **kwargs):
        """Initialize algorithm error.
        
        Args:
            message: Error message
            algorithm_name: Name of algorithm that failed
        """
        super().__init__(message, **kwargs)
        self.algorithm_name = algorithm_name


class GridError(OrthoRouteException):
    """Exception raised for routing grid errors."""
    
    def __init__(self, message: str, grid_bounds: tuple = None, **kwargs):
        """Initialize grid error.
        
        Args:
            message: Error message
            grid_bounds: Grid bounds that caused error
        """
        super().__init__(message, **kwargs)
        self.grid_bounds = grid_bounds