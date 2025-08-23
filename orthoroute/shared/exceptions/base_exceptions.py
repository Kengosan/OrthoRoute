"""Base exceptions for OrthoRoute."""


class OrthoRouteException(Exception):
    """Base exception class for OrthoRoute."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """Initialize exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional additional error details
        """
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of exception."""
        base_msg = super().__str__()
        if self.error_code:
            return f"[{self.error_code}] {base_msg}"
        return base_msg


class ConfigurationError(OrthoRouteException):
    """Exception raised for configuration-related errors."""
    pass


class ValidationError(OrthoRouteException):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, field: str = None, value=None, **kwargs):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
        """
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value


class RoutingError(OrthoRouteException):
    """Exception raised for routing-related errors."""
    
    def __init__(self, message: str, net_id: str = None, **kwargs):
        """Initialize routing error.
        
        Args:
            message: Error message
            net_id: Net ID that failed routing
        """
        super().__init__(message, **kwargs)
        self.net_id = net_id


class KiCadError(OrthoRouteException):
    """Exception raised for KiCad integration errors."""
    
    def __init__(self, message: str, api_mode: str = None, **kwargs):
        """Initialize KiCad error.
        
        Args:
            message: Error message
            api_mode: KiCad API mode being used (ipc, swig, file)
        """
        super().__init__(message, **kwargs)
        self.api_mode = api_mode


class GPUError(OrthoRouteException):
    """Exception raised for GPU-related errors."""
    
    def __init__(self, message: str, gpu_type: str = None, **kwargs):
        """Initialize GPU error.
        
        Args:
            message: Error message
            gpu_type: GPU type (cuda, cpu, etc.)
        """
        super().__init__(message, **kwargs)
        self.gpu_type = gpu_type