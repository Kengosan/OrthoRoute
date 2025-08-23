"""Validation utilities for OrthoRoute."""
import re
from typing import Tuple, Any

from ..exceptions import ValidationError


def validate_coordinates(x: float, y: float, bounds: Tuple[float, float, float, float] = None) -> None:
    """Validate coordinate values.
    
    Args:
        x: X coordinate
        y: Y coordinate  
        bounds: Optional bounds as (min_x, min_y, max_x, max_y)
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    if not isinstance(x, (int, float)):
        raise ValidationError(f"X coordinate must be numeric, got {type(x)}", field="x", value=x)
    
    if not isinstance(y, (int, float)):
        raise ValidationError(f"Y coordinate must be numeric, got {type(y)}", field="y", value=y)
    
    if bounds:
        min_x, min_y, max_x, max_y = bounds
        
        if x < min_x or x > max_x:
            raise ValidationError(
                f"X coordinate {x} out of bounds [{min_x}, {max_x}]",
                field="x", value=x
            )
        
        if y < min_y or y > max_y:
            raise ValidationError(
                f"Y coordinate {y} out of bounds [{min_y}, {max_y}]",
                field="y", value=y
            )


def validate_layer_index(layer: int, max_layers: int = None) -> None:
    """Validate layer index.
    
    Args:
        layer: Layer index
        max_layers: Optional maximum layer count
        
    Raises:
        ValidationError: If layer index is invalid
    """
    if not isinstance(layer, int):
        raise ValidationError(f"Layer must be integer, got {type(layer)}", field="layer", value=layer)
    
    if layer < 0:
        raise ValidationError(f"Layer index must be non-negative, got {layer}", field="layer", value=layer)
    
    if max_layers is not None and layer >= max_layers:
        raise ValidationError(
            f"Layer index {layer} exceeds maximum {max_layers - 1}",
            field="layer", value=layer
        )


def validate_net_id(net_id: str) -> None:
    """Validate net ID format.
    
    Args:
        net_id: Net identifier
        
    Raises:
        ValidationError: If net ID is invalid
    """
    if not isinstance(net_id, str):
        raise ValidationError(f"Net ID must be string, got {type(net_id)}", field="net_id", value=net_id)
    
    if not net_id.strip():
        raise ValidationError("Net ID cannot be empty", field="net_id", value=net_id)
    
    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not re.match(r'^[a-zA-Z0-9_-]+$', net_id):
        raise ValidationError(
            f"Net ID contains invalid characters: {net_id}",
            field="net_id", value=net_id
        )


def validate_positive_number(value: Any, field_name: str) -> None:
    """Validate that a value is a positive number.
    
    Args:
        value: Value to validate
        field_name: Name of field for error reporting
        
    Raises:
        ValidationError: If value is not a positive number
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field_name} must be numeric, got {type(value)}",
            field=field_name, value=value
        )
    
    if value <= 0:
        raise ValidationError(
            f"{field_name} must be positive, got {value}",
            field=field_name, value=value
        )


def validate_non_negative_number(value: Any, field_name: str) -> None:
    """Validate that a value is a non-negative number.
    
    Args:
        value: Value to validate
        field_name: Name of field for error reporting
        
    Raises:
        ValidationError: If value is not non-negative
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field_name} must be numeric, got {type(value)}",
            field=field_name, value=value
        )
    
    if value < 0:
        raise ValidationError(
            f"{field_name} must be non-negative, got {value}",
            field=field_name, value=value
        )


def validate_range(value: Any, field_name: str, min_val: float, max_val: float) -> None:
    """Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        field_name: Name of field for error reporting
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Raises:
        ValidationError: If value is not in range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field_name} must be numeric, got {type(value)}",
            field=field_name, value=value
        )
    
    if value < min_val or value > max_val:
        raise ValidationError(
            f"{field_name} must be between {min_val} and {max_val}, got {value}",
            field=field_name, value=value
        )