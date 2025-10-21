"""
Input Validation Utilities
Ensures data integrity and provides helpful error messages
"""
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass


def validate_coordinates(latitude: float, longitude: float) -> Tuple[bool, Optional[str]]:
    """
    Validate geographic coordinates
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(latitude, (int, float)):
        return False, "Latitude must be a number"
    
    if not isinstance(longitude, (int, float)):
        return False, "Longitude must be a number"
    
    if latitude < -90 or latitude > 90:
        return False, f"Latitude must be between -90 and 90 (got {latitude})"
    
    if longitude < -180 or longitude > 180:
        return False, f"Longitude must be between -180 and 180 (got {longitude})"
    
    return True, None


def validate_system_config(
    system_size: float,
    efficiency: float,
    panel_tilt: float,
    panel_azimuth: float
) -> Tuple[bool, Optional[str]]:
    """
    Validate solar system configuration
    
    Args:
        system_size: System size in kWp
        efficiency: Panel efficiency (0-1)
        panel_tilt: Tilt angle in degrees (0-90)
        panel_azimuth: Azimuth angle in degrees (0-360)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if system_size <= 0 or system_size > 10000:
        return False, f"System size must be between 0 and 10000 kWp (got {system_size})"
    
    if efficiency <= 0 or efficiency > 1:
        return False, f"Efficiency must be between 0 and 1 (got {efficiency})"
    
    if panel_tilt < 0 or panel_tilt > 90:
        return False, f"Panel tilt must be between 0 and 90 degrees (got {panel_tilt})"
    
    if panel_azimuth < 0 or panel_azimuth >= 360:
        return False, f"Panel azimuth must be between 0 and 360 degrees (got {panel_azimuth})"
    
    return True, None


def validate_forecast_params(
    hours: int,
    min_hours: int = 1,
    max_hours: int = 720  # 30 days
) -> Tuple[bool, Optional[str]]:
    """
    Validate forecast parameters
    
    Args:
        hours: Number of hours to forecast
        min_hours: Minimum allowed hours
        max_hours: Maximum allowed hours
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(hours, int):
        return False, "Hours must be an integer"
    
    if hours < min_hours or hours > max_hours:
        return False, f"Hours must be between {min_hours} and {max_hours} (got {hours})"
    
    return True, None


def validate_battery_config(
    capacity: float,
    efficiency: float
) -> Tuple[bool, Optional[str]]:
    """
    Validate battery configuration
    
    Args:
        capacity: Battery capacity in kWh
        efficiency: Battery efficiency (0-1)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if capacity < 0 or capacity > 1000:
        return False, f"Battery capacity must be between 0 and 1000 kWh (got {capacity})"
    
    if efficiency <= 0 or efficiency > 1:
        return False, f"Battery efficiency must be between 0 and 1 (got {efficiency})"
    
    return True, None


def validate_financial_params(
    electricity_tariff: float,
    feed_in_tariff: float
) -> Tuple[bool, Optional[str]]:
    """
    Validate financial parameters
    
    Args:
        electricity_tariff: Cost per kWh
        feed_in_tariff: Payment per kWh exported
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if electricity_tariff < 0 or electricity_tariff > 10:
        return False, f"Electricity tariff must be between 0 and 10 $/kWh (got {electricity_tariff})"
    
    if feed_in_tariff < 0 or feed_in_tariff > 10:
        return False, f"Feed-in tariff must be between 0 and 10 $/kWh (got {feed_in_tariff})"
    
    if feed_in_tariff > electricity_tariff:
        logger.warning(f"Feed-in tariff ({feed_in_tariff}) is higher than electricity tariff ({electricity_tariff})")
    
    return True, None


def sanitize_string(value: str, max_length: int = 100) -> str:
    """
    Sanitize string input
    
    Args:
        value: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Remove control characters
    value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')
    
    # Trim to max length
    if len(value) > max_length:
        value = value[:max_length]
    
    return value.strip()


def validate_all_params(params: Dict) -> List[str]:
    """
    Validate all parameters at once
    
    Args:
        params: Dictionary of parameters to validate
        
    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    
    # Coordinates
    if 'latitude' in params and 'longitude' in params:
        valid, error = validate_coordinates(params['latitude'], params['longitude'])
        if not valid:
            errors.append(error)
    
    # System config
    if all(k in params for k in ['system_size', 'efficiency', 'panel_tilt', 'panel_azimuth']):
        valid, error = validate_system_config(
            params['system_size'],
            params['efficiency'],
            params['panel_tilt'],
            params['panel_azimuth']
        )
        if not valid:
            errors.append(error)
    
    # Forecast params
    if 'hours' in params:
        valid, error = validate_forecast_params(params['hours'])
        if not valid:
            errors.append(error)
    
    # Battery config
    if 'battery_capacity' in params and 'battery_efficiency' in params:
        valid, error = validate_battery_config(
            params['battery_capacity'],
            params['battery_efficiency']
        )
        if not valid:
            errors.append(error)
    
    # Financial params
    if 'electricity_tariff' in params and 'feed_in_tariff' in params:
        valid, error = validate_financial_params(
            params['electricity_tariff'],
            params['feed_in_tariff']
        )
        if not valid:
            errors.append(error)
    
    return errors
