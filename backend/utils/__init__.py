"""
Utility functions for SunShift
"""
from .validators import (
    validate_coordinates,
    validate_system_config,
    validate_forecast_params,
    validate_battery_config,
    validate_financial_params,
    validate_all_params,
    sanitize_string
)

from .cache import (
    TTLCache,
    weather_cache,
    forecast_cache,
    currency_cache,
    cached,
    async_cached,
    location_key
)

from .calculations import (
    calculate_peak_sun_hours,
    calculate_energy_output,
    calculate_cost_savings,
    calculate_carbon_impact,
    calculate_payback_period,
    calculate_battery_metrics,
    calculate_self_consumption_rate,
    calculate_capacity_factor,
    aggregate_hourly_to_daily,
    calculate_performance_metrics
)

__all__ = [
    # Validators
    'validate_coordinates',
    'validate_system_config',
    'validate_forecast_params',
    'validate_battery_config',
    'validate_financial_params',
    'validate_all_params',
    'sanitize_string',
    
    # Calculations
    'calculate_peak_sun_hours',
    'calculate_energy_output',
    'calculate_cost_savings',
    'calculate_carbon_impact',
    'calculate_payback_period',
    'calculate_battery_metrics',
    'calculate_self_consumption_rate',
    'calculate_capacity_factor',
    'aggregate_hourly_to_daily',
    'calculate_performance_metrics',
    
    # Caching
    'TTLCache',
    'weather_cache',
    'forecast_cache',
    'currency_cache',
    'cached',
    'async_cached',
    'location_key',
]

