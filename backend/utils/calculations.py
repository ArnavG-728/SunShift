"""
Calculation Utilities
Reusable calculation functions for energy, carbon, and financial metrics
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


def calculate_peak_sun_hours(irradiance_data: List[float]) -> float:
    """
    Calculate Peak Sun Hours (PSH) from irradiance data
    
    Args:
        irradiance_data: List of irradiance values in W/m²
        
    Returns:
        Peak sun hours
    """
    # PSH = total irradiance / 1000 W/m²
    total_irradiance = sum(irradiance_data)
    psh = total_irradiance / 1000
    return round(psh, 2)


def calculate_energy_output(
    irradiance: float,
    system_size: float,
    efficiency: float,
    temperature: float = 25.0,
    performance_ratio: float = 0.78
) -> float:
    """
    Calculate energy output from solar panels
    
    Args:
        irradiance: Solar irradiance in W/m²
        system_size: System size in kWp
        efficiency: Panel efficiency (0-1)
        temperature: Panel temperature in °C
        performance_ratio: System performance ratio
        
    Returns:
        Energy output in kWh
    """
    # Temperature derating (-0.4% per °C above 25°C)
    temp_factor = 1 - 0.004 * (temperature - 25)
    temp_factor = max(0.7, min(1.0, temp_factor))
    
    # Energy = (Irradiance / 1000) × System Size × Efficiency × Temp Factor × PR
    energy = (irradiance / 1000) * system_size * efficiency * temp_factor * performance_ratio
    
    return max(0, energy)


def calculate_cost_savings(
    solar_energy_kwh: float,
    electricity_tariff: float,
    export_percentage: float = 0.3,
    feed_in_tariff: float = 0.08
) -> Dict[str, float]:
    """
    Calculate financial savings from solar energy
    
    Args:
        solar_energy_kwh: Solar energy generated in kWh
        electricity_tariff: Cost per kWh from grid
        export_percentage: Percentage of energy exported (0-1)
        feed_in_tariff: Payment per kWh exported
        
    Returns:
        Dictionary with savings breakdown
    """
    # Energy consumed on-site
    consumed = solar_energy_kwh * (1 - export_percentage)
    grid_cost_avoided = consumed * electricity_tariff
    
    # Energy exported to grid
    exported = solar_energy_kwh * export_percentage
    export_revenue = exported * feed_in_tariff
    
    total_savings = grid_cost_avoided + export_revenue
    
    return {
        'consumed_kwh': round(consumed, 2),
        'exported_kwh': round(exported, 2),
        'grid_cost_avoided': round(grid_cost_avoided, 2),
        'export_revenue': round(export_revenue, 2),
        'total_savings': round(total_savings, 2)
    }


def calculate_carbon_impact(
    solar_energy_kwh: float,
    grid_carbon_factor: float = 0.5
) -> Dict[str, float]:
    """
    Calculate carbon emissions avoided
    
    Args:
        solar_energy_kwh: Solar energy generated in kWh
        grid_carbon_factor: Grid carbon intensity in kg CO2 per kWh
        
    Returns:
        Dictionary with carbon impact metrics
    """
    co2_avoided_kg = solar_energy_kwh * grid_carbon_factor
    co2_avoided_tons = co2_avoided_kg / 1000
    
    # Equivalents for context
    trees_equivalent = co2_avoided_kg / 21  # 1 tree absorbs ~21kg CO2/year
    car_miles_avoided = co2_avoided_kg / 0.404  # 1 mile = ~0.404kg CO2
    homes_powered = solar_energy_kwh / 30  # Average home uses ~30 kWh/day
    
    return {
        'co2_avoided_kg': round(co2_avoided_kg, 2),
        'co2_avoided_tons': round(co2_avoided_tons, 4),
        'trees_equivalent': round(trees_equivalent, 2),
        'car_miles_avoided': round(car_miles_avoided, 1),
        'homes_powered_days': round(homes_powered, 2)
    }


def calculate_payback_period(
    system_cost: float,
    annual_savings: float,
    annual_degradation: float = 0.005
) -> Dict[str, float]:
    """
    Calculate simple payback period for solar system
    
    Args:
        system_cost: Total system cost in $
        annual_savings: Annual savings in $
        annual_degradation: Annual panel degradation rate (default 0.5%)
        
    Returns:
        Dictionary with payback analysis
    """
    if annual_savings <= 0:
        return {
            'simple_payback_years': float('inf'),
            'discounted_payback_years': float('inf'),
            'total_savings_25_years': 0
        }
    
    # Simple payback (no degradation)
    simple_payback = system_cost / annual_savings
    
    # Calculate with degradation
    cumulative_savings = 0
    year = 0
    current_savings = annual_savings
    
    while cumulative_savings < system_cost and year < 50:
        cumulative_savings += current_savings
        current_savings *= (1 - annual_degradation)
        year += 1
    
    discounted_payback = year if cumulative_savings >= system_cost else float('inf')
    
    # Total savings over 25 years
    total_savings_25y = sum(
        annual_savings * ((1 - annual_degradation) ** y)
        for y in range(25)
    )
    
    return {
        'simple_payback_years': round(simple_payback, 1),
        'discounted_payback_years': round(discounted_payback, 1),
        'total_savings_25_years': round(total_savings_25y, 2),
        'roi_percentage': round((total_savings_25y / system_cost - 1) * 100, 1)
    }


def calculate_battery_metrics(
    battery_capacity: float,
    charge_rate: float,
    discharge_rate: float,
    efficiency: float = 0.95
) -> Dict[str, float]:
    """
    Calculate battery performance metrics
    
    Args:
        battery_capacity: Battery capacity in kWh
        charge_rate: Charging rate in kW
        discharge_rate: Discharging rate in kW
        efficiency: Round-trip efficiency
        
    Returns:
        Dictionary with battery metrics
    """
    # Time to full charge/discharge
    charge_time_hours = battery_capacity / charge_rate if charge_rate > 0 else 0
    discharge_time_hours = battery_capacity / discharge_rate if discharge_rate > 0 else 0
    
    # Usable capacity (accounting for efficiency)
    usable_capacity = battery_capacity * efficiency
    
    # Energy loss per cycle
    energy_loss_per_cycle = battery_capacity * (1 - efficiency)
    
    return {
        'usable_capacity_kwh': round(usable_capacity, 2),
        'charge_time_hours': round(charge_time_hours, 2),
        'discharge_time_hours': round(discharge_time_hours, 2),
        'energy_loss_per_cycle_kwh': round(energy_loss_per_cycle, 2),
        'efficiency_percentage': round(efficiency * 100, 1)
    }


def calculate_self_consumption_rate(
    solar_production: List[float],
    consumption: List[float]
) -> Dict[str, float]:
    """
    Calculate self-consumption and self-sufficiency rates
    
    Args:
        solar_production: Hourly solar production in kWh
        consumption: Hourly consumption in kWh
        
    Returns:
        Dictionary with consumption metrics
    """
    total_production = sum(solar_production)
    total_consumption = sum(consumption)
    
    # Self-consumed energy (minimum of production and consumption each hour)
    self_consumed = sum(min(p, c) for p, c in zip(solar_production, consumption))
    
    # Self-consumption rate (% of production used on-site)
    self_consumption_rate = (self_consumed / total_production * 100) if total_production > 0 else 0
    
    # Self-sufficiency rate (% of consumption met by solar)
    self_sufficiency_rate = (self_consumed / total_consumption * 100) if total_consumption > 0 else 0
    
    # Grid export and import
    grid_export = sum(max(0, p - c) for p, c in zip(solar_production, consumption))
    grid_import = sum(max(0, c - p) for p, c in zip(solar_production, consumption))
    
    return {
        'self_consumption_rate': round(self_consumption_rate, 1),
        'self_sufficiency_rate': round(self_sufficiency_rate, 1),
        'self_consumed_kwh': round(self_consumed, 2),
        'grid_export_kwh': round(grid_export, 2),
        'grid_import_kwh': round(grid_import, 2)
    }


def calculate_capacity_factor(
    actual_energy: float,
    system_size: float,
    hours: int = 24
) -> float:
    """
    Calculate capacity factor (actual vs theoretical maximum)
    
    Args:
        actual_energy: Actual energy produced in kWh
        system_size: System size in kWp
        hours: Time period in hours
        
    Returns:
        Capacity factor as percentage
    """
    theoretical_max = system_size * hours
    capacity_factor = (actual_energy / theoretical_max * 100) if theoretical_max > 0 else 0
    
    return round(capacity_factor, 1)


def aggregate_hourly_to_daily(hourly_data: List[Dict]) -> List[Dict]:
    """
    Aggregate hourly data to daily summaries
    
    Args:
        hourly_data: List of hourly data dictionaries
        
    Returns:
        List of daily aggregated data
    """
    daily_data = {}
    
    for hour in hourly_data:
        timestamp = hour.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        date_key = timestamp.date().isoformat()
        
        if date_key not in daily_data:
            daily_data[date_key] = {
                'date': date_key,
                'total_kwh': 0,
                'avg_temp': [],
                'avg_irradiance': [],
                'peak_kwh': 0
            }
        
        energy = hour.get('predicted_output_kWh', 0) or hour.get('energy_output_kWh', 0)
        daily_data[date_key]['total_kwh'] += energy
        daily_data[date_key]['peak_kwh'] = max(daily_data[date_key]['peak_kwh'], energy)
        
        if 'temperature' in hour:
            daily_data[date_key]['avg_temp'].append(hour['temperature'])
        if 'solar_irradiance' in hour:
            daily_data[date_key]['avg_irradiance'].append(hour['solar_irradiance'])
    
    # Calculate averages
    result = []
    for date_key, data in sorted(daily_data.items()):
        result.append({
            'date': data['date'],
            'total_kwh': round(data['total_kwh'], 2),
            'peak_kwh': round(data['peak_kwh'], 2),
            'avg_temp': round(np.mean(data['avg_temp']), 1) if data['avg_temp'] else 0,
            'avg_irradiance': round(np.mean(data['avg_irradiance']), 1) if data['avg_irradiance'] else 0
        })
    
    return result


def calculate_performance_metrics(
    predicted: List[float],
    actual: List[float]
) -> Dict[str, float]:
    """
    Calculate model performance metrics
    
    Args:
        predicted: Predicted values
        actual: Actual values
        
    Returns:
        Dictionary with performance metrics
    """
    predicted = np.array(predicted)
    actual = np.array(actual)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predicted - actual))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    
    # R² Score
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    return {
        'mae': round(mae, 3),
        'rmse': round(rmse, 3),
        'mape': round(mape, 2),
        'r2_score': round(r2, 3),
        'accuracy': round(max(0, (1 - mape / 100) * 100), 1)
    }
