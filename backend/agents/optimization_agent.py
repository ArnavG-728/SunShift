"""
Smart Energy Optimization Agent
Provides AI-driven recommendations for energy usage, battery management, and grid optimization
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class OptimizationAgent:
    """
    AI agent for smart energy optimization
    - Analyzes solar forecast to recommend optimal appliance usage times
    - Provides battery charging/discharging recommendations
    - Suggests grid import/export strategies
    - Calculates carbon impact and cost savings
    """
    
    def __init__(self, 
                 battery_capacity_kwh: float = 0.0,
                 max_grid_import_kw: float = 10.0,
                 electricity_tariff: float = 0.12,
                 feed_in_tariff: float = 0.08,
                 system_size_kwp: float = 5.0,
                 grid_co2_factor: float = 0.70):
        """
        Args:
            battery_capacity_kwh: Battery storage capacity (0 = no battery)
            max_grid_import_kw: Maximum grid import power
            electricity_tariff: Cost per kWh from grid ($/kWh)
            feed_in_tariff: Payment per kWh exported to grid ($/kWh)
            system_size_kwp: Solar system size in kWp
            grid_co2_factor: Grid carbon intensity (kg CO2 per kWh)
        """
        self.battery_capacity = battery_capacity_kwh
        self.max_grid_import = max_grid_import_kw
        self.electricity_tariff = electricity_tariff
        self.feed_in_tariff = feed_in_tariff
        self.system_size = system_size_kwp
        self.grid_co2_factor = grid_co2_factor
        
    def analyze_forecast(self, hourly_forecast: List[Dict]) -> Dict:
        """
        Analyze solar forecast and generate optimization recommendations
        
        Args:
            hourly_forecast: List of hourly predictions with solar output
            
        Returns:
            Dictionary with recommendations and insights
        """
        if not hourly_forecast:
            return self._empty_recommendations()
        
        df = pd.DataFrame(hourly_forecast)
        
        # Filter to future-only hours so we never recommend past times
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        df = df[df['timestamp'] >= now].reset_index(drop=True)
        
        if len(df) == 0:
            logger.warning("No future hours in forecast data after filtering")
            return self._empty_recommendations()
        
        # Ensure we have the required column
        energy_col = 'predicted_output_kWh' if 'predicted_output_kWh' in df.columns else 'energy_output_kWh'
        
        if energy_col not in df.columns:
            logger.warning("No energy output column found in forecast")
            return self._empty_recommendations()
        
        # Find peak production periods
        peak_hours = self._find_peak_hours(df, energy_col)
        
        # Find low production periods
        low_hours = self._find_low_hours(df, energy_col)
        
        # Generate appliance scheduling recommendations
        appliance_schedule = self._recommend_appliance_schedule(df, energy_col)
        
        # Battery recommendations (if battery exists)
        battery_schedule = self._recommend_battery_schedule(df, energy_col) if self.battery_capacity > 0 else None
        
        # Grid optimization
        grid_strategy = self._recommend_grid_strategy(df, energy_col)
        
        # Calculate potential savings
        savings = self._calculate_savings(df, energy_col)
        
        # Carbon impact
        carbon_impact = self._calculate_carbon_impact(df, energy_col)
        
        # Energy alerts
        alerts = self._generate_alerts(df, energy_col)
        
        # Automation triggers
        automation_triggers = self._generate_automation_triggers(df, energy_col)
        
        return {
            'status': 'success',
            'peak_hours': peak_hours,
            'low_hours': low_hours,
            'appliance_schedule': appliance_schedule,
            'battery_schedule': battery_schedule,
            'grid_strategy': grid_strategy,
            'savings': savings,
            'carbon_impact': carbon_impact,
            'alerts': alerts,
            'automation_triggers': automation_triggers,
            'summary': self._generate_summary(df, energy_col, peak_hours, savings)
        }
    
    def _find_peak_hours(self, df: pd.DataFrame, energy_col: str) -> List[Dict]:
        """Find hours with highest solar production"""
        # Get top 5 peak hours
        top_hours = df.nlargest(5, energy_col)
        
        peak_hours = []
        for _, row in top_hours.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            peak_hours.append({
                'time': timestamp.strftime('%I:%M %p'),
                'date': timestamp.strftime('%Y-%m-%d'),
                'energy_kwh': float(row[energy_col]),
                'hour': timestamp.hour
            })
        
        return peak_hours
    
    def _find_low_hours(self, df: pd.DataFrame, energy_col: str) -> List[Dict]:
        """Find hours with lowest solar production (excluding nighttime)"""
        # Filter out nighttime (assume production < 0.1 kWh is night)
        daytime = df[df[energy_col] > 0.1]
        
        if len(daytime) == 0:
            return []
        
        # Get bottom 5 daytime hours
        low_hours = daytime.nsmallest(5, energy_col)
        
        result = []
        for _, row in low_hours.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            result.append({
                'time': timestamp.strftime('%I:%M %p'),
                'date': timestamp.strftime('%Y-%m-%d'),
                'energy_kwh': float(row[energy_col]),
                'hour': timestamp.hour
            })
        
        return result
    
    def _load_appliances(self) -> List[Dict]:
        """Load appliance configuration from JSON file (flat list)"""
        from config import config
        import json
        
        appliances_path = config.DATA_DIR / 'appliances.json'
        
        if appliances_path.exists():
            try:
                with open(appliances_path, 'r') as f:
                    data = json.load(f)
                    # Support legacy category-based format migration
                    if isinstance(data, dict):
                        flat_list = []
                        for category in data.values():
                            if isinstance(category, list):
                                flat_list.extend(category)
                        return flat_list
                    return data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Error loading appliances config: {e}")
        
        # Fallback to defaults
        return [
            {'name': 'EV Charging', 'consumption_kwh': 7.0, 'duration_hours': 4},
            {'name': 'Water Heater', 'consumption_kwh': 4.0, 'duration_hours': 2},
            {'name': 'Clothes Dryer', 'consumption_kwh': 3.0, 'duration_hours': 1},
            {'name': 'Dishwasher', 'consumption_kwh': 1.8, 'duration_hours': 2},
            {'name': 'Washing Machine', 'consumption_kwh': 1.5, 'duration_hours': 1},
            {'name': 'Pool Pump', 'consumption_kwh': 1.2, 'duration_hours': 3},
            {'name': 'Device Charging', 'consumption_kwh': 0.5, 'duration_hours': 2},
            {'name': 'Vacuum Cleaner', 'consumption_kwh': 0.8, 'duration_hours': 1}
        ]

    def _recommend_appliance_schedule(self, df: pd.DataFrame, energy_col: str) -> Dict:
        """Recommend optimal times to run high-energy appliances with auto-classification"""
        recommendations = {
            'high_energy_appliances': [],
            'medium_energy_appliances': [],
            'flexible_loads': []
        }
        
        # Load flat list of appliances
        all_appliances = self._load_appliances()
        
        # Automatic Classification based on consumption_kwh
        # High: > 2.5 kWh
        # Medium: 1.0 - 2.5 kWh
        # Flexible: < 1.0 kWh
        
        for appliance in all_appliances:
            consumption = appliance.get('consumption_kwh', 1.0)
            
            best_time = self._find_best_time_window(
                df, energy_col, 
                consumption, 
                appliance.get('duration_hours', 1)
            )
            
            if best_time:
                recommendation = {
                    'appliance': appliance['name'],
                    'best_start_time': best_time['start_time'],
                    'expected_solar_coverage': best_time['coverage_percent'],
                    'grid_needed': best_time['grid_needed'],
                    'cost_savings': best_time['savings']
                }
                
                if consumption > 2.5:
                    recommendations['high_energy_appliances'].append(recommendation)
                elif consumption >= 1.0:
                    recommendations['medium_energy_appliances'].append(recommendation)
                else:
                    recommendations['flexible_loads'].append(recommendation)
        
        return recommendations
    
    def _find_best_time_window(self, df: pd.DataFrame, energy_col: str, 
                                consumption_kwh: float, duration_hours: int) -> Optional[Dict]:
        """Find best time window for running an appliance"""
        if len(df) < duration_hours:
            return None
        
        best_window = None
        best_coverage = -1
        max_excess_solar = -float('inf')
        
        # Power drawn per hour (assuming constant draw)
        hourly_consumption = consumption_kwh / duration_hours
        
        # Slide window through forecast
        for i in range(len(df) - duration_hours + 1):
            window = df.iloc[i:i+duration_hours]
            
            # Calculate coverage on an hourly basis
            hourly_coverage = [min(row[energy_col], hourly_consumption) for _, row in window.iterrows()]
            total_coverage = sum(hourly_coverage)
            coverage_percent = (total_coverage / consumption_kwh) * 100
            
            # Calculate total excess solar in this window (to center the load during peak sun)
            total_solar = window[energy_col].sum()
            excess_solar = total_solar - consumption_kwh
            
            # We prefer the window that gives the highest coverage.
            # If multiple windows give the same (e.g., 100%) coverage, we pick the one with the MOST excess solar.
            # This ensures we schedule exactly during the peak of the day rather than at the edges.
            if coverage_percent > best_coverage or (abs(coverage_percent - best_coverage) < 0.1 and excess_solar > max_excess_solar):
                best_coverage = coverage_percent
                max_excess_solar = excess_solar
                grid_needed = max(0, consumption_kwh - total_coverage)
                
                best_window = {
                    'start_time': pd.to_datetime(window.iloc[0]['timestamp']).strftime('%I:%M %p'),
                    'coverage_percent': round(coverage_percent, 1),
                    'grid_needed': round(grid_needed, 2),
                    'savings': round(total_coverage * self.electricity_tariff, 2)
                }
        
        return best_window
    
    def _recommend_battery_schedule(self, df: pd.DataFrame, energy_col: str) -> Dict:
        """Recommend battery charging/discharging schedule"""
        schedule = []
        
        # Simple strategy: charge during peak solar, discharge during low solar
        for _, row in df.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            energy = row[energy_col]
            
            # Charge if production > 50% of max
            max_production = df[energy_col].max()
            
            if energy > max_production * 0.5:
                action = 'charge'
                priority = 'high' if energy > max_production * 0.8 else 'medium'
            elif energy < max_production * 0.2:
                action = 'discharge'
                priority = 'high'
            else:
                action = 'hold'
                priority = 'low'
            
            schedule.append({
                'time': timestamp.strftime('%I:%M %p'),
                'action': action,
                'priority': priority,
                'solar_kwh': round(energy, 2)
            })
        
        return {
            'schedule': schedule[:24],  # Next 24 hours
            'strategy': 'Peak shaving and solar maximization',
            'estimated_cycles': self._estimate_battery_cycles(schedule[:24])
        }
    
    def _estimate_battery_cycles(self, schedule: List[Dict]) -> float:
        """Estimate battery charge/discharge cycles"""
        cycles = 0
        last_action = None
        
        for item in schedule:
            if item['action'] != last_action and item['action'] in ['charge', 'discharge']:
                cycles += 0.5
            last_action = item['action']
        
        return round(cycles, 2)
    
    def _recommend_grid_strategy(self, df: pd.DataFrame, energy_col: str) -> Dict:
        """Recommend grid import/export strategy"""
        total_production = df[energy_col].sum()
        
        # Calculate average household consumption based on system size
        # Rule of thumb: residential consumption ≈ 0.8 * system_size per hour
        avg_consumption_per_hour = 1.2  # realistic average residential consumption (kW)
        total_consumption = avg_consumption_per_hour * len(df)
        
        surplus = total_production - total_consumption
        
        if surplus > 0:
            strategy = 'net_exporter'
            recommendation = f"Export {surplus:.1f} kWh to grid for ${surplus * self.feed_in_tariff:.2f} revenue"
        else:
            deficit = abs(surplus)
            strategy = 'net_importer'
            recommendation = f"Import {deficit:.1f} kWh from grid at ${deficit * self.electricity_tariff:.2f} cost"
        
        return {
            'strategy': strategy,
            'total_production_kwh': round(total_production, 2),
            'estimated_consumption_kwh': round(total_consumption, 2),
            'net_balance_kwh': round(surplus, 2),
            'recommendation': recommendation,
            'peak_export_hours': self._find_peak_hours(df, energy_col)[:3],
            'peak_import_hours': self._find_low_hours(df, energy_col)[:3]
        }
    
    def _calculate_savings(self, df: pd.DataFrame, energy_col: str) -> Dict:
        """Calculate potential cost savings"""
        total_solar = df[energy_col].sum()
        
        # Split solar into self-consumed vs exported
        avg_consumption_per_hour = 1.2  # typical residential kW
        total_consumption = avg_consumption_per_hour * len(df)
        self_consumed = min(total_solar, total_consumption)
        exported = max(0, total_solar - total_consumption)
        
        # Savings from self-consumption (avoided grid purchase)
        grid_cost_avoided = self_consumed * self.electricity_tariff
        # Revenue from exporting surplus
        export_revenue = exported * self.feed_in_tariff
        
        total_savings = grid_cost_avoided + export_revenue
        
        return {
            'total_solar_kwh': round(total_solar, 2),
            'grid_cost_avoided': round(grid_cost_avoided, 2),
            'export_revenue': round(export_revenue, 2),
            'total_savings': round(total_savings, 2),
            'daily_average': round(total_savings / max(1, len(df) / 24), 2),
            'monthly_projection': round(total_savings * 30 / max(1, len(df) / 24), 2)
        }
    
    def _calculate_carbon_impact(self, df: pd.DataFrame, energy_col: str) -> Dict:
        """Calculate carbon emissions avoided"""
        total_solar = df[energy_col].sum()
        
        # Use configured grid carbon intensity
        co2_avoided_kg = total_solar * self.grid_co2_factor
        co2_avoided_tons = co2_avoided_kg / 1000
        
        # Equivalents for context (consistent with GreenMetrics & CarbonWallet)
        trees_daily_absorption = 21 / 365  # ~0.0575 kg CO2 per tree per day
        trees_equivalent = co2_avoided_kg / trees_daily_absorption
        car_km_equivalent = co2_avoided_kg / 0.12  # 1 km = ~0.12 kg CO2
        
        return {
            'co2_avoided_kg': round(co2_avoided_kg, 2),
            'co2_avoided_tons': round(co2_avoided_tons, 4),
            'trees_equivalent': round(trees_equivalent, 2),
            'car_km_avoided': round(car_km_equivalent, 1),
            'monthly_projection_kg': round(co2_avoided_kg * 30 / max(1, len(df) / 24), 2)
        }
    
    def _generate_alerts(self, df: pd.DataFrame, energy_col: str) -> List[Dict]:
        """Generate actionable energy alerts"""
        alerts = []
        
        # Check for low production days
        if len(df) >= 24:
            tomorrow = df.iloc[:24]
            tomorrow_total = tomorrow[energy_col].sum()
            avg_production = df[energy_col].sum() / max(1, len(df) / 24)
            
            if tomorrow_total < avg_production * 0.6:
                alerts.append({
                    'type': 'warning',
                    'priority': 'high',
                    'title': 'Low Solar Production Tomorrow',
                    'message': f'Tomorrow\'s production ({tomorrow_total:.1f} kWh) is 40% below average. Consider charging batteries today.',
                    'action': 'Charge batteries and minimize grid usage'
                })
        
        # Check for peak production
        max_hour = df.loc[df[energy_col].idxmax()]
        max_time = pd.to_datetime(max_hour['timestamp'])
        
        if max_time.date() == datetime.now().date():
            alerts.append({
                'type': 'info',
                'priority': 'medium',
                'title': 'Peak Production Today',
                'message': f'Peak solar at {max_time.strftime("%I:%M %p")} ({max_hour[energy_col]:.2f} kWh). Run high-energy appliances then.',
                'action': 'Schedule dishwasher, laundry, or EV charging'
            })
        
        # Check for weather changes
        if 'clouds' in df.columns:
            current_clouds = df.iloc[0]['clouds']
            future_clouds = df.iloc[min(12, len(df)-1)]['clouds']
            
            if future_clouds - current_clouds > 30:
                alerts.append({
                    'type': 'warning',
                    'priority': 'medium',
                    'title': 'Weather Deteriorating',
                    'message': f'Cloud cover increasing from {current_clouds:.0f}% to {future_clouds:.0f}% in next 12 hours.',
                    'action': 'Use solar energy now before conditions worsen'
                })
        
        return alerts
    
    def _generate_summary(self, df: pd.DataFrame, energy_col: str, 
                         peak_hours: List[Dict], savings: Dict) -> str:
        """Generate human-readable summary"""
        total_energy = df[energy_col].sum()
        avg_energy = df[energy_col].mean()
        
        peak_time = peak_hours[0]['time'] if peak_hours else 'N/A'
        peak_energy = peak_hours[0]['energy_kwh'] if peak_hours else 0
        
        # Calculate carbon impact for summary
        co2_avoided = total_energy * self.grid_co2_factor
        trees_equivalent = co2_avoided / 21
        
        summary = f"""
📊 Energy Optimization Summary

🌞 Solar Production:
   • Total: {total_energy:.1f} kWh over {len(df)} hours
   • Average: {avg_energy:.2f} kWh per hour
   • Peak: {peak_energy:.2f} kWh at {peak_time}

💰 Financial Impact:
   • Potential savings: ${savings['total_savings']:.2f}
   • Monthly projection: ${savings['monthly_projection']:.2f}

🎯 Top Recommendation:
   • Run high-energy appliances during peak hours ({peak_time})
   • Charge batteries when production exceeds 50% of peak
   • Export surplus to grid during mid-day peak production

🌱 Environmental Impact:
   • CO₂ avoided: {co2_avoided:.1f} kg
   • Equivalent to planting {trees_equivalent:.1f} trees
        """.strip()
        
        return summary
    
    def _generate_automation_triggers(self, df: pd.DataFrame, energy_col: str) -> List[Dict]:
        """Generate structured JSON triggers for smart home automation"""
        triggers = []
        if len(df) == 0:
            return triggers
            
        # 1. Excess Solar Trigger
        current_energy = df.iloc[0][energy_col]
        if current_energy > 3.0:
            triggers.append({
                "id": "solar_excess_high",
                "action": "START_LOAD",
                "target": "EV_CHARGER",
                "condition": f"Production ({current_energy:.1f}kW) > 3.0kW",
                "priority": 1,
                "payload": {"current_limit_amps": 16}
            })
            
        # 2. Battery Low + Low Forecast Trigger
        tomorrow_total = df.iloc[:24][energy_col].sum() if len(df) >= 24 else 0
        if tomorrow_total < 5.0 and tomorrow_total > 0:
            triggers.append({
                "id": "grid_charge_reserve",
                "action": "CHARGE_FROM_GRID",
                "target": "STORAGE_BATTERY",
                "condition": f"Tomorrow forecast ({tomorrow_total:.1f}kWh) is low",
                "priority": 2,
                "payload": {"target_soc": 80, "max_rate_kw": 3.0}
            })
            
        # 3. Peak Export Opportunity
        max_idx = df[energy_col].idxmax()
        max_hour = df.loc[max_idx]
        if max_hour[energy_col] > 4.0:
            triggers.append({
                "id": "peak_export_optimize",
                "action": "MAXIMIZE_EXPORT",
                "target": "INVERTER",
                "condition": f"Peak production at {pd.to_datetime(max_hour['timestamp']).strftime('%H:%M')}",
                "priority": 3,
                "payload": {"export_limit_kw": 10.0}
            })
            
        return triggers
    
    def _empty_recommendations(self) -> Dict:
        """Return empty recommendations structure"""
        return {
            'status': 'no_data',
            'peak_hours': [],
            'low_hours': [],
            'appliance_schedule': {
                'high_energy_appliances': [],
                'medium_energy_appliances': [],
                'flexible_loads': []
            },
            'battery_schedule': None,
            'grid_strategy': {},
            'savings': {},
            'carbon_impact': {},
            'alerts': [],
            'summary': 'No forecast data available for optimization analysis.'
        }


if __name__ == "__main__":
    # Test with sample data
    logging.basicConfig(level=logging.INFO)
    
    # Create sample forecast
    sample_forecast = []
    for i in range(24):
        hour = datetime.now() + timedelta(hours=i)
        # Simulate solar curve
        if 6 <= hour.hour <= 18:
            energy = 0.5 * np.sin((hour.hour - 6) * np.pi / 12) * 5
        else:
            energy = 0
        
        sample_forecast.append({
            'timestamp': hour.isoformat(),
            'predicted_output_kWh': energy,
            'clouds': 30
        })
    
    optimizer = OptimizationAgent(
        battery_capacity_kwh=10.0,
        electricity_tariff=0.12,
        feed_in_tariff=0.08
    )
    
    result = optimizer.analyze_forecast(sample_forecast)
    print(result['summary'])
