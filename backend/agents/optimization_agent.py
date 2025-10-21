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
    
    def _recommend_appliance_schedule(self, df: pd.DataFrame, energy_col: str) -> Dict:
        """Recommend optimal times to run high-energy appliances"""
        recommendations = {
            'high_energy_appliances': [],
            'medium_energy_appliances': [],
            'flexible_loads': []
        }
        
        # Define appliance categories with typical consumption
        appliances = {
            'high': [
                {'name': 'Electric Vehicle Charging', 'consumption_kwh': 7.0, 'duration_hours': 4},
                {'name': 'Water Heater', 'consumption_kwh': 4.0, 'duration_hours': 2},
                {'name': 'Clothes Dryer', 'consumption_kwh': 3.0, 'duration_hours': 1},
            ],
            'medium': [
                {'name': 'Dishwasher', 'consumption_kwh': 1.8, 'duration_hours': 2},
                {'name': 'Washing Machine', 'consumption_kwh': 1.5, 'duration_hours': 1},
                {'name': 'Pool Pump', 'consumption_kwh': 1.2, 'duration_hours': 3},
            ],
            'flexible': [
                {'name': 'Battery Charging (devices)', 'consumption_kwh': 0.5, 'duration_hours': 2},
                {'name': 'Vacuum Cleaner', 'consumption_kwh': 0.8, 'duration_hours': 1},
            ]
        }
        
        # Find best time windows for each appliance
        for category, items in appliances.items():
            for appliance in items:
                best_time = self._find_best_time_window(
                    df, energy_col, 
                    appliance['consumption_kwh'], 
                    appliance['duration_hours']
                )
                
                if best_time:
                    recommendation = {
                        'appliance': appliance['name'],
                        'best_start_time': best_time['start_time'],
                        'expected_solar_coverage': best_time['coverage_percent'],
                        'grid_needed_kwh': best_time['grid_needed'],
                        'cost_savings': best_time['savings']
                    }
                    
                    if category == 'high':
                        recommendations['high_energy_appliances'].append(recommendation)
                    elif category == 'medium':
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
        best_coverage = 0
        
        # Slide window through forecast
        for i in range(len(df) - duration_hours + 1):
            window = df.iloc[i:i+duration_hours]
            total_solar = window[energy_col].sum()
            
            # Calculate how much of the consumption can be covered by solar
            coverage = min(total_solar, consumption_kwh)
            coverage_percent = (coverage / consumption_kwh) * 100
            
            if coverage_percent > best_coverage:
                best_coverage = coverage_percent
                grid_needed = max(0, consumption_kwh - total_solar)
                
                best_window = {
                    'start_time': pd.to_datetime(window.iloc[0]['timestamp']).strftime('%I:%M %p'),
                    'coverage_percent': round(coverage_percent, 1),
                    'grid_needed': round(grid_needed, 2),
                    'savings': round((coverage / consumption_kwh) * consumption_kwh * self.electricity_tariff, 2)
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
        avg_consumption_per_hour = 0.8 * self.system_size if self.system_size > 0 else 1.2
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
        
        # Savings from not buying from grid
        grid_cost_avoided = total_solar * self.electricity_tariff
        
        # Potential revenue from export (assume 30% can be exported)
        exportable = total_solar * 0.3
        export_revenue = exportable * self.feed_in_tariff
        
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
        
        # Equivalents for context
        trees_equivalent = co2_avoided_kg / 21  # 1 tree absorbs ~21kg CO2/year
        car_miles_equivalent = co2_avoided_kg / 0.404  # 1 mile = ~0.404kg CO2
        
        return {
            'co2_avoided_kg': round(co2_avoided_kg, 2),
            'co2_avoided_tons': round(co2_avoided_tons, 4),
            'trees_equivalent': round(trees_equivalent, 2),
            'car_miles_avoided': round(car_miles_equivalent, 1),
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
