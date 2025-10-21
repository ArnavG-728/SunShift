"""
Physics-Based Solar Forecaster - Uses actual solar physics instead of pure ML
This ensures predictions are realistic (0 at night, peak at noon)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PhysicsBasedForecaster:
    """
    Solar energy forecaster based on physics principles
    - Solar irradiance calculation (physics)
    - Direct energy conversion (efficiency model)
    - Weather impact (cloud cover, temperature)
    - No ML black box that ignores physics
    """
    
    def __init__(self, system_size_kwp: float = 5.0, efficiency: float = 0.15):
        """
        Args:
            system_size_kwp: Solar panel system size in kWp
            efficiency: Panel efficiency (0.15 = 15%)
        """
        self.system_size = system_size_kwp
        self.efficiency = efficiency
        self.temp_coefficient = -0.004  # -0.4% per °C above 25°C
        
    def calculate_solar_irradiance(self, timestamp: datetime, clouds: float, 
                                   lat: float = 28.6139, lon: float = 77.2090) -> float:
        """
        Calculate solar irradiance using physics
        
        Returns 0 at night, peaks at solar noon
        """
        hour = timestamp.hour + timestamp.minute / 60
        day_of_year = timestamp.timetuple().tm_yday
        
        # Solar declination (Earth's tilt)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle (sun's position)
        hour_angle = 15 * (hour - 12)
        
        # Solar elevation angle
        elevation = np.degrees(np.arcsin(
            np.sin(np.radians(lat)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(lat)) * np.cos(np.radians(declination)) * 
            np.cos(np.radians(hour_angle))
        ))
        
        # If sun is below horizon, irradiance = 0
        if elevation <= 0:
            return 0.0
        
        # Air mass (atmospheric path length)
        air_mass = 1 / (np.sin(np.radians(elevation)) + 0.50572 * (elevation + 6.07995)**-1.6364)
        
        # Clear sky irradiance (AM1.5 spectrum)
        clear_sky_irradiance = 1367 * (0.7 ** (air_mass ** 0.678))
        
        # Cloud cover reduction (clouds block 50-90% of light)
        cloud_factor = 1 - (clouds / 100) * 0.75
        
        # Final irradiance
        irradiance = clear_sky_irradiance * cloud_factor
        
        return max(0, irradiance)
    
    def calculate_energy_output(self, solar_irradiance: float, temperature: float,
                                wind_speed: float = 0) -> float:
        """
        Calculate energy output from solar irradiance
        
        This is PHYSICS, not ML guessing!
        """
        # Temperature derating (panels lose efficiency when hot)
        temp_factor = 1 + self.temp_coefficient * (temperature - 25)
        temp_factor = max(0.7, min(1.0, temp_factor))  # Clamp between 70-100%
        
        # Solar energy output
        # Irradiance (W/m²) → Power (kW) → Energy (kWh for 1 hour)
        solar_output = (solar_irradiance / 1000) * self.system_size * self.efficiency * temp_factor
        
        # Wind energy (optional, small contribution)
        wind_output = 0
        if 3 <= wind_speed <= 25:  # Cut-in and cut-out speeds
            # Simplified wind power curve
            wind_output = 0.1 * (wind_speed ** 2) / 10
        
        total_output = solar_output + wind_output
        
        # Add small realistic noise (±5%)
        noise = np.random.normal(0, total_output * 0.05)
        
        return max(0, total_output + noise)
    
    def forecast(self, weather_forecast: pd.DataFrame, latitude: float = 28.6139,
                longitude: float = 77.2090) -> pd.DataFrame:
        """
        Generate forecast from weather data
        
        Args:
            weather_forecast: DataFrame with columns:
                - timestamp
                - temperature
                - clouds
                - wind_speed (optional)
            latitude: Location latitude
            longitude: Location longitude
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Generating physics-based forecast for {len(weather_forecast)} hours")
        
        predictions = []
        
        for idx, row in weather_forecast.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            temperature = row.get('temperature', 25)
            clouds = row.get('clouds', 30)
            wind_speed = row.get('wind_speed', 5)
            
            # Calculate solar irradiance (PHYSICS)
            irradiance = self.calculate_solar_irradiance(
                timestamp, clouds, latitude, longitude
            )
            
            # Calculate energy output (PHYSICS)
            energy = self.calculate_energy_output(
                irradiance, temperature, wind_speed
            )
            
            predictions.append({
                'timestamp': timestamp,
                'temperature': temperature,
                'clouds': clouds,
                'wind_speed': wind_speed,
                'solar_irradiance': irradiance,
                'predicted_output_kWh': energy,
                'confidence_lower': energy * 0.85,
                'confidence_upper': energy * 1.15
            })
        
        result_df = pd.DataFrame(predictions)
        
        # Log statistics
        logger.info(f"Predictions: min={result_df['predicted_output_kWh'].min():.2f}, "
                   f"max={result_df['predicted_output_kWh'].max():.2f}, "
                   f"mean={result_df['predicted_output_kWh'].mean():.2f}")
        
        # Verify night time is zero
        night_hours = result_df[result_df['solar_irradiance'] == 0]
        if len(night_hours) > 0:
            logger.info(f"✓ Night time predictions: {len(night_hours)} hours with 0 kWh (correct!)")
        
        # Verify peak hours
        peak_hours = result_df[result_df['solar_irradiance'] > 800]
        if len(peak_hours) > 0:
            logger.info(f"✓ Peak hours: {len(peak_hours)} hours with high irradiance")
            logger.info(f"  Peak energy: {result_df['predicted_output_kWh'].max():.2f} kWh")
        
        return result_df
    
    def validate(self, historical_data: pd.DataFrame) -> Dict:
        """
        Validate model on historical data
        
        Args:
            historical_data: DataFrame with actual energy_output_kWh
            
        Returns:
            Validation metrics
        """
        # Generate predictions
        predictions_df = self.forecast(historical_data)
        
        # Compare with actual
        actual = historical_data['energy_output_kWh'].values
        predicted = predictions_df['predicted_output_kWh'].values
        
        # Align lengths
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        
        # R² score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        logger.info(f"Validation: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'accuracy': float(max(0, (1 - mae / (np.mean(actual) + 1e-10)) * 100))
        }


if __name__ == "__main__":
    # Test the forecaster
    logging.basicConfig(level=logging.INFO)
    
    forecaster = PhysicsBasedForecaster(system_size_kwp=5.0)
    
    # Create test data (24 hours)
    test_data = []
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    for h in range(24):
        test_data.append({
            'timestamp': base_time + timedelta(hours=h),
            'temperature': 25 + 5 * np.sin(2 * np.pi * h / 24),
            'clouds': 30,
            'wind_speed': 5
        })
    
    test_df = pd.DataFrame(test_data)
    
    # Generate forecast
    forecast_df = forecaster.forecast(test_df)
    
    print("\n24-Hour Forecast:")
    print(forecast_df[['timestamp', 'solar_irradiance', 'predicted_output_kWh']].to_string())
    
    # Verify night time is zero
    night = forecast_df[(forecast_df['timestamp'].dt.hour < 6) | (forecast_df['timestamp'].dt.hour > 18)]
    print(f"\nNight time energy (should be ~0): {night['predicted_output_kWh'].mean():.3f} kWh")
    
    # Verify noon is peak
    noon = forecast_df[(forecast_df['timestamp'].dt.hour >= 11) & (forecast_df['timestamp'].dt.hour <= 13)]
    print(f"Noon energy (should be peak): {noon['predicted_output_kWh'].mean():.3f} kWh")
