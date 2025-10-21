"""
Data Collection Agent - Fetches and cleans weather and energy data
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
from pathlib import Path

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAgent:
    """Agent responsible for collecting and cleaning energy and weather data"""
    
    def __init__(self):
        self.api_key = config.OPENWEATHER_API_KEY
        self.base_url = config.OPENWEATHER_BASE_URL
        self.data_dir = config.DATA_DIR
        
    def fetch_weather_data(self, lat: float = 28.6139, lon: float = 77.2090, 
                          days: int = 30) -> pd.DataFrame:
        """
        Fetch REAL-TIME weather data from OpenWeather API
        ALWAYS attempts real data first, synthetic only as last resort
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days of historical data (default: 30 for better training)
            
        Returns:
            DataFrame with weather data
        """
        logger.info(f"🌍 Fetching REAL-TIME weather data for ({lat}, {lon})")
        
        # Validate API key
        if not self.api_key or self.api_key == "your_openweather_api_key_here":
            logger.error("❌ No valid API key! Please set OPENWEATHER_API_KEY in .env")
            logger.warning("Falling back to synthetic data...")
            return self._generate_synthetic_data(days)
        
        # ALWAYS try real data first (ignore USE_SYNTHETIC_DATA flag)
        try:
            logger.info(f"📡 Connecting to OpenWeather API...")
            
            # Import real-time agent
            from agents.realtime_data_agent import RealTimeDataAgent
            
            # Create real-time agent with specified coordinates
            rt_agent = RealTimeDataAgent(latitude=lat, longitude=lon)
            
            # Fetch historical data (includes current + forecast + interpolation)
            df = rt_agent.fetch_historical_data(days=days)
            
            if df is not None and len(df) > 0:
                logger.info(f"✅ SUCCESS! Fetched {len(df)} real-time data points")
                logger.info(f"📊 Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                logger.info(f"🌡️  Temp range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
                logger.info(f"☀️  Solar range: {df['solar_irradiance'].min():.0f} to {df['solar_irradiance'].max():.0f} W/m²")
                return df
            else:
                raise Exception("Real-time agent returned empty data")
            
        except Exception as e:
            logger.error(f"❌ Error fetching real-time data: {e}")
            logger.warning("⚠️  Falling back to synthetic data as last resort...")
            return self._generate_synthetic_data(days)
    
    def _generate_synthetic_data(self, days: int = 7) -> pd.DataFrame:
        """
        Generate synthetic renewable energy data for demonstration
        
        Args:
            days: Number of days to generate
            
        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Generating {days} days of synthetic data")
        
        # Generate timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Generate synthetic weather and energy data
        import numpy as np
        np.random.seed(42)
        
        n_samples = len(timestamps)
        
        # Create realistic patterns
        hours = timestamps.hour
        days_of_year = timestamps.dayofyear
        
        # Temperature (Celsius) - varies by hour and season
        temp_base = 15 + 10 * np.sin(2 * np.pi * days_of_year / 365)
        temp_daily = 5 * np.sin(2 * np.pi * hours / 24)
        temperature = temp_base + temp_daily + np.random.normal(0, 2, n_samples)
        
        # Humidity (%) - inverse correlation with temperature
        humidity = 70 - 0.5 * temperature + np.random.normal(0, 5, n_samples)
        humidity = np.clip(humidity, 20, 95)
        
        # Wind Speed (m/s) - random with some daily pattern
        wind_speed = 5 + 3 * np.sin(2 * np.pi * hours / 24) + np.random.exponential(2, n_samples)
        wind_speed = np.clip(wind_speed, 0, 20)
        
        # Solar Irradiance (W/m²) - strong daily pattern, zero at night
        solar_base = np.maximum(0, 800 * np.sin(np.pi * (hours - 6) / 12))
        cloud_factor = 1 - 0.3 * (humidity / 100)
        solar_irradiance = solar_base * cloud_factor + np.random.normal(0, 50, n_samples)
        solar_irradiance = np.clip(solar_irradiance, 0, 1000)
        
        # Energy Output (kWh) - combination of solar and wind
        solar_output = solar_irradiance * 0.15  # 15% efficiency
        wind_output = wind_speed ** 3 * 0.05  # Cubic relationship for wind
        energy_output = solar_output + wind_output + np.random.normal(0, 5, n_samples)
        energy_output = np.clip(energy_output, 0, None)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_irradiance': solar_irradiance,
            'energy_output_kWh': energy_output
        })
        
        return df
    
    def _process_real_weather_data(self, current_data: Dict, forecast_data: Dict, days: int) -> pd.DataFrame:
        """
        Process real weather data from OpenWeather API
        
        Args:
            current_data: Current weather data
            forecast_data: Forecast data
            days: Number of days to generate
            
        Returns:
            DataFrame with processed weather data
        """
        logger.info("Processing real weather data from OpenWeather API...")
        
        data_points = []
        
        # Process forecast data (5-day, 3-hour intervals)
        for item in forecast_data.get('list', []):
            timestamp = datetime.fromtimestamp(item['dt'])
            
            # Extract weather data
            temp = item['main']['temp']
            humidity = item['main']['humidity']
            wind_speed = item['wind']['speed']
            
            # Calculate solar irradiance from cloud coverage
            clouds = item['clouds']['all']  # Cloud coverage percentage
            # Estimate solar irradiance (0-1000 W/m²) based on clouds and time of day
            hour = timestamp.hour
            if 6 <= hour <= 18:  # Daytime
                base_irradiance = 800 * np.sin(np.pi * (hour - 6) / 12)
                solar_irradiance = base_irradiance * (1 - clouds / 100)
            else:
                solar_irradiance = 0
            
            # Estimate energy output based on weather conditions
            # Solar component
            solar_output = solar_irradiance * 0.15  # 15% efficiency
            
            # Wind component (cubic relationship with wind speed)
            wind_output = 0.5 * (wind_speed ** 2)
            
            # Total energy output
            energy_output = solar_output + wind_output
            
            data_points.append({
                'timestamp': timestamp,
                'temperature': temp,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'solar_irradiance': solar_irradiance,
                'energy_output_kWh': energy_output
            })
        
        # Create DataFrame
        df = pd.DataFrame(data_points)
        
        # If we need more historical data, supplement with synthetic data
        if len(df) < days * 24:
            logger.info(f"Supplementing with synthetic data to reach {days} days...")
            synthetic_df = self._generate_synthetic_data(days)
            # Take only the needed amount of synthetic data
            needed_rows = (days * 24) - len(df)
            df = pd.concat([synthetic_df.head(needed_rows), df], ignore_index=True)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Handle missing values
        df = df.ffill().bfill()
        
        # Remove outliers using IQR method
        for col in ['temperature', 'humidity', 'wind_speed', 'solar_irradiance', 'energy_output_kWh']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = "clean_energy_data.csv") -> Path:
        """
        Save cleaned data to CSV
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def run(self, days: int = 30, latitude: float = 28.6139, longitude: float = 77.2090) -> Dict:
        """
        Run data collection pipeline
        
        Args:
            days: Number of days of data to collect
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary with collected data
        """
        logger.info(f"DataAgent: Collecting {days} days of data for ({latitude}, {longitude})...")
        
        # Fetch weather data for specified location
        df = self.fetch_weather_data(lat=latitude, lon=longitude, days=days)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Save data
        filepath = self.save_data(df_clean)
        
        return {
            "status": "success",
            "message": f"Collected and cleaned {len(df_clean)} records",
            "data_path": str(filepath),
            "data": df_clean,
            "shape": df_clean.shape,
            "columns": list(df_clean.columns)
        }


if __name__ == "__main__":
    # Test the agent
    agent = DataAgent()
    result = agent.run(days=30)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Data shape: {result['shape']}")
