"""
Real Weather-Based Solar Forecaster
Uses actual weather data from OpenWeather API + physics calculations
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import requests
import os
import json

logger = logging.getLogger(__name__)


class RealWeatherSolarForecaster:
    """
    Solar forecaster using real weather data
    - Fetches actual weather forecast from OpenWeather
    - Uses physics for solar calculations
    - Accounts for panel orientation (tilt, azimuth)
    - Considers historical patterns
    """
    
    def __init__(self, 
                 system_size_kwp: float = 5.0,
                 efficiency: float = 0.15,
                 panel_tilt: float = 30.0,
                 panel_azimuth: float = 180.0,
                 performance_ratio: float = 0.78):
        """
        Args:
            system_size_kwp: Solar system size in kWp
            efficiency: Panel efficiency (0.15 = 15%)
            panel_tilt: Panel tilt angle in degrees (0=flat, 90=vertical)
            panel_azimuth: Panel direction in degrees (0=North, 90=East, 180=South, 270=West)
        """
        self.system_size = system_size_kwp
        self.efficiency = efficiency
        self.panel_tilt = panel_tilt
        self.panel_azimuth = panel_azimuth
        self.performance_ratio = performance_ratio
        
        # OpenWeather API key
        self.api_key = os.getenv('OPENWEATHER_API_KEY', '9c6f96c360d63c44167435fce9f3a0e6')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # NASA POWER API integration (no API key required)
        self.nasa_power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.solar_cache = {}  # Cache NASA POWER data per location
        
    def fetch_weather_forecast(self, lat: float, lon: float) -> pd.DataFrame:
        """Fetch real weather forecast from OpenWeather API"""
        try:
            logger.info(f"Fetching real weather forecast for ({lat}, {lon})...")
            
            # Get 5-day forecast (3-hour intervals)
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecast_data = []
            for item in data['list']:
                # Extract detailed parameters based on OpenWeather API and parameters.csv
                main = item.get('main', {})
                wind = item.get('wind', {})
                clouds = item.get('clouds', {})
                rain = item.get('rain', {})
                snow = item.get('snow', {})
                sys = item.get('sys', {})
                weather = item.get('weather', [{}])[0]
                
                forecast_data.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': main.get('temp'),
                    'temp_min': main.get('temp_min'),
                    'temp_max': main.get('temp_max'),
                    'feels_like': main.get('feels_like'),
                    'pressure': main.get('pressure'),
                    'sea_level': main.get('sea_level'),
                    'grnd_level': main.get('grnd_level'),
                    'humidity': main.get('humidity'),
                    'clouds': clouds.get('all'),
                    'wind_speed': wind.get('speed'),
                    'wind_deg': wind.get('deg'),
                    'wind_gust': wind.get('gust', 0), # gust is optional
                    'visibility': item.get('visibility'),
                    'pop': item.get('pop', 0), # Probability of precipitation
                    'rain_3h': rain.get('3h', 0),
                    'snow_3h': snow.get('3h', 0),
                    'weather': weather.get('main'),
                    'weather_desc': weather.get('description'),
                    'weather_icon': weather.get('icon'),
                    'pod': sys.get('pod')
                })
            
            df = pd.DataFrame(forecast_data)
            logger.info(f"✓ Fetched {len(df)} forecast points from OpenWeather")
            
            # Interpolate to hourly
            df = self._interpolate_to_hourly(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            # Fallback to synthetic data
            logger.warning("Falling back to synthetic weather data")
            return self._generate_synthetic_weather(168, lat, lon)
    
    def _interpolate_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate 3-hour data to hourly with proper NaN handling"""
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create hourly range
        start = df['timestamp'].min().replace(minute=0, second=0, microsecond=0)
        end = df['timestamp'].max()
        hourly_timestamps = pd.date_range(start=start, end=end, freq='h')
        
        # Create new dataframe with hourly timestamps
        hourly_df = pd.DataFrame({'timestamp': hourly_timestamps})
        
        # Merge with original data
        df_merged = pd.merge(hourly_df, df, on='timestamp', how='left')
        
        # Interpolate numeric columns
        numeric_cols = [
            'temperature', 'temp_min', 'temp_max', 'feels_like', 
            'pressure', 'sea_level', 'grnd_level', 
            'humidity', 'clouds', 
            'wind_speed', 'wind_deg', 'wind_gust', 
            'visibility', 'pop', 'rain_3h', 'snow_3h'
        ]
        
        for col in numeric_cols:
            if col in df_merged.columns:
                # Linear interpolation
                df_merged[col] = df_merged[col].interpolate(method='linear', limit_direction='both')
                # Fill any remaining NaNs
                df_merged[col] = df_merged[col].ffill().bfill()
                # If still NaN, use reasonable default
                if df_merged[col].isna().any():
                    defaults = {
                        'temperature': 25.0,
                        'temp_min': 20.0,
                        'temp_max': 30.0,
                        'feels_like': 25.0,
                        'pressure': 1013.0,
                        'sea_level': 1013.0,
                        'grnd_level': 1013.0,
                        'humidity': 60.0,
                        'clouds': 30.0,
                        'wind_speed': 3.0,
                        'wind_deg': 0.0,
                        'wind_gust': 0.0,
                        'visibility': 10000.0,
                        'pop': 0.0,
                        'rain_3h': 0.0,
                        'snow_3h': 0.0
                    }
                    df_merged[col] = df_merged[col].fillna(defaults.get(col, 0))
        
        # Forward fill non-numeric columns
        non_numeric_cols = ['weather', 'weather_desc', 'weather_icon', 'pod']
        for col in non_numeric_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].ffill().bfill()
                # Default if still NaN
                if col == 'weather':
                    df_merged[col] = df_merged[col].fillna('Clear')
                else:
                    df_merged[col] = df_merged[col].fillna('')
        
        logger.info(f"✓ Interpolated to {len(df_merged)} hourly points")
        
        # Verify no NaN values remain
        nan_count = df_merged.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Still have {nan_count} NaN values after interpolation, filling with zeros")
            df_merged = df_merged.fillna(0)
        
        return df_merged
    
    def _generate_synthetic_weather(self, hours: int, lat: float, lon: float) -> pd.DataFrame:
        """
        Generate location-aware synthetic weather when API fails or for extension.
        Uses latitude-based climatology for realistic patterns.
        """
        current_time = datetime.now()
        forecast_data = []
        
        # Determine climate zone from latitude (affects base temperature and seasonal swing)
        abs_lat = abs(lat)
        if abs_lat < 15:
            # Tropical: Hot, less seasonal variation
            base_temp = 28
            seasonal_swing = 4
            base_humidity = 75
            base_clouds = 40
        elif abs_lat < 35:
            # Subtropical: Warm, moderate seasonal variation
            base_temp = 22
            seasonal_swing = 8
            base_humidity = 60
            base_clouds = 35
        elif abs_lat < 55:
            # Temperate: Moderate, strong seasonal variation
            base_temp = 15
            seasonal_swing = 12
            base_humidity = 65
            base_clouds = 50
        else:
            # Cold/Polar: Cold, extreme seasonal variation
            base_temp = 5
            seasonal_swing = 15
            base_humidity = 70
            base_clouds = 60
        
        # Hemisphere adjustment (southern hemisphere has opposite seasons)
        hemisphere_factor = 1 if lat >= 0 else -1
        
        for h in range(hours):
            future_time = current_time + timedelta(hours=h)
            hour = future_time.hour
            day_of_year = future_time.timetuple().tm_yday
            
            # Seasonal temperature variation (peaks in summer)
            seasonal_offset = seasonal_swing * np.sin(2 * np.pi * (day_of_year - 172) / 365 * hemisphere_factor)
            
            # Daily temperature variation (peaks at 2-3 PM)
            daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            temperature = base_temp + seasonal_offset + daily_variation + np.random.normal(0, 1)
            
            # Humidity (inverse of temperature, with some persistence)
            humidity = base_humidity - (temperature - base_temp) * 1.2 + np.random.normal(0, 3)
            humidity = np.clip(humidity, 25, 98)
            
            # Wind speed (with persistence for realism)
            if h == 0:
                wind_speed = np.random.normal(4, 1.5)
            else:
                prev_wind = forecast_data[-1]['wind_speed']
                wind_speed = prev_wind * 0.7 + np.random.normal(4, 1.5) * 0.3
            wind_speed = np.clip(wind_speed, 0.5, 15)
            
            # Cloud cover (multi-day weather patterns with persistence)
            if h == 0:
                clouds = base_clouds + np.random.normal(0, 15)
            else:
                # Slow random walk for realistic multi-day patterns
                prev_clouds = forecast_data[-1]['clouds']
                clouds = prev_clouds * 0.95 + np.random.normal(base_clouds, 10) * 0.05
            clouds = np.clip(clouds, 0, 100)
            
            forecast_data.append({
                'timestamp': future_time,
                'temperature': round(temperature, 1),
                'temp_min': round(temperature - 2, 1),
                'temp_max': round(temperature + 2, 1),
                'feels_like': round(temperature, 1),
                'pressure': 1013,
                'sea_level': 1013,
                'grnd_level': 1013,
                'humidity': round(humidity, 1),
                'clouds': round(clouds, 1),
                'wind_speed': round(wind_speed, 1),
                'wind_deg': np.random.randint(0, 360),
                'wind_gust': round(wind_speed * 1.5, 1),
                'visibility': 10000,
                'pop': 0.0,
                'rain_3h': 0.0,
                'snow_3h': 0.0,
                'weather': 'Clear' if clouds < 30 else ('Clouds' if clouds < 70 else 'Overcast'),
                'weather_desc': 'clear sky' if clouds < 30 else ('scattered clouds' if clouds < 70 else 'overcast clouds'),
                'weather_icon': '01d' if clouds < 30 else ('03d' if clouds < 70 else '04d')
            })
        
        logger.info(f"Generated {hours}h synthetic weather for ({lat}, {lon}): {base_temp}°C base, {seasonal_swing}° seasonal")
        return pd.DataFrame(forecast_data)

    def _extend_weather_forecast(self, real_df: pd.DataFrame, hours_needed: int, 
                                  lat: float, lon: float) -> pd.DataFrame:
        """
        Extend real weather data with climatology-based synthetic data.
        Uses the last real data point to seed the extension for smooth transition.
        """
        if len(real_df) >= hours_needed:
            return real_df.head(hours_needed)
        
        hours_missing = hours_needed - len(real_df)
        logger.info(f"Extending {len(real_df)}h real data with {hours_missing}h climatology-based forecast")
        
        # Get last real data point to seed extension
        last_real = real_df.iloc[-1]
        last_timestamp = pd.to_datetime(last_real['timestamp'])
        
        # Generate extension starting from next hour
        extension_data = []
        
        # Climate zone parameters (same as synthetic)
        abs_lat = abs(lat)
        if abs_lat < 15:
            base_temp, seasonal_swing, base_humidity, base_clouds = 28, 4, 75, 40
        elif abs_lat < 35:
            base_temp, seasonal_swing, base_humidity, base_clouds = 22, 8, 60, 35
        elif abs_lat < 55:
            base_temp, seasonal_swing, base_humidity, base_clouds = 15, 12, 65, 50
        else:
            base_temp, seasonal_swing, base_humidity, base_clouds = 5, 15, 70, 60
        
        hemisphere_factor = 1 if lat >= 0 else -1
        
        # Initialize with last real values
        prev_wind = last_real.get('wind_speed', 4)
        prev_clouds = last_real.get('clouds', base_clouds)
        
        for h in range(1, hours_missing + 1):
            future_time = last_timestamp + timedelta(hours=h)
            hour = future_time.hour
            day_of_year = future_time.timetuple().tm_yday
            
            # Temperature
            seasonal_offset = seasonal_swing * np.sin(2 * np.pi * (day_of_year - 172) / 365 * hemisphere_factor)
            daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            temperature = base_temp + seasonal_offset + daily_variation + np.random.normal(0, 1)
            
            # Humidity
            humidity = base_humidity - (temperature - base_temp) * 1.2 + np.random.normal(0, 3)
            humidity = np.clip(humidity, 25, 98)
            
            # Wind (persistence from previous)
            wind_speed = prev_wind * 0.7 + np.random.normal(4, 1.5) * 0.3
            wind_speed = np.clip(wind_speed, 0.5, 15)
            prev_wind = wind_speed
            
            # Clouds (persistence from previous)
            clouds = prev_clouds * 0.95 + np.random.normal(base_clouds, 10) * 0.05
            clouds = np.clip(clouds, 0, 100)
            prev_clouds = clouds
            
            extension_data.append({
                'timestamp': future_time,
                'temperature': round(temperature, 1),
                'temp_min': round(temperature - 2, 1),
                'temp_max': round(temperature + 2, 1),
                'feels_like': round(temperature, 1),
                'pressure': 1013,
                'sea_level': 1013,
                'grnd_level': 1013,
                'humidity': round(humidity, 1),
                'clouds': round(clouds, 1),
                'wind_speed': round(wind_speed, 1),
                'wind_deg': np.random.randint(0, 360),
                'wind_gust': round(wind_speed * 1.5, 1),
                'visibility': 10000,
                'pop': 0.0,
                'rain_3h': 0.0,
                'snow_3h': 0.0,
                'weather': 'Clear' if clouds < 30 else ('Clouds' if clouds < 70 else 'Overcast'),
                'weather_desc': 'clear sky' if clouds < 30 else ('scattered clouds' if clouds < 70 else 'overcast clouds'),
                'weather_icon': '01d' if clouds < 30 else ('03d' if clouds < 70 else '04d')
            })
        
        extension_df = pd.DataFrame(extension_data)
        combined_df = pd.concat([real_df, extension_df], ignore_index=True)
        
        logger.info(f"✓ Extended forecast to {len(combined_df)} hours total")
        return combined_df

    
    def calculate_solar_position(self, timestamp: datetime, lat: float, lon: float) -> Dict:
        """Calculate sun's position in the sky using centralized math"""
        from utils.solar_math import calculate_solar_position
        elevation, azimuth, declination = calculate_solar_position(timestamp, lat, lon)
        return {
            'elevation': elevation,
            'azimuth': azimuth,
            'declination': declination
        }

    def fetch_nasa_power_solar_data(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Fetch solar irradiance data from NASA POWER API
        Returns average GHI (Global Horizontal Irradiance) in W/m²
        Global coverage, no API key required
        """
        # Check cache first
        cache_key = f"{lat:.4f},{lon:.4f}"
        if cache_key in self.solar_cache:
            return self.solar_cache[cache_key]
        
        try:
            # NASA POWER API - get last 30 days of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            params = {
                "parameters": "ALLSKY_SFC_SW_DWN",
                "community": "RE",
                "longitude": lon,
                "latitude": lat,
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "format": "JSON"
            }
            
            logger.info(f"Fetching NASA POWER solar data for ({lat}, {lon})...")
            response = requests.get(self.nasa_power_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if "properties" in data and "parameter" in data["properties"]:
                parameter_data = data["properties"]["parameter"]
                ghi_data = parameter_data.get("ALLSKY_SFC_SW_DWN", {})
                
                if not ghi_data:
                    logger.warning("No GHI data in NASA POWER response")
                    return None
                
                # Calculate average GHI (kWh/m²/day)
                ghi_values = [float(v) for v in ghi_data.values() if v != -999]
                if not ghi_values:
                    logger.warning("No valid GHI values in NASA POWER data")
                    return None
                
                avg_ghi_kwh_m2_day = np.mean(ghi_values)
                
                # Convert kWh/m²/day to W/m²
                avg_ghi_w_m2 = (avg_ghi_kwh_m2_day * 1000) / 24.0
                peak_ghi_w_m2 = avg_ghi_w_m2 * 2.0
                
                result = {
                    "daily_avg_kwh_m2_day": avg_ghi_kwh_m2_day,
                    "effective_kwh_m2_day": avg_ghi_kwh_m2_day,
                    "avg_ghi_w_m2": avg_ghi_w_m2,
                    "peak_ghi_w_m2": peak_ghi_w_m2,
                    "source": "NASA POWER"
                }
                
                # Cache the result
                self.solar_cache[cache_key] = result
                
                logger.info(f"✓ NASA POWER data: {avg_ghi_kwh_m2_day:.2f} kWh/m²/day → Peak: {peak_ghi_w_m2:.0f} W/m²")
                return result
            else:
                logger.warning("Invalid NASA POWER response structure")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NASA POWER data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching NASA POWER data: {e}")
            return None
    
    def calculate_angle_of_incidence(self, sun_elevation: float, sun_azimuth: float) -> float:
        """Calculate angle between sun rays and panel surface using centralized math"""
        from utils.solar_math import calculate_angle_of_incidence
        return calculate_angle_of_incidence(
            sun_elevation, sun_azimuth, self.panel_tilt, self.panel_azimuth
        )
    
    def calculate_solar_irradiance(self, timestamp: datetime, clouds: float, 
                                   lat: float, lon: float, nasa_data: Optional[Dict] = None) -> Dict:
        """
        Calculate solar irradiance with panel orientation
        Uses centralized math for consistency
        """
        from utils.solar_math import (
            calculate_solar_position, 
            calculate_angle_of_incidence,
            calculate_clear_sky_irradiance,
            calculate_effective_irradiance
        )
        
        # Get sun position
        elevation, azimuth, _ = calculate_solar_position(timestamp, lat, lon)
        
        # If sun below horizon, no irradiance
        if elevation <= 0:
            return {
                'irradiance': 0.0,
                'direct': 0.0,
                'diffuse': 0.0,
                'sun_elevation': elevation,
                'sun_azimuth': azimuth,
                'angle_of_incidence': 90.0
            }
        
        # Determine base clear sky GHI
        # Always use physical clear sky formulation since NASA data contains average cloud attenuation
        base_clear_sky_ghi = calculate_clear_sky_irradiance(elevation)
        
        # Calculate angle of incidence
        aoi = calculate_angle_of_incidence(elevation, azimuth, self.panel_tilt, self.panel_azimuth)
        
        # Calculate effective tilted panel irradiance with clouds
        irradiance_data = calculate_effective_irradiance(base_clear_sky_ghi, clouds, elevation, aoi)
        
        return {
            'irradiance': irradiance_data['total'],
            'direct': irradiance_data['direct'],
            'diffuse': irradiance_data['diffuse'],
            'sun_elevation': elevation,
            'sun_azimuth': azimuth,
            'angle_of_incidence': aoi
        }
    
    def calculate_energy_output(self, irradiance: float, temperature: float) -> float:
        """Calculate energy output with temperature derating using centralized math"""
        from utils.solar_math import calculate_energy_output
        return calculate_energy_output(irradiance, temperature, self.system_size, self.performance_ratio)
    
    def forecast(self, lat: float, lon: float, hours: int = 168) -> Dict:
        """
        Generate complete forecast using real weather data
        
        Args:
            lat: Latitude
            lon: Longitude
            hours: Hours to forecast (default 168 = 7 days)
        
        Returns:
            Dictionary with predictions and metadata
        """
        logger.info(f"Generating forecast for ({lat}, {lon})")
        logger.info(f"  System: {self.system_size} kWp, PR {self.performance_ratio*100:.0f}%")
        logger.info(f"  Panel: {self.panel_tilt}° tilt, {self.panel_azimuth}° azimuth")
        
        # Fetch NASA POWER solar data for this location (cached)
        nasa_data = self.fetch_nasa_power_solar_data(lat, lon)
        if nasa_data:
            eff = float(nasa_data.get('effective_kwh_m2_day', 0))
            logger.info(f"  Using NASA POWER solar data: {eff:.2f} kWh/m²/day")
        else:
            logger.warning("  NASA POWER data unavailable, using physics-only calculations")
        
        # Fetch real weather forecast
        weather_df = self.fetch_weather_forecast(lat, lon)
        
        # Extend to requested hours if needed (OpenWeather only gives ~5 days)
        weather_df = self._extend_weather_forecast(weather_df, hours, lat, lon)

        
        # Calculate solar irradiance and energy for each hour
        predictions = []
        for idx, row in weather_df.iterrows():
            timestamp = row['timestamp']
            temperature = row['temperature']
            clouds = row['clouds']
            
            # Skip if any critical value is NaN
            if pd.isna(temperature) or pd.isna(clouds):
                logger.warning(f"Skipping row {idx} due to NaN values")
                continue
            
            # Calculate irradiance with panel orientation (pass NASA POWER data)
            solar_data = self.calculate_solar_irradiance(timestamp, clouds, lat, lon, nasa_data)
            
            # Calculate energy
            energy = self.calculate_energy_output(solar_data['irradiance'], temperature)
            
            predictions.append({
                'timestamp': timestamp,
                'temperature': temperature,
                'temp_min': row.get('temp_min', temperature),
                'temp_max': row.get('temp_max', temperature),
                'feels_like': row.get('feels_like', temperature),
                'pressure': row.get('pressure', 1013),
                'sea_level': row.get('sea_level', 1013),
                'grnd_level': row.get('grnd_level', 1013),
                'humidity': row['humidity'],
                'clouds': clouds,
                'wind_speed': row['wind_speed'],
                'wind_deg': row.get('wind_deg', 0),
                'wind_gust': row.get('wind_gust', 0),
                'visibility': row.get('visibility', 10000),
                'pop': row.get('pop', 0),
                'rain_3h': row.get('rain_3h', 0),
                'snow_3h': row.get('snow_3h', 0),
                'weather': row.get('weather', 'Unknown'),
                'weather_desc': row.get('weather_desc', ''),
                'weather_icon': row.get('weather_icon', ''),
                'solar_irradiance': solar_data['irradiance'],
                'direct_irradiance': solar_data['direct'],
                'diffuse_irradiance': solar_data['diffuse'],
                'sun_elevation': solar_data['sun_elevation'],
                'sun_azimuth': solar_data['sun_azimuth'],
                'angle_of_incidence': solar_data['angle_of_incidence'],
                'predicted_output_kWh': energy,
                'confidence_lower': energy * 0.85,
                'confidence_upper': energy * 1.15
            })
        
        predictions_df = pd.DataFrame(predictions)
        
        # Check if we have any predictions
        if len(predictions_df) == 0:
            logger.error("No valid predictions generated - all rows had NaN values")
            logger.warning("Falling back to synthetic weather data")
            # Retry with synthetic data
            weather_df = self._generate_synthetic_weather(hours, lat, lon)
            predictions = []
            for idx, row in weather_df.iterrows():
                timestamp = row['timestamp']
                temperature = row['temperature']
                clouds = row['clouds']
                
                solar_data = self.calculate_solar_irradiance(timestamp, clouds, lat, lon)
                energy = self.calculate_energy_output(solar_data['irradiance'], temperature)
                
                predictions.append({
                    'timestamp': timestamp,
                    'temperature': temperature,
                    'temp_min': row.get('temp_min', temperature),
                    'temp_max': row.get('temp_max', temperature),
                    'feels_like': row.get('feels_like', temperature),
                    'pressure': row.get('pressure', 1013),
                    'sea_level': row.get('sea_level', 1013),
                    'grnd_level': row.get('grnd_level', 1013),
                    'humidity': row['humidity'],
                    'clouds': clouds,
                    'wind_speed': row['wind_speed'],
                    'wind_deg': row.get('wind_deg', 0),
                    'wind_gust': row.get('wind_gust', 0),
                    'visibility': row.get('visibility', 10000),
                    'pop': row.get('pop', 0),
                    'rain_3h': row.get('rain_3h', 0),
                    'snow_3h': row.get('snow_3h', 0),
                    'weather': row.get('weather', 'Unknown'),
                    'weather_desc': row.get('weather_desc', ''),
                    'weather_icon': row.get('weather_icon', ''),
                    'solar_irradiance': solar_data['irradiance'],
                    'direct_irradiance': solar_data['direct'],
                    'diffuse_irradiance': solar_data['diffuse'],
                    'sun_elevation': solar_data['sun_elevation'],
                    'sun_azimuth': solar_data['sun_azimuth'],
                    'angle_of_incidence': solar_data['angle_of_incidence'],
                    'predicted_output_kWh': energy,
                    'confidence_lower': energy * 0.85,
                    'confidence_upper': energy * 1.15
                })
            predictions_df = pd.DataFrame(predictions)
        
        # Log statistics
        logger.info(f"✓ Forecast complete: {len(predictions_df)} hours")
        if len(predictions_df) > 0:
            logger.info(f"  Energy range: {predictions_df['predicted_output_kWh'].min():.2f} - {predictions_df['predicted_output_kWh'].max():.2f} kWh")
            logger.info(f"  Total energy: {predictions_df['predicted_output_kWh'].sum():.1f} kWh")
        
        # Create multi-horizon views
        hourly_24h = predictions_df.head(24).to_dict(orient='records')
        
        # Daily aggregation (7 days)
        daily_data = []
        for day in range(min(7, len(predictions_df) // 24)):
            day_start = day * 24
            day_end = min((day + 1) * 24, len(predictions_df))
            day_df = predictions_df.iloc[day_start:day_end]
            
            if len(day_df) > 0:
                daily_data.append({
                    'date': day_df.iloc[0]['timestamp'].date().isoformat(),
                    'total_kwh': float(day_df['predicted_output_kWh'].sum()),
                    'avg_kwh': float(day_df['predicted_output_kWh'].mean()),
                    'min_kwh': float(day_df['predicted_output_kWh'].min()),
                    'max_kwh': float(day_df['predicted_output_kWh'].max()),
                    'avg_temp': float(day_df['temperature'].mean()),
                    'avg_solar': float(day_df['solar_irradiance'].mean()),
                    'avg_wind': float(day_df['wind_speed'].mean()),
                    'avg_clouds': float(day_df['clouds'].mean())
                })
        
        # Find peak production hour
        peak_hour = predictions_df.loc[predictions_df['predicted_output_kWh'].idxmax()]
        
        return {
            'status': 'success',
            'hourly_24h': hourly_24h,
            'daily_7d': daily_data,
            'metrics': {
                'total_24h': float(sum(p['predicted_output_kWh'] for p in hourly_24h)),
                'avg_24h': float(np.mean([p['predicted_output_kWh'] for p in hourly_24h])),
                'peak_24h': float(max(p['predicted_output_kWh'] for p in hourly_24h)),
                'total_week': float(predictions_df['predicted_output_kWh'].sum()) if len(predictions_df) >= 168 else 0
            },
            'insights': {
                'summary': f"Real weather forecast for {lat:.2f}, {lon:.2f}",
                'peak_hour': peak_hour['timestamp'].strftime('%I:%M %p'),
                'peak_energy': float(peak_hour['predicted_output_kWh']),
                'total_today': float(sum(p['predicted_output_kWh'] for p in hourly_24h)),
                'panel_orientation': f"{self.panel_tilt}° tilt, {self.panel_azimuth}° azimuth",
                'weather_source': 'OpenWeather API'
            }
        }


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    forecaster = RealWeatherSolarForecaster(
        system_size_kwp=5.0,
        efficiency=0.15,
        panel_tilt=30.0,
        panel_azimuth=180.0  # South-facing
    )
    
    result = forecaster.forecast(lat=28.6139, lon=77.2090, hours=168)
    
    print(f"\n✓ Status: {result['status']}")
    print(f"✓ Hourly predictions: {len(result['hourly_24h'])}")
    print(f"✓ Daily predictions: {len(result['daily_7d'])}")
    print(f"✓ Today's total: {result['metrics']['total_24h']:.2f} kWh")
    print(f"✓ Peak hour: {result['insights']['peak_hour']} ({result['insights']['peak_energy']:.2f} kWh)")
    print(f"✓ Panel: {result['insights']['panel_orientation']}")
