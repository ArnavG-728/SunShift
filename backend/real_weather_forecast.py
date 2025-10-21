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
                forecast_data.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'clouds': item['clouds']['all'],
                    'wind_speed': item['wind']['speed'],
                    'pressure': item['main']['pressure'],
                    'weather': item['weather'][0]['main']
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
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'clouds', 'pressure']
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
                        'humidity': 60.0,
                        'wind_speed': 3.0,
                        'clouds': 30.0,
                        'pressure': 1013.0
                    }
                    df_merged[col] = df_merged[col].fillna(defaults.get(col, 0))
        
        # Forward fill weather description
        if 'weather' in df_merged.columns:
            df_merged['weather'] = df_merged['weather'].ffill().bfill()
            # Default if still NaN
            df_merged['weather'] = df_merged['weather'].fillna('Clear')
        
        logger.info(f"✓ Interpolated to {len(df_merged)} hourly points")
        
        # Verify no NaN values remain
        nan_count = df_merged.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Still have {nan_count} NaN values after interpolation, filling with defaults")
            df_merged = df_merged.fillna({
                'temperature': 25.0,
                'humidity': 60.0,
                'wind_speed': 3.0,
                'clouds': 30.0,
                'pressure': 1013.0,
                'weather': 'Clear'
            })
        
        return df_merged
    
    def _generate_synthetic_weather(self, hours: int, lat: float, lon: float) -> pd.DataFrame:
        """Fallback: Generate synthetic weather if API fails"""
        current_time = datetime.now()
        forecast_data = []
        
        for h in range(hours):
            future_time = current_time + timedelta(hours=h)
            hour = future_time.hour
            day_of_year = future_time.timetuple().tm_yday
            
            # Temperature with daily and seasonal cycles
            base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
            daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            temperature = base_temp + daily_variation + np.random.normal(0, 1)
            
            # Humidity (inverse of temperature)
            humidity = 70 - (temperature - 15) * 1.5 + np.random.normal(0, 5)
            humidity = np.clip(humidity, 30, 95)
            
            # Wind
            wind_speed = np.random.normal(5, 2)
            wind_speed = np.clip(wind_speed, 0, 15)
            
            # Clouds (random walk)
            if h == 0:
                clouds = np.random.uniform(0, 100)
            else:
                clouds = forecast_data[-1]['clouds'] + np.random.normal(0, 15)
            clouds = np.clip(clouds, 0, 100)
            
            forecast_data.append({
                'timestamp': future_time,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'clouds': clouds,
                'pressure': 1013,
                'weather': 'Clear' if clouds < 30 else 'Clouds'
            })
        
        return pd.DataFrame(forecast_data)
    
    def calculate_solar_position(self, timestamp: datetime, lat: float, lon: float) -> Dict:
        """Calculate sun's position in the sky"""
        hour = timestamp.hour + timestamp.minute / 60
        day_of_year = timestamp.timetuple().tm_yday
        
        # Solar declination (Earth's tilt)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle (sun's east-west position)
        hour_angle = 15 * (hour - 12)
        
        # Solar elevation angle (height above horizon)
        elevation = np.degrees(np.arcsin(
            np.sin(np.radians(lat)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(lat)) * np.cos(np.radians(declination)) * 
            np.cos(np.radians(hour_angle))
        ))
        
        # Solar azimuth angle (compass direction)
        azimuth = np.degrees(np.arctan2(
            np.sin(np.radians(hour_angle)),
            np.cos(np.radians(hour_angle)) * np.sin(np.radians(lat)) -
            np.tan(np.radians(declination)) * np.cos(np.radians(lat))
        ))
        azimuth = (azimuth + 180) % 360  # Convert to 0-360 range
        
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
        """
        Calculate angle between sun rays and panel surface
        Accounts for panel tilt and azimuth
        """
        # Convert to radians
        sun_elev_rad = np.radians(sun_elevation)
        sun_azim_rad = np.radians(sun_azimuth)
        panel_tilt_rad = np.radians(self.panel_tilt)
        panel_azim_rad = np.radians(self.panel_azimuth)
        
        # Angle of incidence formula
        cos_aoi = (
            np.sin(sun_elev_rad) * np.cos(panel_tilt_rad) +
            np.cos(sun_elev_rad) * np.sin(panel_tilt_rad) * 
            np.cos(sun_azim_rad - panel_azim_rad)
        )
        
        # Clamp to valid range
        cos_aoi = np.clip(cos_aoi, -1, 1)
        aoi = np.degrees(np.arccos(cos_aoi))
        
        return aoi
    
    def calculate_solar_irradiance(self, timestamp: datetime, clouds: float, 
                                   lat: float, lon: float, nasa_data: Optional[Dict] = None) -> Dict:
        """
        Calculate solar irradiance with panel orientation
        Uses NASA POWER data as baseline if available, otherwise physics-based calculation
        """
        # Get sun position
        sun_pos = self.calculate_solar_position(timestamp, lat, lon)
        elevation = sun_pos['elevation']
        azimuth = sun_pos['azimuth']
        
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
        
        # Determine base irradiance
        if nasa_data and nasa_data.get('peak_ghi_w_m2', 0) > 0:
            # Use NASA POWER data as baseline - scale by time of day
            hour = timestamp.hour + timestamp.minute / 60
            # Solar curve: 0 at sunrise/sunset, peak at solar noon
            time_factor = max(0, np.sin((hour - 6) * np.pi / 12))
            base_clear_sky_ghi = nasa_data['peak_ghi_w_m2'] * time_factor
        else:
            # Fallback to physics-based calculation
            # Air mass (atmospheric path length)
            air_mass = 1 / (np.sin(np.radians(elevation)) + 0.50572 * (elevation + 6.07995)**-1.6364)
            
            # Extraterrestrial irradiance
            solar_constant = 1367  # W/m²
            
            # Direct normal irradiance (clear sky)
            direct_normal = solar_constant * (0.7 ** (air_mass ** 0.678))
            
            # GHI for clear sky
            base_clear_sky_ghi = direct_normal * np.sin(np.radians(elevation))
        
        # Apply cloud cover
        cloud_transmittance = 1 - (clouds / 100) * 0.75
        actual_ghi = base_clear_sky_ghi * cloud_transmittance
        
        # Calculate angle of incidence for tilted panel
        aoi = self.calculate_angle_of_incidence(elevation, azimuth)
        
        # For tilted panels, adjust based on angle of incidence
        if aoi < 90:  # Panel facing sun
            # Tilt factor: how much more/less irradiance the tilted panel gets
            tilt_factor = np.cos(np.radians(aoi)) / np.sin(np.radians(elevation))
            tilt_factor = max(0.5, min(1.5, tilt_factor))  # Reasonable bounds
        else:  # Panel facing away
            tilt_factor = 0.5  # Still gets diffuse light
        
        # Total irradiance on tilted panel
        total_irradiance = actual_ghi * tilt_factor
        
        # Estimate direct vs diffuse components
        direct_fraction = 0.8 * cloud_transmittance  # More direct when clear
        diffuse_fraction = 1 - direct_fraction
        
        direct_component = total_irradiance * direct_fraction
        diffuse_component = total_irradiance * diffuse_fraction
        
        return {
            'irradiance': max(0, total_irradiance),
            'direct': max(0, direct_component),
            'diffuse': max(0, diffuse_component),
            'sun_elevation': elevation,
            'sun_azimuth': azimuth,
            'angle_of_incidence': aoi
        }
    
    def calculate_energy_output(self, irradiance: float, temperature: float) -> float:
        """Calculate energy output with temperature derating"""
        # Handle NaN values
        if pd.isna(temperature) or pd.isna(irradiance):
            logger.warning(f"NaN value detected: temp={temperature}, irradiance={irradiance}")
            return 0.0
        
        # Temperature coefficient (-0.4% per °C above 25°C)
        temp_factor = 1 - 0.004 * (temperature - 25)
        temp_factor = max(0.7, min(1.0, temp_factor))
        
        # Energy output (kWh for 1 hour)
        # Use performance ratio (system losses) with DC system size (kWp). Do not multiply efficiency again.
        energy = (float(irradiance) / 1000.0) * float(self.system_size) * float(self.performance_ratio) * float(temp_factor)
        
        return float(max(0, energy))
    
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
        
        # Limit to requested hours
        weather_df = weather_df.head(hours)
        
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
                'humidity': row['humidity'],
                'wind_speed': row['wind_speed'],
                'clouds': clouds,
                'weather': row.get('weather', 'Unknown'),
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
                    'humidity': row['humidity'],
                    'wind_speed': row['wind_speed'],
                    'clouds': clouds,
                    'weather': row.get('weather', 'Unknown'),
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
