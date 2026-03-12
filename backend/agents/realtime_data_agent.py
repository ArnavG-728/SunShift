"""
Real-Time Data Agent - Core component for solar energy prediction
Fetches live weather data from OpenWeather API and calculates solar irradiance
This is the foundation for energy output predictions
"""
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time
from pathlib import Path
import json

from config import config

logger = logging.getLogger(__name__)


class RealTimeDataAgent:
    """
    Continuously fetch and process real-time data from multiple sources
    - OpenWeather API (current + forecast)
    - Historical data for training
    - Real-time telemetry simulation
    """
    
    def __init__(self, latitude: float = 28.6139, longitude: float = 77.2090):
        self.api_key = config.OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.nasa_power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.cache_dir = config.DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Location (can be configured)
        self.default_lat = latitude
        self.default_lon = longitude
        
        # NASA POWER data cache (location-based)
        self.solar_cache = {}
        
        logger.info(f"RealTimeDataAgent initialized for ({latitude}, {longitude})")
        logger.info(f"Using NASA POWER API for solar data (no API key required)")
    
    def fetch_current_weather(self, lat: float = None, lon: float = None) -> Dict:
        """
        Fetch current weather conditions
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Current weather data
        """
        # Use provided coordinates or fall back to default
        use_lat = lat if lat is not None else self.default_lat
        use_lon = lon if lon is not None else self.default_lon
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": use_lat,
                "lon": use_lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            logger.info(f"Fetching current weather for ({use_lat}, {use_lon})...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response
            weather = data['weather'][0]
            main = data['main']
            wind = data['wind']
            clouds = data['clouds']
            
            # Get timezone offset from API (in seconds)
            timezone_offset = data.get('timezone', 0)
            
            # Convert UTC timestamp to location's local time
            utc_time = datetime.utcfromtimestamp(data['dt'])
            local_time = utc_time + timedelta(seconds=timezone_offset)
            
            current = {
                'timestamp': local_time,
                'temperature': main['temp'],
                'humidity': main['humidity'],
                'pressure': main.get('pressure', 1013),
                'wind_speed': wind.get('speed', 0),
                'wind_direction': wind.get('deg', 0),
                'clouds': clouds.get('all', 0),
                'weather': weather['main'],
                'description': weather['description'],
                'timezone_offset': timezone_offset,
                "visibility": data.get("visibility", 10000),
                "sunrise": datetime.utcfromtimestamp(data["sys"]["sunrise"]) + timedelta(seconds=timezone_offset),
                "sunset": datetime.utcfromtimestamp(data["sys"]["sunset"]) + timedelta(seconds=timezone_offset)
            }
            
            logger.info(f"✓ Current weather: {current['temperature']}°C, {current['clouds']}% clouds")
            return current
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching current weather: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def fetch_forecast(self, lat: float = None, lon: float = None, hours: int = 48) -> List[Dict]:
        """
        Fetch weather forecast (5-day, 3-hour intervals)
        
        Args:
            lat: Latitude
            lon: Longitude
            hours: Number of hours to fetch
            
        Returns:
            List of forecast data points
        """
        lat = lat or self.default_lat
        lon = lon or self.default_lon
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            logger.info(f"Fetching forecast for ({lat}, {lon})...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Get timezone offset
            tz_offset_sec = data.get("city", {}).get("timezone", 0)
            
            # Process forecast data
            forecast = []
            for item in data["list"][:hours//3]:  # 3-hour intervals
                utc_time = datetime.utcfromtimestamp(item["dt"])
                local_time = utc_time + timedelta(seconds=tz_offset_sec)
                point = {
                    "timestamp": local_time,
                    "utc_time": utc_time.isoformat(),
                    "timezone_offset": tz_offset_sec,
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "pressure": item["main"]["pressure"],
                    "wind_speed": item["wind"]["speed"],
                    "wind_direction": item["wind"].get("deg", 0),
                    "clouds": item["clouds"]["all"],
                    "weather": item["weather"][0]["main"],
                    "description": item["weather"][0]["description"],
                    "pop": item.get("pop", 0) * 100  # Probability of precipitation
                }
                forecast.append(point)
            
            logger.info(f"✓ Fetched {len(forecast)} forecast points")
            return forecast
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching forecast: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []
    
    def fetch_nasa_power_solar_data(self, lat: float = None, lon: float = None, timestamp: datetime = None) -> Dict:
        """
        Fetch solar irradiance data from NASA POWER API
        Global coverage, no API key required
        
        Args:
            lat: Latitude
            lon: Longitude
            timestamp: Timestamp for time-of-day calculation (defaults to now)
            
        Returns:
            Dictionary with solar irradiance data (GHI)
        """
        use_lat = lat if lat is not None else self.default_lat
        use_lon = lon if lon is not None else self.default_lon
        use_timestamp = timestamp if timestamp is not None else datetime.now()
        
        # Check cache first
        cache_key = f"{use_lat:.4f},{use_lon:.4f}"
        if cache_key in self.solar_cache:
            cached_data = self.solar_cache[cache_key]
            # Recalculate time-of-day factor with provided timestamp
            hour = use_timestamp.hour + use_timestamp.minute / 60
            time_factor = max(0, np.sin((hour - 6) * np.pi / 12))
            current_ghi = cached_data['peak_ghi_w_m2'] * time_factor
            
            return {
                "ghi": float(current_ghi),
                "daily_avg_kwh_m2_day": cached_data['daily_avg_kwh_m2_day'],
                "peak_ghi_w_m2": cached_data['peak_ghi_w_m2'],
                "source": "NASA POWER (cached)"
            }
        
        try:
            # NASA POWER API - get last 30 days of data to calculate average
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            params = {
                "parameters": "ALLSKY_SFC_SW_DWN",  # All Sky Surface Shortwave Downward Irradiance
                "community": "RE",  # Renewable Energy
                "longitude": use_lon,
                "latitude": use_lat,
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "format": "JSON"
            }
            
            logger.info(f"Fetching NASA POWER solar data for ({use_lat}, {use_lon})...")
            response = requests.get(self.nasa_power_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if "properties" in data and "parameter" in data["properties"]:
                parameter_data = data["properties"]["parameter"]
                ghi_data = parameter_data.get("ALLSKY_SFC_SW_DWN", {})
                
                if not ghi_data:
                    logger.warning("No GHI data in NASA POWER response")
                    return None
                
                # Calculate average GHI from the data (kWh/m²/day)
                ghi_values = [float(v) for v in ghi_data.values() if v != -999]
                if not ghi_values:
                    logger.warning("No valid GHI values in NASA POWER data")
                    return None
                
                avg_ghi_kwh_m2_day = np.mean(ghi_values)
                
                # Convert kWh/m²/day to W/m² (peak during daylight)
                avg_ghi_w_m2 = (avg_ghi_kwh_m2_day * 1000) / 24.0
                peak_ghi_w_m2 = avg_ghi_w_m2 * 2.0  # Peak is roughly 2x the 24-hour average
                
                # Cache the base data
                self.solar_cache[cache_key] = {
                    "daily_avg_kwh_m2_day": float(avg_ghi_kwh_m2_day),
                    "peak_ghi_w_m2": float(peak_ghi_w_m2)
                }
                
                # Adjust for time of day
                hour = use_timestamp.hour + use_timestamp.minute / 60
                time_factor = max(0, np.sin((hour - 6) * np.pi / 12))
                current_ghi = peak_ghi_w_m2 * time_factor
                
                logger.info(f"✓ NASA POWER Solar GHI: {current_ghi:.2f} W/m² (daily avg: {avg_ghi_kwh_m2_day:.2f} kWh/m²/day)")
                
                return {
                    "ghi": float(current_ghi),
                    "daily_avg_kwh_m2_day": float(avg_ghi_kwh_m2_day),
                    "peak_ghi_w_m2": float(peak_ghi_w_m2),
                    "source": "NASA POWER"
                }
            else:
                logger.warning("Invalid NASA POWER response structure")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NASA POWER solar data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching NASA POWER data: {e}")
            return None
    
    def calculate_solar_irradiance(self, timestamp: datetime, clouds: float, 
                                   lat: float = None, lon: float = None,
                                   system_size: float = 5.0,
                                   panel_tilt: float = 30.0,
                                   panel_azimuth: float = 180.0) -> float:
        """
        Calculate configured solar irradiance using centralized physics math
        Falls back to physics-based calculation if NREL unavailable
        
        Args:
            timestamp: Current time
            clouds: Cloud coverage percentage (0-100)
            lat: Latitude (optional, uses default if not provided)
            lon: Longitude (optional, uses default if not provided)
            system_size: Size in kWp
            panel_tilt: Degrees tilt
            panel_azimuth: Degrees azimuth
            
        Returns:
            Solar irradiance in W/m² hitting the panel
        """
        use_lat = lat if lat is not None else self.default_lat
        use_lon = lon if lon is not None else self.default_lon
        
        from utils.solar_math import (
            calculate_solar_position, 
            calculate_angle_of_incidence,
            calculate_clear_sky_irradiance,
            calculate_effective_irradiance
        )
        
        # Get sun position
        elevation, azimuth, _ = calculate_solar_position(timestamp, use_lat, use_lon)
        
        # If sun below horizon, no irradiance
        if elevation <= 0:
            return 0.0
            
        # Try NASA POWER data first
        nasa_data = self.fetch_nasa_power_solar_data(use_lat, use_lon, timestamp)
        
        # Always use physical clear sky formulation since NASA data contains average cloud attenuation
        base_ghi = calculate_clear_sky_irradiance(elevation)
            
        # Calculate angle of incidence
        aoi = calculate_angle_of_incidence(elevation, azimuth, panel_tilt, panel_azimuth)
        
        # Calculate effective tilted panel irradiance with clouds
        irradiance_data = calculate_effective_irradiance(base_ghi, clouds, elevation, aoi)
        
        return irradiance_data['total']

    def calculate_cloud_loss(self, timestamp: datetime, clouds: float, 
                             lat: float = None, lon: float = None, 
                             system_size_kwp: float = 5.0, 
                             performance_ratio: float = 0.85,
                             panel_tilt: float = 30.0,
                             panel_azimuth: float = 180.0) -> Dict:
        """
        Calculate energy lost due to cloud cover using accurate tracking
        """
        from utils.solar_math import calculate_solar_position, calculate_clear_sky_irradiance
        use_lat = lat if lat is not None else self.default_lat
        use_lon = lon if lon is not None else self.default_lon
        
        # 1. Clear Sky Potential (Clouds = 0)
        elevation, _, _ = calculate_solar_position(timestamp, use_lat, use_lon)
        
        if elevation <= 0:
            return {"potential_kwh": 0, "actual_kwh": 0, "loss_kwh": 0, "loss_percent": 0}
            
        clear_sky_irradiance = calculate_clear_sky_irradiance(elevation)
        
        # 2. Actual Irradiance
        actual_irradiance = self.calculate_solar_irradiance(
            timestamp, clouds, use_lat, use_lon, system_size_kwp, panel_tilt, panel_azimuth
        )
        
        # 3. Convert to energy
        potential_kwh = (clear_sky_irradiance / 1000.0) * system_size_kwp * performance_ratio
        actual_kwh = (actual_irradiance / 1000.0) * system_size_kwp * performance_ratio
        
        loss_kwh = max(0, potential_kwh - actual_kwh)
        loss_percent = (loss_kwh / potential_kwh * 100) if potential_kwh > 0 else 0
        
        return {
            "potential_kwh": float(potential_kwh),
            "actual_kwh": float(actual_kwh),
            "loss_kwh": float(loss_kwh),
            "loss_percent": float(loss_percent)
        }

    def estimate_energy_output(self, weather_data: Dict, system_size_kwp: float = 5.0, 
                               performance_ratio: float = 0.85, panel_tilt: float = 30.0, 
                               panel_azimuth: float = 180.0) -> float:
        """
        Estimate energy output from weather conditions using centralized match
        """
        irradiance = self.calculate_solar_irradiance(
            weather_data["timestamp"],
            weather_data["clouds"],
            system_size=system_size_kwp,
            panel_tilt=panel_tilt,
            panel_azimuth=panel_azimuth
        )
        
        from utils.solar_math import calculate_energy_output
        return calculate_energy_output(
            irradiance=irradiance, 
            temperature=weather_data.get("temperature", 25.0),
            system_size_kwp=system_size_kwp,
            performance_ratio=performance_ratio
        )
    
    def fetch_historical_data(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch historical weather data for training
        Uses current + forecast data, then fills gaps with realistic interpolation
        
        Args:
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical data
        """
        logger.info(f"Fetching historical data for {days} days...")
        
        # Fetch current weather
        current = self.fetch_current_weather()
        
        # Fetch forecast
        forecast = self.fetch_forecast(hours=120)  # 5 days
        
        if not current or not forecast:
            logger.error("Failed to fetch real-time data")
            return None
        
        # Create historical data by working backwards
        data_points = []
        
        # Add current data
        current_point = {
            "timestamp": current["timestamp"],
            "temperature": current["temperature"],
            "humidity": current["humidity"],
            "wind_speed": current["wind_speed"],
            "solar_irradiance": self.calculate_solar_irradiance(
                current["timestamp"], current["clouds"]
            ),
            "energy_output_kWh": self.estimate_energy_output(current)
        }
        data_points.append(current_point)
        
        # Add forecast data
        for f in forecast:
            point = {
                "timestamp": f["timestamp"],
                "temperature": f["temperature"],
                "humidity": f["humidity"],
                "wind_speed": f["wind_speed"],
                "solar_irradiance": self.calculate_solar_irradiance(
                    f["timestamp"], f["clouds"]
                ),
                "energy_output_kWh": self.estimate_energy_output(f)
            }
            data_points.append(point)
        
        # Create DataFrame
        df = pd.DataFrame(data_points)
        
        # Generate additional historical data by interpolation
        if len(df) < days * 24:
            logger.info("Generating additional historical data through interpolation...")
            df = self._generate_historical_interpolation(df, days)
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"✓ Historical data ready: {len(df)} samples")
        return df
    
    def _generate_historical_interpolation(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """
        Generate historical data by intelligent interpolation
        Uses seasonal patterns and realistic variations
        """
        # Calculate how many hours we need
        needed_hours = days * 24
        current_hours = len(df)
        
        if current_hours >= needed_hours:
            return df
        
        # Get the earliest timestamp
        earliest = df["timestamp"].min()
        
        # Generate timestamps going backwards
        historical_timestamps = [
            earliest - timedelta(hours=i) 
            for i in range(1, needed_hours - current_hours + 1)
        ]
        historical_timestamps.reverse()
        
        # Generate realistic historical data
        historical_data = []
        for ts in historical_timestamps:
            # Use seasonal patterns
            hour = ts.hour
            day_of_year = ts.timetuple().tm_yday
            
            # Temperature (seasonal + daily cycle)
            base_temp = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365)
            daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            temperature = base_temp + daily_variation + np.random.normal(0, 2)
            
            # Humidity (inverse of temperature)
            humidity = 70 - (temperature - 20) * 1.5 + np.random.normal(0, 10)
            humidity = np.clip(humidity, 30, 95)
            
            # Wind speed (random with some persistence)
            wind_speed = 5 + 3 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
            wind_speed = max(0, wind_speed)
            
            # Cloud cover (random)
            clouds = np.random.uniform(0, 100)
            
            # Solar irradiance
            solar_irradiance = self.calculate_solar_irradiance(ts, clouds)
            
            # Energy output
            energy_output = self.estimate_energy_output({
                "timestamp": ts,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "clouds": clouds
            })
            
            historical_data.append({
                "timestamp": ts,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "solar_irradiance": solar_irradiance,
                "energy_output_kWh": energy_output
            })
        
        # Combine historical and real data
        historical_df = pd.DataFrame(historical_data)
        combined_df = pd.concat([historical_df, df], ignore_index=True)
        
        return combined_df
    
    def stream_realtime_data(self, interval_seconds: int = 300) -> Dict:
        """
        Stream real-time data continuously
        
        Args:
            interval_seconds: Update interval (default: 5 minutes)
            
        Yields:
            Real-time data points
        """
        logger.info(f"Starting real-time data stream (interval: {interval_seconds}s)")
        
        while True:
            try:
                # Fetch current weather
                current = self.fetch_current_weather()
                
                if current:
                    # Calculate solar irradiance
                    current["solar_irradiance"] = self.calculate_solar_irradiance(
                        current["timestamp"], current["clouds"]
                    )
                    
                    # Estimate energy output
                    current["energy_output_kWh"] = self.estimate_energy_output(current)
                    
                    logger.info(f"📡 Real-time update: {current['temperature']}°C, "
                              f"{current['solar_irradiance']:.0f} W/m², "
                              f"{current['energy_output_kWh']:.2f} kWh")
                    
                    yield current
                
                # Wait for next update
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Real-time stream stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in real-time stream: {e}")
                time.sleep(interval_seconds)
    
    def get_live_data_for_forecast(self) -> pd.DataFrame:
        """
        Get the latest data formatted for forecasting
        
        Returns:
            DataFrame ready for model input
        """
        logger.info("Fetching live data for forecast...")
        
        # Fetch current + forecast
        current = self.fetch_current_weather()
        forecast = self.fetch_forecast(hours=24)
        
        if not current or not forecast:
            logger.error("Failed to fetch live data")
            return None
        
        # Combine into DataFrame
        data_points = []
        
        # Add current
        data_points.append({
            "timestamp": current["timestamp"],
            "temperature": current["temperature"],
            "humidity": current["humidity"],
            "wind_speed": current["wind_speed"],
            "solar_irradiance": self.calculate_solar_irradiance(
                current["timestamp"], current["clouds"]
            ),
            "energy_output_kWh": self.estimate_energy_output(current)
        })
        
        # Add forecast
        for f in forecast:
            data_points.append({
                "timestamp": f["timestamp"],
                "temperature": f["temperature"],
                "humidity": f["humidity"],
                "wind_speed": f["wind_speed"],
                "solar_irradiance": self.calculate_solar_irradiance(
                    f["timestamp"], f["clouds"]
                ),
                "energy_output_kWh": self.estimate_energy_output(f)
            })
        
        df = pd.DataFrame(data_points)
        logger.info(f"✓ Live data ready: {len(df)} points")
        
        return df


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    agent = RealTimeDataAgent()
    
    # Test current weather
    current = agent.fetch_current_weather()
    print(f"\nCurrent: {current['temperature']}°C, {current['clouds']}% clouds")
    
    # Test forecast
    forecast = agent.fetch_forecast(hours=24)
    print(f"\nForecast: {len(forecast)} points")
    
    # Test historical data
    historical = agent.fetch_historical_data(days=30)
    print(f"\nHistorical: {len(historical)} samples")
    print(historical.head())
