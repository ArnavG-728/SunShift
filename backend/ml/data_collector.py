"""
Solar Data Collector
Collects historical weather + solar irradiance data for ML model training
Sources: NASA POWER API (historical solar), Open-Meteo (historical weather)
"""
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SolarDataCollector:
    """
    Collects historical solar and weather data for ML model training.
    Uses NASA POWER API for solar radiation and Open-Meteo for weather history.
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # NASA POWER API (solar irradiance)
        self.nasa_power_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        
        # Open-Meteo API (historical weather)
        self.open_meteo_url = "https://archive-api.open-meteo.com/v1/archive"
    
    def collect_training_data(
        self, 
        lat: float, 
        lon: float, 
        days: int = 365,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Collect historical data for training the solar forecasting model.
        
        Args:
            lat: Latitude
            lon: Longitude  
            days: Number of days of historical data (default: 365)
            end_date: End date for data collection (default: 2 days ago)
            
        Returns:
            DataFrame with hourly solar and weather data
        """
        if end_date is None:
            # Use 2 days ago to ensure data availability
            end_date = datetime.now() - timedelta(days=2)
        
        start_date = end_date - timedelta(days=days)
        
        cache_key = f"solar_data_{lat:.4f}_{lon:.4f}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        # Check cache
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
        
        logger.info(f"Collecting {days} days of solar/weather data for ({lat}, {lon})...")
        
        # Fetch solar radiation from NASA POWER
        solar_data = self._fetch_nasa_power_data(lat, lon, start_date, end_date)
        
        # Fetch weather data from Open-Meteo
        weather_data = self._fetch_open_meteo_data(lat, lon, start_date, end_date)
        
        # Merge datasets
        if solar_data is not None and weather_data is not None:
            # Ensure both have timestamp column
            solar_data['timestamp'] = pd.to_datetime(solar_data['timestamp'])
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            
            # Merge on timestamp
            merged_data = pd.merge(
                weather_data, 
                solar_data[['timestamp', 'solar_irradiance_wm2', 'clearsky_irradiance_wm2']], 
                on='timestamp', 
                how='left'
            )
            
            # Fill any missing solar data using physics-based estimates
            merged_data = self._fill_missing_solar(merged_data, lat)
            
        elif weather_data is not None:
            # Use weather data with physics-based solar estimates
            merged_data = self._add_physics_solar(weather_data, lat)
        elif solar_data is not None:
            merged_data = solar_data
        else:
            logger.warning("No data available, generating synthetic training data")
            merged_data = self._generate_synthetic_data(lat, lon, start_date, end_date)
        
        # Add derived features
        merged_data = self._add_features(merged_data, lat)
        
        # Calculate target: energy output
        merged_data = self._calculate_energy_output(merged_data, lat)
        
        # Clean data
        merged_data = self._clean_data(merged_data)
        
        # Cache the data
        merged_data.to_parquet(cache_file, index=False)
        logger.info(f"Cached {len(merged_data)} hours of data to {cache_file}")
        
        return merged_data
    
    def _fetch_nasa_power_data(
        self, 
        lat: float, 
        lon: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch hourly solar irradiance from NASA POWER API"""
        try:
            # NASA POWER hourly data
            params = {
                "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN",
                "community": "RE",
                "longitude": lon,
                "latitude": lat,
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "format": "JSON",
                "time-standard": "UTC"
            }
            
            logger.info(f"Fetching NASA POWER data from {start_date} to {end_date}...")
            response = requests.get(self.nasa_power_url, params=params, timeout=60)
            
            if response.status_code != 200:
                logger.warning(f"NASA POWER API returned {response.status_code}")
                return None
            
            data = response.json()
            
            if "properties" not in data or "parameter" not in data["properties"]:
                logger.warning("Invalid NASA POWER response structure")
                return None
            
            parameters = data["properties"]["parameter"]
            allsky = parameters.get("ALLSKY_SFC_SW_DWN", {})
            clearsky = parameters.get("CLRSKY_SFC_SW_DWN", {})
            
            records = []
            for date_hour_str, value in allsky.items():
                if value == -999 or value is None:
                    continue
                    
                try:
                    # Parse NASA POWER date format: YYYYMMDDHH
                    dt = datetime.strptime(date_hour_str, "%Y%m%d%H")
                    clearsky_value = clearsky.get(date_hour_str, value)
                    if clearsky_value == -999:
                        clearsky_value = value
                    
                    records.append({
                        'timestamp': dt,
                        'solar_irradiance_wm2': float(value),
                        'clearsky_irradiance_wm2': float(clearsky_value)
                    })
                except (ValueError, TypeError) as e:
                    continue
            
            if not records:
                logger.warning("No valid records in NASA POWER data")
                return None
            
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp')
            
            logger.info(f"✓ Fetched {len(df)} hours of NASA POWER solar data")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NASA POWER data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with NASA POWER: {e}")
            return None
    
    def _fetch_open_meteo_data(
        self, 
        lat: float, 
        lon: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch historical weather data from Open-Meteo API"""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "cloud_cover",
                    "wind_speed_10m",
                    "precipitation",
                    "pressure_msl",
                    "direct_radiation",
                    "diffuse_radiation",
                    "shortwave_radiation"
                ],
                "timezone": "UTC"
            }
            
            logger.info(f"Fetching Open-Meteo weather data...")
            response = requests.get(self.open_meteo_url, params=params, timeout=60)
            
            if response.status_code != 200:
                logger.warning(f"Open-Meteo API returned {response.status_code}")
                return None
            
            data = response.json()
            
            if "hourly" not in data:
                logger.warning("Invalid Open-Meteo response structure")
                return None
            
            hourly = data["hourly"]
            
            records = []
            for i, time_str in enumerate(hourly["time"]):
                record = {
                    'timestamp': datetime.fromisoformat(time_str.replace('Z', '+00:00').replace('+00:00', '')),
                    'temperature': hourly["temperature_2m"][i] if i < len(hourly.get("temperature_2m", [])) else 25.0,
                    'humidity': hourly["relative_humidity_2m"][i] if i < len(hourly.get("relative_humidity_2m", [])) else 60.0,
                    'clouds': hourly["cloud_cover"][i] if i < len(hourly.get("cloud_cover", [])) else 30.0,
                    'wind_speed': hourly["wind_speed_10m"][i] if i < len(hourly.get("wind_speed_10m", [])) else 3.0,
                    'precipitation': hourly["precipitation"][i] if i < len(hourly.get("precipitation", [])) else 0.0,
                    'pressure': hourly["pressure_msl"][i] if i < len(hourly.get("pressure_msl", [])) else 1013.0,
                }
                
                # Add Open-Meteo's radiation data (in W/m²)
                if "shortwave_radiation" in hourly and i < len(hourly["shortwave_radiation"]):
                    record['shortwave_radiation'] = hourly["shortwave_radiation"][i] or 0.0
                if "direct_radiation" in hourly and i < len(hourly["direct_radiation"]):
                    record['direct_radiation'] = hourly["direct_radiation"][i] or 0.0
                if "diffuse_radiation" in hourly and i < len(hourly["diffuse_radiation"]):
                    record['diffuse_radiation'] = hourly["diffuse_radiation"][i] or 0.0
                
                records.append(record)
            
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp')
            
            # Handle None/NaN values
            numeric_cols = ['temperature', 'humidity', 'clouds', 'wind_speed', 'precipitation', 'pressure']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].ffill().bfill()
            
            logger.info(f"✓ Fetched {len(df)} hours of Open-Meteo weather data")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Open-Meteo data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with Open-Meteo: {e}")
            return None
    
    def _fill_missing_solar(self, df: pd.DataFrame, lat: float) -> pd.DataFrame:
        """Fill missing solar irradiance data using physics-based estimates"""
        df = df.copy()
        
        # If we have shortwave_radiation from Open-Meteo, use it as fallback
        if 'shortwave_radiation' in df.columns:
            df['solar_irradiance_wm2'] = df['solar_irradiance_wm2'].fillna(df['shortwave_radiation'])
        
        # Physics-based filling for remaining NaN values
        mask = df['solar_irradiance_wm2'].isna()
        if mask.sum() > 0:
            for idx in df[mask].index:
                row = df.loc[idx]
                timestamp = pd.to_datetime(row['timestamp'])
                clouds = row.get('clouds', 30)
                df.loc[idx, 'solar_irradiance_wm2'] = self._calculate_physics_irradiance(timestamp, clouds, lat)
        
        # Fill clearsky if missing
        if 'clearsky_irradiance_wm2' not in df.columns:
            df['clearsky_irradiance_wm2'] = df.apply(
                lambda row: self._calculate_physics_irradiance(
                    pd.to_datetime(row['timestamp']), 0, lat
                ), axis=1
            )
        else:
            df['clearsky_irradiance_wm2'] = df['clearsky_irradiance_wm2'].fillna(
                df.apply(lambda row: self._calculate_physics_irradiance(
                    pd.to_datetime(row['timestamp']), 0, lat
                ), axis=1)
            )
        
        return df
    
    def _add_physics_solar(self, df: pd.DataFrame, lat: float) -> pd.DataFrame:
        """Add physics-based solar irradiance when no API data available"""
        df = df.copy()
        
        df['solar_irradiance_wm2'] = df.apply(
            lambda row: self._calculate_physics_irradiance(
                pd.to_datetime(row['timestamp']), 
                row.get('clouds', 30), 
                lat
            ), axis=1
        )
        
        df['clearsky_irradiance_wm2'] = df.apply(
            lambda row: self._calculate_physics_irradiance(
                pd.to_datetime(row['timestamp']), 0, lat
            ), axis=1
        )
        
        return df
    
    def _calculate_physics_irradiance(self, timestamp: datetime, clouds: float, lat: float) -> float:
        """Calculate solar irradiance using physics"""
        hour = timestamp.hour + timestamp.minute / 60
        day_of_year = timestamp.timetuple().tm_yday
        
        # Solar declination
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Solar elevation
        elevation = np.degrees(np.arcsin(
            np.sin(np.radians(lat)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(lat)) * np.cos(np.radians(declination)) * 
            np.cos(np.radians(hour_angle))
        ))
        
        if elevation <= 0:
            return 0.0
        
        # Air mass
        air_mass = 1 / (np.sin(np.radians(elevation)) + 0.50572 * (elevation + 6.07995)**-1.6364)
        
        # Clear sky irradiance
        solar_constant = 1367  # W/m²
        clear_sky = solar_constant * (0.7 ** (air_mass ** 0.678)) * np.sin(np.radians(elevation))
        
        # Cloud factor
        cloud_factor = 1 - (clouds / 100) * 0.75
        
        return max(0.0, clear_sky * cloud_factor)
    
    def _generate_synthetic_data(
        self, 
        lat: float, 
        lon: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate synthetic training data when APIs fail"""
        logger.warning("Generating synthetic training data...")
        
        hours = int((end_date - start_date).total_seconds() / 3600)
        records = []
        
        # Climate zone based on latitude
        abs_lat = abs(lat)
        if abs_lat < 15:
            base_temp, seasonal_swing, base_humidity, base_clouds = 28, 4, 75, 40
        elif abs_lat < 35:
            base_temp, seasonal_swing, base_humidity, base_clouds = 22, 8, 60, 35
        elif abs_lat < 55:
            base_temp, seasonal_swing, base_humidity, base_clouds = 15, 12, 65, 50
        else:
            base_temp, seasonal_swing, base_humidity, base_clouds = 5, 15, 70, 60
        
        hemisphere = 1 if lat >= 0 else -1
        
        for h in range(hours):
            timestamp = start_date + timedelta(hours=h)
            hour = timestamp.hour
            day_of_year = timestamp.timetuple().tm_yday
            
            # Temperature
            seasonal = seasonal_swing * np.sin(2 * np.pi * (day_of_year - 172) / 365 * hemisphere)
            daily = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            temperature = base_temp + seasonal + daily + np.random.normal(0, 2)
            
            # Humidity
            humidity = base_humidity - (temperature - base_temp) * 1.2 + np.random.normal(0, 5)
            humidity = np.clip(humidity, 25, 98)
            
            # Clouds with persistence
            if h == 0:
                clouds = base_clouds + np.random.normal(0, 15)
            else:
                prev_clouds = records[-1]['clouds']
                clouds = prev_clouds * 0.9 + (base_clouds + np.random.normal(0, 15)) * 0.1
            clouds = np.clip(clouds, 0, 100)
            
            # Wind
            if h == 0:
                wind_speed = np.random.normal(4, 1.5)
            else:
                prev_wind = records[-1]['wind_speed']
                wind_speed = prev_wind * 0.7 + np.random.normal(4, 1.5) * 0.3
            wind_speed = np.clip(wind_speed, 0.5, 15)
            
            # Solar irradiance
            irradiance = self._calculate_physics_irradiance(timestamp, clouds, lat)
            clearsky = self._calculate_physics_irradiance(timestamp, 0, lat)
            
            records.append({
                'timestamp': timestamp,
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'clouds': round(clouds, 1),
                'wind_speed': round(wind_speed, 1),
                'precipitation': max(0, np.random.normal(0, 0.5) if clouds > 70 else 0),
                'pressure': 1013 + np.random.normal(0, 5),
                'solar_irradiance_wm2': irradiance,
                'clearsky_irradiance_wm2': clearsky
            })
        
        return pd.DataFrame(records)
    
    def _add_features(self, df: pd.DataFrame, lat: float) -> pd.DataFrame:
        """Add derived features for ML model"""
        df = df.copy()
        
        # Time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Daylight indicator
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        
        # Solar position features
        df['solar_declination'] = 23.45 * np.sin(np.radians(360 * (284 + df['day_of_year']) / 365))
        
        # Clear sky ratio (cloud impact indicator)
        df['clearsky_ratio'] = np.where(
            df['clearsky_irradiance_wm2'] > 0,
            df['solar_irradiance_wm2'] / df['clearsky_irradiance_wm2'],
            0
        )
        df['clearsky_ratio'] = df['clearsky_ratio'].clip(0, 1.5)
        
        # Weather interactions
        df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
        df['cloud_wind'] = df['clouds'] * df['wind_speed'] / 10
        
        # Lagged features (for sequences)
        df['solar_lag_1h'] = df['solar_irradiance_wm2'].shift(1).fillna(0)
        df['solar_lag_24h'] = df['solar_irradiance_wm2'].shift(24).fillna(0)
        df['temp_lag_1h'] = df['temperature'].shift(1).fillna(df['temperature'])
        df['clouds_lag_1h'] = df['clouds'].shift(1).fillna(df['clouds'])
        
        # Rolling statistics
        df['solar_rolling_3h'] = df['solar_irradiance_wm2'].rolling(3, min_periods=1).mean()
        df['solar_rolling_24h'] = df['solar_irradiance_wm2'].rolling(24, min_periods=1).mean()
        df['temp_rolling_24h'] = df['temperature'].rolling(24, min_periods=1).mean()
        df['clouds_rolling_6h'] = df['clouds'].rolling(6, min_periods=1).mean()
        
        return df
    
    def _calculate_energy_output(
        self, 
        df: pd.DataFrame, 
        lat: float,
        system_size_kwp: float = 5.0,
        performance_ratio: float = 0.78,
        panel_tilt: float = 30.0,
        panel_azimuth: float = 180.0
    ) -> pd.DataFrame:
        """Calculate energy output as target variable"""
        df = df.copy()
        
        # Temperature derating factor
        df['temp_factor'] = (1 - 0.004 * (df['temperature'] - 25)).clip(0.7, 1.0)
        
        # Simple tilt factor based on latitude
        optimal_tilt = abs(lat)  # Optimal tilt roughly equals latitude
        tilt_efficiency = 1 - 0.01 * abs(panel_tilt - optimal_tilt)  # 1% loss per degree from optimal
        
        # Energy output (kWh)
        df['energy_output_kWh'] = (
            df['solar_irradiance_wm2'] / 1000.0 *  # Convert W/m² to kW/m²
            system_size_kwp *
            performance_ratio *
            df['temp_factor'] *
            tilt_efficiency
        )
        
        # Ensure non-negative
        df['energy_output_kWh'] = df['energy_output_kWh'].clip(lower=0)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the training data"""
        df = df.copy()
        
        # Remove rows with NaN in critical columns
        critical_cols = ['timestamp', 'temperature', 'humidity', 'clouds', 
                        'solar_irradiance_wm2', 'energy_output_kWh']
        
        initial_len = len(df)
        df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
        
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} rows with NaN values")
        
        # Remove outliers
        for col in ['temperature', 'solar_irradiance_wm2', 'energy_output_kWh']:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Cleaned data: {len(df)} records")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = SolarDataCollector()
    data = collector.collect_training_data(
        lat=28.6139, 
        lon=77.2090, 
        days=90
    )
    
    print(f"\n✓ Collected {len(data)} hours of training data")
    print(f"✓ Columns: {list(data.columns)}")
    print(f"✓ Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"✓ Energy output range: {data['energy_output_kWh'].min():.2f} - {data['energy_output_kWh'].max():.2f} kWh")
