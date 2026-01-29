"""
Hybrid Solar Forecaster
Combines ML predictions with physics-based validation and fallback
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .solar_forecaster import SolarForecasterML
from .data_collector import SolarDataCollector

logger = logging.getLogger(__name__)


class HybridForecaster:
    """
    Hybrid forecaster that combines:
    1. ML-based predictions (LSTM) for pattern recognition
    2. Physics-based calculations for validation and fallback
    3. Ensemble weighting based on model confidence
    
    This provides the best of both worlds:
    - ML captures complex patterns and historical trends
    - Physics ensures predictions are physically plausible
    - Fallback to physics when ML model is unavailable
    """
    
    def __init__(
        self,
        system_size_kwp: float = 5.0,
        performance_ratio: float = 0.78,
        panel_tilt: float = 30.0,
        panel_azimuth: float = 180.0,
        model_dir: str = None
    ):
        """
        Initialize the hybrid forecaster.
        
        Args:
            system_size_kwp: Solar system size in kWp
            performance_ratio: System performance ratio (default: 0.78)
            panel_tilt: Panel tilt angle in degrees
            panel_azimuth: Panel direction (0=N, 90=E, 180=S, 270=W)
            model_dir: Directory for ML models
        """
        self.system_size = system_size_kwp
        self.performance_ratio = performance_ratio
        self.panel_tilt = panel_tilt
        self.panel_azimuth = panel_azimuth
        
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "models" / "ml_saved"
        
        # ML forecaster
        self.ml_forecaster: Optional[SolarForecasterML] = None
        self.ml_available = False
        
        # Physics fallback
        self.data_collector = SolarDataCollector()
        
        # Ensemble weights
        self.ml_weight = 0.7  # Weight for ML predictions when available
        self.physics_weight = 0.3  # Weight for physics predictions
        
        # Try to load existing ML model
        self._try_load_ml_model()
    
    def _try_load_ml_model(self, model_name: str = "solar_forecaster") -> bool:
        """Try to load an existing ML model."""
        self.ml_forecaster = SolarForecasterML(model_dir=str(self.model_dir))
        self.ml_available = self.ml_forecaster.load(model_name)
        
        if self.ml_available:
            logger.info(f"✓ ML model loaded: {model_name}")
            logger.info(f"  MAE: {self.ml_forecaster.metadata.get('val_mae', 'N/A')} kWh")
        else:
            logger.warning("ML model not available, will use physics-based forecasting only")
        
        return self.ml_available
    
    def _calculate_physics_energy(
        self, 
        timestamp: datetime, 
        temperature: float, 
        clouds: float, 
        lat: float, 
        lon: float
    ) -> float:
        """Calculate energy output using physics."""
        # Solar irradiance
        irradiance = self.data_collector._calculate_physics_irradiance(timestamp, clouds, lat)
        
        if irradiance <= 0:
            return 0.0
        
        # Temperature derating
        temp_factor = max(0.7, min(1.0, 1 - 0.004 * (temperature - 25)))
        
        # Panel tilt efficiency
        optimal_tilt = abs(lat)
        tilt_efficiency = 1 - 0.01 * abs(self.panel_tilt - optimal_tilt)
        
        # Energy output
        energy = (
            irradiance / 1000.0 *
            self.system_size *
            self.performance_ratio *
            temp_factor *
            tilt_efficiency
        )
        
        return max(0.0, energy)
    
    def forecast(
        self,
        lat: float,
        lon: float,
        hours: int = 168,
        weather_data: pd.DataFrame = None
    ) -> Dict:
        """
        Generate forecast using hybrid approach.
        
        Args:
            lat: Latitude
            lon: Longitude
            hours: Hours to forecast (default: 168 = 7 days)
            weather_data: Optional pre-fetched weather data
            
        Returns:
            Dictionary with predictions and metadata
        """
        logger.info(f"Generating hybrid forecast for ({lat}, {lon}), {hours} hours")
        logger.info(f"  System: {self.system_size} kWp, PR {self.performance_ratio*100:.0f}%")
        logger.info(f"  ML model available: {self.ml_available}")
        
        # Fetch weather data if not provided
        if weather_data is None:
            from real_weather_forecast import RealWeatherSolarForecaster
            physics_forecaster = RealWeatherSolarForecaster(
                system_size_kwp=self.system_size,
                efficiency=0.15,
                panel_tilt=self.panel_tilt,
                panel_azimuth=self.panel_azimuth,
                performance_ratio=self.performance_ratio
            )
            weather_df = physics_forecaster.fetch_weather_forecast(lat, lon)
            weather_df = physics_forecaster._extend_weather_forecast(weather_df, hours, lat, lon)
        else:
            weather_df = weather_data.copy()
        
        # Generate physics-based predictions
        physics_predictions = []
        for _, row in weather_df.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            energy = self._calculate_physics_energy(
                timestamp,
                row['temperature'],
                row['clouds'],
                lat,
                lon
            )
            physics_predictions.append(energy)
        
        weather_df['physics_energy_kWh'] = physics_predictions
        
        # Generate ML predictions if available
        if self.ml_available:
            try:
                # Get historical data for ML context
                historical_data = self._get_or_generate_historical_context(lat, lon)
                
                # Prepare features for ML
                weather_df = self.ml_forecaster.prepare_features(weather_df, is_future=True)
                
                # Predict using ML
                ml_predictions = self.ml_forecaster.predict_future(
                    historical_data=historical_data,
                    future_weather=weather_df,
                    lat=lat,
                    lon=lon
                )
                
                weather_df = pd.merge(
                    weather_df,
                    ml_predictions[['timestamp', 'predicted_output_kWh']].rename(
                        columns={'predicted_output_kWh': 'ml_energy_kWh'}
                    ),
                    on='timestamp',
                    how='left'
                )
                
                # Ensemble: weighted average
                weather_df['ml_energy_kWh'] = weather_df['ml_energy_kWh'].fillna(weather_df['physics_energy_kWh'])
                weather_df['predicted_output_kWh'] = (
                    self.ml_weight * weather_df['ml_energy_kWh'] +
                    self.physics_weight * weather_df['physics_energy_kWh']
                )
                
                # Validate: physics provides bounds
                # If ML prediction is too different from physics, trust physics more
                diff_ratio = np.abs(weather_df['ml_energy_kWh'] - weather_df['physics_energy_kWh']) / (weather_df['physics_energy_kWh'] + 0.01)
                high_diff_mask = diff_ratio > 1.0  # ML differs by more than 100%
                
                # For high differences, increase physics weight
                weather_df.loc[high_diff_mask, 'predicted_output_kWh'] = (
                    0.3 * weather_df.loc[high_diff_mask, 'ml_energy_kWh'] +
                    0.7 * weather_df.loc[high_diff_mask, 'physics_energy_kWh']
                )
                
                forecasting_method = "hybrid_ml_physics"
                
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}, falling back to physics")
                weather_df['predicted_output_kWh'] = weather_df['physics_energy_kWh']
                forecasting_method = "physics_fallback"
        else:
            weather_df['predicted_output_kWh'] = weather_df['physics_energy_kWh']
            forecasting_method = "physics_only"
        
        # Ensure non-negative
        weather_df['predicted_output_kWh'] = weather_df['predicted_output_kWh'].clip(lower=0)
        
        # Add confidence bounds
        weather_df['confidence_lower'] = weather_df['predicted_output_kWh'] * 0.85
        weather_df['confidence_upper'] = weather_df['predicted_output_kWh'] * 1.15
        
        # Rename for consistency
        weather_df['solar_irradiance'] = weather_df.get('solar_irradiance_wm2', 
            weather_df.apply(lambda r: self.data_collector._calculate_physics_irradiance(
                pd.to_datetime(r['timestamp']), r['clouds'], lat
            ), axis=1)
        )
        
        # Create output structure
        hourly_24h = self._create_hourly_output(weather_df.head(24))
        daily_7d = self._create_daily_output(weather_df.head(168))
        
        # Calculate metrics
        total_24h = float(weather_df.head(24)['predicted_output_kWh'].sum())
        avg_24h = float(weather_df.head(24)['predicted_output_kWh'].mean())
        peak_24h = float(weather_df.head(24)['predicted_output_kWh'].max())
        total_week = float(weather_df.head(168)['predicted_output_kWh'].sum())
        
        # Find peak hour
        peak_idx = weather_df.head(24)['predicted_output_kWh'].idxmax()
        peak_hour = weather_df.loc[peak_idx]
        
        return {
            'status': 'success',
            'hourly_24h': hourly_24h,
            'daily_7d': daily_7d,
            'metrics': {
                'total_24h': total_24h,
                'avg_24h': avg_24h,
                'peak_24h': peak_24h,
                'total_week': total_week
            },
            'insights': {
                'summary': f"{'ML-enhanced' if self.ml_available else 'Physics-based'} forecast for ({lat:.2f}, {lon:.2f})",
                'peak_hour': pd.to_datetime(peak_hour['timestamp']).strftime('%I:%M %p'),
                'peak_energy': float(peak_hour['predicted_output_kWh']),
                'total_today': total_24h,
                'panel_orientation': f"{self.panel_tilt}° tilt, {self.panel_azimuth}° azimuth",
                'forecasting_method': forecasting_method,
                'ml_model_available': self.ml_available
            }
        }
    
    def _get_or_generate_historical_context(self, lat: float, lon: float) -> pd.DataFrame:
        """Get historical context for ML predictions."""
        # Generate synthetic historical data for context
        hours = self.ml_forecaster.sequence_length if self.ml_forecaster else 48
        now = datetime.now()
        
        records = []
        for h in range(hours, 0, -1):
            timestamp = now - timedelta(hours=h)
            hour = timestamp.hour
            
            # Synthetic but physically plausible values
            temperature = 25 + 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            humidity = 60 - (temperature - 25) * 1.5
            clouds = 30 + 20 * np.random.random()
            wind_speed = 5 + 2 * np.random.random()
            
            irradiance = self.data_collector._calculate_physics_irradiance(timestamp, clouds, lat)
            clearsky = self.data_collector._calculate_physics_irradiance(timestamp, 0, lat)
            
            energy = self._calculate_physics_energy(timestamp, temperature, clouds, lat, 0)
            
            records.append({
                'timestamp': timestamp,
                'temperature': temperature,
                'humidity': humidity,
                'clouds': clouds,
                'wind_speed': wind_speed,
                'pressure': 1013,
                'solar_irradiance_wm2': irradiance,
                'clearsky_irradiance_wm2': clearsky,
                'energy_output_kWh': energy
            })
        
        return pd.DataFrame(records)
    
    def _create_hourly_output(self, df: pd.DataFrame) -> List[Dict]:
        """Create hourly output format."""
        output = []
        for _, row in df.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            output.append({
                'timestamp': timestamp.isoformat(),
                'temperature': float(row['temperature']),
                'humidity': float(row.get('humidity', 60)),
                'wind_speed': float(row.get('wind_speed', 3)),
                'clouds': float(row['clouds']),
                'weather': row.get('weather', 'Unknown'),
                'solar_irradiance': float(row.get('solar_irradiance', row.get('solar_irradiance_wm2', 0))),
                'predicted_output_kWh': float(row['predicted_output_kWh']),
                'confidence_lower': float(row['confidence_lower']),
                'confidence_upper': float(row['confidence_upper'])
            })
        return output
    
    def _create_daily_output(self, df: pd.DataFrame) -> List[Dict]:
        """Create daily aggregated output."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        daily = df.groupby('date').agg({
            'predicted_output_kWh': ['sum', 'mean', 'min', 'max'],
            'temperature': 'mean',
            'clouds': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        
        daily.columns = ['date', 'total_kwh', 'avg_kwh', 'min_kwh', 'max_kwh', 
                        'avg_temp', 'avg_clouds', 'avg_wind']
        
        output = []
        for _, row in daily.iterrows():
            output.append({
                'date': str(row['date']),
                'total_kwh': float(row['total_kwh']),
                'avg_kwh': float(row['avg_kwh']),
                'min_kwh': float(row['min_kwh']),
                'max_kwh': float(row['max_kwh']),
                'avg_temp': float(row['avg_temp']),
                'avg_clouds': float(row['avg_clouds']),
                'avg_wind': float(row['avg_wind'])
            })
        
        return output
    
    def get_model_status(self) -> Dict:
        """Get current model status."""
        status = {
            'ml_available': self.ml_available,
            'system_config': {
                'size_kwp': self.system_size,
                'performance_ratio': self.performance_ratio,
                'panel_tilt': self.panel_tilt,
                'panel_azimuth': self.panel_azimuth
            }
        }
        
        if self.ml_available and self.ml_forecaster:
            status['ml_model'] = self.ml_forecaster.get_model_info()
        
        return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    forecaster = HybridForecaster(
        system_size_kwp=5.0,
        performance_ratio=0.78,
        panel_tilt=30.0,
        panel_azimuth=180.0
    )
    
    result = forecaster.forecast(lat=28.6139, lon=77.2090, hours=168)
    
    print(f"\n✓ Status: {result['status']}")
    print(f"✓ Method: {result['insights']['forecasting_method']}")
    print(f"✓ ML available: {result['insights']['ml_model_available']}")
    print(f"✓ Total 24h: {result['metrics']['total_24h']:.2f} kWh")
    print(f"✓ Peak: {result['insights']['peak_hour']} ({result['insights']['peak_energy']:.2f} kWh)")
