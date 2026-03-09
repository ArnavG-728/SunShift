from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from config import config
from utils.validators import validate_coordinates, validate_system_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["forecast"])

class ForecastRequest(BaseModel):
    """Forecast request model"""
    days: Optional[int] = 30
    model_type: Optional[str] = "LSTM"
    latitude: Optional[float] = 28.6139
    longitude: Optional[float] = 77.2090
    system_size: Optional[float] = 5.0
    efficiency: Optional[float] = 0.15
    panel_tilt: Optional[float] = 30.0
    panel_azimuth: Optional[float] = 180.0
    performance_ratio: Optional[float] = 0.78

class HybridForecastRequest(BaseModel):
    """Hybrid ML+Physics Forecast Request"""
    latitude: Optional[float] = 28.6139
    longitude: Optional[float] = 77.2090
    hours: Optional[int] = 168
    system_size: Optional[float] = 5.0
    performance_ratio: Optional[float] = 0.78
    panel_tilt: Optional[float] = 30.0
    panel_azimuth: Optional[float] = 180.0

@router.post("/run")
async def run_forecast(request: ForecastRequest = None):
    """Run the complete forecasting pipeline with multi-horizon predictions"""
    try:
        # Extract coordinates from request or use defaults
        if request:
            lat = request.latitude
            lon = request.longitude
            system_size = request.system_size
            efficiency = request.efficiency
            panel_tilt = request.panel_tilt
            panel_azimuth = request.panel_azimuth
            performance_ratio = request.performance_ratio or 0.78
            logger.info(f"API: Running forecast for ({lat}, {lon})...")
        else:
            lat = 28.6139
            lon = 77.2090
            system_size = 5.0
            efficiency = 0.15
            panel_tilt = 30.0
            panel_azimuth = 180.0
            performance_ratio = 0.78
            logger.info("API: Running forecast with default coordinates...")
        
        # Validate coordinates
        valid, error = validate_coordinates(lat, lon)
        if not valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Validate system configuration
        valid, error = validate_system_config(system_size, efficiency, panel_tilt, panel_azimuth)
        if not valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Use real weather forecaster with panel orientation
        from real_weather_forecast import RealWeatherSolarForecaster
        
        forecaster = RealWeatherSolarForecaster(
            system_size_kwp=system_size,
            efficiency=efficiency,
            panel_tilt=panel_tilt,
            panel_azimuth=panel_azimuth,
            performance_ratio=performance_ratio
        )
        result = forecaster.forecast(lat=lat, lon=lon, hours=168)
        
        logger.info(f"Forecast status: {result.get('status')}")
        logger.info(f"Hourly 24h count: {len(result.get('hourly_24h', []))}")
        logger.info(f"Daily 7d count: {len(result.get('daily_7d', []))}")
        
        # Persist results to CSV files for other endpoints/clients
        try:
            if result.get('hourly_24h'):
                pd.DataFrame(result['hourly_24h']).to_csv(config.DATA_DIR / 'predictions_24h.csv', index=False)
            if result.get('daily_7d'):
                pd.DataFrame(result['daily_7d']).to_csv(config.DATA_DIR / 'predictions_7d.csv', index=False)
        except Exception as save_err:
            logger.warning(f"Unable to persist forecast CSVs: {save_err}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest")
async def get_latest_forecast():
    """Get the latest forecast results"""
    try:
        pred_path = config.DATA_DIR / "predictions_24h.csv"
        
        if not pred_path.exists():
            raise HTTPException(
                status_code=404, 
                detail="No forecast available. Please run the forecast pipeline first."
            )
        
        df = pd.read_csv(pred_path)
        df = df.replace({np.nan: None})
        
        return {
            "status": "success",
            "predictions": df.to_dict(orient="records"),
            "count": len(df)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/24h")
async def get_24h_forecast():
    """Get 24-hour hourly forecast"""
    try:
        pred_path = config.DATA_DIR / "predictions_24h.csv"
        
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail="No 24h forecast available")
        
        df = pd.read_csv(pred_path)
        df = df.replace({np.nan: None})
        return {"status": "success", "data": df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/7d")
async def get_7d_forecast():
    """Get 7-day daily forecast"""
    try:
        pred_path = config.DATA_DIR / "predictions_7d.csv"
        
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail="No 7d forecast available")
        
        df = pd.read_csv(pred_path)
        df = df.replace({np.nan: None})
        return {"status": "success", "data": df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hybrid")
async def run_hybrid_forecast(request: HybridForecastRequest = HybridForecastRequest()):
    """
    Run hybrid ML + Physics-based forecast.
    """
    try:
        from ml.unified_forecaster import HybridForecaster
        
        logger.info(f"Running hybrid forecast for ({request.latitude}, {request.longitude})...")
        
        forecaster = HybridForecaster(
            system_size_kwp=request.system_size,
            performance_ratio=request.performance_ratio,
            panel_tilt=request.panel_tilt,
            panel_azimuth=request.panel_azimuth
        )
        
        result = forecaster.forecast(
            lat=request.latitude,
            lon=request.longitude,
            hours=request.hours
        )
        
        # Persist results
        try:
            if result.get('hourly_24h'):
                pd.DataFrame(result['hourly_24h']).to_csv(
                    config.DATA_DIR / 'predictions_24h.csv', index=False
                )
            if result.get('daily_7d'):
                pd.DataFrame(result['daily_7d']).to_csv(
                    config.DATA_DIR / 'predictions_7d.csv', index=False
                )
        except Exception as save_err:
            logger.warning(f"Unable to persist forecast CSVs: {save_err}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in hybrid forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
