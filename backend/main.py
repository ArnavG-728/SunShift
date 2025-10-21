"""
FastAPI Backend for SunShift - Solar Energy Forecasting System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import logging
import pandas as pd

from config import config
from graph.workflow import workflow_instance
from utils.validators import (
    validate_coordinates,
    validate_system_config,
    validate_forecast_params,
    validate_battery_config,
    validate_financial_params
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="SunShift - AI-Powered Solar Energy Forecasting & Analytics Platform"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str


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


class OptimizationRequest(BaseModel):
    """Optimization request model"""
    latitude: Optional[float] = 28.6139
    longitude: Optional[float] = 77.2090
    battery_capacity: Optional[float] = 0.0
    electricity_tariff: Optional[float] = 0.12
    feed_in_tariff: Optional[float] = 0.08
    system_size: Optional[float] = 5.0
    efficiency: Optional[float] = 0.15
    panel_tilt: Optional[float] = 30.0
    panel_azimuth: Optional[float] = 180.0
    performance_ratio: Optional[float] = 0.78
    grid_co2_factor: Optional[float] = 0.70
    max_grid_import: Optional[float] = 10.0


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {config.APP_NAME}",
        "version": config.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": config.APP_NAME,
        "version": config.APP_VERSION
    }


@app.post("/forecast/run")
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
        logger.info(f"Weekly 4w count: {len(result.get('weekly_4w', []))}")
        
        # Persist results to CSV files for other endpoints/clients
        try:
            import pandas as pd
            if result.get('hourly_24h'):
                pd.DataFrame(result['hourly_24h']).to_csv(config.DATA_DIR / 'predictions_24h.csv', index=False)
            if result.get('daily_7d'):
                pd.DataFrame(result['daily_7d']).to_csv(config.DATA_DIR / 'predictions_7d.csv', index=False)
            if result.get('weekly_4w'):
                pd.DataFrame(result['weekly_4w']).to_csv(config.DATA_DIR / 'predictions_4w.csv', index=False)
        except Exception as save_err:
            logger.warning(f"Unable to persist forecast CSVs: {save_err}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the AI assistant"""
    try:
        logger.info(f"API: Processing chat query: {request.query}")
        result = workflow_instance.chat(request.query)
        
        return {
            "status": "success",
            "query": request.query,
            "response": result["response"]
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/latest")
async def get_latest_forecast():
    """Get the latest forecast results"""
    try:
        import pandas as pd
        from pathlib import Path
        
        pred_path = config.DATA_DIR / "predictions_24h.csv"
        
        if not pred_path.exists():
            raise HTTPException(
                status_code=404, 
                detail="No forecast available. Please run the forecast pipeline first."
            )
        
        df = pd.read_csv(pred_path)
        
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


@app.get("/forecast/24h")
async def get_24h_forecast():
    """Get 24-hour hourly forecast"""
    try:
        import pandas as pd
        pred_path = config.DATA_DIR / "predictions_24h.csv"
        
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail="No 24h forecast available")
        
        df = pd.read_csv(pred_path)
        return {"status": "success", "data": df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/7d")
async def get_7d_forecast():
    """Get 7-day daily forecast"""
    try:
        import pandas as pd
        pred_path = config.DATA_DIR / "predictions_7d.csv"
        
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail="No 7d forecast available")
        
        df = pd.read_csv(pred_path)
        return {"status": "success", "data": df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/4w")
async def get_4w_forecast():
    """Get 4-week weekly forecast"""
    try:
        import pandas as pd
        pred_path = config.DATA_DIR / "predictions_4w.csv"
        
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail="No 4w forecast available")
        
        df = pd.read_csv(pred_path)
        return {"status": "success", "data": df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    try:
        # This would typically come from a database
        # For now, we'll return placeholder metrics
        return {
            "status": "success",
            "metrics": {
                "mae": 0.0,
                "rmse": 0.0,
                "accuracy": 0.0
            },
            "message": "Run the forecast pipeline to get actual metrics"
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/current")
async def get_current_weather(lat: float = 28.6139, lon: float = 77.2090, system_size: float = 5.0, performance_ratio: float = 0.78):
    """Get current real-time weather data for specified coordinates"""
    try:
        from agents.realtime_data_agent import RealTimeDataAgent
        
        logger.info(f"Fetching weather for coordinates: ({lat}, {lon})")
        agent = RealTimeDataAgent(latitude=lat, longitude=lon)
        current = agent.fetch_current_weather(lat=lat, lon=lon)
        
        if not current:
            raise HTTPException(status_code=503, detail="Failed to fetch real-time data")
        
        # Calculate solar irradiance once (using location's local time)
        local_time = current["timestamp"]
        logger.info(f"Calculating solar irradiance for local time: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        solar_irradiance = agent.calculate_solar_irradiance(
            local_time, 
            current["clouds"],
            lat=lat,
            lon=lon
        )
        
        # Calculate energy output using the irradiance we already computed
        temperature = current.get("temperature")
        if temperature is None or pd.isna(temperature):
            temp_factor = 1.0
        else:
            temp_factor = 1 - 0.004 * (float(temperature) - 25.0)
            temp_factor = max(0.7, min(1.0, temp_factor))
        
        energy_output = (float(solar_irradiance) / 1000.0) * float(system_size) * float(performance_ratio) * float(temp_factor)
        energy_output = float(max(0.0, energy_output))
        
        response_data = {
            "status": "success",
            "data": {
                "timestamp": current["timestamp"].isoformat(),
                "local_time": current["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "timezone_offset": current.get("timezone_offset", 0),
                "temperature": float(current["temperature"]),
                "humidity": int(current["humidity"]),
                "wind_speed": float(current["wind_speed"]),
                "clouds": int(current["clouds"]),
                "solar_irradiance": float(solar_irradiance),
                "energy_output_kWh": float(energy_output),
                "weather": current["weather"],
                "description": current["description"]
            }
        }
        
        # Debug: Log the full response
        logger.info(f"API Response: {response_data}")
        
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/forecast")
async def get_realtime_forecast(hours: int = 24):
    """Get real-time weather forecast"""
    try:
        from agents.realtime_data_agent import RealTimeDataAgent
        
        agent = RealTimeDataAgent()
        forecast = agent.fetch_forecast(hours=hours)
        
        if not forecast:
            raise HTTPException(status_code=503, detail="Failed to fetch forecast data")
        
        # Format forecast data
        forecast_data = [
            {
                "timestamp": f["timestamp"].isoformat(),
                "temperature": f["temperature"],
                "humidity": f["humidity"],
                "wind_speed": f["wind_speed"],
                "clouds": f["clouds"],
                "weather": f["weather"],
                "description": f["description"],
                "pop": f.get("pop", 0)
            }
            for f in forecast
        ]
        
        return {
            "status": "success",
            "forecast": forecast_data,
            "count": len(forecast_data)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/status")
async def get_realtime_status():
    """Check real-time data connection status"""
    try:
        from agents.realtime_data_agent import RealTimeDataAgent
        
        agent = RealTimeDataAgent()
        
        # Try to fetch current weather
        current = agent.fetch_current_weather()
        
        if current:
            return {
                "status": "connected",
                "message": "Real-time data connection active",
                "api": "OpenWeather API",
                "last_update": current["timestamp"].isoformat(),
                "location": {
                    "lat": agent.default_lat,
                    "lon": agent.default_lon
                }
            }
        else:
            return {
                "status": "disconnected",
                "message": "Failed to connect to real-time data source",
                "api": "OpenWeather API"
            }
    except Exception as e:
        logger.error(f"Error checking real-time status: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/optimize")
async def optimize_energy(request: OptimizationRequest = OptimizationRequest()):
    """Get AI-powered energy optimization recommendations"""
    try:
        from agents.optimization_agent import OptimizationAgent
        
        # Extract parameters with defaults
        lat = getattr(request, 'latitude', 28.6139)
        lon = getattr(request, 'longitude', 77.2090)
        battery_capacity = getattr(request, 'battery_capacity', 0.0)
        electricity_tariff = getattr(request, 'electricity_tariff', 0.12)
        feed_in_tariff = getattr(request, 'feed_in_tariff', 0.08)
        system_size = getattr(request, 'system_size', 5.0)
        performance_ratio = getattr(request, 'performance_ratio', 0.78)
        efficiency = getattr(request, 'efficiency', 0.15)
        panel_tilt = getattr(request, 'panel_tilt', 30.0)
        panel_azimuth = getattr(request, 'panel_azimuth', 180.0)
        grid_co2_factor = getattr(request, 'grid_co2_factor', 0.70)
        max_grid_import = getattr(request, 'max_grid_import', 10.0)
        
        logger.info(f"Optimize request: lat={lat}, lon={lon}, battery={battery_capacity}, system_size={system_size}, panel={panel_tilt}°/{panel_azimuth}°")
        
        # Validate coordinates
        valid, error = validate_coordinates(lat, lon)
        if not valid:
            logger.error(f"Coordinate validation failed: {error}")
            raise HTTPException(status_code=400, detail=f"Invalid coordinates: {error}")
        
        # Validate battery config if battery exists
        if battery_capacity > 0:
            valid, error = validate_battery_config(battery_capacity, 0.95)
            if not valid:
                logger.error(f"Battery validation failed: {error}")
                raise HTTPException(status_code=400, detail=f"Invalid battery config: {error}")
        
        # Validate financial params
        valid, error = validate_financial_params(electricity_tariff, feed_in_tariff)
        if not valid:
            logger.error(f"Financial validation failed: {error}")
            raise HTTPException(status_code=400, detail=f"Invalid financial params: {error}")
        
        # Get latest forecast data
        import pandas as pd
        pred_path = config.DATA_DIR / "predictions_24h.csv"
        
        # Always generate fresh forecast for the specific location
        logger.info(f"Generating fresh forecast for optimization at ({lat}, {lon})...")
        
        from real_weather_forecast import RealWeatherSolarForecaster
        forecaster = RealWeatherSolarForecaster(
            system_size_kwp=system_size,
            efficiency=efficiency,
            panel_tilt=panel_tilt,
            panel_azimuth=panel_azimuth,
            performance_ratio=performance_ratio
        )
        result = forecaster.forecast(lat=lat, lon=lon, hours=48)
        
        if result['status'] == 'success' and result.get('hourly_24h'):
            hourly_data = result['hourly_24h']
            logger.info(f"✓ Generated {len(hourly_data)} hourly predictions for optimization")
        else:
            raise HTTPException(status_code=404, detail="Unable to generate forecast for optimization")
        
        # Create optimization agent with validated parameters
        optimizer = OptimizationAgent(
            battery_capacity_kwh=battery_capacity,
            electricity_tariff=electricity_tariff,
            feed_in_tariff=feed_in_tariff,
            system_size_kwp=system_size,
            grid_co2_factor=grid_co2_factor,
            max_grid_import_kw=max_grid_import
        )
        
        # Get optimization recommendations
        recommendations = optimizer.analyze_forecast(hourly_data)
        
        return {
            "status": "success",
            "recommendations": recommendations
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/locations/presets")
async def get_location_presets():
    """Get preset locations for quick selection"""
    return {
        "status": "success",
        "locations": [
            {"city": "Delhi (IN)", "lat": 28.6139, "lon": 77.2090, "timezone": "Asia/Kolkata"},
            {"city": "Mumbai (IN)", "lat": 19.0760, "lon": 72.8777, "timezone": "Asia/Kolkata"},
            {"city": "Bangalore (IN)", "lat": 12.9716, "lon": 77.5946, "timezone": "Asia/Kolkata"},
            {"city": "Chennai (IN)", "lat": 13.0827, "lon": 80.2707, "timezone": "Asia/Kolkata"},
            {"city": "New York (US)", "lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York"},
            {"city": "Los Angeles (US)", "lat": 34.0522, "lon": -118.2437, "timezone": "America/Los_Angeles"},
            {"city": "Chicago (US)", "lat": 41.8781, "lon": -87.6298, "timezone": "America/Chicago"},
            {"city": "London (UK)", "lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"},
            {"city": "Paris (FR)", "lat": 48.8566, "lon": 2.3522, "timezone": "Europe/Paris"},
            {"city": "Berlin (DE)", "lat": 52.5200, "lon": 13.4050, "timezone": "Europe/Berlin"},
            {"city": "Tokyo (JP)", "lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
            {"city": "Singapore (SG)", "lat": 1.3521, "lon": 103.8198, "timezone": "Asia/Singapore"},
            {"city": "Sydney (AU)", "lat": -33.8688, "lon": 151.2093, "timezone": "Australia/Sydney"},
            {"city": "Melbourne (AU)", "lat": -37.8136, "lon": 144.9631, "timezone": "Australia/Melbourne"},
        ]
    }


@app.get("/test/nasa-power")
async def test_nasa_power(lat: float = 13.0837, lon: float = 80.2702):
    """Test NASA POWER API integration"""
    try:
        from agents.realtime_data_agent import RealTimeDataAgent
        from datetime import datetime
        
        agent = RealTimeDataAgent(latitude=lat, longitude=lon)
        
        # Test at noon (when sun should be high)
        test_time = datetime.now().replace(hour=12, minute=0, second=0)
        
        # Fetch NASA POWER data
        nasa_data = agent.fetch_nasa_power_solar_data(lat, lon, test_time)
        
        if nasa_data:
            return {
                "status": "success",
                "message": "NASA POWER API is working",
                "data": nasa_data,
                "test_time": test_time.isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "NASA POWER API returned no data",
                "test_time": test_time.isoformat()
            }
    except Exception as e:
        logger.error(f"Error testing NASA POWER: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
