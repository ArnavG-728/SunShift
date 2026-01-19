"""
FastAPI Backend for SunShift - Solar Energy Forecasting System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
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
        
        # Persist results to CSV files for other endpoints/clients
        try:
            import pandas as pd
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
        import numpy as np
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


@app.get("/forecast/24h")
async def get_24h_forecast():
    """Get 24-hour hourly forecast"""
    try:
        import pandas as pd
        pred_path = config.DATA_DIR / "predictions_24h.csv"
        
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail="No 24h forecast available")
        
        df = pd.read_csv(pred_path)
        import numpy as np
        df = df.replace({np.nan: None})
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
        import numpy as np
        df = df.replace({np.nan: None})
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
        
        # Calculate cloud loss
        cloud_loss = agent.calculate_cloud_loss(
            local_time, 
            current["clouds"],
            lat=lat,
            lon=lon,
            system_size_kwp=system_size,
            performance_ratio=performance_ratio
        )
        
        # Get sunrise/sunset times
        sunrise = current.get("sunrise")
        sunset = current.get("sunset")
        
        # Calculate daylight hours
        daylight_hours = 0
        if sunrise and sunset:
            daylight_hours = (sunset - sunrise).total_seconds() / 3600
        
        # Is it currently daytime?
        is_daytime = False
        if sunrise and sunset:
            is_daytime = sunrise <= local_time <= sunset
        
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
                "cloud_loss": cloud_loss,
                "weather": current["weather"],
                "description": current["description"],
                # Sun times
                "sunrise": sunrise.strftime("%H:%M") if sunrise else None,
                "sunset": sunset.strftime("%H:%M") if sunset else None,
                "daylight_hours": round(daylight_hours, 1),
                "is_daytime": is_daytime
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



@app.get("/risk/analysis")
async def get_risk_analysis(lat: float = 28.6139, lon: float = 77.2090):
    """Get solar installation risk analysis"""
    try:
        from agents.realtime_data_agent import RealTimeDataAgent
        from agents.risk_agent import SolarRiskAgent
        
        weather_agent = RealTimeDataAgent(latitude=lat, longitude=lon)
        risk_agent = SolarRiskAgent()
        
        current_weather = weather_agent.fetch_current_weather(lat=lat, lon=lon)
        if not current_weather:
            raise HTTPException(status_code=503, detail="Weather data unavailable for risk analysis")
            
        risk_data = risk_agent.calculate_risk_score(current_weather)
        return {
            "status": "success",
            "risk_analysis": risk_data
        }
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usage/unified")
async def get_unified_usage(
    lat: float = 28.6139, 
    lon: float = 77.2090,
    system_size: float = 5.0,
    has_battery: bool = False,
    battery_capacity: float = 0.0
):
    """Get unified energy truth (Solar + Grid + Battery + EV + Gas + Water)"""
    try:
        import random
        from datetime import datetime
        
        # Try to get real solar generation from weather data
        solar_gen_kw = 0.0
        try:
            from agents.realtime_data_agent import RealTimeDataAgent
            realtime_agent = RealTimeDataAgent(latitude=lat, longitude=lon)
            weather_data = realtime_agent.fetch_current_weather(lat, lon)
            if weather_data:
                clouds = weather_data.get('clouds', 50)
                temp = weather_data.get('temperature', 25)
                
                # Calculate solar using real weather
                solar_irradiance = realtime_agent.calculate_solar_irradiance(datetime.now(), clouds, lat, lon)
                
                # Temperature derating
                temp_factor = 1 - 0.004 * max(0, temp - 25)
                temp_factor = max(0.7, min(1.0, temp_factor))
                
                # Calculate actual output
                performance_ratio = 0.78
                solar_gen_kw = (solar_irradiance / 1000) * system_size * performance_ratio * temp_factor
        except Exception as e:
            logger.warning(f"Could not get real solar data: {e}, using estimate")
            # Fallback: estimate based on time of day
            hour = datetime.now().hour
            if 6 <= hour <= 18:
                solar_gen_kw = system_size * 0.6 * (1 - abs(hour - 12) / 12)
            else:
                solar_gen_kw = 0.0
        
        # Simulate house load (typical residential pattern)
        hour = datetime.now().hour
        base_load = 1.5  # Base load
        if 7 <= hour <= 9 or 18 <= hour <= 22:
            house_load_kw = base_load + random.uniform(1, 2.5)  # Morning/evening peaks
        elif 9 <= hour <= 17:
            house_load_kw = base_load + random.uniform(0.5, 1.5)  # Daytime
        else:
            house_load_kw = base_load + random.uniform(0, 0.5)  # Night
        
        # Calculate grid import/export
        net_flow = solar_gen_kw - house_load_kw
        grid_import_kw = max(0, -net_flow)
        
        # Battery state (if user has one)
        battery_soc = 0.0
        if has_battery and battery_capacity > 0:
            # Simulate battery charging during excess solar
            battery_soc = min(100, 50 + (net_flow * 10))  # Simplified
            battery_soc = max(10, battery_soc)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "location": {"lat": lat, "lon": lon},
            "system_config": {
                "system_size_kwp": system_size,
                "has_battery": has_battery,
                "battery_capacity_kwh": battery_capacity
            },
            "metrics": {
                "electricity": {
                    "solar_gen_kw": round(max(0, solar_gen_kw), 2),
                    "grid_import_kw": round(grid_import_kw, 2),
                    "battery_soc_percent": round(battery_soc, 1),
                    "house_load_kw": round(house_load_kw, 2),
                    "net_flow_kw": round(net_flow, 2),
                    "is_exporting": net_flow > 0
                },
                "transport": {
                    "ev_charge_percent": round(random.uniform(40, 90), 1),
                    "ev_range_km": round(random.uniform(150, 400), 1),
                    "charging_status": random.choice(["Disconnected", "Charging", "Standby"])
                },
                "other_resources": {
                    "gas_usage_m3": round(random.uniform(0.1, 0.5), 3),
                    "water_usage_liters": round(random.uniform(5, 50), 1),
                    "water_leak_alert": False
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in unified usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/currency/rates")
async def get_currency_rates(base: str = "USD"):
    """Get real-time currency exchange rates"""
    try:
        import requests
        from datetime import datetime
        
        # Use free exchangerate API (no key required for limited use)
        url = f"https://api.exchangerate-api.com/v4/latest/{base}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "status": "success",
                "base": base,
                "timestamp": datetime.now().isoformat(),
                "rates": {
                    "USD": data["rates"].get("USD", 1.0),
                    "EUR": data["rates"].get("EUR", 0.85),
                    "GBP": data["rates"].get("GBP", 0.73),
                    "INR": data["rates"].get("INR", 83.0),
                    "AUD": data["rates"].get("AUD", 1.52),
                    "CAD": data["rates"].get("CAD", 1.36),
                    "JPY": data["rates"].get("JPY", 149.0),
                    "CNY": data["rates"].get("CNY", 7.2)
                }
            }
        except:
            # Fallback to approximate rates if API fails
            logger.warning("Currency API failed, using fallback rates")
            return {
                "status": "fallback",
                "base": base,
                "timestamp": datetime.now().isoformat(),
                "rates": {
                    "USD": 1.0,
                    "EUR": 0.92,
                    "GBP": 0.79,
                    "INR": 83.0,
                    "AUD": 1.54,
                    "CAD": 1.36,
                    "JPY": 149.0,
                    "CNY": 7.14
                }
            }
    except Exception as e:
        logger.error(f"Error getting currency rates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


import json
from pathlib import Path

class ApplianceItem(BaseModel):
    name: str
    consumption_kwh: float
    duration_hours: int

class AppliancesConfig(BaseModel):
    appliances: List[ApplianceItem]

@app.get("/appliances")
async def get_appliances():
    """Get the list of appliances used for optimization (always returns a flat list)"""
    try:
        path = config.DATA_DIR / "appliances.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                # Migration: if it's the old category-based format, flatten it
                if isinstance(data, dict):
                    flat_list = []
                    for key in ["high", "medium", "flexible"]:
                        if key in data and isinstance(data[key], list):
                            flat_list.extend(data[key])
                    # If it was some other dict format, try to extract all lists
                    if not flat_list:
                        for val in data.values():
                            if isinstance(val, list):
                                flat_list.extend(val)
                    return flat_list
                return data
        else:
            # Return default flat list
            return [
                {"name": "EV Charging", "consumption_kwh": 7.0, "duration_hours": 4},
                {"name": "Water Heater", "consumption_kwh": 4.0, "duration_hours": 2},
                {"name": "Clothes Dryer", "consumption_kwh": 3.0, "duration_hours": 1},
                {"name": "Dishwasher", "consumption_kwh": 1.8, "duration_hours": 2},
                {"name": "Washing Machine", "consumption_kwh": 1.5, "duration_hours": 1},
                {"name": "Pool Pump", "consumption_kwh": 1.2, "duration_hours": 3},
                {"name": "Device Charging", "consumption_kwh": 0.5, "duration_hours": 2},
                {"name": "Vacuum Cleaner", "consumption_kwh": 0.8, "duration_hours": 1}
            ]
    except Exception as e:
        logger.error(f"Error getting appliances: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/appliances")
async def update_appliances(appliances: List[ApplianceItem]):
    """Update the list of appliances (accepts flat list)"""
    try:
        path = config.DATA_DIR / "appliances.json"
        with open(path, "w") as f:
            json.dump([item.dict() for item in appliances], f, indent=4)
        return {"status": "success", "message": "Appliances updated successfully"}
    except Exception as e:
        logger.error(f"Error updating appliances: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ML Model Training & Hybrid Forecasting Endpoints
# ==============================================================================

class MLTrainRequest(BaseModel):
    """ML Model Training Request"""
    latitude: Optional[float] = 28.6139
    longitude: Optional[float] = 77.2090
    days: Optional[int] = 365
    epochs: Optional[int] = 100
    batch_size: Optional[int] = 32
    sequence_length: Optional[int] = 24
    model_name: Optional[str] = "solar_forecaster"
    force_data_refresh: Optional[bool] = False


@app.post("/ml/train")
async def train_ml_model(request: MLTrainRequest = MLTrainRequest()):
    """
    Train the ML solar forecasting model.
    
    This endpoint triggers the complete ML training pipeline:
    1. Collects historical solar/weather data
    2. Trains the LSTM model
    3. Evaluates and saves the model
    
    Note: Training may take several minutes depending on data size and epochs.
    """
    try:
        logger.info(f"Starting ML model training for ({request.latitude}, {request.longitude})...")
        
        from ml.trainer import ModelTrainer
        
        trainer = ModelTrainer()
        result = trainer.train_model(
            lat=request.latitude,
            lon=request.longitude,
            days=request.days,
            epochs=request.epochs,
            batch_size=request.batch_size,
            sequence_length=request.sequence_length,
            model_name=request.model_name,
            force_data_refresh=request.force_data_refresh
        )
        
        if result['status'] == 'success':
            return {
                "status": "success",
                "message": "ML model trained successfully",
                "model_name": request.model_name,
                "training_samples": result['data_collection']['samples'],
                "metrics": {
                    "mae": result['training']['mae'],
                    "rmse": result['training']['rmse'],
                    "mape": result['training']['mape']
                },
                "model_path": result['model']['path']
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Training failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error training ML model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/status")
async def get_ml_model_status():
    """Get the status of the ML forecasting model."""
    try:
        from ml.hybrid_forecaster import HybridForecaster
        
        forecaster = HybridForecaster()
        status = forecaster.get_model_status()
        
        return {
            "status": "success",
            "ml_model": status
        }
    except Exception as e:
        logger.error(f"Error getting ML status: {e}")
        return {
            "status": "error",
            "ml_model": {
                "ml_available": False,
                "error": str(e)
            }
        }


@app.get("/ml/models")
async def list_ml_models():
    """List all available trained ML models."""
    try:
        from ml.trainer import ModelTrainer
        
        trainer = ModelTrainer()
        models = trainer.get_available_models()
        
        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class HybridForecastRequest(BaseModel):
    """Hybrid ML+Physics Forecast Request"""
    latitude: Optional[float] = 28.6139
    longitude: Optional[float] = 77.2090
    hours: Optional[int] = 168
    system_size: Optional[float] = 5.0
    performance_ratio: Optional[float] = 0.78
    panel_tilt: Optional[float] = 30.0
    panel_azimuth: Optional[float] = 180.0


@app.post("/forecast/hybrid")
async def run_hybrid_forecast(request: HybridForecastRequest = HybridForecastRequest()):
    """
    Run hybrid ML + Physics-based forecast.
    
    This endpoint uses the trained ML model combined with physics-based validation
    for more accurate predictions. If no ML model is available, it falls back to
    physics-only forecasting.
    """
    try:
        from ml.hybrid_forecaster import HybridForecaster
        
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


@app.post("/ml/quick-train")
async def quick_train_model(
    lat: float = 28.6139,
    lon: float = 77.2090,
    days: int = 90,
    epochs: int = 50
):
    """
    Quick train an ML model with minimal configuration.
    
    Uses sensible defaults for quick experimentation:
    - 90 days of training data
    - 50 epochs
    - Default model architecture
    """
    try:
        from ml.trainer import quick_train
        
        logger.info(f"Quick training ML model for ({lat}, {lon})...")
        
        result = quick_train(lat=lat, lon=lon, days=days, epochs=epochs)
        
        if result['status'] == 'success':
            return {
                "status": "success",
                "message": "Model trained successfully",
                "metrics": {
                    "mae": result['training']['mae'],
                    "rmse": result['training']['rmse'],
                    "mape": result['training']['mape']
                }
            }
        else:
            return {
                "status": "failed",
                "error": result.get('error', 'Unknown error')
            }
            
    except Exception as e:
        logger.error(f"Error in quick training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

