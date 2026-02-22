from fastapi import APIRouter, HTTPException
import logging
from typing import Optional
from datetime import datetime
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/realtime", tags=["realtime"])

@router.get("/current")
async def get_current_weather(
    lat: float = 28.6139, 
    lon: float = 77.2090, 
    system_size: float = 5.0, 
    performance_ratio: float = 0.78,
    panel_tilt: float = 30.0,
    panel_azimuth: float = 180.0
):
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
            lon=lon,
            system_size=system_size,
            panel_tilt=panel_tilt,
            panel_azimuth=panel_azimuth
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
            performance_ratio=performance_ratio,
            panel_tilt=panel_tilt,
            panel_azimuth=panel_azimuth
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
                "sunrise": sunrise.strftime("%H:%M") if sunrise else None,
                "sunset": sunset.strftime("%H:%M") if sunset else None,
                "daylight_hours": round(daylight_hours, 1),
                "is_daytime": is_daytime
            }
        }
        
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast")
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

@router.get("/status")
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
