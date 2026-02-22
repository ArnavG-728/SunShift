from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import pandas as pd
from datetime import datetime
import random

from config import config
from utils.validators import validate_coordinates, validate_battery_config, validate_financial_params

logger = logging.getLogger(__name__)

router = APIRouter(tags=["optimization"])

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

@router.post("/optimize")
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

@router.get("/usage/unified")
async def get_unified_usage(
    lat: float = 28.6139, 
    lon: float = 77.2090,
    system_size: float = 5.0,
    has_battery: bool = False,
    battery_capacity: float = 0.0,
    performance_ratio: float = 0.78,
    panel_tilt: float = 30.0,
    panel_azimuth: float = 180.0
):
    """Get unified energy truth (Solar + Grid + Battery + EV + Gas + Water) - USING SIMULATION ENGINE"""
    try:
        from simulation.engine import SimulationEngine
        from utils.cache import weather_cache, location_key
        
        # Create location-specific simulation engine (per-request, not global)
        sim_engine = SimulationEngine(lat=lat, lon=lon)
        
        # 1. Get Real Solar Input (with caching)
        solar_gen_kw = 0.0
        cache_key = location_key(lat, lon)
        
        # Check cache first for weather data
        cached_weather = weather_cache.get(cache_key)
        
        if cached_weather:
            weather_data = cached_weather
            logger.debug(f"Using cached weather for {cache_key}")
        else:
            try:
                from agents.realtime_data_agent import RealTimeDataAgent
                realtime_agent = RealTimeDataAgent(latitude=lat, longitude=lon)
                weather_data = realtime_agent.fetch_current_weather(lat, lon)
                
                if weather_data:
                    # Cache the weather data for 5 minutes
                    weather_cache.set(cache_key, weather_data, ttl=300)
            except Exception as e:
                logger.warning(f"Could not get real solar data: {e}, using estimate")
                weather_data = None
        
        if weather_data:
            clouds = weather_data.get('clouds', 50)
            temp = weather_data.get('temperature', 25)
            
            # Calculate solar using real weather and accurate physics
            from agents.realtime_data_agent import RealTimeDataAgent
            realtime_agent = RealTimeDataAgent(latitude=lat, longitude=lon)
            solar_irradiance = realtime_agent.calculate_solar_irradiance(
                datetime.now(), 
                clouds, 
                lat, 
                lon,
                system_size=system_size,
                panel_tilt=panel_tilt,
                panel_azimuth=panel_azimuth
            )
            
            # Temperature derating
            temp_factor = 1 - 0.004 * max(0, temp - 25)
            temp_factor = max(0.7, min(1.0, temp_factor))
            
            # Calculate actual output dynamically instead of hardcoding efficiency
            solar_gen_kw = (solar_irradiance / 1000) * system_size * performance_ratio * temp_factor

        # 2. Run Simulation Step (isolated per location)
        sim_metrics = sim_engine.tick(
            solar_gen_kw=solar_gen_kw,
            has_battery=has_battery,
            battery_capacity=battery_capacity,
            system_size=system_size
        )
        
        # 3. Format Response
        grid_import_kw = max(0, sim_metrics["grid_exchange_kw"])
        
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
                    "solar_gen_kw": float(round(sim_metrics["solar_gen_kw"], 2)),
                    "grid_import_kw": float(round(grid_import_kw, 2)),
                    "battery_soc_percent": float(round(sim_metrics["battery_soc"], 1)),
                    "house_load_kw": float(round(sim_metrics["house_load_kw"], 2)),
                    "net_flow_kw": float(round(-sim_metrics["grid_exchange_kw"], 2)),
                    "is_exporting": bool(sim_metrics["grid_exchange_kw"] < 0),
                    "battery_power_kw": float(round(sim_metrics["battery_power_kw"], 2)),
                    "ev_charging_kw": float(round(sim_metrics["ev_charging_kw"], 2))
                },
                "transport": {
                    "ev_charge_percent": float(round(sim_metrics["ev_soc"], 1)),
                    "ev_range_km": float(round(sim_metrics["ev_soc"] * 4.0, 1)),
                    "charging_status": "Charging" if sim_metrics["ev_charging_kw"] > 0 else ("Connected" if sim_metrics["ev_connected"] else "Disconnected")
                },
                "other_resources": {
                    "gas_usage_m3": float(round(random.uniform(0.1, 0.15), 3)),
                    "water_usage_liters": float(round(random.uniform(5, 12), 1)),
                    "water_leak_alert": False
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in unified usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))
