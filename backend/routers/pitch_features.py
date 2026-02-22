from fastapi import APIRouter, HTTPException
import logging
from config import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pitch_features"])

@router.get("/risk/analysis")
async def get_risk_analysis(lat: float = 28.6139, lon: float = 77.2090, battery_soc: float = 50.0):
    """Get solar installation risk analysis with defensive charging triggers"""
    try:
        from agents.realtime_data_agent import RealTimeDataAgent
        from agents.risk_agent import SolarRiskAgent
        
        weather_agent = RealTimeDataAgent(latitude=lat, longitude=lon)
        risk_agent = SolarRiskAgent()
        
        current_weather = weather_agent.fetch_current_weather(lat=lat, lon=lon)
        if not current_weather:
            raise HTTPException(status_code=503, detail="Weather data unavailable for risk analysis")
            
        risk_data = risk_agent.calculate_risk_score(current_weather)
        
        # Defensive Battery Charging — scan 12h forecast for storms
        defensive_charging = None
        try:
            forecast = weather_agent.fetch_forecast(hours=12)
            if forecast:
                forecast_dicts = [
                    {
                        "temperature": f.get("temperature", 25),
                        "wind_speed": f.get("wind_speed", 0),
                        "clouds": f.get("clouds", 0),
                        "weather": f.get("weather", ""),
                        "pop": f.get("pop", 0),
                        "visibility": f.get("visibility", 10000),
                        "humidity": f.get("humidity", 50),
                    }
                    for f in forecast
                ]
                defensive_charging = risk_agent.generate_defensive_triggers(
                    forecast_dicts, battery_soc=battery_soc
                )
        except Exception as def_err:
            logger.warning(f"Defensive charging check failed: {def_err}")
            defensive_charging = {"trigger": False, "reason": "Unable to fetch forecast"}
        
        risk_data["defensive_charging"] = defensive_charging
        
        return {
            "status": "success",
            "risk_analysis": risk_data
        }
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/value-gap")
async def get_value_gap(
    lat: float = 28.6139,
    lon: float = 77.2090,
    system_size: float = 5.0,
    electricity_tariff: float = 0.15,
    feed_in_tariff: float = 0.05,
):
    """Calculate the Solar Value Gap and Virtual Battery savings"""
    try:
        from agents.value_gap_agent import ValueGapAgent
        from real_weather_forecast import RealWeatherSolarForecaster

        # Generate 24h forecast for value gap analysis
        forecaster = RealWeatherSolarForecaster(system_size_kwp=system_size)
        result = forecaster.forecast(lat=lat, lon=lon, hours=48)

        if result["status"] != "success" or not result.get("hourly_24h"):
            raise HTTPException(status_code=404, detail="Unable to generate forecast for value gap analysis")

        agent = ValueGapAgent(
            electricity_tariff=electricity_tariff,
            feed_in_tariff=feed_in_tariff,
            system_size_kwp=system_size,
        )
        analysis = agent.analyze(result["hourly_24h"])

        return {"status": "success", "data": analysis}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in value gap analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-health")
async def get_system_health(
    lat: float = 28.6139,
    lon: float = 77.2090,
    system_size: float = 5.0,
    panel_age_years: float = 0,
):
    """Get panel degradation analysis and health score"""
    try:
        from agents.degradation_agent import DegradationAgent
        from real_weather_forecast import RealWeatherSolarForecaster

        # Generate 24h forecast with irradiance data
        forecaster = RealWeatherSolarForecaster(system_size_kwp=system_size)
        result = forecaster.forecast(lat=lat, lon=lon, hours=48)

        if result["status"] != "success" or not result.get("hourly_24h"):
            raise HTTPException(status_code=404, detail="Unable to generate forecast for health analysis")

        agent = DegradationAgent(
            system_size_kwp=system_size,
            panel_age_years=panel_age_years,
        )
        health = agent.analyze_health(
            forecast_data=result["hourly_24h"],
            system_config={"system_size": system_size, "panel_age_years": panel_age_years},
        )

        return {"status": "success", "data": health}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in system health analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/carbon-wallet")
async def get_carbon_wallet(
    lat: float = 28.6139,
    lon: float = 77.2090,
    grid_co2_factor: float = 0.70,
):
    """Get the Carbon Credit Wallet summary and record today's credits"""
    try:
        from agents.carbon_wallet_agent import CarbonWalletAgent

        agent = CarbonWalletAgent()

        # Try to record today's generation (only once per day per location)
        try:
            location_key = f"{lat:.2f},{lon:.2f}"
            if not agent.has_recorded_today(location_key):
                import pandas as pd
                pred_path = config.DATA_DIR / "predictions_24h.csv"
                if pred_path.exists():
                    df = pd.read_csv(pred_path)
                    energy_col = (
                        "predicted_output_kWh"
                        if "predicted_output_kWh" in df.columns
                        else "energy_output_kWh"
                    )
                    if energy_col in df.columns:
                        total_energy = max(0, float(df[energy_col].sum()))
                        if total_energy > 0:
                            agent.record_credits(
                                energy_kwh=total_energy,
                                grid_co2_factor=grid_co2_factor,
                                latitude=lat,
                                longitude=lon,
                            )
        except Exception as rec_err:
            logger.warning(f"Carbon wallet recording failed: {rec_err}")

        wallet = agent.get_wallet(latitude=lat, longitude=lon)

        return {"status": "success", "data": wallet}
    except Exception as e:
        logger.error(f"Error in carbon wallet: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
