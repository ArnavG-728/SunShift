from fastapi import APIRouter, HTTPException
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["locations"])

@router.get("/locations/presets")
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

@router.get("/test/nasa-power")
async def test_nasa_power(lat: float = 13.0837, lon: float = 80.2707):
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
