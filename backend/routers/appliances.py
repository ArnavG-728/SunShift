from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import json
import logging
from config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/appliances", tags=["appliances"])

class ApplianceItem(BaseModel):
    name: str
    consumption_kwh: float
    duration_hours: int

class AppliancesConfig(BaseModel):
    appliances: List[ApplianceItem]

@router.get("/")
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

@router.post("/")
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
