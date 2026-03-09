from fastapi import APIRouter, HTTPException
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/currency", tags=["currency"])

@router.get("/rates")
async def get_currency_rates(base: str = "USD"):
    """Get real-time currency exchange rates"""
    try:
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
