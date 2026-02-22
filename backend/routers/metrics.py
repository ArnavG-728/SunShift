from fastapi import APIRouter, HTTPException
import logging
from config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/")
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
