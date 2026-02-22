from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])

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

@router.post("/train")
async def train_ml_model(request: MLTrainRequest = MLTrainRequest()):
    """Train the ML solar forecasting model."""
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

@router.get("/status")
async def get_ml_model_status():
    """Get the status of the ML forecasting model."""
    try:
        from ml.unified_forecaster import HybridForecaster
        
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

@router.get("/models")
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

@router.post("/quick-train")
async def quick_train_model(
    lat: float = 28.6139,
    lon: float = 77.2090,
    days: int = 90,
    epochs: int = 50
):
    """Quick train an ML model with minimal configuration."""
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
