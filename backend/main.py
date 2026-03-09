"""
FastAPI Backend for SunShift - Solar Energy Forecasting System
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config import config
from database import init_db

# Import routers
from routers import (
    forecast,
    metrics,
    realtime,
    optimize,
    pitch_features,
    chat,
    ml,
    appliances,
    currency,
    locations
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="SunShift - AI-Powered Solar Energy Forecasting & Analytics Platform"
)

@app.on_event("startup")
def on_startup():
    init_db()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecast.router)
app.include_router(metrics.router)
app.include_router(realtime.router)
app.include_router(optimize.router)
app.include_router(pitch_features.router)
app.include_router(chat.router)
app.include_router(ml.router)
app.include_router(appliances.router)
app.include_router(currency.router)
app.include_router(locations.router)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
