"""
Configuration module for GreenCast backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Application
    APP_NAME = os.getenv("APP_NAME", "SunShift")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
    NREL_API_KEY = os.getenv("NREL_API_KEY", "DEMO_KEY")  # Get free key at https://developer.nrel.gov/signup/
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models" / "saved"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Model Settings
    MODEL_TYPE = os.getenv("MODEL_TYPE", "LSTM")
    FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "24"))
    SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "24"))
    TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", "0.8"))
    
    # Data Settings
    DATA_COLLECTION_DAYS = int(os.getenv("DATA_COLLECTION_DAYS", "30"))
    USE_SYNTHETIC_DATA = os.getenv("USE_SYNTHETIC_DATA", "True").lower() == "true"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./greencast.db")
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # OpenWeather API
    OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    # Model Parameters
    LSTM_PARAMS = {
        "units": int(os.getenv("LSTM_UNITS", "64")),
        "dropout": float(os.getenv("LSTM_DROPOUT", "0.2")),
        "epochs": int(os.getenv("LSTM_EPOCHS", "50")),
        "batch_size": int(os.getenv("LSTM_BATCH_SIZE", "32")),
        "sequence_length": int(os.getenv("SEQUENCE_LENGTH", "24"))
    }
    
    PROPHET_PARAMS = {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "daily_seasonality": True,
        "weekly_seasonality": True
    }

config = Config()
