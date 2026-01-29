from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Boolean
from .connection import Base
from datetime import datetime

class SimulationState(Base):
    """Stores the persistent state of the simulation per-location for concurrent multi-user support"""
    __tablename__ = "simulation_state"
    
    id = Column(Integer, primary_key=True, index=True)
    # Location key for per-location isolation (format: "lat,lon" rounded to 2 decimals)
    location_key = Column(String, index=True, unique=True, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # System Components
    battery_soc = Column(Float, default=0.0)
    battery_capacity = Column(Float, default=13.5)
    ev_soc = Column(Float, default=50.0)
    ev_connected = Column(Boolean, default=True)
    
    # Cumulative Totals
    total_solar_kwh = Column(Float, default=0.0)
    total_load_kwh = Column(Float, default=0.0)
    total_grid_import_kwh = Column(Float, default=0.0)
    total_grid_export_kwh = Column(Float, default=0.0)
    
    def to_dict(self):
        return {
            "battery_soc": self.battery_soc,
            "battery_capacity": self.battery_capacity,
            "ev_soc": self.ev_soc,
            "ev_connected": self.ev_connected,
            "totals": {
                "solar_kwh": self.total_solar_kwh,
                "load_kwh": self.total_load_kwh,
                "grid_import_kwh": self.total_grid_import_kwh,
                "grid_export_kwh": self.total_grid_export_kwh
            },
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Forecast(Base):
    """Stores historical forecasts for analysis"""
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    location_lat = Column(Float)
    location_lon = Column(Float)
    
    # 'solar' | 'load' | 'price'
    forecast_type = Column(String) 
    
    # Metadata
    horizon_hours = Column(Integer)
    
    # The actual data points (time -> value)
    data_json = Column(JSON) 
