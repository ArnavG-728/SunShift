"""
Simulation State
Tracks the persistent state of the energy system (Battery, EV, Totals).
Supports per-location isolation for concurrent multi-user access.
"""
from datetime import datetime
from database import SessionLocal
from database.models import SimulationState as DBState
import logging

logger = logging.getLogger(__name__)


def generate_location_key(lat: float, lon: float) -> str:
    """Generate a normalized location key from coordinates (2 decimal precision)."""
    return f"{lat:.2f},{lon:.2f}"


class SystemState:
    def __init__(self, lat: float = 28.6139, lon: float = 77.2090):
        """Initialize state for a specific location."""
        self.location_key = generate_location_key(lat, lon)
        
        # In-memory mirror of state
        self.state = {
            "battery_soc": 50.0,
            "ev_soc": 60.0,
            "last_update": datetime.now().isoformat(),
            "cumulative_solar": 0.0,
            "cumulative_grid_import": 0.0,
            "cumulative_grid_export": 0.0
        }
        self.load()

    def load(self):
        """Load state from SQLite DB by location_key"""
        db = SessionLocal()
        try:
            # Query by location_key for per-location isolation
            db_state = db.query(DBState).filter(DBState.location_key == self.location_key).first()
            
            # Fallback: check for legacy id=1 state if no location-specific state exists
            if not db_state:
                legacy_state = db.query(DBState).filter(DBState.id == 1, DBState.location_key == None).first()
                if legacy_state:
                    logger.info(f"Migrating legacy state to location {self.location_key}")
                    db_state = legacy_state
            
            if db_state:
                self.state["battery_soc"] = db_state.battery_soc
                self.state["ev_soc"] = db_state.ev_soc
                if db_state.updated_at:
                    self.state["last_update"] = db_state.updated_at.isoformat()
                self.state["cumulative_solar"] = db_state.total_solar_kwh
                self.state["cumulative_grid_import"] = db_state.total_grid_import_kwh
                self.state["cumulative_grid_export"] = db_state.total_grid_export_kwh
                logger.debug(f"Loaded simulation state for location {self.location_key}")
            else:
                logger.info(f"No DB state found for {self.location_key}, initializing new state")
                self.save()  # Create initial record
                
        except Exception as e:
            logger.error(f"Failed to load state from DB: {e}")
        finally:
            db.close()

    def save(self):
        """Save current state to SQLite DB with location isolation"""
        db = SessionLocal()
        try:
            db_state = db.query(DBState).filter(DBState.location_key == self.location_key).first()
            
            if not db_state:
                db_state = DBState(location_key=self.location_key)
                db.add(db_state)
            
            # Update fields
            db_state.battery_soc = self.state["battery_soc"]
            db_state.ev_soc = self.state["ev_soc"]
            db_state.updated_at = datetime.fromisoformat(self.state["last_update"])
            db_state.total_solar_kwh = self.state["cumulative_solar"]
            db_state.total_grid_import_kwh = self.state["cumulative_grid_import"]
            db_state.total_grid_export_kwh = self.state["cumulative_grid_export"]
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to save state to DB: {e}")
            db.rollback()
        finally:
            db.close()

    def update_time(self, current_time: datetime) -> float:
        """
        Updates last_update and returns time delta in hours.
        """
        try:
            last_time = datetime.fromisoformat(self.state["last_update"])
            delta_hours = (current_time - last_time).total_seconds() / 3600.0
            
            # Guard against huge jumps (e.g. server restart after days)
            if delta_hours > 24 or delta_hours < 0: 
                delta_hours = 0.01 
        except:
            delta_hours = 0.01
            
        self.state["last_update"] = current_time.isoformat()
        return max(0.0, delta_hours)
    
    @property
    def battery_soc(self):
        return self.state["battery_soc"]
    
    @battery_soc.setter
    def battery_soc(self, value):
        self.state["battery_soc"] = max(0.0, min(100.0, value))

    @property
    def ev_soc(self):
        return self.state["ev_soc"]

    @ev_soc.setter
    def ev_soc(self, value):
        self.state["ev_soc"] = max(0.0, min(100.0, value))

