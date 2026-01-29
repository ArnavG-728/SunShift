"""
Simulation State
Tracks the persistent state of the energy system (Battery, EV, Totals).
Migrated from JSON to SQLite for robustness.
"""
from datetime import datetime
from database import SessionLocal
from database.models import SimulationState as DBState
import logging

logger = logging.getLogger(__name__)

class SystemState:
    def __init__(self):
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
        """Load state from SQLite DB"""
        db = SessionLocal()
        try:
            # We use ID=1 for the singleton simulation state
            db_state = db.query(DBState).filter(DBState.id == 1).first()
            
            if db_state:
                self.state["battery_soc"] = db_state.battery_soc
                self.state["ev_soc"] = db_state.ev_soc
                if db_state.updated_at:
                    self.state["last_update"] = db_state.updated_at.isoformat()
                self.state["cumulative_solar"] = db_state.total_solar_kwh
                self.state["cumulative_grid_import"] = db_state.total_grid_import_kwh
                self.state["cumulative_grid_export"] = db_state.total_grid_export_kwh
                logger.debug("Loaded simulation state from DB")
            else:
                logger.info("No DB state found, initializing new state")
                self.save() # Create initial record
                
        except Exception as e:
            logger.error(f"Failed to load state from DB: {e}")
        finally:
            db.close()

    def save(self):
        """Save current state to SQLite DB"""
        db = SessionLocal()
        try:
            db_state = db.query(DBState).filter(DBState.id == 1).first()
            
            if not db_state:
                db_state = DBState(id=1)
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
