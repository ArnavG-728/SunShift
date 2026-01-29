"""
Simulation Engine
Coordinates Solar, Load, Battery, and Grid logic to produce realistic real-time values.
Supports per-location isolation for concurrent multi-user access.
"""
from datetime import datetime
import random
from .state import SystemState
from .profiles import LoadProfiles

class SimulationEngine:
    def __init__(self, lat: float = 28.6139, lon: float = 77.2090):
        """Create a simulation engine for a specific location."""
        self.state = SystemState(lat=lat, lon=lon)
        self.profiles = LoadProfiles()

    def tick(self, 
             solar_gen_kw: float, 
             has_battery: bool, 
             battery_capacity: float,
             system_size: float) -> dict:
        """
        Run one simulation step based on current time and inputs.
        Returns a dict of instantaneous metrics.
        """
        now = datetime.now()
        hour_float = now.hour + (now.minute / 60.0)
        
        # 1. Calculate Time Delta
        dt_hours = self.state.update_time(now)
        
        # 2. Calculate House Load (Realistic Profile + Small Random Noise)
        profile_factor = self.profiles.get_residential_profile(hour_float)
        # Assuming avg base system load scales with 'system size' implies a larger house
        # Heuristic: 5kW system -> ~1.5kW avg load.
        avg_house_load = system_size * 0.3 
        house_load_kw = (avg_house_load * profile_factor) + random.uniform(-0.1, 0.1)
        house_load_kw = max(0.1, house_load_kw) # Min load
        
        # 3. Calculate EV Load (if connected)
        ev_connected = self.profiles.get_ev_profile(hour_float)
        ev_charging_kw = 0.0
        
        # EV Logic: If connected and not full, charge at 7kW or top up
        if ev_connected and self.state.ev_soc < 90.0:
            ev_charging_kw = 7.0 # Standard Level 2 charger
            
            # Simulate EV battery gain
            # Assume 60kWh EV battery
            ev_energy_added = ev_charging_kw * dt_hours
            soc_added = (ev_energy_added / 60.0) * 100
            self.state.ev_soc += soc_added
        
        # If EV disconnected (driving), simulate drain
        if not ev_connected:
            # Drain 5% per hour driving
            self.state.ev_soc -= (5.0 * dt_hours) 

        # 4. Energy Balance
        total_load_kw = house_load_kw + ev_charging_kw
        net_flow_kw = solar_gen_kw - total_load_kw
        
        battery_power_kw = 0.0
        grid_exchange_kw = 0.0
        
        # 5. Battery Logic (Self-Consumption Optimization)
        if has_battery and battery_capacity > 0:
            if net_flow_kw > 0.0:
                # Excess Solar -> Charge Battery
                max_charge_rate = 5.0 # kW
                charge_power = min(net_flow_kw, max_charge_rate)
                
                # Check capacity limit
                current_kwh = (self.state.battery_soc / 100.0) * battery_capacity
                space_kwh = battery_capacity - current_kwh
                
                # Max energy we can put in during this step
                potential_energy = charge_power * dt_hours
                actual_energy = min(potential_energy, space_kwh)
                
                if dt_hours > 0:
                     actual_power = actual_energy / dt_hours
                else:
                     actual_power = charge_power
                     
                battery_power_kw = -actual_power # Negative = Charging
                
                # Update SOC
                soc_gain = (actual_energy / battery_capacity) * 100
                self.state.battery_soc += soc_gain
                
                # Remainder goes to grid
                grid_exchange_kw = -(net_flow_kw - actual_power) # Negative = Export
                
            else:
                # Deficit -> Discharge Battery
                deficit = -net_flow_kw
                max_discharge_rate = 5.0
                needed_power = min(deficit, max_discharge_rate)
                
                # Check available energy
                current_kwh = (self.state.battery_soc / 100.0) * battery_capacity
                
                potential_energy = needed_power * dt_hours
                actual_energy = min(potential_energy, current_kwh)
                
                if dt_hours > 0:
                     actual_power = actual_energy / dt_hours
                else:
                     actual_power = needed_power
                     
                battery_power_kw = actual_power # Positive = Discharging

                # Update SOC
                soc_loss = (actual_energy / battery_capacity) * 100
                self.state.battery_soc -= soc_loss
                
                # Remainder comes from grid
                grid_exchange_kw = (deficit - actual_power) # Positive = Import
        else:
            # No battery, direct grid exchange
            if net_flow_kw > 0:
                grid_exchange_kw = -net_flow_kw # Export
            else:
                grid_exchange_kw = -net_flow_kw # Import

        # 6. Save State
        self.state.save()
        
        # 7. Return Metrics
        return {
            "solar_gen_kw": max(0.0, solar_gen_kw),
            "house_load_kw": house_load_kw,
            "ev_charging_kw": ev_charging_kw,
            "total_load_kw": total_load_kw,
            "battery_power_kw": battery_power_kw, # +Discharge, -Charge
            "battery_soc": self.state.battery_soc,
            "grid_exchange_kw": grid_exchange_kw, # +Import, -Export
            "ev_soc": self.state.ev_soc,
            "ev_connected": ev_connected
        }
