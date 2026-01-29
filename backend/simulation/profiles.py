"""
Simulation Profiles
Defines realistic usage patterns for different resources.
"""
import numpy as np

class LoadProfiles:
    """Pre-defined load profiles for realistic simulation."""
    
    @staticmethod
    def get_residential_profile(hour: float) -> float:
        """
        Returns a multiplier (0.0 to ~2.0) representing typical residential load intensity at a given hour.
        Based on standard 'Duck Curve' consumption patterns.
        """
        # Hour is 0.0 to 23.99
        
        # Base load (fridge, standby devices) - always running
        base = 0.3
        
        # Morning peak (6 AM - 9 AM) - showers, breakfast
        morning_peak = 1.2 * np.exp(-((hour - 7.5)**2) / (2 * 1.5**2))
        
        # Daytime valley (10 AM - 4 PM) - work/school away
        day_val = 0.4
        if 9 <= hour <= 17:
             # Smooth transition
             pass 
        
        # Evening peak (6 PM - 10 PM) - cooking, TV, lights
        evening_peak = 1.8 * np.exp(-((hour - 20)**2) / (2 * 2.0**2))
        
        return base + morning_peak + evening_peak

    @staticmethod
    def get_ev_profile(hour: float) -> bool:
        """
        Returns True if EV is likely connected at home.
        Typically connected Evening -> Morning (6 PM to 7 AM).
        """
        if hour >= 18 or hour <= 7:
            return True
        return False
