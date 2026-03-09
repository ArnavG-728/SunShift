import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Tuple

def calculate_solar_position(timestamp, lat: float, lon: float = 0.0) -> Tuple[float, float, float]:
    """
    Calculate sun's position accurately based on time and location.
    Handles timezone-naive local timestamps correctly.
    Returns: (elevation, azimuth, declination) in degrees.
    """
    # 1. Get true UTC time from the timestamp
    # IMPORTANT: pandas Timestamp.timestamp() treats tz-naive as UTC,
    # but datetime.timestamp() treats tz-naive as local time.
    # We must convert to plain datetime first for correct local->UTC conversion.
    if hasattr(timestamp, 'to_pydatetime'):
        timestamp = timestamp.to_pydatetime()
    unix_time = timestamp.timestamp()
    utc_dt = datetime.fromtimestamp(unix_time, tz=timezone.utc)
    
    # 2. Fractional UTC hour and day of year
    utc_hour = utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0
    day_of_year = utc_dt.timetuple().tm_yday
    
    # 3. Accurate Solar Declination
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365.0))
    
    # 4. Equation of Time (EoT) in minutes
    B = np.radians((day_of_year - 1) * 360.0 / 365.0)
    E = 229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B) - 0.014615 * np.cos(2*B) - 0.040849 * np.sin(2*B))
    
    # 5. True Solar Time / Local Solar Time (LST) in hours
    lst_hours = (utc_hour + lon / 15.0 + E / 60.0) % 24
    
    # 6. Hour angle (15 degrees per hour, 0 at solar noon)
    hour_angle = 15 * (lst_hours - 12)
    
    # Solar Elevation Angle
    elevation = np.degrees(np.arcsin(
        np.sin(np.radians(lat)) * np.sin(np.radians(declination)) +
        np.cos(np.radians(lat)) * np.cos(np.radians(declination)) * 
        np.cos(np.radians(hour_angle))
    ))
    
    # Solar Azimuth Angle
    azimuth = np.degrees(np.arctan2(
        np.sin(np.radians(hour_angle)),
        np.cos(np.radians(hour_angle)) * np.sin(np.radians(lat)) -
        np.tan(np.radians(declination)) * np.cos(np.radians(lat))
    ))
    azimuth = (azimuth + 180) % 360
    
    return elevation, azimuth, declination

def calculate_angle_of_incidence(sun_elevation: float, sun_azimuth: float, 
                                 panel_tilt: float, panel_azimuth: float) -> float:
    """
    Calculate Angle of Incidence (AOI) - the angle between sun's rays and panel normal.
    Using the standard astronomical formula.
    Returns AOI in degrees.
    """
    sun_elev_rad = np.radians(sun_elevation)
    sun_azim_rad = np.radians(sun_azimuth)
    panel_tilt_rad = np.radians(panel_tilt)
    panel_azim_rad = np.radians(panel_azimuth)
    
    cos_aoi = (
        np.sin(sun_elev_rad) * np.cos(panel_tilt_rad) +
        np.cos(sun_elev_rad) * np.sin(panel_tilt_rad) * 
        np.cos(sun_azim_rad - panel_azim_rad)
    )
    
    cos_aoi = np.clip(cos_aoi, -1, 1)
    aoi = np.degrees(np.arccos(cos_aoi))
    
    return aoi

def calculate_clear_sky_irradiance(sun_elevation: float) -> float:
    """
    Calculate clear sky Global Horizontal Irradiance (GHI) based on air mass.
    Returns Irradiance in W/m².
    """
    if sun_elevation <= 0:
        return 0.0
        
    # Air mass formula (Kasten & Young 1989)
    air_mass = 1 / (np.sin(np.radians(sun_elevation)) + 0.50572 * (sun_elevation + 6.07995)**-1.6364)
    solar_constant = 1367  # W/m²
    
    # Direct Normal Irradiance
    dni = solar_constant * (0.7 ** (air_mass ** 0.678))
    
    # Global Horizontal Irradiance (simplified model)
    ghi = dni * np.sin(np.radians(sun_elevation))
    
    return max(0, ghi)

def calculate_effective_irradiance(base_ghi: float, clouds: float, 
                                   sun_elevation: float, aoi: float) -> Dict[str, float]:
    """
    Calculates final effective irradiance on a tilted panel considering clouds.
    """
    # 1. Cloud Transmittance
    cloud_transmittance = 1 - (clouds / 100) * 0.75
    actual_ghi = base_ghi * cloud_transmittance
    
    # 2. Tilt Factor (based on Angle of Incidence)
    if aoi < 90:  # Direct sun hitting panel
        tilt_factor = np.cos(np.radians(aoi)) / max(0.01, np.sin(np.radians(sun_elevation)))
        tilt_factor = max(0.5, min(1.5, tilt_factor)) # Clamp to realistic bounds
    else:
        tilt_factor = 0.5 # Diffuse light only
        
    total_irradiance = actual_ghi * tilt_factor
    
    # 3. Direct/Diffuse split
    direct_fraction = 0.8 * cloud_transmittance
    diffuse_fraction = 1 - direct_fraction
    
    return {
        'total': max(0.0, total_irradiance),
        'direct': max(0.0, total_irradiance * direct_fraction),
        'diffuse': max(0.0, total_irradiance * diffuse_fraction)
    }

def calculate_energy_output(irradiance: float, temperature: float, 
                           system_size_kwp: float, performance_ratio: float) -> float:
    """Calculate expected kW output at a given moment considering temp derating."""
    if pd.isna(temperature) or pd.isna(irradiance):
        return 0.0
        
    temp_factor = 1 - 0.004 * (temperature - 25)
    temp_factor = max(0.7, min(1.0, temp_factor))
    
    # 1000 W/m² is standard test condition (STC) irradiance
    energy = (irradiance / 1000.0) * system_size_kwp * performance_ratio * temp_factor
    
    return max(0.0, energy)
