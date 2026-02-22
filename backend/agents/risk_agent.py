"""
Solar Risk Agent - Evaluates environmental and performance risks for solar installations
Inspired by OpenWeather Risk Assessment System
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SolarRiskAgent:
    """
    Analyzes weather data and system parameters to generate a 0-100 Risk Score.
    Categorizes risks into:
    - Weather Risks (Wind, Storm, Rain)
    - Production Risks (Cloud cover, Visibility)
    - Maintenance Risks (Temperature extremes)
    """
    
    def __init__(self):
        # Weights for different risk categories
        self.weights = {
            "critical_weather": 0.5,    # High wind, storms
            "production_impact": 0.3,   # Heavy clouds, low visibility
            "environmental": 0.2        # Temp extremes, humidity
        }
        
    def calculate_risk_score(self, weather_data: Dict) -> Dict:
        """
        Calculates a risk score from 0 to 100.
        
        Args:
            weather_data: Dictionary containing temperature, wind_speed, clouds, weather, pop, visibility
            
        Returns:
            Dictionary with score, level, and recommendations
        """
        try:
            # 1. Critical Weather Risk (0-100)
            wind_speed = weather_data.get("wind_speed", 0)
            # Gale force starts ~17m/s (62km/h)
            wind_risk = min(100, (wind_speed / 20) * 100) if wind_speed > 10 else 0
            
            weather_desc = weather_data.get("weather", "").lower()
            storm_risk = 0
            if "storm" in weather_desc or "thunderstorm" in weather_desc:
                storm_risk = 90
            elif "rain" in weather_desc:
                storm_risk = 30
                
            critical_risk = max(wind_risk, storm_risk)
            
            # 2. Production Risk (0-100)
            clouds = weather_data.get("clouds", 0)
            pop = weather_data.get("pop", 0) # Probability of precipitation
            visibility = weather_data.get("visibility", 10000)
            
            visibility_risk = max(0, (1 - (visibility / 10000)) * 100)
            production_risk = (clouds * 0.6) + (pop * 0.2) + (visibility_risk * 0.2)
            
            # 3. Environmental Risk (0-100)
            temp = weather_data.get("temperature", 25)
            # Overheating risk (>40C) or Freezing risk (<0C)
            temp_risk = 0
            if temp > 40:
                temp_risk = min(100, (temp - 40) * 10)
            elif temp < 0:
                temp_risk = min(100, abs(temp) * 5)
                
            humidity = weather_data.get("humidity", 50)
            humidity_risk = (humidity - 80) * 2 if humidity > 80 else 0
            
            environmental_risk = max(temp_risk, humidity_risk)
            
            # Weighted Final Score
            final_score = (critical_risk * self.weights["critical_weather"] + 
                           production_risk * self.weights["production_impact"] + 
                           environmental_risk * self.weights["environmental"])
            
            final_score = round(min(100, final_score), 1)
            
            # Determine Level
            level = "Low"
            if final_score >= 80:
                level = "Extreme"
            elif final_score >= 50:
                level = "High"
            elif final_score >= 25:
                level = "Moderate"
                
            # Generate Recommendations
            recommendations = self._generate_recommendations(level, final_score, weather_data)
            
            return {
                "score": final_score,
                "level": level,
                "categories": {
                    "critical": critical_risk,
                    "production": production_risk,
                    "environmental": environmental_risk
                },
                "recommendations": recommendations,
                "defensive_charging": None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return {"score": 0, "level": "Unknown", "error": str(e)}

    def generate_defensive_triggers(
        self, forecast_hours: List[Dict], battery_soc: float = 50.0
    ) -> Dict:
        """
        Scan the next 12h of weather forecast for storm risk.
        If risk > 50, generate a DEFENSIVE_CHARGE trigger to reserve
        the battery for outage resilience.

        Args:
            forecast_hours: list of hourly weather dicts (next 12-24h).
            battery_soc: current battery state-of-charge (%).

        Returns:
            defensive_charging dict with trigger, target_soc, reason, hours_until_event.
        """
        if not forecast_hours:
            return {"trigger": False, "reason": "No forecast data available"}

        worst_risk = 0.0
        worst_hour = 0
        storm_detected = False
        storm_details = ""

        for i, hour_data in enumerate(forecast_hours[:12]):
            risk_result = self.calculate_risk_score(hour_data)
            score = risk_result.get("score", 0)
            if score > worst_risk:
                worst_risk = score
                worst_hour = i
                if risk_result.get("level") in ("High", "Extreme"):
                    storm_detected = True
                    weather_desc = hour_data.get("weather", "severe weather")
                    wind = hour_data.get("wind_speed", 0)
                    storm_details = f"{weather_desc}, wind {wind:.0f} m/s"

        if storm_detected and worst_risk > 50:
            target_soc = 100.0
            return {
                "trigger": True,
                "action": "DEFENSIVE_CHARGE",
                "target_soc": target_soc,
                "current_soc": battery_soc,
                "charge_needed_pct": round(max(0, target_soc - battery_soc), 1),
                "hours_until_event": worst_hour,
                "risk_score": worst_risk,
                "reason": f"⚡ Severe weather detected in {worst_hour}h: {storm_details}. "
                          f"Reserving battery to 100% for outage resilience.",
            }
        else:
            return {
                "trigger": False,
                "current_soc": battery_soc,
                "max_risk_next_12h": worst_risk,
                "reason": "No severe weather threats in the next 12 hours.",
            }

    def _generate_recommendations(self, level: str, score: float, weather_data: Dict) -> List[str]:
        recs = []
        
        if level == "Extreme":
            recs.append("⚠️ IMMEDIATE ACTION: Secure all loose components around panels.")
            recs.append("🛡️ SYSTEM PROTECTION: Consider manual shutdown if severe lightning persists.")
            recs.append("🔋 DEFENSIVE: Battery should be charged to 100% for outage resilience.")
        elif level == "High":
            recs.append("🔔 MONITOR: High environmental stress detected.")
            recs.append("🔋 DEFENSIVE: Consider pre-charging battery to maximum.")
            if weather_data.get("wind_speed", 0) > 15:
                recs.append("💨 WIND ALERT: Check mounting stability.")
        elif level == "Moderate":
            recs.append("📝 NOTE: Sub-optimal conditions for generation.")
        else:
            recs.append("✅ STATUS: System operating in safe environmental margins.")
            
        # Specific based on drivers
        if weather_data.get("clouds", 0) > 80:
            recs.append("☁️ PRODUCTION: Expect significant cloud-cover losses.")
        if weather_data.get("temperature", 25) > 40:
            recs.append("🌡️ THERMAL: High panel temperatures may reduce efficiency.")
        if weather_data.get("humidity", 50) > 90:
            recs.append("💧 HUMIDITY: High condensation risk; check electrical sealings.")
            
        return recs
