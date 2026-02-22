"""
Degradation Detective Agent — Compares physics-baseline expected output
against actual/simulated output to detect panel health issues such as
dirty panels, inverter degradation, or wiring faults.
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DegradationAgent:
    """
    Analyses the gap between the *physics-expected* energy output and the
    *actual/simulated* energy output to produce:
    1. A 0-100 System Health Score.
    2. A degradation rate (%).
    3. Actionable maintenance alerts.
    """

    # Thresholds (configurable)
    HEALTHY_THRESHOLD = 0.90     # ≥90 % of baseline → healthy
    WARNING_THRESHOLD = 0.75     # 75-90 % → warning
    CRITICAL_THRESHOLD = 0.60    # <60 % → critical

    def __init__(self, system_size_kwp: float = 5.0, panel_age_years: float = 0):
        self.system_size = system_size_kwp
        self.panel_age_years = panel_age_years
        # Standard annual degradation for crystalline silicon panels
        self.expected_annual_degradation = 0.005  # 0.5 % per year

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_health(
        self,
        forecast_data: List[Dict],
        weather_data: Optional[Dict] = None,
        system_config: Optional[Dict] = None,
    ) -> Dict:
        """
        Compare physics-baseline with simulated/forecast output.

        Args:
            forecast_data: hourly dicts with 'predicted_output_kWh' and
                           optionally 'solar_irradiance', 'temperature'.
            weather_data: current weather snapshot (for real-time check).
            system_config: dict with 'system_size', 'panel_age_years', etc.

        Returns:
            dict with health_score, degradation_rate, alerts, recommendations.
        """
        if system_config:
            self.system_size = system_config.get("system_size", self.system_size)
            self.panel_age_years = system_config.get("panel_age_years", self.panel_age_years)

        if not forecast_data:
            return self._empty_result()

        # Compute physics baseline for each hour
        baselines = []
        actuals = []

        for entry in forecast_data:
            irradiance = entry.get("solar_irradiance", 0)
            temperature = entry.get("temperature", 25)
            actual_kwh = entry.get("predicted_output_kWh",
                                   entry.get("energy_output_kWh", 0))

            baseline_kwh = self._physics_baseline(irradiance, temperature)
            baselines.append(baseline_kwh)
            actuals.append(actual_kwh)

        baselines = np.array(baselines)
        actuals = np.array(actuals)

        # Only compare daytime hours (where baseline > 0.05 kWh)
        daytime_mask = baselines > 0.05
        if not np.any(daytime_mask):
            return self._empty_result()

        day_baselines = baselines[daytime_mask]
        day_actuals = actuals[daytime_mask]

        # Performance ratio = actual / baseline
        pr_values = np.divide(
            day_actuals,
            day_baselines,
            out=np.ones_like(day_actuals),
            where=day_baselines > 0,
        )
        mean_pr = float(np.mean(pr_values))

        # Adjust for expected age-related degradation
        age_factor = 1 - (self.expected_annual_degradation * self.panel_age_years)
        adjusted_pr = mean_pr / age_factor if age_factor > 0 else mean_pr

        # Health score (0-100)
        health_score = min(100, max(0, round(adjusted_pr * 100, 1)))

        # Degradation rate (how much below 100 %)
        degradation_pct = round((1 - mean_pr) * 100, 1)

        # Detect anomaly patterns
        anomalies = self._detect_anomalies(pr_values)

        # Build alerts
        alerts = self._build_alerts(health_score, degradation_pct, anomalies)

        # Recommendations
        recommendations = self._build_recommendations(
            health_score, degradation_pct, anomalies
        )

        return {
            "status": "success",
            "health_score": health_score,
            "performance_ratio": round(mean_pr, 3),
            "degradation_rate_pct": degradation_pct,
            "expected_age_degradation_pct": round(
                self.expected_annual_degradation * self.panel_age_years * 100, 1
            ),
            "anomalies": anomalies,
            "alerts": alerts,
            "recommendations": recommendations,
            "details": {
                "daytime_hours_analyzed": int(np.sum(daytime_mask)),
                "avg_baseline_kwh": round(float(np.mean(day_baselines)), 2),
                "avg_actual_kwh": round(float(np.mean(day_actuals)), 2),
                "min_pr": round(float(np.min(pr_values)), 3),
                "max_pr": round(float(np.max(pr_values)), 3),
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _physics_baseline(self, irradiance: float, temperature: float) -> float:
        """Calculate expected output using simple physics model."""
        if irradiance <= 0:
            return 0.0
        efficiency = 0.20  # modern panel
        temp_coeff = -0.004  # -0.4 % per °C above 25
        temp_factor = 1 + temp_coeff * (temperature - 25)
        temp_factor = max(0.7, min(1.0, temp_factor))
        pr = 0.78  # performance ratio
        output = (irradiance / 1000) * self.system_size * efficiency * temp_factor * pr
        return max(0.0, output)

    def _detect_anomalies(self, pr_values: np.ndarray) -> List[Dict]:
        """Detect specific degradation patterns."""
        anomalies: List[Dict] = []

        if len(pr_values) < 3:
            return anomalies

        # 1. Sudden drop — any single hour PR below 50 %
        sudden_drops = np.where(pr_values < 0.50)[0]
        if len(sudden_drops) > 0:
            anomalies.append({
                "type": "sudden_drop",
                "severity": "high",
                "description": f"Sudden output drop detected in {len(sudden_drops)} hour(s). "
                               "Possible inverter fault or shading event.",
                "affected_hours": int(len(sudden_drops)),
            })

        # 2. Gradual decline — PR trending downward
        if len(pr_values) >= 6:
            first_half = np.mean(pr_values[: len(pr_values) // 2])
            second_half = np.mean(pr_values[len(pr_values) // 2 :])
            if second_half < first_half * 0.90:
                anomalies.append({
                    "type": "gradual_decline",
                    "severity": "medium",
                    "description": "Output declining through the day. "
                                   "Possible soiling buildup or thermal degradation.",
                    "decline_pct": round((1 - second_half / first_half) * 100, 1),
                })

        # 3. Consistently low — all hours below warning
        if np.all(pr_values < self.WARNING_THRESHOLD):
            anomalies.append({
                "type": "consistent_underperformance",
                "severity": "high",
                "description": "All daytime hours underperforming. "
                               "Panels may need cleaning or professional inspection.",
            })

        return anomalies

    def _build_alerts(
        self, score: float, deg_pct: float, anomalies: List[Dict]
    ) -> List[Dict]:
        """Generate user-facing alerts."""
        alerts: List[Dict] = []

        if score >= 90:
            alerts.append({
                "type": "info",
                "priority": "low",
                "title": "System Healthy",
                "message": f"Panels operating at {score}% health. No action needed.",
            })
        elif score >= 75:
            alerts.append({
                "type": "warning",
                "priority": "medium",
                "title": "Minor Degradation Detected",
                "message": f"Performance is {deg_pct}% below baseline. "
                           "Consider cleaning panels.",
            })
        else:
            alerts.append({
                "type": "critical",
                "priority": "high",
                "title": "Significant Degradation",
                "message": f"Performance is {deg_pct}% below baseline. "
                           "Professional inspection recommended.",
            })

        for a in anomalies:
            if a["severity"] == "high":
                alerts.append({
                    "type": "critical",
                    "priority": "high",
                    "title": f"Anomaly: {a['type'].replace('_', ' ').title()}",
                    "message": a["description"],
                })

        return alerts

    def _build_recommendations(
        self, score: float, deg_pct: float, anomalies: List[Dict]
    ) -> List[str]:
        recs: List[str] = []

        if score < 90:
            recs.append("🧹 Schedule a panel cleaning to remove dust and debris.")
        if score < 75:
            recs.append("🔧 Book a professional inspection to check wiring and inverters.")
        if any(a["type"] == "sudden_drop" for a in anomalies):
            recs.append("⚡ Check inverter error logs for fault codes.")
        if any(a["type"] == "gradual_decline" for a in anomalies):
            recs.append("🌡️ Monitor panel temperatures — excessive heat may be degrading cells.")
        if self.panel_age_years > 10:
            recs.append("📅 Panels are 10+ years old. Consider efficiency assessment.")
        if not recs:
            recs.append("✅ System is performing well. Continue regular maintenance schedule.")

        return recs

    def _empty_result(self) -> Dict:
        return {
            "status": "no_data",
            "health_score": 0,
            "performance_ratio": 0,
            "degradation_rate_pct": 0,
            "expected_age_degradation_pct": 0,
            "anomalies": [],
            "alerts": [],
            "recommendations": ["No data available for health analysis."],
            "details": {},
        }
