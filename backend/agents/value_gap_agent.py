"""
Value Gap Agent — Calculates the financial "Value Gap" between self-consuming
solar energy vs exporting it, and quantifies "Virtual Battery" savings from
intelligent load-shifting in homes without physical battery storage.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ValueGapAgent:
    """
    Analyses hourly solar forecast against a residential consumption profile
    to quantify:
    1. The Value Gap — difference between retail buy and wholesale sell price.
    2. Virtual Battery Savings — money saved by shifting loads to peak solar.
    3. Self-consumption / export breakdown.
    """

    def __init__(
        self,
        electricity_tariff: float = 0.15,
        feed_in_tariff: float = 0.05,
        system_size_kwp: float = 5.0,
    ):
        self.electricity_tariff = electricity_tariff  # $/kWh retail buy
        self.feed_in_tariff = feed_in_tariff           # $/kWh wholesale sell
        self.system_size = system_size_kwp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, hourly_forecast: List[Dict]) -> Dict:
        """
        Run the full Value-Gap analysis on an hourly solar forecast.

        Args:
            hourly_forecast: list of dicts, each with at least
                             'timestamp' and 'predicted_output_kWh'.

        Returns:
            Dictionary with value-gap metrics, virtual-battery savings,
            self-consumption stats, and per-hour breakdown.
        """
        if not hourly_forecast:
            return self._empty_result()

        df = pd.DataFrame(hourly_forecast)
        energy_col = (
            "predicted_output_kWh"
            if "predicted_output_kWh" in df.columns
            else "energy_output_kWh"
        )
        if energy_col not in df.columns:
            logger.warning("No energy column in forecast data")
            return self._empty_result()

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Build a synthetic residential consumption profile (kWh per hour)
        df["consumption_kwh"] = df["timestamp"].apply(
            lambda t: self._residential_consumption(t.hour + t.minute / 60)
        )

        # --- Scenario A: No optimisation (naive usage) ---
        naive = self._calc_scenario(df, energy_col, optimised=False)

        # --- Scenario B: Optimised load-shifting ---
        optimised = self._calc_scenario(df, energy_col, optimised=True)

        # Virtual Battery Savings = difference between the two
        vb_savings = optimised["total_value"] - naive["total_value"]

        # Per-hour value gap
        hourly_gap = self._per_hour_gap(df, energy_col)

        # Best windows to shift loads into
        shift_windows = self._find_shift_windows(df, energy_col)

        return {
            "status": "success",
            "value_gap": {
                "buy_rate": self.electricity_tariff,
                "sell_rate": self.feed_in_tariff,
                "delta_per_kwh": round(
                    self.electricity_tariff - self.feed_in_tariff, 4
                ),
                "delta_percentage": round(
                    (
                        (self.electricity_tariff - self.feed_in_tariff)
                        / self.feed_in_tariff
                    )
                    * 100
                    if self.feed_in_tariff > 0
                    else 0,
                    1,
                ),
            },
            "naive_scenario": naive,
            "optimised_scenario": optimised,
            "virtual_battery_savings": {
                "daily_savings": round(vb_savings, 2),
                "monthly_projection": round(vb_savings * 30, 2),
                "annual_projection": round(vb_savings * 365, 2),
                "equivalent_battery_kwh": round(
                    vb_savings / max(0.01, self.electricity_tariff - self.feed_in_tariff),
                    1,
                ),
            },
            "self_consumption": {
                "naive_rate": naive["self_consumption_pct"],
                "optimised_rate": optimised["self_consumption_pct"],
                "improvement": round(
                    optimised["self_consumption_pct"]
                    - naive["self_consumption_pct"],
                    1,
                ),
            },
            "optimal_shift_windows": shift_windows,
            "hourly_breakdown": hourly_gap,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _residential_consumption(self, hour_float: float) -> float:
        """Return approx residential consumption in kWh for a given hour."""
        base = 0.3
        morning = 1.2 * np.exp(-((hour_float - 7.5) ** 2) / (2 * 1.5**2))
        evening = 1.8 * np.exp(-((hour_float - 20) ** 2) / (2 * 2.0**2))
        avg_load = 1.0  # Fixed average residential baseline (~1.0 kW)
        return max(0.1, (base + morning + evening) * avg_load)

    def _calc_scenario(
        self, df: pd.DataFrame, energy_col: str, optimised: bool
    ) -> Dict:
        """Calculate financial scenario with or without load-shifting."""
        consumption = df["consumption_kwh"].values.copy()
        production = df[energy_col].values.copy()

        if optimised:
            # Shift 30 % of evening load into peak-solar hours
            consumption = self._shift_loads(df, consumption, production)

        self_consumed = np.minimum(production, consumption)
        exported = np.maximum(0, production - consumption)
        imported = np.maximum(0, consumption - production)

        # Revenue from self-consumption (avoided grid buy)
        self_consume_value = float(np.sum(self_consumed) * self.electricity_tariff)
        # Revenue from export (feed-in)
        export_value = float(np.sum(exported) * self.feed_in_tariff)
        # Cost of grid import
        import_cost = float(np.sum(imported) * self.electricity_tariff)

        total_value = self_consume_value + export_value - import_cost
        total_production = float(np.sum(production))
        total_self = float(np.sum(self_consumed))

        return {
            "self_consumed_kwh": round(total_self, 2),
            "exported_kwh": round(float(np.sum(exported)), 2),
            "imported_kwh": round(float(np.sum(imported)), 2),
            "self_consume_value": round(self_consume_value, 2),
            "export_revenue": round(export_value, 2),
            "import_cost": round(import_cost, 2),
            "total_value": round(total_value, 2),
            "self_consumption_pct": round(
                (total_self / total_production * 100) if total_production > 0 else 0, 1
            ),
        }

    def _shift_loads(
        self,
        df: pd.DataFrame,
        consumption: np.ndarray,
        production: np.ndarray,
    ) -> np.ndarray:
        """Simulate shifting ~30 % of evening load into solar-peak hours."""
        shifted = consumption.copy()
        hours = df["timestamp"].dt.hour.values

        # Identify evening hours (17-22) as "donor"
        evening_mask = (hours >= 17) & (hours <= 22)
        # Identify solar-peak hours (10-15) as "receiver"
        peak_mask = (hours >= 10) & (hours <= 15)

        shiftable = shifted[evening_mask] * 0.30
        total_shift = float(np.sum(shiftable))

        # Remove from evening
        shifted[evening_mask] -= shiftable

        # Distribute evenly into peak hours
        peak_count = int(np.sum(peak_mask))
        if peak_count > 0:
            shifted[peak_mask] += total_shift / peak_count

        return shifted

    def _per_hour_gap(self, df: pd.DataFrame, energy_col: str) -> List[Dict]:
        """Return per-hour value gap breakdown (max 24 hours)."""
        breakdown: List[Dict] = []
        for _, row in df.head(24).iterrows():
            prod = float(row[energy_col])
            cons = float(row["consumption_kwh"])
            net = prod - cons

            if net > 0:
                # Exporting — value = feed-in rate
                value = net * self.feed_in_tariff
                lost = net * (self.electricity_tariff - self.feed_in_tariff)
            else:
                # Importing — cost = retail rate
                value = net * self.electricity_tariff  # negative
                lost = 0

            breakdown.append(
                {
                    "time": row["timestamp"].strftime("%H:%M"),
                    "production_kwh": round(prod, 2),
                    "consumption_kwh": round(cons, 2),
                    "net_kwh": round(net, 2),
                    "value_usd": round(value, 3),
                    "value_lost_usd": round(lost, 3),
                }
            )
        return breakdown

    def _find_shift_windows(
        self, df: pd.DataFrame, energy_col: str
    ) -> List[Dict]:
        """Identify the best windows for shifting heavy loads."""
        windows: List[Dict] = []
        # Look at surplus hours
        for _, row in df.head(24).iterrows():
            surplus = float(row[energy_col]) - float(row["consumption_kwh"])
            if surplus > 0.5:
                windows.append(
                    {
                        "start": row["timestamp"].strftime("%H:%M"),
                        "surplus_kwh": round(surplus, 2),
                        "savings_if_shifted": round(
                            surplus
                            * (self.electricity_tariff - self.feed_in_tariff),
                            3,
                        ),
                    }
                )
        # Sort by savings descending
        windows.sort(key=lambda w: w["savings_if_shifted"], reverse=True)
        return windows[:5]

    def _empty_result(self) -> Dict:
        return {
            "status": "no_data",
            "value_gap": {},
            "naive_scenario": {},
            "optimised_scenario": {},
            "virtual_battery_savings": {},
            "self_consumption": {},
            "optimal_shift_windows": [],
            "hourly_breakdown": [],
        }
