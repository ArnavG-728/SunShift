"""
Carbon Credit Wallet Agent — Maintains a persistent ledger of CO₂ avoided,
assigns a monetary value based on voluntary carbon market rates, and provides
lifetime / monthly summaries.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from config import config

logger = logging.getLogger(__name__)


class CarbonWalletAgent:
    """
    Persistent carbon credit wallet backed by a lightweight SQLite database.
    Records each CO₂-saving event and exposes cumulative wallet summaries.
    """

    # Default voluntary carbon market price (USD per metric ton CO₂)
    DEFAULT_CREDIT_PRICE = 25.0

    def __init__(self, credit_price_per_ton: float = DEFAULT_CREDIT_PRICE):
        self.credit_price = credit_price_per_ton
        self.db_path = config.DATA_DIR / "carbon_wallet.db"
        self._init_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create the carbon_ledger table if it doesn't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS carbon_ledger (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    location_key TEXT   NOT NULL,
                    energy_kwh  REAL    NOT NULL,
                    co2_kg      REAL    NOT NULL,
                    credit_usd  REAL    NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_location
                ON carbon_ledger (location_key)
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    def has_recorded_today(self, location_key: str) -> bool:
        """Check if credits have already been recorded today for this location."""
        today_prefix = datetime.utcnow().strftime("%Y-%m-%d")
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                """SELECT COUNT(*) FROM carbon_ledger
                   WHERE location_key = ? AND timestamp LIKE ?""",
                (location_key, f"{today_prefix}%"),
            ).fetchone()
            return row[0] > 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_credits(
        self,
        energy_kwh: float,
        grid_co2_factor: float = 0.70,
        latitude: float = 0.0,
        longitude: float = 0.0,
    ) -> Dict:
        """
        Record a new carbon-saving event.

        Args:
            energy_kwh: solar energy generated (kWh)
            grid_co2_factor: kg CO₂ per kWh displaced
            latitude / longitude: location for grouping

        Returns:
            The newly created ledger entry.
        """
        co2_kg = energy_kwh * grid_co2_factor
        credit_usd = (co2_kg / 1000) * self.credit_price  # convert kg→ton
        location_key = f"{latitude:.2f},{longitude:.2f}"
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT INTO carbon_ledger
                   (timestamp, location_key, energy_kwh, co2_kg, credit_usd)
                   VALUES (?, ?, ?, ?, ?)""",
                (now, location_key, energy_kwh, co2_kg, credit_usd),
            )
            conn.commit()

        entry = {
            "timestamp": now,
            "location_key": location_key,
            "energy_kwh": round(energy_kwh, 2),
            "co2_avoided_kg": round(co2_kg, 2),
            "credit_value_usd": round(credit_usd, 4),
        }
        logger.info(f"CarbonWallet: recorded {co2_kg:.2f} kg CO₂ → ${credit_usd:.4f}")
        return entry

    def get_wallet(
        self,
        latitude: float = 0.0,
        longitude: float = 0.0,
    ) -> Dict:
        """
        Retrieve a full wallet summary for a location.

        Returns:
            lifetime_co2_kg, lifetime_credits_usd, monthly breakdown, etc.
        """
        location_key = f"{latitude:.2f},{longitude:.2f}"

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            # Lifetime totals
            row = conn.execute(
                """SELECT COALESCE(SUM(energy_kwh), 0)  AS total_energy,
                          COALESCE(SUM(co2_kg), 0)      AS total_co2,
                          COALESCE(SUM(credit_usd), 0)  AS total_credits,
                          COUNT(*)                       AS entries
                   FROM carbon_ledger
                   WHERE location_key = ?""",
                (location_key,),
            ).fetchone()

            total_energy = float(row["total_energy"])
            total_co2 = float(row["total_co2"])
            total_credits = float(row["total_credits"])
            entries = int(row["entries"])

            # Monthly breakdown (last 12 months)
            monthly = conn.execute(
                """SELECT strftime('%%Y-%%m', timestamp) AS month,
                          SUM(co2_kg) AS co2,
                          SUM(credit_usd) AS credits,
                          SUM(energy_kwh) AS energy
                   FROM carbon_ledger
                   WHERE location_key = ?
                   GROUP BY month
                   ORDER BY month DESC
                   LIMIT 12""",
                (location_key,),
            ).fetchall()

        # Equivalents
        trees_lifetime = total_co2 / 21 if total_co2 > 0 else 0
        car_km_avoided = total_co2 / 0.12 if total_co2 > 0 else 0

        return {
            "status": "success",
            "location": location_key,
            "lifetime": {
                "total_energy_kwh": round(total_energy, 2),
                "total_co2_avoided_kg": round(total_co2, 2),
                "total_co2_avoided_tons": round(total_co2 / 1000, 4),
                "total_credit_value_usd": round(total_credits, 2),
                "entries": entries,
            },
            "equivalents": {
                "trees_year_equivalent": round(trees_lifetime, 1),
                "car_km_avoided": round(car_km_avoided, 0),
            },
            "credit_rate": {
                "price_per_ton_usd": self.credit_price,
                "source": "Voluntary Carbon Market (VCM) average",
            },
            "monthly_breakdown": [
                {
                    "month": m["month"],
                    "co2_kg": round(float(m["co2"]), 2),
                    "credit_usd": round(float(m["credits"]), 4),
                    "energy_kwh": round(float(m["energy"]), 2),
                }
                for m in monthly
            ],
        }

    def _empty_result(self) -> Dict:
        return {
            "status": "no_data",
            "lifetime": {},
            "equivalents": {},
            "credit_rate": {},
            "monthly_breakdown": [],
        }
