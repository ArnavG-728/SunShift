# SunShift Project: Comprehensive Data Dictionary

This document provides a complete list of all data values displayed or calculated in the SunShift Energy Dashboard, along with their sources and calculation methods.

---

## Table of Contents

1. [User Configuration Values](#1-user-configuration-values)
2. [Real-Time Weather Data](#2-real-time-weather-data)
3. [Solar Metrics](#3-solar-metrics)
4. [Forecast Values](#4-forecast-values)
5. [Optimization & Recommendations](#5-optimization--recommendations)
6. [Risk Assessment](#6-risk-assessment)
7. [Cloud Loss Analytics](#7-cloud-loss-analytics)
8. [Unified Energy Metrics](#8-unified-energy-metrics)
9. [Green AI Metrics](#9-green-ai-metrics)
10. [API Endpoints Summary](#10-api-endpoints-summary)

---

## 1. User Configuration Values

These are user-defined inputs stored in `localStorage` via `frontend/lib/userPreferences.ts`.

| Value | Type | Default | Description |
|-------|------|---------|-------------|
| `systemSize` | number | 5.0 | Solar system capacity in kWp |
| `panelEfficiency` | number | 0.15 | Panel efficiency (0-1, e.g., 0.15 = 15%) |
| `panelTilt` | number | 30.0 | Panel tilt angle in degrees (0-90) |
| `panelAzimuth` | number | 180.0 | Panel orientation (0=N, 90=E, 180=S, 270=W) |
| `performanceRatio` | number | 0.78 | System performance ratio (typical 0.75-0.85) |
| `city` | string | "Delhi (IN)" | User's city name |
| `latitude` | number | 28.6139 | Latitude coordinate |
| `longitude` | number | 77.2090 | Longitude coordinate |
| `timezone` | string | "Asia/Kolkata" | User's timezone |
| `electricityTariff` | number | 0.12 | Cost per kWh from grid ($/kWh) |
| `feedInTariff` | number | 0.08 | Revenue per kWh exported to grid ($/kWh) |
| `currency` | string | "USD" | Currency for financial calculations |
| `hasBattery` | boolean | false | Whether user has a battery storage system |
| `batteryCapacity` | number | 0 | Battery capacity in kWh |
| `batteryEfficiency` | number | 0.95 | Battery round-trip efficiency (0-1) |
| `gridCO2Factor` | number | 0.70 | kg CO₂ emitted per kWh from grid |
| `maxGridImport` | number | 10.0 | Maximum grid import power (kW) |
| `temperatureUnit` | string | "C" | Temperature display unit (C or F) |
| `energyUnit` | string | "kWh" | Energy display unit (kWh or MWh) |
| `theme` | string | "auto" | UI theme (light, dark, auto) |
| `enableAlerts` | boolean | true | Enable low production alerts |
| `alertThreshold` | number | 2.0 | kWh threshold for low production alerts |

---

## 2. Real-Time Weather Data

**Source:** OpenWeather API via `backend/agents/realtime_data_agent.py`
**Endpoint:** `GET /realtime/current`
**Component:** `frontend/components/RealTimeWeather.tsx`

| Value | Unit | Source/Calculation |
|-------|------|---------------------|
| `temperature` | °C | OpenWeather API `main.temp` |
| `humidity` | % | OpenWeather API `main.humidity` |
| `wind_speed` | m/s | OpenWeather API `wind.speed` |
| `clouds` | % | OpenWeather API `clouds.all` |
| `weather` | string | OpenWeather API `weather[0].main` (e.g., "Clear", "Clouds") |
| `description` | string | OpenWeather API `weather[0].description` |
| `solar_irradiance` | W/m² | Calculated (see below) |
| `energy_output_kWh` | kWh | Calculated (see below) |
| `timestamp` | ISO string | Server time in user's timezone |

### Solar Irradiance Calculation
```python
GHI = nasa_power_data['ALLSKY_SFC_SW_DWN']  # From NASA POWER API
clear_sky_factor = 1 - (clouds / 100) * 0.75
solar_irradiance = GHI * clear_sky_factor
```
Falls back to physics-based calculation if NASA API unavailable:
- Uses solar declination, hour angle, and air mass model.

### Energy Output Calculation
```python
energy_output_kWh = (solar_irradiance / 1000) * system_size_kwp * performance_ratio * temp_factor
# temp_factor: -0.4% efficiency per °C above 25°C
```

---

## 3. Solar Metrics

**Component:** `frontend/components/SolarMetrics.tsx`
**Data Source:** Aggregated from 24-hour forecast

| Value | Unit | Calculation |
|-------|------|-------------|
| `pshToday` (Peak Sun Hours) | kWh/m² | `total_energy_24h / (system_size_kwp * performance_ratio)` |
| `kwhPerM2` | kWh | `total_energy_24h / system_size_kwp` |
| `solarDayClass` | string | Based on PSH: Excellent (≥6), Good (≥5), Typical (≥4), Fair (≥3), Poor (<3) |
| `confidence` | 0-100 | `100 - average_cloud_cover` |
| `estimatedEnergy` | kWh | Sum of `predicted_output_kWh` for next 24 hours |
| `savings` | $ | `estimated_energy * electricity_tariff` |
| `co2Avoided` | kg | `estimated_energy * grid_co2_factor` |

---

## 4. Forecast Values

**Source:** `backend/agents/enhanced_forecast_agent.py`
**Endpoint:** `POST /forecast/run`
**Components:** `EnhancedDashboard.tsx`, `SimpleForecastDashboard.tsx`

### Hourly Forecast (24h)

| Value | Unit | Description |
|-------|------|-------------|
| `timestamp` | ISO string | Hour of prediction |
| `predicted_output_kWh` | kWh | Predicted energy output for that hour |
| `confidence_lower` | kWh | Lower bound (95% CI) |
| `confidence_upper` | kWh | Upper bound (95% CI) |
| `temperature` | °C | Forecasted temperature |
| `solar_irradiance` | W/m² | Forecasted irradiance |
| `clouds` | % | Forecasted cloud cover |

### Daily Forecast (7d)

| Value | Unit | Calculation |
|-------|------|-------------|
| `date` | string | Day of prediction |
| `total_kwh` | kWh | Sum of hourly predictions for that day |
| `avg_kwh` | kWh | Average hourly output |
| `min_kwh` | kWh | Minimum hourly output |
| `max_kwh` | kWh | Maximum hourly output |

### Model Metrics

| Value | Unit | Description |
|-------|------|-------------|
| `mae` (Mean Absolute Error) | kWh | Average absolute prediction error |
| `rmse` (Root Mean Square Error) | kWh | Root mean square of prediction errors |
| `accuracy` | % | `(1 - mae / avg_actual) * 100` |
| `bias_correction` | float | Systematic error correction factor |

---

## 5. Optimization & Recommendations

**Source:** `backend/agents/optimization_agent.py`
**Endpoint:** `POST /optimize`
**Component:** `frontend/components/SmartRecommendations.tsx`

### Appliance Schedule

| Value | Description |
|-------|-------------|
| `appliance` | Name of appliance |
| `best_start_time` | Optimal time to run |
| `expected_solar_coverage` | % of energy from solar |
| `cost_savings` | $ saved by scheduling |
| `grid_needed` | kWh still required from grid |

### Battery Schedule

| Value | Description |
|-------|-------------|
| `time` | Hour of the day |
| `action` | charge, discharge, or hold |
| `solar_kwh` | Available solar at that time |
| `strategy` | Overall battery strategy description |
| `estimated_cycles` | Daily charge/discharge cycles |

### Grid Strategy

| Value | Unit | Calculation |
|-------|------|-------------|
| `strategy` | string | "net_exporter" or "net_importer" |
| `total_production_kwh` | kWh | Sum of predicted output |
| `estimated_consumption_kwh` | kWh | Assumed consumption (default: 15 kWh/day) |
| `net_balance_kwh` | kWh | `production - consumption` |
| `recommendation` | string | Action advice |

### Savings

| Value | Unit | Calculation |
|-------|------|-------------|
| `total_savings` | $ | `grid_cost_avoided + export_revenue` |
| `monthly_projection` | $ | `total_savings * 30` |
| `grid_cost_avoided` | $ | `self_consumed_kwh * electricity_tariff` |
| `export_revenue` | $ | `exported_kwh * feed_in_tariff` |

### Carbon Impact

| Value | Unit | Calculation |
|-------|------|-------------|
| `co2_avoided_kg` | kg | `solar_output * grid_co2_factor` |
| `trees_equivalent` | # | `co2_avoided / 21` (kg CO₂/tree/year) |
| `car_miles_avoided` | miles | `co2_avoided / 0.404` (kg CO₂/mile) |

### Automation Triggers

| Value | Description |
|-------|-------------|
| `id` | Trigger identifier (e.g., "solar_excess_high") |
| `action` | Action type (START_LOAD, CHARGE_BATTERY, MAXIMIZE_EXPORT) |
| `target` | Target device (EV_CHARGER, BATTERY, GRID) |
| `condition` | Condition string (e.g., "Production > 3.0kW") |
| `priority` | 1 (high) to 3 (low) |
| `payload` | JSON object with parameters (e.g., `{"current_limit_amps": 16}`) |

---

## 6. Risk Assessment

**Source:** `backend/agents/risk_agent.py`
**Endpoint:** `GET /risk/analysis`
**Component:** `frontend/components/EnhancedDashboard.tsx`

| Value | Range | Calculation |
|-------|-------|-------------|
| `score` | 0-100 | Weighted sum of risk categories |
| `level` | string | Low (<25), Moderate (25-50), High (50-80), Extreme (>80) |
| `categories.critical` | 0-100 | `max(wind_risk, storm_risk)` |
| `categories.production` | 0-100 | `clouds * 0.6 + pop * 0.2 + visibility_risk * 0.2` |
| `categories.environmental` | 0-100 | `max(temp_risk, humidity_risk)` |
| `recommendations` | array | Context-specific safety actions |

### Risk Weights
- Critical Weather: 50%
- Production Impact: 30%
- Environmental: 20%

### Individual Risk Factors
- **Wind Risk:** `(wind_speed / 20) * 100` if wind > 10 m/s
- **Storm Risk:** 90 if "thunderstorm", 30 if "rain"
- **Temp Risk:** `(temp - 40) * 10` if temp > 40°C; `abs(temp) * 5` if temp < 0°C
- **Humidity Risk:** `(humidity - 80) * 2` if humidity > 80%
- **Visibility Risk:** `(1 - visibility/10000) * 100`

---

## 7. Cloud Loss Analytics

**Source:** `backend/agents/realtime_data_agent.py` → `calculate_cloud_loss()`
**Endpoint:** `GET /realtime/current` (included in response)
**Component:** `frontend/components/EnhancedDashboard.tsx`

| Value | Unit | Calculation |
|-------|------|-------------|
| `potential_kwh` | kWh | Energy output assuming 0% clouds |
| `actual_kwh` | kWh | Actual energy output with current clouds |
| `loss_kwh` | kWh | `potential_kwh - actual_kwh` |
| `loss_percent` | % | `(loss_kwh / potential_kwh) * 100` |

```python
clear_sky_irradiance = calculate_solar_irradiance(timestamp, clouds=0)
cloudy_irradiance = calculate_solar_irradiance(timestamp, clouds)
potential_kwh = (clear_sky_irradiance / 1000) * system_size * performance_ratio
actual_kwh = (cloudy_irradiance / 1000) * system_size * performance_ratio
```

---

## 8. Unified Energy Metrics

**Source:** `backend/main.py` → `/usage/unified` (simulated data)
**Component:** `frontend/components/UnifiedEnergyView.tsx`

### Electricity

| Value | Unit | Description |
|-------|------|-------------|
| `solar_gen_kw` | kW | Current solar generation |
| `grid_import_kw` | kW | Current grid import |
| `battery_soc_percent` | % | Battery state of charge |
| `house_load_kw` | kW | Current house consumption |

### Transport

| Value | Unit | Description |
|-------|------|-------------|
| `ev_charge_percent` | % | EV battery level |
| `ev_range_km` | km | Estimated EV range |
| `charging_status` | string | Disconnected, Charging, Standby |

### Other Resources

| Value | Unit | Description |
|-------|------|-------------|
| `gas_usage_m3` | m³ | Current gas consumption (simulated) |
| `water_usage_liters` | L | Current water usage (simulated) |
| `water_leak_alert` | boolean | Leak detection status |

> **Note:** These values are currently simulated using random values. In a real deployment, they would be integrated with smart meters or IoT sensors.

---

## 9. Green AI Metrics

**Component:** `frontend/components/GreenMetrics.tsx`
**Data Source:** Static/default values (can be enhanced)

| Value | Unit | Calculation |
|-------|------|-------------|
| `energyUsed` | Wh | Energy used per AI forecast (default: 0.3 Wh) |
| `carbonEmissions` | g | CO₂ emitted per training cycle (default: 45g) |
| `netEnergySaved` | ratio | Compute cost savings multiplier (default: 5000x) |
| `treesEquivalent` | trees | `(carbonEmissions / 21000) * 12` |
| `kmDriven` | km | `carbonEmissions / 120` (avg car CO₂/km) |

---

## 10. API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root health check |
| `/health` | GET | Detailed health status |
| `/forecast/run` | POST | Run full forecasting pipeline |
| `/forecast/latest` | GET | Get latest forecast results |
| `/forecast/24h` | GET | Get 24-hour hourly predictions |
| `/forecast/7d` | GET | Get 7-day daily predictions |
| `/realtime/current` | GET | Get current weather + energy output |
| `/realtime/forecast` | GET | Get weather forecast |
| `/realtime/status` | GET | Check real-time data connection |
| `/optimize` | POST | Get energy optimization recommendations |
| `/risk/analysis` | GET | Get solar installation risk score |
| `/usage/unified` | GET | Get unified energy metrics |
| `/chat` | POST | Chat with AI assistant |
| `/locations/presets` | GET | Get preset location options |
| `/test/nasa-power` | GET | Test NASA POWER API |

---

## External APIs Used

| API | Purpose | Endpoint |
|-----|---------|----------|
| **OpenWeather API** | Current weather, forecast | `api.openweathermap.org` |
| **NASA POWER API** | Historical solar irradiance | `power.larc.nasa.gov/api` |
| **Google Gemini API** | AI chat and insights | `generativelanguage.googleapis.com` |

---

## Key Formulas Reference

### Energy Output
```
Energy (kWh) = (Irradiance W/m² / 1000) × System Size (kWp) × Performance Ratio × Temperature Factor
```

### Temperature Factor
```
temp_factor = 1 - 0.004 × max(0, temperature - 25)
```

### Cost Savings
```
Savings = (Self-consumed kWh × Electricity Tariff) + (Exported kWh × Feed-in Tariff)
```

### Carbon Avoided
```
CO₂ Avoided (kg) = Solar Output (kWh) × Grid CO₂ Factor (kg/kWh)
```

### Risk Score
```
Risk Score = (Critical × 0.5) + (Production × 0.3) + (Environmental × 0.2)
```

