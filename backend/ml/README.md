# 🧠 Machine Learning & Forecasting

This directory contains the Machine Learning components that power SunShift's advanced predictive capabilities.

## 🔬 Core Components

### 1. Unified Forecaster (`unified_forecaster.py`)
*   **Description:** The main interface for generating predictions. It integrates the physics-based model with ML enhancements.
*   **Role:** Ensures that every prediction has a "physics baseline" before applying ML corrections.

### 2. LSTM Models
*   **Architecture:** Long Short-Term Memory (LSTM) recurrent neural networks.
*   **Purpose:** Time-series forecasting of solar irradiance.
*   **Training:** Trained on historical weather and solar data to learn complex temporal patterns that pure physics models might miss (e.g., morning fog burn-off patterns).

### 3. Usage
The ML models are loaded by the `ForecastAgent` during runtime. If a model is not available or confidence is low, the system seamlessly falls back to the robust physics-based engine.
