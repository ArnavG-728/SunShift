# 🕵️ SunShift AI Agents

SunShift employs a suite of specialized AI agents to handle complex tasks, orchestrated by **LangGraph**.

## 🧩 Agent Overview

### 1. Realtime Data Agent (`realtime_data_agent.py`)
*   **Role:** The "Senses" of the system.
*   **Function:** Fetches live weather (OpenWeather) and historical solar data (NASA POWER).
*   **Key Feature:** Implements a resilient fallback to synthetic data if external APIs are down.

### 2. Solar Forecast Agent (`solar_forecast_agent.py`)
*   **Role:** The "Brain" of prediction.
*   **Function:** Generates hourly and daily energy forecasts using physics-based models.
*   **Key Feature:** Hybridizes physics calculations with ML-based corrections.

### 3. Optimization Agent (`optimization_agent.py`)
*   **Role:** The "Strategist".
*   **Function:** Analyzes forecasts to provide actionable advice.
*   **Key Feature:** Schedules appliances (EV, dishwasher) and manages battery charging cycles for maximum ROI.

### 4. Chat Agent (`chat_agent.py`)
*   **Role:** The "Voice".
*   **Function:** Provides a natural language interface using **Google Gemini**.
*   **Key Feature:** Context-aware; it knows your specific system configuration and current weather when answering questions.

### 5. Risk Agent (`risk_agent.py`)
*   **Role:** The "Guardian".
*   **Function:** Assesses weather risks (storms, high wind, low visibility).
*   **Key Feature:** Provides safety recommendations based on weather severity.

## 🔄 Workflow

Agents pass state objects between each other. For example:
`DataAgent` (Raw Weather) -> `ForecastAgent` (Energy Prediction) -> `OptimizationAgent` (Schedule & Savings).
