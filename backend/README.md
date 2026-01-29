# ⚙️ SunShift Backend

The backend of SunShift is a sophisticated, multi-agent orchestration system built with **FastAPI**. It fuses physics-based modeling with modern AI agents to deliver laboratory-grade solar predictions.

## 🚀 Key Features

*   **FastAPI & Async:** High-concurrency architecture for real-time dashboards.
*   **Multi-Agent Workflow:** Powered by **LangGraph**, handling data fetching, forecasting, and optimization.
*   **Physics Engine:** Calculates solar geometry (declination, azimuth, air mass) for precise irradiance modeling.
*   **Hybrid Data:** Fuses real-time OpenWeather data with NASA POWER historical records.

## 📂 Project Structure

*   `main.py`: Entry point and API router configuration.
*   `agents/`: Specialized AI agents (see `agents/README.md`).
*   `ml/`: Machine Learning models (see `ml/README.md`).
*   `real_weather_forecast.py`: Core physics-based forecasting engine.
*   `graph/`: LangGraph workflow definitions.

## 🛠️ Setup & Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables:**
    Create a `.env` file with:
    ```env
    OPENWEATHER_API_KEY=your_key
    GOOGLE_API_KEY=your_key
    ```

3.  **Run Server:**
    ```bash
    python main.py
    ```
    The API will be available at `http://localhost:8000`.

## 🧠 System Architecture

### 1. Data Fusion
We combine live weather data with historical solar averages. If APIs fail, the system falls back to synthetic weather models.

### 2. Physics Modeling
Calculates the exact position of the sun and the Angle of Incidence (AOI) on your specific panel setup.
*   **Temperature Coefficient:** Simulates efficiency loss from heat.
*   **Cloud Attenuation:** Non-linear reduction based on cloud density.

### 3. Optimization Engine
Transforms raw energy data into decisions:
*   **Load Shifting:** Matches appliance profiles to production windows.
*   **Battery Strategy:** Peak-shaving logic to maximize savings.

## 🔗 API Documentation

Once running, visit `http://localhost:8000/docs` for the interactive Swagger UI.
