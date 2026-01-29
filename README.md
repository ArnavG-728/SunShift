# SunShift: AI-Powered Solar Energy Forecasting & Optimization

> **Empowering homeowners and businesses to maximize their solar investment with laboratory-grade precision and AI-driven insights.**

## 🌍 Overview

SunShift is a comprehensive solar energy platform that bridges the gap between complex solar physics and everyday user needs. By combining **real-time weather data**, **NASA historical records**, and **advanced AI agents**, SunShift provides:

*   **Accurate Forecasting:** Hybrid physics + ML models (85-95% accuracy).
*   **Smart Optimization:** actionable scheduling for appliances and EV charging.
*   **Financial & Environmental Insights:** Real-time tracking of savings and carbon footprint.
*   **Democratized Access:** A fully free, open-source tool working globally.

## 🏗️ System Architecture

The project is divided into two main components:

### 1. Backend (`/backend`)
A robust **FastAPI** server that orchestrates a multi-agent workflow.
*   **Tech Stack:** Python, FastAPI, LangGraph, TensorFlow/Keras, Pandas.
*   **Key Modules:**
    *   `agents/`: AI agents for data fetching, forecasting, and optimization.
    *   `ml/`: LSTM models and training pipelines.
    *   `real_weather_forecast.py`: The core physics engine.

### 2. Frontend (`/frontend`)
A modern, responsive dashboard built with **Next.js**.
*   **Tech Stack:** Next.js 14, TypeScript, TailwindCSS, Recharts.
*   **Key Features:** Zero-login architecture, real-time visualization, and an AI chat assistant.

## 🚀 Quick Start

### Prerequisites
*   Python 3.9+
*   Node.js 18+
*   API Keys (Optional for basic features, required for full capability): OpenWeather, Google Gemini.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ArnavG-728/SunShift.git
    cd SunShift
    ```

2.  **Setup Backend:**
    ```bash
    cd backend
    pip install -r requirements.txt
    python main.py
    ```

3.  **Setup Frontend:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

4.  **Access the App:**
    Open `http://localhost:3000` in your browser.

## 📂 Project Structure

*   `backend/`: Server-side logic, API, and AI agents.
    *   `agents/`: Documentation on specific AI agents.
    *   `ml/`: Machine learning model details.
*   `frontend/`: Client-side application and UI components.

## 🤝 Contributing

We welcome contributions! Please see the specific READMEs in `backend/` and `frontend/` for deeper technical details.

## 📄 License

MIT License. Built for a sustainable future. 🌍⚡
