# SunShift: Comprehensive Project Explanation

This document provides a detailed, simple-to-understand explanation of the SunShift project, covering both non-technical and technical aspects, and analyzing the primary files that power the system.

---

## 🌍 1. Non-Technical Overview: What is SunShift?

### The Core Problem: The "Solar Black Hole"
We are currently treating cutting-edge solar hardware like 1990s technology. The result? **Massive invisible losses**. Most homeowners and businesses don't know exactly when their panels will make the most energy, so they might run heavy appliances (like washing machines or EV chargers) when solar power is low, forcing them to buy expensive electricity from the grid.

### The Solution: SunShift
**SunShift** is an intelligent, open-source, multi-agent AI platform designed to help people who use solar panels get the absolute most out of them. It bridges the gap between complex solar physics and your wallet. No more "solar blindness". Just pure, optimized energy.

**Our Vision:** Making solar energy *actually* smart. We're turning passive rooftop panels into active, profit-generating, grid-defending assets.

Solar power isn't constantly the same; it changes based on clouds, time of day, and temperature.

### What does SunShift do?
1. **Forecasts the Future:** It looks at weather forecasts and historical NASA data to predict exactly how much solar energy your panels will generate in the next 24 hours to 7 days.
2. **Offers Smart Advice:** It gives you recommendations, like "Run your dishwasher at 11 AM today because your solar power will be at its absolute peak, and it'll cost you nothing."
3. **Optimizes Batteries:** If you have a battery, it calculates when to save solar power and when to use it, maximizing your savings.
4. **Tracks Green Impact:** It shows you exactly how much carbon emission (CO2) you are saving by using solar.

---

## 💻 2. Technical Overview: How is SunShift Built?

SunShift is a full-stack web application. It acts as two major pieces talking to each other:
1. **The Backend (The Brain):** Written in **Python** using **FastAPI**. It handles all the heavy lifting: fetching weather data, running complex AI (Machine Learning) models, and simulating solar physics.
2. **The Frontend (The Face):** Written in **TypeScript/React** using **Next.js 14**. It is the beautiful dashboard that users see in their web browser.

What makes SunShift special is its **Multi-Agent Architecture (LangGraph)**. Instead of having one giant piece of code doing everything, it has a team of specialized AI "Agents" working together.

---

## 📂 3. Detailed File-by-File Explanation

We will explore the code by examining the most important files that make SunShift work.

### A. The Backend Code (`/backend`)

The backend is where the AI and data processing live.

#### 1. `backend/main.py`
* **What it is:** The main entrance to the backend application.
* **What it does:** It creates a web server using **FastAPI**. It defines all the "endpoints" (URLs) that the frontend can talk to, such as `/forecast`, `/chat`, and `/metrics`. When the frontend asks for a forecast, `main.py` receives the request and sends it to the rest of the system securely.

#### 2. `backend/graph/workflow.py`
* **What it is:** The manager of the AI Agents (using a tool called **LangGraph**).
* **What it does:** It creates a pipeline, passing data from one AI agent to the next in a specific order:
  1. `DataAgent` gathers weather data.
  2. `FeatureAgent` processes the data so the AI can read it.
  3. `SolarForecastAgent` makes the actual energy predictions.
  4. `InsightAgent` / `ChatAgent` generate readable advice and answer user questions.

#### 3. `backend/ml/unified_forecaster.py`
* **What it is:** The absolute core forecasting engine. It's a "hybrid" model.
* **What it does:** It combines two things:
  * **PhysicsEngine:** It calculates exactly where the sun will be in the sky based on time and coordinates, and subtracts energy lost by clouds and heat.
  * **SolarForecasterML (LSTM AI):** A Deep Learning AI (Bidirectional LSTM) that learns complex patterns (like morning fog burning off) from historical data. 
  * The file merges the solid rules of physics with the pattern-learning of AI to achieve incredibly accurate (85-95%) solar generation forecasts.

#### 4. `backend/simulation/engine.py`
* **What it is:** The simulator that tracks your house's energy in real-time.
* **What it does:** Think of this as a virtual model of your home. It calculates:
  * How much energy your house naturally uses (`house_load_kw`).
  * If your Electric Vehicle (EV) is plugged in and charging.
  * Whether leftover solar energy should go into your battery (`battery_soc`) or be sold back to the city grid (`grid_exchange_kw`).

#### 5. `backend/agents` (Various Agent files)
* **What they are:** Specialized AI workers.
* **What they do:** Each file here has a specific job. For example, `optimization_agent.py` looks at the forecast and calculates exactly how much money you will save if you shift your power usage. `realtime_data_agent.py` fetches the current cloud cover and temperature from the OpenWeather API.

### B. The Frontend Code (`/frontend`)

The frontend is the visual interface.

#### 1. `frontend/src/app/page.tsx`
* **What it is:** The Landing Page of the website.
* **What it does:** It provides the beautiful, animated welcome screen. It tells new users what SunShift is (with icons highlighting forecasting, eco impact, and smart optimization) and provides a "Let's Get Started" button that takes them into the main dashboard. 

#### 2. `frontend/src/components/Dashboard.tsx`
* **What it is:** The main control panel that users interact with.
* **What it does:** 
  * It shows multiple **Metric Cards** (like Accuracy percentage and Error rates).
  * It displays a beautiful interactive graph (using **Recharts**) showing the actual vs. predicted solar output over 24 hours.
  * It retrieves the "AI Insights" from the backend and prints plain-text advice for the user (e.g., "AI-Generated Insights").
  * It has a "Run Forecast" button that triggers the whole backend AI pipeline.

#### 3. Other Frontend Components
* `SystemConfiguration.tsx`: Allows users to type in their exact solar panel specs (size, tilt, location).
* `ChatInterface.tsx`: A chat window where a user can talk to the Google Gemini-powered AI to ask questions about their energy usage.
* `SmartRecommendations.tsx`: Shows explicit timeslots when the user should run appliances.

---

## 🔄 4. Summary: The Step-by-Step Flow

If a user clicks **"Run Forecast"** on the Dashboard, here is exactly what happens behind the scenes:

1. **Frontend:** The `Dashboard.tsx` file sends a request to the server.
2. **Backend entrance:** `main.py` receives the request.
3. **Workflow starts:** `workflow.py` activates the LangGraph pipeline.
4. **Data Gathering:** The `DataAgent` securely calls weather APIs (OpenWeather) to see if it will be cloudy tomorrow.
5. **AI Prediction:** The data goes to `unified_forecaster.py`, where the Physics Engine and LSTM AI predict exactly how many kilowatts your specific panels will generate at 2 PM, 3 PM, etc.
6. **Simulation & Optimization:** `engine.py` simulates your home's battery, and `optimization_agent.py` creates a schedule on when to do laundry.
7. **Response to User:** The backend bundles the graph data and text advice, sending it back to `Dashboard.tsx`, painting a beautiful, actionable chart on the user's screen.

By combining cutting-edge AI, rigorous solar physics, and an intuitive user interface, SunShift turns complex energy data into simple, daily money-saving actions for everyone.

---

## ❓ 5. Technical Q&A: Addressing the Skeptics

**Q1: "Why do we need AI for this? Can't we just look at the weather forecast and multiply by our panel size?"**  
*Answer:* Weather forecasts only give basic metrics like "30% cloudy". They don't account for complex, non-linear solar physics (the exact angle of the sun, atmospheric interference, and how panels lose efficiency when they get too hot). Our AI predicts how these variables interact with historical patterns (like how morning fog often burns off by 10 AM in specific regions). Simple math gives you a rough guess; our hybrid LSTM/Physics model achieves 85-95% accuracy.

**Q2: "Why use LangGraph instead of just calling a simple Python script? Isn't a Multi-Agent system overkill for this?"**  
*Answer:* It provides crucial modularity, fallback mechanisms, and concurrent processing. For example, if the `OpenWeather` API fails, a monolithic script would just crash. Subsystems in LangGraph isolate that failure: only the `DataAgent` panics, safely falling back to a synthetic weather generator based on latitude, allowing the `OptimizationAgent` and `ChatAgent` to keep running smoothly. 

**Q3: "LSTM sounds complicated. If we just need to guess energy output tomorrow, why not use standard Linear Regression or Random Forest?"**  
*Answer:* Solar energy is fundamentally a "time-series" problem with deep cyclical patterns (day/night cycles, seasonal arcs, rolling cloud patterns). A basic machine learning model cannot remember that 11 AM today will look like 11 AM yesterday but slightly different because of the seasonal shift. A Bidirectional LSTM (Long Short-Term Memory network) remembers sequences of time and captures long-term dependencies that other models ignore.

**Q4: "How is your OptimizationAgent deciding when to turn on appliances? Does it actually know what's plugged into my house?"**  
*Answer:* SunShift doesn't need to physically see or control the appliance to start saving you money. `engine.py` simulates a baseline consumption profile (your house's natural load). It calculates exactly how much leftover power you will have (Solar Generation minus Base House Load). If you have a 2kW surplus from 11 AM to 1 PM, the `OptimizationAgent` mathematically flags that window and recommends running your dishwasher or charging an EV, allowing you to manually intercept that surplus before it sells back to the grid for pennies.

**Q5: "Where exactly is the data for the LSTM coming from? Is it a static CSV file?"**  
*Answer:* No, the dataset is dynamically built for *your specific latitude and longitude*. 
1. **The Source:** When a model is trained (`trainer.py`), the `SolarDataCollector` pulls historical solar irradiance data directly from the **NASA POWER API** and historical weather data (cloud cover, temp, humidity) from the **Open-Meteo API**.
2. **The Processing:** It merges this data, fills missing values using physics equations, and creates "cyclical features" (turning hours and months into sine/cosine waves so the AI understands that 11 PM and 1 AM are close to each other).
3. **Virtual Shading:** Crucially, the code intentionally *injects* "virtual shading" (e.g., simulating a tree blocking the sun at 9 AM or a chimney at 2 PM). This forces the LSTM model to actually learn complex, non-linear obstacles instead of just memorizing the physics baseline.
4. **Storage:** The compiled, cleaned dataset is cached locally as a highly compressed `.parquet` file in `backend/data/cache/`.

**Q6: "Where is the trained model stored, and how is it used during a live forecast?"**  
*Answer:* Once the LSTM finishes training on the historical data, it calculates its exact error metrics (MAE, RMSE) and saves itself as a `.keras` file in `backend/models/ml_saved/`, alongside a JSON training report. 
During a live forecast, when a user clicks "Run Forecast":
1. The `SolarForecastAgent` asks the `DataAgent` for *tomorrow's* weather forecast.
2. It loads the pre-trained `.keras` file for that specific location.
3. The model takes a sequence of the last 24 hours of data plus tomorrow's weather to predict the exact energy output (kW) hour-by-hour.
