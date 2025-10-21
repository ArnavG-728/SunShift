# ☀️ SunShift - Solar Energy Forecasting Platform

AI-powered solar energy forecasting system with real-time weather integration and smart optimization recommendations.

---

## 🚀 Quick Start

### Backend
```powershell
cd backend
python start.py
```
Server runs at `http://localhost:8000`

### Frontend
```powershell
cd frontend
npm install
npm run dev
```
Application runs at `http://localhost:3000`

---

## ⚙️ Setup

### 1. Install Dependencies

**Backend:**
```powershell
cd backend
pip install -r requirements.txt
```

**Frontend:**
```powershell
cd frontend
npm install
```

### 2. Configure API Keys

Copy `backend/.env.example` to `backend/.env` and configure:
```env
# Required for real weather data
OPENWEATHER_API_KEY=your_key_here

# Optional for AI chat features
GOOGLE_API_KEY=your_gemini_key_here
```

**Note:** System works without API keys using fallback data for demonstration.

---

## ✨ Features

- 🌍 **Real-Time Weather** - Live data from OpenWeather API
- ☀️ **Solar Irradiance** - Physics-based calculations with NASA POWER API (no key required)
- 📊 **Multi-Horizon Forecasts** - 24h hourly, 7d daily predictions
- 💡 **Smart Recommendations** - AI-powered optimization tips
- 🤖 **AI Assistant** - Chat interface powered by Google Gemini
- 🔋 **Battery Optimization** - Charge/discharge scheduling recommendations
- 💰 **Financial Metrics** - Savings and cost calculations
- 🌱 **Environmental Impact** - CO₂ avoidance tracking
- 🗺️ **Location Search** - Geocoding search for any location worldwide
- 📱 **Responsive Design** - Mobile-friendly interface

---

## 📁 Project Structure

```
Energy_ReGen_v2/
├── backend/
│   ├── agents/                      # AI agent modules
│   │   ├── realtime_data_agent.py      # Weather & solar data fetching
│   │   ├── enhanced_forecast_agent.py  # LSTM forecasting
│   │   ├── optimization_agent.py       # Smart recommendations
│   │   ├── chat_agent.py               # AI chatbot
│   │   ├── feature_agent.py            # Feature engineering
│   │   └── insight_agent.py            # AI insights
│   ├── models/                      # ML models
│   │   ├── improved_forecaster.py      # LSTM implementation
│   │   └── saved/                      # Model weights
│   ├── graph/                       # LangGraph workflow
│   │   └── workflow.py
│   ├── utils/                       # Utilities
│   │   └── validators.py               # Input validation
│   ├── data/                        # Data storage
│   ├── config.py                    # Configuration
│   ├── main.py                      # FastAPI server
│   ├── real_weather_forecast.py     # Weather-based forecaster
│   ├── start.py                     # Startup script
│   └── requirements.txt             # Python dependencies
│
└── frontend/
    ├── app/                         # Next.js app
    │   ├── page.tsx                    # Main page
    │   ├── layout.tsx                  # Root layout
    │   └── globals.css                 # Global styles
    ├── components/                  # React components
    │   ├── SystemConfiguration.tsx     # System settings
    │   ├── RealTimeWeather.tsx         # Live weather display
    │   ├── SolarMetrics.tsx            # Energy metrics
    │   ├── SimpleForecastDashboard.tsx # Forecast charts
    │   ├── SmartRecommendations.tsx    # Optimization tips
    │   ├── ChatInterface.tsx           # AI chatbot
    │   └── GreenMetrics.tsx            # Environmental impact
    ├── lib/                         # Utilities
    │   └── userPreferences.ts          # Settings management
    └── package.json                 # Node dependencies
```

---

## 🎯 Usage

1. **Configure System** - Set solar panel specifications (size, efficiency, tilt, azimuth)
2. **Select Location** - Choose from 14 preset cities or enter custom coordinates
3. **Run Forecast** - Generate energy predictions for 24h/7d
4. **View Metrics** - Monitor real-time weather, PSH, energy output, savings, CO₂
5. **Get Smart Tips** - AI-powered recommendations for optimal energy usage
6. **Chat with AI** - Ask questions about your forecast and system

---

## 📊 API Endpoints

### Forecasting
```
POST /forecast/run              # Run complete forecast pipeline
GET  /forecast/24h              # 24-hour hourly forecast
GET  /forecast/7d               # 7-day daily forecast
```

### Real-Time Data
```
GET  /realtime/current          # Current weather + solar irradiance
GET  /realtime/forecast         # Weather forecast
GET  /realtime/status           # Connection status
```

### Optimization
```
POST /optimize                  # Get smart recommendations
```

### AI Assistant
```
POST /chat                      # Chat with AI assistant
```

### Utilities
```
GET  /locations/presets         # Get preset city locations
GET  /health                    # Health check
```

---

## 🔬 Technical Details

### Solar Irradiance Calculation
- Fetches NASA POWER API solar data (monthly average GHI) - **No API key required**
- Calculates solar elevation angle based on time and location
- Applies atmospheric attenuation (air mass effect)
- Adjusts for cloud cover from OpenWeather (0-75% reduction)
- Accounts for panel orientation (tilt and azimuth angles)
- Falls back to pure physics-based calculation if NASA POWER unavailable

### Energy Forecasting
- Uses real weather forecast from OpenWeather API (up to 7 days)
- Physics-based solar irradiance calculations for each hour
- Temperature effect on panel efficiency (-0.4% per °C above 25°C)
- System-specific performance ratio
- Multi-horizon aggregation (hourly → daily)

### Energy Output Formula
```
Solar Output (kWh) = (Irradiance / 1000) × System Size × Performance Ratio × Temperature Factor

Temperature Factor = 1 - 0.004 × (Temperature - 25°C)
Clamped between 0.7 and 1.0
```

---

## 🎨 Tech Stack

### Backend
- **FastAPI** - REST API framework
- **TensorFlow/Keras** - LSTM model implementation
- **LangGraph** - AI agent orchestration
- **LangChain** - AI framework
- **Google Gemini** - AI chatbot and insights
- **Pandas/NumPy** - Data processing
- **Scikit-learn** - ML utilities
- **Prophet** - Time series forecasting (optional)

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **TailwindCSS** - Utility-first styling
- **Recharts** - Interactive data visualization
- **Lucide React** - Icon library
- **Axios** - HTTP client

### APIs
- **OpenWeather API** - Real-time weather and forecast data (requires API key)
- **NASA POWER API** - Solar irradiance data (no API key required)
- **Google Gemini API** - AI chat and insights (optional)
- **Nominatim API** - Location geocoding (no API key required)

---

## 🌍 Supported Locations

**Pre-configured cities (14):**
- 🇮🇳 India: Delhi, Mumbai, Bangalore, Chennai
- 🇺🇸 USA: New York, Los Angeles, Chicago
- 🇬🇧 UK: London
- 🇫🇷 France: Paris
- 🇩🇪 Germany: Berlin
- 🇯🇵 Japan: Tokyo
- 🇸🇬 Singapore
- 🇭🇰 Hong Kong
- 🇦🇺 Australia: Sydney, Melbourne

**Custom locations:** Any latitude/longitude coordinates worldwide

---

## 🎯 Use Cases

1. **Homeowners** - Estimate solar panel output and optimize energy usage
2. **Solar Installers** - Provide accurate forecasts to customers
3. **Energy Consultants** - Analyze solar potential for locations
4. **Researchers** - Study solar energy patterns and predictions
5. **Students** - Learn about renewable energy forecasting

---

## 📝 License

MIT License

---

**Built with ❤️ for a sustainable future** 🌍⚡
