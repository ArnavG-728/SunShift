# 🌞 GreenCast - Solar Energy Forecasting Platform

AI-powered solar energy forecasting system with real-time weather integration and smart optimization recommendations.

---

## 🚀 Quick Start

### Backend
```powershell
cd backend
python start.py
```
✅ Runs at `http://localhost:8000`

### Frontend
```powershell
cd frontend
npm run dev
```
✅ Runs at `http://localhost:3000`

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

Copy `backend/.env.example` to `backend/.env` and add:
```env
OPENWEATHER_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here  # Optional for AI chat
```

---

## ✨ Features

- 🌍 **Real Weather Data** - OpenWeather API integration
- 🗺️ **Map-Based Location Selector** - Search cities or drop a pin
- 🧠 **AI Optimization** - Smart appliance scheduling
- ⚡ **Multi-Horizon Forecasts** - 24h, 7d, 4w predictions
- 🔋 **Battery Management** - Charge/discharge optimization
- 💰 **Financial Tracking** - Savings and ROI calculations
- 🌱 **Carbon Impact** - Environmental metrics
- 📱 **Responsive Design** - Works on all devices

---

## 📁 Project Structure

```
Energy_ReGen_v2/
├── backend/
│   ├── agents/              # AI agents
│   ├── models/              # ML models
│   ├── utils/               # Validators, calculations
│   ├── main.py              # FastAPI app
│   └── start.py             # Startup script ⭐
│
└── frontend/
    ├── app/                 # Next.js pages
    ├── components/          # React components
    ├── lib/                 # User preferences
    └── package.json         # Dependencies
```

---

## 🎯 Usage

1. **Configure System** - Enter your solar panel specs
2. **Select Location** - Use map selector or search
3. **Run Forecast** - Get personalized predictions
4. **View Smart Tips** - AI-powered recommendations
5. **Ask AI Assistant** - Get personalized advice

---

## 🔧 Tech Stack
```env
# For real weather data
OPENWEATHER_API_KEY=your_key_here

# For AI insights
GOOGLE_API_KEY=your_gemini_key_here
```

**Without API keys:** System uses synthetic data for demo purposes.

## 🔑 Key Features

### 1. Real-Time Weather
- Live data from OpenWeather API
- Temperature, humidity, wind, clouds
- **Solar irradiance calculation** (physics-based)
- Updates every 10 seconds

### 2. Solar Metrics
- **PSH** (Peak Sun Hours) - Daily solar potential
- **Energy Output** - Estimated kWh generation
- **Savings** - Money saved per day
- **CO₂ Avoided** - Environmental impact

### 3. Multi-Horizon Forecasts
- **24 hours** - Hourly predictions
- **7 days** - Daily aggregates
- **4 weeks** - Weekly trends

### 4. AI Assistant
- Ask questions about forecasts
- Get optimization recommendations
- Understand prediction patterns

### 5. Customizable Settings
- Choose from 15+ cities worldwide
- Set your system size (kWp)
- Configure performance ratio
- Set electricity tariff

## 📊 API Endpoints

```bash
# Forecasting
POST /forecast/run          # Run complete pipeline
GET  /forecast/24h          # 24-hour forecast
GET  /forecast/7d           # 7-day forecast
GET  /forecast/4w           # 4-week forecast

# Real-Time Data
GET  /realtime/current      # Current weather + solar
GET  /realtime/forecast     # Weather forecast

# AI Assistant
POST /chat                  # Chat with AI
```

## 🎯 Use Cases

1. **Homeowners** - Estimate your solar panel output
2. **Solar Installers** - Provide accurate forecasts to customers
3. **Grid Operators** - Predict renewable energy availability
4. **Energy Traders** - Forecast supply for pricing
5. **Researchers** - Study solar energy patterns

## 🌍 Supported Locations

**Pre-configured cities:**
- 🇮🇳 India: Delhi, Mumbai, Bangalore, Chennai
- 🇺🇸 USA: New York, Los Angeles, Chicago
- 🇬🇧 Europe: London, Paris, Berlin
- 🇯🇵 Asia: Tokyo, Singapore, Hong Kong
- 🇦🇺 Australia: Sydney, Melbourne

**Custom locations:** Enter any lat/lon coordinates

## 🔬 Technical Details

### Solar Irradiance Calculation
- Physics-based model using solar geometry
- Accounts for: time, season, latitude, cloud cover
- Calculates solar elevation angle and air mass
- Applies atmospheric attenuation

### LSTM Model
- **Input features:** Weather data + time features + lag features
- **Architecture:** LSTM layers with dropout
- **Training:** 30 days historical data
- **Validation:** 20% holdout set
- **Metrics:** MAE, RMSE, Accuracy

### Energy Output Formula
```
Solar Output = Irradiance × Efficiency × System Size
Wind Output = Wind Speed² × Coefficient
Total = Solar + Wind
```

## 📈 Model Performance

Typical metrics on validation set:
- **MAE:** 0.5-1.5 kWh
- **RMSE:** 0.8-2.0 kWh
- **Accuracy:** 85-95%

## 🎨 Tech Stack

**Backend:** FastAPI • LangGraph • TensorFlow • Pandas • Google Gemini  
**Frontend:** Next.js 14 • TypeScript • TailwindCSS • Recharts  
**APIs:** OpenWeather • Google Gemini

## 📚 Documentation

See `PROJECT_OVERVIEW.md` for detailed architecture and implementation details.

## 🔮 Roadmap

- [ ] Historical data storage (database)
- [ ] Battery storage optimization
- [ ] Grid integration planning
- [ ] PDF report generation
- [ ] Email alerts
- [ ] Mobile app

## 📝 License

MIT License - feel free to use for your projects!

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📧 Support

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for a sustainable future** 🌍⚡
