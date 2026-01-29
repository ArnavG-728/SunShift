# 🖥️ SunShift Frontend

The frontend of SunShift is built with **Next.js 14**, providing a privacy-first, zero-login dashboard for solar monitoring.

## 🎨 Key Features

*   **Zero-Login Architecture:** All user data is stored locally (`localStorage`).
*   **Real-Time:** Updates every 60 seconds with live weather and production data.
*   **Interactive Visualization:** Beautiful charts using **Recharts**.
*   **AI Chat:** Integrated natural language assistant.

## 🛠️ Setup & Installation

1.  **Install Dependencies:**
    ```bash
    npm install
    ```

2.  **Run Development Server:**
    ```bash
    npm run dev
    ```
    Access the app at `http://localhost:3000`.

## 🧩 Component Overview

### **Dashboard (`src/app/dashboard/page.tsx`)**
The command center. Displays:
*   **RealTimeWeather:** Live environmental conditions.
*   **SolarMetrics:** Key KPIs like Peak Sun Hours and Daily Savings.
*   **EnhancedDashboard:** Interactive 24h and 7d forecast charts.
*   **SmartRecommendations:** Artificial Intelligence scheduling advice.

### **Landing Page (`src/app/page.tsx`)**
A high-conversion entry point introducing the project to new users.

### **About Us (`src/app/about-us/page.tsx`)**
Detailed mission statement, problem description, and global impact metrics.

## ⚙️ Configuration

The **SystemConfiguration** component allows users to personalize their setup:
*   Panel Size & Efficiency
*   Tilt & Azimuth (Visual Compass)
*   Battery Capacity
*   Electricity Rates

## 🔐 Privacy
No personal data is sent to our servers. Your configuration lives entirely in your browser.
