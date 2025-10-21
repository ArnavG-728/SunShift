'use client'

import { useState, useEffect } from 'react'
import SimpleForecastDashboard from '@/components/SimpleForecastDashboard'
import ChatInterface from '@/components/ChatInterface'
import RealTimeWeather from '@/components/RealTimeWeather'
import GreenMetrics from '@/components/GreenMetrics'
import SolarMetrics from '@/components/SolarMetrics'
import SystemConfiguration from '@/components/SystemConfiguration'
import SmartRecommendations from '@/components/SmartRecommendations'
import { Sun, Sparkles } from 'lucide-react'
import { loadUserPreferences, SystemConfig } from '@/lib/userPreferences'

export default function Home() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'recommendations' | 'chat'>('dashboard')
  const [mounted, setMounted] = useState(false)
  const [config, setConfig] = useState<SystemConfig>(() => {
    // Provide default config during SSR to prevent hydration mismatch
    if (typeof window === 'undefined') {
      return {
        systemSize: 5.0,
        panelEfficiency: 0.15,
        panelTilt: 30.0,
        panelAzimuth: 180.0,
        performanceRatio: 0.78,
        city: 'Delhi (IN)',
        latitude: 28.6139,
        longitude: 77.2090,
        timezone: 'Asia/Kolkata',
        electricityTariff: 0.12,
        feedInTariff: 0.08,
        currency: 'USD',
        hasBattery: false,
        batteryCapacity: 0,
        batteryEfficiency: 0.95,
        gridCO2Factor: 0.70,
        maxGridImport: 10.0,
        temperatureUnit: 'C',
        energyUnit: 'kWh',
        theme: 'auto',
        enableAlerts: true,
        alertThreshold: 2.0,
      }
    }
    return loadUserPreferences()
  })
  
  useEffect(() => {
    // Load user preferences only on client-side
    setMounted(true)
    const loaded = loadUserPreferences()
    setConfig(loaded)
  }, [])
  
  const handleConfigChange = (newConfig: SystemConfig) => {
    setConfig(newConfig)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-orange-50 via-yellow-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8 py-3 sm:py-4">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-0">
            <div className="flex items-center space-x-2 sm:space-x-3">
              <div className="bg-gradient-to-br from-orange-500 to-yellow-500 p-1.5 sm:p-2 rounded-lg shadow-md">
                <Sun className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-orange-600 to-yellow-500 bg-clip-text text-transparent">SunShift</h1>
                <p className="text-xs sm:text-sm text-gray-500 hidden sm:block">Solar Energy Forecasting & Analytics</p>
              </div>
            </div>
            
            {/* Tab Navigation */}
            <nav className="flex space-x-1 bg-gray-100 p-1 rounded-lg w-full sm:w-auto overflow-x-auto">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`flex-1 sm:flex-none px-3 sm:px-4 py-2 rounded-md text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${
                  activeTab === 'dashboard'
                    ? 'bg-white text-orange-600 shadow-md'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                📊 Dashboard
              </button>
              <button
                onClick={() => setActiveTab('recommendations')}
                className={`flex-1 sm:flex-none px-3 sm:px-4 py-2 rounded-md text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${
                  activeTab === 'recommendations'
                    ? 'bg-white text-orange-600 shadow-md'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                💡 Smart Tips
              </button>
              <button
                onClick={() => setActiveTab('chat')}
                className={`flex-1 sm:flex-none px-3 sm:px-4 py-2 rounded-md text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${
                  activeTab === 'chat'
                    ? 'bg-white text-orange-600 shadow-md'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                🤖 AI Assistant
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8 py-4 sm:py-6 lg:py-8">
        {/* System Configuration */}
        <div className="mb-4 sm:mb-6">
          <SystemConfiguration onConfigChange={handleConfigChange} />
        </div>

        {/* Solar Metrics - Full Width */}
        {mounted && (
          <div className="mb-4 sm:mb-6">
            <SolarMetrics 
              systemSize={config.systemSize}
              performanceRatio={config.performanceRatio}
              tariff={config.electricityTariff}
              gridCO2Factor={config.gridCO2Factor}
              latitude={config.latitude}
              longitude={config.longitude}
              city={config.city}
            />
          </div>
        )}

        {/* Real-Time Weather & Green Metrics */}
        {mounted && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 mb-4 sm:mb-6">
            <div className="w-full">
              <RealTimeWeather 
                latitude={config.latitude}
                longitude={config.longitude}
                city={config.city}
              />
            </div>
            <div className="w-full">
              <GreenMetrics />
            </div>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'dashboard' && (
          <SimpleForecastDashboard 
            latitude={config.latitude}
            longitude={config.longitude}
            city={config.city}
            systemSize={config.systemSize}
            efficiency={config.panelEfficiency}
            panelTilt={config.panelTilt}
            panelAzimuth={config.panelAzimuth}
            performanceRatio={config.performanceRatio}
          />
        )}
        
        {activeTab === 'recommendations' && (
          <SmartRecommendations />
        )}
        
        {activeTab === 'chat' && (
          <ChatInterface />
        )}
      </div>
    </main>
  )
}
