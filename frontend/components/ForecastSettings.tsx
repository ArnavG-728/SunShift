'use client'

import { useState, useEffect } from 'react'
import { MapPin, Zap, Sliders, DollarSign, Leaf, Sun, TrendingUp } from 'lucide-react'

interface ForecastSettingsProps {
  onSettingsChange?: (settings: ForecastSettings) => void
}

export interface ForecastSettings {
  city: string
  latitude: number
  longitude: number
  systemSize: number // kWp
  performanceRatio: number // 0-1
  tariff: number // $/kWh
  gridCO2Factor: number // kg/kWh
}

const CITIES = [
  { name: 'Delhi (IN)', lat: 28.6139, lon: 77.2090, country: 'India' },
  { name: 'Mumbai (IN)', lat: 19.0760, lon: 72.8777, country: 'India' },
  { name: 'Bangalore (IN)', lat: 12.9716, lon: 77.5946, country: 'India' },
  { name: 'Chennai (IN)', lat: 13.0827, lon: 80.2707, country: 'India' },
  { name: 'Kolkata (IN)', lat: 22.5726, lon: 88.3639, country: 'India' },
  { name: 'Hyderabad (IN)', lat: 17.3850, lon: 78.4867, country: 'India' },
  { name: 'Pune (IN)', lat: 18.5204, lon: 73.8567, country: 'India' },
  { name: 'Ahmedabad (IN)', lat: 23.0225, lon: 72.5714, country: 'India' },
  { name: 'Jaipur (IN)', lat: 26.9124, lon: 75.7873, country: 'India' },
  { name: 'Lucknow (IN)', lat: 26.8467, lon: 80.9462, country: 'India' },
  { name: 'New York (US)', lat: 40.7128, lon: -74.0060, country: 'USA' },
  { name: 'Los Angeles (US)', lat: 34.0522, lon: -118.2437, country: 'USA' },
  { name: 'London (UK)', lat: 51.5074, lon: -0.1278, country: 'UK' },
  { name: 'Tokyo (JP)', lat: 35.6762, lon: 139.6503, country: 'Japan' },
  { name: 'Sydney (AU)', lat: -33.8688, lon: 151.2093, country: 'Australia' },
]

export default function ForecastSettings({ onSettingsChange }: ForecastSettingsProps) {
  const [settings, setSettings] = useState<ForecastSettings>({
    city: 'Delhi (IN)',
    latitude: 28.6139,
    longitude: 77.2090,
    systemSize: 3.0,
    performanceRatio: 0.78,
    tariff: 8.0,
    gridCO2Factor: 0.70,
  })

  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (onSettingsChange) {
      onSettingsChange(settings)
    }
  }, [settings, onSettingsChange])

  const handleCityChange = (cityName: string) => {
    const city = CITIES.find(c => c.name === cityName)
    if (city) {
      setSettings(prev => ({
        ...prev,
        city: city.name,
        latitude: city.lat,
        longitude: city.lon,
      }))
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div 
        className="flex items-center justify-between p-4 sm:p-6 cursor-pointer hover:bg-gray-50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <div className="bg-orange-100 p-2 rounded-lg">
            <Sliders className="w-5 h-5 text-orange-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-800">Forecast Settings</h3>
            <p className="text-xs sm:text-sm text-gray-500">
              {settings.city} • {settings.systemSize} kWp • PR: {(settings.performanceRatio * 100).toFixed(0)}%
            </p>
          </div>
        </div>
        <button className="text-gray-400 hover:text-gray-600">
          {expanded ? '▲' : '▼'}
        </button>
      </div>

      {/* Settings Panel */}
      {expanded && (
        <div className="border-t p-4 sm:p-6 space-y-6">
          {/* City Selection */}
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
              <MapPin className="w-4 h-4 text-blue-600" />
              City / Region
            </label>
            <select
              value={settings.city}
              onChange={(e) => handleCityChange(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {CITIES.map(city => (
                <option key={city.name} value={city.name}>
                  {city.name}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Coordinates: {settings.latitude.toFixed(4)}°N, {Math.abs(settings.longitude).toFixed(4)}°{settings.longitude >= 0 ? 'E' : 'W'}
            </p>
          </div>

          {/* System Size */}
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
              <Sun className="w-4 h-4 text-yellow-600" />
              System Size (kWp)
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="1"
                max="10"
                step="0.1"
                value={settings.systemSize}
                onChange={(e) => setSettings(prev => ({ ...prev, systemSize: parseFloat(e.target.value) }))}
                className="flex-1"
              />
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min="1"
                  max="10"
                  step="0.1"
                  value={settings.systemSize}
                  onChange={(e) => setSettings(prev => ({ ...prev, systemSize: parseFloat(e.target.value) }))}
                  className="w-20 px-3 py-1 border border-gray-300 rounded-lg text-center"
                />
                <button
                  onClick={() => setSettings(prev => ({ ...prev, systemSize: Math.min(10, prev.systemSize + 0.1) }))}
                  className="px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded"
                >
                  +
                </button>
              </div>
            </div>
          </div>

          {/* Performance Ratio */}
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
              <Zap className="w-4 h-4 text-purple-600" />
              Performance Ratio (PR)
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0.5"
                max="1.0"
                step="0.01"
                value={settings.performanceRatio}
                onChange={(e) => setSettings(prev => ({ ...prev, performanceRatio: parseFloat(e.target.value) }))}
                className="flex-1"
              />
              <span className="w-16 text-center font-semibold text-purple-600">
                {(settings.performanceRatio * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Accounts for system losses (shading, temperature, inverter efficiency)
            </p>
          </div>

          {/* Tariff */}
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
              <DollarSign className="w-4 h-4 text-green-600" />
              Tariff ($/kWh)
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0"
                max="20"
                step="0.1"
                value={settings.tariff}
                onChange={(e) => setSettings(prev => ({ ...prev, tariff: parseFloat(e.target.value) }))}
                className="flex-1"
              />
              <input
                type="number"
                min="0"
                max="20"
                step="0.1"
                value={settings.tariff}
                onChange={(e) => setSettings(prev => ({ ...prev, tariff: parseFloat(e.target.value) }))}
                className="w-20 px-3 py-1 border border-gray-300 rounded-lg text-center"
              />
            </div>
          </div>

          {/* Grid CO2 Factor */}
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
              <Leaf className="w-4 h-4 text-green-600" />
              Grid CO₂ Factor (kg/kWh)
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0"
                max="1.5"
                step="0.01"
                value={settings.gridCO2Factor}
                onChange={(e) => setSettings(prev => ({ ...prev, gridCO2Factor: parseFloat(e.target.value) }))}
                className="flex-1"
              />
              <input
                type="number"
                min="0"
                max="1.5"
                step="0.01"
                value={settings.gridCO2Factor}
                onChange={(e) => setSettings(prev => ({ ...prev, gridCO2Factor: parseFloat(e.target.value) }))}
                className="w-20 px-3 py-1 border border-gray-300 rounded-lg text-center"
              />
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Carbon intensity of grid electricity in your region
            </p>
          </div>

          {/* Apply Button */}
          <button
            onClick={() => setExpanded(false)}
            className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all font-semibold shadow-lg"
          >
            Apply Settings
          </button>
        </div>
      )}
    </div>
  )
}
