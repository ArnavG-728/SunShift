'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Cloud, Wind, Droplets, Sun, Zap, ThermometerSun, ChevronDown, ChevronUp } from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface RealTimeWeatherProps {
  latitude?: number
  longitude?: number
  city?: string
}

export default function RealTimeWeather({ latitude = 28.6139, longitude = 77.2090, city = 'Delhi (IN)' }: RealTimeWeatherProps) {
  const { config } = useSystemConfig()
  const [weather, setWeather] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [isOpen, setIsOpen] = useState(true)

  const fetchWeather = async () => {
    try {
      if (!loading) setRefreshing(true)
      const response = await axios.get(`${API_BASE_URL}/realtime/current`, {
        params: { 
          lat: latitude, 
          lon: longitude,
          system_size: config.systemSize,
          performance_ratio: config.performanceRatio
        }
      })
      if (response.data.status === 'success') {
        setWeather(response.data.data)
      }
    } catch (error) {
      console.error('Error fetching weather:', error)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchWeather()
    // Refresh every 2 minutes (120 seconds)
    const interval = setInterval(fetchWeather, 120000)
    return () => clearInterval(interval)
  }, [latitude, longitude])

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-4">
          <h3 className="text-lg font-semibold text-white">🌍 Live Weather Data</h3>
        </div>
        <div className="p-6 animate-pulse">Loading...</div>
      </div>
    )
  }

  if (!weather) {
    return (
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-4">
          <h3 className="text-lg font-semibold text-white">🌍 Live Weather Data</h3>
        </div>
        <div className="p-6">
          <p className="text-gray-500">No data available</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-4">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              🌍 Live Weather Data
              {refreshing && (
                <span className="inline-block animate-spin">⟳</span>
              )}
            </h3>
            <p className="text-xs text-blue-100">{city} • {new Date(weather.timestamp).toLocaleTimeString()}</p>
          </div>
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="flex items-center space-x-1 px-3 py-1.5 bg-white/20 hover:bg-white/30 text-white rounded-md text-sm transition-colors"
          >
            <span>{isOpen ? 'Collapse' : 'Expand'}</span>
            {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Content */}
      {isOpen && (
        <div className="p-6 flex-1">

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {/* Temperature */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <ThermometerSun className="w-5 h-5 text-orange-500" />
            <span className="text-sm text-gray-600">Temperature</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {weather.temperature?.toFixed(1)}°C
          </div>
        </div>

        {/* Humidity */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Droplets className="w-5 h-5 text-blue-500" />
            <span className="text-sm text-gray-600">Humidity</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {weather.humidity}%
          </div>
        </div>

        {/* Wind Speed */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Wind className="w-5 h-5 text-cyan-500" />
            <span className="text-sm text-gray-600">Wind Speed</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {weather.wind_speed?.toFixed(1)} m/s
          </div>
        </div>

        {/* Cloud Cover */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Cloud className="w-5 h-5 text-gray-500" />
            <span className="text-sm text-gray-600">Cloud Cover</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {weather.clouds}%
          </div>
        </div>

        {/* Solar Irradiance */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Sun className="w-5 h-5 text-yellow-500" />
            <span className="text-sm text-gray-600">Solar Irradiance</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {weather.solar_irradiance?.toFixed(0)} W/m²
          </div>
        </div>

        {/* Energy Output */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-5 h-5 text-green-500" />
            <span className="text-sm text-gray-600">Energy Output</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {weather.energy_output_kWh?.toFixed(2)} kWh
          </div>
        </div>
      </div>

      {/* Weather Description */}
      <div className="mt-4 p-3 bg-white rounded-lg">
        <div className="flex items-center gap-2">
          <span className="text-2xl">{getWeatherEmoji(weather.weather)}</span>
          <div>
            <div className="font-semibold text-gray-800">{weather.weather}</div>
            <div className="text-sm text-gray-600 capitalize">{weather.description}</div>
          </div>
        </div>
      </div>
        </div>
      )}
    </div>
  )
}

function getWeatherEmoji(weather: string): string {
  const weatherMap: { [key: string]: string } = {
    'Clear': '☀️',
    'Clouds': '☁️',
    'Rain': '🌧️',
    'Drizzle': '🌦️',
    'Thunderstorm': '⛈️',
    'Snow': '❄️',
    'Mist': '🌫️',
    'Haze': '🌫️',
    'Fog': '🌫️'
  }
  return weatherMap[weather] || '🌤️'
}
