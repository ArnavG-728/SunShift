'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Cloud, Wind, Droplets, Sun, Zap, ThermometerSun, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import BoxGuide from './BoxGuide'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function RealTimeWeather() {
  const { config } = useSystemConfig()
  const [weather, setWeather] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [isOpen, setIsOpen] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  const fetchWeather = async () => {
    try {
      if (!loading) setRefreshing(true)
      const response = await axios.get(`${API_BASE_URL}/realtime/current`, {
        params: {
          lat: config.latitude,
          lon: config.longitude,
          system_size: config.systemSize,
          performance_ratio: config.performanceRatio,
          panel_tilt: config.panelTilt,
          panel_azimuth: config.panelAzimuth
        }
      })
      if (response.data.status === 'success') {
        setWeather(response.data.data)
        setLastUpdate(new Date())
      }
    } catch (error) {
      console.error('Error fetching weather:', error)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  // Refetch when location or system config changes
  useEffect(() => {
    fetchWeather()
    // Refresh every 2 minutes
    const interval = setInterval(fetchWeather, 120000)
    return () => clearInterval(interval)
  }, [config.latitude, config.longitude, config.systemSize, config.performanceRatio, config.panelTilt, config.panelAzimuth])

  const manualRefresh = () => {
    if (!refreshing) {
      fetchWeather()
    }
  }

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-4">
          <h3 className="text-lg font-semibold text-white">🌍 Live Weather Data</h3>
          <p className="text-xs text-blue-100">{config.city}</p>
        </div>
        <div className="p-6 animate-pulse space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-20 bg-gray-100 rounded-lg" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (!weather) {
    return (
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-4">
          <h3 className="text-lg font-semibold text-white">🌍 Live Weather Data</h3>
          <p className="text-xs text-blue-100">{config.city}</p>
        </div>
        <div className="p-6 text-center">
          <p className="text-gray-500">Unable to load weather data</p>
          <button
            onClick={manualRefresh}
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            Retry
          </button>
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
                <RefreshCw className="w-4 h-4 animate-spin" />
              )}
            </h3>
            <p className="text-xs text-blue-100">
              {config.city} • {lastUpdate ? lastUpdate.toLocaleTimeString() : 'Loading...'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <BoxGuide title="Live Weather Data">
              <p>This box displays the current meteorological conditions and their direct impact on your solar system's real-time performance.</p>
              <ul className="space-y-3 mt-3">
                <li><strong>Temperature:</strong> Outside air temperature. Values above 25°C marginally decrease solar panel efficiency.</li>
                <li><strong>Humidity & Wind Speed:</strong> Environmental factors affecting panel temperatures and weather patterns.</li>
                <li><strong>Cloud Cover (%):</strong> The percentage of sky covered by clouds. High cloud cover directly reduces the sunlight reaching the panels.</li>
                <li><strong>Solar Irradiance (W/m²):</strong> The actual sun power hitting the earth's surface at your location. Optimal clear-sky noon values approach 1,000 W/m².</li>
                <li><strong>Current Output (kW):</strong> Your system's live power generation, calculated from current irradiance, system size, and panel efficiency.</li>
                <li><strong>Sunrise/Sunset & Daylight Hours:</strong> Indicates the active solar production window. The progress bar visualizes how much daylight has passed.</li>
              </ul>
            </BoxGuide>
            <button
              onClick={manualRefresh}
              disabled={refreshing}
              className="p-1.5 bg-white/20 hover:bg-white/30 text-white rounded-md transition-colors disabled:opacity-50"
              title="Refresh"
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="flex items-center space-x-1 px-3 py-1.5 bg-white/20 hover:bg-white/30 text-white rounded-md text-sm transition-colors"
            >
              <span>{isOpen ? 'Collapse' : 'Expand'}</span>
              {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      {isOpen && (
        <div className="p-6 flex-1">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {/* Temperature */}
            <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-lg p-4 border border-orange-100">
              <div className="flex items-center gap-2 mb-2">
                <ThermometerSun className="w-5 h-5 text-orange-500" />
                <span className="text-sm text-gray-600">Temperature</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {weather.temperature?.toFixed(1)}°{config.temperatureUnit}
              </div>
              {weather.temperature > 35 && (
                <p className="text-xs text-orange-600 mt-1">⚠️ High temp may reduce efficiency</p>
              )}
            </div>

            {/* Humidity */}
            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-4 border border-blue-100">
              <div className="flex items-center gap-2 mb-2">
                <Droplets className="w-5 h-5 text-blue-500" />
                <span className="text-sm text-gray-600">Humidity</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {weather.humidity}%
              </div>
            </div>

            {/* Wind Speed */}
            <div className="bg-gradient-to-br from-cyan-50 to-teal-50 rounded-lg p-4 border border-cyan-100">
              <div className="flex items-center gap-2 mb-2">
                <Wind className="w-5 h-5 text-cyan-500" />
                <span className="text-sm text-gray-600">Wind Speed</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {weather.wind_speed?.toFixed(1)} m/s
              </div>
            </div>

            {/* Cloud Cover */}
            <div className="bg-gradient-to-br from-gray-50 to-slate-50 rounded-lg p-4 border border-gray-200">
              <div className="flex items-center gap-2 mb-2">
                <Cloud className="w-5 h-5 text-gray-500" />
                <span className="text-sm text-gray-600">Cloud Cover</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {weather.clouds}%
              </div>
              <div className="w-full bg-gray-200 h-1 rounded-full mt-2">
                <div
                  className="bg-gray-500 h-full rounded-full"
                  style={{ width: `${weather.clouds}%` }}
                />
              </div>
            </div>

            {/* Solar Irradiance */}
            <div className="bg-gradient-to-br from-yellow-50 to-amber-50 rounded-lg p-4 border border-yellow-100">
              <div className="flex items-center gap-2 mb-2">
                <Sun className="w-5 h-5 text-yellow-500" />
                <span className="text-sm text-gray-600">Solar Irradiance</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {weather.solar_irradiance?.toFixed(0)} W/m²
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {weather.solar_irradiance > 800 ? '☀️ Excellent' :
                  weather.solar_irradiance > 400 ? '🌤️ Good' :
                    weather.solar_irradiance > 100 ? '⛅ Fair' : '🌙 Low'}
              </p>
            </div>

            {/* Energy Output */}
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-4 border border-green-100">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-5 h-5 text-green-500" />
                <span className="text-sm text-gray-600">Current Output</span>
              </div>
              <div className="text-2xl font-bold text-green-600">
                {weather.energy_output_kWh?.toFixed(2)} kW
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {((weather.energy_output_kWh / config.systemSize) * 100).toFixed(0)}% of {config.systemSize} kWp
              </p>
            </div>
          </div>

          {/* Weather Description with Sun Times */}
          <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg border border-blue-100">
            <div className="flex items-center gap-3">
              <span className="text-3xl">{getWeatherEmoji(weather.weather)}</span>
              <div className="flex-1">
                <div className="font-semibold text-gray-800">{weather.weather}</div>
                <div className="text-sm text-gray-600 capitalize">{weather.description}</div>
              </div>

              {/* Sunrise/Sunset */}
              <div className="flex items-center gap-4 text-sm">
                {weather.sunrise && (
                  <div className="flex items-center gap-1 text-orange-600">
                    <span>🌅</span>
                    <span>{weather.sunrise}</span>
                  </div>
                )}
                {weather.sunset && (
                  <div className="flex items-center gap-1 text-purple-600">
                    <span>🌇</span>
                    <span>{weather.sunset}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Daylight hours bar */}
            {weather.daylight_hours && (
              <div className="mt-3 pt-3 border-t border-blue-100">
                <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                  <span>☀️ Daylight: {weather.daylight_hours}h</span>
                  <span className={weather.is_daytime ? 'text-green-600 font-medium' : 'text-gray-400'}>
                    {weather.is_daytime ? '● Active Solar Production' : '● Night Mode'}
                  </span>
                </div>
                <div className="w-full bg-gray-200 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-orange-400 via-yellow-400 to-orange-400 h-full transition-all"
                    style={{ width: `${Math.min(100, (weather.daylight_hours / 16) * 100)}%` }}
                  />
                </div>
              </div>
            )}
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
