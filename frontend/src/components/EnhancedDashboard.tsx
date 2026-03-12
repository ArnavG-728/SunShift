'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Activity, RefreshCw, Calendar, Clock, Shield, CloudOff, Info, User } from 'lucide-react'
import UnifiedEnergyView from './UnifiedEnergyView'
import { useSystemConfig } from '@/lib/SystemConfigContext'

// Imported subcomponents
import SimpleModeView from './EnhancedDashboard/SimpleModeView'
import InsightsPanel from './EnhancedDashboard/InsightsPanel'
import ForecastChart from './EnhancedDashboard/ForecastChart'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface EnhancedDashboardProps {
  // Props are now optional - will use system config by default
}

export default function EnhancedDashboard(props: EnhancedDashboardProps = {}) {
  const { config } = useSystemConfig()
  const [loading, setLoading] = useState(false)
  const [activeHorizon, setActiveHorizon] = useState<'24h' | '7d'>('24h')

  // Data states
  const [hourly24h, setHourly24h] = useState<any[]>([])
  const [daily7d, setDaily7d] = useState<any[]>([])

  // Insights
  const [insights, setInsights] = useState<any>(null)
  const [metrics, setMetrics] = useState<any>(null)

  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  // New States for Innovation
  const [simpleMode, setSimpleMode] = useState(false)
  const [riskData, setRiskData] = useState<any>(null)
  const [currentWeather, setCurrentWeather] = useState<any>(null)
  const [hasInitialLoad, setHasInitialLoad] = useState(false)

  const fetchLatestData = async (): Promise<boolean> => {
    try {
      // Fetch all horizons
      const [h24, d7] = await Promise.all([
        axios.get(`${API_BASE_URL}/forecast/24h`).catch(() => ({ data: { data: [] } })),
        axios.get(`${API_BASE_URL}/forecast/7d`).catch(() => ({ data: { data: [] } }))
      ])

      const hasData = h24.data.data && h24.data.data.length > 0
      if (hasData) {
        setHourly24h(h24.data.data)
      }
      if (d7.data.data && d7.data.data.length > 0) {
        setDaily7d(d7.data.data)
      }
      setLastUpdate(new Date())

      // Fetch Risk Analysis
      const riskRes = await axios.get(`${API_BASE_URL}/risk/analysis`, {
        params: { lat: config.latitude, lon: config.longitude }
      }).catch(() => null)
      if (riskRes?.data?.risk_analysis) setRiskData(riskRes.data.risk_analysis)

      // Fetch Current Weather (for cloud loss)
      const weatherRes = await axios.get(`${API_BASE_URL}/realtime/current`, {
        params: {
          lat: config.latitude,
          lon: config.longitude,
          system_size: config.systemSize,
          performance_ratio: config.performanceRatio,
          panel_tilt: config.panelTilt,
          panel_azimuth: config.panelAzimuth
        }
      }).catch(() => null)
      if (weatherRes?.data?.data) setCurrentWeather(weatherRes.data.data)

      return hasData
    } catch (error) {
      console.error('Error fetching latest data:', error)
      return false
    }
  }

  const runForecast = async () => {
    setLoading(true)
    try {
      // Send system configuration with the forecast request
      const response = await axios.post(`${API_BASE_URL}/forecast/run`, {
        latitude: config.latitude,
        longitude: config.longitude,
        system_size: config.systemSize,
        efficiency: config.panelEfficiency,
        panel_tilt: config.panelTilt,
        panel_azimuth: config.panelAzimuth,
        performance_ratio: config.performanceRatio,
        days: 30
      })

      console.log('Forecast response:', response.data)

      if (response.data.hourly_24h) {
        setHourly24h(response.data.hourly_24h)
      }
      if (response.data.daily_7d) {
        setDaily7d(response.data.daily_7d)
      }
      if (response.data.insights) {
        setInsights(response.data.insights)
      }
      if (response.data.metrics) {
        setMetrics(response.data.metrics)
      }

      setLastUpdate(new Date())

      // Also fetch from endpoints to ensure data is loaded
      await fetchLatestData()

      // Only show alert if this was a manual run (not auto-run on first load)
      if (hasInitialLoad) {
        alert(`✅ Forecast completed successfully for ${config.city}!`)
      }
    } catch (error: any) {
      console.error('Error running forecast:', error)
      alert(`❌ Error: ${error.response?.data?.detail || error.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Load data on mount and when location changes
  useEffect(() => {
    const loadData = async () => {
      const hasData = await fetchLatestData()
      // Auto-run forecast if no cached data exists (first time load)
      if (!hasData && !hasInitialLoad) {
        await runForecast()
      }
      setHasInitialLoad(true)
    }
    loadData()
  }, [config.latitude, config.longitude, config.systemSize, config.performanceRatio, config.panelTilt, config.panelAzimuth])

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && hourly24h.length > 0) {
      const interval = setInterval(fetchLatestData, 120000) // 2 minutes
      return () => clearInterval(interval)
    }
  }, [autoRefresh, hourly24h, config.latitude, config.longitude, config.systemSize, config.performanceRatio, config.panelTilt, config.panelAzimuth])

  const formatHourlyData = (data: any[]) => {
    return data.map(d => ({
      time: new Date(d.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      predicted: parseFloat(d.predicted_output_kWh?.toFixed(2) || 0),
      lower: parseFloat(d.confidence_lower?.toFixed(2) || 0),
      upper: parseFloat(d.confidence_upper?.toFixed(2) || 0),
      temp: parseFloat(d.temperature?.toFixed(1) || 0),
      solar: parseFloat(d.solar_irradiance?.toFixed(0) || 0)
    }))
  }

  const formatDailyData = (data: any[]) => {
    return data.map(d => ({
      date: new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      total: parseFloat(d.total_kwh?.toFixed(1) || 0),
      avg: parseFloat(d.avg_kwh?.toFixed(2) || 0),
      min: parseFloat(d.min_kwh?.toFixed(2) || 0),
      max: parseFloat(d.max_kwh?.toFixed(2) || 0)
    }))
  }

  const formatWeeklyData = (data: any[]) => {
    return data.map(d => ({
      week: `Week ${d.week}`,
      total: parseFloat(d.total_kwh?.toFixed(1) || 0),
      avg: parseFloat(d.avg_kwh?.toFixed(2) || 0)
    }))
  }

  const currentData = activeHorizon === '24h'
    ? formatHourlyData(hourly24h)
    : formatDailyData(daily7d)

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Dynamic Header with Simple Mode Toggle */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-2">
        <div>
          <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-blue-500">
            SunShift Intelligence
          </h1>
          <p className="text-gray-500 text-sm">Actionable solar analytics for {config.city}</p>
        </div>

        <div className="flex items-center bg-gray-100 p-1 rounded-xl border border-gray-200">
          <button
            onClick={() => setSimpleMode(false)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${!simpleMode ? 'bg-white shadow-sm text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
          >
            Power User
          </button>
          <button
            onClick={() => setSimpleMode(true)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${simpleMode ? 'bg-white shadow-sm text-green-600' : 'text-gray-500 hover:text-gray-700'}`}
          >
            Simple Mode
          </button>
        </div>
      </div>

      {!simpleMode && <UnifiedEnergyView />}

<<<<<<< HEAD
      {/* Innovation Section: Risk & Cloud Impact */}
      {!simpleMode && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <RiskScoreCard riskData={riskData} />
          <CloudImpactCard currentWeather={currentWeather} />
        </div>
      )}
=======
      {/* Removing Risk & Cloud Impact sections per user request */}
>>>>>>> dadb8469b898939895b08ce8661c88f2164da40f

      {/* Conditional Rendering for Simple/Power Mode */}
      {simpleMode ? (
        <SimpleModeView currentWeather={currentWeather} daily7d={daily7d} />
      ) : (
        <>
          {/* Header Controls */}
          <div className="bg-white rounded-lg shadow p-4 sm:p-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4 mb-4">
              <h2 className="text-xl sm:text-2xl font-bold text-gray-800">Energy Forecast</h2>
              <div className="flex items-center gap-2 sm:gap-4 w-full sm:w-auto">
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg text-sm ${autoRefresh ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
                    }`}
                >
                  <div className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                  {autoRefresh ? 'Live' : 'Paused'}
                </button>
                <button
                  onClick={runForecast}
                  disabled={loading}
                  className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 sm:px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 text-sm sm:text-base"
                >
                  <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                  {loading ? 'Running...' : 'Run Forecast'}
                </button>
              </div>
            </div>

            {lastUpdate && (
              <p className="text-sm text-gray-500">
                Last updated: {lastUpdate.toLocaleTimeString()}
              </p>
            )}

            {/* Horizon Selector */}
            <div className="grid grid-cols-3 gap-2 mt-4">
              <button
                onClick={() => setActiveHorizon('24h')}
                className={`flex items-center justify-center gap-1 sm:gap-2 px-2 sm:px-4 py-2 rounded-lg text-xs sm:text-sm ${activeHorizon === '24h' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'
                  }`}
              >
                <Clock className="w-3 h-3 sm:w-4 sm:h-4" />
                <span className="hidden sm:inline">24 Hours</span>
                <span className="sm:hidden">24h</span>
              </button>
              <button
                onClick={() => setActiveHorizon('7d')}
                className={`flex items-center justify-center gap-1 sm:gap-2 px-2 sm:px-4 py-2 rounded-lg text-xs sm:text-sm ${activeHorizon === '7d' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'
                  }`}
              >
                <Calendar className="w-3 h-3 sm:w-4 sm:h-4" />
                <span className="hidden sm:inline">7 Days</span>
                <span className="sm:hidden">7d</span>
              </button>
            </div>
          </div>

          {/* Main Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">
              {activeHorizon === '24h' ? '24-Hour Hourly Forecast' : '7-Day Daily Forecast'}
            </h3>
            <ForecastChart currentData={currentData} activeHorizon={activeHorizon} />
          </div>

          <InsightsPanel insights={insights} />
        </>
      )}
    </div>
  )
}
