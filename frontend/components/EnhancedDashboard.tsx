'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { Activity, TrendingUp, AlertCircle, RefreshCw, Calendar, Clock, Zap } from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface EnhancedDashboardProps {
  // Props are now optional - will use system config by default
}

export default function EnhancedDashboard(props: EnhancedDashboardProps = {}) {
  const { config } = useSystemConfig()
  const [loading, setLoading] = useState(false)
  const [activeHorizon, setActiveHorizon] = useState<'24h' | '7d' | '4w'>('24h')
  
  // Data states
  const [hourly24h, setHourly24h] = useState<any[]>([])
  const [daily7d, setDaily7d] = useState<any[]>([])
  const [weekly4w, setWeekly4w] = useState<any[]>([])
  
  // Insights
  const [insights, setInsights] = useState<any>(null)
  const [metrics, setMetrics] = useState<any>(null)
  
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  const fetchLatestData = async () => {
    try {
      // Fetch all horizons
      const [h24, d7, w4] = await Promise.all([
        axios.get(`${API_BASE_URL}/forecast/24h`).catch(() => ({ data: { data: [] } })),
        axios.get(`${API_BASE_URL}/forecast/7d`).catch(() => ({ data: { data: [] } })),
        axios.get(`${API_BASE_URL}/forecast/4w`).catch(() => ({ data: { data: [] } }))
      ])
      
      if (h24.data.data) setHourly24h(h24.data.data)
      if (d7.data.data) setDaily7d(d7.data.data)
      if (w4.data.data) setWeekly4w(w4.data.data)
      
      setLastUpdate(new Date())
    } catch (error) {
      console.error('Error fetching latest data:', error)
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
        console.log('Setting hourly data:', response.data.hourly_24h.length, 'points')
        setHourly24h(response.data.hourly_24h)
      }
      if (response.data.daily_7d) {
        console.log('Setting daily data:', response.data.daily_7d.length, 'points')
        setDaily7d(response.data.daily_7d)
      }
      if (response.data.weekly_4w) {
        console.log('Setting weekly data:', response.data.weekly_4w.length, 'points')
        setWeekly4w(response.data.weekly_4w)
      }
      
      if (response.data.insights) {
        console.log('Insights:', response.data.insights)
        setInsights(response.data.insights)
      }
      if (response.data.metrics) {
        console.log('Metrics:', response.data.metrics)
        setMetrics(response.data.metrics)
      }
      
      setLastUpdate(new Date())
      
      // Also fetch from endpoints to ensure data is loaded
      await fetchLatestData()
      
      alert(`✅ Forecast completed successfully for ${config.city}!`)
    } catch (error: any) {
      console.error('Error running forecast:', error)
      console.error('Error details:', error.response?.data)
      alert(`❌ Error: ${error.response?.data?.detail || error.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Load data on mount
  useEffect(() => {
    fetchLatestData()
  }, [])

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && hourly24h.length > 0) {
      const interval = setInterval(fetchLatestData, 120000) // 2 minutes
      return () => clearInterval(interval)
    }
  }, [autoRefresh, hourly24h])

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

  const currentData = activeHorizon === '24h' ? formatHourlyData(hourly24h) :
                      activeHorizon === '7d' ? formatDailyData(daily7d) :
                      formatWeeklyData(weekly4w)

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="bg-white rounded-lg shadow p-4 sm:p-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4 mb-4">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-800">Energy Forecast</h2>
          <div className="flex items-center gap-2 sm:gap-4 w-full sm:w-auto">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg text-sm ${
                autoRefresh ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
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
            className={`flex items-center justify-center gap-1 sm:gap-2 px-2 sm:px-4 py-2 rounded-lg text-xs sm:text-sm ${
              activeHorizon === '24h' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'
            }`}
          >
            <Clock className="w-3 h-3 sm:w-4 sm:h-4" />
            <span className="hidden sm:inline">24 Hours</span>
            <span className="sm:hidden">24h</span>
          </button>
          <button
            onClick={() => setActiveHorizon('7d')}
            className={`flex items-center justify-center gap-1 sm:gap-2 px-2 sm:px-4 py-2 rounded-lg text-xs sm:text-sm ${
              activeHorizon === '7d' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'
            }`}
          >
            <Calendar className="w-3 h-3 sm:w-4 sm:h-4" />
            <span className="hidden sm:inline">7 Days</span>
            <span className="sm:hidden">7d</span>
          </button>
          <button
            onClick={() => setActiveHorizon('4w')}
            className={`flex items-center justify-center gap-1 sm:gap-2 px-2 sm:px-4 py-2 rounded-lg text-xs sm:text-sm ${
              activeHorizon === '4w' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'
            }`}
          >
            <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4" />
            <span className="hidden sm:inline">4 Weeks</span>
            <span className="sm:hidden">4w</span>
          </button>
        </div>
      </div>

      {/* Main Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">
          {activeHorizon === '24h' ? '24-Hour Hourly Forecast' :
           activeHorizon === '7d' ? '7-Day Daily Forecast' :
           '4-Week Weekly Forecast'}
        </h3>
        
        {currentData.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            {activeHorizon === '24h' ? (
              <AreaChart data={currentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="upper" stackId="1" stroke="none" fill="#10b98120" name="Upper Bound" />
                <Area type="monotone" dataKey="lower" stackId="1" stroke="none" fill="#10b98120" name="Lower Bound" />
                <Line type="monotone" dataKey="predicted" stroke="#10b981" strokeWidth={3} name="Predicted" dot={false} />
              </AreaChart>
            ) : (
              <BarChart data={currentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={activeHorizon === '7d' ? 'date' : 'week'} />
                <YAxis label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="total" fill="#10b981" name="Total Energy" />
                {activeHorizon === '7d' && (
                  <>
                    <Bar dataKey="min" fill="#3b82f6" name="Min" />
                    <Bar dataKey="max" fill="#ef4444" name="Max" />
                  </>
                )}
              </BarChart>
            )}
          </ResponsiveContainer>
        ) : (
          <div className="h-96 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <Activity className="w-12 h-12 mx-auto mb-2 text-gray-400" />
              <p>No forecast data available</p>
              <p className="text-sm">Click "Run Forecast" to generate predictions</p>
            </div>
          </div>
        )}
      </div>

      {/* Metrics */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-600">Accuracy</div>
            <div className="text-2xl font-bold text-green-600">{metrics.accuracy?.toFixed(1)}%</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-600">MAE</div>
            <div className="text-2xl font-bold text-blue-600">{metrics.mae?.toFixed(2)} kWh</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-600">RMSE</div>
            <div className="text-2xl font-bold text-purple-600">{metrics.rmse?.toFixed(2)} kWh</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-600">Bias Correction</div>
            <div className="text-2xl font-bold text-orange-600">{metrics.bias_correction?.toFixed(3)}</div>
          </div>
        </div>
      )}

      {/* AI Insights */}
      {insights && typeof insights === 'object' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Summary */}
          {insights.summary && (
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Zap className="w-5 h-5 text-blue-600" />
                Forecast Summary
              </h3>
              <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                {String(insights.summary).replace(/\*\*/g, '').replace(/^#+\s/gm, '')}
              </div>
            </div>
          )}

          {/* Next 24h */}
          {insights.next_24h && (
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Clock className="w-5 h-5 text-green-600" />
                Next 24 Hours
              </h3>
              <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                {String(insights.next_24h).replace(/\*\*/g, '').replace(/^#+\s/gm, '')}
              </div>
            </div>
          )}

          {/* Next 7 Days */}
          {insights.next_7d && (
            <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-purple-600" />
                7-Day Outlook
              </h3>
              <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                {String(insights.next_7d).replace(/\*\*/g, '').replace(/^#+\s/gm, '')}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {insights.recommendations && (
            <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-orange-600" />
                Recommendations
              </h3>
              <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                {String(insights.recommendations).replace(/\*\*/g, '').replace(/^#+\s/gm, '')}
              </div>
            </div>
          )}

          {/* Model Performance */}
          {insights.model_performance && (
            <div className="bg-gradient-to-br from-cyan-50 to-blue-50 rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Activity className="w-5 h-5 text-cyan-600" />
                Model Performance
              </h3>
              <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                {String(insights.model_performance).replace(/\*\*/g, '').replace(/^#+\s/gm, '')}
              </div>
            </div>
          )}

          {/* Weather Impact */}
          {insights.weather_impact && (
            <div className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-yellow-600" />
                Weather Impact
              </h3>
              <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                {String(insights.weather_impact).replace(/\*\*/g, '').replace(/^#+\s/gm, '')}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
