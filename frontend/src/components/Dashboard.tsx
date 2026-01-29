'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, TrendingUp, AlertCircle, RefreshCw } from 'lucide-react'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Dashboard() {
  const [loading, setLoading] = useState(false)
  const [forecastData, setForecastData] = useState<any>(null)
  const [insights, setInsights] = useState<string>('')
  const [metrics, setMetrics] = useState<any>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  const fetchLatestData = async () => {
    try {
      const predResponse = await axios.get(`${API_BASE_URL}/forecast/latest`)
      if (predResponse.data.predictions) {
        setForecastData(predResponse.data.predictions)
        setLastUpdate(new Date())
      }
    } catch (error) {
      console.error('Error fetching latest data:', error)
    }
  }

  const runForecast = async () => {
    setLoading(true)
    try {
      const response = await axios.post(`${API_BASE_URL}/forecast/run`)
      setInsights(response.data.insights?.analysis || 'No insights available')
      setMetrics(response.data.metrics)
      
      // Fetch latest predictions
      await fetchLatestData()
    } catch (error) {
      console.error('Error running forecast:', error)
      alert('Error running forecast. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  // Load data on mount (persist across tab switches)
  useEffect(() => {
    fetchLatestData()
  }, [])

  // Auto-refresh every 2 minutes
  useEffect(() => {
    if (autoRefresh && forecastData) {
      const interval = setInterval(() => {
        fetchLatestData()
      }, 120000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, forecastData])

  return (
    <div className="space-y-6">
      {/* Action Bar */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Energy Forecast Dashboard</h2>
            <p className="text-sm text-gray-500 mt-1">
              Monitor and analyze renewable energy generation predictions
              {lastUpdate && (
                <span className="ml-2 text-xs text-green-600">
                  • Last updated: {lastUpdate.toLocaleTimeString()}
                </span>
              )}
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-colors ${
                autoRefresh 
                  ? 'bg-green-50 border-green-300 text-green-700' 
                  : 'bg-gray-50 border-gray-300 text-gray-700'
              }`}
            >
              <Activity className={`h-4 w-4 ${autoRefresh ? 'animate-pulse' : ''}`} />
              <span className="text-sm">{autoRefresh ? 'Live' : 'Paused'}</span>
            </button>
            <button
              onClick={runForecast}
              disabled={loading}
              className="flex items-center space-x-2 bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <RefreshCw className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
              <span>{loading ? 'Running...' : 'Run Forecast'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <MetricCard
            title="Mean Absolute Error"
            value={metrics.mae?.toFixed(2) || '0.00'}
            unit="kWh"
            icon={<Activity className="h-6 w-6" />}
            color="blue"
          />
          <MetricCard
            title="RMSE"
            value={metrics.rmse?.toFixed(2) || '0.00'}
            unit="kWh"
            icon={<TrendingUp className="h-6 w-6" />}
            color="green"
          />
          <MetricCard
            title="Accuracy"
            value={metrics.mae ? ((1 - metrics.mae / 100) * 100).toFixed(1) : '0.0'}
            unit="%"
            icon={<AlertCircle className="h-6 w-6" />}
            color="purple"
          />
        </div>
      )}

      {/* Chart */}
      {forecastData && forecastData.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            24-Hour Energy Output Forecast
          </h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: any) => [`${value.toFixed(2)} kWh`, '']}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="energy_output_kWh" 
                stroke="#3b82f6" 
                name="Actual Output"
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                dataKey="predicted_output_kWh" 
                stroke="#10b981" 
                name="Predicted Output"
                strokeWidth={2}
                strokeDasharray="5 5"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* AI Insights */}
      {insights && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">AI-Generated Insights</h3>
          <div className="prose prose-sm max-w-none">
            <pre className="whitespace-pre-wrap text-gray-700 font-sans">{insights}</pre>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!forecastData && !loading && (
        <div className="bg-white rounded-lg shadow-sm p-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
            <Activity className="h-8 w-8 text-green-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No Forecast Data</h3>
          <p className="text-gray-500 mb-6">
            Click "Run Forecast" to generate energy predictions and insights
          </p>
        </div>
      )}
    </div>
  )
}

function MetricCard({ title, value, unit, icon, color }: any) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
  }

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-lg ${colorClasses[color as keyof typeof colorClasses]}`}>
          {icon}
        </div>
      </div>
      <h3 className="text-sm font-medium text-gray-500 mb-1">{title}</h3>
      <p className="text-3xl font-bold text-gray-900">
        {value} <span className="text-lg text-gray-500">{unit}</span>
      </p>
    </div>
  )
}
