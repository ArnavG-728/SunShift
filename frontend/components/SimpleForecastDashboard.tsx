'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Sun, Cloud, Zap, Calendar, TrendingUp, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface SimpleForecastDashboardProps {
  latitude?: number
  longitude?: number
  city?: string
  systemSize?: number
  efficiency?: number
  panelTilt?: number
  panelAzimuth?: number
  performanceRatio?: number
}

export default function SimpleForecastDashboard({ 
  latitude = 28.6139, 
  longitude = 77.2090,
  city = 'Delhi (IN)',
  systemSize = 5.0,
  efficiency = 0.15,
  panelTilt = 30.0,
  panelAzimuth = 180.0,
  performanceRatio = 0.78
}: SimpleForecastDashboardProps) {
  const [loading, setLoading] = useState(false)
  const [view, setView] = useState<'today' | 'week'>('today')
  
  const [todayData, setTodayData] = useState<any[]>([])
  const [weekData, setWeekData] = useState<any[]>([])
  const [summary, setSummary] = useState<any>(null)
  
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  const runForecast = async () => {
    setLoading(true)
    try {
      const response = await axios.post(`${API_BASE_URL}/forecast/run`, {
        latitude,
        longitude,
        days: 30,
        system_size: systemSize,
        efficiency: efficiency,
        panel_tilt: panelTilt,
        panel_azimuth: panelAzimuth,
        performance_ratio: performanceRatio
      })
      
      if (response.data.hourly_24h) {
        processData(response.data.hourly_24h, response.data.daily_7d)
      }
      
      setLastUpdate(new Date())
      alert(`✅ Forecast ready for ${city}!`)
    } catch (error: any) {
      console.error('Error:', error)
      alert(`❌ Error: ${error.response?.data?.detail || error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const processData = (hourly: any[], daily: any[]) => {
    // Process today's data (next 24 hours)
    const today = hourly.slice(0, 24).map((d: any) => {
      const hour = new Date(d.timestamp).getHours()
      const energy = parseFloat(d.predicted_output_kWh?.toFixed(2) || 0)
      
      return {
        time: formatHour(hour),
        hour: hour,
        energy: energy,
        status: getEnergyStatus(energy),
        icon: getTimeIcon(hour)
      }
    })
    
    // Process week data
    const week = daily.slice(0, 7).map((d: any) => {
      const date = new Date(d.date)
      const total = parseFloat(d.total_kwh?.toFixed(1) || 0)
      
      return {
        day: date.toLocaleDateString('en-US', { weekday: 'short' }),
        fullDate: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        energy: total,
        status: getDayStatus(total)
      }
    })
    
    // Calculate summary
    const todayTotal = today.reduce((sum, h) => sum + h.energy, 0)
    const weekTotal = week.reduce((sum, d) => sum + d.energy, 0)
    const bestDay = week.reduce((max, d) => d.energy > max.energy ? d : max, week[0])
    const peakHour = today.reduce((max, h) => h.energy > max.energy ? h : max, today[0])
    
    setSummary({
      todayTotal: todayTotal.toFixed(1),
      weekTotal: weekTotal.toFixed(0),
      bestDay: bestDay?.fullDate || 'N/A',
      bestDayEnergy: bestDay?.energy.toFixed(1) || '0',
      peakHour: peakHour?.time || 'N/A',
      peakEnergy: peakHour?.energy.toFixed(2) || '0'
    })
    
    setTodayData(today)
    setWeekData(week)
  }

  const formatHour = (hour: number) => {
    if (hour === 0) return '12 AM'
    if (hour === 12) return '12 PM'
    if (hour < 12) return `${hour} AM`
    return `${hour - 12} PM`
  }

  const getTimeIcon = (hour: number) => {
    if (hour >= 6 && hour < 18) return '☀️'
    return '🌙'
  }

  const getEnergyStatus = (energy: number) => {
    if (energy > 3) return 'excellent'
    if (energy > 2) return 'good'
    if (energy > 1) return 'moderate'
    return 'low'
  }

  const getDayStatus = (energy: number) => {
    if (energy > 40) return 'excellent'
    if (energy > 30) return 'good'
    if (energy > 20) return 'moderate'
    return 'low'
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'bg-green-100 text-green-800 border-green-300'
      case 'good': return 'bg-blue-100 text-blue-800 border-blue-300'
      case 'moderate': return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      case 'low': return 'bg-gray-100 text-gray-800 border-gray-300'
      default: return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'excellent': return '⭐ Excellent'
      case 'good': return '✓ Good'
      case 'moderate': return '○ Moderate'
      case 'low': return '◯ Low'
      default: return 'Unknown'
    }
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Simple Header */}
      <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-lg shadow-lg p-4 sm:p-6 text-white">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold">Solar Energy Forecast</h2>
            <p className="text-sm sm:text-base text-green-100 mt-1">When will you get power?</p>
          </div>
          <Sun className="w-10 h-10 sm:w-12 sm:h-12 opacity-80" />
        </div>
        
        <button
          onClick={runForecast}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-white text-green-600 rounded-lg hover:bg-green-50 disabled:opacity-50 font-semibold text-sm sm:text-base"
        >
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          {loading ? 'Getting Forecast...' : 'Get Latest Forecast'}
        </button>
        
        {lastUpdate && (
          <p className="text-xs sm:text-sm text-green-100 mt-2 text-center">
            Updated: {lastUpdate.toLocaleTimeString()}
          </p>
        )}
      </div>

      {/* Quick Summary Cards */}
      {summary && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          <div className="bg-white rounded-lg shadow p-3 sm:p-4">
            <div className="flex items-center gap-2 text-orange-600 mb-2">
              <Sun className="w-4 h-4 sm:w-5 sm:h-5" />
              <span className="text-xs sm:text-sm font-medium">Today's Total</span>
            </div>
            <p className="text-xl sm:text-2xl font-bold text-gray-800">{summary.todayTotal} kWh</p>
            <p className="text-xs text-gray-500 mt-1">Next 24 hours</p>
          </div>
          
          <div className="bg-white rounded-lg shadow p-3 sm:p-4">
            <div className="flex items-center gap-2 text-blue-600 mb-2">
              <Calendar className="w-4 h-4 sm:w-5 sm:h-5" />
              <span className="text-xs sm:text-sm font-medium">This Week</span>
            </div>
            <p className="text-xl sm:text-2xl font-bold text-gray-800">{summary.weekTotal} kWh</p>
            <p className="text-xs text-gray-500 mt-1">Next 7 days</p>
          </div>
          
          <div className="bg-white rounded-lg shadow p-3 sm:p-4">
            <div className="flex items-center gap-2 text-green-600 mb-2">
              <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5" />
              <span className="text-xs sm:text-sm font-medium">Best Day</span>
            </div>
            <p className="text-base sm:text-lg font-bold text-gray-800">{summary.bestDay}</p>
            <p className="text-xs text-gray-500 mt-1">{summary.bestDayEnergy} kWh</p>
          </div>
          
          <div className="bg-white rounded-lg shadow p-3 sm:p-4">
            <div className="flex items-center gap-2 text-purple-600 mb-2">
              <Zap className="w-4 h-4 sm:w-5 sm:h-5" />
              <span className="text-xs sm:text-sm font-medium">Peak Time</span>
            </div>
            <p className="text-base sm:text-lg font-bold text-gray-800">{summary.peakHour}</p>
            <p className="text-xs text-gray-500 mt-1">{summary.peakEnergy} kWh</p>
          </div>
        </div>
      )}

      {/* View Selector */}
      <div className="flex gap-2 sm:gap-3">
        <button
          onClick={() => setView('today')}
          className={`flex-1 py-2 sm:py-3 px-4 rounded-lg font-semibold text-sm sm:text-base transition-colors ${
            view === 'today' 
              ? 'bg-green-600 text-white shadow-lg' 
              : 'bg-white text-gray-700 hover:bg-gray-50'
          }`}
        >
          📅 Today (Hour by Hour)
        </button>
        <button
          onClick={() => setView('week')}
          className={`flex-1 py-2 sm:py-3 px-4 rounded-lg font-semibold text-sm sm:text-base transition-colors ${
            view === 'week' 
              ? 'bg-green-600 text-white shadow-lg' 
              : 'bg-white text-gray-700 hover:bg-gray-50'
          }`}
        >
          📆 This Week (Day by Day)
        </button>
      </div>

      {/* Today's Hourly View */}
      {view === 'today' && todayData.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4 sm:p-6">
          <h3 className="text-lg sm:text-xl font-bold text-gray-800 mb-4">
            ⏰ Today's Power Schedule (Hour by Hour)
          </h3>
          
          {/* Simple Chart */}
          <div className="mb-6">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={todayData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  dataKey="time" 
                  tick={{ fontSize: 11 }}
                  interval={2}
                />
                <YAxis 
                  label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip 
                  contentStyle={{ fontSize: 12 }}
                  formatter={(value: any) => [`${value} kWh`, 'Energy']}
                />
                <Bar 
                  dataKey="energy" 
                  fill="#10b981" 
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Hour Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 sm:gap-3">
            {todayData.map((hour, idx) => (
              <div 
                key={idx}
                className={`border-2 rounded-lg p-3 ${getStatusColor(hour.status)}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-lg">{hour.icon}</span>
                  <span className="text-xs font-semibold">{getStatusText(hour.status)}</span>
                </div>
                <p className="text-sm font-bold text-gray-800">{hour.time}</p>
                <p className="text-xl sm:text-2xl font-bold mt-1">{hour.energy.toFixed(1)}</p>
                <p className="text-xs text-gray-600">kWh</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Week's Daily View */}
      {view === 'week' && weekData.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4 sm:p-6">
          <h3 className="text-lg sm:text-xl font-bold text-gray-800 mb-4">
            📅 This Week's Power (Day by Day)
          </h3>
          
          {/* Simple Chart */}
          <div className="mb-6">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={weekData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  dataKey="day" 
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  contentStyle={{ fontSize: 12 }}
                  formatter={(value: any) => [`${value} kWh`, 'Total Energy']}
                />
                <Bar 
                  dataKey="energy" 
                  fill="#3b82f6" 
                  radius={[6, 6, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Day Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
            {weekData.map((day, idx) => (
              <div 
                key={idx}
                className={`border-2 rounded-lg p-4 ${getStatusColor(day.status)}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold">{getStatusText(day.status)}</span>
                  {day.status === 'excellent' && <span className="text-xl">🌟</span>}
                  {day.status === 'good' && <span className="text-xl">☀️</span>}
                  {day.status === 'moderate' && <span className="text-xl">⛅</span>}
                  {day.status === 'low' && <span className="text-xl">☁️</span>}
                </div>
                <p className="text-lg font-bold text-gray-800">{day.day}</p>
                <p className="text-sm text-gray-600 mb-2">{day.fullDate}</p>
                <p className="text-2xl sm:text-3xl font-bold">{day.energy.toFixed(0)}</p>
                <p className="text-sm text-gray-600">kWh total</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Help Section */}
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-900">
            <p className="font-semibold mb-1">How to read this:</p>
            <ul className="space-y-1 text-xs sm:text-sm">
              <li>• <strong>Green bars</strong> = More solar power available</li>
              <li>• <strong>⭐ Excellent</strong> = Best time to use heavy equipment</li>
              <li>• <strong>✓ Good</strong> = Good for normal use</li>
              <li>• <strong>○ Moderate</strong> = Limited power, use carefully</li>
              <li>• <strong>◯ Low</strong> = Very little power (night time)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* No Data State */}
      {!todayData.length && !loading && (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <Sun className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-800 mb-2">No Forecast Yet</h3>
          <p className="text-gray-600 mb-4">Click "Get Latest Forecast" to see when you'll get solar power</p>
        </div>
      )}
    </div>
  )
}
