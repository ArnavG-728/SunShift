'use client'

import { useState, useEffect } from 'react'
import { Sun, Zap, TrendingUp, Cloud, Activity, ChevronDown, ChevronUp } from 'lucide-react'
import axios from 'axios'
import { useSystemConfig } from '@/lib/SystemConfigContext'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface SolarMetricsProps {
  systemSize?: number
  performanceRatio?: number
  tariff?: number
  gridCO2Factor?: number
  latitude?: number
  longitude?: number
  city?: string
}

export default function SolarMetrics({ 
  systemSize = 3.0, 
  performanceRatio = 0.78,
  tariff = 8.0,
  gridCO2Factor = 0.70,
  latitude = 28.6139,
  longitude = 77.2090,
  city = 'Delhi (IN)'
}: SolarMetricsProps) {
  const { config } = useSystemConfig()
  const [metrics, setMetrics] = useState({
    pshToday: 5.04,
    kwhPerM2: 3.78,
    solarDayClass: 'Typical',
    confidence: 54,
    estimatedEnergy: 11.34,
    savings: 91,
    co2Avoided: 7.9,
  })

  const [loading, setLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(true)

  useEffect(() => {
    fetchSolarMetrics()
    const interval = setInterval(fetchSolarMetrics, 120000) // Update every 2 minutes
    return () => clearInterval(interval)
  }, [
    systemSize, 
    performanceRatio, 
    latitude, 
    longitude,
    tariff,
    gridCO2Factor,
    config.panelEfficiency,
    config.panelTilt,
    config.panelAzimuth
  ])

  const fetchSolarMetrics = async () => {
    try {
      // 1) Try to read latest 24h forecast first (fast path)
      let hourly: any[] = []
      try {
        const h24 = await axios.get(`${API_BASE_URL}/forecast/24h`)
        if (h24.data && Array.isArray(h24.data.data)) {
          hourly = h24.data.data
        }
      } catch (e) {
        // ignore; will fall back to run forecast
      }

      // 2) If not available, run a forecast with current system/location config
      if (!hourly || hourly.length === 0) {
        const body = {
          latitude,
          longitude,
          days: 30,
          system_size: systemSize,
          efficiency: config.panelEfficiency,
          panel_tilt: config.panelTilt,
          panel_azimuth: config.panelAzimuth,
          performance_ratio: performanceRatio
        }
        const resp = await axios.post(`${API_BASE_URL}/forecast/run`, body)
        if (resp.data && Array.isArray(resp.data.hourly_24h)) {
          hourly = resp.data.hourly_24h
        }
      }

      if (!hourly || hourly.length === 0) return

      // 3) Use next 24 hours from forecast and compute totals
      const today = hourly.slice(0, 24)
      const getEnergy = (row: any) => {
        const v = row?.predicted_output_kWh ?? row?.energy_output_kWh ?? 0
        return typeof v === 'number' ? v : parseFloat(v) || 0
      }
      const totalEnergy = today.reduce((sum: number, r: any) => sum + getEnergy(r), 0)
      const cloudsArr = today.map((r: any) => (typeof r.clouds === 'number' ? r.clouds : parseFloat(r.clouds) || 0))
      const avgClouds = cloudsArr.length ? cloudsArr.reduce((a, b) => a + b, 0) / cloudsArr.length : 50

      // PSH aligned with forecast: total_kWh / (kWp * PR)
      const denom = Math.max(1e-6, systemSize * performanceRatio)
      const pshToday = totalEnergy / denom

      // kWh @ 1 kWp per day = totalEnergy / systemSize
      const kwhPerM2 = systemSize > 0 ? totalEnergy / systemSize : 0

      // Day class based on PSH
      let solarDayClass = 'Poor'
      if (pshToday >= 6) solarDayClass = 'Excellent'
      else if (pshToday >= 5) solarDayClass = 'Good'
      else if (pshToday >= 4) solarDayClass = 'Typical'
      else if (pshToday >= 3) solarDayClass = 'Fair'

      const confidence = Math.round(100 - avgClouds)
      const estimatedEnergy = totalEnergy // align exactly to forecast
      const savings = estimatedEnergy * tariff
      const co2Avoided = estimatedEnergy * gridCO2Factor

      setMetrics({
        pshToday: parseFloat(pshToday.toFixed(2)),
        kwhPerM2: parseFloat(kwhPerM2.toFixed(2)),
        solarDayClass,
        confidence: Math.min(100, Math.max(0, confidence)),
        estimatedEnergy: parseFloat(estimatedEnergy.toFixed(2)),
        savings: parseFloat(savings.toFixed(0)),
        co2Avoided: parseFloat(co2Avoided.toFixed(1)),
      })
    } catch (error) {
      console.error('Error fetching solar metrics:', error)
    }
  }

  const getClassColor = (className: string) => {
    switch (className) {
      case 'Excellent': return 'text-green-600 bg-green-100'
      case 'Good': return 'text-blue-600 bg-blue-100'
      case 'Typical': return 'text-yellow-600 bg-yellow-100'
      case 'Fair': return 'text-orange-600 bg-orange-100'
      case 'Poor': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-1">
            <Sun className="w-6 h-6 text-white" />
            <div>
              <h3 className="text-lg sm:text-xl font-bold text-white">
                Peak Sun Hours (PSH) & Solar Energy
              </h3>
              <p className="text-xs sm:text-sm text-purple-100">
                {city} • Real-time solar performance metrics
              </p>
            </div>
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
        <div className="p-4 sm:p-6">

      {/* Top Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
        {/* PSH Today */}
        <div className="bg-white rounded-lg p-3 sm:p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Sun className="w-4 h-4 text-orange-600" />
            <p className="text-xs text-gray-600">PSH Today</p>
          </div>
          <p className="text-xl sm:text-2xl font-bold text-orange-600">
            {metrics.pshToday}
          </p>
          <p className="text-xs text-gray-500">kWh/m²</p>
        </div>

        {/* kWh @ 1 kWp */}
        <div className="bg-white rounded-lg p-3 sm:p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-yellow-600" />
            <p className="text-xs text-gray-600">kWh @ 1 kWp</p>
          </div>
          <p className="text-xl sm:text-2xl font-bold text-yellow-600">
            {metrics.kwhPerM2}
          </p>
          <p className="text-xs text-gray-500">per day</p>
        </div>

        {/* Solar Day Class */}
        <div className="bg-white rounded-lg p-3 sm:p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Cloud className="w-4 h-4 text-blue-600" />
            <p className="text-xs text-gray-600">Solar Day Class</p>
          </div>
          <div className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${getClassColor(metrics.solarDayClass)}`}>
            {metrics.solarDayClass}
          </div>
        </div>

        {/* Confidence */}
        <div className="bg-white rounded-lg p-3 sm:p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-purple-600" />
            <p className="text-xs text-gray-600">Confidence</p>
          </div>
          <p className="text-xl sm:text-2xl font-bold text-purple-600">
            {metrics.confidence}/100
          </p>
        </div>
      </div>

      {/* Estimated Energy Section */}
      <div className="bg-white rounded-lg p-4 shadow-sm">
        <h4 className="text-sm font-semibold text-gray-700 mb-3">
          Estimated energy (your system {systemSize} kWp):
        </h4>
        
        <div className="space-y-3">
          {/* Energy Output */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Daily Output:</span>
            <span className="text-lg font-bold text-blue-600">
              {metrics.estimatedEnergy} kWh
            </span>
          </div>

          {/* Savings */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Savings:</span>
            <span className="text-lg font-bold text-green-600">
              ${metrics.savings}
            </span>
          </div>

          {/* CO2 Avoided */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">CO₂ Avoided:</span>
            <span className="text-lg font-bold text-emerald-600">
              {metrics.co2Avoided} kg
            </span>
          </div>
        </div>

        {/* Performance Ratio Badge */}
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>Performance Ratio:</span>
            <span className="font-semibold text-purple-600">
              {(performanceRatio * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
        </div>
      )}
    </div>
  )
}
