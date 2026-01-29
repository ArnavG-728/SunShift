'use client'

import { useState, useEffect } from 'react'
import { Sun, Zap, TrendingUp, Cloud, Activity, ChevronDown, ChevronUp, DollarSign } from 'lucide-react'
import axios from 'axios'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import { useCurrency } from '@/lib/useCurrency'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function SolarMetrics() {
  const { config } = useSystemConfig()
  const { convert, formatCurrency } = useCurrency()
  const [metrics, setMetrics] = useState({
    pshToday: 0,
    kwhPerM2: 0,
    solarDayClass: 'Loading...',
    confidence: 0,
    estimatedEnergy: 0,
    savings: 0,
    co2Avoided: 0,
  })

  const [loading, setLoading] = useState(true)
  const [isOpen, setIsOpen] = useState(true)

  // Refetch when any config value changes
  useEffect(() => {
    fetchSolarMetrics()
    const interval = setInterval(fetchSolarMetrics, 120000) // Update every 2 minutes
    return () => clearInterval(interval)
  }, [
    config.systemSize,
    config.performanceRatio,
    config.latitude,
    config.longitude,
    config.electricityTariff,
    config.gridCO2Factor,
    config.panelEfficiency,
    config.panelTilt,
    config.panelAzimuth
  ])

  const fetchSolarMetrics = async () => {
    setLoading(true)
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
          latitude: config.latitude,
          longitude: config.longitude,
          days: 30,
          system_size: config.systemSize,
          efficiency: config.panelEfficiency,
          panel_tilt: config.panelTilt,
          panel_azimuth: config.panelAzimuth,
          performance_ratio: config.performanceRatio
        }
        const resp = await axios.post(`${API_BASE_URL}/forecast/run`, body)
        if (resp.data && Array.isArray(resp.data.hourly_24h)) {
          hourly = resp.data.hourly_24h
        }
      }

      if (!hourly || hourly.length === 0) {
        setLoading(false)
        return
      }

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
      const denom = Math.max(1e-6, config.systemSize * config.performanceRatio)
      const pshToday = totalEnergy / denom

      // kWh @ 1 kWp per day = totalEnergy / systemSize
      const kwhPerM2 = config.systemSize > 0 ? totalEnergy / config.systemSize : 0

      // Day class based on PSH
      let solarDayClass = 'Poor'
      if (pshToday >= 6) solarDayClass = 'Excellent'
      else if (pshToday >= 5) solarDayClass = 'Good'
      else if (pshToday >= 4) solarDayClass = 'Typical'
      else if (pshToday >= 3) solarDayClass = 'Fair'

      const confidence = Math.round(100 - avgClouds)
      const estimatedEnergy = totalEnergy
      const savings = estimatedEnergy * config.electricityTariff
      const co2Avoided = estimatedEnergy * config.gridCO2Factor

      setMetrics({
        pshToday: parseFloat(pshToday.toFixed(2)),
        kwhPerM2: parseFloat(kwhPerM2.toFixed(2)),
        solarDayClass,
        confidence: Math.min(100, Math.max(0, confidence)),
        estimatedEnergy: parseFloat(estimatedEnergy.toFixed(2)),
        savings: parseFloat(savings.toFixed(2)),
        co2Avoided: parseFloat(co2Avoided.toFixed(1)),
      })
    } catch (error) {
      console.error('Error fetching solar metrics:', error)
    } finally {
      setLoading(false)
    }
  }

  const getClassColor = (className: string) => {
    switch (className) {
      case 'Excellent': return 'text-green-600 bg-green-100 border-green-200'
      case 'Good': return 'text-blue-600 bg-blue-100 border-blue-200'
      case 'Typical': return 'text-yellow-600 bg-yellow-100 border-yellow-200'
      case 'Fair': return 'text-orange-600 bg-orange-100 border-orange-200'
      case 'Poor': return 'text-red-600 bg-red-100 border-red-200'
      default: return 'text-gray-600 bg-gray-100 border-gray-200'
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
                Peak Sun Hours & Solar Energy
              </h3>
              <p className="text-xs sm:text-sm text-purple-100">
                {config.city} • {config.systemSize} kWp System
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
          {loading ? (
            <div className="animate-pulse space-y-4">
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {[1, 2, 3, 4].map(i => (
                  <div key={i} className="h-24 bg-gray-100 rounded-lg" />
                ))}
              </div>
            </div>
          ) : (
            <>
              {/* Top Metrics Row */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
                {/* PSH Today */}
                <div className="bg-gradient-to-br from-orange-50 to-yellow-50 rounded-lg p-3 sm:p-4 border border-orange-100">
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
                <div className="bg-gradient-to-br from-yellow-50 to-amber-50 rounded-lg p-3 sm:p-4 border border-yellow-100">
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
                <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-3 sm:p-4 border border-blue-100">
                  <div className="flex items-center gap-2 mb-2">
                    <Cloud className="w-4 h-4 text-blue-600" />
                    <p className="text-xs text-gray-600">Solar Day Class</p>
                  </div>
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-semibold border ${getClassColor(metrics.solarDayClass)}`}>
                    {metrics.solarDayClass}
                  </div>
                </div>

                {/* Confidence */}
                <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-3 sm:p-4 border border-purple-100">
                  <div className="flex items-center gap-2 mb-2">
                    <Activity className="w-4 h-4 text-purple-600" />
                    <p className="text-xs text-gray-600">Confidence</p>
                  </div>
                  <p className="text-xl sm:text-2xl font-bold text-purple-600">
                    {metrics.confidence}%
                  </p>
                  <p className="text-xs text-gray-500">clear sky</p>
                </div>
              </div>

              {/* Estimated Energy Section */}
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4 border border-green-100">
                <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-green-600" />
                  Today's Projections ({config.systemSize} kWp system)
                </h4>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {/* Energy Output */}
                  <div className="flex items-center justify-between sm:flex-col sm:items-start">
                    <span className="text-sm text-gray-600">Daily Output</span>
                    <span className="text-xl font-bold text-blue-600">
                      {metrics.estimatedEnergy} kWh
                    </span>
                  </div>

                  {/* Savings */}
                  <div className="flex items-center justify-between sm:flex-col sm:items-start">
                    <span className="text-sm text-gray-600 flex items-center gap-1">
                      <DollarSign className="w-3 h-3" /> Savings
                    </span>
                    <span className="text-xl font-bold text-green-600">
                      {formatCurrency(convert(metrics.savings, 'USD', config.currency), config.currency)}
                    </span>
                  </div>

                  {/* CO2 Avoided */}
                  <div className="flex items-center justify-between sm:flex-col sm:items-start">
                    <span className="text-sm text-gray-600">CO₂ Avoided</span>
                    <span className="text-xl font-bold text-emerald-600">
                      {metrics.co2Avoided} kg
                    </span>
                  </div>
                </div>

                {/* System Info */}
                <div className="mt-4 pt-4 border-t border-green-200 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-gray-500">
                  <div className="flex justify-between">
                    <span>PR:</span>
                    <span className="font-semibold text-purple-600">{(config.performanceRatio * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Tilt:</span>
                    <span className="font-semibold">{config.panelTilt}°</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Tariff:</span>
                    <span className="font-semibold">{formatCurrency(convert(config.electricityTariff, 'USD', config.currency), config.currency)}/kWh</span>
                  </div>
                  <div className="flex justify-between">
                    <span>CO₂ Factor:</span>
                    <span className="font-semibold">{(config.gridCO2Factor * 1000).toFixed(0)}g/kWh</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
