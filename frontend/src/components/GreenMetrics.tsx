'use client'

import { useState, useEffect } from 'react'
import { Leaf, Zap, TrendingDown, TreePine, ChevronDown, ChevronUp } from 'lucide-react'
import axios from 'axios'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import { useCurrency } from '@/lib/useCurrency'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function GreenMetrics() {
  const { config } = useSystemConfig()
  const { convert, formatCurrency } = useCurrency()
  const [isOpen, setIsOpen] = useState(true)
  const [metrics, setMetrics] = useState({
    energyGenerated: 0,
    co2Avoided: 0,
    treesEquivalent: 0,
    kmDriven: 0,
    gridCO2Factor: 0.70,
    savingsToday: 0  // Always stored in USD
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchGreenMetrics()
    const interval = setInterval(fetchGreenMetrics, 120000) // Update every 2 minutes
    return () => clearInterval(interval)
  }, [config.latitude, config.longitude, config.systemSize, config.gridCO2Factor, config.electricityTariff])

  const fetchGreenMetrics = async () => {
    try {
      // Fetch today's forecast to calculate real environmental impact
      const forecastRes = await axios.get(`${API_BASE_URL}/forecast/24h`).catch(() => null)

      let totalEnergyToday = 0
      if (forecastRes?.data?.data && Array.isArray(forecastRes.data.data)) {
        // Sum up predicted energy for next 24 hours
        totalEnergyToday = forecastRes.data.data.reduce((sum: number, hour: any) => {
          const energy = hour.predicted_output_kWh || hour.energy_output_kWh || 0
          return sum + (typeof energy === 'number' ? energy : parseFloat(energy) || 0)
        }, 0)
      }

      // If no forecast data, fetch current weather and estimate
      if (totalEnergyToday === 0) {
        const currentRes = await axios.get(`${API_BASE_URL}/realtime/current`, {
          params: {
            lat: config.latitude,
            lon: config.longitude,
            system_size: config.systemSize,
            performance_ratio: config.performanceRatio
          }
        }).catch(() => null)

        if (currentRes?.data?.data?.energy_output_kWh) {
          // Estimate daily based on current hour output * daylight hours
          totalEnergyToday = currentRes.data.data.energy_output_kWh * 8 // ~8 productive hours
        }
      }

      // Calculate environmental metrics using user's config
      const co2Factor = config.gridCO2Factor || 0.70 // kg CO2 per kWh
      const co2Avoided = totalEnergyToday * co2Factor

      // Trees absorb ~21kg CO2/year = ~0.058 kg/day
      const treesEquivalent = co2Avoided / 0.058

      // Average car emits ~0.12 kg CO2/km
      const kmDriven = co2Avoided / 0.12

      // Financial savings - store in USD for conversion
      const savingsToday = totalEnergyToday * config.electricityTariff

      setMetrics({
        energyGenerated: totalEnergyToday,
        co2Avoided,
        treesEquivalent,
        kmDriven,
        gridCO2Factor: co2Factor,
        savingsToday
      })

    } catch (error) {
      console.error('Error fetching green metrics:', error)
    } finally {
      setLoading(false)
    }
  }

  // Convert savings to user's selected currency
  const displaySavings = convert(metrics.savingsToday, 'USD', config.currency)
  const displayMonthlySavings = convert(metrics.savingsToday * 30, 'USD', config.currency)

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-500 to-emerald-500 p-4">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              🌱 Environmental Impact
            </h3>
            <p className="text-xs text-green-100">Real-time carbon savings for {config.city}</p>
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
          {loading ? (
            <div className="animate-pulse text-center py-8 text-gray-400">Loading real data...</div>
          ) : (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {/* Energy Generated */}
                <div className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-lg p-4 border border-yellow-100">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-5 h-5 text-yellow-500" />
                    <span className="text-xs text-gray-600">Generated Today</span>
                  </div>
                  <div className="text-xl font-bold text-gray-800">
                    {metrics.energyGenerated.toFixed(1)} kWh
                  </div>
                  <div className="text-xs text-green-600 mt-1">
                    {formatCurrency(displaySavings, config.currency)} saved
                  </div>
                </div>

                {/* CO2 Avoided */}
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-4 border border-green-100">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingDown className="w-5 h-5 text-green-500" />
                    <span className="text-xs text-gray-600">CO₂ Avoided</span>
                  </div>
                  <div className="text-xl font-bold text-green-700">
                    {metrics.co2Avoided.toFixed(1)} kg
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    @ {(metrics.gridCO2Factor * 1000).toFixed(0)} g/kWh
                  </div>
                </div>

                {/* Trees Equivalent */}
                <div className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-lg p-4 border border-emerald-100">
                  <div className="flex items-center gap-2 mb-2">
                    <TreePine className="w-5 h-5 text-emerald-500" />
                    <span className="text-xs text-gray-600">Trees Equivalent</span>
                  </div>
                  <div className="text-xl font-bold text-gray-800">
                    {metrics.treesEquivalent.toFixed(1)} 🌳
                  </div>
                  <div className="text-xs text-gray-500 mt-1">tree-days</div>
                </div>

                {/* Driving Offset */}
                <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-4 border border-blue-100">
                  <div className="flex items-center gap-2 mb-2">
                    <Leaf className="w-5 h-5 text-blue-500" />
                    <span className="text-xs text-gray-600">Driving Offset</span>
                  </div>
                  <div className="text-xl font-bold text-gray-800">
                    {metrics.kmDriven.toFixed(0)} km
                  </div>
                  <div className="text-xs text-gray-500 mt-1">car equivalent</div>
                </div>
              </div>

              {/* Impact Summary */}
              <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-100">
                <h4 className="font-semibold text-sm text-gray-700 mb-2">Your System's Impact</h4>
                <div className="space-y-2 text-sm text-gray-600">
                  <div className="flex items-center justify-between">
                    <span>☀️ System Size:</span>
                    <span className="font-semibold">{config.systemSize} kWp</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>⚡ Grid CO₂ Factor:</span>
                    <span className="font-semibold">{(config.gridCO2Factor * 1000).toFixed(0)} g/kWh</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>💰 Monthly Projection:</span>
                    <span className="font-semibold text-green-600">
                      {formatCurrency(displayMonthlySavings, config.currency)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>Currency:</span>
                    <span>{config.currency}</span>
                  </div>
                </div>
              </div>

              {/* Green Badge */}
              <div className="mt-4 flex items-center justify-center gap-2 p-3 bg-green-100 rounded-lg border-2 border-green-300">
                <Leaf className="w-5 h-5 text-green-700" />
                <span className="text-sm font-semibold text-green-800">
                  {metrics.co2Avoided > 0 ? 'Net Positive Environmental Impact ✓' : 'Run forecast to see impact'}
                </span>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
