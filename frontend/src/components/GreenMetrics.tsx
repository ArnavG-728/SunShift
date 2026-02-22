'use client'

import { useState, useEffect } from 'react'
import { Leaf, Zap, TrendingDown, TreePine, Coins, ChevronDown, ChevronUp } from 'lucide-react'
import axios from 'axios'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import { useCurrency } from '@/lib/useCurrency'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface WalletLifetime {
  total_energy_kwh: number
  total_co2_avoided_kg: number
  total_co2_avoided_tons: number
  total_credit_value_usd: number
  entries: number
}

interface WalletData {
  lifetime: WalletLifetime
  equivalents: { trees_year_equivalent: number; car_km_avoided: number }
  credit_rate: { price_per_ton_usd: number; source: string }
  monthly_breakdown: Array<{ month: string; co2_kg: number; credit_usd: number; energy_kwh: number }>
}

export default function GreenMetrics() {
  const { config } = useSystemConfig()
  const { convert, formatCurrency } = useCurrency()
  const [isOpen, setIsOpen] = useState(true)
  const [loading, setLoading] = useState(true)

  // Today's real-time metrics
  const [today, setToday] = useState({
    energyGenerated: 0,
    co2Avoided: 0,
    treesEquivalent: 0,
    kmDriven: 0,
    savingsToday: 0,
  })

  // Lifetime wallet data from Carbon Wallet backend
  const [wallet, setWallet] = useState<WalletData | null>(null)

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, 120000)
    return () => clearInterval(interval)
  }, [config.latitude, config.longitude, config.systemSize, config.gridCO2Factor, config.electricityTariff])

  const fetchAll = async () => {
    try {
      setLoading(true)
      await Promise.all([fetchTodayMetrics(), fetchWallet()])
    } finally {
      setLoading(false)
    }
  }

  const fetchTodayMetrics = async () => {
    try {
      const forecastRes = await axios.get(`${API_BASE_URL}/forecast/24h`).catch(() => null)

      let totalEnergyToday = 0
      if (forecastRes?.data?.data && Array.isArray(forecastRes.data.data)) {
        totalEnergyToday = forecastRes.data.data.reduce((sum: number, hour: any) => {
          const energy = hour.predicted_output_kWh || hour.energy_output_kWh || 0
          return sum + (typeof energy === 'number' ? energy : parseFloat(energy) || 0)
        }, 0)
      }

      if (totalEnergyToday === 0) {
        const currentRes = await axios.get(`${API_BASE_URL}/realtime/current`, {
          params: {
            lat: config.latitude,
            lon: config.longitude,
            system_size: config.systemSize,
            performance_ratio: config.performanceRatio,
          },
        }).catch(() => null)

        if (currentRes?.data?.data?.energy_output_kWh) {
          const daylightHours = currentRes.data.data.daylight_hours || 6
          totalEnergyToday = currentRes.data.data.energy_output_kWh * daylightHours
        }
      }

      const co2Factor = config.gridCO2Factor || 0.70
      const co2Avoided = totalEnergyToday * co2Factor
      // Tree absorbs ~21 kg CO₂/year — daily equivalent = 21/365 ≈ 0.058 kg
      const treesEquivalent = co2Avoided / 0.058
      const kmDriven = co2Avoided / 0.12

      // Split into self-consumed vs exported (consistent with SmartRecommendations)
      const avgConsumptionPerHour = 1.2 // typical residential kW
      const totalConsumption = avgConsumptionPerHour * 24
      const selfConsumed = Math.min(totalEnergyToday, totalConsumption)
      const exported = Math.max(0, totalEnergyToday - totalConsumption)
      const savingsToday = (selfConsumed * config.electricityTariff) + (exported * config.feedInTariff)

      setToday({ energyGenerated: totalEnergyToday, co2Avoided, treesEquivalent, kmDriven, savingsToday })
    } catch (err) {
      console.error('Error fetching today metrics:', err)
    }
  }

  const fetchWallet = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/carbon-wallet`, {
        params: { lat: config.latitude, lon: config.longitude, grid_co2_factor: config.gridCO2Factor },
      })
      if (res.data?.data?.status === 'success') {
        setWallet(res.data.data)
      }
    } catch (err) {
      console.error('Error fetching carbon wallet:', err)
    }
  }

  const displaySavings = convert(today.savingsToday, 'USD', config.currency)
  const displayMonthlySavings = convert(today.savingsToday * 30, 'USD', config.currency)  // estimate

  const displayUSD = (usd: number) => {
    const converted = convert(usd, 'USD', config.currency)
    return formatCurrency(converted, config.currency)
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-500 to-emerald-500 p-4">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              🌱 Environmental Impact & Carbon Wallet
            </h3>
            <p className="text-xs text-green-100">Real-time carbon savings · Lifetime credit portfolio</p>
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
              {/* ── Today's Impact ── */}
              <p className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-2">☀️ Today</p>
              <div className="grid grid-cols-2 gap-3 mb-5">
                <div className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-lg p-3 border border-yellow-100">
                  <div className="flex items-center gap-1.5 mb-1">
                    <Zap className="w-4 h-4 text-yellow-500" />
                    <span className="text-[11px] text-gray-500">Generated</span>
                  </div>
                  <p className="text-lg font-bold text-gray-800">{today.energyGenerated.toFixed(1)} kWh</p>
                  <p className="text-[10px] text-green-600">{formatCurrency(displaySavings, config.currency)} saved</p>
                </div>

                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-3 border border-green-100">
                  <div className="flex items-center gap-1.5 mb-1">
                    <TrendingDown className="w-4 h-4 text-green-500" />
                    <span className="text-[11px] text-gray-500">CO₂ Avoided</span>
                  </div>
                  <p className="text-lg font-bold text-green-700">{today.co2Avoided.toFixed(1)} kg</p>
                  <p className="text-[10px] text-gray-400">@ {(config.gridCO2Factor * 1000).toFixed(0)} g/kWh</p>
                </div>
              </div>

              {/* ── Lifetime Carbon Wallet ── */}
              {wallet && (
                <>
                  <p className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-2">🌍 Carbon Credit Wallet — Lifetime</p>

                  {/* Wallet Balance */}
                  <div className="bg-gradient-to-br from-teal-50 to-cyan-50 rounded-xl p-4 border border-teal-200 mb-4 flex items-center justify-between">
                    <div>
                      <p className="text-[10px] text-gray-400 uppercase">Wallet Balance</p>
                      <p className="text-2xl font-bold text-teal-700">
                        {displayUSD(wallet.lifetime.total_credit_value_usd)}
                      </p>
                      <p className="text-[10px] text-gray-400">
                        @ {displayUSD(wallet.credit_rate.price_per_ton_usd)}/ton • {wallet.credit_rate.source}
                      </p>
                    </div>
                    <div className="text-right space-y-1">
                      <p className="text-sm">
                        <span className="text-gray-400">CO₂: </span>
                        <span className="font-semibold text-green-700">{wallet.lifetime.total_co2_avoided_kg.toFixed(1)} kg</span>
                      </p>
                      <p className="text-sm">
                        <span className="text-gray-400">🌳 </span>
                        <span className="font-semibold">{wallet.equivalents.trees_year_equivalent.toFixed(0)} tree-yrs</span>
                      </p>
                      <p className="text-sm">
                        <span className="text-gray-400">🚗 </span>
                        <span className="font-semibold">{wallet.equivalents.car_km_avoided.toFixed(0)} km</span>
                      </p>
                    </div>
                  </div>

                  {/* Monthly Ledger */}
                  {wallet.monthly_breakdown.length > 0 && (
                    <div className="space-y-1 max-h-28 overflow-y-auto">
                      {wallet.monthly_breakdown.map((m, i) => (
                        <div key={i} className="flex items-center justify-between bg-gray-50 rounded px-3 py-1.5 text-xs">
                          <span className="font-medium text-gray-600">{m.month}</span>
                          <span className="text-gray-400">{m.energy_kwh.toFixed(1)} kWh</span>
                          <span className="text-gray-500">{m.co2_kg.toFixed(1)} kg CO₂</span>
                          <span className="text-teal-600 font-semibold">{displayUSD(m.credit_usd)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}

              {/* System Info + Monthly Projection */}
              <div className="mt-4 p-3 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-100">
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <span>☀️ {config.systemSize} kWp · ⚡ {(config.gridCO2Factor * 1000).toFixed(0)} g/kWh</span>
                  <span className="font-semibold text-green-600">
                    Monthly (est.): {formatCurrency(displayMonthlySavings, config.currency)}
                  </span>
                </div>
              </div>

              {/* Green Badge */}
              <div className="mt-3 flex items-center justify-center gap-2 p-2.5 bg-green-100 rounded-lg border-2 border-green-300">
                <Leaf className="w-4 h-4 text-green-700" />
                <span className="text-xs font-semibold text-green-800">
                  {today.co2Avoided > 0 ? 'Net Positive Environmental Impact ✓' : 'Run forecast to see impact'}
                </span>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
