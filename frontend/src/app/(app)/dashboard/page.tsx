'use client'

import { useState, useEffect } from 'react'
import SolarMetrics from '@/components/SolarMetrics'
import RealTimeWeather from '@/components/RealTimeWeather'
import GreenMetrics from '@/components/GreenMetrics'
import { Sun } from 'lucide-react'

export default function DashboardOverview() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <div className="space-y-4 sm:space-y-5 md:space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2.5 sm:gap-3 mb-1">
          <div className="p-1.5 sm:p-2 rounded-lg sm:rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 shadow-lg shadow-orange-500/20">
            <Sun className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg sm:text-xl md:text-2xl 2xl:text-3xl font-bold text-gray-900">Dashboard Overview</h1>
            <p className="text-xs sm:text-sm text-gray-500">Your solar system at a glance</p>
          </div>
        </div>
      </div>

      {/* Solar Metrics - Full Width Hero */}
      {mounted && (
        <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
          <SolarMetrics />
        </div>
      )}

      {/* Weather & Green Metrics Grid */}
      {mounted && (
        <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-2 gap-3 sm:gap-4 md:gap-5 lg:gap-6 animate-in fade-in slide-in-from-bottom-3 duration-700">
          <div className="w-full min-w-0">
            <RealTimeWeather />
          </div>
          <div className="w-full min-w-0">
            <GreenMetrics />
          </div>
        </div>
      )}
    </div>
  )
}
