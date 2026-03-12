'use client'

import { useState, useEffect } from 'react'
import EnhancedDashboard from '@/components/EnhancedDashboard'
import ValueGapDashboard from '@/components/ValueGapDashboard'
import SystemHealth from '@/components/SystemHealth'
import { BarChart3 } from 'lucide-react'

export default function AnalyticsPage() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <div className="space-y-4 sm:space-y-5 md:space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2.5 sm:gap-3 mb-1">
          <div className="p-1.5 sm:p-2 rounded-lg sm:rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg shadow-blue-500/20">
            <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg sm:text-xl md:text-2xl 2xl:text-3xl font-bold text-gray-900">Analytics</h1>
            <p className="text-xs sm:text-sm text-gray-500">Forecasts, risk analysis, and system performance</p>
          </div>
        </div>
      </div>

      {/* Enhanced Dashboard (Forecasts, Risk, Cloud, Charts) */}
      {mounted && (
        <div className="animate-in fade-in slide-in-from-bottom-2 duration-500 min-w-0 overflow-x-hidden">
          <EnhancedDashboard />
        </div>
      )}

      {/* Value Gap & System Health */}
      {mounted && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4 md:gap-5 lg:gap-6 animate-in fade-in slide-in-from-bottom-3 duration-700">
          <div className="w-full min-w-0">
            <ValueGapDashboard />
          </div>
          <div className="w-full min-w-0">
            <SystemHealth />
          </div>
        </div>
      )}
    </div>
  )
}
