'use client'

import SmartRecommendations from '@/components/SmartRecommendations'
import { Lightbulb } from 'lucide-react'

export default function SmartTipsPage() {
  return (
    <div className="space-y-4 sm:space-y-5 md:space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2.5 sm:gap-3 mb-1">
          <div className="p-1.5 sm:p-2 rounded-lg sm:rounded-xl bg-gradient-to-br from-yellow-500 to-orange-500 shadow-lg shadow-yellow-500/20">
            <Lightbulb className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg sm:text-xl md:text-2xl 2xl:text-3xl font-bold text-gray-900">Smart Tips</h1>
            <p className="text-xs sm:text-sm text-gray-500">AI-driven optimization for your energy usage</p>
          </div>
        </div>
      </div>

      {/* Smart Recommendations */}
      <div className="animate-in fade-in slide-in-from-bottom-2 duration-500 min-w-0">
        <SmartRecommendations />
      </div>
    </div>
  )
}
