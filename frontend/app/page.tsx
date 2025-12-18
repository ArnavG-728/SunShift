'use client'

import { useState, useEffect } from 'react'
import EnhancedDashboard from '@/components/EnhancedDashboard'
import ChatInterface from '@/components/ChatInterface'
import RealTimeWeather from '@/components/RealTimeWeather'
import GreenMetrics from '@/components/GreenMetrics'
import SolarMetrics from '@/components/SolarMetrics'
import SystemConfiguration from '@/components/SystemConfiguration'
import SmartRecommendations from '@/components/SmartRecommendations'
import UserGuide from '@/components/UserGuide'
import { Sun } from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'

export default function Home() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'recommendations' | 'chat'>('dashboard')
  const [mounted, setMounted] = useState(false)
  const { config, isLoading } = useSystemConfig()

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <main className="min-h-screen bg-gradient-to-br from-orange-50 via-yellow-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8 py-3 sm:py-4">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-0">
            <div className="flex items-center space-x-2 sm:space-x-3">
              <div className="bg-gradient-to-br from-orange-500 to-yellow-500 p-1.5 sm:p-2 rounded-lg shadow-md">
                <Sun className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-orange-600 to-yellow-500 bg-clip-text text-transparent">SunShift</h1>
                <p className="text-xs sm:text-sm text-gray-500 hidden sm:block">
                  {isLoading ? 'Loading...' : `${config.city} • ${config.systemSize} kWp System`}
                </p>
              </div>
            </div>

            {/* Tab Navigation */}
            <nav className="flex space-x-1 bg-gray-100 p-1 rounded-lg w-full sm:w-auto overflow-x-auto">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`flex-1 sm:flex-none px-3 sm:px-4 py-2 rounded-md text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${activeTab === 'dashboard'
                  ? 'bg-white text-orange-600 shadow-md'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
              >
                📊 Dashboard
              </button>
              <button
                onClick={() => setActiveTab('recommendations')}
                className={`flex-1 sm:flex-none px-3 sm:px-4 py-2 rounded-md text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${activeTab === 'recommendations'
                  ? 'bg-white text-orange-600 shadow-md'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
              >
                💡 Smart Tips
              </button>
              <button
                onClick={() => setActiveTab('chat')}
                className={`flex-1 sm:flex-none px-3 sm:px-4 py-2 rounded-md text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${activeTab === 'chat'
                  ? 'bg-white text-orange-600 shadow-md'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
              >
                🤖 AI Assistant
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8 py-4 sm:py-6 lg:py-8">
        {/* System Configuration - Now uses global context */}
        <div className="mb-4 sm:mb-6">
          <SystemConfiguration />
        </div>

        {/* Solar Metrics - Full Width */}
        {mounted && (
          <div className="mb-4 sm:mb-6">
            <SolarMetrics />
          </div>
        )}

        {/* Real-Time Weather & Green Metrics */}
        {mounted && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 mb-4 sm:mb-6">
            <div className="w-full">
              <RealTimeWeather />
            </div>
            <div className="w-full">
              <GreenMetrics />
            </div>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'dashboard' && (
          <EnhancedDashboard />
        )}

        {activeTab === 'recommendations' && (
          <SmartRecommendations />
        )}

        {activeTab === 'chat' && (
          <ChatInterface />
        )}
      </div>

      {/* Floating User Guide */}
      <UserGuide />
    </main>
  )
}
