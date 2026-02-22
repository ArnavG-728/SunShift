'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import EnhancedDashboard from '@/components/EnhancedDashboard'
import ChatInterface from '@/components/ChatInterface'
import RealTimeWeather from '@/components/RealTimeWeather'
import GreenMetrics from '@/components/GreenMetrics'
import SolarMetrics from '@/components/SolarMetrics'
import SystemConfiguration from '@/components/SystemConfiguration'
import SmartRecommendations from '@/components/SmartRecommendations'
import UserGuide from '@/components/UserGuide'
import ValueGapDashboard from '@/components/ValueGapDashboard'
import SystemHealth from '@/components/SystemHealth'
import { Sun, Moon } from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Dashboard() {
    const [activeTab, setActiveTab] = useState<'dashboard' | 'recommendations' | 'chat'>('dashboard')
    const [mounted, setMounted] = useState(false)
    const [isNight, setIsNight] = useState(false)
    const { config, isLoading } = useSystemConfig()

    // Fetch day/night status based on location
    useEffect(() => {
        const fetchDayNightStatus = async () => {
            try {
                const response = await axios.get(`${API_BASE_URL}/realtime/current`, {
                    params: {
                        lat: config.latitude,
                        lon: config.longitude,
                        system_size: config.systemSize,
                        performance_ratio: config.performanceRatio
                    }
                })
                if (response.data.status === 'success') {
                    setIsNight(!response.data.data.is_daytime)
                }
            } catch (error) {
                console.error('Error fetching day/night status:', error)
            }
        }

        if (config.latitude && config.longitude) {
            fetchDayNightStatus()
            // Check every 5 minutes for day/night transitions
            const interval = setInterval(fetchDayNightStatus, 300000)
            return () => clearInterval(interval)
        }
    }, [config.latitude, config.longitude, config.systemSize, config.performanceRatio])

    useEffect(() => {
        setMounted(true)
    }, [])

    // Dynamic background classes based on day/night
    const bgClasses = isNight
        ? 'bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900'
        : 'bg-gradient-to-br from-orange-50 via-yellow-50 to-blue-50'

    // Dynamic header classes
    const headerClasses = isNight
        ? 'bg-slate-900/95 shadow-lg border-b border-slate-700/50 backdrop-blur-sm'
        : 'bg-white shadow-sm border-b'

    return (
        <main className={`min-h-screen transition-colors duration-1000 ${bgClasses}`}>
            {/* Header */}
            <header className={`sticky top-0 z-50 transition-colors duration-500 ${headerClasses}`}>
                <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8 py-3 sm:py-4">
                    <div className="flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-0">
                        <div className="flex items-center space-x-2 sm:space-x-3">
                            <div className={`p-1.5 sm:p-2 rounded-lg shadow-md transition-colors duration-500 ${isNight
                                ? 'bg-gradient-to-br from-indigo-500 to-purple-600'
                                : 'bg-gradient-to-br from-orange-500 to-yellow-500'
                                }`}>
                                {isNight ? (
                                    <Moon className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
                                ) : (
                                    <Sun className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
                                )}
                            </div>
                            <div>
                                <h1 className={`text-xl sm:text-2xl font-bold bg-clip-text text-transparent transition-colors duration-500 ${isNight
                                    ? 'bg-gradient-to-r from-indigo-400 to-purple-400'
                                    : 'bg-gradient-to-r from-orange-600 to-yellow-500'
                                    }`}>SunShift</h1>
                                <p className={`text-xs sm:text-sm hidden sm:block transition-colors duration-500 ${isNight ? 'text-slate-400' : 'text-gray-500'
                                    }`}>
                                    {isLoading ? 'Loading...' : `${config.city} • ${config.systemSize} kWp System`}
                                </p>
                            </div>
                        </div>

                        {/* Tab Navigation */}
                        <nav className={`flex space-x-1 p-1 rounded-lg w-full sm:w-auto overflow-x-auto transition-colors duration-500 ${isNight ? 'bg-slate-800/80' : 'bg-gray-100'
                            }`}>
                            <button
                                onClick={() => setActiveTab('dashboard')}
                                className={`flex-1 sm:flex-none px-3 sm:px-4 py-2 rounded-md text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${activeTab === 'dashboard'
                                    ? isNight
                                        ? 'bg-slate-700 text-indigo-400 shadow-md'
                                        : 'bg-white text-orange-600 shadow-md'
                                    : isNight
                                        ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
                                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                                    }`}
                            >
                                📊 Dashboard
                            </button>
                            <button
                                onClick={() => setActiveTab('recommendations')}
                                className={`flex-1 sm:flex-none px-3 sm:px-4 py-2 rounded-md text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${activeTab === 'recommendations'
                                    ? isNight
                                        ? 'bg-slate-700 text-indigo-400 shadow-md'
                                        : 'bg-white text-orange-600 shadow-md'
                                    : isNight
                                        ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
                                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                                    }`}
                            >
                                💡 Smart Tips
                            </button>
                            <button
                                onClick={() => setActiveTab('chat')}
                                className={`flex-1 sm:flex-none px-3 sm:px-4 py-2 rounded-md text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${activeTab === 'chat'
                                    ? isNight
                                        ? 'bg-slate-700 text-indigo-400 shadow-md'
                                        : 'bg-white text-orange-600 shadow-md'
                                    : isNight
                                        ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
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

                {/* Pitch Feature Cards: Value Gap & System Health */}
                {mounted && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 mb-4 sm:mb-6">
                        <div className="w-full">
                            <ValueGapDashboard />
                        </div>
                        <div className="w-full">
                            <SystemHealth />
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
