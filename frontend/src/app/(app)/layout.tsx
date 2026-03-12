'use client'

import { useState, useEffect, useCallback } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import axios from 'axios'
import {
  Sun, Moon, LayoutDashboard, BarChart3, Lightbulb,
  Bot, Settings, ChevronLeft, ChevronRight, Menu, X, Zap
} from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import UserGuide from '@/components/UserGuide'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const NAV_ITEMS = [
  { href: '/dashboard', label: 'Overview', icon: LayoutDashboard, description: 'At-a-glance metrics' },
  { href: '/analytics', label: 'Analytics', icon: BarChart3, description: 'Forecasts & deep data' },
  { href: '/smart-tips', label: 'Smart Tips', icon: Lightbulb, description: 'Optimization advice' },
  { href: '/ai-assistant', label: 'AI Assistant', icon: Bot, description: 'Chat with SunShift AI' },
  { href: '/settings', label: 'Settings', icon: Settings, description: 'System configuration' },
]

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const { config, isLoading } = useSystemConfig()
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [isNight, setIsNight] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [isTablet, setIsTablet] = useState(false)

  // Responsive breakpoint detection
  useEffect(() => {
    const checkSize = () => {
      const w = window.innerWidth
      setIsMobile(w < 768)
      setIsTablet(w >= 768 && w < 1024)
      // Auto-collapse sidebar on tablet
      if (w >= 768 && w < 1024) setCollapsed(true)
      if (w >= 1024) setCollapsed(false)
    }
    checkSize()
    window.addEventListener('resize', checkSize)
    return () => window.removeEventListener('resize', checkSize)
  }, [])

  // Fetch day/night status
  useEffect(() => {
    const fetchDayNightStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/realtime/current`, {
          params: {
            lat: config.latitude,
            lon: config.longitude,
            system_size: config.systemSize,
            performance_ratio: config.performanceRatio,
            panel_tilt: config.panelTilt,
            panel_azimuth: config.panelAzimuth
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
      const interval = setInterval(fetchDayNightStatus, 300000)
      return () => clearInterval(interval)
    }
  }, [config.latitude, config.longitude, config.systemSize, config.performanceRatio, config.panelTilt, config.panelAzimuth])

  // Close mobile drawer on route change
  useEffect(() => {
    setMobileOpen(false)
  }, [pathname])

  // Prevent body scroll when mobile drawer is open
  useEffect(() => {
    if (mobileOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => { document.body.style.overflow = '' }
  }, [mobileOpen])

  // Theme classes
  const themeBg = isNight
    ? 'bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950'
    : 'bg-gradient-to-br from-orange-50/80 via-amber-50/50 to-sky-50/80'

  const sidebarBg = isNight
    ? 'bg-slate-900/95 border-slate-700/50'
    : 'bg-white/90 border-gray-200/80'

  const sidebarText = isNight ? 'text-slate-300' : 'text-gray-600'
  const sidebarTextActive = isNight ? 'text-white' : 'text-gray-900'
  const sidebarHover = isNight ? 'hover:bg-slate-800/80' : 'hover:bg-orange-50/80'
  const sidebarActiveBg = isNight
    ? 'bg-gradient-to-r from-indigo-600/30 to-purple-600/20 border-l-[3px] border-indigo-400'
    : 'bg-gradient-to-r from-orange-100/80 to-amber-50/60 border-l-[3px] border-orange-500'

  const brandGradient = isNight
    ? 'from-indigo-400 to-purple-400'
    : 'from-orange-600 to-amber-500'

  const iconBg = isNight
    ? 'from-indigo-500 to-purple-600'
    : 'from-orange-500 to-amber-500'

  // Bottom bar colors
  const bottomBarBg = isNight
    ? 'bg-slate-900/98 border-slate-700/50'
    : 'bg-white/98 border-gray-200'

  const bottomBarText = isNight ? 'text-slate-500' : 'text-gray-400'
  const bottomBarActive = isNight ? 'text-indigo-400' : 'text-orange-600'

  // Sidebar width
  const sidebarWidth = collapsed ? 'w-[72px]' : 'w-64'
  const mainMargin = isMobile ? '' : collapsed ? 'md:ml-[72px]' : 'lg:ml-64 md:ml-[72px]'

  return (
    <div className={`min-h-screen transition-colors duration-700 ${themeBg}`}>

      {/* ========== MOBILE: Top Bar ========== */}
      {isMobile && (
        <header
          className={`fixed top-0 left-0 right-0 z-40 border-b backdrop-blur-md safe-top
            ${isNight ? 'bg-slate-900/95 border-slate-700/50' : 'bg-white/95 border-gray-200'}
          `}
        >
          <div className="flex items-center justify-between px-4 h-14">
            <Link href="/dashboard" className="flex items-center gap-2">
              <div className={`p-1.5 rounded-lg shadow-md bg-gradient-to-br ${iconBg}`}>
                {isNight ? <Moon className="h-4 w-4 text-white" /> : <Sun className="h-4 w-4 text-white" />}
              </div>
              <span className={`text-base font-bold bg-clip-text text-transparent bg-gradient-to-r ${brandGradient}`}>
                SunShift
              </span>
            </Link>
            <div className="flex items-center gap-2">
              <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-medium ${isNight ? 'bg-green-900/30 text-green-400' : 'bg-green-50 text-green-700 border border-green-200'}`}>
                <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                Online
              </div>
            </div>
          </div>
        </header>
      )}

      {/* ========== MOBILE: Bottom Tab Bar ========== */}
      {isMobile && (
        <nav
          className={`fixed bottom-0 left-0 right-0 z-40 border-t backdrop-blur-md safe-bottom ${bottomBarBg}`}
          id="mobile-bottom-nav"
        >
          <div className="flex items-stretch justify-around h-16 px-1">
            {NAV_ITEMS.map((item) => {
              const isActive = pathname === item.href
              const Icon = item.icon
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex flex-col items-center justify-center flex-1 min-w-0 py-1 transition-colors touch-target
                    ${isActive ? bottomBarActive : bottomBarText}
                  `}
                  id={`mobile-nav-${item.label.toLowerCase().replace(' ', '-')}`}
                >
                  <Icon className={`w-5 h-5 mb-0.5 ${isActive ? 'scale-110' : ''} transition-transform`} />
                  <span className={`text-[10px] font-medium truncate max-w-full px-1 ${isActive ? 'font-bold' : ''}`}>
                    {item.label}
                  </span>
                  {isActive && (
                    <span className={`absolute top-0 w-8 h-0.5 rounded-full ${isNight ? 'bg-indigo-400' : 'bg-orange-500'}`} />
                  )}
                </Link>
              )
            })}
          </div>
        </nav>
      )}

      {/* ========== TABLET/DESKTOP: Sidebar ========== */}
      {!isMobile && (
        <>
          {/* Mobile Overlay (for tablet sheet) */}
          {isTablet && mobileOpen && (
            <div
              className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
              onClick={() => setMobileOpen(false)}
            />
          )}

          <aside
            className={`fixed top-0 left-0 z-50 h-full border-r backdrop-blur-md transition-all duration-300 ease-in-out
              ${sidebarBg}
              ${isTablet ? 'w-[72px]' : collapsed ? 'w-[72px]' : 'w-64'}
            `}
            id="sidebar-nav"
          >
            {/* Sidebar Header */}
            <div className="h-16 flex items-center justify-between px-3 lg:px-4 border-b border-inherit">
              <Link href="/dashboard" className="flex items-center gap-2.5 overflow-hidden">
                <div className={`p-1.5 rounded-lg shadow-md bg-gradient-to-br ${iconBg} flex-shrink-0`}>
                  {isNight ? <Moon className="h-5 w-5 text-white" /> : <Sun className="h-5 w-5 text-white" />}
                </div>
                {!collapsed && !isTablet && (
                  <div className="min-w-0">
                    <h1 className={`text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r ${brandGradient} truncate`}>
                      SunShift
                    </h1>
                    <p className={`text-[10px] ${isNight ? 'text-slate-500' : 'text-gray-400'} truncate`}>
                      {isLoading ? 'Loading...' : `${config.city} • ${config.systemSize} kWp`}
                    </p>
                  </div>
                )}
              </Link>
            </div>

            {/* Nav Links */}
            <nav className="flex-1 px-2 py-4 space-y-1">
              {NAV_ITEMS.map((item) => {
                const isActive = pathname === item.href
                const Icon = item.icon
                const showLabels = !collapsed && !isTablet
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200
                      ${isActive
                        ? `${sidebarActiveBg} ${sidebarTextActive}`
                        : `${sidebarText} ${sidebarHover}`
                      }
                      ${!showLabels ? 'justify-center' : ''}
                    `}
                    title={!showLabels ? item.label : undefined}
                    id={`nav-${item.label.toLowerCase().replace(' ', '-')}`}
                  >
                    <Icon className={`w-5 h-5 flex-shrink-0 transition-colors ${
                      isActive
                        ? isNight ? 'text-indigo-400' : 'text-orange-600'
                        : isNight ? 'text-slate-400 group-hover:text-slate-200' : 'text-gray-400 group-hover:text-gray-700'
                    }`} />
                    {showLabels && (
                      <div className="min-w-0">
                        <span className="block truncate">{item.label}</span>
                        {!isActive && (
                          <span className={`block text-[10px] truncate ${isNight ? 'text-slate-500' : 'text-gray-400'}`}>
                            {item.description}
                          </span>
                        )}
                      </div>
                    )}
                  </Link>
                )
              })}
            </nav>

            {/* Collapse Button (Desktop only, not tablet) */}
            {!isTablet && (
              <div className="hidden lg:block px-2 pb-4">
                <button
                  onClick={() => setCollapsed(!collapsed)}
                  className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all
                    ${isNight ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'}
                  `}
                  id="sidebar-collapse-btn"
                >
                  {collapsed ? <ChevronRight className="w-4 h-4" /> : (
                    <>
                      <ChevronLeft className="w-4 h-4" />
                      <span>Collapse</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Status indicator (expanded desktop only) */}
            {!collapsed && !isTablet && (
              <div className={`mx-3 mb-4 p-3 rounded-xl ${isNight ? 'bg-slate-800/80' : 'bg-gradient-to-r from-green-50 to-emerald-50 border border-green-100'}`}>
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Zap className={`w-4 h-4 ${isNight ? 'text-green-400' : 'text-green-600'}`} />
                    <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  </div>
                  <span className={`text-xs font-medium ${isNight ? 'text-green-400' : 'text-green-700'}`}>
                    System Online
                  </span>
                </div>
              </div>
            )}
          </aside>
        </>
      )}

      {/* ========== MAIN CONTENT ========== */}
      <div className={`transition-all duration-300 ${mainMargin}`}>
        <main className="min-h-screen">
          <div className={`
            mx-auto
            px-3 sm:px-4 md:px-6 lg:px-8 2xl:px-10
            py-4 sm:py-5 md:py-6 lg:py-8
            ${isMobile ? 'pt-[72px] pb-24' : 'pt-6'}
            max-w-[1800px]
          `}>
            {children}
          </div>
        </main>
      </div>

      {/* Floating User Guide */}
      <UserGuide />
    </div>
  )
}
