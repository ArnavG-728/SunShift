'use client'

import SystemConfiguration from '@/components/SystemConfiguration'
import { Settings } from 'lucide-react'

export default function SettingsPage() {
  return (
    <div className="space-y-4 sm:space-y-5 md:space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2.5 sm:gap-3 mb-1">
          <div className="p-1.5 sm:p-2 rounded-lg sm:rounded-xl bg-gradient-to-br from-teal-500 to-indigo-500 shadow-lg shadow-teal-500/20">
            <Settings className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg sm:text-xl md:text-2xl 2xl:text-3xl font-bold text-gray-900">Settings</h1>
            <p className="text-xs sm:text-sm text-gray-500">Configure your solar system parameters</p>
          </div>
        </div>
      </div>

      {/* System Configuration */}
      <div className="animate-in fade-in slide-in-from-bottom-2 duration-500 min-w-0">
        <SystemConfiguration />
      </div>
    </div>
  )
}
