'use client'

/**
 * System Configuration Context
 * Provides global access to user preferences and system configuration
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { SystemConfig, loadUserPreferences, saveUserPreferences, DEFAULT_CONFIG } from './userPreferences'

interface SystemConfigContextType {
  config: SystemConfig
  updateConfig: (updates: Partial<SystemConfig>) => void
  resetConfig: () => void
  isLoading: boolean
}

const SystemConfigContext = createContext<SystemConfigContextType | undefined>(undefined)

export function SystemConfigProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<SystemConfig>(DEFAULT_CONFIG)
  const [isLoading, setIsLoading] = useState(true)

  // Load preferences on mount
  useEffect(() => {
    const loadedConfig = loadUserPreferences()
    setConfig(loadedConfig)
    setIsLoading(false)
  }, [])

  const updateConfig = (updates: Partial<SystemConfig>) => {
    const newConfig = { ...config, ...updates }
    setConfig(newConfig)
    saveUserPreferences(newConfig)
  }

  const resetConfig = () => {
    setConfig(DEFAULT_CONFIG)
    saveUserPreferences(DEFAULT_CONFIG)
  }

  return (
    <SystemConfigContext.Provider value={{ config, updateConfig, resetConfig, isLoading }}>
      {children}
    </SystemConfigContext.Provider>
  )
}

export function useSystemConfig() {
  const context = useContext(SystemConfigContext)
  if (context === undefined) {
    throw new Error('useSystemConfig must be used within a SystemConfigProvider')
  }
  return context
}
