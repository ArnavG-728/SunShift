'use client'

import { useState, useEffect } from 'react'
import { Settings, Save, RotateCcw, Download, Upload, Info, Zap, Battery, MapPin, DollarSign, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'
import LocationSelector from './LocationSelector'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import BoxGuide from './BoxGuide'
import { getCurrencySymbol } from '@/lib/useCurrency'
import {
  exportPreferences,
  importPreferences,
  validateConfig,
  getLocationRecommendations,
  calculateOptimalTilt,
  calculateOptimalAzimuth,
  PRESET_CONFIGS,
  SystemConfig
} from '@/lib/userPreferences'

export default function SystemConfiguration() {
  const { config, updateConfig, resetConfig, isLoading } = useSystemConfig()
  const [isOpen, setIsOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<'system' | 'location' | 'financial' | 'battery'>('system')
  const [errors, setErrors] = useState<string[]>([])
  const [showRecommendations, setShowRecommendations] = useState(false)

  // Local pending state for manual saves
  const [pendingConfig, setPendingConfig] = useState<SystemConfig>(config)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)

  // Keep pendingConfig in sync with global config when it loads/changes externally
  useEffect(() => {
    setPendingConfig(config)
    setHasUnsavedChanges(false)
  }, [config])

  // Validate on pending config changes
  useEffect(() => {
    const validationErrors = validateConfig(pendingConfig)
    setErrors(validationErrors)

    // Check if pending differs from actual
    setHasUnsavedChanges(JSON.stringify(pendingConfig) !== JSON.stringify(config))
  }, [pendingConfig, config])

  const updatePendingConfig = (updates: Partial<SystemConfig>) => {
    setPendingConfig(prev => ({ ...prev, ...updates }))
  }

  const handleSave = () => {
    if (errors.length === 0) {
      updateConfig(pendingConfig)
      setHasUnsavedChanges(false)
    }
  }

  const handleDiscard = () => {
    setPendingConfig(config)
    setHasUnsavedChanges(false)
  }

  const handleExport = () => {
    exportPreferences()
  }

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      importPreferences(file)
        .then((imported) => {
          // Update all config at once
          updateConfig(imported)
          alert('✅ Configuration imported successfully!')
        })
        .catch((error) => {
          alert(`❌ Error importing: ${error.message}`)
        })
    }
  }

  const handleLocationChange = (location: { latitude: number; longitude: number; city: string; timezone?: string }) => {
    // Update pending location
    updatePendingConfig({
      latitude: location.latitude,
      longitude: location.longitude,
      city: location.city,
      ...(location.timezone ? { timezone: location.timezone } : {})
    })
  }

  const applyPreset = (presetKey: keyof typeof PRESET_CONFIGS) => {
    const preset = PRESET_CONFIGS[presetKey]
    updatePendingConfig({
      systemSize: preset.systemSize,
      panelEfficiency: preset.panelEfficiency,
      performanceRatio: preset.performanceRatio
    })
  }

  const applyOptimalOrientation = () => {
    const optimalTilt = calculateOptimalTilt(pendingConfig.latitude)
    const optimalAzimuth = calculateOptimalAzimuth(pendingConfig.latitude)

    updatePendingConfig({
      panelTilt: optimalTilt,
      panelAzimuth: optimalAzimuth
    })

    alert(`✅ Applied optimal orientation:\nTilt: ${optimalTilt}°\nAzimuth: ${optimalAzimuth}°`)
  }

  const recommendations = getLocationRecommendations(pendingConfig.latitude)

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm">
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center space-x-2 text-gray-700">
            <Settings className="h-5 w-5 animate-spin" />
            <span className="font-medium">System Configuration</span>
            <span className="text-xs text-gray-500">Loading...</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden h-full">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-500 to-indigo-500 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3 flex-1">
            <Settings className="h-6 w-6 text-white" />
            <div className="flex-1">
              <div className="flex items-center space-x-2 flex-wrap">
                <span className="font-semibold text-white text-lg">⚙️ System Configuration</span>
                <span className="text-xs px-2 py-0.5 bg-white/20 text-white rounded-full">
                  {config.systemSize} kWp
                </span>
                <span className="text-xs text-teal-100">• {config.city}</span>
                <span className="text-xs text-teal-100">• {config.timezone}</span>
                <span className="text-xs px-2 py-0.5 bg-green-500/50 text-white rounded-full animate-pulse">
                  Live Updates
                </span>
              </div>
              <p className="text-xs text-teal-100 mt-1">
                Changes apply instantly to all components
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <BoxGuide title="System Configuration">
              <p>This section allows you to customize the core parameters of your solar setup. It controls all simulations and financial estimates across the dashboard.</p>
              <ul className="space-y-2 mt-2">
                <li><strong>Configuration Display:</strong> The top header shows active settings and their live sync status.</li>
                <li><strong>Expand/Collapse:</strong> Open the panel to edit your settings.</li>
                <li><strong>Save/Discard Changes:</strong> Edits are held locally. Click "Save Configuration" to apply them globally.</li>
                <li><strong>Tabs (Solar System, Location, Financial, Battery):</strong> Navigate through each category to refine your system.
                  <ul className="list-disc pl-5 mt-1 space-y-1 text-gray-600">
                    <li><em>Solar System:</em> Set system size (kWp), efficiency, tilt, and azimuth. Use presets or the "Suggest Optimal Orientation" feature for quick setup.</li>
                    <li><em>Location:</em> Defines coordinates for weather matching and solar positions.</li>
                    <li><em>Financial:</em> Configure utility tariffs and currency for accurate ROI and savings estimations.</li>
                    <li><em>Battery:</em> If you have energy storage, input its capacity and efficiency here.</li>
                  </ul>
                </li>
                <li><strong>Import/Export:</strong> Save your final configuration to a JSON file, or load a previously saved one.</li>
              </ul>
            </BoxGuide>

            <button
              onClick={() => resetConfig()}
              className="p-1.5 text-white hover:bg-white/20 rounded-md transition-colors"
              title="Reset to defaults"
            >
              <RotateCcw className="h-4 w-4" />
            </button>

            <button
              onClick={() => setIsOpen(!isOpen)}
              className="flex items-center space-x-1 px-3 py-1.5 bg-white/20 hover:bg-white/30 text-white rounded-md text-sm transition-colors"
              title={isOpen ? "Collapse" : "Expand"}
            >
              <span>{isOpen ? 'Collapse' : 'Expand'}</span>
              {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      {isOpen && (
        <div className="p-4">
          {/* Header Action Buttons inside Config Panel */}
          <div className="flex items-center justify-between mb-4 pb-4 border-b">
            <h3 className="text-lg font-medium text-gray-800">Configuration Options</h3>
            <div className="flex items-center gap-2">
              {hasUnsavedChanges && (
                <button
                  onClick={handleDiscard}
                  className="px-4 py-2 text-sm text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors font-medium"
                >
                  Discard Changes
                </button>
              )}
              <button
                onClick={handleSave}
                disabled={!hasUnsavedChanges || errors.length > 0}
                className={`flex items-center space-x-2 px-6 py-2 rounded-md text-sm font-medium transition-colors ${hasUnsavedChanges && errors.length === 0
                  ? 'bg-gradient-to-r from-teal-500 to-indigo-500 text-white shadow-md hover:shadow-lg transform hover:-translate-y-0.5'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  }`}
              >
                <Save className="h-4 w-4" />
                <span>Save Configuration</span>
              </button>
            </div>
          </div>

          {/* Errors */}
          {errors.length > 0 && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm font-medium text-red-800 mb-1">Validation Errors:</p>
              <ul className="text-sm text-red-700 list-disc list-inside">
                {errors.map((error, i) => <li key={i}>{error}</li>)}
              </ul>
            </div>
          )}

          {/* Tabs */}
          <div className="flex space-x-1 mb-4 border-b overflow-x-auto">
            {[
              { key: 'system', label: 'Solar System', icon: Zap },
              { key: 'location', label: 'Location', icon: MapPin },
              { key: 'financial', label: 'Financial', icon: DollarSign },
              { key: 'battery', label: 'Battery', icon: Battery }
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key as any)}
                className={`flex items-center space-x-1 px-4 py-2 text-sm font-medium transition-colors whitespace-nowrap ${activeTab === key
                  ? 'text-green-600 border-b-2 border-green-600'
                  : 'text-gray-600 hover:text-gray-900'
                  }`}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="space-y-4">
            {/* Solar System Tab */}
            {activeTab === 'system' && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      System Size (kWp)
                    </label>
                    <input
                      type="number"
                      value={pendingConfig.systemSize}
                      onChange={(e) => updatePendingConfig({ systemSize: parseFloat(e.target.value) || 0 })}
                      className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                      step="0.1"
                      min="0"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Panel Efficiency (%)
                    </label>
                    <input
                      type="number"
                      value={(pendingConfig.panelEfficiency * 100).toFixed(1)}
                      onChange={(e) => updatePendingConfig({ panelEfficiency: (parseFloat(e.target.value) || 0) / 100 })}
                      className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                      step="0.1"
                      min="0"
                      max="100"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Panel Tilt (degrees)
                    </label>
                    <input
                      type="number"
                      value={pendingConfig.panelTilt}
                      onChange={(e) => updatePendingConfig({ panelTilt: parseFloat(e.target.value) || 0 })}
                      className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                      step="1"
                      min="0"
                      max="90"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Panel Azimuth (degrees)
                    </label>
                    <input
                      type="number"
                      value={pendingConfig.panelAzimuth}
                      onChange={(e) => updatePendingConfig({ panelAzimuth: parseFloat(e.target.value) || 0 })}
                      className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                      step="1"
                      min="0"
                      max="359"
                    />
                    <p className="text-xs text-gray-500 mt-1">0=North, 90=East, 180=South, 270=West</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Performance Ratio
                    </label>
                    <input
                      type="number"
                      value={pendingConfig.performanceRatio}
                      onChange={(e) => updatePendingConfig({ performanceRatio: parseFloat(e.target.value) || 0 })}
                      className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                      step="0.01"
                      min="0"
                      max="1"
                    />
                  </div>
                </div>

                {/* Presets */}
                <div className="border-t pt-4">
                  <p className="text-sm font-medium text-gray-700 mb-2">Quick Presets:</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(PRESET_CONFIGS).map(([key, preset]) => (
                      <button
                        key={key}
                        onClick={() => applyPreset(key as any)}
                        className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-green-100 rounded-md transition-colors"
                      >
                        {preset.description}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Optimal Orientation */}
                <div className="border-t pt-4">
                  <button
                    onClick={applyOptimalOrientation}
                    className="flex items-center space-x-2 px-4 py-2 bg-blue-50 text-blue-700 rounded-md hover:bg-blue-100"
                  >
                    <Info className="h-4 w-4" />
                    <span className="text-sm">Suggest Optimal Orientation for {pendingConfig.city}</span>
                  </button>
                </div>
              </>
            )}

            {/* Location Tab */}
            {activeTab === 'location' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Location
                  </label>
                  <LocationSelector
                    latitude={pendingConfig.latitude}
                    longitude={pendingConfig.longitude}
                    city={pendingConfig.city}
                    onLocationChange={handleLocationChange}
                  />
                  <p className="text-xs text-gray-500 mt-2">
                    Finding a new location will update coordinates below, click Save to apply.
                  </p>
                </div>

                {/* Manual Entry */}
                <div className="border-t pt-4">
                  <p className="text-sm font-medium text-gray-700 mb-3">Manual Coordinates</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-600 mb-1">City Name</label>
                      <input
                        type="text"
                        value={pendingConfig.city}
                        onChange={(e) => updatePendingConfig({ city: e.target.value })}
                        className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500 text-sm"
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-600 mb-1">Timezone</label>
                      <input
                        type="text"
                        value={pendingConfig.timezone}
                        onChange={(e) => updatePendingConfig({ timezone: e.target.value })}
                        className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500 text-sm"
                        placeholder="e.g., Asia/Kolkata"
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-600 mb-1">Latitude</label>
                      <input
                        type="number"
                        value={pendingConfig.latitude}
                        onChange={(e) => updatePendingConfig({ latitude: parseFloat(e.target.value) || 0 })}
                        className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500 text-sm"
                        step="0.0001"
                        min="-90"
                        max="90"
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-600 mb-1">Longitude</label>
                      <input
                        type="number"
                        value={pendingConfig.longitude}
                        onChange={(e) => updatePendingConfig({ longitude: parseFloat(e.target.value) || 0 })}
                        className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500 text-sm"
                        step="0.0001"
                        min="-180"
                        max="180"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Financial Tab */}
            {activeTab === 'financial' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Electricity Tariff ({getCurrencySymbol(pendingConfig.currency)}/kWh)
                  </label>
                  <input
                    type="number"
                    value={pendingConfig.electricityTariff}
                    onChange={(e) => updatePendingConfig({ electricityTariff: parseFloat(e.target.value) || 0 })}
                    className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                    step="0.01"
                    min="0"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Feed-in Tariff ({getCurrencySymbol(pendingConfig.currency)}/kWh)
                  </label>
                  <input
                    type="number"
                    value={pendingConfig.feedInTariff}
                    onChange={(e) => updatePendingConfig({ feedInTariff: parseFloat(e.target.value) || 0 })}
                    className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                    step="0.01"
                    min="0"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Grid CO₂ Factor (kg/kWh)
                  </label>
                  <input
                    type="number"
                    value={pendingConfig.gridCO2Factor}
                    onChange={(e) => updatePendingConfig({ gridCO2Factor: parseFloat(e.target.value) || 0 })}
                    className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                    step="0.01"
                    min="0"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Currency
                  </label>
                  <select
                    value={pendingConfig.currency}
                    onChange={(e) => updatePendingConfig({ currency: e.target.value })}
                    className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                  >
                    <option value="USD">USD ($)</option>
                    <option value="EUR">EUR (€)</option>
                    <option value="GBP">GBP (£)</option>
                    <option value="INR">INR (₹)</option>
                    <option value="AUD">AUD (A$)</option>
                    <option value="CAD">CAD (C$)</option>
                    <option value="JPY">JPY (¥)</option>
                    <option value="CNY">CNY (¥)</option>
                  </select>
                </div>
              </div>
            )}

            {/* Battery Tab */}
            {activeTab === 'battery' && (
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="hasBattery"
                    checked={pendingConfig.hasBattery}
                    onChange={(e) => updatePendingConfig({ hasBattery: e.target.checked })}
                    className="rounded text-green-600 focus:ring-green-500"
                  />
                  <label htmlFor="hasBattery" className="text-sm font-medium text-gray-700">
                    I have a battery storage system
                  </label>
                </div>

                {pendingConfig.hasBattery && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pl-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Battery Capacity (kWh)
                      </label>
                      <input
                        type="number"
                        value={pendingConfig.batteryCapacity}
                        onChange={(e) => updatePendingConfig({ batteryCapacity: parseFloat(e.target.value) || 0 })}
                        className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                        step="0.1"
                        min="0"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Battery Efficiency (%)
                      </label>
                      <input
                        type="number"
                        value={(pendingConfig.batteryEfficiency * 100).toFixed(0)}
                        onChange={(e) => updatePendingConfig({ batteryEfficiency: (parseFloat(e.target.value) || 0) / 100 })}
                        className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-green-500"
                        step="1"
                        min="0"
                        max="100"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Import/Export */}
          <div className="mt-6 pt-4 border-t flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <button
                onClick={handleExport}
                className="flex items-center space-x-1 px-3 py-1.5 text-sm text-gray-700 border rounded-md hover:bg-gray-50"
              >
                <Download className="h-4 w-4" />
                <span>Export</span>
              </button>

              <label className="flex items-center space-x-1 px-3 py-1.5 text-sm text-gray-700 border rounded-md hover:bg-gray-50 cursor-pointer">
                <Upload className="h-4 w-4" />
                <span>Import</span>
                <input
                  type="file"
                  accept=".json"
                  onChange={handleImport}
                  className="hidden"
                />
              </label>
            </div>

            <button
              onClick={() => setShowRecommendations(!showRecommendations)}
              className="text-sm text-blue-600 hover:text-blue-700"
            >
              {showRecommendations ? 'Hide' : 'Show'} Recommendations
            </button>
          </div>

          {showRecommendations && (
            <div className="mt-4 p-3 bg-blue-50 rounded-md text-sm text-blue-800">
              <p><strong>Recommendations for {config.city}:</strong></p>
              <p>• Optimal Tilt: {recommendations.optimalTilt}°</p>
              <p>• Optimal Azimuth: {recommendations.optimalAzimuth}°</p>
              <p className="mt-1 text-xs">{recommendations.seasonalAdjustment}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
