'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Lightbulb, Zap, TrendingDown,
  Settings2, Play, Battery, SearchCheck, Gauge
} from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import { useCurrency } from '@/lib/useCurrency'

import AlertBanner from './SmartRecommendations/AlertBanner'
import ApplianceSchedule from './SmartRecommendations/ApplianceSchedule'
import BatterySchedule from './SmartRecommendations/BatterySchedule'
import GridStrategy from './SmartRecommendations/GridStrategy'
import AutomationTriggers from './SmartRecommendations/AutomationTriggers'
import ApplianceEditorModal from './SmartRecommendations/ApplianceEditorModal'

interface SmartRecommendationsProps { }

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function SmartRecommendations(props: SmartRecommendationsProps = {}) {
  const { config } = useSystemConfig()
  const { convert, formatCurrency } = useCurrency()
  const currency = config?.currency || 'USD'
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState<any>(null)
  const [activeSection, setActiveSection] = useState<'appliances' | 'battery' | 'grid' | 'savings' | 'automation'>('appliances')

  // Appliance management state
  const [showApplianceEditor, setShowApplianceEditor] = useState(false)
  const [applianceConfig, setApplianceConfig] = useState<any[]>([])
  const [isSaving, setIsSaving] = useState(false)

  const fetchAppliances = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/appliances`)
      if (Array.isArray(response.data)) {
        setApplianceConfig(response.data)
      } else {
        setApplianceConfig([]) // Fallback to empty array if the format is unexpected
      }
    } catch (error) {
      console.error('Error fetching appliances:', error)
    }
  }

  const saveAppliances = async () => {
    setIsSaving(true)
    try {
      console.log("Saving flat appliance list:", applianceConfig)
      await axios.post(`${API_BASE_URL}/appliances`, applianceConfig)
      await fetchRecommendations() // Refresh to get updated savings
      setShowApplianceEditor(false)
    } catch (error) {
      console.error('Error saving appliances:', error)
      alert("Failed to save appliances.")
    } finally {
      setIsSaving(false)
    }
  }

  const updateAppliance = (index: number, field: string, value: any) => {
    const newConfig = [...applianceConfig]
    newConfig[index] = { ...newConfig[index], [field]: value }
    setApplianceConfig(newConfig)
  }

  const addAppliance = () => {
    setApplianceConfig([...applianceConfig, { name: '', consumption_kwh: 1.0, duration_hours: 1 }])
  }

  const removeAppliance = (index: number) => {
    const newConfig = [...applianceConfig]
    newConfig.splice(index, 1)
    setApplianceConfig(newConfig)
  }

  const getCategory = (consumption: number) => {
    if (consumption >= 3) return { label: 'High', color: 'bg-red-500', text: 'text-red-700' };
    if (consumption >= 1) return { label: 'Medium', color: 'bg-orange-500', text: 'text-orange-700' };
    return { label: 'Low', color: 'bg-blue-500', text: 'text-blue-700' };
  }

  useEffect(() => {
    fetchAppliances()
  }, [])

  const fetchRecommendations = async () => {
    setLoading(true)
    try {
      const payload = {
        latitude: config.latitude,
        longitude: config.longitude,
        battery_capacity: config.hasBattery ? config.batteryCapacity : 0,
        electricity_tariff: config.electricityTariff,
        feed_in_tariff: config.feedInTariff,
        system_size: config.systemSize,
        performance_ratio: config.performanceRatio,
        efficiency: config.panelEfficiency,
        panel_tilt: config.panelTilt,
        panel_azimuth: config.panelAzimuth,
        grid_co2_factor: config.gridCO2Factor,
        max_grid_import: config.maxGridImport
      }

      console.log('Fetching recommendations with config:', payload)

      const response = await axios.post(`${API_BASE_URL}/optimize`, payload)

      if (response.data.status === 'success') {
        setRecommendations(response.data.recommendations)
        console.log('Recommendations received:', response.data.recommendations)
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRecommendations()
    // Auto-refresh every 5 minutes to stay in sync with weather data
    const interval = setInterval(fetchRecommendations, 300000)
    return () => clearInterval(interval)
  }, [
    config.latitude,
    config.longitude,
    config.batteryCapacity,
    config.electricityTariff,
    config.feedInTariff,
    config.systemSize,
    config.panelEfficiency,
    config.panelTilt,
    config.panelAzimuth,
    config.gridCO2Factor,
    config.maxGridImport
  ])

  // Custom renderer for markdown-like text
  const renderFormattedText = (text: string) => {
    if (!text) return null;

    return text.split('\n').map((line, i) => {
      if (line.startsWith('## ')) {
        return <h3 key={i} className="text-sm font-bold text-gray-900 mt-4 mb-2">{line.replace('## ', '')}</h3>
      }
      if (line.startsWith('- ')) {
        // highlight numbers and currencies
        let formattedLine = line.replace('- ', '');
        formattedLine = formattedLine.replace(/(\$[\d\.]+)/g, '<span class="text-green-600 font-bold">$1</span>');
        formattedLine = formattedLine.replace(/(\d+%)/g, '<span class="text-blue-600 font-bold">$1</span>');

        return (
          <li key={i} className="flex items-start text-sm text-gray-700 mb-2 ml-2">
            <span className="text-green-500 mr-2 mt-0.5">•</span>
            <span dangerouslySetInnerHTML={{ __html: formattedLine }} />
          </li>
        )
      }
      if (line.trim() === '') return null;
      return <p key={i} className="text-sm text-gray-700 mb-2">{line}</p>
    })
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden flex flex-col h-full">
      <div className="p-5 border-b flex items-center justify-between pb-4">
        <div className="flex flex-col">
          <div className="flex items-center space-x-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h3 className="font-semibold text-gray-900">Smart Optimization</h3>
          </div>
          <p className="text-xs text-gray-500 mt-1">AI-driven scheduling based on your local weather</p>
        </div>
        <button
          onClick={fetchRecommendations}
          disabled={loading}
          className="text-xs flex items-center bg-gray-50 border hover:bg-gray-100 hover:text-gray-900 px-3 py-1.5 rounded-md transition-colors text-gray-600"
        >
          {loading ? 'Analyzing Forecast...' : 'Refresh Insights'}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto bg-gray-50/50">
        {!recommendations ? (
          <div className="flex items-center justify-center h-48 text-gray-500 text-sm flex-col">
            <SearchCheck className="w-8 h-8 text-blue-200 mb-2" />
            Generating personalized optimizations...
          </div>
        ) : (
          <div>
            <AlertBanner alerts={recommendations.alerts} />

            <div className="p-4 grid grid-cols-5 gap-1 border-b bg-white">
              {[
                { id: 'appliances', icon: Zap, label: 'Appliances' },
                { id: 'battery', icon: Battery, label: 'Battery' },
                { id: 'grid', icon: Gauge, label: 'Grid' },
                { id: 'savings', icon: TrendingDown, label: 'Savings' },
                { id: 'automation', icon: Play, label: 'Automation' }
              ].map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveSection(tab.id as any)}
                    className={`flex flex-col items-center p-2 rounded-lg transition-all ${activeSection === tab.id
                      ? 'bg-blue-50 text-blue-700'
                      : 'text-gray-500 hover:bg-gray-50'
                      }`}
                  >
                    <Icon className="h-4 w-4 mb-1" />
                    <span className="text-[10px] uppercase font-bold tracking-wider">{tab.label}</span>
                  </button>
                )
              })}
            </div>

            <div className="p-5">
              {activeSection === 'appliances' && (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-gray-900">Appliance Scheduling</h3>
                    <button
                      onClick={() => setShowApplianceEditor(true)}
                      className="text-xs flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-200 hover:border-gray-300 hover:bg-gray-50 text-gray-700 rounded-lg shadow-sm transition-all shadow-black/5"
                    >
                      <Settings2 className="w-3.5 h-3.5" /> Configure Household
                    </button>
                  </div>
                  <ApplianceSchedule
                    schedule={recommendations.appliance_schedule}
                    formatCurrency={formatCurrency}
                    convert={convert}
                    currency={currency}
                  />
                </div>
              )}

              {activeSection === 'battery' && (
                <BatterySchedule schedule={recommendations.battery_schedule} />
              )}

              {activeSection === 'grid' && (
                <GridStrategy strategy={recommendations.grid_strategy} />
              )}

              {activeSection === 'savings' && recommendations.summary && (
                <div className="prose prose-sm max-w-none text-gray-700">
                  {renderFormattedText(recommendations.summary)}
                </div>
              )}

              {activeSection === 'automation' && (
                <AutomationTriggers triggers={recommendations.automation_triggers} />
              )}
            </div>
          </div>
        )}
      </div>

      <ApplianceEditorModal
        showApplianceEditor={showApplianceEditor}
        setShowApplianceEditor={setShowApplianceEditor}
        applianceConfig={applianceConfig}
        setApplianceConfig={setApplianceConfig}
        getCategory={getCategory}
        updateAppliance={updateAppliance}
        removeAppliance={removeAppliance}
        addAppliance={addAppliance}
        saveAppliances={saveAppliances}
        isSaving={isSaving}
      />
    </div>
  )
}
