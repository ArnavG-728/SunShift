'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Lightbulb, Battery, Zap, TrendingUp, AlertTriangle, Clock, DollarSign, Leaf, Cpu, Play, Sparkles, Cross, CrossIcon, Minus, MinusCircleIcon, SidebarClose, PanelTopClose, ShieldClose, X } from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import { useCurrency } from '@/lib/useCurrency'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface SmartRecommendationsProps {
  // Props are now optional - will use system config by default
}

export default function SmartRecommendations(props: SmartRecommendationsProps = {}) {
  const { config } = useSystemConfig()
  const { convert, formatCurrency } = useCurrency()
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState<any>(null)
  const [activeSection, setActiveSection] = useState<'appliances' | 'battery' | 'grid' | 'savings' | 'automation'>('appliances')

  // Appliance management state
  const [showApplianceEditor, setShowApplianceEditor] = useState(false)
  const [applianceConfig, setApplianceConfig] = useState<any[]>([])
  const [isSaving, setIsSaving] = useState(false)

  const fetchAppliances = async () => {
    try {
      const resp = await axios.get(`${API_BASE_URL}/appliances`)
      setApplianceConfig(Array.isArray(resp.data) ? resp.data : [])
    } catch (err) {
      console.error('Error fetching appliances:', err)
    }
  }

  const saveAppliances = async () => {
    setIsSaving(true)
    try {
      await axios.post(`${API_BASE_URL}/appliances`, applianceConfig)
      setShowApplianceEditor(false)
      fetchRecommendations() // Refresh with new config
    } catch (err) {
      console.error('Error saving appliances:', err)
      alert('Failed to save appliance configuration')
    } finally {
      setIsSaving(false)
    }
  }

  const updateAppliance = (index: number, field: string, value: any) => {
    const newConfig = [...applianceConfig]
    newConfig[index][field] = value
    setApplianceConfig(newConfig)
  }

  const addAppliance = () => {
    setApplianceConfig([...applianceConfig, { name: 'New Device', consumption_kwh: 1.0, duration_hours: 1 }])
  }

  const removeAppliance = (index: number) => {
    const newConfig = [...applianceConfig]
    newConfig.splice(index, 1)
    setApplianceConfig(newConfig)
  }

  const getCategory = (consumption: number) => {
    if (consumption > 2.5) return { label: 'High', color: 'bg-red-400', text: 'text-red-600' }
    if (consumption >= 1.0) return { label: 'Medium', color: 'bg-orange-400', text: 'text-orange-600' }
    return { label: 'Flexible', color: 'bg-blue-400', text: 'text-blue-600' }
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

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-600"></div>
          <span className="ml-3 text-gray-600">Generating smart recommendations...</span>
        </div>
      </div>
    )
  }

  if (!recommendations || recommendations.status === 'no_data') {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="text-center text-gray-500">
          <Lightbulb className="h-12 w-12 mx-auto mb-3 text-gray-400" />
          <p>Run a forecast first to get smart energy recommendations</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-sm">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h3 className="text-lg font-semibold text-gray-900">Smart Energy Recommendations</h3>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowApplianceEditor(true)}
              className="text-xs flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
            >
              <Cpu className="w-3.5 h-3.5" />
              Manage Appliances
            </button>
            <button
              onClick={fetchRecommendations}
              className="text-sm text-green-600 hover:text-green-700 font-medium"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Appliance Editor Modal */}
      {showApplianceEditor && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
            <div className="p-6 border-b flex items-center justify-between bg-gradient-to-r from-green-50 to-blue-50">
              <div>
                <h3 className="text-xl font-bold text-gray-900">Configure Appliances</h3>
                <p className="text-sm text-gray-500">Add or edit appliances to personalize your energy optimization</p>
              </div>
              <button
                onClick={() => setShowApplianceEditor(false)}
                className="p-2 hover:bg-white/50 rounded-full transition-colors"
              >
                <X className="w-5 h-5 text-gray-400 rotate-180" />
              </button>
            </div>

            <div className="p-6 overflow-y-auto flex-1 space-y-6">
              <div className="space-y-3">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-bold uppercase tracking-wider text-gray-400 flex items-center gap-2">
                    <Cpu className="w-4 h-4" />
                    Appliance Inventory
                  </h4>
                  <p className="text-[10px] text-gray-400 font-medium">Automatic classification based on kWh</p>
                </div>

                <div className="grid gap-3">
                  {applianceConfig?.map((app: any, idx: number) => {
                    const cat = getCategory(app.consumption_kwh);
                    return (
                      <div key={idx} className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl border border-gray-100 group transition-all hover:bg-white hover:shadow-sm">
                        <div className="flex-1 grid grid-cols-12 gap-3 items-center">
                          <div className="col-span-5">
                            <input
                              className="w-full bg-transparent font-bold text-sm focus:ring-0 border-none p-0 text-gray-800 placeholder:text-gray-300"
                              value={app.name}
                              placeholder="Appliance Name"
                              onChange={(e) => updateAppliance(idx, 'name', e.target.value)}
                            />
                            <div className="flex items-center gap-1.5 mt-0.5">
                              <span className={`w-1.5 h-1.5 rounded-full ${cat.color}`} />
                              <span className={`text-[10px] font-bold uppercase tracking-tight ${cat.text}`}>{cat.label} Load</span>
                            </div>
                          </div>

                          <div className="col-span-3 flex items-center gap-2 bg-white/50 px-2 py-1 rounded-lg border border-gray-100">
                            <Zap className="w-3.5 h-3.5 text-yellow-500" />
                            <input
                              type="number"
                              step="0.1"
                              className="w-full bg-transparent text-sm font-medium focus:ring-0 border-none p-0"
                              value={app.consumption_kwh}
                              onChange={(e) => updateAppliance(idx, 'consumption_kwh', parseFloat(e.target.value))}
                            />
                            <span className="text-[10px] text-gray-400 font-bold uppercase">kWh</span>
                          </div>

                          <div className="col-span-3 flex items-center gap-2 bg-white/50 px-2 py-1 rounded-lg border border-gray-100">
                            <Clock className="w-3.5 h-3.5 text-blue-500" />
                            <input
                              type="number"
                              className="w-full bg-transparent text-sm font-medium focus:ring-0 border-none p-0"
                              value={app.duration_hours}
                              onChange={(e) => updateAppliance(idx, 'duration_hours', parseInt(e.target.value))}
                            />
                            <span className="text-[10px] text-gray-400 font-bold uppercase">hrs</span>
                          </div>
                        </div>

                        <button
                          onClick={() => removeAppliance(idx)}
                          className="p-2 hover:bg-red-50 text-red-300 hover:text-red-500 rounded-lg transition-all"
                          title="Remove appliance"
                        >
                          <Minus className="w-4 h-4" />
                        </button>
                      </div>
                    );
                  })}

                  <button
                    onClick={addAppliance}
                    className="flex items-center justify-center gap-2 p-4 border-2 border-dashed border-gray-100 rounded-xl text-sm font-bold text-gray-400 hover:border-green-200 hover:text-green-600 hover:bg-green-50/30 transition-all active:scale-[0.98]"
                  >
                    <Sparkles className="w-4 h-4" />
                    Add New Appliance
                  </button>
                </div>
              </div>
            </div>

            <div className="p-6 border-t bg-gray-50 flex items-center justify-between">
              <p className="text-xs text-gray-500">Changes will be saved to your profile and used for future tips.</p>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowApplianceEditor(false)}
                  className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900"
                >
                  Cancel
                </button>
                <button
                  onClick={saveAppliances}
                  disabled={isSaving}
                  className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-bold rounded-xl shadow-lg shadow-green-200 transition-all active:scale-95 disabled:opacity-50"
                >
                  {isSaving ? 'Saving...' : 'Save Configuration'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Alerts Section */}
      {recommendations.alerts && recommendations.alerts.length > 0 && (
        <div className="p-4 bg-yellow-50 border-b">
          <div className="space-y-2">
            {recommendations.alerts.map((alert: any, index: number) => (
              <div
                key={index}
                className={`flex items-start space-x-3 p-3 rounded-md ${alert.type === 'warning' ? 'bg-orange-50 border border-orange-200' : 'bg-blue-50 border border-blue-200'
                  }`}
              >
                <AlertTriangle className={`h-5 w-5 mt-0.5 ${alert.type === 'warning' ? 'text-orange-600' : 'text-blue-600'
                  }`} />
                <div className="flex-1">
                  <p className="font-medium text-sm text-gray-900">{alert.title}</p>
                  <p className="text-sm text-gray-600 mt-1">{alert.message}</p>
                  <p className="text-xs text-gray-500 mt-1">💡 {alert.action}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      {recommendations.summary && (
        <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 border-b">
          <pre className="text-sm text-gray-700 whitespace-pre-wrap font-sans">
            {recommendations.summary}
          </pre>
        </div>
      )}

      {/* Tabs */}
      <div className="flex border-b">
        {[
          { key: 'appliances', label: 'Appliances', icon: Zap },
          { key: 'battery', label: 'Battery', icon: Battery, show: config.hasBattery && config.batteryCapacity > 0 },
          { key: 'grid', label: 'Grid Strategy', icon: TrendingUp },
          { key: 'automation', label: 'Automation', icon: Cpu },
          { key: 'savings', label: 'Savings', icon: DollarSign }
        ].filter(tab => tab.show !== false).map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveSection(key as any)}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 text-sm font-medium transition-colors ${activeSection === key
              ? 'text-green-600 border-b-2 border-green-600 bg-green-50'
              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
          >
            <Icon className="h-4 w-4" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-4">
        {/* Appliances Schedule */}
        {activeSection === 'appliances' && recommendations.appliance_schedule && (
          <div className="space-y-4">
            {/* High Energy Appliances */}
            {recommendations.appliance_schedule.high_energy_appliances?.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                  <Zap className="h-4 w-4 mr-2 text-red-500" />
                  High Energy Appliances
                </h4>
                <div className="space-y-2">
                  {recommendations.appliance_schedule.high_energy_appliances.map((item: any, index: number) => (
                    <div key={index} className="p-3 bg-red-50 border border-red-100 rounded-md">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-sm text-gray-900">{item.appliance}</span>
                        <span className="text-xs font-medium text-green-600">
                          {item.expected_solar_coverage.toFixed(0)}% Solar
                        </span>
                      </div>
                      <div className="text-xs text-gray-600 space-y-1">
                        <p>⏰ Best time: <strong>{item.best_start_time}</strong></p>
                        <p>💰 Save: <strong>{formatCurrency(convert(item.cost_savings, 'USD', config.currency), config.currency)}</strong></p>
                        {item.grid_needed > 0 && (
                          <p>⚡ Grid needed: {item.grid_needed.toFixed(2)} kWh</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Medium Energy Appliances */}
            {recommendations.appliance_schedule.medium_energy_appliances?.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                  <Zap className="h-4 w-4 mr-2 text-orange-500" />
                  Medium Energy Appliances
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {recommendations.appliance_schedule.medium_energy_appliances.map((item: any, index: number) => (
                    <div key={index} className="p-3 bg-orange-50 border border-orange-100 rounded-md">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-sm text-gray-900">{item.appliance}</span>
                        <span className="text-xs font-medium text-green-600">
                          {item.expected_solar_coverage.toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-xs text-gray-600">⏰ {item.best_start_time}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Flexible Loads */}
            {recommendations.appliance_schedule.flexible_loads?.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                  <Clock className="h-4 w-4 mr-2 text-blue-500" />
                  Flexible Loads
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {recommendations.appliance_schedule.flexible_loads.map((item: any, index: number) => (
                    <div key={index} className="p-2 bg-blue-50 border border-blue-100 rounded-md text-center">
                      <p className="text-xs font-medium text-gray-900">{item.appliance}</p>
                      <p className="text-xs text-gray-600 mt-1">{item.best_start_time}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Battery Schedule */}
        {activeSection === 'battery' && recommendations.battery_schedule && (
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 rounded-md">
              <p className="text-sm font-medium text-gray-900 mb-1">Strategy</p>
              <p className="text-sm text-gray-700">{recommendations.battery_schedule.strategy}</p>
              <p className="text-xs text-gray-600 mt-2">
                Estimated cycles: {recommendations.battery_schedule.estimated_cycles}
              </p>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-gray-900 mb-3">24-Hour Schedule</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {recommendations.battery_schedule.schedule?.slice(0, 12).map((item: any, index: number) => (
                  <div
                    key={index}
                    className={`p-2 rounded-md text-center ${item.action === 'charge'
                      ? 'bg-green-100 border border-green-200'
                      : item.action === 'discharge'
                        ? 'bg-orange-100 border border-orange-200'
                        : 'bg-gray-100 border border-gray-200'
                      }`}
                  >
                    <p className="text-xs font-medium text-gray-900">{item.time}</p>
                    <p className="text-xs text-gray-600 mt-1 capitalize">{item.action}</p>
                    <p className="text-xs text-gray-500">{item.solar_kwh} kWh</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Grid Strategy */}
        {activeSection === 'grid' && recommendations.grid_strategy && (
          <div className="space-y-4">
            <div className={`p-4 rounded-md ${recommendations.grid_strategy.strategy === 'net_exporter'
              ? 'bg-green-50 border border-green-200'
              : 'bg-orange-50 border border-orange-200'
              }`}>
              <p className="text-sm font-medium text-gray-900 mb-2">
                {recommendations.grid_strategy.strategy === 'net_exporter' ? '📤 Net Exporter' : '📥 Net Importer'}
              </p>
              <p className="text-sm text-gray-700">{recommendations.grid_strategy.recommendation}</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-gray-50 rounded-md">
                <p className="text-xs text-gray-600">Total Production</p>
                <p className="text-lg font-semibold text-gray-900">
                  {recommendations.grid_strategy.total_production_kwh} kWh
                </p>
              </div>
              <div className="p-3 bg-gray-50 rounded-md">
                <p className="text-xs text-gray-600">Est. Consumption</p>
                <p className="text-lg font-semibold text-gray-900">
                  {recommendations.grid_strategy.estimated_consumption_kwh} kWh
                </p>
              </div>
            </div>

            <div className="p-3 bg-blue-50 rounded-md">
              <p className="text-xs text-gray-600">Net Balance</p>
              <p className={`text-xl font-bold ${recommendations.grid_strategy.net_balance_kwh >= 0 ? 'text-green-600' : 'text-orange-600'
                }`}>
                {recommendations.grid_strategy.net_balance_kwh >= 0 ? '+' : ''}
                {recommendations.grid_strategy.net_balance_kwh} kWh
              </p>
            </div>
          </div>
        )}

        {/* Savings & Impact */}
        {activeSection === 'savings' && (
          <div className="space-y-4">
            {/* Financial Savings */}
            {recommendations.savings && (
              <div>
                <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                  <DollarSign className="h-4 w-4 mr-2 text-green-600" />
                  Financial Impact
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-green-50 rounded-md">
                    <p className="text-xs text-gray-600">Total Savings</p>
                    <p className="text-2xl font-bold text-green-600">
                      {formatCurrency(convert(recommendations.savings.total_savings || 0, 'USD', config.currency), config.currency)}
                    </p>
                  </div>
                  <div className="p-3 bg-blue-50 rounded-md">
                    <p className="text-xs text-gray-600">Monthly Projection</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {formatCurrency(convert(recommendations.savings.monthly_projection || 0, 'USD', config.currency), config.currency)}
                    </p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-md">
                    <p className="text-xs text-gray-600">Grid Cost Avoided</p>
                    <p className="text-lg font-semibold text-gray-900">
                      {formatCurrency(convert(recommendations.savings.grid_cost_avoided || 0, 'USD', config.currency), config.currency)}
                    </p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-md">
                    <p className="text-xs text-gray-600">Export Revenue</p>
                    <p className="text-lg font-semibold text-gray-900">
                      {formatCurrency(convert(recommendations.savings.export_revenue || 0, 'USD', config.currency), config.currency)}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Carbon Impact */}
            {recommendations.carbon_impact && (
              <div>
                <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                  <Leaf className="h-4 w-4 mr-2 text-green-600" />
                  Environmental Impact
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-green-50 rounded-md">
                    <p className="text-xs text-gray-600">CO₂ Avoided</p>
                    <p className="text-xl font-bold text-green-600">
                      {recommendations.carbon_impact.co2_avoided_kg?.toFixed(1)} kg
                    </p>
                  </div>
                  <div className="p-3 bg-blue-50 rounded-md">
                    <p className="text-xs text-gray-600">Trees Equivalent</p>
                    <p className="text-xl font-bold text-blue-600">
                      🌳 {recommendations.carbon_impact.trees_equivalent?.toFixed(1)}
                    </p>
                  </div>
                  <div className="p-3 bg-purple-50 rounded-md col-span-2">
                    <p className="text-xs text-gray-600">Car Miles Avoided</p>
                    <p className="text-xl font-bold text-purple-600">
                      🚗 {recommendations.carbon_impact.car_miles_avoided?.toFixed(0)} miles
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Automation Triggers */}
        {activeSection === 'automation' && (
          <div className="space-y-4">
            <div className="p-4 bg-purple-50 rounded-md border border-purple-100">
              <p className="text-sm font-semibold text-purple-900 mb-1">Prescriptive Intelligence</p>
              <p className="text-sm text-purple-700">These triggers can be linked to Home Assistant or Google Home to automate your savings.</p>
            </div>

            {recommendations.automation_triggers && recommendations.automation_triggers.length > 0 ? (
              <div className="space-y-3">
                {recommendations.automation_triggers.map((trigger: any, index: number) => (
                  <div key={index} className="p-4 bg-white border border-gray-200 rounded-xl hover:border-purple-300 transition-colors shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className="p-1.5 bg-purple-100 rounded-lg text-purple-600">
                          <Cpu className="w-4 h-4" />
                        </div>
                        <span className="font-bold text-sm text-gray-800">{trigger.action}</span>
                      </div>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${trigger.priority === 1 ? 'bg-red-100 text-red-600' : 'bg-gray-100 text-gray-500'
                        }`}>
                        Priority {trigger.priority}
                      </span>
                    </div>

                    <p className="text-xs text-gray-600 mb-3">
                      <b>Condition:</b> {trigger.condition}
                    </p>

                    <div className="flex items-center justify-between">
                      <div className="text-[10px] py-1 px-2 bg-gray-50 rounded text-gray-500 font-mono">
                        Target: {trigger.target}
                      </div>
                      <button className="flex items-center gap-1 text-xs font-bold text-purple-600 hover:text-purple-700 bg-purple-50 px-3 py-1.5 rounded-lg transition-all active:scale-95">
                        <Play className="w-3 h-3" /> Execute Mock
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-10 text-gray-500 bg-gray-50 rounded-xl border border-dashed border-gray-200">
                <p>No active automation triggers.</p>
                <p className="text-xs">Conditions for automation (like high solar excess) are not met currently.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
