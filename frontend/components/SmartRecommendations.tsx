'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Lightbulb, Battery, Zap, TrendingUp, AlertTriangle, Clock, DollarSign, Leaf } from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface SmartRecommendationsProps {
  // Props are now optional - will use system config by default
}

export default function SmartRecommendations(props: SmartRecommendationsProps = {}) {
  const { config } = useSystemConfig()
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState<any>(null)
  const [activeSection, setActiveSection] = useState<'appliances' | 'battery' | 'grid' | 'savings'>('appliances')

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
          <button
            onClick={fetchRecommendations}
            className="text-sm text-green-600 hover:text-green-700"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Alerts Section */}
      {recommendations.alerts && recommendations.alerts.length > 0 && (
        <div className="p-4 bg-yellow-50 border-b">
          <div className="space-y-2">
            {recommendations.alerts.map((alert: any, index: number) => (
              <div
                key={index}
                className={`flex items-start space-x-3 p-3 rounded-md ${
                  alert.type === 'warning' ? 'bg-orange-50 border border-orange-200' : 'bg-blue-50 border border-blue-200'
                }`}
              >
                <AlertTriangle className={`h-5 w-5 mt-0.5 ${
                  alert.type === 'warning' ? 'text-orange-600' : 'text-blue-600'
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
          { key: 'savings', label: 'Savings', icon: DollarSign }
        ].filter(tab => tab.show !== false).map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveSection(key as any)}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeSection === key
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
                        <p>💰 Save: <strong>${item.cost_savings.toFixed(2)}</strong></p>
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
                    className={`p-2 rounded-md text-center ${
                      item.action === 'charge'
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
            <div className={`p-4 rounded-md ${
              recommendations.grid_strategy.strategy === 'net_exporter'
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
              <p className={`text-xl font-bold ${
                recommendations.grid_strategy.net_balance_kwh >= 0 ? 'text-green-600' : 'text-orange-600'
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
                      ${recommendations.savings.total_savings?.toFixed(2)}
                    </p>
                  </div>
                  <div className="p-3 bg-blue-50 rounded-md">
                    <p className="text-xs text-gray-600">Monthly Projection</p>
                    <p className="text-2xl font-bold text-blue-600">
                      ${recommendations.savings.monthly_projection?.toFixed(2)}
                    </p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-md">
                    <p className="text-xs text-gray-600">Grid Cost Avoided</p>
                    <p className="text-lg font-semibold text-gray-900">
                      ${recommendations.savings.grid_cost_avoided?.toFixed(2)}
                    </p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-md">
                    <p className="text-xs text-gray-600">Export Revenue</p>
                    <p className="text-lg font-semibold text-gray-900">
                      ${recommendations.savings.export_revenue?.toFixed(2)}
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
      </div>
    </div>
  )
}
