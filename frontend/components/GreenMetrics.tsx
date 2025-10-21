'use client'

import { useState } from 'react'
import { Leaf, Zap, TrendingDown, TreePine, ChevronDown, ChevronUp } from 'lucide-react'

interface GreenMetricsProps {
  energyUsed?: number
  carbonEmissions?: number
  netEnergySaved?: number
}

export default function GreenMetrics({ 
  energyUsed = 0.3, 
  carbonEmissions = 45,
  netEnergySaved = 5000 
}: GreenMetricsProps) {
  const [isOpen, setIsOpen] = useState(true)
  
  const treesEquivalent = (carbonEmissions / 21000 * 12).toFixed(1) // Trees absorb ~21kg CO2/year
  const kmDriven = (carbonEmissions / 120).toFixed(1) // Average car emits ~120g CO2/km

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-500 to-emerald-500 p-4">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              🌱 Green AI Metrics
            </h3>
            <p className="text-xs text-green-100">AI-powered environmental impact tracking</p>
          </div>
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="flex items-center space-x-1 px-3 py-1.5 bg-white/20 hover:bg-white/30 text-white rounded-md text-sm transition-colors"
          >
            <span>{isOpen ? 'Collapse' : 'Expand'}</span>
            {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Content */}
      {isOpen && (
        <div className="p-6 flex-1">

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Energy Used */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-5 h-5 text-yellow-500" />
            <span className="text-xs text-gray-600">Energy Used</span>
          </div>
          <div className="text-xl font-bold text-gray-800">
            {energyUsed.toFixed(2)} Wh
          </div>
          <div className="text-xs text-gray-500 mt-1">per forecast</div>
        </div>

        {/* Carbon Emissions */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-5 h-5 text-blue-500" />
            <span className="text-xs text-gray-600">CO₂ Emissions</span>
          </div>
          <div className="text-xl font-bold text-gray-800">
            {carbonEmissions.toFixed(0)} g
          </div>
          <div className="text-xs text-gray-500 mt-1">per training</div>
        </div>

        {/* Net Energy Saved */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Leaf className="w-5 h-5 text-green-500" />
            <span className="text-xs text-gray-600">Net Saved</span>
          </div>
          <div className="text-xl font-bold text-green-600">
            {netEnergySaved}x
          </div>
          <div className="text-xs text-gray-500 mt-1">compute cost</div>
        </div>

        {/* Trees Equivalent */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <TreePine className="w-5 h-5 text-emerald-500" />
            <span className="text-xs text-gray-600">Equivalent</span>
          </div>
          <div className="text-xl font-bold text-gray-800">
            {treesEquivalent} 🌳
          </div>
          <div className="text-xs text-gray-500 mt-1">tree-months</div>
        </div>
      </div>

      {/* Impact Summary */}
      <div className="mt-4 p-4 bg-white rounded-lg">
        <h4 className="font-semibold text-sm text-gray-700 mb-2">Environmental Impact</h4>
        <div className="space-y-2 text-sm text-gray-600">
          <div className="flex items-center justify-between">
            <span>🚗 Equivalent km driven:</span>
            <span className="font-semibold">{kmDriven} km</span>
          </div>
          <div className="flex items-center justify-between">
            <span>♻️ Carbon intensity:</span>
            <span className="font-semibold">400 gCO₂/kWh</span>
          </div>
          <div className="flex items-center justify-between">
            <span>💚 Grid efficiency gain:</span>
            <span className="font-semibold text-green-600">+15%</span>
          </div>
        </div>
      </div>

      {/* Green AI Badge */}
      <div className="mt-4 flex items-center justify-center gap-2 p-3 bg-green-100 rounded-lg border-2 border-green-300">
        <Leaf className="w-5 h-5 text-green-700" />
        <span className="text-sm font-semibold text-green-800">
          Net Positive Environmental Impact ✓
        </span>
      </div>
        </div>
      )}
    </div>
  )
}
