'use client'

import { useState } from 'react'
import {
    HelpCircle, X, Sun, Zap, DollarSign, Leaf, Battery,
    MapPin, Thermometer, Cloud, Wind, TrendingUp, Settings,
    ChevronDown, ChevronUp
} from 'lucide-react'

interface GuideSection {
    title: string
    icon: React.ReactNode
    description: string
    items: { label: string; effect: string; tip?: string }[]
}

const guideSections: GuideSection[] = [
    {
        title: "Location Settings",
        icon: <MapPin className="w-5 h-5 text-blue-500" />,
        description: "Your location determines weather data, sunrise/sunset times, and optimal panel orientation.",
        items: [
            { label: "City/Location", effect: "Fetches local weather, calculates sun position, determines peak sun hours", tip: "Choose your exact location for accurate predictions" },
            { label: "Latitude", effect: "Affects solar panel efficiency and optimal tilt angle", tip: "Higher latitudes need steeper panel tilt" },
            { label: "Longitude", effect: "Determines local solar noon and sun path", tip: "Used for precise sunrise/sunset calculations" },
            { label: "Timezone", effect: "Syncs forecasts with your local time", tip: "Auto-detected but can be adjusted" }
        ]
    },
    {
        title: "System Configuration",
        icon: <Sun className="w-5 h-5 text-yellow-500" />,
        description: "Your solar system's specifications directly impact energy calculations.",
        items: [
            { label: "System Size (kWp)", effect: "Scales all energy output calculations proportionally", tip: "Check your inverter rating for accurate value" },
            { label: "Panel Efficiency (%)", effect: "Modern panels: 15-22%. Higher = more output per m²", tip: "Check your panel datasheet" },
            { label: "Panel Tilt (°)", effect: "Optimal angle equals your latitude. Affects seasonal output", tip: "Flat roof: 10-15°, Pitched roof: match roof angle" },
            { label: "Panel Azimuth (°)", effect: "South-facing (180°) is optimal in Northern Hemisphere", tip: "180° = South, 0° = North, 90° = East, 270° = West" },
            { label: "Performance Ratio", effect: "Accounts for real-world losses (wiring, dirt, shading). Usually 0.75-0.85", tip: "New systems: 0.80-0.85, Older: 0.70-0.78" }
        ]
    },
    {
        title: "Financial Settings",
        icon: <DollarSign className="w-5 h-5 text-green-500" />,
        description: "These values calculate your actual savings and ROI.",
        items: [
            { label: "Electricity Tariff", effect: "Your grid purchase price. Higher = more savings from solar", tip: "Check your electricity bill for exact rate" },
            { label: "Feed-in Tariff", effect: "What you earn for exporting excess power to grid", tip: "Contact your utility for current rates" },
            { label: "Currency", effect: "All financial calculations shown in your chosen currency", tip: "Uses real-time exchange rates" }
        ]
    },
    {
        title: "Battery Settings",
        icon: <Battery className="w-5 h-5 text-purple-500" />,
        description: "If you have battery storage, configure it here for accurate optimization.",
        items: [
            { label: "Has Battery", effect: "Enables battery charge/discharge optimization", tip: "Enable to see battery recommendations" },
            { label: "Battery Capacity (kWh)", effect: "Determines how much energy can be stored", tip: "Check your battery specifications" },
            { label: "Battery Efficiency", effect: "Round-trip efficiency (typically 90-95%)", tip: "Lithium: 95%, Lead-acid: 80-85%" }
        ]
    },
    {
        title: "Environmental Impact",
        icon: <Leaf className="w-5 h-5 text-emerald-500" />,
        description: "Track your carbon footprint reduction.",
        items: [
            { label: "Grid CO₂ Factor", effect: "kg CO₂ per kWh from your grid. Varies by country/region", tip: "Clean grid: 0.2-0.4, Coal-heavy: 0.7-1.0" },
            { label: "CO₂ Avoided", effect: "Solar output × Grid CO₂ Factor = Your carbon savings", tip: "1 kWh solar saves 0.3-0.8 kg CO₂ depending on grid" }
        ]
    },
    {
        title: "Weather & Real-Time Data",
        icon: <Cloud className="w-5 h-5 text-cyan-500" />,
        description: "Live weather data affects current and predicted output.",
        items: [
            { label: "Temperature", effect: "Panels lose ~0.4% efficiency per °C above 25°C", tip: "Hot days: lower output despite more sun" },
            { label: "Cloud Cover (%)", effect: "Directly reduces solar irradiance. 100% clouds ≈ 25% of clear sky power", tip: "Scattered clouds still allow significant output" },
            { label: "Solar Irradiance (W/m²)", effect: "The actual sunlight hitting your panels right now", tip: "Clear noon: 800-1000 W/m², Cloudy: 100-300 W/m²" },
            { label: "Sunrise/Sunset", effect: "Defines your productive solar window", tip: "Summer: 14+ hours, Winter: 8-10 hours" }
        ]
    }
]

export default function UserGuide() {
    const [isOpen, setIsOpen] = useState(false)
    const [expandedSection, setExpandedSection] = useState<number | null>(0)

    return (
        <>
            {/* Floating Guide Button */}
            <button
                onClick={() => setIsOpen(true)}
                className="fixed bottom-6 right-6 bg-gradient-to-r from-blue-500 to-purple-500 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110 z-40"
                title="How to use SunShift"
            >
                <HelpCircle className="w-6 h-6" />
            </button>

            {/* Guide Modal */}
            {isOpen && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-2xl max-w-3xl w-full max-h-[85vh] overflow-hidden">
                        {/* Header */}
                        <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-6 text-white">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="bg-white/20 p-2 rounded-lg">
                                        <HelpCircle className="w-6 h-6" />
                                    </div>
                                    <div>
                                        <h2 className="text-2xl font-bold">SunShift User Guide</h2>
                                        <p className="text-blue-100 text-sm">Understanding what each value does</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setIsOpen(false)}
                                    className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                                >
                                    <X className="w-6 h-6" />
                                </button>
                            </div>
                        </div>

                        {/* Content */}
                        <div className="overflow-y-auto max-h-[calc(85vh-120px)] p-6">
                            <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-100">
                                <h3 className="font-semibold text-green-800 mb-2">🔄 Real-Time Data Flow</h3>
                                <p className="text-sm text-green-700">
                                    When you change any setting in System Configuration, all components automatically
                                    refresh with new calculations. No manual refresh needed!
                                </p>
                            </div>

                            <div className="space-y-4">
                                {guideSections.map((section, idx) => (
                                    <div key={idx} className="border border-gray-200 rounded-xl overflow-hidden">
                                        <button
                                            onClick={() => setExpandedSection(expandedSection === idx ? null : idx)}
                                            className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
                                        >
                                            <div className="flex items-center gap-3">
                                                {section.icon}
                                                <span className="font-semibold text-gray-800">{section.title}</span>
                                            </div>
                                            {expandedSection === idx ? (
                                                <ChevronUp className="w-5 h-5 text-gray-400" />
                                            ) : (
                                                <ChevronDown className="w-5 h-5 text-gray-400" />
                                            )}
                                        </button>

                                        {expandedSection === idx && (
                                            <div className="p-4 bg-white">
                                                <p className="text-sm text-gray-600 mb-4">{section.description}</p>
                                                <div className="space-y-3">
                                                    {section.items.map((item, i) => (
                                                        <div key={i} className="p-3 bg-gray-50 rounded-lg">
                                                            <div className="flex items-start justify-between gap-4">
                                                                <div>
                                                                    <span className="font-medium text-gray-800">{item.label}</span>
                                                                    <p className="text-sm text-gray-600 mt-1">{item.effect}</p>
                                                                </div>
                                                            </div>
                                                            {item.tip && (
                                                                <div className="mt-2 text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded inline-block">
                                                                    💡 {item.tip}
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>

                            {/* Quick Reference */}
                            <div className="mt-6 p-4 bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl border border-amber-100">
                                <h3 className="font-semibold text-amber-800 mb-3">⚡ Quick Reference</h3>
                                <div className="grid grid-cols-2 gap-3 text-sm">
                                    <div className="p-2 bg-white rounded-lg">
                                        <span className="text-gray-500">1 kWp system →</span>
                                        <span className="font-semibold text-gray-800 ml-1">3-5 kWh/day</span>
                                    </div>
                                    <div className="p-2 bg-white rounded-lg">
                                        <span className="text-gray-500">Peak Sun Hours →</span>
                                        <span className="font-semibold text-gray-800 ml-1">4-6 hours/day avg</span>
                                    </div>
                                    <div className="p-2 bg-white rounded-lg">
                                        <span className="text-gray-500">Clear sky noon →</span>
                                        <span className="font-semibold text-gray-800 ml-1">~1000 W/m²</span>
                                    </div>
                                    <div className="p-2 bg-white rounded-lg">
                                        <span className="text-gray-500">1 kWh solar →</span>
                                        <span className="font-semibold text-gray-800 ml-1">0.5-0.8 kg CO₂ saved</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    )
}
