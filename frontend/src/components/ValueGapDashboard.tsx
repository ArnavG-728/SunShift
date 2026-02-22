'use client'

import { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown, Zap, Battery, Clock, DollarSign, ChevronDown, ChevronUp } from 'lucide-react'
import axios from 'axios'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import { useCurrency } from '@/lib/useCurrency'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface ValueGapData {
    value_gap: {
        buy_rate: number
        sell_rate: number
        delta_per_kwh: number
        delta_percentage: number
    }
    naive_scenario: {
        self_consumed_kwh: number
        exported_kwh: number
        imported_kwh: number
        total_value: number
        self_consumption_pct: number
    }
    optimised_scenario: {
        self_consumed_kwh: number
        exported_kwh: number
        imported_kwh: number
        total_value: number
        self_consumption_pct: number
    }
    virtual_battery_savings: {
        daily_savings: number
        monthly_projection: number
        annual_projection: number
        equivalent_battery_kwh: number
    }
    self_consumption: {
        naive_rate: number
        optimised_rate: number
        improvement: number
    }
    optimal_shift_windows: Array<{
        start: string
        surplus_kwh: number
        savings_if_shifted: number
    }>
}

export default function ValueGapDashboard() {
    const { config } = useSystemConfig()
    const { convert, formatCurrency } = useCurrency()
    const [isOpen, setIsOpen] = useState(true)
    const [data, setData] = useState<ValueGapData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetchValueGap()
    }, [config.latitude, config.longitude, config.systemSize, config.electricityTariff, config.feedInTariff])

    const fetchValueGap = async () => {
        try {
            setLoading(true)
            setError(null)
            const res = await axios.get(`${API_BASE_URL}/value-gap`, {
                params: {
                    lat: config.latitude,
                    lon: config.longitude,
                    system_size: config.systemSize,
                    electricity_tariff: config.electricityTariff,
                    feed_in_tariff: config.feedInTariff,
                }
            })
            if (res.data?.data?.status === 'success') {
                setData(res.data.data)
            }
        } catch (err: any) {
            console.error('Value gap fetch error:', err)
            setError('Unable to load value gap analysis')
        } finally {
            setLoading(false)
        }
    }

    const displayValue = (usd: number) => {
        const converted = convert(usd, 'USD', config.currency)
        return formatCurrency(converted, config.currency)
    }

    return (
        <div className="bg-white rounded-lg shadow-lg overflow-hidden h-full flex flex-col">
            {/* Header */}
            <div className="bg-gradient-to-r from-amber-500 to-orange-500 p-4">
                <div className="flex items-center justify-between">
                    <div className="flex-1">
                        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            💰 Solar Value Gap
                        </h3>
                        <p className="text-xs text-amber-100">Buy vs Sell rate analysis & Virtual Battery</p>
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

            {isOpen && (
                <div className="p-6 flex-1">
                    {loading ? (
                        <div className="animate-pulse text-center py-8 text-gray-400">Analyzing value gap...</div>
                    ) : error ? (
                        <div className="text-center py-8 text-red-400 text-sm">{error}</div>
                    ) : data ? (
                        <>
                            {/* Value Gap Delta */}
                            <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-lg p-4 border border-red-200 mb-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs text-gray-500 uppercase tracking-wide">The Value Gap</p>
                                        <p className="text-2xl font-bold text-red-600">
                                            {data.value_gap.delta_percentage.toFixed(0)}% Markup
                                        </p>
                                        <p className="text-xs text-gray-500 mt-1">
                                            Buy @ {displayValue(data.value_gap.buy_rate)}/kWh → Sell @ {displayValue(data.value_gap.sell_rate)}/kWh
                                        </p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-xs text-gray-500">Delta per kWh</p>
                                        <p className="text-xl font-bold text-orange-600">
                                            {displayValue(data.value_gap.delta_per_kwh)}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Virtual Battery Savings */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-3 border border-green-100">
                                    <div className="flex items-center gap-1 mb-1">
                                        <Battery className="w-4 h-4 text-green-500" />
                                        <span className="text-xs text-gray-500">Daily Savings</span>
                                    </div>
                                    <p className="text-lg font-bold text-green-700">
                                        {displayValue(data.virtual_battery_savings.daily_savings)}
                                    </p>
                                </div>
                                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-3 border border-green-100">
                                    <div className="flex items-center gap-1 mb-1">
                                        <DollarSign className="w-4 h-4 text-green-500" />
                                        <span className="text-xs text-gray-500">Monthly</span>
                                    </div>
                                    <p className="text-lg font-bold text-green-700">
                                        {displayValue(data.virtual_battery_savings.monthly_projection)}
                                    </p>
                                </div>
                                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-3 border border-green-100">
                                    <div className="flex items-center gap-1 mb-1">
                                        <TrendingUp className="w-4 h-4 text-green-500" />
                                        <span className="text-xs text-gray-500">Annual</span>
                                    </div>
                                    <p className="text-lg font-bold text-green-700">
                                        {displayValue(data.virtual_battery_savings.annual_projection)}
                                    </p>
                                </div>
                                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-3 border border-blue-100">
                                    <div className="flex items-center gap-1 mb-1">
                                        <Zap className="w-4 h-4 text-blue-500" />
                                        <span className="text-xs text-gray-500">Battery Equiv</span>
                                    </div>
                                    <p className="text-lg font-bold text-blue-700">
                                        {data.virtual_battery_savings.equivalent_battery_kwh} kWh
                                    </p>
                                </div>
                            </div>

                            {/* Self-Consumption Improvement */}
                            <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 border border-indigo-100 mb-4">
                                <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Self-Consumption Boost</p>
                                <div className="flex items-center gap-4">
                                    <div className="flex-1">
                                        <div className="flex justify-between text-xs text-gray-500 mb-1">
                                            <span>Without SunShift</span>
                                            <span>{data.self_consumption.naive_rate}%</span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-2">
                                            <div
                                                className="bg-gray-400 h-2 rounded-full transition-all"
                                                style={{ width: `${Math.min(100, data.self_consumption.naive_rate)}%` }}
                                            />
                                        </div>
                                    </div>
                                    <TrendingUp className="w-5 h-5 text-green-500 flex-shrink-0" />
                                    <div className="flex-1">
                                        <div className="flex justify-between text-xs text-gray-500 mb-1">
                                            <span>With SunShift</span>
                                            <span>{data.self_consumption.optimised_rate}%</span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-2">
                                            <div
                                                className="bg-green-500 h-2 rounded-full transition-all"
                                                style={{ width: `${Math.min(100, data.self_consumption.optimised_rate)}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>
                                <p className="text-xs text-green-600 font-semibold mt-2 text-center">
                                    +{data.self_consumption.improvement}% improvement by shifting loads
                                </p>
                            </div>

                            {/* Best Shift Windows */}
                            {data.optimal_shift_windows.length > 0 && (
                                <div>
                                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">⏰ Best Times to Shift Loads</p>
                                    <div className="space-y-1">
                                        {data.optimal_shift_windows.slice(0, 3).map((w, i) => (
                                            <div key={i} className="flex items-center justify-between bg-gray-50 rounded px-3 py-2 text-sm">
                                                <div className="flex items-center gap-2">
                                                    <Clock className="w-3.5 h-3.5 text-orange-500" />
                                                    <span className="font-medium">{w.start}</span>
                                                </div>
                                                <span className="text-gray-500">{w.surplus_kwh} kWh surplus</span>
                                                <span className="text-green-600 font-semibold">
                                                    +{displayValue(w.savings_if_shifted)}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </>
                    ) : null}
                </div>
            )}
        </div>
    )
}
