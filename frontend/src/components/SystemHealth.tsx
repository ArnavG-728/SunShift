'use client'

import { useState, useEffect } from 'react'
import { Shield, AlertTriangle, CheckCircle, Activity, Wrench, ChevronDown, ChevronUp } from 'lucide-react'
import axios from 'axios'
import { useSystemConfig } from '@/lib/SystemConfigContext'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface HealthData {
    health_score: number
    performance_ratio: number
    degradation_rate_pct: number
    expected_age_degradation_pct: number
    anomalies: Array<{
        type: string
        severity: string
        description: string
    }>
    alerts: Array<{
        type: string
        priority: string
        title: string
        message: string
    }>
    recommendations: string[]
    details: {
        daytime_hours_analyzed: number
        avg_baseline_kwh: number
        avg_actual_kwh: number
        min_pr: number
        max_pr: number
    }
}

export default function SystemHealth() {
    const { config } = useSystemConfig()
    const [isOpen, setIsOpen] = useState(true)
    const [data, setData] = useState<HealthData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetchHealth()
    }, [config.latitude, config.longitude, config.systemSize])

    const fetchHealth = async () => {
        try {
            setLoading(true)
            setError(null)
            const res = await axios.get(`${API_BASE_URL}/system-health`, {
                params: {
                    lat: config.latitude,
                    lon: config.longitude,
                    system_size: config.systemSize,
                    panel_age_years: 0,
                }
            })
            if (res.data?.data?.status === 'success') {
                setData(res.data.data)
            }
        } catch (err: any) {
            console.error('System health fetch error:', err)
            setError('Unable to load health analysis')
        } finally {
            setLoading(false)
        }
    }

    const getScoreColor = (score: number) => {
        if (score >= 90) return 'text-green-600'
        if (score >= 75) return 'text-yellow-600'
        return 'text-red-600'
    }

    const getScoreBg = (score: number) => {
        if (score >= 90) return 'from-green-500 to-emerald-500'
        if (score >= 75) return 'from-yellow-500 to-amber-500'
        return 'from-red-500 to-rose-500'
    }

    const getScoreLabel = (score: number) => {
        if (score >= 90) return 'Excellent'
        if (score >= 75) return 'Good'
        if (score >= 60) return 'Fair'
        return 'Needs Attention'
    }

    const getAlertIcon = (type: string) => {
        switch (type) {
            case 'critical': return <AlertTriangle className="w-4 h-4 text-red-500" />
            case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />
            default: return <CheckCircle className="w-4 h-4 text-green-500" />
        }
    }

    return (
        <div className="bg-white rounded-lg shadow-lg overflow-hidden h-full flex flex-col">
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-500 to-indigo-500 p-4">
                <div className="flex items-center justify-between">
                    <div className="flex-1">
                        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            🔍 System Health
                        </h3>
                        <p className="text-xs text-blue-100">Degradation Detective — Panel diagnostics</p>
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
                        <div className="animate-pulse text-center py-8 text-gray-400">Analyzing panel health...</div>
                    ) : error ? (
                        <div className="text-center py-8 text-red-400 text-sm">{error}</div>
                    ) : data ? (
                        <>
                            {/* Health Score Gauge */}
                            <div className="flex items-center gap-6 mb-5">
                                <div className="relative">
                                    <div className={`w-24 h-24 rounded-full bg-gradient-to-br ${getScoreBg(data.health_score)} flex items-center justify-center shadow-lg`}>
                                        <div className="w-20 h-20 rounded-full bg-white flex flex-col items-center justify-center">
                                            <span className={`text-2xl font-bold ${getScoreColor(data.health_score)}`}>
                                                {data.health_score}
                                            </span>
                                            <span className="text-[10px] text-gray-400 uppercase">Score</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="flex-1">
                                    <p className={`text-lg font-semibold ${getScoreColor(data.health_score)}`}>
                                        {getScoreLabel(data.health_score)}
                                    </p>
                                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-sm">
                                        <div className="flex justify-between text-gray-500">
                                            <span>PR:</span>
                                            <span className="font-medium text-gray-700">{(data.performance_ratio * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="flex justify-between text-gray-500">
                                            <span>Degradation:</span>
                                            <span className="font-medium text-gray-700">{data.degradation_rate_pct}%</span>
                                        </div>
                                        <div className="flex justify-between text-gray-500">
                                            <span>Baseline:</span>
                                            <span className="font-medium text-gray-700">{data.details.avg_baseline_kwh} kWh</span>
                                        </div>
                                        <div className="flex justify-between text-gray-500">
                                            <span>Actual:</span>
                                            <span className="font-medium text-gray-700">{data.details.avg_actual_kwh} kWh</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Alerts */}
                            {data.alerts.length > 0 && (
                                <div className="space-y-2 mb-4">
                                    {data.alerts.map((alert, i) => (
                                        <div
                                            key={i}
                                            className={`flex items-start gap-3 p-3 rounded-lg border text-sm ${alert.type === 'critical'
                                                    ? 'bg-red-50 border-red-200'
                                                    : alert.type === 'warning'
                                                        ? 'bg-yellow-50 border-yellow-200'
                                                        : 'bg-green-50 border-green-200'
                                                }`}
                                        >
                                            {getAlertIcon(alert.type)}
                                            <div>
                                                <p className="font-semibold text-gray-800">{alert.title}</p>
                                                <p className="text-gray-600 text-xs mt-0.5">{alert.message}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Recommendations */}
                            {data.recommendations.length > 0 && (
                                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-100">
                                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                                        <Wrench className="w-3.5 h-3.5" /> Maintenance Recommendations
                                    </p>
                                    <ul className="space-y-1">
                                        {data.recommendations.map((rec, i) => (
                                            <li key={i} className="text-sm text-gray-700">{rec}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </>
                    ) : null}
                </div>
            )}
        </div>
    )
}
