"use client";

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Zap,
    Battery,
    Car,
    Droplet,
    Flame,
    AlertTriangle,
    Sun,
    ArrowDownRight,
    ArrowUpRight
} from 'lucide-react';
import { useSystemConfig } from '@/lib/SystemConfigContext';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface UnifiedMetrics {
    electricity: {
        solar_gen_kw: number;
        grid_import_kw: number;
        battery_soc_percent: number;
        house_load_kw: number;
    };
    transport: {
        ev_charge_percent: number;
        ev_range_km: number;
        charging_status: string;
    };
    other_resources: {
        gas_usage_m3: number;
        water_usage_liters: number;
        water_leak_alert: boolean;
    };
}

const UnifiedEnergyView = () => {
    const { config } = useSystemConfig();
    const [data, setData] = useState<UnifiedMetrics | null>(null);
    const [loading, setLoading] = useState(true);
    const [realTimeData, setRealTimeData] = useState<any>(null);

    useEffect(() => {
        fetchAllData();
        const interval = setInterval(fetchAllData, 30000); // Update every 30s
        return () => clearInterval(interval);
    }, [config.latitude, config.longitude, config.systemSize, config.batteryCapacity, config.hasBattery, config.performanceRatio, config.panelTilt, config.panelAzimuth]);

    const fetchAllData = async () => {
        try {
            // Fetch real-time solar data
            const realtimeRes = await axios.get(`${API_BASE_URL}/realtime/current`, {
                params: {
                    lat: config.latitude,
                    lon: config.longitude,
                    system_size: config.systemSize,
                    performance_ratio: config.performanceRatio,
                    panel_tilt: config.panelTilt,
                    panel_azimuth: config.panelAzimuth
                }
            }).catch(() => null);

            if (realtimeRes?.data?.data) {
                setRealTimeData(realtimeRes.data.data);
            }

            // Fetch unified metrics (EV, Gas, Water)
            const unifiedRes = await axios.get(`${API_BASE_URL}/usage/unified`, {
                params: {
                    lat: config.latitude,
                    lon: config.longitude,
                    system_size: config.systemSize,
                    has_battery: config.hasBattery,
                    battery_capacity: config.batteryCapacity,
                    performance_ratio: config.performanceRatio,
                    panel_tilt: config.panelTilt,
                    panel_azimuth: config.panelAzimuth
                }
            }).catch(() => null);

            if (unifiedRes?.data?.status === 'success') {
                // Merge real-time solar data with unified metrics
                const mergedData = { ...unifiedRes.data.metrics };

                if (realTimeData || realtimeRes?.data?.data) {
                    const rtData = realtimeRes?.data?.data || realTimeData;
                    mergedData.electricity = {
                        ...mergedData.electricity,
                        solar_gen_kw: rtData.energy_output_kWh || mergedData.electricity.solar_gen_kw,
                        // Calculate grid import: house load - solar (negative = exporting)
                        grid_import_kw: Math.max(0, mergedData.electricity.house_load_kw - (rtData.energy_output_kWh || 0)),
                        battery_soc_percent: config.hasBattery ? mergedData.electricity.battery_soc_percent : 0
                    };
                }

                setData(mergedData);
            }
        } catch (error) {
            console.error("Error fetching unified data:", error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="animate-pulse grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {[1, 2, 3].map(i => (
                    <div key={i} className="bg-gray-100 rounded-2xl p-6 h-48" />
                ))}
            </div>
        );
    }

    if (!data) {
        return (
            <div className="text-center py-8 text-gray-500">
                <Sun className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>Unable to load unified energy data</p>
                <p className="text-xs mt-1">Check that the backend is running</p>
            </div>
        );
    }

    const { electricity, transport, other_resources } = data;

    // Calculate net energy flow
    const netFlow = electricity.solar_gen_kw - electricity.house_load_kw;
    const isExporting = netFlow > 0;

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Electricity Section - Enhanced with real data */}
            <div className="bg-gradient-to-br from-yellow-50 to-orange-50 p-6 rounded-2xl border border-yellow-200 shadow-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold flex items-center gap-2 text-gray-800">
                        <Zap className="text-yellow-500" /> Electricity
                    </h3>
                    <span className="text-xs px-2 py-1 bg-yellow-400/20 text-yellow-700 rounded-full font-medium">
                        Live • {config.city}
                    </span>
                </div>

                <div className="space-y-4">
                    <div className="flex justify-between items-end">
                        <div>
                            <p className="text-gray-500 text-sm flex items-center gap-1">
                                <Sun className="w-3 h-3" /> Solar Gen
                            </p>
                            <p className="text-2xl font-bold text-gray-800">
                                {electricity.solar_gen_kw.toFixed(2)} <span className="text-sm font-normal text-gray-500">kW</span>
                            </p>
                        </div>
                        <div className="text-right">
                            <p className="text-gray-500 text-sm">Grid</p>
                            <p className={`text-xl font-bold flex items-center gap-1 ${isExporting ? 'text-green-600' : 'text-orange-600'}`}>
                                {isExporting ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                                {Math.abs(netFlow).toFixed(2)} <span className="text-sm font-normal text-gray-500">kW</span>
                            </p>
                            <p className="text-xs text-gray-400">{isExporting ? 'Exporting' : 'Importing'}</p>
                        </div>
                    </div>

                    {/* Progress bar based on system capacity */}
                    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                        <div
                            className="bg-gradient-to-r from-yellow-400 to-orange-400 h-full transition-all duration-1000"
                            style={{ width: `${Math.min(100, (electricity.solar_gen_kw / config.systemSize) * 100)}%` }}
                        />
                    </div>
                    <p className="text-xs text-gray-500 text-right">
                        {((electricity.solar_gen_kw / config.systemSize) * 100).toFixed(0)}% of {config.systemSize} kWp capacity
                    </p>

                    {/* Battery status - only show if user has a battery */}
                    {config.hasBattery && config.batteryCapacity > 0 && (
                        <div className="flex items-center gap-2 text-sm text-gray-600 bg-white/50 p-2 rounded-lg">
                            <Battery className={`w-4 h-4 ${electricity.battery_soc_percent > 20 ? 'text-green-500' : 'text-red-500'}`} />
                            <span>Battery: {electricity.battery_soc_percent}%</span>
                            <span className="text-xs text-gray-400">({config.batteryCapacity} kWh)</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Transport Section */}
            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-6 rounded-2xl border border-blue-200 shadow-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold flex items-center gap-2 text-gray-800">
                        <Car className="text-blue-500" /> Transport
                    </h3>
                    <span className={`text-xs px-2 py-1 rounded-full font-medium ${transport.charging_status === 'Charging'
                        ? 'bg-green-100 text-green-700'
                        : transport.charging_status === 'Standby'
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-gray-100 text-gray-600'
                        }`}>
                        {transport.charging_status}
                    </span>
                </div>

                <div className="space-y-4">
                    <div>
                        <p className="text-gray-500 text-sm">EV Battery</p>
                        <div className="flex items-baseline gap-2">
                            <p className="text-2xl font-bold text-gray-800">{transport.ev_charge_percent}%</p>
                            <p className="text-blue-600 text-sm">~{transport.ev_range_km} km range</p>
                        </div>
                    </div>

                    {/* EV Battery progress bar */}
                    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                        <div
                            className={`h-full transition-all duration-1000 ${transport.ev_charge_percent > 50 ? 'bg-gradient-to-r from-blue-400 to-cyan-400' :
                                transport.ev_charge_percent > 20 ? 'bg-yellow-400' : 'bg-red-400'
                                }`}
                            style={{ width: `${transport.ev_charge_percent}%` }}
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="p-2 bg-white/50 rounded-lg border border-white/50">
                            <p className="text-gray-500 text-[10px] uppercase">Efficiency</p>
                            <p className="font-medium text-gray-800">18 kWh/100km</p>
                        </div>
                        <div className="p-2 bg-white/50 rounded-lg border border-white/50">
                            <p className="text-gray-500 text-[10px] uppercase">Solar Charged</p>
                            <p className="font-medium text-green-600">
                                {isExporting ? '+' : ''}{(netFlow * 0.5).toFixed(1)} kW
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Other Resources */}
            <div className="bg-gradient-to-br from-purple-50 to-pink-50 p-6 rounded-2xl border border-purple-200 shadow-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold flex items-center gap-2 text-gray-800">
                        <Droplet className="text-purple-500" /> Resources
                    </h3>
                    {other_resources.water_leak_alert && (
                        <AlertTriangle className="text-red-500 animate-pulse" />
                    )}
                </div>

                <div className="space-y-4">
                    <div className="flex justify-between items-center p-3 bg-white/50 rounded-lg">
                        <div className="flex items-center gap-2">
                            <Flame className="w-5 h-5 text-orange-500" />
                            <span className="text-gray-700">Gas</span>
                        </div>
                        <p className="font-bold text-gray-800">
                            {other_resources.gas_usage_m3.toFixed(1)} <span className="text-xs font-normal text-gray-500">m³</span>
                        </p>
                    </div>

                    <div className="flex justify-between items-center p-3 bg-white/50 rounded-lg">
                        <div className="flex items-center gap-2">
                            <Droplet className="w-5 h-5 text-blue-500" />
                            <span className="text-gray-700">Water</span>
                        </div>
                        <p className="font-bold text-gray-800">
                            {other_resources.water_usage_liters.toFixed(0)} <span className="text-xs font-normal text-gray-500">L</span>
                        </p>
                    </div>

                    <div className="mt-4 pt-4 border-t border-purple-100">
                        <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-1">System Status</p>
                        <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${other_resources.water_leak_alert ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
                            <span className={`text-xs font-medium ${other_resources.water_leak_alert ? 'text-red-600' : 'text-green-600'}`}>
                                {other_resources.water_leak_alert ? 'Water Leak Detected!' : 'All Systems Nominal'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UnifiedEnergyView;
