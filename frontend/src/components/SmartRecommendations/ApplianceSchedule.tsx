import { Zap, Clock } from 'lucide-react'

export default function ApplianceSchedule({
    schedule,
    formatCurrency,
    convert,
    currency
}: {
    schedule: any;
    formatCurrency: any;
    convert: any;
    currency: string;
}) {
    if (!schedule) return null;

    return (
        <div className="space-y-4">
            {/* High Energy Appliances */}
            {schedule.high_energy_appliances?.length > 0 && (
                <div>
                    <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                        <Zap className="h-4 w-4 mr-2 text-red-500" />
                        High Energy Appliances
                    </h4>
                    <div className="space-y-2">
                        {schedule.high_energy_appliances.map((item: any, index: number) => (
                            <div key={index} className="p-3 bg-red-50 border border-red-100 rounded-md">
                                <div className="flex items-center justify-between mb-1">
                                    <span className="font-medium text-sm text-gray-900">{item.appliance}</span>
                                    <span className="text-xs font-medium text-green-600">
                                        {item.expected_solar_coverage.toFixed(0)}% Solar
                                    </span>
                                </div>
                                <div className="text-xs text-gray-600 space-y-1">
                                    <p>⏰ Best time: <strong>{item.best_start_time}</strong></p>
                                    <p>💰 Save: <strong>{formatCurrency(convert(item.cost_savings, 'USD', currency), currency)}</strong></p>
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
            {schedule.medium_energy_appliances?.length > 0 && (
                <div>
                    <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                        <Zap className="h-4 w-4 mr-2 text-orange-500" />
                        Medium Energy Appliances
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {schedule.medium_energy_appliances.map((item: any, index: number) => (
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
            {schedule.flexible_loads?.length > 0 && (
                <div>
                    <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                        <Clock className="h-4 w-4 mr-2 text-blue-500" />
                        Flexible Loads
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                        {schedule.flexible_loads.map((item: any, index: number) => (
                            <div key={index} className="p-2 bg-blue-50 border border-blue-100 rounded-md text-center">
                                <p className="text-xs font-medium text-gray-900">{item.appliance}</p>
                                <p className="text-xs text-gray-600 mt-1">{item.best_start_time}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
