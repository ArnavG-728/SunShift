export default function BatterySchedule({ schedule }: { schedule: any }) {
    if (!schedule) return null;

    return (
        <div className="space-y-4">
            <div className="p-4 bg-blue-50 rounded-md">
                <p className="text-sm font-medium text-gray-900 mb-1">Strategy</p>
                <p className="text-sm text-gray-700">{schedule.strategy}</p>
                <p className="text-xs text-gray-600 mt-2">
                    Estimated cycles: {schedule.estimated_cycles}
                </p>
            </div>

            <div>
                <h4 className="text-sm font-semibold text-gray-900 mb-3">24-Hour Schedule</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {schedule.schedule?.slice(0, 12).map((item: any, index: number) => (
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
    )
}
