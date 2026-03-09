export default function GridStrategy({ strategy }: { strategy: any }) {
    if (!strategy) return null;

    return (
        <div className="space-y-4">
            <div className={`p-4 rounded-md ${strategy.strategy === 'net_exporter'
                ? 'bg-green-50 border border-green-200'
                : 'bg-orange-50 border border-orange-200'
                }`}>
                <p className="text-sm font-medium text-gray-900 mb-2">
                    {strategy.strategy === 'net_exporter' ? '📤 Net Exporter' : '📥 Net Importer'}
                </p>
                <p className="text-sm text-gray-700">{strategy.recommendation}</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-gray-50 rounded-md">
                    <p className="text-xs text-gray-600">Total Production</p>
                    <p className="text-lg font-semibold text-gray-900">
                        {strategy.total_production_kwh} kWh
                    </p>
                </div>
                <div className="p-3 bg-gray-50 rounded-md">
                    <p className="text-xs text-gray-600">Est. Consumption</p>
                    <p className="text-lg font-semibold text-gray-900">
                        {strategy.estimated_consumption_kwh} kWh
                    </p>
                </div>
            </div>

            <div className="p-3 bg-blue-50 rounded-md">
                <p className="text-xs text-gray-600">Net Balance</p>
                <p className={`text-xl font-bold ${strategy.net_balance_kwh >= 0 ? 'text-green-600' : 'text-orange-600'
                    }`}>
                    {strategy.net_balance_kwh >= 0 ? '+' : ''}
                    {strategy.net_balance_kwh} kWh
                </p>
            </div>
        </div>
    )
}
