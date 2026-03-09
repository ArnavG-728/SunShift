import { CloudOff, Info } from 'lucide-react'

export default function CloudImpactCard({ currentWeather }: { currentWeather: any }) {
    return (
        <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm overflow-hidden relative">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-50 rounded-full -mr-16 -mt-16 z-0" />
            <div className="relative z-10">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        <CloudOff className="text-blue-500" /> Cloud Generation Impact
                    </h3>
                    <Info className="text-gray-300 w-4 h-4 cursor-help" />
                </div>

                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <p className="text-gray-500 text-xs uppercase font-semibold">Today's Loss</p>
                        <p className="text-3xl font-bold text-blue-600">
                            {currentWeather?.cloud_loss?.loss_kwh?.toFixed(2) || '0.00'} <span className="text-sm font-normal text-gray-400">kWh</span>
                        </p>
                    </div>
                    <div>
                        <p className="text-gray-500 text-xs uppercase font-semibold">Efficiency Dip</p>
                        <p className="text-3xl font-bold text-orange-600">
                            -{currentWeather?.cloud_loss?.loss_percent?.toFixed(1) || '0.0'}%
                        </p>
                    </div>
                </div>

                <div className="mt-6 pt-4 border-t border-gray-100">
                    <p className="text-sm text-gray-600 italic">
                        "You could have generated <b>{currentWeather?.cloud_loss?.potential_kwh?.toFixed(1) || '0.0'} kWh</b> today with clear skies."
                    </p>
                </div>
            </div>
        </div>
    )
}
