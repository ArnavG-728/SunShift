import { User } from 'lucide-react'

export default function SimpleModeView({ currentWeather }: { currentWeather: any }) {
    return (
        <div className="bg-white rounded-3xl shadow-xl p-8 border border-green-100">
            <div className="text-center mb-10">
                <div className="inline-flex items-center justify-center p-4 bg-green-100 rounded-full mb-4">
                    <User className="w-8 h-8 text-green-600" />
                </div>
                <h2 className="text-3xl font-bold text-gray-800">Hi, how's your energy today?</h2>
                <p className="text-gray-500">Everything you need to know in a second.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="p-8 rounded-2xl bg-gradient-to-br from-green-500 to-green-600 text-white text-center shadow-lg shadow-green-200">
                    <p className="text-green-100 text-sm uppercase tracking-widest font-bold mb-2">Solar Health</p>
                    <div className="text-6xl font-black mb-2">Excellent</div>
                    <p className="text-green-50 opacity-90">Producing {currentWeather?.energy_output_kWh?.toFixed(1) || '0.0'} kW right now</p>
                </div>

                <div className="p-8 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-600 text-white text-center shadow-lg shadow-blue-200">
                    <p className="text-blue-100 text-sm uppercase tracking-widest font-bold mb-2">Wallet Status</p>
                    <div className="text-6xl font-black mb-2">${((currentWeather?.energy_output_kWh || 0) * 0.12).toFixed(2)}</div>
                    <p className="text-blue-50 opacity-90">Estimated savings today</p>
                </div>
            </div>

            <div className="mt-12 text-center">
                <div className="inline-block px-10 py-5 bg-gray-50 rounded-2xl border border-gray-100">
                    <p className="text-sm text-gray-500 mb-1">Grandma Test Recommendation:</p>
                    <p className="text-xl font-bold text-gray-800 italic">
                        "It's a great time to do the laundry! 🧺 Sunlight is at its peak."
                    </p>
                </div>
            </div>
        </div>
    )
}
