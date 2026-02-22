import { Shield } from 'lucide-react'

export default function RiskScoreCard({ riskData }: { riskData: any }) {
    return (
        <div className={`p-6 rounded-2xl border transition-all duration-500 ${riskData?.level === 'Extreme' ? 'bg-red-50 border-red-200' :
            riskData?.level === 'High' ? 'bg-orange-50 border-orange-200' :
                'bg-green-50 border-green-200'
            }`}>
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold flex items-center gap-2">
                    <Shield className={riskData?.level === 'Extreme' ? 'text-red-500' : 'text-green-500'} />
                    Solar Installation Risk
                </h3>
                <span className={`text-xs px-2 py-1 rounded-full font-bold uppercase ${riskData?.level === 'Extreme' ? 'bg-red-500 text-white' :
                    riskData?.level === 'High' ? 'bg-orange-500 text-white' :
                        'bg-green-500 text-white'
                    }`}>
                    {riskData?.level || 'Safe'}
                </span>
            </div>

            <div className="flex items-end gap-4 mb-4">
                <span className="text-5xl font-black text-gray-800">{riskData?.score || 0}</span>
                <div className="flex-1">
                    <div className="w-full bg-gray-200 h-3 rounded-full overflow-hidden">
                        <div
                            className={`h-full transition-all duration-1000 ${riskData?.score > 75 ? 'bg-red-500' : riskData?.score > 40 ? 'bg-orange-500' : 'bg-green-500'
                                }`}
                            style={{ width: `${riskData?.score || 0}%` }}
                        />
                    </div>
                </div>
            </div>

            <ul className="space-y-2">
                {riskData?.recommendations?.map((rec: string, i: number) => (
                    <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                        <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-gray-400 flex-shrink-0" />
                        {rec}
                    </li>
                ))}
            </ul>
        </div>
    )
}
