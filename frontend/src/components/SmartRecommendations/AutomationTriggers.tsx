import { Cpu, Play } from 'lucide-react'

export default function AutomationTriggers({ triggers }: { triggers: any[] }) {
    return (
        <div className="space-y-4">
            <div className="p-4 bg-purple-50 rounded-md border border-purple-100">
                <p className="text-sm font-semibold text-purple-900 mb-1">Prescriptive Intelligence</p>
                <p className="text-sm text-purple-700">These triggers can be linked to Home Assistant or Google Home to automate your savings.</p>
            </div>

            {triggers && triggers.length > 0 ? (
                <div className="space-y-3">
                    {triggers.map((trigger: any, index: number) => (
                        <div key={index} className="p-4 bg-white border border-gray-200 rounded-xl hover:border-purple-300 transition-colors shadow-sm">
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <div className="p-1.5 bg-purple-100 rounded-lg text-purple-600">
                                        <Cpu className="w-4 h-4" />
                                    </div>
                                    <span className="font-bold text-sm text-gray-800">{trigger.action}</span>
                                </div>
                                <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${trigger.priority === 1 ? 'bg-red-100 text-red-600' : 'bg-gray-100 text-gray-500'
                                    }`}>
                                    Priority {trigger.priority}
                                </span>
                            </div>

                            <p className="text-xs text-gray-600 mb-3">
                                <b>Condition:</b> {trigger.condition}
                            </p>

                            <div className="flex items-center justify-between">
                                <div className="text-[10px] py-1 px-2 bg-gray-50 rounded text-gray-500 font-mono">
                                    Target: {trigger.target}
                                </div>
                                <button className="flex items-center gap-1 text-xs font-bold text-purple-600 hover:text-purple-700 bg-purple-50 px-3 py-1.5 rounded-lg transition-all active:scale-95">
                                    <Play className="w-3 h-3" /> Execute Mock
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="text-center py-10 text-gray-500 bg-gray-50 rounded-xl border border-dashed border-gray-200">
                    <p>No active automation triggers.</p>
                    <p className="text-xs">Conditions for automation (like high solar excess) are not met currently.</p>
                </div>
            )}
        </div>
    )
}
