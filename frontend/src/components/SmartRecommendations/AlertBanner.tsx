import { AlertTriangle } from 'lucide-react'

export default function AlertBanner({ alerts }: { alerts: any[] }) {
    if (!alerts || alerts.length === 0) return null;

    return (
        <div className="p-4 bg-yellow-50 border-b">
            <div className="space-y-2">
                {alerts.map((alert: any, index: number) => (
                    <div
                        key={index}
                        className={`flex items-start space-x-3 p-3 rounded-md ${alert.type === 'warning' ? 'bg-orange-50 border border-orange-200' : 'bg-blue-50 border border-blue-200'
                            }`}
                    >
                        <AlertTriangle className={`h-5 w-5 mt-0.5 ${alert.type === 'warning' ? 'text-orange-600' : 'text-blue-600'
                            }`} />
                        <div className="flex-1">
                            <p className="font-medium text-sm text-gray-900">{alert.title}</p>
                            <p className="text-sm text-gray-600 mt-1">{alert.message}</p>
                            <p className="text-xs text-gray-500 mt-1">💡 {alert.action}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
