import { Clock, Calendar, TrendingUp, Activity, AlertCircle } from 'lucide-react'

export default function InsightsPanel({ insights }: { insights: any }) {
    if (!insights || typeof insights !== 'object') return null;

    const formatInsightText = (text: string) => {
        return String(text).replace(/\*\*/g, '').replace(/^#+\s/gm, '');
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            {insights.next_24h && (
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg shadow p-6 border border-green-100">
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <Clock className="w-5 h-5 text-green-600" />
                        Next 24 Hours
                    </h3>
                    <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                        {formatInsightText(insights.next_24h)}
                    </div>
                </div>
            )}

            {insights.next_7d && (
                <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg shadow p-6 border border-purple-100">
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <Calendar className="w-5 h-5 text-purple-600" />
                        7-Day Outlook
                    </h3>
                    <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                        {formatInsightText(insights.next_7d)}
                    </div>
                </div>
            )}

            {insights.recommendations && (
                <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-lg shadow p-6 border border-orange-100">
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-orange-600" />
                        Recommendations
                    </h3>
                    <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                        {formatInsightText(insights.recommendations)}
                    </div>
                </div>
            )}

            {insights.model_performance && (
                <div className="bg-gradient-to-br from-cyan-50 to-blue-50 rounded-lg shadow p-6 border border-cyan-100">
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-cyan-600" />
                        Model Performance
                    </h3>
                    <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                        {formatInsightText(insights.model_performance)}
                    </div>
                </div>
            )}

            {insights.weather_impact && (
                <div className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-lg shadow p-6 border border-yellow-100 md:col-span-2">
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <AlertCircle className="w-5 h-5 text-yellow-600" />
                        Weather Impact
                    </h3>
                    <div className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                        {formatInsightText(insights.weather_impact)}
                    </div>
                </div>
            )}
        </div>
    )
}
