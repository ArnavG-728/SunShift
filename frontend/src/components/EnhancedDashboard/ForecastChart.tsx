import { Activity } from 'lucide-react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts'

interface ForecastChartProps {
    currentData: any[];
    activeHorizon: '24h' | '7d';
}

export default function ForecastChart({ currentData, activeHorizon }: ForecastChartProps) {
    if (!currentData || currentData.length === 0) {
        return (
            <div className="h-96 flex items-center justify-center text-gray-500">
                <div className="text-center">
                    <Activity className="w-12 h-12 mx-auto mb-2 text-gray-400" />
                    <p>No forecast data available</p>
                    <p className="text-sm">Click "Run Forecast" to generate predictions</p>
                </div>
            </div>
        );
    }

    return (
        <ResponsiveContainer width="100%" height={400}>
            {activeHorizon === '24h' ? (
                <AreaChart data={currentData}>
                    <defs>
                        <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.1} />
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis
                        dataKey="time"
                        axisLine={{ stroke: '#e2e8f0' }}
                        tickLine={false}
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        dy={5}
                    />
                    <YAxis
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11, offset: 10 }}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#ffffff',
                            borderRadius: '12px',
                            border: '1px solid #f1f5f9',
                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.05)',
                            fontSize: '12px'
                        }}
                    />
                    <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ paddingBottom: '20px', fontSize: '12px' }} />
                    <Area type="monotone" dataKey="upper" stackId="1" stroke="none" fill="url(#colorConfidence)" name="Upper Range" />
                    <Area type="monotone" dataKey="lower" stackId="1" stroke="none" fill="url(#colorConfidence)" name="Lower Range" />
                    <Area type="monotone" dataKey="predicted" stroke="#f97316" strokeWidth={4} fill="url(#colorPredicted)" name="Forecasted Output" dot={false} />
                </AreaChart>
            ) : (
                <BarChart data={currentData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis
                        dataKey={'date'}
                        axisLine={{ stroke: '#e2e8f0' }}
                        tickLine={false}
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        dy={5}
                    />
                    <YAxis
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        label={{ value: 'Total kWh', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11, offset: 10 }}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#ffffff',
                            borderRadius: '12px',
                            border: '1px solid #f1f5f9',
                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.05)',
                            fontSize: '12px'
                        }}
                    />
                    <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ paddingBottom: '20px', fontSize: '12px' }} />
                    <Bar dataKey="total" fill="#f97316" radius={[4, 4, 0, 0]} name="Daily Total" />
                    {activeHorizon === '7d' && (
                        <>
                            <Bar dataKey="min" fill="#10b981" radius={[4, 4, 0, 0]} name="Minimum" />
                            <Bar dataKey="max" fill="#fbbf24" radius={[4, 4, 0, 0]} name="Maximum" />
                        </>
                    )}
                </BarChart>
            )}
        </ResponsiveContainer>
    )
}
