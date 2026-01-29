'use client'

import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { Send, Bot, User, Loader2, Sparkles, RefreshCw, Zap, Sun, Battery, TrendingUp } from 'lucide-react'
import { useSystemConfig } from '@/lib/SystemConfigContext'
import { useCurrency, getCurrencySymbol } from '@/lib/useCurrency'

// Simple markdown renderer for bold (**text**) and bullet points
const renderMarkdown = (text: string) => {
  const lines = text.split('\n')
  return lines.map((line, lineIdx) => {
    // Process bold text: **text** -> <strong>text</strong>
    const parts = line.split(/(\*\*[^*]+\*\*)/g)
    const processedParts = parts.map((part, partIdx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={partIdx} className="font-semibold">{part.slice(2, -2)}</strong>
      }
      // Handle bullet points
      if (part.startsWith('• ') || part.startsWith('- ')) {
        return <span key={partIdx}>{part}</span>
      }
      return part
    })
    return (
      <span key={lineIdx}>
        {processedParts}
        {lineIdx < lines.length - 1 && <br />}
      </span>
    )
  })
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export default function ChatInterface() {
  const { config } = useSystemConfig()
  const { convert, formatCurrency } = useCurrency()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [quickStats, setQuickStats] = useState<any>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Fetch quick stats for context
  const fetchQuickStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/realtime/current`, {
        params: {
          lat: config.latitude,
          lon: config.longitude,
          system_size: config.systemSize,
          performance_ratio: config.performanceRatio
        }
      })
      if (response.data.status === 'success') {
        setQuickStats(response.data.data)
      }
    } catch (error) {
      console.error('Error fetching stats:', error)
    }
  }

  // Initialize with personalized greeting when config loads
  useEffect(() => {
    fetchQuickStats()

    const greeting = `Hello! I'm your **SunShift AI Assistant** for **${config.city}**.

**Your System:**
• ${config.systemSize} kWp solar array
• Panel tilt: ${config.panelTilt}° at ${config.panelAzimuth === 180 ? 'South' : config.panelAzimuth === 0 ? 'North' : config.panelAzimuth + '°'}
${config.hasBattery ? `• Battery: ${config.batteryCapacity} kWh` : '• No battery storage'}
• Electricity rate: ${formatCurrency(convert(config.electricityTariff, 'USD', config.currency), config.currency)}/kWh

I can help you:
- Understand your solar forecast and production
- Optimize energy usage patterns
- Calculate savings and ROI
- Recommend battery charging strategies
- Answer questions about your specific setup

**What would you like to know?**`

    setMessages([{
      role: 'assistant',
      content: greeting,
      timestamp: new Date()
    }])
  }, [config.city, config.systemSize, config.hasBattery, config.batteryCapacity, config.currency])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      // Build comprehensive context for the AI
      const currentTime = new Date().toLocaleTimeString()
      const currentDate = new Date().toLocaleDateString()

      let realtimeContext = ''
      if (quickStats) {
        realtimeContext = `
Current Real-Time Data (as of ${currentTime}):
- Current solar output: ${quickStats.energy_output_kWh?.toFixed(2)} kW
- Solar irradiance: ${quickStats.solar_irradiance?.toFixed(0)} W/m²
- Temperature: ${quickStats.temperature?.toFixed(1)}°C
- Cloud cover: ${quickStats.clouds}%
- Weather: ${quickStats.weather} - ${quickStats.description}
- Sunrise: ${quickStats.sunrise || 'N/A'}
- Sunset: ${quickStats.sunset || 'N/A'}
- Is daytime: ${quickStats.is_daytime ? 'Yes' : 'No'}
`
      }

      const systemContext = `
You are SunShift AI, an expert solar energy assistant. Be helpful, concise, and specific to this user's system.

User's System Configuration:
- Location: ${config.city} (Lat: ${config.latitude}, Lon: ${config.longitude})
- System Size: ${config.systemSize} kWp
- Panel Efficiency: ${(config.panelEfficiency * 100).toFixed(0)}%
- Panel Tilt: ${config.panelTilt}° 
- Panel Azimuth: ${config.panelAzimuth}° (${config.panelAzimuth === 180 ? 'South' : config.panelAzimuth === 0 ? 'North' : config.panelAzimuth === 90 ? 'East' : 'West'})
- Performance Ratio: ${(config.performanceRatio * 100).toFixed(0)}%
- Has Battery: ${config.hasBattery ? `Yes, ${config.batteryCapacity} kWh capacity` : 'No'}
- Electricity Tariff: ${formatCurrency(convert(config.electricityTariff, 'USD', config.currency), config.currency)}/kWh
- Feed-in Tariff: ${formatCurrency(convert(config.feedInTariff, 'USD', config.currency), config.currency)}/kWh
- Grid CO2 Factor: ${config.gridCO2Factor} kg/kWh

${realtimeContext}

Current Date/Time: ${currentDate} ${currentTime}

When giving financial advice, always use ${config.currency} (${getCurrencySymbol(config.currency)}) for all monetary values.
When asked about forecasts, be specific about timeframes.
If asked about battery, ${config.hasBattery ? 'provide battery-specific advice' : 'suggest they might benefit from battery storage'}.

User's Question: ${input}

Respond concisely and helpfully. Use bullet points for lists. Be specific to their system.
`.trim()

      const response = await axios.post(`${API_BASE_URL}/chat`, {
        query: systemContext
      })

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error connecting to the AI service. Please ensure the backend is running and try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Suggested questions based on user's system and time
  const hour = new Date().getHours()
  const suggestedQuestions = [
    hour < 12 ? "What's my expected production today?" : "How did my system perform today?",
    "What appliances should I run now to maximize solar usage?",
    config.hasBattery
      ? "What's the optimal battery strategy for today?"
      : "Would a battery system make sense for me?",
    `How can I maximize savings with my ${config.systemSize} kWp system?`
  ]

  return (
    <div className="bg-white rounded-lg shadow-lg min-h-[calc(100vh-4rem)] md:min-h-[calc(100vh-6rem)] flex flex-col">
      {/* Chat Header */}
      <div className="border-b px-6 py-4 bg-gradient-to-r from-green-50 to-emerald-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-br from-green-500 to-emerald-600 p-2.5 rounded-xl shadow-md">
              <Bot className="h-6 w-6 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                SunShift AI
                <Sparkles className="h-4 w-4 text-yellow-500" />
              </h2>
              <p className="text-sm text-gray-500">Your personal solar energy expert</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {quickStats && (
              <div className="hidden md:flex items-center gap-2 text-xs bg-white px-3 py-1.5 rounded-full border">
                <Sun className="w-3 h-3 text-yellow-500" />
                <span className="text-gray-600">Now: {quickStats.energy_output_kWh?.toFixed(1)} kW</span>
              </div>
            )}
            <button
              onClick={fetchQuickStats}
              className="p-2 hover:bg-white rounded-lg transition-colors"
              title="Refresh stats"
            >
              <RefreshCw className="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex space-x-3 max-w-3xl ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
              <div className={`flex-shrink-0 ${message.role === 'user' ? 'bg-gradient-to-br from-green-500 to-emerald-600' : 'bg-gray-100'} p-2 rounded-xl h-fit shadow-sm`}>
                {message.role === 'user' ? (
                  <User className="h-5 w-5 text-white" />
                ) : (
                  <Bot className="h-5 w-5 text-gray-600" />
                )}
              </div>
              <div>
                <div
                  className={`rounded-2xl px-4 py-3 ${message.role === 'user'
                    ? 'bg-gradient-to-br from-green-500 to-emerald-600 text-white'
                    : 'bg-gray-100 text-gray-900'
                    }`}
                >
                  <div className="text-sm whitespace-pre-wrap leading-relaxed">{renderMarkdown(message.content)}</div>
                </div>
                <p className="text-xs text-gray-400 mt-1 px-2">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="flex space-x-3 max-w-3xl">
              <div className="bg-gray-100 p-2 rounded-xl h-fit">
                <Bot className="h-5 w-5 text-gray-600" />
              </div>
              <div className="bg-gray-100 rounded-2xl px-4 py-3">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 text-green-600 animate-spin" />
                  <span className="text-sm text-gray-500">Thinking...</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions */}
      {messages.length <= 1 && (
        <div className="px-4 md:px-6 pb-4">
          <p className="text-xs text-gray-400 mb-2">Suggested questions:</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => setInput(question)}
                className="text-left text-sm bg-gradient-to-r from-green-50 to-emerald-50 hover:from-green-100 hover:to-emerald-100 text-gray-700 px-3 py-2 rounded-lg transition-colors border border-green-100"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="border-t px-6 py-4 bg-gray-50">
        <div className="flex space-x-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={`Ask about your ${config.systemSize} kWp system in ${config.city}...`}
            className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent bg-white"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-3 rounded-xl hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center space-x-2 shadow-md hover:shadow-lg"
          >
            <Send className="h-5 w-5" />
            <span className="hidden sm:inline">Send</span>
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2 text-center">
          Powered by AI • Using real-time data from {config.city}
        </p>
      </div>
    </div>
  )
}
