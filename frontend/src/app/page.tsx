'use client'

import React from 'react'
import Link from 'next/link'
import { Sun, ArrowRight, Zap, Leaf, BarChart3, CloudSun } from 'lucide-react'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-yellow-50 to-blue-50 relative overflow-hidden">
      {/* Decorative Background Elements */}
      <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] bg-orange-200/30 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-[-10%] left-[-5%] w-[600px] h-[600px] bg-blue-200/20 rounded-full blur-3xl pointer-events-none" />

      {/* Navigation */}
      <nav className="relative z-10 max-w-7xl mx-auto px-6 py-6 flex justify-between items-center">
        <div className="flex items-center gap-2 group cursor-default">
          <div className="bg-gradient-to-br from-orange-500 to-yellow-500 p-2 rounded-xl shadow-lg group-hover:shadow-orange-500/20 transition-all duration-300">
            <Sun className="h-6 w-6 text-white" />
          </div>
          <span className="text-2xl font-bold bg-gradient-to-r from-orange-600 to-yellow-600 bg-clip-text text-transparent">
            SunShift
          </span>
        </div>
        <Link
          href="/dashboard"
          className="hidden md:flex items-center gap-2 px-5 py-2.5 rounded-full bg-white/50 hover:bg-white border border-white/60 hover:border-white shadow-sm hover:shadow-md transition-all duration-300 text-gray-700 font-medium backdrop-blur-sm"
        >
          Open Dashboard <ArrowRight className="w-4 h-4" />
        </Link>
      </nav>

      <main className="relative z-10 max-w-7xl mx-auto px-6 pt-10 pb-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">

          {/* Left Column: Content */}
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-5 duration-700">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-100/50 border border-green-200 text-green-700 text-sm font-medium">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              AI-Powered Solar Intelligence
            </div>

            <h1 className="text-5xl lg:text-7xl font-extrabold leading-tight text-gray-900">
              Optimize Your <br />
              <span className="bg-gradient-to-r from-orange-500 via-yellow-500 to-orange-500 bg-clip-text text-transparent bg-[length:200%_auto] animate-gradient">
                Solar Energy
              </span>
            </h1>

            <p className="text-lg text-gray-600 leading-relaxed max-w-xl">
              Harness the power of advanced AI to forecast solar production, optimize energy usage, and maximize your savings. SunShift bridges the gap between complex data and actionable insights.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <Link
                href="/dashboard"
                className="group relative px-8 py-4 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-full text-white font-semibold text-lg shadow-xl shadow-orange-500/20 hover:shadow-orange-500/40 hover:-translate-y-0.5 transition-all duration-300 overflow-hidden"
              >
                <span className="relative z-10 flex items-center justify-center gap-2">
                  Let's Get Started <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-orange-600 to-yellow-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </Link>
              <Link
                href="/analytics"
                className="px-8 py-4 bg-white/60 hover:bg-white rounded-full text-gray-700 font-semibold text-lg border border-white/60 hover:border-white shadow-sm hover:shadow-lg backdrop-blur-sm transition-all duration-300 flex items-center justify-center"
              >
                Learn More
              </Link>
            </div>

            <div className="pt-8 grid grid-cols-3 gap-6 border-t border-gray-200/50">
              <div>
                <p className="text-3xl font-bold text-gray-900">98%</p>
                <p className="text-sm text-gray-500">Forecast Accuracy</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-gray-900">24/7</p>
                <p className="text-sm text-gray-500">Real-time Monitoring</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-gray-900">10k+</p>
                <p className="text-sm text-gray-500">Data Points Analyzed</p>
              </div>
            </div>
          </div>

          {/* Right Column: Visuals */}
          <div className="relative animate-in fade-in slide-in-from-right-5 duration-1000 delay-200 hidden lg:block">
            <div className="relative z-10 bg-white/40 backdrop-blur-xl border border-white/50 rounded-3xl p-6 shadow-2xl">
              {/* Feature Cards Grid within the "Glass" container */}
              <div className="grid gap-4">
                {/* Card 1 */}
                <div className="flex items-start gap-4 p-4 bg-white/80 rounded-2xl shadow-sm hover:shadow-md transition-shadow">
                  <div className="p-3 bg-blue-100 rounded-xl text-blue-600">
                    <BarChart3 className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">Energy Forecasting</h3>
                    <p className="text-sm text-gray-500">Precise production estimates using ML.</p>
                  </div>
                </div>

                {/* Card 2 */}
                <div className="flex items-start gap-4 p-4 bg-white/80 rounded-2xl shadow-sm hover:shadow-md transition-shadow">
                  <div className="p-3 bg-green-100 rounded-xl text-green-600">
                    <Leaf className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">Eco Impact</h3>
                    <p className="text-sm text-gray-500">Track your carbon footprint reduction.</p>
                  </div>
                </div>

                {/* Card 3 */}
                <div className="flex items-start gap-4 p-4 bg-white/80 rounded-2xl shadow-sm hover:shadow-md transition-shadow">
                  <div className="p-3 bg-yellow-100 rounded-xl text-yellow-600">
                    <Zap className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">Smart Optimization</h3>
                    <p className="text-sm text-gray-500">Automated appliance scheduling.</p>
                  </div>
                </div>
              </div>

              {/* Simulated Chart/Graph Graphic */}
              <div className="mt-6 p-4 bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl text-white">
                <div className="flex justify-between items-center mb-4">
                  <div className="flex items-center gap-2">
                    <CloudSun className="w-5 h-5 text-yellow-500" />
                    <span className="text-sm font-medium">Solar Production</span>
                  </div>
                  <span className="text-xs text-green-400">+12% vs last week</span>
                </div>
                <div className="h-32 flex items-end justify-between gap-1">
                  {[40, 65, 45, 80, 55, 90, 70].map((h, i) => (
                    <div key={i} className="w-full bg-gradient-to-t from-orange-500/20 to-orange-500 rounded-t-sm transition-all hover:bg-orange-400" style={{ height: `${h}%` }}></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Floating Elements */}
            <div className="absolute -top-6 -right-6 p-4 bg-white rounded-2xl shadow-xl animate-bounce duration-[3000ms]">
              <Sun className="w-8 h-8 text-orange-500" />
            </div>
            <div className="absolute -bottom-6 -left-6 p-4 bg-white rounded-2xl shadow-xl animate-bounce duration-[4000ms]">
              <Zap className="w-8 h-8 text-yellow-500" />
            </div>
          </div>
        </div>
      </main>

      <style jsx global>{`
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient {
          animation: gradient 3s ease infinite;
        }
      `}</style>
    </div>
  )
}
