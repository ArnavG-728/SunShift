'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, Target, Users, Zap, Globe, Cpu, Sun, BarChart3, Leaf, Battery, Sparkles } from 'lucide-react'

export default function AboutUs() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-orange-50 via-yellow-50 to-blue-50 relative">

            {/* Navigation */}
            <nav className="sticky top-0 z-50 bg-white/50 backdrop-blur-md border-b border-white/20">
                <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
                    <Link
                        href="/"
                        className="flex items-center gap-2 text-gray-600 hover:text-orange-600 transition-colors duration-300"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span className="font-medium">Back to Home</span>
                    </Link>
                    <div className="flex items-center gap-2">
                        <div className="bg-gradient-to-br from-orange-500 to-yellow-500 p-1.5 rounded-lg shadow-md">
                            <Sun className="h-5 w-5 text-white" />
                        </div>
                        <span className="text-xl font-bold bg-gradient-to-r from-orange-600 to-yellow-600 bg-clip-text text-transparent">
                            SunShift
                        </span>
                    </div>
                </div>
            </nav>

            <main className="max-w-7xl mx-auto px-6 py-12 space-y-24">

                {/* Header Section */}
                <section className="text-center space-y-6 animate-in fade-in slide-in-from-bottom-5 duration-700">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100 text-blue-700 text-sm font-medium">
                        <Globe className="w-4 h-4" />
                        Our Mission
                    </div>
                    <h1 className="text-4xl md:text-6xl font-extrabold text-gray-900 leading-tight">
                        Empowering the World with <br />
                        <span className="bg-gradient-to-r from-orange-500 to-yellow-500 bg-clip-text text-transparent">
                            Solar Intelligence
                        </span>
                    </h1>
                    <p className="max-w-3xl mx-auto text-xl text-gray-600 leading-relaxed">
                        We are democratizing solar energy optimization. By bridging the gap between complex data and everyday users, we help you save money and reduce your carbon footprint—one kilowatt-hour at a time.
                    </p>
                </section>

                {/* The Problem & Solution Grid */}
                <section className="grid md:grid-cols-2 gap-12 items-stretch">
                    <div className="bg-white/60 backdrop-blur-sm p-8 rounded-3xl border border-white/50 shadow-xl hover:shadow-2xl transition-all duration-500 group">
                        <div className="w-12 h-12 bg-red-100 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                            <Target className="w-6 h-6 text-red-600" />
                        </div>
                        <h2 className="text-2xl font-bold text-gray-900 mb-4">The Challenge</h2>
                        <p className="text-gray-600 leading-relaxed">
                            Homeowners and businesses lose <strong>30-40%</strong> of potential savings because they lack real-time optimization tools.
                            Most existing solutions are expensive, complex, or require specialized hardware.
                            This leads to inefficient consumption, missed financial returns, and unnecessary reliance on the grid.
                        </p>
                    </div>

                    <div className="bg-white/60 backdrop-blur-sm p-8 rounded-3xl border border-white/50 shadow-xl hover:shadow-2xl transition-all duration-500 group">
                        <div className="w-12 h-12 bg-green-100 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                            <Sparkles className="w-6 h-6 text-green-600" />
                        </div>
                        <h2 className="text-2xl font-bold text-gray-900 mb-4">Our Solution</h2>
                        <p className="text-gray-600 leading-relaxed">
                            SunShift combines <strong>real-time weather data</strong>, <strong>NASA historical records</strong>, and <strong>smart AI agents</strong> to provide accurate production forecasts.
                            We give you actionable insights on when to run your appliances, charge your EV, or store energy—completely free.
                        </p>
                    </div>
                </section>

                {/* How It Works / Key Features */}
                <section className="space-y-12">
                    <div className="text-center">
                        <h2 className="text-3xl font-bold text-gray-900 mb-4">How SunShift Works</h2>
                        <p className="text-gray-600">Advanced technology working seamlessly in the background.</p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {[
                            {
                                icon: <Cpu className="w-6 h-6 text-purple-600" />,
                                bg: "bg-purple-100",
                                title: "Hybrid Forecasting",
                                desc: "We merge live weather feeds with physics-based models to predict solar output with 85-95% accuracy."
                            },
                            {
                                icon: <Battery className="w-6 h-6 text-yellow-600" />,
                                bg: "bg-yellow-100",
                                title: "AI Optimization",
                                desc: "Our intelligent agents analyze your usage and suggest the perfect time for high-energy tasks."
                            },
                            {
                                icon: <BarChart3 className="w-6 h-6 text-blue-600" />,
                                bg: "bg-blue-100",
                                title: "Actionable Insights",
                                desc: "Visualize your production, savings, and environmental impact in a simple, intuitive dashboard."
                            }
                        ].map((feature, idx) => (
                            <div key={idx} className="bg-white/40 p-6 rounded-2xl border border-white/30 shadow-lg hover:-translate-y-1 transition-transform duration-300">
                                <div className={`w-12 h-12 ${feature.bg} rounded-xl flex items-center justify-center mb-4`}>
                                    {feature.icon}
                                </div>
                                <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
                                <p className="text-gray-600">{feature.desc}</p>
                            </div>
                        ))}
                    </div>
                </section>

                {/* Impact Section */}
                <section className="bg-gradient-to-r from-gray-900 to-gray-800 rounded-[2.5rem] p-8 md:p-12 text-white overflow-hidden relative">
                    <div className="absolute top-0 right-0 w-96 h-96 bg-white/5 rounded-full blur-3xl pointer-events-none -translate-y-1/2 translate-x-1/2"></div>

                    <div className="grid md:grid-cols-2 gap-12 items-center relative z-10">
                        <div>
                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 text-white/90 text-sm font-medium mb-6">
                                <Leaf className="w-4 h-4" />
                                Global Impact
                            </div>
                            <h2 className="text-3xl md:text-4xl font-bold mb-6">Making a Real Difference</h2>
                            <p className="text-gray-300 mb-8 leading-relaxed">
                                By optimizing energy usage, we're not just saving money; we're reducing grid dependency and lowering carbon emissions.
                                Every kilowatt-hour of solar energy used efficiently is a step towards a greener planet.
                            </p>
                            <div className="space-y-4">
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center font-bold text-green-400">1</div>
                                    <p>Empowering homeowners to maximize their solar investment.</p>
                                </div>
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center font-bold text-green-400">2</div>
                                    <p>Reducing reliance on fossil-fuel heavy grid power.</p>
                                </div>
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center font-bold text-green-400">3</div>
                                    <p>Creating a sustainable community of energy-conscious users.</p>
                                </div>
                            </div>
                        </div>

                        <div className="bg-white/10 rounded-3xl p-8 backdrop-blur-md border border-white/10">
                            <h3 className="text-xl font-bold mb-6">Who We Help</h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-colors">
                                    <Users className="w-6 h-6 text-orange-400 mb-2" />
                                    <p className="font-semibold">Homeowners</p>
                                </div>
                                <div className="p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-colors">
                                    <Zap className="w-6 h-6 text-yellow-400 mb-2" />
                                    <p className="font-semibold">Businesses</p>
                                </div>
                                <div className="p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-colors">
                                    <Sun className="w-6 h-6 text-blue-400 mb-2" />
                                    <p className="font-semibold">Installers</p>
                                </div>
                                <div className="p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-colors">
                                    <Leaf className="w-6 h-6 text-green-400 mb-2" />
                                    <p className="font-semibold">The Planet</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* CTA */}
                <section className="text-center py-12">
                    <h2 className="text-3xl font-bold text-gray-900 mb-6">Ready to Optimize Your Energy?</h2>
                    <Link
                        href="/dashboard"
                        className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-full text-white font-bold text-lg shadow-xl hover:shadow-2xl hover:-translate-y-1 transition-all duration-300"
                    >
                        Get Started Now
                    </Link>
                </section>

            </main>
        </div>
    )
}
