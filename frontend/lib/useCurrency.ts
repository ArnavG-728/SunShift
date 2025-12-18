'use client'

import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface CurrencyRates {
    USD: number
    EUR: number
    GBP: number
    INR: number
    AUD: number
    CAD: number
    JPY: number
    CNY: number
}

interface UseCurrencyResult {
    rates: CurrencyRates
    convert: (amount: number, from: string, to: string) => number
    formatCurrency: (amount: number, currency: string) => string
    loading: boolean
    error: string | null
    lastUpdated: Date | null
}

// Currency symbols
const CURRENCY_SYMBOLS: Record<string, string> = {
    USD: '$',
    EUR: '€',
    GBP: '£',
    INR: '₹',
    AUD: 'A$',
    CAD: 'C$',
    JPY: '¥',
    CNY: '¥'
}

// Fallback rates (approximate Dec 2024)
const FALLBACK_RATES: CurrencyRates = {
    USD: 1.0,
    EUR: 0.92,
    GBP: 0.79,
    INR: 83.5,
    AUD: 1.54,
    CAD: 1.36,
    JPY: 149.0,
    CNY: 7.14
}

// Cache for rates
let cachedRates: CurrencyRates | null = null
let cacheTimestamp: Date | null = null
const CACHE_DURATION_MS = 60 * 60 * 1000 // 1 hour

export function useCurrency(): UseCurrencyResult {
    const [rates, setRates] = useState<CurrencyRates>(FALLBACK_RATES)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

    useEffect(() => {
        // Check cache first
        if (cachedRates && cacheTimestamp) {
            const age = Date.now() - cacheTimestamp.getTime()
            if (age < CACHE_DURATION_MS) {
                setRates(cachedRates)
                setLastUpdated(cacheTimestamp)
                setLoading(false)
                return
            }
        }

        fetchRates()
    }, [])

    const fetchRates = async () => {
        try {
            setLoading(true)
            const response = await axios.get(`${API_BASE_URL}/currency/rates`, {
                params: { base: 'USD' }
            })

            if (response.data.rates) {
                const newRates = response.data.rates as CurrencyRates
                setRates(newRates)
                cachedRates = newRates
                cacheTimestamp = new Date()
                setLastUpdated(cacheTimestamp)
                setError(null)
            }
        } catch (err) {
            console.error('Error fetching currency rates:', err)
            setError('Using offline rates')
            // Keep using fallback rates
        } finally {
            setLoading(false)
        }
    }

    const convert = useCallback((amount: number, from: string, to: string): number => {
        if (from === to) return amount

        // Convert to USD first, then to target
        const fromRate = rates[from as keyof CurrencyRates] || 1
        const toRate = rates[to as keyof CurrencyRates] || 1

        // amount in FROM currency -> USD -> TO currency
        const amountInUSD = amount / fromRate
        const amountInTarget = amountInUSD * toRate

        return amountInTarget
    }, [rates])

    const formatCurrency = useCallback((amount: number, currency: string): string => {
        const symbol = CURRENCY_SYMBOLS[currency] || currency

        // Format based on currency
        if (currency === 'JPY') {
            return `${symbol}${Math.round(amount).toLocaleString()}`
        }

        return `${symbol}${amount.toFixed(2)}`
    }, [])

    return {
        rates,
        convert,
        formatCurrency,
        loading,
        error,
        lastUpdated
    }
}

// Utility function for one-time conversion without hook
export function convertCurrency(
    amount: number,
    from: string,
    to: string,
    rates: CurrencyRates = FALLBACK_RATES
): number {
    if (from === to) return amount

    const fromRate = rates[from as keyof CurrencyRates] || 1
    const toRate = rates[to as keyof CurrencyRates] || 1

    const amountInUSD = amount / fromRate
    return amountInUSD * toRate
}

export function getCurrencySymbol(currency: string): string {
    return CURRENCY_SYMBOLS[currency] || currency
}
