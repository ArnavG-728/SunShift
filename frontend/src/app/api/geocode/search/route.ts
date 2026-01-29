import { NextRequest, NextResponse } from 'next/server'

// Rate limiting: track last request time
let lastRequestTime = 0
const MIN_REQUEST_INTERVAL = 1000 // 1 second between requests

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const query = searchParams.get('q')
  const limit = searchParams.get('limit') || '5'

  if (!query) {
    return NextResponse.json({ error: 'Query parameter is required' }, { status: 400 })
  }

  try {
    // Rate limiting: ensure minimum interval between requests
    const now = Date.now()
    const timeSinceLastRequest = now - lastRequestTime
    if (timeSinceLastRequest < MIN_REQUEST_INTERVAL) {
      await new Promise(resolve => setTimeout(resolve, MIN_REQUEST_INTERVAL - timeSinceLastRequest))
    }
    lastRequestTime = Date.now()

    const response = await fetch(
      `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=${limit}&addressdetails=1`,
      {
        headers: {
          'User-Agent': 'SunShift/1.0 (Solar Energy Forecasting Application)',
          'Accept': 'application/json',
          'Accept-Language': 'en',
          'Referer': request.headers.get('referer') || 'http://localhost:3000',
        },
        cache: 'no-store',
      }
    )

    if (!response.ok) {
      console.error(`Nominatim returned status ${response.status}`)
      throw new Error(`Nominatim API error: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error fetching from Nominatim:', error)
    return NextResponse.json(
      { error: 'Failed to search location. Please try again in a moment.' },
      { status: 500 }
    )
  }
}
