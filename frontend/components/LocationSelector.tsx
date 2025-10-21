'use client'

import { useState, useEffect, useRef } from 'react'
import { MapPin, Search, X, Navigation, Loader } from 'lucide-react'

interface LocationSelectorProps {
  latitude: number
  longitude: number
  city: string
  onLocationChange: (location: { latitude: number; longitude: number; city: string }) => void
}

interface SearchResult {
  name: string
  display_name: string
  lat: string
  lon: string
  type: string
}

export default function LocationSelector({
  latitude,
  longitude,
  city,
  onLocationChange
}: LocationSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [selectedLat, setSelectedLat] = useState(latitude)
  const [selectedLon, setSelectedLon] = useState(longitude)
  const [mapCenter, setMapCenter] = useState({ lat: latitude, lon: longitude })
  const [zoom, setZoom] = useState(10)
  const searchTimeoutRef = useRef<NodeJS.Timeout>()

  // Search for locations using Nominatim (OpenStreetMap)
  const searchLocation = async (query: string) => {
    if (!query || query.length < 3) {
      setSearchResults([])
      return
    }

    setSearching(true)
    try {
      const response = await fetch(
        `/api/geocode/search?q=${encodeURIComponent(query)}&limit=5`
      )
      const data = await response.json()
      setSearchResults(data)
    } catch (error) {
      console.error('Error searching location:', error)
    } finally {
      setSearching(false)
    }
  }

  // Debounced search
  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current)
    }

    searchTimeoutRef.current = setTimeout(() => {
      searchLocation(searchQuery)
    }, 500)

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current)
      }
    }
  }, [searchQuery])

  // Get current location from browser
  const getCurrentLocation = () => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const lat = position.coords.latitude
          const lon = position.coords.longitude
          
          setSelectedLat(lat)
          setSelectedLon(lon)
          setMapCenter({ lat, lon })
          
          // Reverse geocode to get city name
          try {
            const response = await fetch(
              `/api/geocode/reverse?lat=${lat}&lon=${lon}`
            )
            const data = await response.json()
            const cityName = data.address?.city || data.address?.town || data.address?.village || 'Unknown Location'
            
            onLocationChange({
              latitude: lat,
              longitude: lon,
              city: cityName
            })
            
            setIsOpen(false)
          } catch (error) {
            console.error('Error reverse geocoding:', error)
          }
        },
        (error) => {
          console.error('Error getting location:', error)
          alert('Unable to get your location. Please enable location services.')
        }
      )
    } else {
      alert('Geolocation is not supported by your browser.')
    }
  }

  const selectSearchResult = (result: SearchResult) => {
    const lat = parseFloat(result.lat)
    const lon = parseFloat(result.lon)
    
    setSelectedLat(lat)
    setSelectedLon(lon)
    setMapCenter({ lat, lon })
    setSearchQuery('')
    setSearchResults([])
    
    onLocationChange({
      latitude: lat,
      longitude: lon,
      city: result.name || result.display_name.split(',')[0]
    })
    
    setIsOpen(false)
  }

  const handleMapClick = async (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    // Convert pixel coordinates to lat/lon (simplified)
    const mapWidth = rect.width
    const mapHeight = rect.height
    
    const lonRange = 360 / Math.pow(2, zoom)
    const latRange = 180 / Math.pow(2, zoom)
    
    const lon = mapCenter.lon + ((x / mapWidth) - 0.5) * lonRange
    const lat = mapCenter.lat - ((y / mapHeight) - 0.5) * latRange
    
    setSelectedLat(lat)
    setSelectedLon(lon)
    
    // Reverse geocode
    try {
      const response = await fetch(
        `/api/geocode/reverse?lat=${lat}&lon=${lon}`
      )
      const data = await response.json()
      const cityName = data.address?.city || data.address?.town || data.address?.village || 'Custom Location'
      
      onLocationChange({
        latitude: lat,
        longitude: lon,
        city: cityName
      })
    } catch (error) {
      console.error('Error reverse geocoding:', error)
    }
  }

  const applyLocation = () => {
    setIsOpen(false)
  }

  return (
    <div className="relative">
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="flex items-center space-x-2 px-4 py-2 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors"
      >
        <MapPin className="h-4 w-4" />
        <span className="text-sm font-medium">
          {city} ({latitude.toFixed(4)}, {longitude.toFixed(4)})
        </span>
      </button>

      {/* Modal */}
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
            {/* Header */}
            <div className="p-4 border-b flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Select Location</h3>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1 hover:bg-gray-100 rounded-md"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Content */}
            <div className="p-4 space-y-4 max-h-[calc(90vh-8rem)] overflow-y-auto">
              {/* Search Bar */}
              <div className="space-y-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search for a city or address..."
                    className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  {searching && (
                    <Loader className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400 animate-spin" />
                  )}
                </div>

                {/* Search Results */}
                {searchResults.length > 0 && (
                  <div className="border rounded-lg divide-y max-h-48 overflow-y-auto">
                    {searchResults.map((result, index) => (
                      <button
                        key={index}
                        onClick={() => selectSearchResult(result)}
                        className="w-full px-4 py-2 text-left hover:bg-gray-50 transition-colors"
                      >
                        <p className="font-medium text-sm text-gray-900">{result.name}</p>
                        <p className="text-xs text-gray-500">{result.display_name}</p>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Quick Actions */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={getCurrentLocation}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  <Navigation className="h-4 w-4" />
                  <span className="text-sm">Use My Location</span>
                </button>
                
                <div className="flex-1 text-sm text-gray-600">
                  or click on the map below to drop a pin
                </div>
              </div>

              {/* Map Display */}
              <div className="space-y-2">
                <div className="relative bg-gray-100 rounded-lg overflow-hidden border-2 border-gray-300">
                  {/* OpenStreetMap Tile */}
                  <div
                    className="relative w-full h-96 cursor-crosshair"
                    onClick={handleMapClick}
                    style={{
                      backgroundImage: `url(https://tile.openstreetmap.org/${zoom}/${Math.floor((mapCenter.lon + 180) / 360 * Math.pow(2, zoom))}/${Math.floor((1 - Math.log(Math.tan(mapCenter.lat * Math.PI / 180) + 1 / Math.cos(mapCenter.lat * Math.PI / 180)) / Math.PI) / 2 * Math.pow(2, zoom))}.png)`,
                      backgroundSize: 'cover',
                      backgroundPosition: 'center'
                    }}
                  >
                    {/* Selected Pin */}
                    <div
                      className="absolute transform -translate-x-1/2 -translate-y-full"
                      style={{
                        left: '50%',
                        top: '50%'
                      }}
                    >
                      <MapPin className="h-8 w-8 text-red-600 drop-shadow-lg" fill="currentColor" />
                    </div>

                    {/* Zoom Controls */}
                    <div className="absolute top-4 right-4 flex flex-col space-y-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setZoom(Math.min(18, zoom + 1))
                        }}
                        className="bg-white p-2 rounded-md shadow-md hover:bg-gray-50"
                      >
                        <span className="text-lg font-bold">+</span>
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setZoom(Math.max(3, zoom - 1))
                        }}
                        className="bg-white p-2 rounded-md shadow-md hover:bg-gray-50"
                      >
                        <span className="text-lg font-bold">−</span>
                      </button>
                    </div>

                    {/* Coordinates Display */}
                    <div className="absolute bottom-4 left-4 bg-white px-3 py-2 rounded-md shadow-md">
                      <p className="text-xs font-mono">
                        {selectedLat.toFixed(6)}, {selectedLon.toFixed(6)}
                      </p>
                    </div>
                  </div>
                </div>

                <p className="text-xs text-gray-500 text-center">
                  Map tiles © OpenStreetMap contributors
                </p>
              </div>

              {/* Current Selection */}
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm font-medium text-gray-900 mb-1">Selected Location:</p>
                <p className="text-sm text-gray-700">
                  Latitude: <span className="font-mono">{selectedLat.toFixed(6)}</span>
                </p>
                <p className="text-sm text-gray-700">
                  Longitude: <span className="font-mono">{selectedLon.toFixed(6)}</span>
                </p>
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t flex items-center justify-end space-x-3">
              <button
                onClick={() => setIsOpen(false)}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={applyLocation}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Apply Location
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
