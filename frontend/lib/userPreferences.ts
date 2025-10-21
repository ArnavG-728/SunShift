/**
 * User Preferences Management
 * Stores user settings in localStorage (no database required)
 */

export interface SystemConfig {
  // Solar System Configuration
  systemSize: number // kWp
  panelEfficiency: number // 0-1 (e.g., 0.15 = 15%)
  panelTilt: number // degrees (0-90)
  panelAzimuth: number // degrees (0=North, 90=East, 180=South, 270=West)
  performanceRatio: number // 0-1 (typical 0.75-0.85)
  
  // Location
  city: string
  latitude: number
  longitude: number
  timezone: string
  
  // Financial
  electricityTariff: number // $/kWh
  feedInTariff: number // $/kWh for grid export
  currency: string
  
  // Battery (optional)
  hasBattery: boolean
  batteryCapacity: number // kWh
  batteryEfficiency: number // 0-1
  
  // Grid
  gridCO2Factor: number // kg CO2 per kWh
  maxGridImport: number // kW
  
  // Display Preferences
  temperatureUnit: 'C' | 'F'
  energyUnit: 'kWh' | 'MWh'
  theme: 'light' | 'dark' | 'auto'
  
  // Notifications
  enableAlerts: boolean
  alertThreshold: number // kWh threshold for low production alerts
}

export const DEFAULT_CONFIG: SystemConfig = {
  // Solar System
  systemSize: 5.0,
  panelEfficiency: 0.15,
  panelTilt: 30.0,
  panelAzimuth: 180.0,
  performanceRatio: 0.78,
  
  // Location (Delhi, India by default)
  city: 'Delhi (IN)',
  latitude: 28.6139,
  longitude: 77.2090,
  timezone: 'Asia/Kolkata',
  
  // Financial
  electricityTariff: 0.12,
  feedInTariff: 0.08,
  currency: 'USD',
  
  // Battery
  hasBattery: false,
  batteryCapacity: 0,
  batteryEfficiency: 0.95,
  
  // Grid
  gridCO2Factor: 0.70,
  maxGridImport: 10.0,
  
  // Display
  temperatureUnit: 'C',
  energyUnit: 'kWh',
  theme: 'auto',
  
  // Notifications
  enableAlerts: true,
  alertThreshold: 2.0,
}

const STORAGE_KEY = 'sunshift_user_preferences'

/**
 * Load user preferences from localStorage
 */
export function loadUserPreferences(): SystemConfig {
  if (typeof window === 'undefined') {
    return DEFAULT_CONFIG
  }
  
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      const parsed = JSON.parse(stored)
      // Merge with defaults to handle new fields
      return { ...DEFAULT_CONFIG, ...parsed }
    }
  } catch (error) {
    console.error('Error loading user preferences:', error)
  }
  
  return DEFAULT_CONFIG
}

/**
 * Save user preferences to localStorage
 */
export function saveUserPreferences(config: Partial<SystemConfig>): void {
  if (typeof window === 'undefined') {
    return
  }
  
  try {
    const current = loadUserPreferences()
    const updated = { ...current, ...config }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
  } catch (error) {
    console.error('Error saving user preferences:', error)
  }
}

/**
 * Reset to default preferences
 */
export function resetUserPreferences(): void {
  if (typeof window === 'undefined') {
    return
  }
  
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch (error) {
    console.error('Error resetting user preferences:', error)
  }
}

/**
 * Export preferences as JSON file
 */
export function exportPreferences(): void {
  const config = loadUserPreferences()
  const dataStr = JSON.stringify(config, null, 2)
  const dataBlob = new Blob([dataStr], { type: 'application/json' })
  const url = URL.createObjectURL(dataBlob)
  
  const link = document.createElement('a')
  link.href = url
  link.download = `sunshift-config-${new Date().toISOString().split('T')[0]}.json`
  link.click()
  
  URL.revokeObjectURL(url)
}

/**
 * Import preferences from JSON file
 */
export function importPreferences(file: File): Promise<SystemConfig> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    
    reader.onload = (e) => {
      try {
        const config = JSON.parse(e.target?.result as string)
        saveUserPreferences(config)
        resolve(config)
      } catch (error) {
        reject(new Error('Invalid configuration file'))
      }
    }
    
    reader.onerror = () => reject(new Error('Error reading file'))
    reader.readAsText(file)
  })
}

/**
 * Calculate optimal panel tilt based on latitude
 */
export function calculateOptimalTilt(latitude: number): number {
  // Rule of thumb: tilt ≈ latitude for year-round optimization
  return Math.abs(latitude)
}

/**
 * Calculate optimal panel azimuth based on hemisphere
 */
export function calculateOptimalAzimuth(latitude: number): number {
  // Northern hemisphere: South (180°)
  // Southern hemisphere: North (0°)
  return latitude >= 0 ? 180 : 0
}

/**
 * Validate system configuration
 */
export function validateConfig(config: Partial<SystemConfig>): string[] {
  const errors: string[] = []
  
  if (config.systemSize !== undefined && (config.systemSize <= 0 || config.systemSize > 1000)) {
    errors.push('System size must be between 0 and 1000 kWp')
  }
  
  if (config.panelEfficiency !== undefined && (config.panelEfficiency <= 0 || config.panelEfficiency > 1)) {
    errors.push('Panel efficiency must be between 0 and 1')
  }
  
  if (config.panelTilt !== undefined && (config.panelTilt < 0 || config.panelTilt > 90)) {
    errors.push('Panel tilt must be between 0 and 90 degrees')
  }
  
  if (config.panelAzimuth !== undefined && (config.panelAzimuth < 0 || config.panelAzimuth >= 360)) {
    errors.push('Panel azimuth must be between 0 and 360 degrees')
  }
  
  if (config.latitude !== undefined && (config.latitude < -90 || config.latitude > 90)) {
    errors.push('Latitude must be between -90 and 90')
  }
  
  if (config.longitude !== undefined && (config.longitude < -180 || config.longitude > 180)) {
    errors.push('Longitude must be between -180 and 180')
  }
  
  if (config.batteryCapacity !== undefined && config.batteryCapacity < 0) {
    errors.push('Battery capacity cannot be negative')
  }
  
  return errors
}

/**
 * Get location-specific recommendations
 */
export function getLocationRecommendations(latitude: number): {
  optimalTilt: number
  optimalAzimuth: number
  seasonalAdjustment: string
} {
  const optimalTilt = calculateOptimalTilt(latitude)
  const optimalAzimuth = calculateOptimalAzimuth(latitude)
  
  let seasonalAdjustment = ''
  const absLat = Math.abs(latitude)
  
  if (absLat < 15) {
    seasonalAdjustment = 'Tropical region: Keep tilt low (10-15°) for year-round optimization'
  } else if (absLat < 35) {
    seasonalAdjustment = 'Subtropical: Consider seasonal adjustment ±15° for summer/winter'
  } else if (absLat < 55) {
    seasonalAdjustment = 'Temperate: Seasonal adjustment recommended (summer: -15°, winter: +15°)'
  } else {
    seasonalAdjustment = 'High latitude: Steep tilt needed, consider seasonal tracking'
  }
  
  return {
    optimalTilt: Math.round(optimalTilt),
    optimalAzimuth,
    seasonalAdjustment
  }
}

/**
 * Preset configurations for common scenarios
 */
export const PRESET_CONFIGS = {
  residential_small: {
    systemSize: 3.0,
    panelEfficiency: 0.15,
    performanceRatio: 0.75,
    description: 'Small residential (3 kWp)'
  },
  residential_medium: {
    systemSize: 5.0,
    panelEfficiency: 0.18,
    performanceRatio: 0.78,
    description: 'Medium residential (5 kWp)'
  },
  residential_large: {
    systemSize: 10.0,
    panelEfficiency: 0.20,
    performanceRatio: 0.80,
    description: 'Large residential (10 kWp)'
  },
  commercial_small: {
    systemSize: 25.0,
    panelEfficiency: 0.18,
    performanceRatio: 0.82,
    description: 'Small commercial (25 kWp)'
  },
  commercial_medium: {
    systemSize: 50.0,
    panelEfficiency: 0.20,
    performanceRatio: 0.85,
    description: 'Medium commercial (50 kWp)'
  }
}
