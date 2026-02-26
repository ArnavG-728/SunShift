'use client'

import { useState } from 'react'
import { HelpCircle, X, Info } from 'lucide-react'

interface BoxGuideProps {
  title: string
  children: React.ReactNode
}

export default function BoxGuide({ title, children }: BoxGuideProps) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      <button
        onClick={(e) => {
          e.stopPropagation()
          setIsOpen(true)
        }}
        className="p-1.5 text-white/80 hover:text-white hover:bg-white/20 rounded-full transition-colors z-10"
        title="Guide & Information"
      >
        <HelpCircle className="w-5 h-5" />
      </button>

      {isOpen && (
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6 bg-black/50 backdrop-blur-sm"
          onClick={(e) => {
            e.stopPropagation()
            setIsOpen(false)
          }}
        >
          <div 
            className="bg-white rounded-2xl shadow-2xl w-full max-w-xl max-h-[85vh] overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-4 sm:p-5 flex items-center justify-between text-white shrink-0">
              <div className="flex items-center gap-3">
                <div className="bg-white/20 p-2 rounded-lg">
                  <Info className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-lg">{title}</h3>
                  <p className="text-blue-100 text-xs">Understanding this component</p>
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setIsOpen(false)
                }}
                className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                title="Close Guide"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {/* Content Body */}
            <div className="p-5 sm:p-6 overflow-y-auto text-gray-700 text-sm space-y-4 flex-1 guide-content">
              {children}
            </div>
          </div>
        </div>
      )}
    </>
  )
}
