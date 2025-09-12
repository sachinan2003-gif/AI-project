"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Eye, EyeOff } from "lucide-react"

interface HeatmapDisplayProps {
  originalImage: string
  heatmapImage: string
}

export function HeatmapDisplay({ originalImage, heatmapImage }: HeatmapDisplayProps) {
  const [showHeatmap, setShowHeatmap] = useState(true)

  return (
    <div className="space-y-4">
      <div className="flex justify-center">
        <Button variant="outline" size="sm" onClick={() => setShowHeatmap(!showHeatmap)}>
          {showHeatmap ? (
            <>
              <EyeOff className="w-4 h-4 mr-2" />
              Hide Heatmap
            </>
          ) : (
            <>
              <Eye className="w-4 h-4 mr-2" />
              Show Heatmap
            </>
          )}
        </Button>
      </div>

      <div className="relative inline-block mx-auto">
        <img
          src={originalImage || "/placeholder.svg"}
          alt="Original X-ray"
          className="max-w-full max-h-96 rounded-lg shadow-lg"
        />
        {showHeatmap && (
          <img
            src={`data:image/png;base64,${heatmapImage}`}
            alt="Heatmap overlay"
            className="absolute inset-0 max-w-full max-h-96 rounded-lg opacity-60 mix-blend-multiply"
          />
        )}
      </div>

      <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
        <div className="text-center">
          <div className="w-4 h-4 bg-blue-500 rounded mx-auto mb-2"></div>
          <p className="text-xs text-gray-600 dark:text-gray-400">Low Attention</p>
        </div>
        <div className="text-center">
          <div className="w-4 h-4 bg-red-500 rounded mx-auto mb-2"></div>
          <p className="text-xs text-gray-600 dark:text-gray-400">High Attention</p>
        </div>
      </div>
    </div>
  )
}
