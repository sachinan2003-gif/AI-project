"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { HeatmapDisplay } from "@/components/heatmap-display"
import { Download, RotateCcw } from "lucide-react"

interface PredictionResult {
  result: "TB" | "Normal"
  confidence: number
  heatmap?: string
}

interface ResultCardProps {
  prediction: PredictionResult
  originalImage: string
  onReset: () => void
}

export function ResultCard({ prediction, originalImage, onReset }: ResultCardProps) {
  const isPositive = prediction.result === "TB"
  const confidencePercentage = Math.round(prediction.confidence * 100)

  const handleDownloadReport = () => {
    // Create a simple text report
    const report = `
TB Chest X-Ray Detection Report
==============================

Prediction: ${prediction.result}
Confidence: ${confidencePercentage}%
Date: ${new Date().toLocaleDateString()}
Time: ${new Date().toLocaleTimeString()}

Note: This is an AI-generated prediction for educational purposes only.
Please consult with healthcare professionals for medical diagnosis.
    `.trim()

    const blob = new Blob([report], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `tb-detection-report-${Date.now()}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <Card className="p-6 space-y-6">
      <div className="text-center space-y-4">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Prediction Results</h3>

        <div className="flex items-center justify-center gap-4">
          <Badge variant={isPositive ? "destructive" : "default"} className="text-lg px-4 py-2">
            {prediction.result}
          </Badge>
          <div className="text-right">
            <p className="text-sm text-gray-600 dark:text-gray-400">Confidence</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{confidencePercentage}%</p>
          </div>
        </div>

        {isPositive && (
          <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <p className="text-red-700 dark:text-red-400 text-sm">
              <strong>Important:</strong> This AI prediction suggests possible tuberculosis signs. Please consult a
              healthcare professional immediately for proper medical evaluation and diagnosis.
            </p>
          </div>
        )}
      </div>

      {/* Heatmap Visualization */}
      {prediction.heatmap && (
        <div className="space-y-4">
          <h4 className="text-lg font-semibold text-gray-900 dark:text-white text-center">AI Analysis Visualization</h4>
          <HeatmapDisplay originalImage={originalImage} heatmapImage={prediction.heatmap} />
          <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
            The highlighted areas show regions the AI model focused on during analysis
          </p>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-4 justify-center">
        <Button onClick={onReset} variant="outline">
          <RotateCcw className="w-4 h-4 mr-2" />
          Analyze Another
        </Button>
        <Button onClick={handleDownloadReport}>
          <Download className="w-4 h-4 mr-2" />
          Download Report
        </Button>
      </div>
    </Card>
  )
}
