"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Upload, X, Loader2 } from "lucide-react"
import { ResultCard } from "@/components/result-card"
import { cn } from "@/lib/utils"

interface PredictionResult {
  result: "TB" | "Normal"
  confidence: number
  heatmap?: string
}

export function UploadForm() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const handleFileSelect = useCallback((file: File) => {
    // Validate file type
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file")
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be less than 10MB")
      return
    }

    setError(null)
    setSelectedFile(file)
    setPrediction(null)

    // Create preview URL
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)

      const files = Array.from(e.dataTransfer.files)
      if (files.length > 0) {
        handleFileSelect(files[0])
      }
    },
    [handleFileSelect],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const handlePredict = async () => {
    if (!selectedFile) return

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Prediction failed")
      }

      const result: PredictionResult = await response.json()
      setPrediction(result)
    } catch (err) {
      setError("Failed to get prediction. Please try again.")
      console.error("Prediction error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setPrediction(null)
    setError(null)
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        className={cn(
          "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
          isDragOver
            ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20"
            : "border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500",
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {previewUrl ? (
          <div className="space-y-4">
            <div className="relative inline-block">
              <img
                src={previewUrl || "/placeholder.svg"}
                alt="Uploaded X-ray"
                className="max-w-full max-h-64 rounded-lg shadow-md"
              />
              <Button
                variant="destructive"
                size="sm"
                className="absolute -top-2 -right-2 rounded-full w-8 h-8 p-0"
                onClick={handleReset}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">{selectedFile?.name}</p>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="w-12 h-12 text-gray-400 mx-auto" />
            <div>
              <p className="text-lg font-medium text-gray-700 dark:text-gray-300">Drop your chest X-ray image here</p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">or click to browse files</p>
            </div>
            <input type="file" accept="image/*" onChange={handleFileInput} className="hidden" id="file-input" />
            <Button asChild variant="outline">
              <label htmlFor="file-input" className="cursor-pointer">
                Browse Files
              </label>
            </Button>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <Card className="p-4 border-red-200 bg-red-50 dark:bg-red-950/20 dark:border-red-800">
          <p className="text-red-700 dark:text-red-400 text-sm">{error}</p>
        </Card>
      )}

      {/* Predict Button */}
      {selectedFile && !prediction && (
        <div className="text-center">
          <Button onClick={handlePredict} disabled={isLoading} size="lg" className="px-8">
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              "Predict"
            )}
          </Button>
        </div>
      )}

      {/* Results */}
      {prediction && previewUrl && (
        <ResultCard prediction={prediction} originalImage={previewUrl} onReset={handleReset} />
      )}
    </div>
  )
}
