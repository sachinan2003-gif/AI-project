import { UploadForm } from "@/components/upload-form"
import { Card } from "@/components/ui/card"

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">TB Chest X-Ray Detector</h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Upload chest X-ray images to get real-time tuberculosis detection using advanced AI technology. Our system
            provides accurate predictions with visual explanations.
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto">
          <Card className="p-8 shadow-xl bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
            <UploadForm />
          </Card>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 text-sm text-gray-500 dark:text-gray-400">
          <p>This tool is for educational purposes. Always consult healthcare professionals for medical diagnosis.</p>
        </div>
      </div>
    </main>
  )
}
