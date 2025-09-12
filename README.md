# TB Chest X-Ray Detection Platform

A full-stack web application for tuberculosis detection in chest X-ray images using AI technology. The system provides real-time predictions with visual explanations through Grad-CAM heatmaps.

## Features

- **Modern Web Interface**: Clean, responsive UI built with Next.js and Tailwind CSS
- **AI-Powered Detection**: Deep learning model for TB classification
- **Visual Explanations**: Grad-CAM heatmaps showing model attention areas
- **Real-time Predictions**: Instant analysis of uploaded X-ray images
- **Professional Reports**: Downloadable prediction reports
- **Drag & Drop Upload**: Intuitive file upload with validation

## Architecture

### Frontend (Next.js)
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS with shadcn/ui components
- **Components**: Modular React components for upload, results, and visualization
- **API Integration**: RESTful communication with backend services

### Backend (Python/FastAPI)
- **Framework**: FastAPI for high-performance API
- **AI Model**: TensorFlow/Keras CNN for TB detection
- **Explainability**: Grad-CAM for visual model interpretation
- **Image Processing**: OpenCV and PIL for preprocessing

## Project Structure

\`\`\`
tb-detection-app/
├── app/                          # Next.js app directory
│   ├── api/predict/route.ts      # API endpoint for predictions
│   ├── layout.tsx                # Root layout
│   ├── page.tsx                  # Main page
│   └── globals.css               # Global styles
├── components/                   # React components
│   ├── upload-form.tsx           # File upload component
│   ├── result-card.tsx           # Results display
│   ├── heatmap-display.tsx       # Grad-CAM visualization
│   └── ui/                       # shadcn/ui components
├── scripts/                      # Python backend
│   ├── tb_model.py               # AI model implementation
│   ├── train_model.py            # Model training script
│   ├── api_server.py             # FastAPI server
│   └── requirements.txt          # Python dependencies
└── README.md                     # This file
\`\`\`

## Setup Instructions

### Frontend Setup (Next.js)

1. **Install Dependencies**
   \`\`\`bash
   npm install
   \`\`\`

2. **Run Development Server**
   \`\`\`bash
   npm run dev
   \`\`\`

3. **Access Application**
   - Open http://localhost:3000 in your browser

### Backend Setup (Python/FastAPI)

1. **Create Virtual Environment**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

2. **Install Dependencies**
   \`\`\`bash
   cd scripts
   pip install -r requirements.txt
   \`\`\`

3. **Train Model (Optional)**
   \`\`\`bash
   python train_model.py
   \`\`\`

4. **Start API Server**
   \`\`\`bash
   python api_server.py
   \`\`\`

5. **API Access**
   - API runs on http://localhost:8000
   - Documentation: http://localhost:8000/docs

## Usage

### For Development

1. **Start Backend Server**
   \`\`\`bash
   cd scripts
   python api_server.py
   \`\`\`

2. **Start Frontend Server**
   \`\`\`bash
   npm run dev
   \`\`\`

3. **Upload X-ray Images**
   - Navigate to http://localhost:3000
   - Upload chest X-ray images via drag-and-drop or file browser
   - Click "Predict" to get AI analysis
   - View results with confidence scores and heatmap visualization

### For Production

1. **Build Frontend**
   \`\`\`bash
   npm run build
   npm start
   \`\`\`

2. **Deploy Backend**
   - Use Docker or cloud services (AWS, GCP, Azure)
   - Configure CORS for your production domain
   - Set up proper model storage and loading

## API Endpoints

### POST /api/predict
Upload chest X-ray image for TB detection.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file

**Response:**
\`\`\`json
{
  "result": "TB" | "Normal",
  "confidence": 0.95,
  "heatmap": "base64_encoded_heatmap_image"
}
\`\`\`

### GET /model/info
Get information about the loaded AI model.

**Response:**
\`\`\`json
{
  "input_size": [224, 224],
  "classes": ["Normal", "TB"],
  "model_loaded": true
}
\`\`\`

## Model Details

### Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224x3 (RGB images)
- **Classes**: Binary classification (Normal, TB)
- **Framework**: TensorFlow/Keras

### Training Data
- **Format**: Chest X-ray images in standard medical imaging formats
- **Preprocessing**: Resize, normalize, data augmentation
- **Validation**: Train/validation split with cross-validation

### Explainability
- **Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Purpose**: Visual explanation of model decisions
- **Output**: Heatmap overlay showing attention regions

## Important Notes

### Medical Disclaimer
⚠️ **This application is for educational and research purposes only.**
- Not intended for clinical diagnosis
- Always consult healthcare professionals for medical decisions
- AI predictions should be validated by medical experts

### Data Privacy
- Images are processed locally and not stored permanently
- Implement proper data handling for production use
- Follow HIPAA and other medical data regulations

### Model Performance
- Current model is a demonstration version
- For production use, train with large, validated medical datasets
- Regular model updates and validation required

## Development

### Adding New Features
1. **Frontend**: Add components in `components/` directory
2. **Backend**: Extend API in `scripts/api_server.py`
3. **Model**: Modify `scripts/tb_model.py` for model changes

### Testing
\`\`\`bash
# Frontend tests
npm test

# Backend tests
cd scripts
python -m pytest
\`\`\`

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is for educational purposes. Please ensure compliance with medical software regulations for any clinical use.

## Support

For technical issues or questions:
- Check the documentation
- Review API endpoints at http://localhost:8000/docs
- Ensure all dependencies are properly installed
\`\`\`

\`\`\`json file="" isHidden
