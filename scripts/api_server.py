"""
FastAPI server for TB detection model.
This provides the backend API that the Next.js frontend calls.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tb_model import TBDetectionModel
import io

# Initialize FastAPI app
app = FastAPI(title="TB Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the TB detection model
tb_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global tb_model
    try:
        # Try to load pre-trained model, fallback to new model
        tb_model = TBDetectionModel("tb_model.h5")
        print("TB detection model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        tb_model = TBDetectionModel()
        print("Created new TB detection model")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "TB Detection API is running", "status": "healthy"}

@app.post("/predict")
async def predict_tb(file: UploadFile = File(...)):
    """
    Predict tuberculosis from chest X-ray image.
    
    Args:
        file: Uploaded chest X-ray image file
        
    Returns:
        JSON response with prediction result, confidence, and heatmap
    """
    if tb_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Make prediction
        result = tb_model.predict(image_data)
        
        return {
            "result": result["result"],
            "confidence": result["confidence"],
            "heatmap": result["heatmap"],
            "filename": file.filename
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if tb_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "input_size": tb_model.input_size,
        "classes": tb_model.class_names,
        "model_loaded": tb_model.model is not None
    }

if __name__ == "__main__":
    print("Starting TB Detection API server...")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
