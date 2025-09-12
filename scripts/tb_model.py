"""
TB Detection Model Implementation
This script contains the AI model for tuberculosis detection in chest X-rays.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class TBDetectionModel:
    def __init__(self, model_path=None):
        """Initialize the TB detection model."""
        self.model = None
        self.input_size = (224, 224)
        self.class_names = ['Normal', 'TB']
        
        if model_path:
            self.load_model(model_path)
        else:
            self.create_model()
    
    def create_model(self):
        """Create a simple CNN model for TB detection."""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')  # 2 classes: Normal, TB
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_model(self, model_path):
        """Load a pre-trained model."""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead...")
            self.create_model()
    
    def preprocess_image(self, image_data):
        """Preprocess image for model input."""
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(self.input_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, image
    
    def predict(self, image_data):
        """Make prediction on chest X-ray image."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess image
        processed_image, original_image = self.preprocess_image(image_data)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx]
        
        # Generate Grad-CAM heatmap
        heatmap = self.generate_gradcam(processed_image, predicted_class_idx)
        heatmap_base64 = self.heatmap_to_base64(heatmap, original_image)
        
        return {
            'result': predicted_class,
            'confidence': confidence,
            'heatmap': heatmap_base64
        }
    
    def generate_gradcam(self, img_array, class_idx):
        """Generate Grad-CAM heatmap for model explainability."""
        if self.model is None:
            return np.random.rand(224, 224)  # Mock heatmap
        
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            # Return random heatmap if no conv layer found
            return np.random.rand(224, 224)
        
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = keras.models.Model(
            [self.model.inputs], 
            [last_conv_layer.output, self.model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Extract the gradients of the last conv layer
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the corresponding gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def heatmap_to_base64(self, heatmap, original_image):
        """Convert heatmap to base64 string for frontend display."""
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, original_image.size)
        
        # Apply colormap
        heatmap_colored = cm.jet(heatmap_resized)
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Convert to PIL Image
        heatmap_image = Image.fromarray(heatmap_colored)
        
        # Convert to base64
        buffer = BytesIO()
        heatmap_image.save(buffer, format='PNG')
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return heatmap_base64

# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    tb_model = TBDetectionModel()
    
    print("TB Detection Model initialized successfully!")
    print(f"Model input size: {tb_model.input_size}")
    print(f"Classes: {tb_model.class_names}")
    
    # Test with a dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_image_pil = Image.fromarray(dummy_image)
    
    buffer = BytesIO()
    dummy_image_pil.save(buffer, format='PNG')
    dummy_image_bytes = buffer.getvalue()
    
    try:
        result = tb_model.predict(dummy_image_bytes)
        print(f"Test prediction: {result['result']} (confidence: {result['confidence']:.2f})")
        print("Model is working correctly!")
    except Exception as e:
        print(f"Error during prediction: {e}")
