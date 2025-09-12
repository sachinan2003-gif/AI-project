"""
Training script for TB detection model.
This script demonstrates how to train the model with chest X-ray data.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tb_model import TBDetectionModel

def create_sample_data():
    """Create sample training data for demonstration."""
    # In a real scenario, you would load actual chest X-ray images
    # This creates dummy data for demonstration
    
    # Generate random images (224x224x3)
    num_samples = 1000
    X_train = np.random.rand(num_samples, 224, 224, 3)
    
    # Generate random labels (0: Normal, 1: TB)
    y_train = np.random.randint(0, 2, num_samples)
    
    # Convert to categorical
    y_train_categorical = keras.utils.to_categorical(y_train, 2)
    
    return X_train, y_train_categorical

def train_tb_model():
    """Train the TB detection model."""
    print("Starting TB detection model training...")
    
    # Create model
    tb_model = TBDetectionModel()
    
    # Create sample data (replace with real data loading)
    X_train, y_train = create_sample_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    # Prepare training and validation generators
    train_generator = datagen.flow(
        X_train, y_train,
        batch_size=32,
        subset='training'
    )
    
    validation_generator = datagen.flow(
        X_train, y_train,
        batch_size=32,
        subset='validation'
    )
    
    # Train the model
    history = tb_model.model.fit(
        train_generator,
        epochs=10,  # Reduced for demo
        validation_data=validation_generator,
        verbose=1
    )
    
    # Save the trained model
    model_path = 'tb_model.h5'
    tb_model.model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Print training results
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"Final training accuracy: {final_accuracy:.4f}")
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")
    
    return tb_model, history

def evaluate_model():
    """Evaluate the trained model."""
    print("Evaluating model performance...")
    
    # Load the trained model
    tb_model = TBDetectionModel('tb_model.h5')
    
    # Create test data
    X_test, y_test = create_sample_data()
    
    # Evaluate
    test_loss, test_accuracy = tb_model.model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return test_accuracy, test_loss

if __name__ == "__main__":
    print("TB Detection Model Training Script")
    print("=" * 40)
    
    # Train the model
    model, training_history = train_tb_model()
    
    print("\nTraining completed!")
    
    # Evaluate the model
    test_acc, test_loss = evaluate_model()
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Model ready for deployment!")
