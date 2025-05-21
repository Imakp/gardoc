import os
import numpy as np
import tensorflow as tf

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU name: {device.name}")
else:
    print("No GPU found. Running on CPU")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

# Configuration
MODEL_PATH = 'plant_classification_model.h5'
IMG_SIZE = 224

def load_class_mapping(file_path='class_mapping.txt'):
    class_mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            idx, name = line.strip().split(': ')
            class_mapping[int(idx)] = name
    return class_mapping

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img, img_array

def predict_image(model, img_path, class_mapping):
    # Load and preprocess the image
    original_img, processed_img = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get class name
    predicted_class = class_mapping[predicted_class_idx]
    
    # Parse plant name and condition
    parts = predicted_class.split('_')
    plant_name = parts[0]
    condition = parts[1]
    
    return {
        'plant_name': plant_name,
        'condition': condition,
        'full_class': predicted_class,
        'confidence': confidence,
        'original_img': original_img,
        'all_predictions': predictions[0]
    }

def display_prediction(result, class_mapping):
    # Display the image with prediction
    plt.figure(figsize=(10, 6))
    plt.imshow(result['original_img'])
    plt.title(f"Prediction: {result['plant_name'].capitalize()} - {result['condition'].capitalize()}\nConfidence: {result['confidence']:.2f}%")
    plt.axis('off')
    
    # Display top 3 predictions
    top_indices = np.argsort(result['all_predictions'])[-3:][::-1]
    top_predictions = [(class_mapping[i], result['all_predictions'][i] * 100) for i in top_indices]
    
    for i, (class_name, prob) in enumerate(top_predictions):
        parts = class_name.split('_')
        plant = parts[0].capitalize()
        condition = parts[1].capitalize()
        plt.text(10, 30 + i * 30, f"{i+1}. {plant} - {condition}: {prob:.2f}%", 
                 color='white', backgroundcolor='black', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the model
    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    # Load class mapping
    class_mapping = load_class_mapping()
    
    # Test with a single image
    test_image_path = input("Enter the path to the test image: ")
    
    if os.path.exists(test_image_path):
        result = predict_image(model, test_image_path, class_mapping)
        display_prediction(result, class_mapping)
        
        print(f"\nPrediction Details:")
        print(f"Plant: {result['plant_name'].capitalize()}")
        print(f"Condition: {result['condition'].capitalize()}")
        print(f"Confidence: {result['confidence']:.2f}%")
    else:
        print(f"Error: File {test_image_path} does not exist.")

if __name__ == "__main__":
    main()