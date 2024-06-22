from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import io

app = Flask(__name__)

# Path to the model file
model_path = 'model.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the pre-trained model
model = load_model(model_path)

# Ensure the model's input shape is correct
input_shape = model.input_shape[1:3]  # assuming the model has an input shape like (None, height, width, channels)

# Function to preprocess the image
def preprocess_image(image):
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(input_shape)  # Resize the image as required by the model
    image = np.array(image)
    
    # Add logging to check the shape of the image array
    print(f"Preprocessed image shape: {image.shape}")
    
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match the model input
    
    # Add logging to check the shape after expanding dimensions
    print(f"Shape after expanding dimensions: {image.shape}")
    
    return image

# Categories that the model predicts
categories = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
    'Normal'
]

# API endpoint for prediction
@app.route('/xray/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)[0]
        
        # Convert predictions to human-readable format
        results = {category: float(prediction) * 100 for category, prediction in zip(categories, predictions)}
        
        return jsonify(results)
    
    except Exception as e:
        # Add more detailed error information
        return jsonify({'error': str(e), 'message': 'An error occurred during prediction. Please check the server logs for more details.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
