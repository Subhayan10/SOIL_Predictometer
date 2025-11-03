from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os

# Initialize Flask app
app = Flask(__name__)

# Load model
MODEL_PATH = 'Soil_MobileNetV2.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define your classes (same order as during training)
class_names = ['Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil', 'Mountain_Soil', 'Red_Soil', 'Yellow_Soil']

@app.route('/')
def home():
    return jsonify({"message": "Soil classification API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Please send an image file with key "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Read image safely
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))

        # Convert to model input format
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        preds = model.predict(img_array)
        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]
        confidence = float(np.max(preds) * 100)

        return jsonify({
            'prediction': pred_class,
            'confidence': confidence
        })

    except UnidentifiedImageError:
        return jsonify({'error': 'Uploaded file is not a valid image.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)