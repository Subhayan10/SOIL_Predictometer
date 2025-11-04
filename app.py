from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load model
MODEL_PATH = "model/Soil_MobileNetV2.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define soil-to-plant mapping (you can edit or extend this)
plant_recommendations = {
     
    "Alluvial_Soil": ["Rice", "Wheat", "Sugarcane", "Maize", "Pulses", "Jute"],
    "Arid_Soil": ["Millets", "Barley", "Cotton", "Dates", "Cactus", "Mustard"],
    "Black_Soil": ["Cotton", "Soybean", "Sunflower", "Tobacco", "Citrus Fruits"],
    "Laterite_Soil": ["Tea", "Coffee", "Cashew", "Coconut", "Rubber"],
    "Mountain_Soil": ["Apples", "Barley", "Tea", "Maize", "Fruits and Vegetables"],
    "Red_Soil": ["Groundnut", "Potato", "Millets", "Pulses", "Cotton"],
    "Yellow_Soil": ["Maize", "Peas", "Groundnut", "Paddy", "Vegetables"]

}

# Classes (should match your model’s training classes)
class_labels = list(plant_recommendations.keys())

@app.route('/')
def home():
    return "🌱 Soil Classification API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_index]

        # Recommended plants
        plants = plant_recommendations.get(predicted_class, [])

        return jsonify({
            'predicted_soil': predicted_class,
            'recommended_plants': plants
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
