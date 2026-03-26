from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "plant_model.h5"),compile=False
)

# Load class names
dataset_path = os.path.join(os.path.dirname(__file__), "../dataset/train")
class_names = sorted([
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
])

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # Upload folder path (static)
    upload_folder = os.path.join(os.path.dirname(__file__), "static")

    # Create static folder if not exists
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, "uploaded.jpg")
    file.save(file_path)

    # Image processing
    img = Image.open(file_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]
    result = result.replace("_", " ")
    confidence = round(float(np.max(prediction) * 100), 2)

    return render_template(
        'index.html',
        prediction=result,
        confidence=confidence,
        image_path="uploaded.jpg"
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
