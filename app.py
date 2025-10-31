from flask import Flask, request, jsonify, render_template
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Disable TensorFlow optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("my_model.keras")

# Define your class names
class_name = ['Early Blight', 'Late Blight', 'Healthy']


def read_file_as_image(data) -> dict:
    image = np.array(Image.open(BytesIO(data)))
    img_batch = np.expand_dims(image, 0)

    predictions = model.predict(img_batch)
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    contents = file.read()
    result = read_file_as_image(contents)

    return render_template(
        'result.html',
        filename=file.filename,
        predicted_class=result['class'],
        confidence=round(result['confidence'] * 100, 2)
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
