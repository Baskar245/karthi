from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load pretrained model
MODEL_PATH = 'model/deepfake_model.h5'
model = load_model(MODEL_PATH)

# Detect deepfake function
def detect_deepfake(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0][0]
    print(f"Prediction: {prediction:.4f}")
    return "Deepfake" if prediction > 0.5 else "Real"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            result = detect_deepfake(path)
            image_path = path
    return render_template('index.html', result=result, image_path=image_path)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
