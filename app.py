from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO
import numpy as np
import cv2
import supervision as sv
import io
from PIL import Image

app = Flask(__name__)

# Load models
keras_model = load_model('./model/model_leukovision.keras')
yolo_model = YOLO('./model/leukovision.pt')

# Mapping class index to class names for Keras model
class_labels = {
    0: 'Early_Malignant_early_Pre-B',
    2: 'Pre_Malignant_Pro-B',
    3: 'Pro_Malignant_Pro-B'
}

def preprocess_image(image_file):
    """Preprocess the image for Keras model prediction."""
    img_bytes = image_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_class(image_array):
    """Predict the class of the image using the Keras model."""
    predictions = keras_model.predict(image_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    predicted_class = class_labels.get(class_index, 'Unknown Class')
    return predicted_class, float(confidence)

def create_response(status, message, data=None):
    """Create a standardized JSON response."""
    response = {
        'status': status,
        'message': message,
    }
    if data:
        response['data'] = data
    return jsonify(response)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the uploaded image and return the predicted class."""
    if 'image' not in request.files:
        return create_response('error', 'No image file provided'), 400

    file = request.files['image']
    if file.filename == '':
        return create_response('error', 'No selected file'), 400

    try:
        image_array = preprocess_image(file)
        predicted_class, confidence = predict_class(image_array)
        return create_response('success', 'Model is analyze successfully', {
            'confidence': confidence,
            'predicted_class': predicted_class
        })
    except Exception as e:
        return create_response('error', str(e)), 400

@app.route('/detect', methods=['POST'])
def detect():
    """Detect objects in the uploaded image and return the annotated image."""
    if 'image' not in request.files:
        return create_response('error', 'No image file provided'), 400

    file = request.files['image']
    if file.filename == '':
        return create_response('error', 'No selected file'), 400

    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = yolo_model(image)
        detections = sv.Detections.from_ultralytics(results[0])

        # Annotate the image
        class_names = ['benign', 'early', 'pre', 'pro']
        labels = [f"{class_names[class_id]}: {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
        oriented_box_annotator = sv.OrientedBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = oriented_box_annotator.annotate(scene=image, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Save annotated image to buffer
        img_byte_arr = io.BytesIO()
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', annotated_frame_bgr)
        img_byte_arr.write(buffer)
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype='image/png', as_attachment=False)
    except Exception as e:
        return create_response('error', 'Invalid image file'), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
