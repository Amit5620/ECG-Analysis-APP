from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Roboflow Models
rf = Roboflow(api_key="kqlgTsdyBapHPYnoxznG")
project1 = rf.workspace().project("ecg-classification-ygs4v")
model1 = project1.version(1).model

project2 = rf.workspace().project("ecg_detection")
model2 = project2.version(3).model

# Home Route
@app.route('/')
def home():
    return render_template('login.html')

# Login Route
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == "amit" and password == "12345":
        session['user'] = username
        return redirect(url_for('dashboard'))
    return render_template('login.html', error="Invalid Credentials")

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('index.html')
    return redirect(url_for('home'))

# Upload Page Route
@app.route('/upload')
def upload():
    if 'user' in session:
        return render_template('upload.html')
    return redirect(url_for('home'))

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Resize the uploaded image to a manageable size
    image = cv2.imread(file_path)
    image = cv2.resize(image, (500, 500))  # Resize to 500x500 pixels
    cv2.imwrite(file_path, image)
    
    # Collect Patient Details
    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    patient_gender = request.form.get('patient_gender')
    patient_id = request.form.get('patient_id')
    doctor_name = request.form.get('doctor_name')
    patient_symptoms = request.form.get('patient_symptoms')
    
    # Model Predictions
    result1 = model1.predict(file_path, confidence=40, overlap=30).json()
    result2 = model2.predict(file_path, confidence=40, overlap=30).json()
    
    def process_result(result, filename):
        predictions = result.get("predictions", [])

        # If no predictions are found, return a blank image
        if not predictions:
            print(f"No predictions found for {filename}")
            return filename, ["No abnormality detected"]

        xyxy, confidence, class_id, labels = [], [], [], []
        predicted_classes = []

        for pred in predictions:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            xyxy.append([x - w/2, y - h/2, x + w/2, y + h/2])
            confidence.append(pred["confidence"])
            class_id.append(pred["class_id"])
            labels.append(pred["class"])
            predicted_classes.append(pred["class"])

        # Convert lists to NumPy arrays
        xyxy = np.array(xyxy, dtype=np.float32) if xyxy else np.zeros((1, 4), dtype=np.float32)
        confidence = np.array(confidence, dtype=np.float32) if confidence else np.zeros((1,), dtype=np.float32)
        class_id = np.array(class_id, dtype=int) if class_id else np.zeros((1,), dtype=int)

        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        image = cv2.imread(file_path)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Annotate Image
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(output_path, annotated_image)

        return filename, predicted_classes

    # Process results for both models
    pred_img1, predicted_classes1 = process_result(result1, 'prediction1.jpg')
    pred_img2, predicted_classes2 = process_result(result2, 'prediction2.jpg')
    
    # Render the result template with all necessary data
    return render_template('result.html', original=file.filename, pred1=pred_img1, pred2=pred_img2,
                           patient_name=patient_name, patient_age=patient_age, 
                           patient_gender=patient_gender, patient_id=patient_id, doctor_name=doctor_name, patient_symptoms=patient_symptoms,
                           predicted_classes1=predicted_classes1, predicted_classes2=predicted_classes2)

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)