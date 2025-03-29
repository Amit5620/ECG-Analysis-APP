# ECG Analysis Web App

## Overview
This is a web-based **ECG Analysis Application** built with **Flask**. It allows users to **upload an ECG image**, enter patient details, and get predictions using **two AI models**. The models detect abnormalities in ECG images using **Roboflow and Supervision**.

## Features
✅ **User Authentication** – Login system for access control.  
✅ **ECG Image Upload** – Users can upload ECG images for analysis.  
✅ **Patient Details Form** – Collects patient name, age, gender, and symptoms.  
✅ **Dual Model Prediction** – Uses two AI models to analyze the ECG image.  
✅ **Annotated Result Images** – Displays **original ECG, prediction images, and detected abnormalities**.  
✅ **Download Option** – Users can download the prediction images.  
✅ **Responsive UI** – Built with **HTML, CSS, Bootstrap**.  
✅ **Deployment Ready** – Configured with **Gunicorn** for hosting on **Render**.  

## Tech Stack
- **Backend**: Flask, Gunicorn  
- **Frontend**: HTML, CSS, Bootstrap  
- **AI Models**: Roboflow, Supervision  
- **Deployment**: Render  

## Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ecg-analysis-app.git
cd ecg-analysis-app
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App
```bash
python your_script.py
```
**Access the App:**  
Visit `http://127.0.0.1:5000/` in your browser.

## Deployment on Render
### 1️⃣ Install Gunicorn
```bash
pip install gunicorn
```
### 2️⃣ Create a `Procfile`
```text
web: gunicorn -w 4 -b 0.0.0.0:5000 your_script:app
```
### 3️⃣ Push to GitHub and Deploy on Render
1. Create a **Render Web Service**
2. Select your **GitHub repo**
3. Set `Start Command`:  
   ```bash
   gunicorn your_script:app
   ```
4. Deploy 🚀

## Folder Structure
```
/ecg-analysis-app
│── static/
│   ├── uploads/  # Stores uploaded and processed images
│   ├── style.css  # Stylesheet
│── templates/
│   ├── login.html
│   ├── index.html
│   ├── upload.html
│   ├── result.html
│── your_script.py  # Main Flask app
│── requirements.txt
│── README.md
│── Procfile
```

## Future Enhancements
🔹 Add **Database Integration** (Store patient history).  
🔹 Improve **Model Performance** with better datasets.  
🔹 Add **Real-time ECG Signal Processing**.  

## License
This project is **open-source** under the MIT License.

---
💡 **Have ideas or improvements?** Feel free to contribute! 🤝
