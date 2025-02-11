from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import os

app = Flask(__name__, template_folder="templates")  # Ensure template folder exists
CORS(app)

# ✅ Safely Load Model & Scaler
try:
    with open('model.pkl', 'rb') as model_file:
        logistic = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print("🚨 Model or Scaler file missing! Ensure 'model.pkl' & 'scaler.pkl' exist.")
    logistic, scaler = None, None

# 🔹 Prediction Function
def predict_anemia(gender, hemoglobin, mch, mchc, mcv):
    input_data = pd.DataFrame([[gender, hemoglobin, mch, mchc, mcv]], 
                              columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
    input_data_scaled = scaler.transform(input_data)
    prediction = logistic.predict(input_data_scaled)
    return "Anaemic" if prediction[0] == 1 else "Not Anaemic"

# 🔹 Serve HTML Webpage
@app.route('/')
def home():
    return render_template('index.html')

# 🔹 API Route for Predictions
@app.route('/predict', methods=['POST'])
def predict():
    if logistic is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    gender = data.get('Gender')
    hemoglobin = data.get('Hemoglobin')
    mch = data.get('MCH')
    mchc = data.get('MCHC')
    mcv = data.get('MCV')

    if None in [gender, hemoglobin, mch, mchc, mcv]:
        return jsonify({'error': 'Missing parameters'}), 400

    result = predict_anemia(gender, hemoglobin, mch, mchc, mcv)
    return jsonify({'Prediction': result})

# ✅ Run Flask App
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use Render's assigned port if available
    app.run(host='0.0.0.0', port=port)
