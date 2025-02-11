from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__, template_folder="templates")  # Add template folder
CORS(app)

# Load model and scaler
with open('model.pkl', 'rb') as model_file:
    logistic = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Prediction function
def predict_anemia(gender, hemoglobin, mch, mchc, mcv):
    input_data = pd.DataFrame([[gender, hemoglobin, mch, mchc, mcv]], 
                              columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
    
    input_data_scaled = scaler.transform(input_data)
    prediction = logistic.predict(input_data_scaled)

    return "Anaemic" if prediction[0] == 1 else "Not Anaemic"

# Serve HTML webpage
@app.route('/')
def home():
    return render_template('index.html')  # Load your webpage

# API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
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

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
