from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this line
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
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

# Define an API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from request
    
    # Extract values from JSON
    gender = data.get('Gender')
    hemoglobin = data.get('Hemoglobin')
    mch = data.get('MCH')
    mchc = data.get('MCHC')
    mcv = data.get('MCV')

    if None in [gender, hemoglobin, mch, mchc, mcv]:
        return jsonify({'error': 'Missing parameters'}), 400

    result = predict_anemia(gender, hemoglobin, mch, mchc, mcv)
    
    return jsonify({'Prediction': result})

# ✅ Add a default route for testing
@app.route('/')
def home():
    return "Anemia Prediction API is running!"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
