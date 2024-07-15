from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load(r"C:\Users\alani\Downloads\model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json 
    input_features = np.array(data['features']).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
