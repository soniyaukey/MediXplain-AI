from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import shap
import os

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app, origins=['http://localhost:5000', 'file://*'])  # Enable CORS

# Load the trained model and preprocessing objects
try:
    model = joblib.load('models/diabetes_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    background = joblib.load('models/background.pkl')
    
    # Initialize SHAP explainer for linear model
    explainer = shap.LinearExplainer(model, background)
except Exception as e:
    print(f"Warning: Could not load model files. Please run training first. Error: {e}")

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"PREDICT INPUT: {data}")

        # Convert input data to array in correct order
        input_data = []
        for feature in feature_names:
            input_data.append(data.get(feature, 0))

        input_array = np.array([input_data])

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Predict probability
        prob = model.predict_proba(input_scaled)[0][1]
        risk_percentage = round(float(prob * 100), 1)
        print(f"PREDICT RISK: {risk_percentage}%")

        # Determine risk class and level
        if risk_percentage >= 70:
            risk_level = "High"
            risk_class = "high-risk"
        elif risk_percentage >= 30:
            risk_level = "Medium"
            risk_class = "medium-risk"
        else:
            risk_level = "Low"
            risk_class = "low-risk"

        # Get SHAP values for explanation
        shap_values = explainer.shap_values(input_scaled)

        # Prepare feature importance for frontend
        # For LinearExplainer, shap_values is an array of shape (1, n_features)
        importance = shap_values[0]

        top_features = []
        for i, name in enumerate(feature_names):
            top_features.append({
                "feature": name,
                "importance": float(importance[i])
            })

        # Sort by absolute importance and take top 5
        top_features.sort(key=lambda x: abs(x['importance']), reverse=True)
        top_features = top_features[:5]

        # Generate simple explanation text
        most_important = top_features[0]['feature'].replace('_', ' ')
        explanation = f"The most significant factor contributing to this prediction is {most_important}."

        return jsonify({
            "risk_level": risk_level,
            "risk_percentage": risk_percentage,
            "risk_class": risk_class,
            "explanation": explanation,
            "top_features": top_features
        })
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
