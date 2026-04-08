# MediXplain AI 🚀

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.8-orange.svg)](https://scikit-learn.org/)

## 🌟 Overview
MediXplain AI is an **explainable AI web application** for **Diabetes Risk Prediction** using the PIMA Indians Diabetes Dataset.

**Key Features**:
- ML predictions (Logistic Regression)
- **SHAP explanations** – understand feature importance
- Interactive web UI
- Risk assessment with visual charts
- Production-ready Flask API

## 📋 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (data + train + serve)
```bash
python scripts/run.py
```

### 3. **One-Command Run** (Backend + Frontend)
```bash
python backend/app.py
```
**Now**: http://localhost:5000/ serves complete app (UI + API)!
- Backend: `http://localhost:5000`
- Enter patient features → **Predict** → See risk % + top features chart!

## 🏗️ Architecture
```
MediXplain AI
├── data/          # Dataset + preprocessing
├── models/        # Trained model + scaler + SHAP background
├── backend/       # Flask API (/predict)
├── frontend/      # HTML/JS UI + Chart.js
├── scripts/       # run.py pipeline
└── requirements.txt
```

**Flow**:
1. `data/download.py` → PIMA CSV
2. `data/preprocess.py` → Clean features
3. `models/train.py` → LogisticRegression + StandardScaler
4. `backend/app.py` → Predict + SHAP LinearExplainer
5. `frontend/index.html` → UI fetches /predict

## 🔧 API Endpoints
```
POST /predict
Content-Type: application/json

Body: {
  "pregnancies": 6,
  "glucose": 148,
  ...
}

Response: {
  "risk_level": "High",
  "risk_percentage": 90.4,
  "risk_class": "high-risk",
  "explanation": "...",
  "top_features": [{"feature": "glucose", "importance": 2.1}, ...]
}
```

## 🧪 Sample Test Data
```
Pregnancies: 6
Glucose: 148
Blood Pressure: 72
Skin Thickness: 35
Insulin: 0
BMI: 33.6
Diabetes Pedigree: 0.627
Age: 50
```
**Expected**: ~90% High Risk (tested!)

## 🔍 Troubleshooting
- **Prediction error?** Check terminal logs (`PREDICT INPUT/ERROR`)
- **Models missing?** Run `python scripts/run.py`
- **404 on /**: Normal (API-only, use frontend.html)
- **CORS?** `flask-cors` enabled

## 📈 Model Performance
- **Accuracy**: ~77% (from training)
- **SHAP**: Top features like Glucose, BMI drive predictions

## 🤝 Contributing
1. Fork & PR
2. Add features/tests
3. Update README

## 📄 License
MIT

**Built with ❤️ for explainable healthcare AI!**

