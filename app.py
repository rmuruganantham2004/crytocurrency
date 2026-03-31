import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.pipeline import DataLoader, FeatureEngineer, PreProcessor, CryptoLSTM
from datetime import datetime, timedelta

app = Flask(__name__)

# Load models and preprocessing objects
SCALER_PATH = 'models/scaler.pkl'
MODEL_PATH = 'models/crypto_lstm.pt'
FEATURES_PATH = 'models/features.pkl'

try:
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    # The LSTM takes sequence_length=60 and features_count=len(features)
    model = CryptoLSTM.load(MODEL_PATH, sequence_length=60, features_count=len(features))
    preprocessor = PreProcessor(sequence_length=60)
    preprocessor.scaler = scaler
    preprocessor.features = features
    preprocessor.target_idx = features.index('Close')
except Exception as e:
    print(f"Error loading models: {e}")
    scaler = None
    model = None

@app.route('/')
def home():
    # Fetch recent data to display on the dashboard
    loader = DataLoader(ticker="BTC-USD", start_date=(datetime.today() - timedelta(days=100)).strftime('%Y-%m-%d'))
    df = loader.fetch_data()
    dates = df.index.strftime('%Y-%m-%d').tolist()
    closes = df['Close'].tolist()
    
    return render_template('index.html', dates=dates, closes=closes)

@app.route('/predict', methods=['GET'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
        
    # Fetch the last 150 days of data to compute features properly
    loader = DataLoader(ticker="BTC-USD", start_date=(datetime.today() - timedelta(days=150)).strftime('%Y-%m-%d'))
    df = loader.fetch_data()
    
    # Feature engineering
    df = FeatureEngineer.add_technical_indicators(df)
    df = df.dropna()
    
    if len(df) < 60:
        return jsonify({"error": "Not enough data after dropping NA to form a sequence."}), 500
        
    df = df[features]
    last_60 = df.iloc[-60:]
    
    # Scale Data
    scaled_data = scaler.transform(last_60)
    X_pred = np.array([scaled_data])
    
    # Predict
    scaled_pred = model.predict(X_pred)
    
    # Inverse transform
    dummy = np.zeros((1, len(features)))
    dummy[0, preprocessor.target_idx] = scaled_pred[0, 0]
    predicted_price = scaler.inverse_transform(dummy)[0, preprocessor.target_idx]
    
    current_price = last_60['Close'].iloc[-1]
    trend = "UP" if predicted_price > current_price else "DOWN"
    diff_percent = abs(predicted_price - current_price) / current_price * 100
    
    predicted_date = (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
    
    return jsonify({
        "current_price": round(float(current_price), 2),
        "predicted_price": round(float(predicted_price), 2),
        "predicted_date": predicted_date,
        "trend": trend,
        "diff_percent": round(float(diff_percent), 2)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
