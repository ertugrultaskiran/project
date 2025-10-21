"""
REST API for Model Inference
Flask ile model deployment - Production ready!
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('..')
from utils import basic_clean

app = Flask(__name__)

# Load models
print("Loading models...")
with open("../models/baseline_tfidf_logreg.pkl", "rb") as f:
    baseline_model = pickle.load(f)

lstm_model = load_model("../models/word2vec_lstm_model.h5")

with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
with open("../models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 80

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "ticket_classifier"}), 200

@app.route('/predict/baseline', methods=['POST'])
def predict_baseline():
    """Baseline model prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Predict
        prediction = baseline_model.predict([text])[0]
        probabilities = baseline_model.predict_proba([text])[0]
        
        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_predictions = [
            {
                "label": label_encoder.inverse_transform([idx])[0],
                "probability": float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            "prediction": prediction,
            "confidence": float(max(probabilities)),
            "top_3": top_3_predictions,
            "model": "baseline_tfidf_logreg"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/lstm', methods=['POST'])
def predict_lstm():
    """LSTM model prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Preprocess
        cleaned_text = basic_clean(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
        
        # Predict
        probabilities = lstm_model.predict(padded)[0]
        prediction_idx = probabilities.argmax()
        prediction = label_encoder.inverse_transform([prediction_idx])[0]
        
        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_predictions = [
            {
                "label": label_encoder.inverse_transform([idx])[0],
                "probability": float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            "prediction": prediction,
            "confidence": float(probabilities[prediction_idx]),
            "top_3": top_3_predictions,
            "model": "word2vec_lstm"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/ensemble', methods=['POST'])
def predict_ensemble():
    """Ensemble prediction (average of both models)"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Baseline prediction
        baseline_probs = baseline_model.predict_proba([text])[0]
        
        # LSTM prediction
        cleaned_text = basic_clean(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
        lstm_probs = lstm_model.predict(padded)[0]
        
        # Ensemble (average)
        ensemble_probs = (baseline_probs + lstm_probs) / 2
        prediction_idx = ensemble_probs.argmax()
        prediction = label_encoder.inverse_transform([prediction_idx])[0]
        
        # Get top 3
        top_3_indices = ensemble_probs.argsort()[-3:][::-1]
        top_3_predictions = [
            {
                "label": label_encoder.inverse_transform([idx])[0],
                "probability": float(ensemble_probs[idx])
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            "prediction": prediction,
            "confidence": float(ensemble_probs[prediction_idx]),
            "top_3": top_3_predictions,
            "model": "ensemble"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Ticket Classification API...")
    print("Endpoints:")
    print("  - GET  /health")
    print("  - POST /predict/baseline")
    print("  - POST /predict/lstm")
    print("  - POST /predict/ensemble")
    app.run(host='0.0.0.0', port=5000, debug=False)

