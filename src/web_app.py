"""
IT Ticket Classification - Web Dashboard
=========================================

Modern web application with chatbot interface for ticket classification.

Features:
- Real-time ticket classification
- Interactive dashboard
- Chatbot widget (bottom-right)
- Classification history
- Confidence scores visualization

Usage:
    python src/web_app.py

Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import pickle
import pandas as pd
from datetime import datetime
import uuid
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys
sys.path.append('..')
from utils import basic_clean
from conversational_assistant_v2 import ConversationalAssistantV2

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
CORS(app)

# Global storage for tickets (in production, use database)
tickets_history = []

# Session-based conversation assistants (each user gets their own)
session_assistants = {}

# Initialize conversational assistant V2.0
print("Initializing conversational assistant v2.0...")
try:
    conversational_assistant = ConversationalAssistantV2()
    print("✅ Conversational assistant v2.0 ready!")
    print("   Features: Sentiment analysis, Escalation logic, Context tracking")
except Exception as e:
    print(f"⚠️  Conversational assistant not available: {e}")
    conversational_assistant = None

# Load models at startup
print("Loading models...")
try:
    # Baseline model
    with open("../models/baseline_tfidf_logreg.pkl", "rb") as f:
        baseline_model = pickle.load(f)
    
    # LSTM model
    lstm_model = load_model("../models/word2vec_lstm_model.h5")
    
    with open("../models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    with open("../models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    # Try to load BERT model (MULTILINGUAL!)
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_tokenizer = BertTokenizer.from_pretrained("../models/bert_tokenizer")
        bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-multilingual-cased',
            num_labels=len(label_encoder.classes_)
        )
        bert_model.load_state_dict(torch.load("../models/bert_model.pt", map_location=device))
        bert_model.to(device)
        bert_model.eval()
        has_bert_model = True
        print(f"   - BERT (Multilingual): ✓ (Device: {device})")
    except Exception as e:
        bert_model = None
        bert_tokenizer = None
        has_bert_model = False
        print(f"   - BERT (Multilingual): ✗ ({str(e)[:50]}...)")
    
    # Try to load custom attention model if available
    try:
        custom_attention_model = load_model("../models/custom_attention_lstm.h5",
                                           custom_objects={'CustomAttentionLayer': None})
        has_custom_model = True
    except:
        custom_attention_model = None
        has_custom_model = False
    
    print("✅ Models loaded successfully!")
    print(f"   - Baseline: ✓")
    print(f"   - LSTM: ✓")
    print(f"   - BERT: {'✓' if has_bert_model else '✗ (not trained yet)'}")
    print(f"   - Custom Attention: {'✓' if has_custom_model else '✗ (not trained yet)'}")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please train models first!")
    baseline_model = None
    lstm_model = None


MAX_LEN = 80

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def classify_ticket():
    """
    Classify a ticket using all available models
    
    Request JSON:
        {
            "text": "ticket description",
            "model": "ensemble" | "baseline" | "lstm" | "custom"
        }
    
    Response JSON:
        {
            "id": "unique-id",
            "text": "ticket description",
            "category": "predicted category",
            "confidence": 0.95,
            "model_used": "ensemble",
            "all_predictions": {...},
            "timestamp": "2025-11-16 12:30:45"
        }
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_choice = data.get('model', 'ensemble')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Generate unique ID
        ticket_id = str(uuid.uuid4())[:8]
        
        # Get predictions from all models
        predictions = {}
        
        # 1. Baseline prediction
        if baseline_model:
            baseline_probs = baseline_model.predict_proba([text])[0]
            baseline_pred = baseline_model.predict([text])[0]
            baseline_conf = float(max(baseline_probs))
            predictions['baseline'] = {
                'category': baseline_pred,
                'confidence': baseline_conf,
                'probabilities': {label_encoder.classes_[i]: float(baseline_probs[i]) 
                                 for i in range(len(baseline_probs))}
            }
        
        # 2. LSTM prediction
        if lstm_model:
            cleaned_text = basic_clean(text)
            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
            lstm_probs = lstm_model.predict(padded, verbose=0)[0]
            lstm_pred_idx = lstm_probs.argmax()
            lstm_pred = label_encoder.inverse_transform([lstm_pred_idx])[0]
            lstm_conf = float(lstm_probs[lstm_pred_idx])
            predictions['lstm'] = {
                'category': lstm_pred,
                'confidence': lstm_conf,
                'probabilities': {label_encoder.inverse_transform([i])[0]: float(lstm_probs[i]) 
                                 for i in range(len(lstm_probs))}
            }
        
        # 3. Ensemble prediction
        if baseline_model and lstm_model:
            ensemble_probs = (baseline_probs + lstm_probs) / 2
            ensemble_pred_idx = ensemble_probs.argmax()
            ensemble_pred = label_encoder.inverse_transform([ensemble_pred_idx])[0]
            ensemble_conf = float(ensemble_probs[ensemble_pred_idx])
            predictions['ensemble'] = {
                'category': ensemble_pred,
                'confidence': ensemble_conf,
                'probabilities': {label_encoder.inverse_transform([i])[0]: float(ensemble_probs[i]) 
                                 for i in range(len(ensemble_probs))}
            }
        
        # 4. Custom attention model (if available)
        if has_custom_model and custom_attention_model:
            custom_probs = custom_attention_model.predict(padded, verbose=0)[0]
            custom_pred_idx = custom_probs.argmax()
            custom_pred = label_encoder.inverse_transform([custom_pred_idx])[0]
            custom_conf = float(custom_probs[custom_pred_idx])
            predictions['custom'] = {
                'category': custom_pred,
                'confidence': custom_conf,
                'probabilities': {label_encoder.inverse_transform([i])[0]: float(custom_probs[i]) 
                                 for i in range(len(custom_probs))}
            }
        
        # Select prediction based on model choice
        if model_choice in predictions:
            selected = predictions[model_choice]
        else:
            selected = predictions.get('ensemble', predictions.get('lstm', predictions.get('baseline')))
        
        # Create ticket object
        ticket = {
            'id': ticket_id,
            'text': text,
            'category': selected['category'],
            'confidence': selected['confidence'],
            'model_used': model_choice,
            'all_predictions': predictions,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'classified'
        }
        
        # Store in history
        tickets_history.insert(0, ticket)  # Insert at beginning
        if len(tickets_history) > 100:  # Keep last 100
            tickets_history.pop()
        
        return jsonify(ticket), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Get all classified tickets"""
    return jsonify({
        'tickets': tickets_history,
        'total': len(tickets_history)
    }), 200


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics"""
    if not tickets_history:
        return jsonify({
            'total_tickets': 0,
            'categories': {},
            'average_confidence': 0
        }), 200
    
    categories = {}
    total_confidence = 0
    
    for ticket in tickets_history:
        cat = ticket['category']
        categories[cat] = categories.get(cat, 0) + 1
        total_confidence += ticket['confidence']
    
    return jsonify({
        'total_tickets': len(tickets_history),
        'categories': categories,
        'average_confidence': total_confidence / len(tickets_history),
        'models_available': {
            'baseline': baseline_model is not None,
            'lstm': lstm_model is not None,
            'ensemble': baseline_model is not None and lstm_model is not None,
            'custom': has_custom_model
        }
    }), 200


@app.route('/api/chat', methods=['POST'])
def chat_with_assistant():
    """
    Conversational response endpoint
    
    Request JSON:
        {
            "text": "ticket description",
            "model": "ensemble" | "baseline" | "lstm"
        }
    
    Response JSON:
        {
            "response": "conversational response text",
            "category": "predicted category",
            "confidence": 0.95,
            "intent": "urgent" | "question" | "standard"
        }
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_choice = data.get('model', 'ensemble')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get classification (PRIORITIZE BERT for best results!)
        if model_choice == 'bert' and has_bert_model:
            # BERT prediction (supports Turkish + English!)
            inputs = bert_tokenizer(
                text,
                return_tensors='pt',
                max_length=128,
                padding='max_length',
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                pred_idx = probs.argmax().item()
                category = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])
        
        elif model_choice == 'lstm' and lstm_model:
            cleaned_text = basic_clean(text)
            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
            probs = lstm_model.predict(padded, verbose=0)[0]
            pred_idx = probs.argmax()
            category = label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
        elif model_choice == 'baseline' and baseline_model:
            category = baseline_model.predict([text])[0]
            probs = baseline_model.predict_proba([text])[0]
            confidence = float(max(probs))
        else:  # ensemble (default)
            if baseline_model and lstm_model:
                baseline_probs = baseline_model.predict_proba([text])[0]
                cleaned_text = basic_clean(text)
                sequence = tokenizer.texts_to_sequences([cleaned_text])
                padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
                lstm_probs = lstm_model.predict(padded, verbose=0)[0]
                ensemble_probs = (baseline_probs + lstm_probs) / 2
                pred_idx = ensemble_probs.argmax()
                category = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(ensemble_probs[pred_idx])
            else:
                return jsonify({"error": "Models not available"}), 503
        
        # Get or create session-based assistant (context tracking per user)
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        # Get user's assistant (maintains conversation context)
        if session_id not in session_assistants and conversational_assistant:
            session_assistants[session_id] = ConversationalAssistantV2()
        
        user_assistant = session_assistants.get(session_id)
        
        # Generate ADVANCED conversational response (v2.0)
        if user_assistant:
            result = user_assistant.generate_response(
                ticket_text=text,
                category=category,
                confidence=confidence
            )
            
            # v2.0 returns structured response
            conversational_response = result['response']
            intent = result['intent']
            sentiment = result['sentiment']
            should_escalate = result['should_escalate']
            escalation_reason = result.get('escalation_reason', '')
            follow_up_questions = result.get('follow_up_questions', [])
        else:
            # Fallback if assistant not available
            conversational_response = f"Kategori: {category} (Güven: %{confidence*100:.1f})"
            intent = "standard"
            sentiment = "neutral"
            should_escalate = False
            escalation_reason = ""
            follow_up_questions = []
        
        # CREATE TICKET OBJECT and SAVE TO HISTORY
        ticket_id = str(uuid.uuid4())[:8]
        ticket = {
            'id': ticket_id,
            'text': text,
            'category': category,
            'confidence': confidence,
            'model_used': model_choice,
            'intent': intent,
            'sentiment': sentiment,
            'escalated': should_escalate,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'escalated' if should_escalate else 'resolved'
        }
        
        # Add to history
        tickets_history.insert(0, ticket)
        if len(tickets_history) > 100:
            tickets_history.pop()
        
        return jsonify({
            "id": ticket_id,
            "response": conversational_response,
            "category": category,
            "confidence": confidence,
            "intent": intent,
            "sentiment": sentiment,
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
            "follow_up_questions": follow_up_questions,
            "model_used": model_choice,
            "timestamp": ticket['timestamp']
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset conversation context for current user"""
    session_id = session.get('session_id')
    if session_id and session_id in session_assistants:
        session_assistants[session_id].reset_conversation()
        return jsonify({
            "status": "success",
            "message": "Conversation context reset"
        }), 200
    return jsonify({"status": "no_session"}), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": baseline_model is not None and lstm_model is not None,
        "conversational_assistant": conversational_assistant is not None,
        "active_sessions": len(session_assistants),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }), 200


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 IT TICKET CLASSIFICATION - WEB DASHBOARD")
    print("=" * 70)
    print("\n📊 Starting web application...")
    print("\n🌐 Open your browser and go to:")
    print("   http://127.0.0.1:5000")
    print("   or")
    print("   http://localhost:5000")
    print("\n💡 Features:")
    print("   ✓ Real-time ticket classification")
    print("   ✓ Interactive dashboard")
    print("   ✓ Chatbot interface (bottom-right)")
    print("   ✓ Multiple model support")
    print("   ✓ Classification history")
    print("\n⏹  Press CTRL+C to stop")
    print("=" * 70 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)

