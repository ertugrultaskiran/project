"""
IT Ticket Classification - Web Dashboard (DEMO MODE)
====================================================

DEMO MODE: Works without trained models!
Shows UI, but classification will return dummy data.

Perfect for demonstrating the interface to professors!
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime
import uuid
import random

app = Flask(__name__)
app.secret_key = 'demo-secret-key'
CORS(app)

# Demo mode - no models needed!
tickets_history = []

# Demo categories
CATEGORIES = [
    'Access', 'Administrative rights', 'HR Support', 
    'Hardware', 'Internal Project', 'Miscellaneous', 
    'Purchase', 'Storage'
]

print("=" * 70)
print("🎭 DEMO MODE - No models required!")
print("=" * 70)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def classify_ticket():
    """
    DEMO: Classify a ticket (returns dummy prediction)
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_choice = data.get('model', 'ensemble')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Generate unique ID
        ticket_id = str(uuid.uuid4())[:8]
        
        # DEMO: Smart dummy prediction based on keywords
        category = predict_category_demo(text)
        confidence = random.uniform(0.85, 0.95)
        
        # Generate dummy probabilities
        probs = {}
        remaining = 1.0 - confidence
        for cat in CATEGORIES:
            if cat == category:
                probs[cat] = confidence
            else:
                probs[cat] = remaining / (len(CATEGORIES) - 1) * random.uniform(0.5, 1.5)
        
        # Normalize
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        # Create predictions object
        predictions = {
            model_choice: {
                'category': category,
                'confidence': probs[category],
                'probabilities': probs
            }
        }
        
        # Create ticket object
        ticket = {
            'id': ticket_id,
            'text': text,
            'category': category,
            'confidence': probs[category],
            'model_used': model_choice,
            'all_predictions': predictions,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'classified (demo)'
        }
        
        # Store in history
        tickets_history.insert(0, ticket)
        if len(tickets_history) > 100:
            tickets_history.pop()
        
        return jsonify(ticket), 200
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


def predict_category_demo(text):
    """
    DEMO: Simple keyword-based prediction
    """
    text_lower = text.lower()
    
    # Simple keyword matching
    if any(word in text_lower for word in ['access', 'login', 'password', 'permission']):
        return 'Access'
    elif any(word in text_lower for word in ['laptop', 'computer', 'screen', 'keyboard', 'mouse', 'hardware']):
        return 'Hardware'
    elif any(word in text_lower for word in ['leave', 'vacation', 'hr', 'payroll', 'timesheet']):
        return 'HR Support'
    elif any(word in text_lower for word in ['storage', 'disk', 'space', 'drive']):
        return 'Storage'
    elif any(word in text_lower for word in ['purchase', 'buy', 'order']):
        return 'Purchase'
    elif any(word in text_lower for word in ['project', 'development']):
        return 'Internal Project'
    elif any(word in text_lower for word in ['admin', 'administrator', 'rights']):
        return 'Administrative rights'
    else:
        return 'Miscellaneous'


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
            'average_confidence': 0,
            'models_available': {
                'baseline': True,
                'lstm': True,
                'ensemble': True,
                'custom': False
            }
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
            'baseline': True,
            'lstm': True,
            'ensemble': True,
            'custom': False
        }
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "mode": "DEMO",
        "models_loaded": True,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }), 200


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎭 IT TICKET CLASSIFICATION - DEMO MODE")
    print("=" * 70)
    print("\n⚠️  DEMO MODE: No trained models required!")
    print("   Classification uses simple keyword matching.")
    print("   Perfect for demonstrating the UI!")
    print("\n🌐 Open your browser and go to:")
    print("   👉 http://127.0.0.1:5000")
    print("\n💡 Features:")
    print("   ✓ Full UI working")
    print("   ✓ Interactive dashboard")
    print("   ✓ Chatbot interface")
    print("   ✓ Real-time updates")
    print("   ✓ No models needed!")
    print("\n⏹  Press CTRL+C to stop")
    print("=" * 70 + "\n")
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

