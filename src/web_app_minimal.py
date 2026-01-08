"""
Minimal Web App - Analytics Dashboard Test
==========================================
Quick test version without heavy model loading
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from collections import defaultdict
import random
from ai_features import IntelligentTicketAssistant

app = Flask(__name__)
app.secret_key = 'test-secret-key'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
CORS(app)

# Mock tickets data for testing
tickets_history = []

# Generate some mock data for testing
def generate_mock_data():
    global tickets_history
    categories = ['Access', 'Hardware', 'HR Support', 'Network', 'Purchase', 'Software', 'Storage', 'Administrative Rights']
    models = ['bert', 'ensemble', 'lstm', 'baseline']
    
    # Generate tickets for last 7 days
    for i in range(50):
        days_ago = random.randint(0, 6)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))
        
        ticket = {
            'id': f'mock-{i:03d}',
            'text': f'Sample ticket {i} - testing analytics',
            'category': random.choice(categories),
            'confidence': random.uniform(0.75, 0.98),
            'model_used': random.choice(models),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'classified'
        }
        tickets_history.append(ticket)

# Generate mock data on startup
generate_mock_data()
print("[OK] Mock data generated!")

# Initialize AI Assistant
ai_assistant = IntelligentTicketAssistant()
print("[OK] AI Assistant initialized!")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/manifest.json')
def manifest():
    """Serve PWA manifest"""
    return app.send_static_file('manifest.json')


@app.route('/service-worker.js')
def service_worker():
    """Serve service worker"""
    return app.send_static_file('service-worker.js')


@app.route('/test')
def test_chatbot():
    """Test chatbot page"""
    return render_template('test_chatbot.html')


@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Get all classified tickets"""
    return jsonify({
        'tickets': tickets_history[:20],  # Return last 20
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
            'baseline': True,
            'lstm': True,
            'ensemble': True,
            'custom': False
        }
    }), 200


@app.route('/api/analytics/trends', methods=['GET'])
def get_trends():
    """Get ticket trends over time"""
    try:
        days = int(request.args.get('days', 7))
        
        # Group tickets by date
        daily_counts = defaultdict(int)
        daily_confidence = defaultdict(list)
        
        for ticket in tickets_history:
            try:
                ticket_date = datetime.strptime(ticket['timestamp'], '%Y-%m-%d %H:%M:%S').date()
                date_str = ticket_date.strftime('%Y-%m-%d')
                daily_counts[date_str] += 1
                daily_confidence[date_str].append(ticket['confidence'])
            except:
                continue
        
        # Generate last N days
        today = datetime.now().date()
        dates = []
        counts = []
        avg_confidences = []
        
        for i in range(days - 1, -1, -1):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            dates.append(date_str)
            counts.append(daily_counts.get(date_str, 0))
            
            if date_str in daily_confidence and daily_confidence[date_str]:
                avg_conf = sum(daily_confidence[date_str]) / len(daily_confidence[date_str])
                avg_confidences.append(round(avg_conf * 100, 1))
            else:
                avg_confidences.append(0)
        
        return jsonify({
            'dates': dates,
            'counts': counts,
            'avg_confidences': avg_confidences
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/categories', methods=['GET'])
def get_category_distribution():
    """Get category distribution for pie chart"""
    try:
        if not tickets_history:
            return jsonify({
                'categories': [],
                'counts': []
            }), 200
        
        categories = {}
        for ticket in tickets_history:
            cat = ticket['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        # Sort by count
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'categories': [cat for cat, _ in sorted_cats],
            'counts': [count for _, count in sorted_cats]
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/models', methods=['GET'])
def get_model_performance():
    """Get model performance comparison"""
    try:
        if not tickets_history:
            return jsonify({
                'models': [],
                'avg_confidences': [],
                'usage_counts': []
            }), 200
        
        model_stats = {}
        for ticket in tickets_history:
            model = ticket.get('model_used', 'unknown')
            if model not in model_stats:
                model_stats[model] = {'confidences': [], 'count': 0}
            
            model_stats[model]['confidences'].append(ticket['confidence'])
            model_stats[model]['count'] += 1
        
        # Calculate averages
        models = []
        avg_confidences = []
        usage_counts = []
        
        for model, stats in model_stats.items():
            models.append(model.upper())
            avg_conf = sum(stats['confidences']) / len(stats['confidences'])
            avg_confidences.append(round(avg_conf * 100, 1))
            usage_counts.append(stats['count'])
        
        return jsonify({
            'models': models,
            'avg_confidences': avg_confidences,
            'usage_counts': usage_counts
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/confidence-distribution', methods=['GET'])
def get_confidence_distribution():
    """Get confidence score distribution"""
    try:
        if not tickets_history:
            return jsonify({
                'ranges': [],
                'counts': []
            }), 200
        
        bins = {
            '0-50%': 0,
            '50-60%': 0,
            '60-70%': 0,
            '70-80%': 0,
            '80-90%': 0,
            '90-100%': 0
        }
        
        for ticket in tickets_history:
            conf = ticket['confidence'] * 100
            if conf < 50:
                bins['0-50%'] += 1
            elif conf < 60:
                bins['50-60%'] += 1
            elif conf < 70:
                bins['60-70%'] += 1
            elif conf < 80:
                bins['70-80%'] += 1
            elif conf < 90:
                bins['80-90%'] += 1
            else:
                bins['90-100%'] += 1
        
        return jsonify({
            'ranges': list(bins.keys()),
            'counts': list(bins.values())
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get comprehensive analytics summary"""
    try:
        if not tickets_history:
            return jsonify({
                'total_tickets': 0,
                'today_tickets': 0,
                'avg_confidence': 0,
                'top_category': 'N/A',
                'most_used_model': 'N/A',
                'growth_rate': 0
            }), 200
        
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        today_count = 0
        yesterday_count = 0
        categories = {}
        models = {}
        confidences = []
        
        for ticket in tickets_history:
            try:
                ticket_date = datetime.strptime(ticket['timestamp'], '%Y-%m-%d %H:%M:%S').date()
                if ticket_date == today:
                    today_count += 1
                elif ticket_date == yesterday:
                    yesterday_count += 1
            except:
                pass
            
            cat = ticket['category']
            categories[cat] = categories.get(cat, 0) + 1
            
            model = ticket.get('model_used', 'unknown')
            models[model] = models.get(model, 0) + 1
            
            confidences.append(ticket['confidence'])
        
        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else 'N/A'
        most_used_model = max(models.items(), key=lambda x: x[1])[0] if models else 'N/A'
        
        growth_rate = 0
        if yesterday_count > 0:
            growth_rate = round(((today_count - yesterday_count) / yesterday_count) * 100, 1)
        elif today_count > 0:
            growth_rate = 100
        
        return jsonify({
            'total_tickets': len(tickets_history),
            'today_tickets': today_count,
            'avg_confidence': round(sum(confidences) / len(confidences) * 100, 1) if confidences else 0,
            'top_category': top_category,
            'most_used_model': most_used_model.upper(),
            'growth_rate': growth_rate
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ai/analyze', methods=['POST'])
def ai_analyze():
    """
    AI-powered ticket analysis
    
    Request JSON:
        {
            "text": "ticket description",
            "category": "ticket category"
        }
    
    Response: Complete AI analysis
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        category = data.get('category', 'Misc')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Perform AI analysis
        result = ai_assistant.analyze_ticket(text, category, tickets_history)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/classify-with-ai', methods=['POST'])
def classify_with_ai():
    """
    Enhanced classification with AI features
    
    Combines classification with AI analysis
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Mock classification (random for demo)
        categories = ['Access', 'Hardware', 'HR Support', 'Network', 
                     'Purchase', 'Software', 'Storage', 'Administrative Rights']
        category = random.choice(categories)
        confidence = random.uniform(0.80, 0.98)
        model_used = random.choice(['bert', 'ensemble', 'lstm'])
        
        # AI Analysis
        ai_analysis = ai_assistant.analyze_ticket(text, category, tickets_history)
        
        # Create enhanced ticket
        ticket_id = f'ai-{len(tickets_history):03d}'
        ticket = {
            'id': ticket_id,
            'text': text,
            'category': category,
            'confidence': confidence,
            'model_used': model_used,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'escalated' if ai_analysis['priority']['priority'] == 'HIGH' else 'classified',
            # AI Features
            'sentiment': ai_analysis['sentiment']['sentiment'],
            'sentiment_score': ai_analysis['sentiment']['score'],
            'is_urgent': ai_analysis['sentiment']['is_urgent'],
            'priority': ai_analysis['priority']['priority'],
            'priority_score': ai_analysis['priority']['score'],
            'priority_reasons': ai_analysis['priority']['reasons'],
            'route_to': ai_analysis['routing']['primary_team'],
            'sla_hours': ai_analysis['sla']['hours'],
            'sla_description': ai_analysis['sla']['description'],
            'sla_deadline': ai_analysis['sla']['deadline'],
            'similar_tickets': ai_analysis['similar_tickets']
        }
        
        # Add to history
        tickets_history.insert(0, ticket)
        if len(tickets_history) > 100:
            tickets_history.pop()
        
        return jsonify(ticket), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "mode": "minimal_test",
        "mock_data": True,
        "ai_features": True,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }), 200


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("\n" + "=" * 70)
    print("IT TICKET CLASSIFICATION - WEB APPLICATION")
    print("=" * 70)
    print("\nAdvanced Analytics Dashboard + AI Features")
    print("[OK] Mock data generated (50 tickets)")
    print("[OK] AI Assistant initialized")
    print(f"\nRunning on:")
    print(f"   http://{host}:{port}")
    print("\nFeatures:")
    print("   [OK] Analytics Dashboard with Charts")
    print("   [OK] AI-Powered Classification")
    print("   [OK] Sentiment Analysis")
    print("   [OK] Priority Detection")
    print("   [OK] Smart Routing & SLA Prediction")
    print("\nPress CTRL+C to stop")
    print("=" * 70 + "\n")
    
    app.run(host=host, port=port, debug=False)

