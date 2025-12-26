"""
AI-Powered Intelligent Features
================================

Advanced AI features for ticket classification:
- Sentiment Analysis
- Priority Detection  
- Smart Routing
- SLA Prediction
- Similar Tickets Finder

Author: Your Name
Date: December 2025
"""

import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SentimentAnalyzer:
    """
    Sentiment Analysis for IT Tickets
    
    Analyzes ticket text to determine:
    - Sentiment: positive, negative, neutral, urgent
    - Emotional tone
    - Urgency level
    """
    
    def __init__(self):
        # Positive keywords
        self.positive_keywords = [
            'thank', 'thanks', 'appreciate', 'great', 'good', 'excellent',
            'perfect', 'working', 'resolved', 'fixed', 'helpful', 'quick'
        ]
        
        # Negative keywords
        self.negative_keywords = [
            'not working', 'broken', 'error', 'fail', 'issue', 'problem',
            'cannot', 'unable', 'down', 'crash', 'stuck', 'slow', 'won\'t',
            'doesn\'t', 'don\'t', 'can\'t', 'isn\'t', 'urgent', 'critical'
        ]
        
        # Urgent keywords
        self.urgent_keywords = [
            'urgent', 'asap', 'immediately', 'critical', 'emergency',
            'now', 'today', 'deadline', 'production', 'down', 'outage',
            'losing', 'loss', 'crash', 'severe', 'major'
        ]
        
        # Frustration keywords
        self.frustration_keywords = [
            'again', 'still', 'always', 'never', 'frustrated', 'annoying',
            'unacceptable', 'ridiculous', 'waste', 'terrible', 'awful'
        ]
    
    def analyze(self, text):
        """
        Analyze sentiment of ticket text
        
        Returns:
            dict: {
                'sentiment': 'positive' | 'negative' | 'neutral' | 'urgent',
                'score': float (-1 to 1),
                'is_urgent': bool,
                'is_frustrated': bool,
                'confidence': float (0 to 1)
            }
        """
        text_lower = text.lower()
        
        # Count keyword matches
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        urgent_count = sum(1 for kw in self.urgent_keywords if kw in text_lower)
        frustration_count = sum(1 for kw in self.frustration_keywords if kw in text_lower)
        
        # Check for exclamation marks and CAPS (indicates urgency/frustration)
        exclamations = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Calculate sentiment score
        score = positive_count - negative_count
        
        # Normalize score
        total_keywords = positive_count + negative_count
        if total_keywords > 0:
            score = score / total_keywords
        else:
            score = 0
        
        # Determine is_urgent
        is_urgent = (urgent_count > 0 or 
                    exclamations > 2 or 
                    caps_ratio > 0.3 or
                    'asap' in text_lower or
                    'urgent' in text_lower)
        
        # Determine is_frustrated
        is_frustrated = (frustration_count > 0 or 
                        exclamations > 3 or
                        'again' in text_lower or
                        'still' in text_lower)
        
        # Determine overall sentiment
        if is_urgent:
            sentiment = 'urgent'
        elif score > 0.2:
            sentiment = 'positive'
        elif score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Calculate confidence (based on keyword density)
        word_count = len(text.split())
        keyword_density = total_keywords / word_count if word_count > 0 else 0
        confidence = min(keyword_density * 3, 1.0)  # Cap at 1.0
        
        return {
            'sentiment': sentiment,
            'score': round(score, 2),
            'is_urgent': is_urgent,
            'is_frustrated': is_frustrated,
            'confidence': round(confidence, 2)
        }


class PriorityDetector:
    """
    Priority Detection for IT Tickets
    
    Determines ticket priority based on:
    - Sentiment analysis
    - Keywords
    - Category
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # High priority categories
        self.high_priority_categories = [
            'access', 'network', 'administrative rights'
        ]
        
        # Critical keywords
        self.critical_keywords = [
            'production', 'outage', 'down', 'critical', 'security',
            'breach', 'losing money', 'cannot work', 'blocking'
        ]
    
    def detect(self, text, category=None):
        """
        Detect priority level
        
        Returns:
            dict: {
                'priority': 'HIGH' | 'MEDIUM' | 'LOW',
                'reasons': list of reasons,
                'score': int (0-100)
            }
        """
        text_lower = text.lower()
        reasons = []
        score = 50  # Start at medium
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze(text)
        
        # Urgent sentiment -> High priority
        if sentiment['is_urgent']:
            score += 30
            reasons.append('Urgent language detected')
        
        # Frustrated user -> Increase priority
        if sentiment['is_frustrated']:
            score += 15
            reasons.append('User frustration detected')
        
        # Critical keywords
        critical_matches = sum(1 for kw in self.critical_keywords if kw in text_lower)
        if critical_matches > 0:
            score += critical_matches * 10
            reasons.append(f'{critical_matches} critical keyword(s)')
        
        # High priority categories
        if category and category.lower() in self.high_priority_categories:
            score += 10
            reasons.append(f'High-priority category: {category}')
        
        # Determine priority level
        if score >= 75:
            priority = 'HIGH'
        elif score >= 40:
            priority = 'MEDIUM'
        else:
            priority = 'LOW'
        
        return {
            'priority': priority,
            'reasons': reasons,
            'score': min(score, 100)
        }


class SmartRouter:
    """
    Smart Routing for IT Tickets
    
    Routes tickets to appropriate teams/departments
    """
    
    def __init__(self):
        self.routing_map = {
            'access': ['IT Security', 'Access Management'],
            'hardware': ['Hardware Support', 'IT Assets'],
            'software': ['Software Support', 'Application Team'],
            'network': ['Network Team', 'Infrastructure'],
            'hr support': ['HR Department', 'People Operations'],
            'purchase': ['Procurement', 'Finance'],
            'storage': ['Storage Team', 'Infrastructure'],
            'administrative rights': ['IT Security', 'System Admin'],
            'misc': ['General IT Support']
        }
    
    def route(self, category, priority='MEDIUM'):
        """
        Route ticket to appropriate team
        
        Returns:
            dict: {
                'primary_team': str,
                'secondary_team': str or None,
                'escalate': bool
            }
        """
        category_lower = category.lower()
        
        teams = self.routing_map.get(category_lower, ['General IT Support', None])
        
        return {
            'primary_team': teams[0],
            'secondary_team': teams[1] if len(teams) > 1 else None,
            'escalate': priority == 'HIGH'
        }


class SLAPredictor:
    """
    SLA (Service Level Agreement) Prediction
    
    Predicts resolution time based on priority and category
    """
    
    def __init__(self):
        # SLA times in hours
        self.sla_matrix = {
            'HIGH': {
                'access': 2,
                'network': 1,
                'hardware': 4,
                'software': 3,
                'default': 2
            },
            'MEDIUM': {
                'access': 8,
                'network': 4,
                'hardware': 24,
                'software': 12,
                'default': 12
            },
            'LOW': {
                'access': 48,
                'network': 24,
                'hardware': 72,
                'software': 48,
                'default': 48
            }
        }
    
    def predict(self, priority, category):
        """
        Predict SLA resolution time
        
        Returns:
            dict: {
                'hours': int,
                'deadline': str (ISO format),
                'description': str
            }
        """
        category_lower = category.lower().replace(' ', '')
        
        # Get SLA hours
        priority_sla = self.sla_matrix.get(priority, self.sla_matrix['MEDIUM'])
        hours = priority_sla.get(category_lower, priority_sla['default'])
        
        # Calculate deadline
        deadline = datetime.now() + timedelta(hours=hours)
        
        # Description
        if hours < 4:
            description = f'Within {hours} hours (Same day)'
        elif hours < 24:
            description = f'Within {hours} hours (Today/Tomorrow)'
        elif hours < 48:
            description = f'Within {hours//24} day(s)'
        else:
            description = f'Within {hours//24} days'
        
        return {
            'hours': hours,
            'deadline': deadline.strftime('%Y-%m-%d %H:%M:%S'),
            'description': description
        }


class SimilarTicketsFinder:
    """
    Find Similar Tickets using TF-IDF + Cosine Similarity
    
    Helps find previously resolved similar issues
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.ticket_texts = []
        self.ticket_metadata = []
        self.fitted = False
    
    def fit(self, tickets):
        """
        Fit the vectorizer with historical tickets
        
        Args:
            tickets: List of dicts with 'text', 'category', 'id', etc.
        """
        if not tickets:
            return
        
        self.ticket_texts = [t['text'] for t in tickets]
        self.ticket_metadata = tickets
        
        if len(self.ticket_texts) > 0:
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.ticket_texts)
                self.fitted = True
            except:
                self.fitted = False
    
    def find_similar(self, text, top_k=3, category_filter=None):
        """
        Find similar tickets
        
        Returns:
            list: [
                {
                    'id': str,
                    'text': str,
                    'category': str,
                    'similarity': float,
                    'timestamp': str
                },
                ...
            ]
        """
        if not self.fitted or len(self.ticket_texts) == 0:
            return []
        
        try:
            # Transform query
            query_vec = self.vectorizer.transform([text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            
            # Get top K indices
            top_indices = similarities.argsort()[-top_k-1:-1][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    ticket = self.ticket_metadata[idx]
                    
                    # Filter by category if specified
                    if category_filter and ticket.get('category') != category_filter:
                        continue
                    
                    results.append({
                        'id': ticket.get('id', 'N/A'),
                        'text': ticket['text'][:100] + '...' if len(ticket['text']) > 100 else ticket['text'],
                        'category': ticket.get('category', 'N/A'),
                        'similarity': round(float(similarities[idx]) * 100, 1),
                        'timestamp': ticket.get('timestamp', 'N/A')
                    })
            
            return results[:top_k]
        except:
            return []


# Integrated AI Assistant
class IntelligentTicketAssistant:
    """
    Complete AI Assistant combining all features
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.priority_detector = PriorityDetector()
        self.smart_router = SmartRouter()
        self.sla_predictor = SLAPredictor()
        self.similar_finder = SimilarTicketsFinder()
    
    def analyze_ticket(self, text, category, ticket_history=None):
        """
        Complete ticket analysis
        
        Returns:
            dict: All AI features combined
        """
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.analyze(text)
        
        # Priority detection
        priority = self.priority_detector.detect(text, category)
        
        # Smart routing
        routing = self.smart_router.route(category, priority['priority'])
        
        # SLA prediction
        sla = self.sla_predictor.predict(priority['priority'], category)
        
        # Similar tickets
        similar = []
        if ticket_history:
            self.similar_finder.fit(ticket_history)
            similar = self.similar_finder.find_similar(text, top_k=3)
        
        return {
            'sentiment': sentiment,
            'priority': priority,
            'routing': routing,
            'sla': sla,
            'similar_tickets': similar
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("AI-POWERED INTELLIGENT FEATURES - DEMONSTRATION")
    print("=" * 70)
    
    assistant = IntelligentTicketAssistant()
    
    # Test ticket
    test_ticket = "URGENT! Production server is down and we're losing money! Need immediate help!"
    test_category = "Network"
    
    print(f"\nTest Ticket: {test_ticket}")
    print(f"Category: {test_category}\n")
    
    result = assistant.analyze_ticket(test_ticket, test_category)
    
    print("ANALYSIS RESULTS:")
    print("-" * 70)
    print(f"Sentiment: {result['sentiment']['sentiment']} (Score: {result['sentiment']['score']})")
    print(f"Is Urgent: {result['sentiment']['is_urgent']}")
    print(f"Priority: {result['priority']['priority']} (Score: {result['priority']['score']})")
    print(f"Reasons: {', '.join(result['priority']['reasons'])}")
    print(f"Route to: {result['routing']['primary_team']}")
    print(f"SLA: {result['sla']['description']}")
    print(f"Deadline: {result['sla']['deadline']}")
    
    print("\n" + "=" * 70)
    print("READY FOR INTEGRATION!")
    print("=" * 70)


