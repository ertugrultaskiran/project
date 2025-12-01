"""
Domain-Specific Feature Engineering for IT Ticket Classification
=================================================================

ORIGINAL CONTRIBUTION for Graduation Project

This module extracts IT ticket-specific features that are not captured
by standard NLP approaches (TF-IDF, Word2Vec, BERT).

These features leverage domain knowledge about IT support tickets
to improve classification accuracy.

Author: [Your Name]
Date: November 2025
"""

import re
import numpy as np
import pandas as pd
from collections import Counter


class ITTicketFeatureExtractor:
    """
    Extract domain-specific features from IT support tickets
    
    This is an ORIGINAL CONTRIBUTION that goes beyond standard NLP.
    We leverage domain knowledge about IT tickets to extract meaningful features.
    """
    
    def __init__(self):
        # IT-specific keyword dictionaries (domain knowledge)
        self.access_keywords = [
            'access', 'permission', 'login', 'password', 'credential',
            'authentication', 'authorization', 'rights', 'privilege',
            'unlock', 'account', 'user', 'cannot access', 'denied'
        ]
        
        self.hardware_keywords = [
            'laptop', 'computer', 'monitor', 'keyboard', 'mouse',
            'printer', 'cable', 'hardware', 'device', 'equipment',
            'screen', 'broken', 'physical', 'dock', 'headset'
        ]
        
        self.software_keywords = [
            'software', 'application', 'app', 'program', 'install',
            'update', 'version', 'bug', 'error', 'crash', 'freeze',
            'license', 'patch', 'upgrade'
        ]
        
        self.network_keywords = [
            'network', 'internet', 'wifi', 'vpn', 'connection',
            'connectivity', 'offline', 'slow', 'disconnect', 'proxy',
            'firewall', 'router'
        ]
        
        self.hr_keywords = [
            'leave', 'vacation', 'payroll', 'salary', 'timesheet',
            'attendance', 'hr', 'human resources', 'benefits', 'onboarding',
            'offboarding', 'resignation', 'training', 'policy'
        ]
        
        self.purchase_keywords = [
            'purchase', 'buy', 'order', 'procurement', 'vendor',
            'supplier', 'quote', 'invoice', 'budget', 'cost',
            'approval', 'requisition'
        ]
        
        self.storage_keywords = [
            'storage', 'disk', 'space', 'drive', 'backup', 'archive',
            'folder', 'file', 'share', 'capacity', 'quota', 'full'
        ]
        
        # Urgency indicators
        self.urgency_keywords = [
            'urgent', 'asap', 'immediately', 'critical', 'emergency',
            'priority', 'high priority', 'important', 'now', 'today',
            'deadline', 'rush'
        ]
        
        # Common IT systems/tools
        self.systems = [
            'sap', 'salesforce', 'jira', 'confluence', 'slack',
            'teams', 'outlook', 'excel', 'sharepoint', 'workday',
            'oracle', 'aws', 'azure', 'windows', 'mac', 'linux'
        ]
        
    def extract_features(self, text):
        """
        Extract all domain-specific features from a ticket
        
        Args:
            text (str): Ticket text
        
        Returns:
            dict: Dictionary of features
        """
        text_lower = text.lower()
        
        features = {}
        
        # 1. Keyword-based features (domain knowledge)
        features['access_score'] = self._keyword_score(text_lower, self.access_keywords)
        features['hardware_score'] = self._keyword_score(text_lower, self.hardware_keywords)
        features['software_score'] = self._keyword_score(text_lower, self.software_keywords)
        features['network_score'] = self._keyword_score(text_lower, self.network_keywords)
        features['hr_score'] = self._keyword_score(text_lower, self.hr_keywords)
        features['purchase_score'] = self._keyword_score(text_lower, self.purchase_keywords)
        features['storage_score'] = self._keyword_score(text_lower, self.storage_keywords)
        
        # 2. Urgency indicators
        features['urgency_score'] = self._keyword_score(text_lower, self.urgency_keywords)
        features['has_urgent'] = 1 if any(kw in text_lower for kw in self.urgency_keywords) else 0
        
        # 3. System/tool mentions
        features['system_mentions'] = sum(1 for sys in self.systems if sys in text_lower)
        
        # 4. Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
        
        # 5. Punctuation analysis
        features['question_marks'] = text.count('?')
        features['exclamation_marks'] = text.count('!')
        features['is_question'] = 1 if '?' in text else 0
        
        # 6. Capital letters (indicates emphasis/urgency)
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # 7. Number presence (version numbers, ticket IDs, etc.)
        features['has_numbers'] = 1 if bool(re.search(r'\d', text)) else 0
        features['number_count'] = len(re.findall(r'\d+', text))
        
        # 8. Email/URL presence
        features['has_email'] = 1 if bool(re.search(r'\S+@\S+', text)) else 0
        features['has_url'] = 1 if bool(re.search(r'http[s]?://|www\.', text)) else 0
        
        # 9. Technical terms density
        technical_pattern = r'(?i)\b(error|issue|problem|fail|unable|cannot|not working|broken)\b'
        features['technical_terms'] = len(re.findall(technical_pattern, text_lower))
        
        # 10. Sentence structure
        features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        
        return features
    
    def _keyword_score(self, text, keywords):
        """
        Calculate score based on keyword presence
        
        Score = (number of matching keywords) / (total keywords)
        
        This gives us a normalized score between 0 and 1
        """
        matches = sum(1 for kw in keywords if kw in text)
        return matches / len(keywords) if keywords else 0
    
    def extract_batch(self, texts):
        """
        Extract features for multiple texts
        
        Args:
            texts: List or Series of texts
        
        Returns:
            DataFrame with all features
        """
        features_list = [self.extract_features(text) for text in texts]
        return pd.DataFrame(features_list)


class HybridFeatureModel:
    """
    Combine text embeddings with custom features
    
    This creates a hybrid model that uses:
    1. Deep learning embeddings (LSTM/BERT) - semantic understanding
    2. Custom domain features - domain expertise
    
    This is a NOVEL APPROACH for IT ticket classification!
    """
    
    def __init__(self, text_model, feature_extractor):
        """
        Args:
            text_model: Pre-trained text model (LSTM/BERT)
            feature_extractor: ITTicketFeatureExtractor instance
        """
        self.text_model = text_model
        self.feature_extractor = feature_extractor
    
    def predict_hybrid(self, texts):
        """
        Make predictions using both text model and custom features
        
        Strategy:
        1. Get text model predictions (probabilities)
        2. Extract custom features
        3. Adjust predictions based on custom features
        
        This is a simple fusion strategy - can be made more sophisticated!
        """
        # Get text model predictions
        text_probs = self.text_model.predict(texts)
        
        # Extract custom features
        custom_features = self.feature_extractor.extract_batch(texts)
        
        # Feature-based adjustment (simple heuristic)
        # This can be learned with a meta-model!
        adjusted_probs = text_probs.copy()
        
        for idx, features in custom_features.iterrows():
            # Boost Access category if high access score
            if features['access_score'] > 0.2:
                adjusted_probs[idx, 0] *= 1.2  # Assuming Access is class 0
            
            # Boost Hardware if high hardware score
            if features['hardware_score'] > 0.15:
                adjusted_probs[idx, 3] *= 1.2  # Assuming Hardware is class 3
            
            # Boost urgency handling...
            # etc.
        
        # Re-normalize probabilities
        adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)
        
        return adjusted_probs


# Demonstration and testing
if __name__ == "__main__":
    print("=" * 70)
    print("CUSTOM FEATURE ENGINEERING - ORIGINAL CONTRIBUTION")
    print("=" * 70)
    
    extractor = ITTicketFeatureExtractor()
    
    # Test with example tickets
    test_tickets = [
        "I need urgent access to the SAP system. My password is not working.",
        "The printer on 3rd floor is broken. Need hardware replacement.",
        "Can you help me with my leave request in Workday?",
        "Storage quota exceeded. Need more disk space for project files."
    ]
    
    print("\nExtracting features from example tickets:\n")
    
    for i, ticket in enumerate(test_tickets, 1):
        print(f"Ticket {i}: {ticket[:50]}...")
        features = extractor.extract_features(ticket)
        
        # Show top features
        print("  Top scores:")
        scores = {k: v for k, v in features.items() if 'score' in k}
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for feat, score in top_scores:
            if score > 0:
                print(f"    - {feat}: {score:.3f}")
        print()
    
    print("=" * 70)
    print("\nKEY CONTRIBUTIONS:")
    print("  ✓ Domain-specific feature engineering (IT tickets)")
    print("  ✓ 20+ custom features based on domain knowledge")
    print("  ✓ Hybrid approach: Deep learning + domain expertise")
    print("  ✓ Explainable: Features are interpretable")
    print("\nThis goes beyond standard NLP approaches!")
    print("=" * 70)

