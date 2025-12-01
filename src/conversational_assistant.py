"""
Conversational IT Ticket Assistant
===================================

Template-based conversational response generator for IT tickets.
Provides empathetic, actionable, and context-aware responses.

Author: Your Project
Date: November 2025
"""

import yaml
import re
from pathlib import Path


class ConversationalAssistant:
    """
    Generates conversational responses for IT ticket classification.
    
    Features:
    - Template-based responses (no LLM cost)
    - Intent detection (urgent, question, complaint)
    - Confidence-aware tone
    - Actionable steps
    - Follow-up suggestions
    """
    
    def __init__(self, template_path='../templates/response_templates.yaml'):
        """Load response templates."""
        template_file = Path(__file__).parent / template_path
        
        with open(template_file, 'r', encoding='utf-8') as f:
            self.templates = yaml.safe_load(f)
        
        self.intent_keywords = self.templates['intents']
        self.followup_messages = self.templates['followup']
    
    def detect_intent(self, text):
        """
        Detect user intent from text.
        
        Args:
            text: User's ticket text
            
        Returns:
            str: 'urgent', 'question', 'complaint', or 'standard'
        """
        text_lower = text.lower()
        
        # Check for urgent intent
        if any(keyword in text_lower for keyword in self.intent_keywords['urgent']):
            return 'urgent'
        
        # Check for question intent
        if any(keyword in text_lower for keyword in self.intent_keywords['question']):
            return 'question'
        
        # Check for complaint intent
        if any(keyword in text_lower for keyword in self.intent_keywords['complaint']):
            return 'complaint'
        
        return 'standard'
    
    def get_confidence_level(self, confidence):
        """
        Categorize confidence level.
        
        Args:
            confidence: Float between 0 and 1
            
        Returns:
            str: 'high', 'medium', or 'low'
        """
        if confidence >= 0.85:
            return 'high'
        elif confidence >= 0.70:
            return 'medium'
        else:
            return 'low'
    
    def generate_response(self, ticket_text, category, confidence):
        """
        Generate conversational response.
        
        Args:
            ticket_text: Original ticket text
            category: Predicted category
            confidence: Prediction confidence (0-1)
            
        Returns:
            str: Formatted conversational response
        """
        # Detect intent
        intent = self.detect_intent(ticket_text)
        confidence_level = self.get_confidence_level(confidence)
        
        # Get template for this category
        template = self.templates.get(category, self.templates['Miscellaneous'])
        
        # Build response parts
        parts = []
        
        # 1. Greeting (empathetic opening)
        greeting = self._get_greeting(template, intent, confidence_level)
        parts.append(greeting)
        
        # 2. Category explanation
        if confidence_level == 'high':
            parts.append(f"\n**Kategori:** {category} (Güven: %{confidence*100:.1f})")
        elif confidence_level == 'medium':
            parts.append(f"\n**Muhtemel kategori:** {category} (%{confidence*100:.1f} güven)")
        else:
            parts.append(f"\n⚠️ **Kategori belirsiz** (en yakın: {category}, %{confidence*100:.1f} güven)")
            parts.append("\n💡 **Daha iyi yardımcı olabilmem için:**")
            parts.append("   • Sorununuzu daha detaylı açıklar mısınız?")
            parts.append("   • Hangi cihaz/sistem ile ilgili?")
            parts.append("   • Ne zaman başladı?")
            parts.append("   • Hata mesajı var mı?")
        
        # 3. Main content (steps or information)
        content = self._get_main_content(template, intent)
        parts.append(content)
        
        # 4. Estimated time
        if 'eta' in template:
            parts.append(f"\n⏱️ **Tahmini çözüm süresi:** {template['eta']}")
        
        # 5. Follow-up
        followup = self._get_followup(intent, confidence_level)
        parts.append(f"\n{followup}")
        
        return "\n".join(parts)
    
    def _get_greeting(self, template, intent, confidence_level):
        """Get appropriate greeting based on intent and confidence."""
        if intent == 'urgent':
            return template.get('greeting_urgent', '🚨 Acil durumunuzu anlıyorum!')
        elif confidence_level == 'high':
            return template.get('greeting_confident', '✅ Sorununuzu tespit ettim.')
        else:
            return template.get('greeting_uncertain', '📋 Sorununuzu inceliyorum.')
    
    def _get_main_content(self, template, intent):
        """Get main response content based on intent."""
        parts = []
        
        if intent == 'urgent':
            # Urgent steps
            parts.append("\n🚨 **Acil çözüm adımları:**")
            steps = template.get('urgent_steps', template.get('standard_steps', []))
            for i, step in enumerate(steps, 1):
                parts.append(f"   {i}. {step}")
        
        elif intent == 'question':
            # Informative response
            parts.append("\n❓ **Bilgi:**")
            info = template.get('question_response', '')
            if info:
                parts.append(info)
            else:
                # Fallback to standard steps
                steps = template.get('standard_steps', [])
                for i, step in enumerate(steps, 1):
                    parts.append(f"   {i}. {step}")
        
        else:
            # Standard steps
            parts.append("\n📋 **Önerilen adımlar:**")
            steps = template.get('standard_steps', [])
            for i, step in enumerate(steps, 1):
                parts.append(f"   {i}. {step}")
        
        return "\n".join(parts)
    
    def _get_followup(self, intent, confidence_level):
        """Get appropriate follow-up message."""
        if intent == 'urgent':
            return "❗ Bu adımlar işe yaramazsa, size **Priority-1** destek talebi oluşturayım mı?"
        elif confidence_level == 'low':
            return "⚠️ Kategori tahmini belirsiz. Sorununuzu **daha detaylı** anlatabilir misiniz? Veya size bir destek uzmanı atayayım mı?"
        else:
            return self.followup_messages['solved']
    
    def generate_short_response(self, category, confidence):
        """
        Generate short response for quick replies.
        
        Args:
            category: Predicted category
            confidence: Confidence score
            
        Returns:
            str: Short formatted response
        """
        if confidence >= 0.85:
            return f"✅ Bu **{category}** kategorisine giriyor. Hemen yardımcı olayım!"
        else:
            return f"📋 Muhtemelen **{category}** ile ilgili. Detaylı bakalım..."


# Example usage
if __name__ == "__main__":
    # Initialize assistant
    assistant = ConversationalAssistant()
    
    # Test cases
    test_cases = [
        {
            "text": "Bilgisayarım açılmıyor acil yardım",
            "category": "Hardware",
            "confidence": 0.89
        },
        {
            "text": "Şifremi unuttum nasıl sıfırlarım",
            "category": "Access",
            "confidence": 0.92
        },
        {
            "text": "İnternet çok yavaş çalışmıyor",
            "category": "Network",
            "confidence": 0.76
        }
    ]
    
    # Generate responses
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 70)
    print("CONVERSATIONAL ASSISTANT TEST")
    print("=" * 70)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST CASE {i}")
        print(f"{'='*70}")
        print(f"\nUser: {case['text']}")
        print(f"\nAssistant:\n")
        
        response = assistant.generate_response(
            ticket_text=case['text'],
            category=case['category'],
            confidence=case['confidence']
        )
        
        print(response)

