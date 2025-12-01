"""
Conversational IT Ticket Assistant v2.0
========================================

Advanced conversational AI with:
- Multi-turn conversation support
- Context tracking
- Sentiment analysis
- Dynamic follow-up questions
- Escalation logic

Author: Your Project
Date: November 2025
"""

import yaml
import re
from pathlib import Path
from datetime import datetime


class ConversationalAssistantV2:
    """
    Advanced conversational assistant with context awareness.
    """
    
    def __init__(self, template_path='../templates/response_templates.yaml'):
        """Initialize assistant with templates and conversation memory."""
        template_file = Path(__file__).parent / template_path
        
        with open(template_file, 'r', encoding='utf-8') as f:
            self.templates = yaml.safe_load(f)
        
        self.intent_keywords = self.templates['intents']
        self.followup_messages = self.templates['followup']
        
        # Conversation memory (stores context)
        self.conversation_history = []
        self.escalation_count = 0
        
    def detect_intent(self, text):
        """Detect user intent with priority ordering."""
        text_lower = text.lower()
        
        # Priority 1: Urgent (most important)
        if any(keyword in text_lower for keyword in self.intent_keywords['urgent']):
            return 'urgent'
        
        # Priority 2: Complaint (needs empathy)
        if any(keyword in text_lower for keyword in self.intent_keywords['complaint']):
            return 'complaint'
        
        # Priority 3: Question (needs information)
        if any(keyword in text_lower for keyword in self.intent_keywords['question']):
            return 'question'
        
        return 'standard'
    
    def detect_sentiment(self, text):
        """
        Simple sentiment detection.
        Returns: 'positive', 'negative', or 'neutral'
        """
        text_lower = text.lower()
        
        # Negative indicators
        negative_words = [
            'kötü', 'berbat', 'çalışmıyor', 'bozuk', 'yavaş', 'sorun', 'sinir', 'sinirliyim', 'kızgın', 
            'kızgınım', 'öfkeliyim', 'bıktım', 'yoruldum', 'hata', 'arıza', 'problem', 'kırık', 'kırıldı',
            'bad', 'terrible', 'broken', 'not working', 'slow', 'problem', 'error', 'crash',
            'frustrated', 'angry', 'upset', 'annoyed', 'mad', 'furious'
        ]
        
        # Positive indicators
        positive_words = [
            'teşekkür', 'sağol', 'iyi', 'güzel', 'mükemmel',
            'thank', 'thanks', 'good', 'great', 'excellent', 'perfect'
        ]
        
        neg_count = sum(1 for word in negative_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)
        
        if neg_count > pos_count and neg_count >= 2:
            return 'negative'
        elif pos_count > neg_count:
            return 'positive'
        else:
            return 'neutral'
    
    def should_escalate(self, category, confidence, intent):
        """
        Determine if ticket should be escalated to human agent.
        
        Returns: (bool, reason)
        """
        # Low confidence → escalate
        if confidence < 0.65:
            return True, "Düşük güven seviyesi - insan desteği önerilir"
        
        # Urgent + certain categories → immediate escalate
        if intent == 'urgent' and category in ['Hardware', 'Access', 'Network']:
            self.escalation_count += 1
            if self.escalation_count >= 1:  # First urgent = escalate
                return True, "Acil durum - anında destek gerekli"
        
        # Multiple attempts without resolution
        if len(self.conversation_history) >= 3:
            return True, "Çoklu deneme - uzman desteği önerilir"
        
        return False, None
    
    def add_to_history(self, user_message, bot_response, category, confidence):
        """Track conversation history for context."""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'bot': bot_response,
            'category': category,
            'confidence': confidence
        })
        
        # Keep last 10 messages only
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
    
    def generate_response(self, ticket_text, category, confidence, is_followup=False):
        """
        Generate advanced conversational response.
        
        Args:
            ticket_text: User's message
            category: Predicted category
            confidence: Prediction confidence
            is_followup: Whether this is a follow-up message
            
        Returns:
            dict: {
                'response': str,
                'intent': str,
                'sentiment': str,
                'should_escalate': bool,
                'escalation_reason': str,
                'suggested_actions': list,
                'follow_up_questions': list
            }
        """
        # Detect intent and sentiment
        intent = self.detect_intent(ticket_text)
        sentiment = self.detect_sentiment(ticket_text)
        confidence_level = self._get_confidence_level(confidence)
        
        # Check if escalation needed
        should_escalate, escalation_reason = self.should_escalate(category, confidence, intent)
        
        # Get template
        template = self.templates.get(category, self.templates['Miscellaneous'])
        
        # Build response
        response_parts = []
        
        # 1. Opening (context-aware)
        opening = self._generate_opening(
            intent, sentiment, confidence_level, is_followup
        )
        response_parts.append(opening)
        
        # 2. Category information
        category_info = self._generate_category_info(
            category, confidence, confidence_level
        )
        response_parts.append(category_info)
        
        # 3. Main content (steps or information)
        main_content = self._generate_main_content(
            template, intent, sentiment, confidence_level
        )
        response_parts.append(main_content)
        
        # 4. Estimated time
        if 'eta' in template and confidence_level != 'low':
            response_parts.append(f"\n⏱️ **Tahmini çözüm süresi:** {template['eta']}")
        
        # 5. Escalation or follow-up
        if should_escalate:
            escalation_msg = self._generate_escalation(escalation_reason, intent)
            response_parts.append(f"\n{escalation_msg}")
        else:
            followup = self._generate_followup(intent, confidence_level, sentiment)
            response_parts.append(f"\n{followup}")
        
        # Combine all parts
        full_response = "\n".join(response_parts)
        
        # Track in history
        self.add_to_history(ticket_text, full_response, category, confidence)
        
        # Return structured response
        return {
            'response': full_response,
            'intent': intent,
            'sentiment': sentiment,
            'confidence_level': confidence_level,
            'should_escalate': should_escalate,
            'escalation_reason': escalation_reason,
            'category': category,
            'confidence': confidence,
            'suggested_actions': template.get('urgent_steps' if intent == 'urgent' else 'standard_steps', []),
            'follow_up_questions': self._get_followup_questions(category, intent)
        }
    
    def _get_confidence_level(self, confidence):
        """Categorize confidence level."""
        if confidence >= 0.85:
            return 'high'
        elif confidence >= 0.70:
            return 'medium'
        else:
            return 'low'
    
    def _generate_opening(self, intent, sentiment, confidence_level, is_followup):
        """Generate context-aware opening."""
        
        # Follow-up message
        if is_followup:
            if sentiment == 'negative':
                return "😔 Üzgünüm, sorun devam ediyor. Başka bir yöntem deneyelim..."
            else:
                return "👍 Teşekkürler! Ek bilgiyle daha iyi yardımcı olabilirim."
        
        # First message
        if sentiment == 'negative' and intent == 'complaint':
            return "😔 Yaşadığınız sorunu anlıyorum, bu gerçekten sinir bozucu olmalı. Hemen çözelim!"
        
        elif intent == 'urgent':
            return "🚨 Acil durumunuzu görüyorum, hemen yardımcı oluyorum!"
        
        elif intent == 'question':
            return "👋 Merhaba! Sorunuza yanıt vereyim."
        
        elif confidence_level == 'high':
            return "✅ Sorununuzu net bir şekilde anladım."
        
        else:
            return "📋 Talebinizi inceliyorum..."
    
    def _generate_category_info(self, category, confidence, confidence_level):
        """Generate category explanation."""
        conf_pct = confidence * 100
        
        if confidence_level == 'high':
            return f"\n**📂 Kategori:** {category} (%{conf_pct:.1f} güven)"
        elif confidence_level == 'medium':
            return f"\n**📂 Muhtemel kategori:** {category} (%{conf_pct:.1f} güven)"
        else:
            return f"\n⚠️ **Kategori belirsiz** (en yakın: {category}, %{conf_pct:.1f})\n" \
                   "Daha iyi yardımcı olabilmem için sorununuzu detaylandırabilir misiniz?"
    
    def _generate_main_content(self, template, intent, sentiment, confidence_level):
        """Generate main response content."""
        parts = []
        
        # Low confidence → ask for more info
        if confidence_level == 'low':
            parts.append("\n💡 **Daha fazla bilgiye ihtiyacım var:**")
            parts.append("   • Hangi cihaz/sistem ile ilgili?")
            parts.append("   • Sorun ne zaman başladı?")
            parts.append("   • Hata mesajı var mı?")
            parts.append("   • Daha önce denediğiniz çözümler var mı?")
            return "\n".join(parts)
        
        # Urgent → immediate action steps
        if intent == 'urgent':
            parts.append("\n🚨 **Acil çözüm adımları:**")
            steps = template.get('urgent_steps', template.get('standard_steps', []))
            for i, step in enumerate(steps, 1):
                parts.append(f"   {i}. {step}")
        
        # Question → informative response
        elif intent == 'question':
            parts.append("\n❓ **Bilgi ve Rehber:**")
            info = template.get('question_response', '')
            if info:
                parts.append(info)
            
            # Add standard steps too
            parts.append("\n**Adım adım:**")
            steps = template.get('standard_steps', [])
            for i, step in enumerate(steps[:5], 1):  # Max 5 steps for questions
                parts.append(f"   {i}. {step}")
        
        # Complaint → empathy + solution
        elif intent == 'complaint' or sentiment == 'negative':
            parts.append("\n😔 **Sorununuzu anlıyorum. Hemen çözelim:**")
            steps = template.get('standard_steps', [])
            for i, step in enumerate(steps, 1):
                parts.append(f"   {i}. {step}")
            parts.append("\nBu adımlar yardımcı olmazsa, size öncelikli destek sağlayacağım.")
        
        # Standard → normal steps
        else:
            parts.append("\n📋 **Önerilen çözüm adımları:**")
            steps = template.get('standard_steps', [])
            for i, step in enumerate(steps, 1):
                parts.append(f"   {i}. {step}")
        
        return "\n".join(parts)
    
    def _generate_escalation(self, reason, intent):
        """Generate escalation message."""
        if intent == 'urgent':
            return f"""
🚨 **Acil Destek Gerekiyor!**

Bu sorunu hemen çözmek için size bir **IT uzmanı atıyorum**.

📞 Acil destek hattı: **(555) 123-4567**
📧 E-posta: **priority@support.com**
🎫 Ticket numaranız: **#URG-{datetime.now().strftime('%Y%m%d-%H%M')}**

⏰ Bir uzman **5-10 dakika** içinde sizinle iletişime geçecek.

Sebep: {reason}
"""
        else:
            return f"""
👤 **İnsan Desteğine Yönlendirme**

Bu sorun için bir destek uzmanının yardımı daha uygun olacak.

🎫 Destek talebi oluşturuldu: **#TKT-{datetime.now().strftime('%Y%m%d-%H%M')}**
📧 E-posta bildirimi gönderildi
⏰ Yanıt süresi: **2-4 saat** içinde

Sebep: {reason}
"""
    
    def _generate_followup(self, intent, confidence_level, sentiment):
        """Generate smart follow-up."""
        
        if intent == 'urgent' and confidence_level == 'high':
            return "❗ Bu adımları denediniz mi? Sonuç nasıl oldu?"
        
        elif intent == 'question':
            return "❓ Bu bilgiler yeterli oldu mu? Başka soru var mı?"
        
        elif sentiment == 'negative':
            return "💬 Bu çözümler yardımcı olmazsa, hemen bir uzmanla görüştürebilirim."
        
        elif confidence_level == 'low':
            return "🔍 Daha detaylı bilgi verirseniz, daha spesifik çözümler önerebilirim."
        
        else:
            return self.followup_messages.get('solved', 'Bu adımlar yardımcı oldu mu?')
    
    def _get_followup_questions(self, category, intent):
        """Generate dynamic follow-up questions."""
        
        questions = []
        
        if category == 'Hardware':
            questions = [
                "Sorun son yazılım güncellemesinden sonra mı başladı?",
                "Daha önce benzer bir sorun yaşadınız mı?",
                "Cihaz garanti kapsamında mı?"
            ]
        
        elif category == 'Access':
            questions = [
                "Şifrenizi en son ne zaman değiştirdiniz?",
                "Başka bir cihazdan giriş yapmayı denediniz mi?",
                "VPN üzerinden mi bağlanmaya çalışıyorsunuz?"
            ]
        
        elif category == 'Network':
            questions = [
                "Kablolu bağlantı da yavaş mı?",
                "Sadece belirli sitelerde mi yavaş?",
                "Sorun belli saatlerde mi oluyor?"
            ]
        
        elif category == 'Software':
            questions = [
                "Hangi yazılım versiyonunu kullanıyorsunuz?",
                "Hata kodu veya mesajı var mı?",
                "Başka kullanıcılarda da aynı sorun var mı?"
            ]
        
        return questions[:2]  # Return max 2 questions
    
    def generate_simple_response(self, ticket_text, category, confidence):
        """
        Simplified response for quick replies (backwards compatible).
        """
        result = self.generate_response(ticket_text, category, confidence)
        return result['response']
    
    def reset_conversation(self):
        """Reset conversation context (new user/session)."""
        self.conversation_history = []
        self.escalation_count = 0


# Backwards compatibility wrapper
class ConversationalAssistant(ConversationalAssistantV2):
    """Alias for backwards compatibility."""
    pass


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Initialize assistant
    assistant = ConversationalAssistantV2()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Acil Hardware - Kızgın Kullanıcı",
            "text": "Bilgisayarım çalışmıyor acil toplantıya giremiyorum çok kötü durum",
            "category": "Hardware",
            "confidence": 0.91
        },
        {
            "name": "Soru - Access",
            "text": "Şifremi nasıl değiştirebilirim sistem nerede",
            "category": "Access",
            "confidence": 0.88
        },
        {
            "name": "Düşük Güven - Belirsiz",
            "text": "Bir şey çalışmıyor ama ne olduğunu bilmiyorum",
            "category": "Miscellaneous",
            "confidence": 0.58
        },
        {
            "name": "Şikayet - Network",
            "text": "İnternet berbat yavaş çalışamıyorum böyle olmaz",
            "category": "Network",
            "confidence": 0.76
        },
        {
            "name": "Pozitif - Teşekkür",
            "text": "Önceki çözüm çok iyi oldu teşekkürler ama başka sorum var",
            "category": "Software",
            "confidence": 0.85
        }
    ]
    
    print("=" * 80)
    print("🤖 CONVERSATIONAL ASSISTANT V2.0 - ADVANCED TEST")
    print("=" * 80)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n\n{'='*80}")
        print(f"📋 TEST SCENARIO {i}: {scenario['name']}")
        print(f"{'='*80}")
        print(f"\n👤 User: {scenario['text']}")
        print(f"\n🤖 Assistant:\n")
        
        result = assistant.generate_response(
            ticket_text=scenario['text'],
            category=scenario['category'],
            confidence=scenario['confidence']
        )
        
        print(result['response'])
        
        # Show metadata
        print(f"\n📊 Metadata:")
        print(f"   Intent: {result['intent']}")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Escalate: {result['should_escalate']}")
        if result['should_escalate']:
            print(f"   Reason: {result['escalation_reason']}")
        
        # Show follow-up questions
        if result['follow_up_questions']:
            print(f"\n💬 Suggested follow-up questions:")
            for q in result['follow_up_questions']:
                print(f"   • {q}")
        
        print("\n" + "-" * 80)
    
    print("\n\n✅ All tests completed!")
    print(f"📈 Conversation history: {len(assistant.conversation_history)} messages tracked")

