"""
Simple Keyword-Based Classifier
================================
Intelligent mock classification based on keywords
"""

def classify_ticket(text):
    """
    Classify ticket based on keywords
    
    Returns:
        tuple: (category, confidence)
    """
    text_lower = text.lower()
    
    # Category keywords with confidence scores
    categories = {
        'Hardware': {
            'keywords': [
                'laptop', 'computer', 'screen', 'monitor', 'keyboard', 'mouse',
                'printer', 'scanner', 'cable', 'broken', 'fell', 'dropped',
                'damage', 'repair', 'replace', 'hardware', 'device', 'equipment',
                'headset', 'webcam', 'charger', 'battery', 'power', 'fan',
                'noise', 'overheating', 'physical'
            ],
            'weight': 1.0
        },
        'Software': {
            'keywords': [
                'software', 'application', 'app', 'program', 'install', 'update',
                'crash', 'error', 'bug', 'slow', 'freeze', 'hang', 'loading',
                'excel', 'word', 'outlook', 'chrome', 'browser', 'antivirus',
                'license', 'activation', 'version', 'upgrade', 'download'
            ],
            'weight': 1.0
        },
        'Network': {
            'keywords': [
                'network', 'internet', 'wifi', 'connection', 'connect', 'vpn',
                'ethernet', 'cable', 'router', 'switch', 'ip', 'dns', 'firewall',
                'bandwidth', 'speed', 'slow internet', 'disconnected', 'offline',
                'cannot connect', 'no internet', 'network drive', 'shared folder'
            ],
            'weight': 1.0
        },
        'Access': {
            'keywords': [
                'access', 'login', 'password', 'username', 'credential', 'permission',
                'cannot access', 'locked out', 'forgot password', 'reset password',
                'account', 'authentication', 'authorization', 'denied', 'restricted',
                'unlock', 'enable', 'grant', 'sap', 'system access', 'database access'
            ],
            'weight': 1.0
        },
        'Administrative Rights': {
            'keywords': [
                'admin', 'administrator', 'admin rights', 'elevated', 'privilege',
                'sudo', 'root', 'administrator access', 'admin permission',
                'install software', 'need admin', 'run as administrator'
            ],
            'weight': 1.0
        },
        'Storage': {
            'keywords': [
                'storage', 'disk', 'drive', 'space', 'full', 'capacity', 'backup',
                'file', 'folder', 'delete', 'save', 'cloud', 'onedrive', 'sharepoint',
                'out of space', 'disk full', 'quota', 'expand', 'cleanup'
            ],
            'weight': 1.0
        },
        'Purchase': {
            'keywords': [
                'purchase', 'buy', 'order', 'request', 'procurement', 'vendor',
                'supplier', 'cost', 'price', 'invoice', 'payment', 'budget',
                'approve', 'approval', 'po', 'purchase order', 'need new'
            ],
            'weight': 1.0
        },
        'HR Support': {
            'keywords': [
                'hr', 'human resources', 'payroll', 'salary', 'leave', 'vacation',
                'absence', 'timesheet', 'employee', 'onboarding', 'offboarding',
                'benefits', 'insurance', 'contract', 'resign', 'hire', 'recruitment'
            ],
            'weight': 1.0
        }
    }
    
    # Calculate scores for each category
    scores = {}
    for category, data in categories.items():
        score = 0
        matched_keywords = []
        
        for keyword in data['keywords']:
            if keyword in text_lower:
                score += data['weight']
                matched_keywords.append(keyword)
        
        scores[category] = {
            'score': score,
            'matched': matched_keywords
        }
    
    # Find best match
    if all(s['score'] == 0 for s in scores.values()):
        # No keywords matched - default to Software
        return 'Software', 0.65, []
    
    # Get category with highest score
    best_category = max(scores.items(), key=lambda x: x[1]['score'])
    category = best_category[0]
    score = best_category[1]['score']
    matched = best_category[1]['matched']
    
    # Calculate confidence (0.75 - 0.98)
    # More matches = higher confidence
    base_confidence = 0.75
    match_bonus = min(score * 0.05, 0.23)  # Max +0.23
    confidence = base_confidence + match_bonus
    
    return category, round(confidence, 2), matched


def get_top_predictions(text, top_k=3):
    """
    Get top K predictions
    
    Returns:
        list: [(category, confidence), ...]
    """
    text_lower = text.lower()
    
    categories = {
        'Hardware': ['laptop', 'computer', 'screen', 'monitor', 'keyboard', 'mouse', 'printer', 'broken', 'fell'],
        'Software': ['software', 'application', 'app', 'program', 'install', 'update', 'crash', 'error', 'bug'],
        'Network': ['network', 'internet', 'wifi', 'connection', 'vpn', 'ethernet', 'router', 'dns'],
        'Access': ['access', 'login', 'password', 'username', 'credential', 'permission', 'locked out'],
        'Administrative Rights': ['admin', 'administrator', 'admin rights', 'elevated', 'privilege', 'sudo'],
        'Storage': ['storage', 'disk', 'drive', 'space', 'full', 'capacity', 'backup', 'file'],
        'Purchase': ['purchase', 'buy', 'order', 'request', 'procurement', 'vendor', 'cost'],
        'HR Support': ['hr', 'human resources', 'payroll', 'salary', 'leave', 'vacation', 'employee']
    }
    
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score
    
    # Sort by score
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Generate confidences
    results = []
    total_score = sum(s for _, s in sorted_cats[:top_k])
    
    if total_score == 0:
        # No matches - return defaults
        return [
            ('Software', 0.65),
            ('Hardware', 0.20),
            ('Access', 0.15)
        ]
    
    for i, (cat, score) in enumerate(sorted_cats[:top_k]):
        if i == 0:
            conf = 0.75 + min(score * 0.05, 0.23)
        elif i == 1:
            conf = 0.15 + (score / total_score) * 0.10
        else:
            conf = 0.05 + (score / total_score) * 0.05
        
        results.append((cat, round(conf, 2)))
    
    return results


# Test
if __name__ == '__main__':
    test_tickets = [
        "My laptop fell and screen broken",
        "I need access to SAP system",
        "Internet is not working",
        "Need to install Microsoft Office",
        "My computer is very slow"
    ]
    
    print("TESTING SIMPLE CLASSIFIER")
    print("=" * 60)
    
    for ticket in test_tickets:
        category, confidence, matched = classify_ticket(ticket)
        print(f"\nTicket: {ticket}")
        print(f"Category: {category} ({confidence*100:.1f}%)")
        print(f"Matched keywords: {', '.join(matched[:3])}")

