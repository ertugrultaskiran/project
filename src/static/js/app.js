// Global state
let tickets = [];
let stats = {};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadTickets();
    loadStats();
    
    // Refresh every 10 seconds
    setInterval(() => {
        loadStats();
    }, 10000);
    
    // Enter key to send
    document.getElementById('chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            sendMessage();
        }
    });
});

// Toggle chatbot
function toggleChatbot() {
    const widget = document.getElementById('chatbotWidget');
    widget.classList.toggle('minimized');
}

// Send message and classify ticket
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    const modelSelect = document.getElementById('modelSelect');
    const model = modelSelect.value;
    
    if (!text) {
        showToast('Please enter a ticket description', 'error');
        return;
    }
    
    // Add user message to chat
    addChatMessage(text, 'user');
    
    // Clear input
    input.value = '';
    
    // Show loading
    showLoading(true);
    
    // Add typing indicator
    addTypingIndicator();
    
    try {
        // Call CONVERSATIONAL API (not just classification!)
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text, model })
        });
        
        if (!response.ok) {
            throw new Error('Classification failed');
        }
        
        const result = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add CONVERSATIONAL bot response (formatted with line breaks)
        addConversationalResponse(result);
        
        // Refresh tickets table
        await loadTickets();
        await loadStats();
        
        // Show success toast
        showToast('Ticket classified successfully!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addChatMessage('❌ Sorry, there was an error classifying your ticket. Please try again.', 'bot');
        showToast('Error classifying ticket', 'error');
    } finally {
        showLoading(false);
    }
}

// Typing indicator functions
function addTypingIndicator() {
    const messagesDiv = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator-msg';
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// Add user message to chat
function addChatMessage(text, sender) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(text)}</p>
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Add CONVERSATIONAL bot response with v2.0 features
function addConversationalResponse(result) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const confidence = (result.confidence * 100).toFixed(1);
    const categoryBadgeClass = getCategoryBadgeClass(result.category);
    
    // Format conversational response (preserve line breaks and formatting)
    const formattedResponse = result.response
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold text
        .replace(/• /g, '&nbsp;&nbsp;• ');  // Bullet points
    
    // Build metadata badges
    let badges = '';
    
    // Sentiment badge
    if (result.sentiment) {
        const sentimentEmoji = {
            'positive': '😊',
            'negative': '😔',
            'neutral': '😐'
        };
        badges += `<span class="badge badge-${result.sentiment}">${sentimentEmoji[result.sentiment]} ${result.sentiment}</span> `;
    }
    
    // Intent badge
    if (result.intent) {
        const intentEmoji = {
            'urgent': '🚨',
            'question': '❓',
            'complaint': '😔',
            'standard': '📋'
        };
        badges += `<span class="badge badge-${result.intent}">${intentEmoji[result.intent]} ${result.intent}</span> `;
    }
    
    // Escalation badge
    if (result.should_escalate) {
        badges += `<span class="badge badge-danger">🚨 Escalated</span>`;
    }
    
    // Follow-up questions section (clickable!)
    let followupSection = '';
    if (result.follow_up_questions && result.follow_up_questions.length > 0) {
        followupSection = `
            <div class="follow-up-questions" style="margin-top: 1rem; padding: 0.75rem; background: #f0f9ff; border-left: 3px solid #2563eb; border-radius: 4px;">
                <strong>💬 Size sorularım var:</strong>
                ${result.follow_up_questions.map((q, idx) => `
                    <button class="followup-btn" onclick="answerFollowup('${q.replace(/'/g, "\\'")}')" 
                            style="display: block; margin-top: 0.5rem; padding: 0.5rem; width: 100%; text-align: left; 
                                   background: white; border: 1px solid #2563eb; border-radius: 4px; cursor: pointer;
                                   transition: all 0.2s;">
                        ${idx + 1}. ${q}
                    </button>
                `).join('')}
            </div>
        `;
    }
    
    // Escalation warning (if needed)
    let escalationSection = '';
    if (result.should_escalate && result.escalation_reason) {
        escalationSection = `
            <div class="escalation-warning" style="margin-top: 1rem; padding: 0.75rem; background: #fef2f2; 
                 border-left: 3px solid #dc2626; border-radius: 4px;">
                <strong>⚠️ Dikkat:</strong> ${result.escalation_reason}
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-content conversational-response">
            ${formattedResponse}
            ${escalationSection}
            ${followupSection}
            <div class="meta-info" style="margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #e0e0e0; font-size: 0.75rem; color: #666;">
                ${badges}
                <span style="margin-left: 0.5rem;">Model: ${capitalize(result.model_used)}</span> •
                <span>${result.timestamp}</span>
            </div>
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Keep old function for backwards compatibility
function addBotResponse(result) {
    // Fallback to conversational if response field exists
    if (result.response) {
        addConversationalResponse(result);
        return;
    }
    
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const confidence = (result.confidence * 100).toFixed(1);
    const categoryBadgeClass = getCategoryBadgeClass(result.category);
    
    // Get top 3 predictions
    const allPreds = result.all_predictions[result.model_used];
    const probs = allPreds.probabilities;
    const sortedProbs = Object.entries(probs)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <p><strong>✅ Ticket Classified!</strong></p>
            <div class="result-card">
                <div class="result-item">
                    <span class="result-label">Category:</span>
                    <span class="badge ${categoryBadgeClass}">${result.category}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Confidence:</span>
                    <span class="result-value">${confidence}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Model:</span>
                    <span class="result-value">${capitalize(result.model_used)}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Ticket ID:</span>
                    <span class="result-value">#${result.id}</span>
                </div>
            </div>
            <p style="margin-top: 1rem; font-size: 0.875rem;">
                <strong>Top 3 Predictions:</strong>
            </p>
            ${sortedProbs.map(([cat, prob]) => `
                <div class="result-item">
                    <span class="result-label">${cat}:</span>
                    <span class="result-value">${(prob * 100).toFixed(1)}%</span>
                </div>
            `).join('')}
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Load tickets from API
async function loadTickets() {
    try {
        const response = await fetch('/api/tickets');
        const data = await response.json();
        tickets = data.tickets;
        renderTicketsTable();
    } catch (error) {
        console.error('Error loading tickets:', error);
    }
}

// Load stats from API
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        stats = await response.json();
        updateStatsDisplay();
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Render tickets table
function renderTicketsTable() {
    const tbody = document.getElementById('ticketsTableBody');
    
    if (tickets.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-state">
                <td colspan="7">
                    <div class="empty-message">
                        <i class="fas fa-inbox"></i>
                        <p>No tickets classified yet</p>
                        <p class="hint">Use the chatbot below to classify your first ticket!</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = tickets.map(ticket => {
        const confidence = (ticket.confidence * 100).toFixed(1);
        const badgeClass = getCategoryBadgeClass(ticket.category);
        
        return `
            <tr>
                <td><strong>#${ticket.id}</strong></td>
                <td class="ticket-text" title="${escapeHtml(ticket.text)}">${escapeHtml(ticket.text)}</td>
                <td><span class="badge ${badgeClass}">${ticket.category}</span></td>
                <td>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                    <div class="confidence-text">${confidence}%</div>
                </td>
                <td>${capitalize(ticket.model_used)}</td>
                <td>${ticket.timestamp}</td>
                <td>
                    <button class="btn-action" onclick="viewDetails('${ticket.id}')" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

// Update stats display
function updateStatsDisplay() {
    document.getElementById('totalTickets').textContent = stats.total_tickets || 0;
    document.getElementById('totalTicketsCard').textContent = stats.total_tickets || 0;
    
    const avgConf = stats.average_confidence ? (stats.average_confidence * 100).toFixed(1) + '%' : '0%';
    document.getElementById('avgConfidence').textContent = avgConf;
    document.getElementById('avgConfidenceCard').textContent = avgConf;
    
    // Find most common category
    if (stats.categories && Object.keys(stats.categories).length > 0) {
        const topCat = Object.entries(stats.categories)
            .sort((a, b) => b[1] - a[1])[0][0];
        document.getElementById('topCategory').textContent = topCat;
    } else {
        document.getElementById('topCategory').textContent = '-';
    }
    
    // Count active models
    let activeModels = 0;
    if (stats.models_available) {
        if (stats.models_available.baseline) activeModels++;
        if (stats.models_available.lstm) activeModels++;
        if (stats.models_available.custom) activeModels++;
    }
    document.getElementById('modelsActive').textContent = activeModels || 3;
}

// View ticket details (could open modal)
function viewDetails(ticketId) {
    const ticket = tickets.find(t => t.id === ticketId);
    if (!ticket) return;
    
    alert(`Ticket #${ticketId}\n\nText: ${ticket.text}\n\nCategory: ${ticket.category}\nConfidence: ${(ticket.confidence * 100).toFixed(1)}%\nModel: ${ticket.model_used}\nTime: ${ticket.timestamp}`);
}

// Refresh tickets
async function refreshTickets() {
    await loadTickets();
    await loadStats();
    showToast('Refreshed successfully', 'success');
}

// Show loading overlay
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('show');
    } else {
        overlay.classList.remove('show');
    }
}

// Show toast notification
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Get category badge class
function getCategoryBadgeClass(category) {
    const normalized = category.toLowerCase().replace(/\s+/g, '-');
    const classMap = {
        'access': 'badge-access',
        'hardware': 'badge-hardware',
        'hr-support': 'badge-hr',
        'administrative-rights': 'badge-admin',
        'internal-project': 'badge-project',
        'miscellaneous': 'badge-misc',
        'purchase': 'badge-purchase',
        'storage': 'badge-storage'
    };
    return classMap[normalized] || 'badge-misc';
}

// Utility: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Utility: Capitalize
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Answer follow-up question (clicked from suggestion)
function answerFollowup(question) {
    const input = document.getElementById('chatInput');
    input.value = question;
    input.focus();
    
    // Optional: Auto-send after 1 second
    setTimeout(() => {
        if (input.value === question) {
            // User didn't modify, auto-send
            // sendMessage(); // Uncomment to auto-send
        }
    }, 1000);
}

// Reset conversation context
async function resetConversation() {
    try {
        await fetch('/api/reset_conversation', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        // Clear chat UI
        const messagesDiv = document.getElementById('chatMessages');
        messagesDiv.innerHTML = `
            <div class="welcome-message" style="text-align: center; padding: 2rem; color: #6b7280;">
                <h3>👋 Welcome!</h3>
                <p style="margin-top: 0.5rem;">I can help you classify IT support tickets.</p>
                <p>Just describe your issue below!</p>
                <p style="margin-top: 1rem; font-size: 0.875rem;">
                    Example: "I need access to SAP system"
                </p>
            </div>
        `;
        
        showToast('Conversation reset', 'success');
    } catch (error) {
        console.error('Error resetting conversation:', error);
    }
}

// Add simple button action styling
const style = document.createElement('style');
style.textContent = `
    .btn-action {
        background: transparent;
        border: none;
        color: var(--primary-color);
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 4px;
        transition: background 0.2s;
    }
    .btn-action:hover {
        background: var(--light-bg);
    }
`;
document.head.appendChild(style);

