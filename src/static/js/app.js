// Global state
let tickets = [];
let stats = {};
let charts = {};  // Store chart instances
let currentView = 'dashboard';  // Track current view
let currentMode = 'ai';  // 'classic' or 'ai'

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadTickets();
    loadStats();
    
    // Refresh every 10 seconds
    setInterval(() => {
        loadStats();
        if (currentView === 'analytics') {
            loadAnalytics();
        }
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

// Set mode (classic or AI)
function setMode(mode) {
    currentMode = mode;
    
    // Update button styles
    document.getElementById('modeClassic').classList.toggle('active', mode === 'classic');
    document.getElementById('modeAI').classList.toggle('active', mode === 'ai');
    
    // Update send button text
    document.getElementById('sendButtonText').textContent = 
        mode === 'ai' ? 'Classify with AI' : 'Classify';
    
    showToast(mode === 'ai' ? 'AI Mode Enabled' : 'Classic Mode', 'success');
}

// Send message and classify ticket
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    
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
        // Call API based on mode
        const endpoint = currentMode === 'ai' ? '/api/classify-with-ai' : '/api/classify';
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) {
            throw new Error('Classification failed');
        }
        
        const result = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add response based on mode
        if (currentMode === 'ai') {
            addAIResponse(result);
        } else {
            addBasicResponse(result);
        }
        
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

// Add AI-enhanced response
function addAIResponse(result) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const confidence = (result.confidence * 100).toFixed(1);
    const categoryBadgeClass = getCategoryBadgeClass(result.category);
    
    // Sentiment emoji
    const sentimentEmoji = {
        'positive': '😊',
        'negative': '😟',
        'neutral': '😐',
        'urgent': '🚨'
    };
    
    // Priority badge
    const priorityBadge = {
        'HIGH': '<span class="badge" style="background: #ef4444; color: white; font-weight: 700;">🔴 HIGH PRIORITY</span>',
        'MEDIUM': '<span class="badge" style="background: #f59e0b; color: white;">🟡 MEDIUM</span>',
        'LOW': '<span class="badge" style="background: #10b981; color: white;">🟢 LOW</span>'
    };
    
    // Build similar tickets section
    let similarSection = '';
    if (result.similar_tickets && result.similar_tickets.length > 0) {
        similarSection = `
            <div style="margin-top: 1rem; padding: 1rem; background: #f0f9ff; border-left: 3px solid #3b82f6; border-radius: 4px;">
                <strong>📋 Similar Past Tickets:</strong>
                ${result.similar_tickets.map((t, idx) => `
                    <div style="margin-top: 0.5rem; padding: 0.5rem; background: white; border-radius: 4px; font-size: 0.85rem;">
                        <div><strong>#${t.id}</strong> - ${t.category} (${t.similarity}% match)</div>
                        <div style="color: #6b7280; margin-top: 0.25rem;">${t.text}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <p><strong>✅ Ticket Analyzed with AI!</strong></p>
            
            <!-- Category & Confidence -->
            <div style="margin: 1rem 0; padding: 1rem; background: #f9fafb; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span><strong>Category:</strong></span>
                    <span class="badge ${categoryBadgeClass}">${result.category}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span><strong>Confidence:</strong></span>
                    <span style="font-weight: 700; color: #10b981;">${confidence}%</span>
                </div>
            </div>
            
            <!-- AI Analysis -->
            <div style="margin: 1rem 0; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; color: white;">
                <div style="font-weight: 700; margin-bottom: 0.75rem; font-size: 1.05rem;">🤖 AI Analysis</div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                    <div>
                        <div style="opacity: 0.9; font-size: 0.8rem;">Sentiment</div>
                        <div style="font-weight: 700; font-size: 1.1rem;">
                            ${sentimentEmoji[result.sentiment] || '😐'} ${result.sentiment.toUpperCase()}
                        </div>
                    </div>
                    
                    <div>
                        <div style="opacity: 0.9; font-size: 0.8rem;">Priority</div>
                        <div style="font-weight: 700; font-size: 1.1rem;">
                            ${result.priority}
                        </div>
                    </div>
                    
                    <div>
                        <div style="opacity: 0.9; font-size: 0.8rem;">Route To</div>
                        <div style="font-weight: 600;">👥 ${result.route_to}</div>
                    </div>
                    
                    <div>
                        <div style="opacity: 0.9; font-size: 0.8rem;">SLA</div>
                        <div style="font-weight: 600;">⏱️ ${result.sla_description}</div>
                    </div>
                </div>
                
                ${result.is_urgent ? '<div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(255,255,255,0.2); border-radius: 4px; font-weight: 600;">⚠️ URGENT TICKET - Escalated automatically!</div>' : ''}
            </div>
            
            ${similarSection}
            
            <div style="margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid #e5e7eb; font-size: 0.75rem; color: #6b7280;">
                Ticket ID: <strong>#${result.id}</strong> • 
                Model: ${result.model_used.toUpperCase()} • 
                ${result.timestamp}
            </div>
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Add basic response (classic mode)
function addBasicResponse(result) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const confidence = (result.confidence * 100).toFixed(1);
    const categoryBadgeClass = getCategoryBadgeClass(result.category);
    
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
                    <span class="result-label">Ticket ID:</span>
                    <span class="result-value">#${result.id}</span>
                </div>
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

// Switch between Dashboard and Analytics views
function switchView(view) {
    currentView = view;
    
    // Toggle views
    document.getElementById('dashboardView').style.display = view === 'dashboard' ? 'block' : 'none';
    document.getElementById('analyticsView').style.display = view === 'analytics' ? 'block' : 'none';
    
    // Toggle nav buttons
    document.getElementById('navDashboard').classList.toggle('active', view === 'dashboard');
    document.getElementById('navAnalytics').classList.toggle('active', view === 'analytics');
    
    // Load analytics if switching to analytics view
    if (view === 'analytics') {
        loadAnalytics();
    }
}

// Load Analytics Data
async function loadAnalytics() {
    try {
        const days = document.getElementById('dateRange')?.value || 7;
        
        // Load all analytics data in parallel
        const [trendsRes, categoriesRes, modelsRes, confidenceRes, summaryRes] = await Promise.all([
            fetch(`/api/analytics/trends?days=${days}`),
            fetch('/api/analytics/categories'),
            fetch('/api/analytics/models'),
            fetch('/api/analytics/confidence-distribution'),
            fetch('/api/analytics/summary')
        ]);
        
        const trends = await trendsRes.json();
        const categories = await categoriesRes.json();
        const models = await modelsRes.json();
        const confidence = await confidenceRes.json();
        const summary = await summaryRes.json();
        
        // Update summary stats
        updateAnalyticsSummary(summary);
        
        // Update charts
        updateTrendsChart(trends);
        updateCategoryChart(categories);
        updateModelChart(models);
        updateConfidenceChart(confidence);
        
        // Update insights
        updateInsights(summary, categories, models);
        
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

// Update Analytics Summary Cards
function updateAnalyticsSummary(summary) {
    document.getElementById('analyticsTotal').textContent = summary.total_tickets;
    document.getElementById('analyticsToday').textContent = summary.today_tickets;
    document.getElementById('analyticsConfidence').textContent = summary.avg_confidence + '%';
    document.getElementById('analyticsTopModel').textContent = summary.most_used_model;
    
    // Update trend indicators
    const trendTotal = document.getElementById('trendTotal');
    if (summary.growth_rate > 0) {
        trendTotal.className = 'stat-trend positive';
        trendTotal.innerHTML = `<i class="fas fa-arrow-up"></i><span>+${summary.growth_rate}% vs yesterday</span>`;
    } else if (summary.growth_rate < 0) {
        trendTotal.className = 'stat-trend negative';
        trendTotal.innerHTML = `<i class="fas fa-arrow-down"></i><span>${summary.growth_rate}% vs yesterday</span>`;
    } else {
        trendTotal.className = 'stat-trend';
        trendTotal.innerHTML = `<i class="fas fa-minus"></i><span>No change</span>`;
    }
}

// Update Trends Chart (Line Chart)
function updateTrendsChart(data) {
    const ctx = document.getElementById('trendsChart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (charts.trends) {
        charts.trends.destroy();
    }
    
    charts.trends = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Tickets Count',
                    data: data.counts,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y'
                },
                {
                    label: 'Avg. Confidence (%)',
                    data: data.avg_confidences,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 12,
                            weight: 600
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    borderColor: '#3b82f6',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Ticket Count',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Confidence (%)',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        drawOnChartArea: false
                    },
                    min: 0,
                    max: 100
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Update Category Chart (Doughnut Chart)
function updateCategoryChart(data) {
    const ctx = document.getElementById('categoryChart');
    if (!ctx) return;
    
    if (charts.category) {
        charts.category.destroy();
    }
    
    // Beautiful gradient colors
    const colors = [
        '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
        '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'
    ];
    
    charts.category = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.categories,
            datasets: [{
                data: data.counts,
                backgroundColor: colors,
                borderWidth: 3,
                borderColor: '#ffffff',
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'right',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 12,
                            weight: 600
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Update Model Performance Chart (Bar Chart)
function updateModelChart(data) {
    const ctx = document.getElementById('modelChart');
    if (!ctx) return;
    
    if (charts.model) {
        charts.model.destroy();
    }
    
    charts.model = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.models,
            datasets: [
                {
                    label: 'Avg. Confidence (%)',
                    data: data.avg_confidences,
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(139, 92, 246, 0.8)'
                    ],
                    borderColor: [
                        '#3b82f6',
                        '#10b981',
                        '#f59e0b',
                        '#8b5cf6'
                    ],
                    borderWidth: 2,
                    borderRadius: 8,
                    hoverBackgroundColor: [
                        'rgba(59, 130, 246, 1)',
                        'rgba(16, 185, 129, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(139, 92, 246, 1)'
                    ]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    callbacks: {
                        afterLabel: function(context) {
                            const index = context.dataIndex;
                            const usage = data.usage_counts[index];
                            return `Used: ${usage} times`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Confidence (%)',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Update Confidence Distribution Chart (Bar Chart - Histogram style)
function updateConfidenceChart(data) {
    const ctx = document.getElementById('confidenceChart');
    if (!ctx) return;
    
    if (charts.confidence) {
        charts.confidence.destroy();
    }
    
    charts.confidence = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.ranges,
            datasets: [{
                label: 'Number of Tickets',
                data: data.counts,
                backgroundColor: 'rgba(16, 185, 129, 0.7)',
                borderColor: '#10b981',
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Ticket Count',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Confidence Range',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Update Insights
function updateInsights(summary, categories, models) {
    // Update usage insight
    const usageInsight = document.getElementById('insightUsage');
    if (summary.growth_rate > 10) {
        usageInsight.textContent = `Ticket volume growing rapidly: +${summary.growth_rate}% vs yesterday!`;
    } else if (summary.growth_rate > 0) {
        usageInsight.textContent = `Ticket volume increasing: +${summary.growth_rate}% vs yesterday`;
    } else if (summary.growth_rate < -10) {
        usageInsight.textContent = `Ticket volume decreasing: ${summary.growth_rate}% vs yesterday`;
    } else {
        usageInsight.textContent = 'Ticket volume is steady';
    }
    
    // Update top category insight
    const topCatInsight = document.getElementById('insightTopCategory');
    if (summary.top_category !== 'N/A') {
        topCatInsight.textContent = `Most tickets are categorized as "${summary.top_category}"`;
    } else {
        topCatInsight.textContent = 'No tickets classified yet';
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

