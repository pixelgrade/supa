// Supa Terminal Dashboard
// ========================

const ADMIN_KEY_STORAGE = 'supa_admin_key';

// Greetings based on time of day
const GREETINGS = {
    morning: [
        "Good morning!",
        "Morning! Coffee ready?",
        "Rise and shine!",
        "Early bird gets the tickets!"
    ],
    afternoon: [
        "Good afternoon!",
        "How's the day going?",
        "Afternoon shift, let's go!",
        "Ready when you are!"
    ],
    evening: [
        "Good evening!",
        "Evening session!",
        "Burning the midnight oil?",
        "Still at it? Let's go!"
    ],
    generic: [
        "Ready when you are!",
        "Let's get to work!",
        "At your service!",
        "What's on the queue?",
        "Tickets await!",
        "Support mode: ON"
    ]
};

// Status words for different phases
const STATUS_WORDS = {
    fetch: ['Fetching', 'Retrieving', 'Loading', 'Grabbing'],
    think: ['Mulling', 'Pondering', 'Reasoning', 'Contemplating', 'Thinking'],
    search: ['Digging', 'Searching', 'Scanning', 'Exploring', 'Probing'],
    create: ['Crafting', 'Composing', 'Assembling', 'Forging', 'Building'],
    done: ['Done', 'Complete', 'Ready', 'Finished']
};

// Loading words for queue
const LOADING_WORDS = ['Loading', 'Fetching', 'Checking', 'Scanning', 'Syncing'];
let loadingInterval = null;


// Command history
let commandHistory = [];
let historyIndex = -1;

// Current run result (for post command)
let lastTestResult = null;

// Current draft being reviewed
let currentDraft = null;
let draftsQueue = [];

// ===================
// Auth Functions
// ===================

function getAdminKey() {
    return localStorage.getItem(ADMIN_KEY_STORAGE);
}

function setAdminKey(key) {
    localStorage.setItem(ADMIN_KEY_STORAGE, key);
}

function showAuthModal() {
    document.getElementById('auth-modal').showModal();
    document.getElementById('admin-key-input').focus();
}

function hideAuthModal() {
    document.getElementById('auth-modal').close();
}

function submitAdminKey() {
    const key = document.getElementById('admin-key-input').value;
    if (key) {
        setAdminKey(key);
        hideAuthModal();
        loadDashboard();
    }
}

// ===================
// API Functions
// ===================

async function apiCall(endpoint, options = {}) {
    const adminKey = getAdminKey();
    const headers = {
        'Content-Type': 'application/json',
        ...(adminKey && { 'X-Admin-Key': adminKey })
    };

    const response = await fetch(endpoint, { ...options, headers });

    if (response.status === 401) {
        showAuthModal();
        throw new Error('Unauthorized');
    }

    return response.json();
}

// ===================
// Status Line (in terminal history)
// ===================

let currentLogEntry = null;

function getRandomWord(phase) {
    const words = STATUS_WORDS[phase] || STATUS_WORDS.think;
    return words[Math.floor(Math.random() * words.length)];
}

function startCommand(command) {
    // Create a new log entry for this command
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <div class="log-command">
            <span class="log-prefix">❯</span>
            <span class="log-text">${escapeHtml(command)}</span>
        </div>
        <div class="log-status">
            <span class="status-icon">●</span>
            <span class="status-msg">Starting...</span>
        </div>
        <div class="log-result"></div>
    `;

    const container = document.getElementById('terminal-content');
    const inputLine = document.getElementById('input-line');
    container.insertBefore(entry, inputLine);
    container.scrollTop = container.scrollHeight;

    currentLogEntry = entry;
}

function updateStatus(phase, suffix = '...') {
    if (!currentLogEntry) return;

    const statusDiv = currentLogEntry.querySelector('.log-status');
    if (statusDiv) {
        statusDiv.innerHTML = `
            <span class="status-icon">●</span>
            <span class="status-msg">${getRandomWord(phase)}${suffix}</span>
        `;
    }
}

function finishCommand(resultHtml) {
    if (!currentLogEntry) return;

    // Update status to done
    const statusDiv = currentLogEntry.querySelector('.log-status');
    if (statusDiv) {
        statusDiv.innerHTML = `
            <span class="status-icon done">✓</span>
            <span class="status-msg done">${getRandomWord('done')}.</span>
        `;
    }

    // Add result
    const resultDiv = currentLogEntry.querySelector('.log-result');
    if (resultDiv) {
        resultDiv.innerHTML = resultHtml;
    }

    const container = document.getElementById('terminal-content');
    container.scrollTop = container.scrollHeight;

    currentLogEntry = null;
}

function finishCommandWithError(errorHtml) {
    if (!currentLogEntry) return;

    // Update status to error
    const statusDiv = currentLogEntry.querySelector('.log-status');
    if (statusDiv) {
        statusDiv.innerHTML = `
            <span class="status-icon error">✗</span>
            <span class="status-msg error">Failed.</span>
        `;
    }

    // Add error result
    const resultDiv = currentLogEntry.querySelector('.log-result');
    if (resultDiv) {
        resultDiv.innerHTML = errorHtml;
    }

    const container = document.getElementById('terminal-content');
    container.scrollTop = container.scrollHeight;

    currentLogEntry = null;
}

// ===================
// Terminal Output Functions
// ===================

function clearTerminal() {
    const container = document.getElementById('terminal-content');
    const inputLine = document.getElementById('input-line');
    // Remove all children except the input line
    while (container.firstChild && container.firstChild !== inputLine) {
        container.removeChild(container.firstChild);
    }
    lastTestResult = null;
    currentLogEntry = null;
}

function addSimpleLog(command, resultHtml) {
    // For simple commands that don't need status updates
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <div class="log-command">
            <span class="log-prefix">❯</span>
            <span class="log-text">${escapeHtml(command)}</span>
        </div>
        <div class="log-result">${resultHtml}</div>
    `;

    const container = document.getElementById('terminal-content');
    const inputLine = document.getElementById('input-line');
    container.insertBefore(entry, inputLine);
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function confidenceClass(draft) {
    if (draft.confidence_status === 'manual_review') return 'manual';
    if ((draft.confidence_score || 0) >= 9) return 'high';
    if ((draft.confidence_score || 0) >= 8) return 'medium';
    return 'low';
}

function renderConfidenceBadge(draft) {
    if (draft.confidence_score == null) return '';
    const label = `${draft.confidence_score}/10`;
    return `<span class="confidence-badge ${confidenceClass(draft)}">${label}</span>`;
}

// ===================
// Result Renderers
// ===================

function renderTicketResult(data) {
    const classification = data.classification || 'unknown';
    const ticket = data.ticket || {};
    const theme = ticket.theme_tag || 'not detected';
    const customerName = ticket.customer_name ? ticket.customer_name.split(' ')[0] : 'Unknown';
    const newSubject = data.new_subject;
    const confidenceBadge = renderConfidenceBadge(data);
    const confidenceSummary = data.confidence_summary ? escapeHtml(data.confidence_summary) : '';
    const deliveryMode = data.delivery_mode ? escapeHtml(data.delivery_mode) : '';
    const providerSummary = [data.investigator_provider, data.evaluator_provider].filter(Boolean).join(' / ');
    const evidenceSummary = data.evidence_summary ? escapeHtml(data.evidence_summary) : '';
    const decisionNote = data.decision_note ? escapeHtml(data.decision_note) : '';
    const riskFlags = Array.isArray(data.risk_flags) ? data.risk_flags.filter(Boolean) : [];
    const sources = Array.isArray(data.sources) ? data.sources.filter(Boolean) : [];

    // Knowledge matches
    let knowledgeHtml = '';
    if (data.knowledge_results && data.knowledge_results.length > 0) {
        knowledgeHtml = `
            <ul class="match-list">
                ${data.knowledge_results.map((k, i) => {
                    const title = escapeHtml(k.title || 'Document');
                    const url = k.metadata?.url || k.url;
                    const titleHtml = url
                        ? `<a href="${escapeHtml(url)}" target="_blank" class="match-link">${title}</a>`
                        : `<span class="match-title">${title}</span>`;
                    return `
                        <li class="match-item">
                            <span class="match-num">${i + 1}.</span>
                            ${titleHtml}
                            <span class="match-score">${(k.score || 0).toFixed(2)}</span>
                        </li>
                    `;
                }).join('')}
            </ul>
        `;
    } else {
        knowledgeHtml = '<p class="no-matches">No knowledge matches found</p>';
    }

    // Similar tickets
    let ticketsHtml = '';
    if (data.similar_tickets && data.similar_tickets.length > 0) {
        ticketsHtml = `
            <ul class="match-list">
                ${data.similar_tickets.map((t, i) => {
                    const subject = escapeHtml(t.subject || 'Ticket');
                    const ticketUrl = t.id ? `${document.documentElement.dataset.mailboxWeb}/conversation/${encodeURIComponent(t.id)}/` : null;
                    const titleHtml = ticketUrl
                        ? `<a href="${ticketUrl}" target="_blank" class="match-link">${subject}</a>`
                        : `<span class="match-title">${subject}</span>`;
                    return `
                        <li class="match-item">
                            <span class="match-num">${i + 1}.</span>
                            ${titleHtml}
                        </li>
                    `;
                }).join('')}
            </ul>
        `;
    } else {
        ticketsHtml = '<p class="no-matches">No similar tickets found</p>';
    }

    // Response preview
    let responseHtml = '';
    if (data.response_html) {
        responseHtml = `
            <div class="result-section">
                <div class="section-title">Generated Response</div>
                ${confidenceBadge || confidenceSummary ? `
                <div class="confidence-row">
                    ${confidenceBadge}
                    ${confidenceSummary ? `<span class="confidence-summary">${confidenceSummary}</span>` : ''}
                </div>
                ` : ''}
                <div class="response-preview">${data.response_html}</div>
            </div>
        `;
    }

    // Action buttons (only if not posted and has response)
    let actionsHtml = '';
    if (data.response_html && !data.posted) {
        actionsHtml = `
            <div class="result-actions">
                <button class="btn btn-primary" onclick="postLastResult()">Post to HelpScout</button>
                <button class="btn btn-secondary" onclick="clearTerminal()">Clear</button>
            </div>
        `;
    }

    // Badge
    const badgeClass = data.posted ? 'posted' : classification;
    const badgeText = data.posted ? 'Posted' : classification;

    return `
        <div class="result-block">
            <div class="result-header">
                <span class="result-title">Ticket Analysis</span>
                <span class="result-badge ${badgeClass}">${badgeText}</span>
            </div>
            <div class="result-body">
                <div class="result-field">
                    <span class="field-label">Customer</span>
                    <span class="field-value">${escapeHtml(customerName)}</span>
                </div>
                <div class="result-field">
                    <span class="field-label">Theme</span>
                    <span class="field-value">${escapeHtml(theme)}</span>
                </div>
                <div class="result-field">
                    <span class="field-label">Subject</span>
                    <span class="field-value">${escapeHtml(ticket.subject || 'No subject')}</span>
                </div>
                ${newSubject ? `
                <div class="result-field">
                    <span class="field-label">New Subject</span>
                    <span class="field-value new-subject">${escapeHtml(newSubject)}</span>
                </div>
                ` : ''}
                ${deliveryMode ? `
                <div class="result-field">
                    <span class="field-label">Decision Preview</span>
                    <span class="field-value">${deliveryMode}</span>
                </div>
                ` : ''}
                ${providerSummary ? `
                <div class="result-field">
                    <span class="field-label">Providers</span>
                    <span class="field-value">${escapeHtml(providerSummary)}</span>
                </div>
                ` : ''}

                <div class="result-section">
                    <div class="section-title">Knowledge Matches</div>
                    ${knowledgeHtml}
                </div>

                <div class="result-section">
                    <div class="section-title">Similar Tickets</div>
                    ${ticketsHtml}
                </div>

                ${evidenceSummary ? `
                <div class="result-section">
                    <div class="section-title">Evidence Summary</div>
                    <div class="confidence-summary-block">${evidenceSummary}</div>
                </div>
                ` : ''}
                ${riskFlags.length ? `
                <div class="result-section">
                    <div class="section-title">Risk Flags</div>
                    <div class="confidence-summary-block">${escapeHtml(riskFlags.join(', '))}</div>
                </div>
                ` : ''}
                ${sources.length ? `
                <div class="result-section">
                    <div class="section-title">Sources</div>
                    <ul class="match-list">
                        ${sources.map((source, i) => `
                            <li class="match-item">
                                <span class="match-num">${i + 1}.</span>
                                <span class="match-title">${escapeHtml(source)}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                ` : ''}
                ${decisionNote ? `
                <div class="result-section">
                    <div class="section-title">Decision Note</div>
                    <div class="confidence-summary-block">${decisionNote.replace(/\n/g, '<br>')}</div>
                </div>
                ` : ''}

                ${responseHtml}
                ${actionsHtml}
            </div>
        </div>
    `;
}

function renderSearchResults(results) {
    if (!results || results.length === 0) {
        return `
            <div class="result-block">
                <div class="result-header">
                    <span class="result-title">Search Results</span>
                </div>
                <div class="result-body">
                    <p class="no-matches">No results found</p>
                </div>
            </div>
        `;
    }

    return `
        <div class="result-block">
            <div class="result-header">
                <span class="result-title">Search Results</span>
                <span class="result-badge support">${results.length} found</span>
            </div>
            <div class="result-body">
                <ul class="match-list">
                    ${results.map((r, i) => `
                        <li class="match-item">
                            <span class="match-num">${i + 1}.</span>
                            <span class="match-title">${escapeHtml(r.title || r.id || 'Document')}</span>
                            <span class="match-score">${(r.score || 0).toFixed(2)}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        </div>
    `;
}

function renderStats(stats) {
    return `
        <div class="result-block">
            <div class="result-header">
                <span class="result-title">Today's Statistics</span>
            </div>
            <div class="result-body">
                <div class="result-field">
                    <span class="field-label">Total</span>
                    <span class="field-value">${stats.total || 0}</span>
                </div>
                <div class="result-field">
                    <span class="field-label">Support</span>
                    <span class="field-value">${stats.support || 0}</span>
                </div>
                <div class="result-field">
                    <span class="field-label">Marketing</span>
                    <span class="field-value">${stats.marketing || 0}</span>
                </div>
                <div class="result-field">
                    <span class="field-label">Spam</span>
                    <span class="field-value">${stats.spam || 0}</span>
                </div>
            </div>
        </div>
    `;
}

function renderError(message) {
    return `<div class="error-message">Error: ${escapeHtml(message)}</div>`;
}

function renderSimple(message) {
    return `
        <div class="result-block">
            <div class="result-body">
                <p>${escapeHtml(message)}</p>
            </div>
        </div>
    `;
}

function renderPostConfirmation(message) {
    return `<div class="post-confirmation">${escapeHtml(message)}</div>`;
}

function renderDraftReview(draft) {
    const customerName = escapeHtml(draft.customer_name || 'Customer');
    const customerEmail = escapeHtml(draft.customer_email || '');
    const subject = escapeHtml(draft.subject || 'No subject');
    const theme = escapeHtml(draft.theme || 'not detected');
    const customerMessage = escapeHtml(draft.customer_message || 'No message content');
    const draftContent = draft.draft_content || '';
    const confidenceBadge = renderConfidenceBadge(draft);
    const confidenceSummary = draft.confidence_summary ? escapeHtml(draft.confidence_summary) : '';
    const confidenceAttempts = draft.confidence_attempts || 1;

    return `
        <div class="result-block draft-review">
            <div class="result-header">
                <span class="result-title">Draft Review</span>
                <span class="result-badge support">${theme}</span>
            </div>
            <div class="result-body">
                <div class="result-field">
                    <span class="field-label">From</span>
                    <span class="field-value">${customerName} &lt;${customerEmail}&gt;</span>
                </div>
                <div class="result-field">
                    <span class="field-label">Subject</span>
                    <span class="field-value">${subject}</span>
                </div>
                ${confidenceBadge ? `
                <div class="result-field">
                    <span class="field-label">Confidence</span>
                    <span class="field-value confidence-field">
                        ${confidenceBadge}
                        <span class="confidence-attempts">attempts: ${confidenceAttempts}</span>
                    </span>
                </div>
                ` : ''}
                ${confidenceSummary ? `
                <div class="result-section">
                    <div class="section-title">Confidence Summary</div>
                    <div class="confidence-summary-block">${confidenceSummary}</div>
                </div>
                ` : ''}

                <div class="result-section">
                    <div class="section-title">Customer Message</div>
                    <div class="customer-message">${customerMessage}</div>
                </div>

                <div class="result-section">
                    <div class="section-title">Draft Reply</div>
                    <div class="draft-editor-container">
                        <div class="draft-content" id="draft-content" contenteditable="true">${draftContent}</div>
                    </div>
                </div>

                <div class="result-actions draft-actions">
                    <button class="btn btn-primary" onclick="sendDraftAsPending()">Send as Pending</button>
                    <button class="btn" onclick="skipDraft()">Skip</button>
                    <a href="${escapeHtml(draft.url)}" target="_blank" class="btn">Edit in HelpScout</a>
                </div>
            </div>
        </div>
    `;
}

function renderHelp() {
    return `
        <div class="help-output">
            <div class="help-line"><span class="help-cmd">run &lt;id&gt;</span> Run a HelpScout ticket (dry run)</div>
            <div class="help-line"><span class="help-cmd">test &lt;id&gt;</span> Alias for <code>run &lt;id&gt;</code></div>
            <div class="help-line"><span class="help-cmd">post</span> Post last run result to HelpScout</div>
            <div class="help-line"><span class="help-cmd">list &lt;type&gt;</span> List tickets (received, auto_closed, drafts, manual)</div>
            <div class="help-line"><span class="help-cmd">search &lt;query&gt;</span> Search the knowledge base</div>
            <div class="help-line"><span class="help-cmd">stats</span> Show today's statistics</div>
            <div class="help-line"><span class="help-cmd">refresh</span> Re-index knowledge base</div>
            <div class="help-line"><span class="help-cmd">reprocess</span> Find tickets stranded by a provider outage (add <code>send</code> to recover)</div>
            <div class="help-line"><span class="help-cmd">clear</span> Clear the terminal</div>
            <div class="help-line"><span class="help-cmd">?</span> Show this help</div>
        </div>
    `;
}

// ===================
// Command Handling
// ===================

function extractConversationId(input) {
    const urlMatch = input.match(/conversation\/(\d+)/);
    if (urlMatch) return urlMatch[1];
    if (/^\d+$/.test(input.trim())) return input.trim();
    return null;
}

async function executeCommand(command) {
    const parts = command.trim().split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1).join(' ');

    // Add to history
    if (command.trim()) {
        commandHistory.unshift(command);
        if (commandHistory.length > 50) commandHistory.pop();
    }
    historyIndex = -1;

    switch (cmd) {
        case 'help':
        case '?':
            addSimpleLog(command, renderHelp());
            break;

        case 'clear':
        case 'cls':
            clearTerminal();
            break;

        case 'run':
        case 'test':
            await runTicket(command, args);
            break;

        case 'post':
            await postLastResult();
            break;

        case 'search':
            await searchKnowledge(command, args);
            break;

        case 'stats':
            await showStats(command);
            break;

        case 'refresh':
            await refreshKnowledge(command);
            break;

        case 'index':
            await indexDocs(command);
            break;

        case 'list':
            await listTickets(command, args);
            break;

        case 'reprocess':
        case 'recover':
            await reprocessStuck(command, args.trim() === 'send');
            break;

        default:
            // If it looks like a ticket ID/URL, treat as run
            if (/^\d+$/.test(cmd) || cmd.includes('helpscout')) {
                await runTicket(command, command);
            } else if (cmd) {
                addSimpleLog(command, renderError(`Unknown command: ${cmd}`));
            }
    }
}

async function runTicket(command, input) {
    const conversationId = extractConversationId(input);
    if (!conversationId) {
        addSimpleLog(command, renderError('Please provide a valid HelpScout ticket ID or URL'));
        return;
    }

    startCommand(command);

    try {
        updateStatus('fetch', ' ticket data...');
        await sleep(300);

        updateStatus('think', ' over the ticket...');

        const response = await fetch('/dashboard/test-ticket', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Admin-Key': getAdminKey()
            },
            body: JSON.stringify({
                conversation_id: parseInt(conversationId)
            })
        });

        if (response.status === 401) {
            finishCommandWithError(renderError('Unauthorized'));
            showAuthModal();
            return;
        }

        updateStatus('search', ' through knowledge...');
        await sleep(200);

        if (!response.ok) {
            let errorMsg = `HTTP ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.detail || errorMsg;
            } catch (e) {
                // Response wasn't JSON
                const text = await response.text().catch(() => '');
                if (text) errorMsg = text.substring(0, 100);
            }
            finishCommandWithError(renderError(errorMsg));
            return;
        }

        const data = await response.json();

        updateStatus('create', ' response...');
        await sleep(300);

        // Store full result for post command
        lastTestResult = {
            conversationId: parseInt(conversationId),
            customerId: data.ticket?.customer_id,
            responseHtml: data.response_html,
            newSubject: data.new_subject,
            themeTag: data.ticket?.theme_tag,
            existingTags: data.ticket?.tags || [],
            classification: data.classification,
            subject: data.ticket?.subject,
            customerName: data.ticket?.customer_name,
            confidenceScore: data.confidence_score,
            confidenceStatus: data.confidence_status,
            confidenceAttempts: data.confidence_attempts,
            confidenceSummary: data.confidence_summary,
            confidenceTrace: data.confidence_trace || [],
            confidenceProvider: data.confidence_provider,
            evidenceSummary: data.evidence_summary,
            deliveryMode: data.delivery_mode,
            investigatorProvider: data.investigator_provider,
            evaluatorProvider: data.evaluator_provider,
            decisionNote: data.decision_note,
            investigationSteps: data.investigation_steps || [],
            riskFlags: data.risk_flags || [],
            sources: data.sources || [],
            data: data
        };

        finishCommand(renderTicketResult(data));

    } catch (error) {
        finishCommandWithError(renderError(error.message));
    }
}

async function postLastResult() {
    if (!lastTestResult) {
        addSimpleLog('post', renderError('No run result to post. Run "run <id>" first.'));
        return;
    }

    if (!lastTestResult.responseHtml) {
        addSimpleLog('post', renderError('No response to post (ticket may be spam/marketing).'));
        return;
    }

    startCommand('post');

    try {
        updateStatus('create', ' posting to HelpScout...');

        const response = await fetch('/dashboard/post-cached', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Admin-Key': getAdminKey()
            },
            body: JSON.stringify({
                conversation_id: lastTestResult.conversationId,
                customer_id: lastTestResult.customerId,
                response_html: lastTestResult.responseHtml,
                new_subject: lastTestResult.newSubject,
                theme_tag: lastTestResult.themeTag,
                existing_tags: lastTestResult.existingTags,
                classification: lastTestResult.classification,
                subject: lastTestResult.subject,
                customer_name: lastTestResult.customerName,
                confidence_score: lastTestResult.confidenceScore,
                confidence_status: lastTestResult.confidenceStatus,
                confidence_attempts: lastTestResult.confidenceAttempts,
                confidence_summary: lastTestResult.confidenceSummary,
                confidence_trace: lastTestResult.confidenceTrace,
                confidence_provider: lastTestResult.confidenceProvider,
                evidence_summary: lastTestResult.evidenceSummary,
                delivery_mode: lastTestResult.deliveryMode,
                investigator_provider: lastTestResult.investigatorProvider,
                evaluator_provider: lastTestResult.evaluatorProvider,
                decision_note: lastTestResult.decisionNote,
                investigation_steps: lastTestResult.investigationSteps,
                risk_flags: lastTestResult.riskFlags,
                sources: lastTestResult.sources
            })
        });

        if (response.status === 401) {
            finishCommandWithError(renderError('Unauthorized'));
            showAuthModal();
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            finishCommandWithError(renderError(result.detail || `HTTP ${response.status}`));
            return;
        }

        // Show simple confirmation
        const subject = lastTestResult.data?.ticket?.subject || 'ticket';
        const newSubject = lastTestResult.newSubject;

        let message = `Done! I've posted a draft reply for "${subject}".`;
        if (newSubject) {
            message += ` Also updated the subject to "${newSubject}".`;
        }
        message += ` You can review and send it from HelpScout.`;

        lastTestResult = null;
        finishCommand(renderPostConfirmation(message));

    } catch (error) {
        finishCommandWithError(renderError(error.message));
    }
}

async function searchKnowledge(command, query) {
    if (!query) {
        addSimpleLog(command, renderError('Please provide a search query'));
        return;
    }

    startCommand(command);

    try {
        updateStatus('search', ` for "${query}"...`);

        const response = await apiCall(`/dashboard/search?q=${encodeURIComponent(query)}`);

        finishCommand(renderSearchResults(response.results || response));

    } catch (error) {
        if (error.message !== 'Unauthorized') {
            finishCommandWithError(renderError(error.message));
        }
    }
}

async function showStats(command) {
    startCommand(command);

    try {
        updateStatus('fetch', ' statistics...');

        const data = await apiCall('/dashboard/stats');

        finishCommand(renderStats(data));

    } catch (error) {
        if (error.message !== 'Unauthorized') {
            finishCommandWithError(renderError(error.message));
        }
    }
}

async function reprocessStuck(command, send) {
    startCommand(command || (send ? 'reprocess send' : 'reprocess'));
    try {
        updateStatus(send ? 'create' : 'search',
                     send ? ' recovering stranded tickets...' : ' scanning for stranded tickets...');
        const result = await apiCall('/dashboard/reprocess-stuck', {
            method: 'POST',
            body: JSON.stringify({ send: !!send })
        });
        finishCommand(renderReprocess(result));
        loadDashboard();
    } catch (error) {
        if (error.message !== 'Unauthorized') {
            finishCommandWithError(renderError(error.message));
        }
    }
}

function renderReprocess(data) {
    const results = data.results || [];
    if (!data.count) {
        return renderSimple('No tickets stranded by a provider outage. Nothing to recover. ✅');
    }
    const rows = results.map(r => {
        let detail;
        if (r.action === 'sent') detail = `<span class="conf-high">sent ${r.new_score}/10</span>`;
        else if (r.action === 'drafted') detail = `drafted ${r.new_score}/10`;
        else if (r.action === 'dry_run') detail = `would ${r.would} (${r.new_score}/10)`;
        else if (r.action === 'failed') detail = `<span class="conf-low">failed: ${escapeHtml(r.reason || '')}</span>`;
        else detail = `skipped — ${escapeHtml(r.reason || '')}`;
        return `<div class="help-line"><span class="help-cmd">#${r.conversation_id}</span> ${escapeHtml((r.subject || '').slice(0, 48))} — ${detail}</div>`;
    }).join('');

    const actionable = results.filter(r => r.action === 'dry_run').length;
    const button = (!data.send && actionable)
        ? `<div class="result-actions" style="margin-top:10px">
             <button class="btn btn-primary" onclick="reprocessStuck('reprocess send', true)">Recover all (${actionable})</button>
           </div>`
        : '';
    const header = data.send
        ? `Recovered ${results.filter(r => r.action === 'sent' || r.action === 'drafted').length}/${data.count} stranded ticket(s):`
        : `Found ${data.count} stranded ticket(s) (dry run):`;
    return `<div class="help-output"><div style="margin-bottom:6px">${header}</div>${rows}${button}</div>`;
}

async function refreshKnowledge(command) {
    startCommand(command);

    try {
        updateStatus('search', ' and indexing examples...');

        const result = await apiCall('/dashboard/refresh-knowledge', { method: 'POST' });

        finishCommand(renderSimple(`Indexed ${result.indexed} examples`));
        loadDashboard();

    } catch (error) {
        if (error.message !== 'Unauthorized') {
            finishCommandWithError(renderError(error.message));
        }
    }
}

async function indexDocs(command) {
    startCommand(command);

    try {
        updateStatus('search', ' and indexing docs...');

        const result = await apiCall('/dashboard/index-docs', { method: 'POST' });

        finishCommand(renderSimple(`Indexed ${result.indexed} documents`));
        loadDashboard();

    } catch (error) {
        if (error.message !== 'Unauthorized') {
            finishCommandWithError(renderError(error.message));
        }
    }
}

// ===================
// List Tickets Command
// ===================

async function listTickets(command, typeArg) {
    const validTypes = ['received', 'auto_closed', 'drafts', 'manual'];
    const ticketType = typeArg?.toLowerCase() || 'received';

    if (!validTypes.includes(ticketType)) {
        addSimpleLog(command, renderError(`Invalid type. Use: ${validTypes.join(', ')}`));
        return;
    }

    startCommand(command);

    try {
        updateStatus('fetch', ` ${ticketType} tickets...`);

        const data = await apiCall(`/dashboard/list/${ticketType}`);

        finishCommand(renderTicketList(data));

    } catch (error) {
        if (error.message !== 'Unauthorized') {
            finishCommandWithError(renderError(error.message));
        }
    }
}

function renderTicketList(data) {
    const typeLabels = {
        received: 'All Received',
        auto_closed: 'Auto-Closed',
        drafts: 'Drafts Created',
        manual: 'Manual Review'
    };

    const title = typeLabels[data.type] || data.type;

    if (!data.tickets || data.tickets.length === 0) {
        return `
            <div class="result-block">
                <div class="result-header">
                    <span class="result-title">${escapeHtml(title)}</span>
                    <span class="result-badge">0 tickets</span>
                </div>
                <div class="result-body">
                    <p class="no-matches">No tickets in the last 7 days</p>
                </div>
            </div>
        `;
    }

    const ticketsHtml = data.tickets.map(t => {
        const statusClass = t.response_posted ? 'posted' : (t.classification || 'support');
        const ticketUrl = `${document.documentElement.dataset.mailboxWeb}/conversation/${encodeURIComponent(t.id)}/`;
        return `
            <li class="ticket-item">
                <a href="${ticketUrl}" target="_blank" class="ticket-link">
                    <span class="ticket-customer">${escapeHtml(t.customer)}</span>
                    <span class="ticket-subject">${escapeHtml(t.subject)}</span>
                    <span class="ticket-badge ${statusClass}">${escapeHtml(t.classification)}</span>
                </a>
            </li>
        `;
    }).join('');

    return `
        <div class="result-block">
            <div class="result-header">
                <span class="result-title">${escapeHtml(title)}</span>
                <span class="result-badge support">${data.count} tickets</span>
            </div>
            <div class="result-body">
                <ul class="ticket-list">
                    ${ticketsHtml}
                </ul>
            </div>
        </div>
    `;
}

// ===================
// Typing Animation
// ===================

async function typeCommand(command) {
    const input = document.getElementById('command-input');
    input.value = '';
    input.focus();

    for (const char of command) {
        input.value += char;
        await sleep(30 + Math.random() * 20);  // 30-50ms per character
    }

    await sleep(100);
    input.value = '';
    executeCommand(command);
}

// ===================
// Draft Review Functions
// ===================

async function loadDrafts() {
    try {
        const drafts = await apiCall('/dashboard/drafts');
        draftsQueue = drafts;
        return drafts;
    } catch (error) {
        console.error('Failed to load drafts:', error);
        return [];
    }
}

async function openDraft(draftId) {
    startCommand(`review ${draftId}`);

    try {
        updateStatus('fetch', ' draft details...');

        // Load full drafts if not already loaded
        if (draftsQueue.length === 0) {
            await loadDrafts();
        }

        // Find the draft
        const draft = draftsQueue.find(d => d.id === draftId);
        if (!draft) {
            finishCommandWithError(renderError('Draft not found'));
            return;
        }

        currentDraft = draft;
        finishCommand(renderDraftReview(draft));

    } catch (error) {
        finishCommandWithError(renderError(error.message));
    }
}

async function sendDraftAsPending() {
    if (!currentDraft) {
        addSimpleLog('send', renderError('No draft selected'));
        return;
    }

    // Get potentially edited content
    const contentEl = document.getElementById('draft-content');
    const replyHtml = contentEl ? contentEl.innerHTML : currentDraft.draft_content;

    startCommand('send');

    try {
        updateStatus('create', ' sending reply...');

        const response = await fetch('/dashboard/send-reply', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Admin-Key': getAdminKey()
            },
            body: JSON.stringify({
                conversation_id: currentDraft.id,
                reply_html: replyHtml,
                draft_thread_id: currentDraft.draft_thread_id,
                theme_tag: currentDraft.theme,
                existing_tags: currentDraft.tags || []
            })
        });

        if (response.status === 401) {
            finishCommandWithError(renderError('Unauthorized'));
            showAuthModal();
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            finishCommandWithError(renderError(result.detail || `HTTP ${response.status}`));
            return;
        }

        const subject = currentDraft.subject || 'ticket';
        finishCommand(renderPostConfirmation(`Sent! Reply posted for "${subject}" and set to Pending.`));

        // Remove from queue and show next
        draftsQueue = draftsQueue.filter(d => d.id !== currentDraft.id);
        currentDraft = null;

        // Refresh dashboard
        loadDashboard();

        // Auto-load next draft if available
        if (draftsQueue.length > 0) {
            setTimeout(() => openDraft(draftsQueue[0].id), 500);
        }

    } catch (error) {
        finishCommandWithError(renderError(error.message));
    }
}

function skipDraft() {
    if (!currentDraft) return;

    const skippedId = currentDraft.id;
    addSimpleLog('skip', renderSimple(`Skipped "${currentDraft.subject}"`));

    // Move to end of queue
    draftsQueue = draftsQueue.filter(d => d.id !== skippedId);
    draftsQueue.push(currentDraft);
    currentDraft = null;

    // Load next draft
    if (draftsQueue.length > 0 && draftsQueue[0].id !== skippedId) {
        openDraft(draftsQueue[0].id);
    }
}

// ===================
// Dashboard Loading
// ===================

function renderQueueGroup(title, items, clickable = false) {
    if (!items || items.length === 0) {
        return '';
    }

    const itemsHtml = items.map(item => {
        const theme = item.theme ? `<span class="theme">${escapeHtml(item.theme)}:</span> ` : '';
        const subject = escapeHtml(item.subject || 'No subject');
        const confidence = renderConfidenceBadge(item);

        if (clickable) {
            // Clickable draft - opens in Supa for review
            return `<a href="#" class="queue-item" onclick="openDraft(${item.id}); return false;">${theme}${subject}${confidence}</a>`;
        } else {
            // Link to HelpScout
            return `<a href="${escapeHtml(item.url)}" target="_blank" class="queue-item">${theme}${subject}${confidence}</a>`;
        }
    }).join('');

    return `
        <div class="queue-group">
            <div class="queue-group-title">${title}</div>
            ${itemsHtml}
        </div>
    `;
}

function renderQueue(data) {
    const summaryEl = document.getElementById('queue-summary');
    const listEl = document.getElementById('queue-list');
    const statsEl = document.getElementById('stats-summary');

    // Summary line
    const draftsCount = data.drafts?.length || 0;
    const followUpsCount = data.follow_ups?.length || 0;

    const draftsClass = draftsCount === 0 ? 'zero' : '';
    const followUpsClass = followUpsCount === 0 ? 'zero' : '';

    const warning = data.warning;
    const warningHtml = warning ? `
        <div class="queue-warning">
            <div class="queue-warning-title">${escapeHtml(warning.message)}</div>
            ${warning.detail ? `<div class="queue-warning-detail">${escapeHtml(warning.detail)}</div>` : ''}
        </div>
    ` : '';

    summaryEl.innerHTML = `
        ${warningHtml}
        <span class="count ${draftsClass}">${draftsCount}</span> drafts to review ·
        <span class="count ${followUpsClass}">${followUpsCount}</span> follow-ups
    `;

    // Queue lists - drafts are clickable for in-Supa review
    let listHtml = '';
    listHtml += renderQueueGroup('Drafts', data.drafts, true);  // clickable
    listHtml += renderQueueGroup('Follow-ups', data.follow_ups, false);  // link to HelpScout

    if (!listHtml) {
        listHtml = '<div class="queue-empty">All clear</div>';
    }

    listEl.innerHTML = listHtml;

    // Stats summary (7 days) - clickable
    if (data.stats) {
        const s = data.stats;
        statsEl.innerHTML = `Last 7 days:
            <a href="#" class="stat-link" onclick="typeCommand('list received'); return false;"><span class="num">${s.received}</span> received</a> ·
            <a href="#" class="stat-link" onclick="typeCommand('list auto_closed'); return false;"><span class="num">${s.auto_closed}</span> auto-closed</a> ·
            <a href="#" class="stat-link" onclick="typeCommand('list drafts'); return false;"><span class="num">${s.drafts_created}</span> drafts</a> ·
            <a href="#" class="stat-link" onclick="typeCommand('list manual'); return false;"><span class="num">${s.manual}</span> manual</a>`;
    }
}

function startLoadingAnimation() {
    const wordEl = document.getElementById('loading-word');
    if (!wordEl) return;

    let index = 0;
    loadingInterval = setInterval(() => {
        index = (index + 1) % LOADING_WORDS.length;
        wordEl.textContent = LOADING_WORDS[index];
    }, 600);
}

function stopLoadingAnimation() {
    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }
}

async function loadDashboard() {
    startLoadingAnimation();

    try {
        // Load queue data (includes stats) and full drafts in parallel
        const [queue, drafts] = await Promise.all([
            apiCall('/dashboard/queue'),
            apiCall('/dashboard/drafts').catch(() => [])
        ]);

        stopLoadingAnimation();
        renderQueue(queue);

        // Show the recovery button only when there are stranded tickets to recover.
        const reprocessBtn = document.getElementById('reprocess-btn');
        if (reprocessBtn) {
            const stuck = queue.stuck_count || 0;
            reprocessBtn.style.display = stuck ? '' : 'none';
            reprocessBtn.textContent = `↻ Recover ${stuck} stuck ticket${stuck === 1 ? '' : 's'}`;
        }

        // Store full drafts for review
        draftsQueue = drafts;

        // Knowledge base stats
        const kb = queue.knowledge || {};
        const statusKnowledge = document.getElementById('status-knowledge');
        if (statusKnowledge) {
            const parts = [];
            if (kb.docs) parts.push(`${kb.docs} doc pages`);
            if (kb.examples) parts.push(`${kb.examples} response templates`);
            if (kb.internal) parts.push(`${kb.internal} internal notes`);
            statusKnowledge.textContent = parts.join(' · ') || 'no knowledge';
        }

    } catch (error) {
        stopLoadingAnimation();
        if (error.message !== 'Unauthorized') {
            const summaryEl = document.getElementById('queue-summary');
            if (summaryEl) {
                summaryEl.innerHTML = '<span style="color: #ff5f56">Error loading queue</span>';
            }
        }
    }
}

// ===================
// Utilities
// ===================

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ===================
// Event Handlers
// ===================

function handleCommandInput(e) {
    const input = document.getElementById('command-input');

    if (e.key === 'Enter') {
        const command = input.value.trim();
        if (command) {
            executeCommand(command);
            input.value = '';
        }
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (historyIndex < commandHistory.length - 1) {
            historyIndex++;
            input.value = commandHistory[historyIndex];
        }
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (historyIndex > 0) {
            historyIndex--;
            input.value = commandHistory[historyIndex];
        } else if (historyIndex === 0) {
            historyIndex = -1;
            input.value = '';
        }
    }
}

// ===================
// Initialization
// ===================

function setGreeting() {
    const hour = new Date().getHours();
    let pool;

    if (hour >= 5 && hour < 12) {
        pool = GREETINGS.morning;
    } else if (hour >= 12 && hour < 18) {
        pool = GREETINGS.afternoon;
    } else {
        pool = GREETINGS.evening;
    }

    // 50% chance to use generic greeting instead
    if (Math.random() > 0.5) {
        pool = GREETINGS.generic;
    }

    const greeting = pool[Math.floor(Math.random() * pool.length)];
    const el = document.getElementById('greeting');
    if (el) el.textContent = greeting;
}

document.addEventListener('DOMContentLoaded', function() {
    // Set random greeting
    setGreeting();

    // Load dashboard data
    loadDashboard();

    // Set up event listeners
    document.getElementById('command-input').addEventListener('keydown', handleCommandInput);
    document.getElementById('auth-submit-btn').addEventListener('click', submitAdminKey);
    const reprocessBtn = document.getElementById('reprocess-btn');
    if (reprocessBtn) reprocessBtn.addEventListener('click', () => reprocessStuck('reprocess', false));

    document.getElementById('admin-key-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') submitAdminKey();
    });

    // Focus command input
    document.getElementById('command-input').focus();

    // Auto-refresh dashboard every 30 seconds
    setInterval(loadDashboard, 30000);
});

// Make functions available globally for button onclick
window.postLastResult = postLastResult;
window.reprocessStuck = reprocessStuck;
window.clearTerminal = clearTerminal;
window.openDraft = openDraft;
window.sendDraftAsPending = sendDraftAsPending;
window.skipDraft = skipDraft;
window.typeCommand = typeCommand;
