/**
 * Support Triage Agent - Interactive Demo Console
 * Static, framework-free vanilla JavaScript client
 */

(function () {
  'use strict';

  // State
  let demoData = null;
  let activeRunId = null;
  let filteredTickets = [];
  let selectedTicketId = null;
  let currentPage = 1;
  const pageSize = 25;

  // DOM Elements
  const el = {
    themeToggle: document.getElementById('theme-toggle'),
    runSelect: document.getElementById('run-select'),
    runBadge: document.getElementById('run-badge'),
    evalCountBadge: document.getElementById('eval-count-badge'),
    kpiCards: document.getElementById('kpi-cards'),
    comparisonSection: document.getElementById('comparison-section'),
    comparisonTableBody: document.getElementById('comparison-tbody'),
    compThRun1: document.getElementById('comp-th-run1'),
    compThRun2: document.getElementById('comp-th-run2'),
    reasonsCard: document.getElementById('reasons-card'),
    reasonsPills: document.getElementById('reasons-pills'),
    reasonsTotalLabel: document.getElementById('reasons-total-label'),
    // Filters
    filterSearch: document.getElementById('filter-search'),
    clearSearch: document.getElementById('clear-search'),
    filterIntent: document.getElementById('filter-intent'),
    filterAccuracy: document.getElementById('filter-accuracy'),
    filterResolution: document.getElementById('filter-resolution'),
    filterGrounded: document.getElementById('filter-grounded'),
    resetFilters: document.getElementById('reset-filters'),
    filteredCount: document.getElementById('filtered-count'),
    // Table & Pagination
    ticketsTbody: document.getElementById('tickets-tbody'),
    noResults: document.getElementById('no-results'),
    paginationInfo: document.getElementById('pagination-info'),
    pageIndicator: document.getElementById('page-indicator'),
    pagePrev: document.getElementById('page-prev'),
    pageNext: document.getElementById('page-next'),
    // Detail Trace Panel
    traceTicketId: document.getElementById('trace-ticket-id'),
    traceStatusPill: document.getElementById('trace-status-pill'),
    traceEmptyState: document.getElementById('trace-empty-state'),
    traceTimeline: document.getElementById('trace-timeline'),
    traceInstruction: document.getElementById('trace-instruction'),
    traceCategory: document.getElementById('trace-category'),
    traceTrueIntent: document.getElementById('trace-true-intent'),
    tracePredIntent: document.getElementById('trace-pred-intent'),
    traceAccuracyBadge: document.getElementById('trace-accuracy-badge'),
    traceConfFill: document.getElementById('trace-conf-fill'),
    traceConfText: document.getElementById('trace-conf-text'),
    traceTopScore: document.getElementById('trace-top-score'),
    traceKbHitBadge: document.getElementById('trace-kb-hit-badge'),
    traceCitedArticles: document.getElementById('trace-cited-articles'),
    traceReplyBox: document.getElementById('trace-reply-box'),
    traceGroundedStatus: document.getElementById('trace-grounded-status'),
    traceDecisionOutcome: document.getElementById('trace-decision-outcome'),
    traceReasonsList: document.getElementById('trace-reasons-list'),
  };

  /* -------------------------------------------------------------
   * 1. Theme Toggle (Persistent Light / Dark mode, Light by default)
   * ------------------------------------------------------------- */
  function initTheme() {
    const savedTheme = localStorage.getItem('sta_theme') || 'light';
    setTheme(savedTheme);

    el.themeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      setTheme(next);
    });
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('sta_theme', theme);
  }

  /* -------------------------------------------------------------
   * 2. Data Fetching
   * ------------------------------------------------------------- */
  async function loadData() {
    try {
      const resp = await fetch('data/demo.json');
      if (!resp.ok) {
        throw new Error(`HTTP error ${resp.status}`);
      }
      demoData = await resp.json();
      onDataLoaded();
    } catch (err) {
      console.error('Failed to load demo data:', err);
      el.kpiCards.innerHTML = `
        <div class="kpi-card" style="grid-column: 1 / -1; background-color: var(--danger-bg); border-color: var(--danger-border); color: var(--danger);">
          <div class="kpi-label" style="color: var(--danger);">Data Loading Error</div>
          <p class="text-sm">Unable to load <code>data/demo.json</code>. Ensure the export script has generated the dataset.</p>
        </div>
      `;
    }
  }

  function onDataLoaded() {
    const runIds = Object.keys(demoData.runs || {});
    if (runIds.length === 0) {
      el.runSelect.innerHTML = '<option value="">No runs available</option>';
      return;
    }

    // Populate run selector
    el.runSelect.innerHTML = runIds
      .map(id => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`)
      .join('');

    activeRunId = runIds[runIds.length - 1];
    el.runSelect.value = activeRunId;

    el.runSelect.addEventListener('change', (e) => {
      activeRunId = e.target.value;
      renderCurrentRun();
    });

    renderCurrentRun();

    // Setup Run Comparison if 2 or more runs exist
    if (runIds.length >= 2) {
      renderRunComparison(runIds[0], runIds[1]);
    } else {
      el.comparisonSection.classList.add('hidden');
    }
  }

  /* -------------------------------------------------------------
   * 3. Render Active Run
   * ------------------------------------------------------------- */
  function renderCurrentRun() {
    const runData = demoData.runs[activeRunId];
    if (!runData) return;

    if (el.runSelect.value !== activeRunId) {
      el.runSelect.value = activeRunId;
    }
    el.runBadge.textContent = `Run: ${activeRunId}`;
    el.evalCountBadge.textContent = `${runData.tickets.length} tickets evaluated`;

    // Render KPIs
    renderKpiCards(runData.kpis);

    // Populate Intent filter options
    populateIntentFilter(runData.tickets);

    // Reset filters
    resetFiltersState();

    // Filter tickets and render table
    applyFilters();
  }

  /* -------------------------------------------------------------
   * 4. KPI Dashboard Cards & Reasons Breakdown
   * ------------------------------------------------------------- */
  function renderKpiCards(kpis) {
    if (!kpis) {
      el.kpiCards.innerHTML = '';
      return;
    }

    const cards = [
      {
        label: 'Intent Accuracy',
        value: formatPercent(kpis.intent_accuracy),
        subtext: `${kpis.classification_error_count || 0} misclassifications`,
        progress: (kpis.intent_accuracy || 0) * 100,
        variant: (kpis.intent_accuracy >= 0.9) ? 'success' : 'warning',
      },
      {
        label: 'KB Hit Rate (@4)',
        value: formatPercent(kpis.kb_hit_rate_at_4),
        subtext: 'Target article in top 4 chunks',
        progress: (kpis.kb_hit_rate_at_4 || 0) * 100,
        variant: (kpis.kb_hit_rate_at_4 >= 0.85) ? 'success' : 'info',
      },
      {
        label: 'Grounded Citations',
        value: formatPercent(kpis.grounded_rate),
        subtext: 'Supported by retrieved KB articles',
        progress: (kpis.grounded_rate || 0) * 100,
        variant: (kpis.grounded_rate >= 0.8) ? 'success' : 'warning',
      },
      {
        label: 'Escalation Rate',
        value: formatPercent(kpis.escalation_rate),
        subtext: `${kpis.non_escalated_count || 0} auto-answered`,
        progress: (kpis.escalation_rate || 0) * 100,
        variant: (kpis.escalation_rate < 0.3) ? 'info' : 'warning',
      },
      {
        label: 'Auto-Answer Accuracy',
        value: formatPercent(kpis.accuracy_on_non_escalated),
        subtext: 'Accuracy on non-escalated tickets',
        progress: (kpis.accuracy_on_non_escalated || 0) * 100,
        variant: (kpis.accuracy_on_non_escalated >= 0.9) ? 'success' : 'warning',
      },
      {
        label: 'Error Recall',
        value: formatPercent(kpis.escalation_recall_on_errors),
        subtext: 'Misclassified caught by guardrails',
        progress: (kpis.escalation_recall_on_errors || 0) * 100,
        variant: (kpis.escalation_recall_on_errors >= 0.4) ? 'success' : 'danger',
      },
      {
        label: 'Total Run Cost',
        value: `$${(kpis.cost_usd || 0).toFixed(4)}`,
        subtext: `~$${((kpis.cost_usd || 0) / (kpis.count || 1)).toFixed(5)} / ticket`,
        progress: null,
      },
    ];

    el.kpiCards.innerHTML = cards.map(c => `
      <div class="kpi-card">
        <span class="kpi-label">${escapeHtml(c.label)}</span>
        <div class="kpi-value-row">
          <span class="kpi-value">${c.value}</span>
        </div>
        <span class="kpi-subtext">${escapeHtml(c.subtext)}</span>
        ${c.progress !== null ? `
          <div class="kpi-bar">
            <div class="kpi-bar-fill ${c.variant || ''}" style="width: ${Math.min(100, Math.max(0, c.progress))}%;"></div>
          </div>
        ` : ''}
      </div>
    `).join('');

    // Escalation Reasons
    const reasons = kpis.escalation_reasons || [];
    const totalEscalated = reasons.reduce((sum, r) => sum + r.count, 0);
    el.reasonsTotalLabel.textContent = `${totalEscalated} policy triggers logged`;

    if (reasons.length === 0) {
      el.reasonsPills.innerHTML = '<span class="text-muted text-sm">No escalation reasons recorded.</span>';
    } else {
      el.reasonsPills.innerHTML = reasons.map(r => `
        <div class="reason-pill" title="${r.count} tickets (${(r.rate * 100).toFixed(1)}%)">
          <strong>${formatReason(r.reason)}</strong>
          <span class="reason-count">${r.count}</span>
          <span class="text-muted">(${(r.rate * 100).toFixed(1)}%)</span>
        </div>
      `).join('');
    }
  }

  /* -------------------------------------------------------------
   * 5. Before / After Comparison Table
   * ------------------------------------------------------------- */
  function renderRunComparison(runId1, runId2) {
    const r1 = demoData.runs[runId1]?.kpis;
    const r2 = demoData.runs[runId2]?.kpis;
    if (!r1 || !r2) return;

    el.compThRun1.textContent = runId1;
    el.compThRun2.textContent = runId2;

    const metrics = [
      { key: 'intent_accuracy', name: 'Intent Accuracy', val1: r1.intent_accuracy, val2: r2.intent_accuracy, isPercent: true, higherIsBetter: true },
      { key: 'kb_hit_rate', name: 'KB Hit Rate (@4)', val1: r1.kb_hit_rate_at_4, val2: r2.kb_hit_rate_at_4, isPercent: true, higherIsBetter: true },
      { key: 'grounded_rate', name: 'Grounded Rate', val1: r1.grounded_rate, val2: r2.grounded_rate, isPercent: true, higherIsBetter: true },
      { key: 'escalation_rate', name: 'Escalation Rate', val1: r1.escalation_rate, val2: r2.escalation_rate, isPercent: true, higherIsBetter: false },
      { key: 'accuracy_on_non_escalated', name: 'Accuracy on Auto-Answered', val1: r1.accuracy_on_non_escalated, val2: r2.accuracy_on_non_escalated, isPercent: true, higherIsBetter: true },
      { key: 'error_recall', name: 'Error Recall', val1: r1.escalation_recall_on_errors, val2: r2.escalation_recall_on_errors, isPercent: true, higherIsBetter: true },
      { key: 'cost', name: 'Total Cost (USD)', val1: r1.cost_usd, val2: r2.cost_usd, isPercent: false, isCurrency: true, higherIsBetter: false },
    ];

    el.comparisonTableBody.innerHTML = metrics.map(m => {
      const v1 = m.val1 ?? 0;
      const v2 = m.val2 ?? 0;
      const delta = v2 - v1;
      const improved = m.higherIsBetter ? delta > 0.0001 : delta < -0.0001;
      const regressed = m.higherIsBetter ? delta < -0.0001 : delta > 0.0001;

      let deltaClass = 'delta-neutral';
      let statusBadge = '<span class="badge badge-neutral">No change</span>';
      if (improved) {
        deltaClass = 'delta-positive';
        statusBadge = '<span class="badge badge-success">Improved</span>';
      } else if (regressed) {
        deltaClass = 'delta-negative';
        statusBadge = '<span class="badge badge-danger">Regression</span>';
      }

      const fmt1 = m.isPercent ? formatPercent(v1) : (m.isCurrency ? `$${v1.toFixed(4)}` : v1.toFixed(3));
      const fmt2 = m.isPercent ? formatPercent(v2) : (m.isCurrency ? `$${v2.toFixed(4)}` : v2.toFixed(3));
      const fmtDelta = (delta > 0 ? '+' : '') + (m.isPercent ? (delta * 100).toFixed(1) + '%' : (m.isCurrency ? `$${delta.toFixed(4)}` : delta.toFixed(3)));

      if (m.key === 'error_recall' && typeof r2.classification_error_count === 'number' && r2.classification_error_count < 5) {
        deltaClass = 'delta-neutral';
        statusBadge = `<span class="badge badge-neutral">Low sample (n=${r2.classification_error_count} errors)</span>`;
      } else if (m.key === 'cost') {
        deltaClass = 'delta-neutral';
        const costPerTicket = (v2 / (r2.count || 200)).toFixed(4);
        statusBadge = `<span class="badge badge-neutral">${fmtDelta} (still ~$${costPerTicket}/ticket)</span>`;
      }

      return `
        <tr>
          <td><strong>${escapeHtml(m.name)}</strong></td>
          <td>${fmt1}</td>
          <td>${fmt2}</td>
          <td><span class="delta-badge ${deltaClass}">${fmtDelta}</span></td>
          <td>${statusBadge}</td>
        </tr>
      `;
    }).join('');

    el.comparisonSection.classList.remove('hidden');
  }

  /* -------------------------------------------------------------
   * 6. Filters & Search
   * ------------------------------------------------------------- */
  function populateIntentFilter(tickets) {
    const intents = new Set();
    tickets.forEach(t => {
      if (t.true_intent) intents.add(t.true_intent);
      if (t.pred_intent) intents.add(t.pred_intent);
    });

    const sortedIntents = Array.from(intents).sort();
    const currentVal = el.filterIntent.value;

    el.filterIntent.innerHTML = '<option value="all">All Intents</option>' +
      sortedIntents.map(i => `<option value="${escapeHtml(i)}">${escapeHtml(i)}</option>`).join('');

    if (sortedIntents.includes(currentVal)) {
      el.filterIntent.value = currentVal;
    }
  }

  function resetFiltersState() {
    el.filterSearch.value = '';
    el.clearSearch.classList.add('hidden');
    el.filterIntent.value = 'all';
    el.filterAccuracy.value = 'all';
    el.filterResolution.value = 'all';
    el.filterGrounded.value = 'all';
    currentPage = 1;
  }

  function applyFilters() {
    const runData = demoData?.runs[activeRunId];
    if (!runData) return;

    const query = el.filterSearch.value.trim().toLowerCase();
    const intent = el.filterIntent.value;
    const accuracy = el.filterAccuracy.value;
    const resolution = el.filterResolution.value;
    const grounded = el.filterGrounded.value;

    filteredTickets = runData.tickets.filter(t => {
      // Search query
      if (query) {
        const idMatch = String(t.ticket_id).includes(query);
        const textMatch = (t.text || '').toLowerCase().includes(query);
        const trueIntentMatch = (t.true_intent || '').toLowerCase().includes(query);
        const predIntentMatch = (t.pred_intent || '').toLowerCase().includes(query);
        const replyMatch = (t.reply || '').toLowerCase().includes(query);
        if (!idMatch && !textMatch && !trueIntentMatch && !predIntentMatch && !replyMatch) {
          return false;
        }
      }

      // Intent
      if (intent !== 'all' && t.true_intent !== intent && t.pred_intent !== intent) {
        return false;
      }

      // Classification accuracy
      if (accuracy === 'correct' && !t.correct) return false;
      if (accuracy === 'wrong' && t.correct) return false;

      // Resolution
      if (resolution === 'auto' && t.escalated) return false;
      if (resolution === 'escalated' && !t.escalated) return false;

      // Grounding
      if (grounded === 'grounded' && !t.grounded) return false;
      if (grounded === 'ungrounded' && t.grounded) return false;

      return true;
    });

    el.filteredCount.textContent = `${filteredTickets.length} / ${runData.tickets.length} tickets`;
    currentPage = 1;
    renderTable();

    // Auto-select first ticket if current selected ticket is not in filtered list
    if (filteredTickets.length > 0) {
      const stillSelected = filteredTickets.some(t => t.ticket_id === selectedTicketId);
      if (!stillSelected) {
        selectTicket(filteredTickets[0].ticket_id);
      }
    } else {
      clearTicketDetail();
    }
  }

  /* -------------------------------------------------------------
   * 7. Tickets Table & Pagination
   * ------------------------------------------------------------- */
  function renderTable() {
    const total = filteredTickets.length;
    if (total === 0) {
      el.ticketsTbody.innerHTML = '';
      el.noResults.classList.remove('hidden');
      el.paginationInfo.textContent = 'Showing 0 tickets';
      el.pageIndicator.textContent = 'Page 0 of 0';
      el.pagePrev.disabled = true;
      el.pageNext.disabled = true;
      return;
    }

    el.noResults.classList.add('hidden');
    const totalPages = Math.ceil(total / pageSize);
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const startIdx = (currentPage - 1) * pageSize;
    const endIdx = Math.min(startIdx + pageSize, total);
    const pageTickets = filteredTickets.slice(startIdx, endIdx);

    el.ticketsTbody.innerHTML = pageTickets.map(t => {
      const isSelected = t.ticket_id === selectedTicketId;
      const isCorrect = t.correct;

      // Classification column badge
      const accuracyBadge = isCorrect
        ? `<span class="badge badge-success" title="Predicted matched Ground Truth">✓ Correct</span>`
        : `<span class="badge badge-danger" title="Mismatch: True is ${escapeHtml(t.true_intent)}">✗ Mismatch</span>`;

      // Confidence badge / text
      const confPercent = Math.round(t.confidence * 100);

      // Retrieval
      const kbHitBadge = t.kb_hit
        ? `<span class="badge badge-success">Hit</span>`
        : `<span class="badge badge-neutral">Miss</span>`;
      const kbScore = t.top_kb_score != null ? t.top_kb_score.toFixed(2) : '—';

      // Grounding
      const groundedBadge = t.grounded
        ? `<span class="badge badge-success">Grounded</span>`
        : `<span class="badge badge-warning">Ungrounded</span>`;

      // Resolution
      const resolutionBadge = t.escalated
        ? `<span class="badge badge-danger" title="Escalated: ${escapeHtml((t.reasons || []).join(', '))}">Escalated</span>`
        : `<span class="badge badge-success" title="Auto-answered safely">Auto-reply</span>`;

      return `
        <tr data-id="${t.ticket_id}" class="${isSelected ? 'selected' : ''}">
          <td class="ticket-cell-id">#${t.ticket_id}</td>
          <td>
            <div class="ticket-cell-text" title="${escapeHtml(t.text)}">${escapeHtml(t.text)}</div>
          </td>
          <td>
            <div class="intent-tag-group">
              <span class="intent-code">${escapeHtml(t.pred_intent)}</span>
              ${accuracyBadge}
            </div>
          </td>
          <td>
            <span class="font-mono">${confPercent}%</span>
          </td>
          <td>
            <div style="display: flex; align-items: center; gap: 4px;">
              <span class="font-mono text-sm">${kbScore}</span>
              ${kbHitBadge}
            </div>
          </td>
          <td>${groundedBadge}</td>
          <td>${resolutionBadge}</td>
        </tr>
      `;
    }).join('');

    // Pagination Info
    el.paginationInfo.textContent = `Showing ${startIdx + 1}-${endIdx} of ${total}`;
    el.pageIndicator.textContent = `Page ${currentPage} of ${totalPages}`;
    el.pagePrev.disabled = currentPage <= 1;
    el.pageNext.disabled = currentPage >= totalPages;

    // Attach row click listeners
    el.ticketsTbody.querySelectorAll('tr').forEach(row => {
      row.addEventListener('click', () => {
        const id = parseInt(row.getAttribute('data-id'), 10);
        selectTicket(id);
      });
    });
  }

  /* -------------------------------------------------------------
   * 8. Ticket Detail Panel (Trace Step by Step)
   * ------------------------------------------------------------- */
  function selectTicket(ticketId) {
    selectedTicketId = ticketId;

    // Highlight row in table
    el.ticketsTbody.querySelectorAll('tr').forEach(r => {
      const id = parseInt(r.getAttribute('data-id'), 10);
      r.classList.toggle('selected', id === ticketId);
    });

    const runData = demoData?.runs[activeRunId];
    if (!runData) return;

    const ticket = runData.tickets.find(t => t.ticket_id === ticketId);
    if (!ticket) return;

    renderTicketTrace(ticket);
  }

  function clearTicketDetail() {
    selectedTicketId = null;
    el.traceEmptyState.classList.remove('hidden');
    el.traceTimeline.classList.add('hidden');
    el.traceTicketId.textContent = 'Select a ticket';
    el.traceStatusPill.innerHTML = '';
  }

  function renderTicketTrace(ticket) {
    el.traceEmptyState.classList.add('hidden');
    el.traceTimeline.classList.remove('hidden');

    el.traceTicketId.textContent = `Ticket #${ticket.ticket_id}`;

    // Status pill
    if (ticket.escalated) {
      el.traceStatusPill.innerHTML = `<span class="badge badge-danger">Escalated to Human</span>`;
    } else {
      el.traceStatusPill.innerHTML = `<span class="badge badge-success">Auto-Answered</span>`;
    }

    // Step 1: Inbound Ticket
    el.traceInstruction.textContent = ticket.text;
    el.traceCategory.textContent = ticket.category || 'SUPPORT';
    el.traceTrueIntent.textContent = ticket.true_intent;

    // Step 2: Classify
    el.tracePredIntent.textContent = ticket.pred_intent;
    if (ticket.correct) {
      el.traceAccuracyBadge.className = 'badge badge-success';
      el.traceAccuracyBadge.textContent = '✓ Correct (Matches Ground Truth)';
    } else {
      el.traceAccuracyBadge.className = 'badge badge-danger';
      el.traceAccuracyBadge.textContent = `✗ Mismatch (Ground truth: ${ticket.true_intent})`;
    }

    const confPct = Math.round(ticket.confidence * 100);
    el.traceConfFill.style.width = `${confPct}%`;
    el.traceConfText.textContent = `${confPct}%`;

    // Step 3: Retrieve
    el.traceTopScore.textContent = ticket.top_kb_score != null ? ticket.top_kb_score.toFixed(4) : 'N/A';
    if (ticket.kb_hit) {
      el.traceKbHitBadge.className = 'badge badge-success';
      el.traceKbHitBadge.textContent = 'Hit (@4)';
    } else {
      el.traceKbHitBadge.className = 'badge badge-warning';
      el.traceKbHitBadge.textContent = 'Miss';
    }

    // Retrieved & Cited articles chips
    const citations = ticket.citations || [];
    if (citations.length === 0) {
      el.traceCitedArticles.innerHTML = '<span class="text-muted text-sm">No KB articles cited in reply.</span>';
    } else {
      el.traceCitedArticles.innerHTML = citations.map(kbId => renderKbChip(kbId)).join('');
    }

    // Step 4: Draft Reply with [KB-xx] chips
    el.traceReplyBox.innerHTML = renderReplyWithChips(ticket.reply);

    // Step 5: Citation Verification
    if (ticket.grounded) {
      el.traceGroundedStatus.innerHTML = `
        <div style="display: flex; align-items: center; gap: 6px;">
          <span class="badge badge-success">✓ Grounded</span>
          <span class="text-sm">All inline citations correspond to retrieved KB articles with zero unsupported factual claims.</span>
        </div>
      `;
    } else {
      el.traceGroundedStatus.innerHTML = `
        <div style="display: flex; align-items: center; gap: 6px;">
          <span class="badge badge-warning">⚠ Not Grounded</span>
          <span class="text-sm">Failed citation audit: either citations were missing, outside retrieved context, or contained unverified sentences.</span>
        </div>
      `;
    }

    // Step 6: Escalation Decision
    if (ticket.escalated) {
      el.traceDecisionOutcome.innerHTML = `
        <span class="badge badge-danger" style="font-size: 0.8125rem; padding: 4px 10px;">Escalate to Tier-2 Agent</span>
        <span class="text-sm text-muted">Response withheld from customer; ticket forwarded to human agent queue.</span>
      `;
    } else {
      el.traceDecisionOutcome.innerHTML = `
        <span class="badge badge-success" style="font-size: 0.8125rem; padding: 4px 10px;">Auto-reply Approved</span>
        <span class="text-sm text-muted">Directly delivered to customer with high confidence and verified citations.</span>
      `;
    }

    const reasons = ticket.reasons || [];
    if (reasons.length === 0) {
      el.traceReasonsList.innerHTML = '<span class="badge badge-neutral">No safety triggers tripped (Safe)</span>';
    } else {
      el.traceReasonsList.innerHTML = reasons.map(r => `
        <span class="badge badge-danger">
          <svg style="width: 12px; height: 12px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          ${formatReason(r)}
        </span>
      `).join('');
    }
  }

  /* -------------------------------------------------------------
   * 9. Chip Rendering & String Helpers
   * ------------------------------------------------------------- */
  function renderKbChip(rawKbId) {
    const normId = normalizeKbId(rawKbId);
    const kbItem = demoData?.kb?.[normId];
    const title = kbItem?.title || 'Knowledge Base Article';
    const intent = kbItem?.intent ? `(${kbItem.intent})` : '';

    return `
      <span class="kb-chip" title="${escapeHtml(normId)}: ${escapeHtml(title)} ${escapeHtml(intent)}">
        <svg style="width: 12px; height: 12px; flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
        </svg>
        <span class="chip-id">${escapeHtml(normId)}</span>
        <span class="chip-title">${escapeHtml(title)}</span>
      </span>
    `;
  }

  function renderReplyWithChips(replyText) {
    if (!replyText) return '<span class="text-muted">No reply generated.</span>';

    // First escape HTML in reply text to prevent XSS
    const escaped = escapeHtml(replyText);

    // Replace [KB-xx] or [KB-XX] with chips
    return escaped.replace(/\[KB-(\d+)\]/gi, (match, digits) => {
      const num = parseInt(digits, 10);
      const kbId = `KB-${String(num).padStart(2, '0')}`;
      const kbItem = demoData?.kb?.[kbId];
      const title = kbItem?.title || 'Knowledge Base Article';

      return `
        <span class="kb-chip" title="${escapeHtml(kbId)}: ${escapeHtml(title)}" data-kb="${escapeHtml(kbId)}">
          <span class="chip-id">[${escapeHtml(kbId)}]</span>
          <span class="chip-title">${escapeHtml(title)}</span>
        </span>
      `;
    });
  }

  function normalizeKbId(val) {
    if (!val) return 'KB-01';
    const s = String(val).trim().toUpperCase();
    const match = s.match(/^KB-?(\d+)$/);
    if (match) {
      return `KB-${String(parseInt(match[1], 10)).padStart(2, '0')}`;
    }
    return s;
  }

  function formatPercent(val) {
    if (val == null || isNaN(val)) return '—';
    return `${(val * 100).toFixed(1)}%`;
  }

  function formatReason(reasonStr) {
    if (!reasonStr) return 'Unknown';
    return reasonStr
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  }

  function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  /* -------------------------------------------------------------
   * 10. Event Listeners Setup
   * ------------------------------------------------------------- */
  function setupEventListeners() {
    // Search input
    let searchTimeout = null;
    el.filterSearch.addEventListener('input', () => {
      const hasVal = el.filterSearch.value.length > 0;
      el.clearSearch.classList.toggle('hidden', !hasVal);

      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        applyFilters();
      }, 150);
    });

    el.clearSearch.addEventListener('click', () => {
      el.filterSearch.value = '';
      el.clearSearch.classList.add('hidden');
      applyFilters();
    });

    // Select filters
    el.filterIntent.addEventListener('change', applyFilters);
    el.filterAccuracy.addEventListener('change', applyFilters);
    el.filterResolution.addEventListener('change', applyFilters);
    el.filterGrounded.addEventListener('change', applyFilters);

    // Reset button
    el.resetFilters.addEventListener('click', () => {
      resetFiltersState();
      applyFilters();
    });

    // Pagination
    el.pagePrev.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        renderTable();
      }
    });

    el.pageNext.addEventListener('click', () => {
      const totalPages = Math.ceil(filteredTickets.length / pageSize);
      if (currentPage < totalPages) {
        currentPage++;
        renderTable();
      }
    });
  }

  /* -------------------------------------------------------------
   * Entry Point
   * ------------------------------------------------------------- */
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupEventListeners();
    loadData();
  });
})();
