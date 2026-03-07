const API = '';
function get(u) { return fetch(API + u).then(r => r.ok ? r.json() : Promise.reject(new Error(r.statusText))); }

function showView(id) {
  document.querySelectorAll('main section').forEach(s => s.classList.add('hidden'));
  const el = document.getElementById(id);
  if (el) el.classList.remove('hidden');
}

// Leaderboard
document.querySelectorAll('nav button[data-view]').forEach(btn => {
  btn.addEventListener('click', () => {
    const v = btn.getAttribute('data-view');
    if (v === 'leaderboard') loadLeaderboard();
    if (v === 'runs') loadRuns();
    if (v === 'compare') loadCompare();
    showView(v + '-view');
  });
});

function loadLeaderboard() {
  get('/leaderboard').then(data => {
    const tbody = document.querySelector('#leaderboard-table tbody');
    tbody.innerHTML = '';
    (data.leaderboard || []).forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${row.model_name}</td><td>${row.rubric_name}</td><td>${Number(row.avg_score).toFixed(2)}</td><td>${row.run_count}</td>`;
      tbody.appendChild(tr);
    });
  }).catch(e => { document.querySelector('#leaderboard-table tbody').innerHTML = '<tr><td colspan="4">Failed to load</td></tr>'; });
}

function loadRuns() {
  get('/runs').then(data => {
    const list = document.getElementById('runs-list');
    list.innerHTML = '';
    (data.runs || []).forEach(run => {
      const card = document.createElement('div');
      card.className = 'run-card';
      const cost = run.cost_metadata || {};
      card.innerHTML = `<strong>${run.model_name}</strong> / ${run.rubric_name} — ${run.timestamp || run.run_id}<br><small>Bias: ${run.position_bias_rate != null ? run.position_bias_rate.toFixed(2) : '-'} | Verbosity: ${run.verbosity_correlation != null ? run.verbosity_correlation.toFixed(2) : '-'} | Cost: $${(cost.total_estimated_cost != null ? cost.total_estimated_cost : 0).toFixed(4)}</small>`;
      card.addEventListener('click', () => loadRunDetail(run.run_id));
      list.appendChild(card);
    });
    document.getElementById('run-detail').classList.add('hidden');
  }).catch(() => { document.getElementById('runs-list').innerHTML = 'Failed to load runs'; });
}

function loadRunDetail(runId) {
  get('/runs/' + runId).then(report => {
    document.getElementById('run-detail-id').textContent = runId;
    const biasCost = document.getElementById('run-bias-cost');
    const cm = report.cost_metadata || {};
    const n = (report.items || []).length;
    const avgCost = n > 0 && cm.total_estimated_cost != null ? (cm.total_estimated_cost / n) : 0;
    biasCost.innerHTML = `
      <dl>
        <dt>Position bias rate</dt><dd>${report.position_bias_rate != null ? report.position_bias_rate.toFixed(3) : '-'}</dd>
        <dt>Verbosity correlation</dt><dd>${report.verbosity_correlation != null ? report.verbosity_correlation.toFixed(3) : '-'}</dd>
        <dt>Cohen's κ</dt><dd>${report.cohens_kappa != null ? report.cohens_kappa.toFixed(3) : '-'}</dd>
        <dt>Total estimated cost</dt><dd>$${(cm.total_estimated_cost != null ? cm.total_estimated_cost : 0).toFixed(4)}</dd>
        <dt>Avg cost per item</dt><dd>$${avgCost.toFixed(4)}</dd>
        <dt>Tokens in/out</dt><dd>${cm.total_tokens_in || 0} / ${cm.total_tokens_out || 0}</dd>
      </dl>
    `;
    const tbody = document.querySelector('#run-items-table tbody');
    tbody.innerHTML = '';
    (report.items || []).forEach((it, i) => {
      const tr = document.createElement('tr');
      const scores = typeof it.scores === 'object' ? JSON.stringify(it.scores) : '';
      const reasoning = typeof it.reasoning === 'object' ? JSON.stringify(it.reasoning).slice(0, 80) + '...' : '';
      tr.innerHTML = `<td>${i + 1}</td><td>${(it.question || '').slice(0, 60)}...</td><td>${it.overall_score}</td><td>${scores}</td><td>${reasoning}</td>`;
      tbody.appendChild(tr);
    });
    document.getElementById('run-detail').classList.remove('hidden');
  }).catch(() => {});
}

function loadCompare() {
  get('/leaderboard').then(data => {
    const models = [...new Set((data.leaderboard || []).map(r => r.model_name))];
    const selA = document.getElementById('compare-model-a');
    const selB = document.getElementById('compare-model-b');
    selA.innerHTML = selB.innerHTML = '<option value="">Select...</option>' + models.map(m => `<option value="${m}">${m}</option>`).join('');
  });
}

document.getElementById('compare-btn').addEventListener('click', () => {
  const a = document.getElementById('compare-model-a').value;
  const b = document.getElementById('compare-model-b').value;
  if (!a || !b) return;
  Promise.all([get('/models/' + encodeURIComponent(a) + '/items'), get('/models/' + encodeURIComponent(b) + '/items')])
    .then(([resA, resB]) => {
      const itemsA = (resA.items || []).reduce((acc, it) => { acc[it.question] = it; return acc; }, {});
      const itemsB = (resB.items || []).reduce((acc, it) => { acc[it.question] = it; return acc; }, {});
      const questions = [...new Set([...Object.keys(itemsA), ...Object.keys(itemsB)])];
      let html = '<table><thead><tr><th>Question</th><th>' + a + '</th><th>' + b + '</th><th>Delta</th></tr></thead><tbody>';
      questions.forEach(q => {
        const sa = itemsA[q] ? (itemsA[q].overall_score != null ? itemsA[q].overall_score : '-') : '-';
        const sb = itemsB[q] ? (itemsB[q].overall_score != null ? itemsB[q].overall_score : '-') : '-';
        let delta = '';
        if (typeof sa === 'number' && typeof sb === 'number') {
          const d = sa - sb;
          delta = '<span class="' + (d > 0 ? 'delta-positive' : d < 0 ? 'delta-negative' : '') + '">' + (d >= 0 ? '+' : '') + d.toFixed(2) + '</span>';
        }
        html += `<tr><td>${(q || '').slice(0, 50)}...</td><td>${sa}</td><td>${sb}</td><td>${delta}</td></tr>`;
      });
      html += '</tbody></table>';
      document.getElementById('compare-result').innerHTML = html;
    })
    .catch(() => { document.getElementById('compare-result').innerHTML = 'Failed to load'; });
});

// Initial load
loadLeaderboard();
showView('leaderboard-view');
