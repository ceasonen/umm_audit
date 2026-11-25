(function () {
  const comparison = document.getElementById('comparison');
  const task = document.getElementById('task');
  const caption = document.getElementById('result-caption');
  const head = document.getElementById('result-head');
  const body = document.getElementById('result-body');
  const note = document.getElementById('result-note');
  const status = document.getElementById('result-status');

  const views = {
    endpoints: {
      labels: ['Answer (pp)', 'Image (100×)'],
      metrics: ['answer_pp', 'image_100x'],
      description: 'Trained update minus frozen baseline. Answer accuracy is in percentage points (pp). Image scores use the 100× scale.',
      spatialNote: ' Spatial image scores average 2D and 3D prompts.'
    },
    swaps: {
      labels: ['Original request', 'Donor request'],
      metrics: ['recipient', 'donor'],
      description: 'Donor swap minus own update on 128 paired examples per run. Single-run rows include all pairs, including any magnitude-matching failures.',
      countingNote: ' Counting measures exact-hit changes in percentage points (pp).',
      spatialNote: ' The 2D spatial task reports changes on a 100× score scale.'
    }
  };

  function parseRows(csv) {
    const [header, ...lines] = csv.trim().split(/\r?\n/);
    const columns = header.split(',');
    return lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [columns[i], value])));
  }

  function cell(tag, text, scope) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (scope) node.scope = scope;
    return node;
  }

  function render(rows) {
    const mode = comparison.value;
    const dimension = task.value;
    const view = views[mode];
    const unit = dimension === 'counting' ? 'pp' : '100x';
    const metricKeys = view.metrics.map(metric =>
      mode === 'endpoints' ? dimension + '_' + metric : dimension + '_' + metric + '_' + unit
    );
    caption.textContent = comparison.selectedOptions[0].text + ' · ' + task.selectedOptions[0].text;
    const headerRow = document.createElement('tr');
    headerRow.append(...['Backbone', 'Runs', ...view.labels].map(label => cell('th', label, 'col')));
    head.replaceChildren(headerRow);
    body.replaceChildren(...rows.map(result => {
      const row = document.createElement('tr');
      row.append(cell('th', result.backbone, 'row'), cell('td', result.runs));
      for (const key of metricKeys) {
        const value = Number(result[key]);
        const entry = cell('td', (value > 0 ? '+' : '') + value.toFixed(2).replace('-', '−'));
        if (value !== 0) entry.className = value > 0 ? 'rise' : 'fall';
        row.append(entry);
      }
      return row;
    }));
    note.textContent = view.description + (view[dimension + 'Note'] || '');
  }

  status.textContent = 'Loading results…';
  fetch('data/results.csv')
    .then(response => {
      if (!response.ok) throw new Error('Results are unavailable');
      return response.text();
    })
    .then(csv => {
      const rows = parseRows(csv);
      for (const control of [comparison, task]) {
        control.disabled = false;
        control.addEventListener('change', () => render(rows));
      }
      render(rows);
      document.getElementById('result-table').hidden = false;
      status.textContent = '';
    })
    .catch(() => {
      status.textContent = 'The results table could not be loaded. The CSV above is also available to download.';
    });
})();
