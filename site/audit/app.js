(function () {
  const { blank, error, complete, csv, restore } = window.DonorAudit;
  const packet = {
    id: 'umm-audit-counting-demo-v1',
    items: ['original-update', 'donor-update'].map((name, index) => ({
      item_id: 'example-' + (index + 1),
      image: '../images/' + name + '.webp',
      capability: 'counting',
      objects: 'Trucks',
      statement_a: '',
      statement_b: ''
    }))
  };
  const withLabels = [['yes', 'Yes'], ['no', 'No'], ['unclear', 'Unsure']];
  const get = id => document.getElementById(id);
  let state = { packet: packet.id, index: 0, rows: blank(packet.items) };
  let storageMessage = '';

  try {
    const stored = localStorage.getItem(packet.id);
    if (stored) state = restore(packet, JSON.parse(stored));
  } catch {
    storageMessage = 'The browser draft could not be opened. You can import a saved draft file.';
  }

  function update() {
    const finished = complete(state.rows);
    get('progress').textContent = finished + ' of ' + state.rows.length + ' images scored';
    get('bar').max = state.rows.length;
    get('bar').value = finished;
    get('export').disabled = finished !== state.rows.length;
    get('message').textContent = storageMessage || error(state.rows[state.index]);
  }

  function save() {
    try {
      localStorage.setItem(packet.id, JSON.stringify(state));
      storageMessage = '';
    } catch {
      storageMessage = 'Browser storage is unavailable. Save a draft file to keep your work.';
    }
    update();
  }

  function choice(field, value, title, checked, onChange, type = 'radio') {
    const wrapper = document.createElement('span');
    wrapper.className = 'choice';
    const input = document.createElement('input');
    input.type = type;
    input.id = field + '-' + value;
    input.name = field;
    input.value = value;
    input.checked = checked;
    input.addEventListener('change', () => onChange(input.checked));
    const label = document.createElement('label');
    label.htmlFor = input.id;
    label.textContent = title;
    wrapper.append(input, label);
    return wrapper;
  }

  function question(row, field, text, values) {
    const group = document.createElement('fieldset');
    const legend = document.createElement('legend');
    legend.textContent = text;
    group.append(legend, ...values.map(([value, title]) =>
      choice(field, value, title, row[field] === value, () => {
        row[field] = value;
        save();
      })
    ));
    return group;
  }

  function render() {
    const row = state.rows[state.index];
    get('image').src = row.image;
    get('item').textContent = 'Image ' + (state.index + 1) + ' of ' + state.rows.length;
    get('objects').textContent = row.objects;
    const count = document.createElement('div');
    count.className = 'count-field';
    const label = document.createElement('label');
    label.htmlFor = 'count';
    label.textContent = 'How many trucks can you distinguish?';
    const input = document.createElement('input');
    input.id = 'count';
    input.type = 'number';
    input.min = 0;
    input.max = 99;
    input.disabled = row.observed_count === 'unclear';
    input.value = input.disabled ? '' : row.observed_count;
    input.addEventListener('input', () => {
      row.observed_count = input.value;
      save();
    });
    const unknown = choice('observed_count', 'unclear', 'Unsure', input.disabled, checked => {
      row.observed_count = checked ? 'unclear' : '';
      input.disabled = checked;
      input.value = '';
      save();
    }, 'checkbox');
    count.append(label, input, ' ', unknown);
    get('questions').replaceChildren(
      question(row, 'objects_present', 'Are any trucks present?', withLabels),
      count,
      question(row, 'object_form', 'Are the truck shapes intact?', [
        ['coherent', 'Intact'], ['deformed', 'Deformed or fused'], ['unclear', 'Unsure']
      ])
    );
    get('notes').value = row.notes;
    get('prev').disabled = state.index === 0;
    get('next').disabled = state.index === state.rows.length - 1;
    update();
  }

  function goTo(index) {
    state.index = index;
    save();
    render();
  }

  function download(filename, content, type) {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([content], { type }));
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }

  get('notes').addEventListener('input', event => {
    state.rows[state.index].notes = event.target.value;
    save();
  });
  get('prev').onclick = () => goTo(state.index - 1);
  get('next').onclick = () => goTo(state.index + 1);
  get('missing').onclick = () => {
    const index = state.rows.findIndex(error);
    if (index !== -1) goTo(index);
  };
  get('backup').onclick = () => download(packet.id + '-draft.json', JSON.stringify(state), 'application/json');
  get('restore').onclick = () => get('draft').click();
  get('draft').onchange = async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      state = restore(packet, JSON.parse(await file.text()));
      save();
      render();
    } catch (failure) {
      get('message').textContent = failure.message;
    } finally {
      event.target.value = '';
    }
  };
  get('export').onclick = () => download(packet.id + '-ratings.csv', csv(state.rows), 'text/csv;charset=utf-8');
  render();
})();
