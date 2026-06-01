const SOLVERS = ['rootnode', 'ruge_stuben', 'smoothed_aggregation'];
const SOLVER_LABELS = {
  rootnode: 'Root Node',
  ruge_stuben: 'Ruge–Stüben',
  smoothed_aggregation: 'Smoothed Aggregation',
};
const METHODS = ['amjax', 'amjax_pcg', 'pyamg', 'pyamg_pcg'];
const METHOD_LABELS = {
  amjax: 'AMJax',
  amjax_pcg: 'AMJax + PCG',
  pyamg: 'PyAMG',
  pyamg_pcg: 'PyAMG + PCG',
};
const METHOD_COLORS = {
  amjax: '#2563eb',
  amjax_pcg: '#0ea5e9',
  pyamg: '#16a34a',
  pyamg_pcg: '#65a30d',
};
const GRID_SIZES = [50, 100, 200, 500, 1000];

const PINV_VS_JACOBI = {
  f32: [
    { method: 'AMJax',       jacobi_time: 0.1376095319999422,   pinv_time: 0.13853287400002046, jacobi_residual: 1.8337039509788156e-3, pinv_residual: 1.8278160132467747e-3 },
    { method: 'AMJax + PCG', jacobi_time: 0.020511815999952887,  pinv_time: null,                jacobi_residual: 2.3913965560495853e-3, pinv_residual: null },
    { method: 'PyAMG',       jacobi_time: 0.4719979559999956,    pinv_time: 0.5141297729999224,  jacobi_residual: 9.310109298097912e-11, pinv_residual: 9.438143487178253e-11 },
    { method: 'PyAMG + PCG', jacobi_time: 0.4085199680000642,    pinv_time: 0.4288588739998431,  jacobi_residual: 1.3283514263917804e-11, pinv_residual: 1.3547022363352355e-11 },
  ],
  f64: [
    { method: 'AMJax',       jacobi_time: 0.03199397699995643,   pinv_time: 0.032377573000076154, jacobi_residual: 3.523029448135565e-11,  pinv_residual: 3.521999887324147e-11 },
    { method: 'AMJax + PCG', jacobi_time: 0.024145035999936226,  pinv_time: 0.023089609999942695, jacobi_residual: 4.2388481708353174e-11, pinv_residual: 4.2251518098646775e-11 },
    { method: 'PyAMG',       jacobi_time: 0.8429085650000161,    pinv_time: 0.519526285999973,    jacobi_residual: 9.314766041232512e-11,  pinv_residual: 9.440596306223817e-11 },
    { method: 'PyAMG + PCG', jacobi_time: 0.38465043899998363,   pinv_time: 0.41908607200002734,  jacobi_residual: 1.3474994454087835e-11, pinv_residual: 1.346787669111285e-11 },
  ],
};

function formatTime(t) {
  if (t < 0.001) return (t * 1e6).toFixed(1) + ' µs';
  if (t < 1) return (t * 1000).toFixed(2) + ' ms';
  return t.toFixed(3) + ' s';
}

function filterData(solver, dtype, mode, device) {
  return BENCHMARK_DATA.filter(function(e) {
    return e.solver === solver && e.dtype === dtype && e.mode === mode && e.device === device;
  });
}

function makeFilterGroup(label) {
  var g = document.createElement('div');
  g.className = 'bm-filter-group';
  var l = document.createElement('span');
  l.className = 'bm-filter-label';
  l.textContent = label;
  g.appendChild(l);
  return g;
}

function makeBtn(text, active, onClick) {
  var btn = document.createElement('button');
  btn.className = 'bm-btn' + (active ? ' active' : '');
  btn.textContent = text;
  btn.addEventListener('click', onClick);
  return btn;
}

function makeFilters(containerId, state, onChange) {
  var container = document.getElementById(containerId);
  if (!container) return;

  function render() {
    container.innerHTML = '';

    var gSolver = makeFilterGroup('Solver');
    SOLVERS.forEach(function(s) {
      gSolver.appendChild(makeBtn(SOLVER_LABELS[s], state.solver === s, function() {
        state.solver = s; onChange(); render();
      }));
    });
    container.appendChild(gSolver);

    var gPrec = makeFilterGroup('Precision');
    ['f32', 'f64'].forEach(function(d) {
      gPrec.appendChild(makeBtn(d, state.dtype === d, function() {
        state.dtype = d; onChange(); render();
      }));
    });
    container.appendChild(gPrec);

    var gMode = makeFilterGroup('Mode');
    ['single', 'vmap'].forEach(function(m) {
      gMode.appendChild(makeBtn(m, state.mode === m, function() {
        state.mode = m; onChange(); render();
      }));
    });
    container.appendChild(gMode);

    var gDevice = makeFilterGroup('Device');
    ['cpu', 'gpu'].forEach(function(d) {
      gDevice.appendChild(makeBtn(d.toUpperCase(), state.device === d, function() {
        state.device = d; onChange(); render();
      }));
    });
    container.appendChild(gDevice);
  }

  render();
}

var solveTimeChart = null;
var speedupChart = null;
var pinvChart = null;

function initSolveTimeChart(state) {
  var ctx = document.getElementById('chart-solve-time');
  if (!ctx) return;
  if (solveTimeChart) { solveTimeChart.destroy(); solveTimeChart = null; }

  var filtered = filterData(state.solver, state.dtype, state.mode, state.device);

  var datasets = METHODS.map(function(method) {
    return {
      label: METHOD_LABELS[method],
      data: GRID_SIZES.map(function(n) {
        var e = filtered.find(function(e) { return e.grid_size === n && e.method === method; });
        return e ? e.time * 1000 : null;
      }),
      borderColor: METHOD_COLORS[method],
      backgroundColor: METHOD_COLORS[method],
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.1,
      spanGaps: true,
    };
  });

  solveTimeChart = new Chart(ctx, {
    type: 'line',
    data: { labels: GRID_SIZES, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Grid size (n)', font: { size: 12 } },
          ticks: { font: { size: 12 } },
        },
        y: {
          type: 'logarithmic',
          title: { display: false },
          ticks: {
            font: { size: 11 },
            callback: function(v) {
              if (v < 0.001) return (v * 1e6).toFixed(1) + ' µs';
              if (v < 1) return (v * 1000).toFixed(0) + ' ms';
              return v.toFixed(0) + ' ms';
            },
          },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            title: function(items) { return 'Grid size: ' + items[0].label; },
            label: function(ctx) {
              var v = ctx.parsed.y;
              if (v === null) return null;
              return ctx.dataset.label + ': ' + formatTime(v / 1000);
            },
          },
        },
        legend: { labels: { font: { size: 12 } } },
      },
    },
  });

  var note = document.getElementById('note-chart1');
  if (note) {
    note.textContent = 'Time in ms (log scale) · ' + SOLVER_LABELS[state.solver] + ' · ' + state.dtype + ' · ' + state.mode + ' · ' + state.device.toUpperCase() + (state.mode === 'vmap' ? ' · k=64 RHS total' : '');
  }
}

function initSpeedupChart(state) {
  var ctx = document.getElementById('chart-speedup');
  if (!ctx) return;
  if (speedupChart) { speedupChart.destroy(); speedupChart = null; }

  var filtered = filterData(state.solver, state.dtype, state.mode, state.device);

  var amjaxData = GRID_SIZES.map(function(n) {
    var pyamg = filtered.find(function(e) { return e.grid_size === n && e.method === 'pyamg'; });
    var amjax = filtered.find(function(e) { return e.grid_size === n && e.method === 'amjax'; });
    return (pyamg && amjax) ? parseFloat((pyamg.time / amjax.time).toFixed(2)) : null;
  });

  var amjaxPcgData = GRID_SIZES.map(function(n) {
    var pyamg = filtered.find(function(e) { return e.grid_size === n && e.method === 'pyamg'; });
    var pcg = filtered.find(function(e) { return e.grid_size === n && e.method === 'amjax_pcg'; });
    return (pyamg && pcg) ? parseFloat((pyamg.time / pcg.time).toFixed(2)) : null;
  });

  speedupChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: GRID_SIZES,
      datasets: [
        {
          label: 'AMJax vs PyAMG',
          data: amjaxData,
          borderColor: '#2563eb',
          backgroundColor: '#2563eb',
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.1,
          spanGaps: true,
        },
        {
          label: 'AMJax+PCG vs PyAMG',
          data: amjaxPcgData,
          borderColor: '#0ea5e9',
          backgroundColor: '#0ea5e9',
          borderWidth: 2,
          borderDash: [4, 2],
          pointRadius: 3,
          tension: 0.1,
          spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Grid size (n)', font: { size: 12 } },
          ticks: { font: { size: 12 } },
        },
        y: {
          title: { display: true, text: 'Speedup ×', font: { size: 12 } },
          ticks: { font: { size: 11 } },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            title: function(items) { return 'Grid size: ' + items[0].label; },
          },
        },
        legend: { labels: { font: { size: 12 } } },
      },
    },
  });

  var note = document.getElementById('note-chart2');
  if (note) {
    note.textContent = 'Speedup = PyAMG time / AMJax time · ' + SOLVER_LABELS[state.solver] + ' · ' + state.dtype + ' · ' + state.mode + ' · ' + state.device.toUpperCase();
  }
}

function renderTimesTable(state) {
  var table = document.getElementById('table-times');
  if (!table) return;

  var filtered = filterData(state.solver, state.dtype, state.mode, state.device);

  var bestPerGrid = {};
  GRID_SIZES.forEach(function(n) {
    var times = filtered.filter(function(e) { return e.grid_size === n; }).map(function(e) { return e.time; });
    if (times.length > 0) bestPerGrid[n] = Math.min.apply(null, times);
  });

  var html = '<thead><tr><th>Method</th>' + GRID_SIZES.map(function(n) { return '<th>n = ' + n + '</th>'; }).join('') + '</tr></thead><tbody>';

  METHODS.forEach(function(method, mi) {
    var bg = mi % 2 !== 0 ? 'background:rgba(0,0,0,0.015)' : '';
    html += '<tr style="' + bg + '"><td class="bm-td-label"><span class="bm-method-dot" style="background:' + METHOD_COLORS[method] + '"></span>' + METHOD_LABELS[method] + '</td>';
    GRID_SIZES.forEach(function(n) {
      var e = filtered.find(function(e) { return e.grid_size === n && e.method === method; });
      var isBest = e && e.time === bestPerGrid[n];
      var style = isBest ? 'font-weight:700;color:#2563eb' : '';
      html += '<td style="' + style + '">' + (e ? formatTime(e.time) : '—') + '</td>';
    });
    html += '</tr>';
  });

  html += '</tbody>';
  table.innerHTML = html;

  var note = document.getElementById('note-table1');
  if (note) {
    note.textContent = 'Fastest time per grid size highlighted in blue. ' + SOLVER_LABELS[state.solver] + ' · ' + state.dtype + ' · ' + state.mode + ' · ' + state.device.toUpperCase() + (state.mode === 'vmap' ? ' · total time for k=64 RHS' : ' · time for 1 solve') + '.';
  }
}

function renderResidualsTable(state) {
  var table = document.getElementById('table-residuals');
  if (!table) return;

  var filtered = filterData(state.solver, state.dtype, state.mode, state.device);

  var html = '<thead><tr><th>Method</th>' + GRID_SIZES.map(function(n) { return '<th>n = ' + n + '</th>'; }).join('') + '</tr></thead><tbody>';

  METHODS.forEach(function(method, mi) {
    var bg = mi % 2 !== 0 ? 'background:rgba(0,0,0,0.015)' : '';
    html += '<tr style="' + bg + '"><td class="bm-td-label">' + METHOD_LABELS[method] + '</td>';
    GRID_SIZES.forEach(function(n) {
      var e = filtered.find(function(e) { return e.grid_size === n && e.method === method; });
      var color = '';
      if (e) color = e.residual < 1e-9 ? '#16a34a' : e.residual > 0.01 ? '#dc2626' : '#f59e0b';
      html += '<td style="color:' + color + '">' + (e ? e.residual.toExponential(2) : '—') + '</td>';
    });
    html += '</tr>';
  });

  html += '</tbody>';
  table.innerHTML = html;

  var note = document.getElementById('note-table2');
  if (note) {
    note.innerHTML = 'Final relative residual ‖b − Ax‖ / ‖b‖. <span style="color:#16a34a">Green</span> = converged (&lt;1e-9), <span style="color:#f59e0b">amber</span> = partial, <span style="color:#dc2626">red</span> = not converged.';
  }
}

function initPinvChart(dtype, view) {
  var ctx = document.getElementById('chart-pinv');
  if (!ctx) return;
  if (pinvChart) { pinvChart.destroy(); pinvChart = null; }

  var raw = PINV_VS_JACOBI[dtype];
  var isLog = view === 'residual';

  var jacobiData = raw.map(function(r) {
    if (view === 'time') return r.jacobi_time !== null ? parseFloat((r.jacobi_time * 1000).toFixed(3)) : null;
    return r.jacobi_residual;
  });
  var pinvData = raw.map(function(r) {
    if (view === 'time') return r.pinv_time !== null ? parseFloat((r.pinv_time * 1000).toFixed(3)) : null;
    return r.pinv_residual;
  });

  pinvChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: raw.map(function(r) { return r.method; }),
      datasets: [
        { label: 'Jacobi', data: jacobiData, backgroundColor: '#2563eb', borderRadius: 3 },
        { label: 'Pinv',   data: pinvData,   backgroundColor: '#f59e0b', borderRadius: 3 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { font: { size: 12 } } },
        y: {
          type: isLog ? 'logarithmic' : 'linear',
          ticks: {
            font: { size: 11 },
            callback: function(v) {
              return view === 'time'
                ? (v < 1 ? (v * 1000).toFixed(0) + ' µs' : v.toFixed(0) + ' ms')
                : v.toExponential(1);
            },
          },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function(ctx) {
              var v = ctx.parsed.y;
              if (v === null || v === undefined) return null;
              return ctx.dataset.label + ': ' + (view === 'time' ? formatTime(v / 1000) : v.toExponential(2));
            },
          },
        },
        legend: { labels: { font: { size: 12 } } },
      },
    },
  });

  var note = document.getElementById('note-pinv');
  if (note) {
    note.textContent = (view === 'time' ? 'Solve time (ms)' : 'Final relative residual') + ' · Ruge–Stüben · n = 500 · ' + dtype + ' · single · GPU';
  }
}

function renderPinvTable(dtype) {
  var table = document.getElementById('table-pinv');
  if (!table) return;

  var raw = PINV_VS_JACOBI[dtype];

  var html = '<thead><tr><th>Method</th><th>Jacobi : time</th><th>Pinv : time</th><th>Jacobi : residual</th><th>Pinv : residual</th></tr></thead><tbody>';

  raw.forEach(function(row, i) {
    var bg = i % 2 !== 0 ? 'background:rgba(0,0,0,0.015)' : '';
    var jrColor = row.jacobi_residual !== null && row.jacobi_residual < 1e-9 ? '#16a34a' : '#dc2626';
    var prColor = row.pinv_residual !== null && row.pinv_residual < 1e-9 ? '#16a34a' : '#dc2626';
    html += '<tr style="' + bg + '">';
    html += '<td class="bm-td-label">' + row.method + '</td>';
    html += '<td>' + (row.jacobi_time !== null ? formatTime(row.jacobi_time) : '') + '</td>';
    html += '<td>' + (row.pinv_time !== null ? formatTime(row.pinv_time) : '') + '</td>';
    html += '<td style="color:' + jrColor + '">' + (row.jacobi_residual !== null ? row.jacobi_residual.toExponential(2) : '') + '</td>';
    html += '<td style="color:' + (row.pinv_residual !== null ? prColor : '') + '">' + (row.pinv_residual !== null ? row.pinv_residual.toExponential(2) : '') + '</td>';
    html += '</tr>';
  });

  html += '</tbody>';
  table.innerHTML = html;
}

function initBenchmarks() {
  var state1 = { solver: 'ruge_stuben', dtype: 'f64', mode: 'single', device: 'gpu' };
  var state2 = { solver: 'ruge_stuben', dtype: 'f64', mode: 'single', device: 'gpu' };
  var state3 = { solver: 'ruge_stuben', dtype: 'f64', mode: 'single', device: 'gpu' };
  var state4 = { solver: 'ruge_stuben', dtype: 'f64', mode: 'single', device: 'gpu' };
  var statePinv = { dtype: 'f64', view: 'time' };

  makeFilters('filters-chart1', state1, function() { initSolveTimeChart(state1); });
  initSolveTimeChart(state1);

  makeFilters('filters-chart2', state2, function() { initSpeedupChart(state2); });
  initSpeedupChart(state2);

  makeFilters('filters-table1', state3, function() { renderTimesTable(state3); });
  renderTimesTable(state3);

  makeFilters('filters-table2', state4, function() { renderResidualsTable(state4); });
  renderResidualsTable(state4);

  var pinvContainer = document.getElementById('filters-pinv');
  if (pinvContainer) {
    function renderPinvFilters() {
      pinvContainer.innerHTML = '';

      var gPrec = makeFilterGroup('Precision');
      ['f32', 'f64'].forEach(function(d) {
        gPrec.appendChild(makeBtn(d, statePinv.dtype === d, function() {
          statePinv.dtype = d;
          initPinvChart(statePinv.dtype, statePinv.view);
          renderPinvTable(statePinv.dtype);
          renderPinvFilters();
        }));
      });
      pinvContainer.appendChild(gPrec);

      var gView = makeFilterGroup('View');
      [['time', 'Time'], ['residual', 'Residual']].forEach(function(pair) {
        gView.appendChild(makeBtn(pair[1], statePinv.view === pair[0], function() {
          statePinv.view = pair[0];
          initPinvChart(statePinv.dtype, statePinv.view);
          renderPinvFilters();
        }));
      });
      pinvContainer.appendChild(gView);
    }
    renderPinvFilters();
  }

  initPinvChart(statePinv.dtype, statePinv.view);
  renderPinvTable(statePinv.dtype);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initBenchmarks);
} else {
  initBenchmarks();
}
