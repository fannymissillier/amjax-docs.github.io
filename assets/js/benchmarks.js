const SOLVERS = ['rootnode', 'ruge_stuben', 'smoothed_aggregation'];
const SOLVER_LABELS = {
  rootnode: 'Root Node',
  ruge_stuben: 'Ruge-Stüben',
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

// skipDtype: omit the precision filter (used when both dtypes are shown simultaneously)
function makeFilters(containerId, state, onChange, skipDtype) {
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

    if (!skipDtype) {
      var gPrec = makeFilterGroup('Precision');
      ['f32', 'f64'].forEach(function(d) {
        gPrec.appendChild(makeBtn(d, state.dtype === d, function() {
          state.dtype = d; onChange(); render();
        }));
      });
      container.appendChild(gPrec);
    }

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

var speedupChart = null;
var residualsChart = null;

function initSpeedupChart(state) {
  var ctx = document.getElementById('chart-speedup');
  if (!ctx) return;
  if (speedupChart) { speedupChart.destroy(); speedupChart = null; }

  var f64 = filterData(state.solver, 'f64', state.mode, state.device);
  var f32 = filterData(state.solver, 'f32', state.mode, state.device);

  function ratio(data, numMethod, denMethod) {
    return GRID_SIZES.map(function(n) {
      var num = data.find(function(e) { return e.grid_size === n && e.method === numMethod; });
      var den = data.find(function(e) { return e.grid_size === n && e.method === denMethod; });
      return (num && den) ? parseFloat((num.time / den.time).toFixed(3)) : null;
    });
  }

  function f64overf32(method) {
    return GRID_SIZES.map(function(n) {
      var e64 = f64.find(function(e) { return e.grid_size === n && e.method === method; });
      var e32 = f32.find(function(e) { return e.grid_size === n && e.method === method; });
      return (e64 && e32) ? parseFloat((e64.time / e32.time).toFixed(3)) : null;
    });
  }

  speedupChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: GRID_SIZES,
      datasets: [
        {
          label: 'AMJax / PyAMG',
          data: ratio(f64, 'pyamg', 'amjax'),
          borderColor: '#16a34a',
          backgroundColor: '#16a34a',
          borderWidth: 2, pointRadius: 3, tension: 0.1, spanGaps: true,
        },
        {
          label: 'AMJax+PCG / PyAMG+PCG',
          data: ratio(f64, 'pyamg_pcg', 'amjax_pcg'),
          borderColor: '#65a30d',
          backgroundColor: '#65a30d',
          borderDash: [4, 2],
          borderWidth: 2, pointRadius: 3, tension: 0.1, spanGaps: true,
        },
        {
          label: 'AMJax f32 / AMJax f64',
          data: f64overf32('amjax'),
          borderColor: '#2563eb',
          backgroundColor: '#2563eb',
          borderWidth: 2, pointRadius: 3, tension: 0.1, spanGaps: true,
        },
        {
          label: 'AMJax+PCG f32 / AMJax+PCG f64',
          data: f64overf32('amjax_pcg'),
          borderColor: '#0ea5e9',
          backgroundColor: '#0ea5e9',
          borderDash: [4, 2],
          borderWidth: 2, pointRadius: 3, tension: 0.1, spanGaps: true,
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
          title: { display: true, text: 'Speedup ratio', font: { size: 12 } },
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

  var note = document.getElementById('note-speedup');
  if (note) {
    note.textContent = SOLVER_LABELS[state.solver] + ' | ' + state.mode + ' | ' + state.device.toUpperCase() + ' | Blue: AMJax speedup over PyAMG (f64) | Green: f32 speedup over f64 (> 1 = f32 faster)';
  }
}

function initResidualsChart(state) {
  var ctx = document.getElementById('chart-residuals');
  if (!ctx) return;
  if (residualsChart) { residualsChart.destroy(); residualsChart = null; }

  var f64 = filterData(state.solver, 'f64', state.mode, state.device);
  var f32 = filterData(state.solver, 'f32', state.mode, state.device);

  var datasets = [
    {
      label: 'AMJax f64',
      data: GRID_SIZES.map(function(n) {
        var e = f64.find(function(e) { return e.grid_size === n && e.method === 'amjax'; });
        return e ? e.residual : null;
      }),
      backgroundColor: '#2563eb',
      borderRadius: 3,
    },
    {
      label: 'AMJax f32',
      data: GRID_SIZES.map(function(n) {
        var e = f32.find(function(e) { return e.grid_size === n && e.method === 'amjax'; });
        return e ? e.residual : null;
      }),
      backgroundColor: '#93c5fd',
      borderRadius: 3,
    },
    {
      label: 'AMJax+PCG f64',
      data: GRID_SIZES.map(function(n) {
        var e = f64.find(function(e) { return e.grid_size === n && e.method === 'amjax_pcg'; });
        return e ? e.residual : null;
      }),
      backgroundColor: '#0ea5e9',
      borderRadius: 3,
    },
    {
      label: 'AMJax+PCG f32',
      data: GRID_SIZES.map(function(n) {
        var e = f32.find(function(e) { return e.grid_size === n && e.method === 'amjax_pcg'; });
        return e ? e.residual : null;
      }),
      backgroundColor: '#7dd3fc',
      borderRadius: 3,
    },
  ];

  residualsChart = new Chart(ctx, {
    type: 'bar',
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
          title: { display: true, text: '||b - Ax|| / ||b||', font: { size: 12 } },
          ticks: {
            font: { size: 11 },
            callback: function(v) { return v.toExponential(0); },
          },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            title: function(items) { return 'Grid size: ' + items[0].label; },
            label: function(ctx) {
              var v = ctx.parsed.y;
              return ctx.dataset.label + ': ' + (v !== null ? v.toExponential(2) : '-');
            },
          },
        },
        legend: { labels: { font: { size: 11 } } },
      },
    },
  });

  var note = document.getElementById('note-residuals');
  if (note) {
    note.textContent = 'Solid = f64, dashed = f32. ' + SOLVER_LABELS[state.solver] + ' | ' + state.mode + ' | ' + state.device.toUpperCase();
  }
}

function initBenchmarks() {
  var stateSpeedup   = { solver: 'ruge_stuben', mode: 'single', device: 'gpu' };
  var stateResiduals = { solver: 'ruge_stuben', mode: 'single', device: 'gpu' };

  makeFilters('filters-speedup', stateSpeedup, function() { initSpeedupChart(stateSpeedup); }, true);
  initSpeedupChart(stateSpeedup);

  makeFilters('filters-residuals', stateResiduals, function() { initResidualsChart(stateResiduals); }, true);
  initResidualsChart(stateResiduals);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initBenchmarks);
} else {
  initBenchmarks();
}
