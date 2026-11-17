/**
 * NIDS Intelligence Dashboard — Bespoke Chart Engine
 * Plotly.js Visualizations + GSAP Entrance Animations
 *
 * Implements:
 * - displayModeBar: false (clean, bespoke chart styling with no default toolbar)
 * - Deep custom Plotly theming for Flight Tracker Dark Material
 * - Custom center cutout & annotations for Class Imbalance Donut
 * - Wide margins and rounded breathing room for Attack Vector Bar Chart
 * - Dynamic update of HorizonX telemetry progress indicators
 */

/* ── Plotly Bespoke Config: Toolbar Completely Hidden ──────────────────── */
const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: false, // Strict requirement: remove default hover toolbar
  displaylogo: false
};

/* ── Dark Base Theme Layout ────────────────────────────────────────────── */
const DARK_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font: {
    family: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    color:  '#8c96a8',
    size:   11
  }
};

const GRID_COLOR = 'rgba(255, 255, 255, 0.05)';
const ZERO_COLOR = 'rgba(255, 255, 255, 0.09)';

/* ── Utility Helpers ───────────────────────────────────────────────────── */
async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
    return await res.json();
  } catch (err) {
    console.error('[NIDS API Error]', err.message);
    return null;
  }
}

function fmt(n) {
  if (n === null || n === undefined) return '—';
  return Number(n).toLocaleString();
}

function el(id) {
  return document.getElementById(id);
}

/* ── Main Dashboard Initialization ─────────────────────────────────────── */
async function initDashboard() {

  /* 1. GSAP Entrance Animation (with reduced-motion guard) */
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (window.gsap && !prefersReducedMotion) {
    gsap.fromTo(
      '.animate-card',
      { opacity: 0, y: 16 },
      {
        opacity: 1,
        y: 0,
        duration: 0.5,
        stagger: 0.06,
        ease: 'power2.out',
        clearProps: 'all' // Prevents inline styles from ever trapping the DOM
      }
    );
  }

  /* 2. Health Check & Live Beacon */
  const statusDot   = el('statusDot');
  const statusText  = el('statusText');
  const emptyBanner = el('emptyStateBanner');

  const health = await fetchJSON('/api/health');
  if (health && health.db_exists) {
    if (statusDot)  statusDot.className = 'status-dot online';
    if (statusText) statusText.textContent = 'ONLINE (DEPLOY DB)';
    if (emptyBanner) emptyBanner.style.display = 'none';
  } else {
    if (statusDot)  statusDot.className = 'status-dot offline';
    if (statusText) statusText.textContent = 'NO DB DETECTED';
    if (emptyBanner) emptyBanner.style.display = 'flex';
  }

  /* 3. KPI Telemetry Metrics */
  const kpis = await fetchJSON('/api/kpis');
  if (kpis && kpis.ready) {
    const total = fmt(kpis.total_flows);

    if (el('valTotalFlows'))   el('valTotalFlows').textContent   = total;
    if (el('valBenignFlows'))  el('valBenignFlows').textContent  = fmt(kpis.benign_count);
    if (el('valAttackFlows'))  el('valAttackFlows').textContent  = fmt(kpis.attack_count);
    if (el('valAttackTypes'))  el('valAttackTypes').textContent  = fmt(kpis.unique_attack_types);

    const benignPct = (100 - kpis.attack_percentage).toFixed(1);
    const attackPct = Number(kpis.attack_percentage).toFixed(1);

    if (el('valBenignPct')) el('valBenignPct').textContent = `${benignPct}% Normal Operational Traffic`;
    if (el('valAttackPct')) el('valAttackPct').textContent = `${attackPct}% Threat Ratio (Class 1)`;

    /* Update Dynamic Progress Bar in Hero KPI Card */
    if (el('kpiFlowBarBenign')) el('kpiFlowBarBenign').style.width = `${benignPct}%`;
    if (el('kpiFlowBarAttack')) el('kpiFlowBarAttack').style.width = `${attackPct}%`;
    if (el('kpiFlowTextBenign')) el('kpiFlowTextBenign').textContent = `${benignPct}% Benign`;
    if (el('kpiFlowTextAttack')) el('kpiFlowTextAttack').textContent = `${attackPct}% Malicious`;

    /* Dynamic Sample Size Callouts in Navbar */
    if (el('sampleFlowsCount'))  el('sampleFlowsCount').textContent  = total;
    if (el('sampleCountBanner')) el('sampleCountBanner').textContent = total;
  }

  /* 4. Render All Visualizations in Parallel */
  await Promise.all([
    renderClassDistribution(),
    renderCorrelationHeatmap(),
    renderVolumetricScatter(),
    initMlDashboard()
  ]);
}

/* ══════════════════════════════════════════════════════════════════════
   VISUALIZATION 1: Binary Class Imbalance Donut + Attack Breakdown Bar
   ══════════════════════════════════════════════════════════════════════ */
async function renderClassDistribution() {
  const data = await fetchJSON('/api/class-distribution');
  if (!data || data.length === 0) return;

  let benignCount = 0;
  let attackCount = 0;
  const breakdown = [];

  data.forEach(item => {
    if (item.is_attack === 0 || item.label.toUpperCase() === 'BENIGN') {
      benignCount += item.count;
    } else {
      attackCount += item.count;
      breakdown.push({ label: item.label, count: item.count });
    }
  });

  const totalEvaluated = benignCount + attackCount;
  const benignRatio = totalEvaluated > 0 ? Math.round((benignCount / totalEvaluated) * 100) : 80;
  const attackRatio = totalEvaluated > 0 ? Math.round((attackCount / totalEvaluated) * 100) : 20;

  /* ── A) Donut Chart: Clean Center Cutout & Bespoke Center Readout ── */
  Plotly.newPlot('chartBinaryDist', [{
    labels: ['Benign Traffic', 'Malicious (Attack)'],
    values: [benignCount, attackCount],
    type: 'pie',
    hole: 0.72, // Clean deep center cutout
    marker: {
      colors: ['#10b981', '#f43f5e'],
      line: { color: '#14161c', width: 3 } // Matching dark card background gap
    },
    textinfo: 'label+percent',
    textposition: 'outside',
    outsidetextfont: { color: '#8c96a8', size: 11, family: 'Inter, sans-serif' },
    hoverinfo: 'label+value+percent'
  }], {
    ...DARK_LAYOUT,
    showlegend: false,
    margin: { t: 30, r: 45, b: 30, l: 45 },
    annotations: [
      {
        text: 'RATIO',
        x: 0.5,
        y: 0.58,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: {
          family: 'Inter, sans-serif',
          size: 10,
          color: '#64748b'
        }
      },
      {
        text: `<b>${benignRatio} / ${attackRatio}</b>`,
        x: 0.5,
        y: 0.44,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: {
          family: 'JetBrains Mono, monospace',
          size: 20,
          color: '#f1f3f8'
        }
      }
    ]
  }, PLOTLY_CONFIG);

  /* ── B) Horizontal Bar Chart: Bespoke Breathing Room & Margins ── */
  breakdown.sort((a, b) => a.count - b.count);

  const barColors = breakdown.map(item => {
    const lbl = item.label.toLowerCase();
    if (lbl.includes('ddos'))         return '#f59e0b'; // Neon amber
    if (lbl.includes('portscan'))     return '#06b6d4'; // Electric cyan
    if (lbl.includes('web'))          return '#f43f5e'; // Crimson
    if (lbl.includes('bot'))          return '#a855f7'; // Purple
    if (lbl.includes('patator'))      return '#38bdf8'; // Sky blue
    if (lbl.includes('hulk'))         return '#f97316'; // Orange
    if (lbl.includes('infiltration')) return '#ec4899'; // Pink
    return '#818cf8';                                   // Indigo
  });

  Plotly.newPlot('chartAttackBreakdown', [{
    y: breakdown.map(item => item.label),
    x: breakdown.map(item => item.count),
    type: 'bar',
    orientation: 'h',
    marker: {
      color: barColors,
      line: { color: 'rgba(255, 255, 255, 0.12)', width: 1 }
    },
    hovertemplate: '<b>%{y}</b><br>Detected flows: %{x:,}<extra></extra>'
  }], {
    ...DARK_LAYOUT,
    bargap: 0.35, // Generous breathing room between bars
    /* Left margin expanded to 220px so long labels like "Web Attack - Brute Force" are 100% visible */
    margin: { t: 15, r: 25, b: 40, l: 220 },
    xaxis: {
      title: { text: 'Flow Count', font: { size: 10, color: '#8c96a8' } },
      gridcolor: GRID_COLOR,
      zerolinecolor: ZERO_COLOR,
      tickfont: { color: '#8c96a8', size: 10 }
    },
    yaxis: {
      automargin: true,
      tickfont: { size: 11, color: '#e2e8f0', family: 'Inter, sans-serif' },
      gridcolor: GRID_COLOR
    }
  }, PLOTLY_CONFIG);
}

/* ══════════════════════════════════════════════════════════════════════
   VISUALIZATION 2: 13×13 Pearson Correlation Heatmap
   ══════════════════════════════════════════════════════════════════════ */
async function renderCorrelationHeatmap() {
  const data = await fetchJSON('/api/correlation-matrix');
  if (!data || !data.features || data.features.length === 0) return;

  const names = data.features.map(f =>
    f.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase())
  );

  const flatVals = data.matrix.flat().filter(v => v !== null && v !== undefined);
  const zMin = flatVals.length > 0 ? Math.max(0, Math.floor(Math.min(...flatVals) * 10) / 10 - 0.05) : 0.3;

  Plotly.newPlot('chartCorrelationHeatmap', [{
    z: data.matrix,
    x: names,
    y: names,
    type: 'heatmap',
    colorscale: [
      [0.0,  '#14161c'],  // Deep card neutral (minimum correlation)
      [0.25, '#1e293b'],  // Slate blue
      [0.5,  '#1d4ed8'],  // Rich royal blue
      [0.75, '#06b6d4'],  // Electric cyan
      [1.0,  '#a5f3fc']   // Bright specular cyan (perfect 1.0 correlation)
    ],
    zmin: zMin,
    zmax: 1.0,
    colorbar: {
      title: { text: 'Pearson (r)', font: { color: '#f1f3f8', size: 11 } },
      titleside: 'top',
      tickfont: { color: '#8c96a8', size: 10 },
      thickness: 12,
      len: 0.85
    },
    hovertemplate: '<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>'
  }], {
    ...DARK_LAYOUT,
    margin: { t: 25, r: 40, b: 140, l: 160 },
    xaxis: {
      tickangle: -40,
      tickfont: { color: '#8c96a8', size: 10 },
      gridcolor: 'rgba(255, 255, 255, 0.03)'
    },
    yaxis: {
      autorange: 'reversed',
      tickfont: { color: '#8c96a8', size: 10 },
      gridcolor: 'rgba(255, 255, 255, 0.03)'
    }
  }, PLOTLY_CONFIG);
}

/* ══════════════════════════════════════════════════════════════════════
   VISUALIZATION 3: Volumetric Profiling — Log-Log Scatter Plot
   ══════════════════════════════════════════════════════════════════════ */
async function renderVolumetricScatter() {
  const records = await fetchJSON('/api/scatter-sample');
  if (!records || records.length === 0) return;

  const COLOR_MAP = {
    BENIGN:       '#10b981',
    DDoS:         '#f59e0b',
    PortScan:     '#06b6d4',
    Bot:          '#ec4899',
    Infiltration: '#a855f7'
  };

  /* Group points by attack label */
  const groups = {};
  records.forEach(row => {
    const lbl = row.label || 'Unknown';
    if (!groups[lbl]) groups[lbl] = [];
    groups[lbl].push(row);
  });

  const traces = [];

  /* Benign flows (baseline background points) */
  if (groups['BENIGN']) {
    const rows = groups['BENIGN'];
    traces.push({
      name: 'BENIGN',
      x: rows.map(r => Math.max(1, r.flow_duration || 0)),
      y: rows.map(r => Math.max(0.01, r.flow_packets_s || 0.01)),
      mode: 'markers',
      type: 'scatter',
      marker: { color: COLOR_MAP.BENIGN, size: 5, opacity: 0.38 },
      text: rows.map(r => `Port: ${r.destination_port ?? 'N/A'}<br>Mean Pkt Len: ${r.packet_length_mean?.toFixed(1) ?? 'N/A'}`),
      hovertemplate: '<b>BENIGN</b><br>Duration: %{x:,} µs<br>Rate: %{y:,.2f} pkt/s<br>%{text}<extra></extra>'
    });
  }

  /* Malicious attack traces */
  for (const [lbl, rows] of Object.entries(groups)) {
    if (lbl.toUpperCase() === 'BENIGN') continue;
    const isDDoS = lbl.toLowerCase().includes('ddos');
    const color = isDDoS ? COLOR_MAP.DDoS : (COLOR_MAP[lbl] || '#818cf8');
    traces.push({
      name: lbl,
      x: rows.map(r => Math.max(1, r.flow_duration || 0)),
      y: rows.map(r => Math.max(0.01, r.flow_packets_s || 0.01)),
      mode: 'markers',
      type: 'scatter',
      marker: {
        color,
        size:    isDDoS ? 7 : 6,
        opacity: isDDoS ? 0.88 : 0.72,
        line:    isDDoS ? { color: '#ffffff', width: 0.7 } : undefined
      },
      text: rows.map(r => `Port: ${r.destination_port ?? 'N/A'}<br>Mean Pkt Len: ${r.packet_length_mean?.toFixed(1) ?? 'N/A'}`),
      hovertemplate: `<b>${lbl}</b><br>Duration: %{x:,} µs<br>Rate: %{y:,.2f} pkt/s<br>%{text}<extra></extra>`
    });
  }

  Plotly.newPlot('chartVolumetricScatter', traces, {
    ...DARK_LAYOUT,
    margin: { t: 30, r: 30, b: 60, l: 70 },
    xaxis: {
      title: { text: 'Flow Duration (µs, log scale)', font: { size: 11, color: '#8c96a8' } },
      type: 'log',
      gridcolor: GRID_COLOR,
      zerolinecolor: ZERO_COLOR,
      tickfont: { color: '#8c96a8', size: 10 }
    },
    yaxis: {
      title: { text: 'Packet Rate (pkt/s, log scale)', font: { size: 11, color: '#8c96a8' } },
      type: 'log',
      gridcolor: GRID_COLOR,
      zerolinecolor: ZERO_COLOR,
      tickfont: { color: '#8c96a8', size: 10 }
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.12,
      font: { color: '#cbd5e1', size: 11 }
    }
  }, PLOTLY_CONFIG);
}

/* ══════════════════════════════════════════════════════════════════════
   PHASE 2: ML EVALUATION DASHBOARD & PIPELINE TAB CONTROLLER
   ══════════════════════════════════════════════════════════════════════ */

window.switchPipelineTab = function(tab) {
  const panelEda = el('panelEda');
  const panelMl = el('panelMl');
  const btnEda = el('tabBtnEda');
  const btnMl = el('tabBtnMl');

  if (tab === 'eda') {
    if (panelEda) panelEda.style.display = 'block';
    if (panelMl) panelMl.style.display = 'none';
    if (btnEda) btnEda.classList.add('active');
    if (btnMl) btnMl.classList.remove('active');
    ['chartBinaryDist', 'chartAttackBreakdown', 'chartCorrelationHeatmap', 'chartVolumetricScatter'].forEach(id => {
      const node = document.getElementById(id);
      if (node && node.data) Plotly.Plots.resize(node);
    });
  } else {
    if (panelEda) panelEda.style.display = 'none';
    if (panelMl) panelMl.style.display = 'block';
    if (btnEda) btnEda.classList.remove('active');
    if (btnMl) btnMl.classList.add('active');
    ['chartMlRocCurve', 'chartMlFeatureImportance'].forEach(id => {
      const node = document.getElementById(id);
      if (node && node.data) Plotly.Plots.resize(node);
    });
  }
};

async function initMlDashboard() {
  const data = await fetchJSON('/api/ml/metrics');
  if (!data || !data.ready) {
    console.warn('[ML Dashboard] Evaluation metrics pending or not available yet.');
    return;
  }

  const metrics = data.metrics || {};
  const cm = data.confusion_matrix || {};
  const bestParams = data.best_params || {};

  /* 1. KPI Cards */
  if (el('mlValAccuracy'))  el('mlValAccuracy').textContent  = `${((metrics.accuracy || 0) * 100).toFixed(2)}%`;
  if (el('mlValPrecision')) el('mlValPrecision').textContent = `${((metrics.precision || 0) * 100).toFixed(2)}%`;
  if (el('mlValRecall'))    el('mlValRecall').textContent    = `${((metrics.recall || 0) * 100).toFixed(2)}%`;
  if (el('mlValF1'))        el('mlValF1').textContent        = `${((metrics.f1 || 0) * 100).toFixed(2)}%`;
  if (el('mlValCvF1'))      el('mlValCvF1').textContent      = `${((data.best_cv_f1 || 0) * 100).toFixed(2)}%`;
  if (el('mlValRocAuc'))    el('mlValRocAuc').textContent    = (metrics.roc_auc || 0).toFixed(4);
  if (el('mlValPrAuc'))     el('mlValPrAuc').textContent     = (metrics.pr_auc || 0).toFixed(4);

  /* 2. Confusion Matrix */
  if (el('cmValTN')) el('cmValTN').textContent = fmt(cm.tn);
  if (el('cmValFP')) el('cmValFP').textContent = fmt(cm.fp);
  if (el('cmValFN')) el('cmValFN').textContent = fmt(cm.fn);
  if (el('cmValTP')) el('cmValTP').textContent = fmt(cm.tp);

  if (el('cmTotalFlows'))   el('cmTotalFlows').textContent   = fmt(data.n_test || 5684);
  if (el('cmActualBenign')) el('cmActualBenign').textContent = fmt((cm.tn || 0) + (cm.fp || 0));
  if (el('cmActualAttack')) el('cmActualAttack').textContent = fmt((cm.tp || 0) + (cm.fn || 0));

  /* 3. ROC Curve Chart */
  if (data.roc_curve && data.roc_curve.fpr && data.roc_curve.tpr) {
    if (el('rocBadgeAuc')) el('rocBadgeAuc').textContent = (metrics.roc_auc || 0).toFixed(4);

    const rocTrace = {
      x: data.roc_curve.fpr,
      y: data.roc_curve.tpr,
      mode: 'lines',
      name: `Random Forest (AUC = ${(metrics.roc_auc || 0).toFixed(4)})`,
      line: {
        color: '#8b5cf6',
        width: 3,
        shape: 'spline'
      },
      fill: 'tozeroy',
      fillcolor: 'rgba(139, 92, 246, 0.12)',
      hovertemplate: 'FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra></extra>'
    };

    const baselineTrace = {
      x: [0, 1],
      y: [0, 1],
      mode: 'lines',
      name: 'Random Guess (AUC = 0.50)',
      line: {
        dash: 'dash',
        color: 'rgba(255, 255, 255, 0.25)',
        width: 1.5
      },
      hoverinfo: 'none'
    };

    Plotly.newPlot('chartMlRocCurve', [baselineTrace, rocTrace], {
      ...DARK_LAYOUT,
      margin: { t: 20, r: 20, b: 50, l: 60 },
      xaxis: {
        title: { text: 'False Positive Rate (FPR)', font: { size: 11, color: '#8c96a8' } },
        range: [-0.02, 1.02],
        gridcolor: GRID_COLOR,
        zerolinecolor: ZERO_COLOR,
        tickfont: { color: '#8c96a8', size: 10 }
      },
      yaxis: {
        title: { text: 'True Positive Rate (Recall / TPR)', font: { size: 11, color: '#8c96a8' } },
        range: [-0.02, 1.02],
        gridcolor: GRID_COLOR,
        zerolinecolor: ZERO_COLOR,
        tickfont: { color: '#8c96a8', size: 10 }
      },
      legend: {
        x: 0.35,
        y: 0.12,
        font: { color: '#cbd5e1', size: 11 },
        bgcolor: 'rgba(20, 22, 28, 0.7)'
      }
    }, PLOTLY_CONFIG);
  }

  /* 4. Top-20 Gini Feature Importance Horizontal Bar Chart */
  if (data.top_features && data.top_features.length > 0) {
    if (el('mlFeatureCountBadge')) el('mlFeatureCountBadge').textContent = data.num_features || 77;

    // Reverse array so rank 1 appears at the top of horizontal bar chart
    const reversed = data.top_features.slice().reverse();
    const xVals = reversed.map(f => f.importance);
    const yVals = reversed.map(f => f.feature);

    const barTrace = {
      x: xVals,
      y: yVals,
      type: 'bar',
      orientation: 'h',
      marker: {
        color: xVals,
        colorscale: [
          [0.0, '#0284c7'],
          [0.5, '#06b6d4'],
          [1.0, '#10b981']
        ],
        line: {
          color: 'rgba(255, 255, 255, 0.12)',
          width: 1
        }
      },
      hovertemplate: '<b>%{y}</b><br>Gini Importance: %{x:.6f}<extra></extra>'
    };

    Plotly.newPlot('chartMlFeatureImportance', [barTrace], {
      ...DARK_LAYOUT,
      margin: { t: 20, r: 30, b: 50, l: 200 },
      xaxis: {
        title: { text: 'Mean Decrease in Impurity (Gini Importance)', font: { size: 11, color: '#8c96a8' } },
        gridcolor: GRID_COLOR,
        zerolinecolor: ZERO_COLOR,
        tickfont: { color: '#8c96a8', size: 10 }
      },
      yaxis: {
        tickfont: { color: '#cbd5e1', size: 11, family: 'JetBrains Mono, monospace' },
        automargin: true
      }
    }, PLOTLY_CONFIG);
  }

  /* 5. Hyperparameter Badges & Methodology Note */
  if (el('hpEstimators')) el('hpEstimators').textContent = `n_estimators: ${bestParams.classifier__n_estimators ?? 100}`;
  if (el('hpDepth'))      el('hpDepth').textContent      = `max_depth: ${bestParams.classifier__max_depth ?? 'None'}`;
  if (el('hpSplit'))      el('hpSplit').textContent      = `min_samples_split: ${bestParams.classifier__min_samples_split ?? 2}`;
  if (el('hpWeight'))     el('hpWeight').textContent     = `class_weight: ${bestParams.classifier__class_weight ?? 'balanced'}`;
  if (el('hpCvF1'))       el('hpCvF1').textContent       = `CV F1: ${((data.best_cv_f1 || 0) * 100).toFixed(2)}%`;

  if (el('mlMethodologyNoteText') && data.methodological_note) {
    el('mlMethodologyNoteText').textContent = data.methodological_note;
  }
}

/* ── Responsive Reflow on Resize ────────────────────────────────────────── */
const CHART_CONTAINERS = [
  'chartBinaryDist',
  'chartAttackBreakdown',
  'chartCorrelationHeatmap',
  'chartVolumetricScatter',
  'chartMlRocCurve',
  'chartMlFeatureImportance'
];

window.addEventListener('resize', () => {
  CHART_CONTAINERS.forEach(id => {
    const node = document.getElementById(id);
    if (node && node.data) Plotly.Plots.resize(node);
  });
});

/* ── Boot ───────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', initDashboard);
