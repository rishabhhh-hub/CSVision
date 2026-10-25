/**
 * script.js — CSVision frontend controller
 * All API calls use fetch(). No data is hardcoded.
 */

/* ═══════════════════════════════════════════════
   State
══════════════════════════════════════════════ */
const state = {
  filename: "",
  numericCols: [],
  categoricalCols: [],
  allCols: [],
  modelTrained: false,
  trainMeta: null,   // { feature_cols, numeric_cols, categorical_cols, cat_unique, task_type }
};

/* ═══════════════════════════════════════════════
   Helpers
══════════════════════════════════════════════ */
function $id(id) { return document.getElementById(id); }

function showEl(id, show = true) {
  const el = $id(id);
  if (el) el.style.display = show ? "" : "none";
}

function setHTML(id, html) {
  const el = $id(id);
  if (el) el.innerHTML = html;
}

function showError(id, msg) {
  const el = $id(id);
  if (!el) return;
  el.textContent = msg;
  el.style.display = msg ? "" : "none";
}

function showLoading(id, show) {
  showEl(id, show);
}

async function apiFetch(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function fmt(val) {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") {
    if (Number.isInteger(val)) return val.toLocaleString();
    return val.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  return String(val);
}

/* ═══════════════════════════════════════════════
   Upload
══════════════════════════════════════════════ */
const fileInput  = $id("file-input");
const dropZone   = $id("drop-zone");

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
});

dropZone.addEventListener("dragover", e => {
  e.preventDefault(); dropZone.classList.add("drag-over");
  $id("drop-label").textContent = "Release to upload";
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
  $id("drop-label").textContent = "Drop CSV here";
});
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag-over");
  $id("drop-label").textContent = "Drop CSV here";
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelect(file);
});

function handleFileSelect(file) {
  showError("upload-error", "");
  if (!file.name.endsWith(".csv")) {
    showError("upload-error", "Please select a .csv file.");
    return;
  }
  // Show file info
  const sizeKB = (file.size / 1024).toFixed(1);
  $id("upload-filename").textContent = `📄 ${file.name}`;
  $id("upload-size").textContent = `${sizeKB} KB`;
  showEl("upload-info");
  uploadFile(file);
}

async function uploadFile(file) {
  showLoading("upload-loading", true);
  showError("upload-error", "");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const data = await apiFetch("/api/upload", { method: "POST", body: formData });
    state.filename = data.filename;
    showLoading("upload-loading", false);
    onUploadSuccess();
  } catch (err) {
    showLoading("upload-loading", false);
    showError("upload-error", err.message);
  }
}

async function onUploadSuccess() {
  // Update header
  $id("header-filename").textContent = state.filename;
  showEl("header-file");
  // Switch view
  showEl("upload-section", false);
  showEl("dashboard");
  // Load everything
  await Promise.all([loadOverview(), loadInsights()]);
  loadAutoCharts();
  populateMlTargets();
}

/* ── Reset ────────────────────────────────────── */
$id("btn-reset").addEventListener("click", async () => {
  try { await apiFetch("/api/reset", { method: "POST" }); } catch (_) {}
  // Reset UI
  showEl("upload-section");
  showEl("dashboard", false);
  showEl("header-file", false);
  showError("upload-error", "");
  showEl("upload-info", false);
  fileInput.value = "";
  state.modelTrained = false;
  state.trainMeta = null;
  showEl("pred-no-model");
  showEl("pred-form-section", false);
  showEl("ml-results", false);
  switchTab("overview");
});

/* ═══════════════════════════════════════════════
   Tabs
══════════════════════════════════════════════ */
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tabName));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${tabName}`));
}

/* ═══════════════════════════════════════════════
   Overview tab
══════════════════════════════════════════════ */
async function loadOverview() {
  // Stat cards
  try {
    const overview = await apiFetch("/api/overview");
    renderStatCards(overview);
  } catch (err) { console.error("overview", err); }

  // Preview table (default 10 rows)
  await loadPreview(10);
}

function renderStatCards(o) {
  const cards = [
    { label: "Rows",           value: fmt(o.rows) },
    { label: "Columns",        value: fmt(o.cols) },
    { label: "Missing Values", value: fmt(o.missing_total) },
    { label: "Duplicates",     value: fmt(o.duplicates) },
    { label: "Numeric Cols",   value: fmt(o.numeric_count) },
    { label: "Categorical Cols", value: fmt(o.categorical_count) },
  ];
  $id("stat-cards").innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="stat-label">${c.label}</div>
      <div class="stat-value">${c.value}</div>
    </div>`).join("");
}

async function loadPreview(n) {
  try {
    const data = await apiFetch(`/api/preview?n=${n}`);
    renderPreviewTable(data);
  } catch (err) { console.error("preview", err); }
}

function renderPreviewTable(data) {
  const { columns, dtypes, rows, total_rows } = data;
  if (!rows.length) { setHTML("preview-table-wrap", "<p class='muted'>No rows to display.</p>"); return; }

  const headerRow = columns.map(c => `<th>${c}<br><span class="badge-dtype">${dtypes[c]}</span></th>`).join("");
  const bodyRows  = rows.map(r =>
    "<tr>" + columns.map(c => {
      const v = r[c];
      const isNum = typeof v === "number";
      return `<td class="${isNum ? "num" : ""}">${v === null ? "—" : fmt(v)}</td>`;
    }).join("") + "</tr>"
  ).join("");

  setHTML("preview-table-wrap", `<table><thead><tr>${headerRow}</tr></thead><tbody>${bodyRows}</tbody></table>`);
  $id("preview-meta").textContent = `Showing ${rows.length} of ${total_rows.toLocaleString()} rows.`;
}

$id("preview-rows").addEventListener("change", e => loadPreview(parseInt(e.target.value)));

/* ── Auto charts ─────────────────────────────── */
async function loadAutoCharts() {
  const container = $id("auto-charts");
  container.innerHTML = '<p class="muted">Generating charts…</p>';

  const charts = [
    { type: "auto_histograms", label: "Numeric Distributions" },
    { type: "auto_boxplots",   label: "Outlier Check" },
    { type: "correlation",     label: "Correlation Heatmap" },
  ];

  const results = await Promise.allSettled(
    charts.map(c => apiFetch("/api/visualize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chart_type: c.type }),
    }).then(d => ({ ...c, image: d.image })))
  );

  container.innerHTML = "";
  results.forEach((r, i) => {
    if (r.status === "fulfilled") {
      const { label, image } = r.value;
      const div = document.createElement("div");
      div.className = "chart-card";
      div.innerHTML = `<div class="chart-card-label">${label}</div><img src="data:image/png;base64,${image}" alt="${label}" />`;
      container.appendChild(div);
    }
  });
  if (!container.children.length) container.innerHTML = '<p class="muted">No numeric columns to chart.</p>';
}

/* ═══════════════════════════════════════════════
   Insights tab
══════════════════════════════════════════════ */
async function loadInsights() {
  try {
    const data = await apiFetch("/api/insights");
    state.numericCols     = data.numeric_cols || [];
    state.categoricalCols = data.categorical_cols || [];
    state.allCols         = [...state.numericCols, ...state.categoricalCols];

    renderInsightBullets(data.insights || []);
    renderDescribeTable(data.describe || []);
    renderMissingTable(data.missing || []);
    renderCatSummary(data.categorical_summary || []);
    populateVizSelectors();
  } catch (err) { console.error("insights", err); }
}

function renderInsightBullets(bullets) {
  if (!bullets.length) { setHTML("insight-bullets", "<p class='muted'>No insights generated.</p>"); return; }
  const ul = document.createElement("ul");
  ul.className = "insight-bullets";
  bullets.forEach(b => { const li = document.createElement("li"); li.textContent = b; ul.appendChild(li); });
  $id("insight-bullets").innerHTML = "";
  $id("insight-bullets").appendChild(ul);
}

function renderDescribeTable(describe) {
  if (!describe.length) { setHTML("describe-table", "<p class='muted'>No numeric columns.</p>"); return; }
  const keys = Object.keys(describe[0]).filter(k => k !== "column");
  const header = `<tr><th>Column</th>${keys.map(k => `<th>${k}</th>`).join("")}</tr>`;
  const rows = describe.map(r =>
    `<tr><td>${r.column}</td>${keys.map(k => `<td class="num">${fmt(r[k])}</td>`).join("")}</tr>`
  ).join("");
  setHTML("describe-table", `<table><thead>${header}</thead><tbody>${rows}</tbody></table>`);
}

function renderMissingTable(missing) {
  if (!missing.length) {
    setHTML("missing-table", "<p class='muted'>✅ No missing values found.</p>");
    return;
  }
  const rows = missing.map(m => {
    const cls = m.pct > 30 ? "missing-row-danger" : m.pct > 10 ? "missing-row-warn" : "";
    return `<tr class="${cls}"><td>${m.column}</td><td class="num">${m.missing}</td><td class="num">${m.pct}%</td></tr>`;
  }).join("");
  setHTML("missing-table", `<table><thead><tr><th>Column</th><th>Missing</th><th>%</th></tr></thead><tbody>${rows}</tbody></table>`);
}

function renderCatSummary(cats) {
  const wrap = $id("cat-summary");
  if (!cats.length) { wrap.innerHTML = "<p class='muted'>No categorical columns.</p>"; return; }
  wrap.innerHTML = "";
  cats.forEach(c => {
    const div = document.createElement("div"); div.className = "cat-item";
    let barsHTML = "";
    if (c.value_counts) {
      const max = c.value_counts[0].count;
      barsHTML = `<div class="cat-bar-wrap">${c.value_counts.slice(0, 8).map(vc => {
        const pct = max > 0 ? (vc.count / max * 100).toFixed(1) : 0;
        return `<div class="cat-bar-row">
          <span class="cat-bar-label">${vc.value}</span>
          <div class="cat-bar-bg"><div class="cat-bar-fill" style="width:${pct}%"></div></div>
          <span class="cat-bar-count">${vc.count}</span>
        </div>`;
      }).join("")}</div>`;
    }
    div.innerHTML = `<div class="cat-item-header">${c.column} <span style="color:var(--text3);font-weight:400">(${c.unique} unique)</span></div>${barsHTML}`;
    wrap.appendChild(div);
  });
}

/* ═══════════════════════════════════════════════
   Visualizations tab
══════════════════════════════════════════════ */
function populateVizSelectors() {
  const chartType  = $id("viz-chart-type").value;
  const colSelect  = $id("viz-column");
  const yColSelect = $id("viz-y-col");

  // Populate x-col options based on chart type
  const isCatChart = ["count_plot"].includes(chartType);
  const isNumChart = ["histogram","boxplot","line"].includes(chartType);
  const isTwoCol   = ["scatter","line","bar_chart"].includes(chartType);

  let colOptions = state.allCols;
  if (isCatChart) colOptions = state.categoricalCols.length ? state.categoricalCols : state.allCols;
  if (isNumChart) colOptions = state.numericCols.length ? state.numericCols : state.allCols;

  colSelect.innerHTML = colOptions.map(c => `<option value="${c}">${c}</option>`).join("");

  if (isTwoCol) {
    const yOptions = state.numericCols.map(c => `<option value="${c}">${c}</option>`).join("");
    yColSelect.innerHTML = yOptions;
    showEl("viz-y-wrap", true);
    $id("viz-col-label").textContent = "X Column";
  } else {
    showEl("viz-y-wrap", false);
    $id("viz-col-label").textContent = "Column";
  }

  const isCorr = chartType === "correlation";
  showEl("viz-col-wrap", !isCorr);
}

$id("viz-chart-type").addEventListener("change", populateVizSelectors);

$id("btn-generate-chart").addEventListener("click", async () => {
  const chartType = $id("viz-chart-type").value;
  const column    = $id("viz-column").value;
  const yCol      = $id("viz-y-col").value;
  const display   = $id("chart-display");

  display.innerHTML = `<div class="loading-bar" style="max-width:400px;margin:1rem auto"><div class="loading-bar-inner"></div></div>`;

  const body = { chart_type: chartType };
  if (chartType !== "correlation") {
    body.column = column;
    body.x_col  = column;
  }
  if (["scatter","line","bar_chart"].includes(chartType)) body.y_col = yCol;

  try {
    const data = await apiFetch("/api/visualize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    display.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="${chartType}" />`;
  } catch (err) {
    display.innerHTML = `<p style="color:var(--danger);font-size:.9rem">⚠ ${err.message}</p>`;
  }
});

/* ═══════════════════════════════════════════════
   Machine Learning tab
══════════════════════════════════════════════ */
const METRIC_DESCS = {
  "Accuracy":  "Overall fraction of correct predictions.",
  "Precision": "Fraction of positive predictions that were correct.",
  "Recall":    "Fraction of actual positives correctly identified.",
  "F1 Score":  "Harmonic mean of Precision and Recall.",
  "R² Score":  "Proportion of variance explained (1 = perfect).",
  "MAE":       "Average absolute difference between actual and predicted.",
  "MSE":       "Average squared difference (penalises large errors).",
  "RMSE":      "Square root of MSE — in the same units as the target.",
};

async function populateMlTargets() {
  const sel = $id("ml-target");
  sel.innerHTML = state.allCols.map(c => `<option value="${c}">${c}</option>`).join("");
  await loadAlgorithms();
}

async function loadAlgorithms() {
  try {
    const data = await apiFetch("/api/algorithms");
    updateAlgorithmList(data);
  } catch (_) {}
}

function updateAlgorithmList(data) {
  const task = $id("ml-task").value;
  const list = data[task] || [];
  $id("ml-algo").innerHTML = list.map(a => `<option value="${a.key}">${a.label}</option>`).join("");
}

$id("ml-task").addEventListener("change", async () => {
  try {
    const data = await apiFetch("/api/algorithms");
    updateAlgorithmList(data);
  } catch (_) {}
});

$id("btn-train").addEventListener("click", async () => {
  showError("train-error", "");
  showEl("ml-results", false);
  showLoading("train-loading", true);

  const body = {
    target_col:    $id("ml-target").value,
    task_type:     $id("ml-task").value,
    algorithm_key: $id("ml-algo").value,
    test_size:     parseFloat($id("ml-split").value),
  };

  try {
    const result = await apiFetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    showLoading("train-loading", false);
    renderMlResults(result);
    // Store for predictions
    state.modelTrained = true;
    state.trainMeta = {
      feature_cols:     result.feature_cols,
      numeric_cols:     result.numeric_cols,
      categorical_cols: result.categorical_cols,
      cat_unique:       result.cat_unique,
      task_type:        result.task_type,
    };
    buildPredictionForm();
  } catch (err) {
    showLoading("train-loading", false);
    showError("train-error", err.message);
  }
});

function renderMlResults(r) {
  $id("ml-result-title").textContent = `${r.algorithm_label} — ${r.task_type === "classification" ? "Classification" : "Regression"} Results`;

  // Metric cards
  const cards = Object.entries(r.metrics).map(([label, val]) => `
    <div class="metric-card">
      <div class="m-label">${label}</div>
      <div class="m-value">${typeof val === "number" ? val.toFixed(3) : val}</div>
      <div class="m-desc">${METRIC_DESCS[label] || ""}</div>
    </div>`).join("");
  setHTML("metric-cards", cards);

  $id("split-info").textContent =
    `Train set: ${r.train_rows} rows — Test set: ${r.test_rows} rows (${Math.round(r.test_size * 100)}% held out)`;

  if (r.chart) {
    setHTML("ml-chart", `<img src="data:image/png;base64,${r.chart}" alt="Model chart" />`);
  } else {
    setHTML("ml-chart", "");
  }

  showEl("ml-results");
}

/* ═══════════════════════════════════════════════
   Predictions tab
══════════════════════════════════════════════ */
function buildPredictionForm() {
  const meta = state.trainMeta;
  if (!meta) return;

  showEl("pred-no-model", false);
  showEl("pred-form-section");
  showEl("pred-result", false);
  showError("pred-error", "");

  const form = $id("pred-form");
  form.innerHTML = "";

  meta.feature_cols.forEach(col => {
    const group = document.createElement("div");
    group.className = "form-group";
    const label = document.createElement("label");
    label.textContent = col;

    let input;
    if (meta.categorical_cols.includes(col)) {
      // Dropdown for categorical
      input = document.createElement("select");
      input.name = col;
      const opts = (meta.cat_unique[col] || []);
      input.innerHTML = opts.map(v => `<option value="${v}">${v}</option>`).join("");
    } else {
      // Number input for numeric
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.name = col;
      input.placeholder = "Enter value";
    }

    group.appendChild(label);
    group.appendChild(input);
    form.appendChild(group);
  });
}

$id("btn-predict").addEventListener("click", async () => {
  showError("pred-error", "");
  showEl("pred-result", false);

  const meta   = state.trainMeta;
  const inputs = $id("pred-form").querySelectorAll("input, select");

  const features = {};
  let valid = true;
  inputs.forEach(inp => {
    if (inp.type === "number" && inp.value === "") {
      inp.style.borderColor = "var(--danger)";
      valid = false;
    } else {
      inp.style.borderColor = "";
      features[inp.name] = inp.type === "number" ? parseFloat(inp.value) : inp.value;
    }
  });

  if (!valid) {
    showError("pred-error", "Please fill in all numeric fields.");
    return;
  }

  try {
    const result = await apiFetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const display = typeof result.prediction === "number"
      ? result.prediction.toLocaleString(undefined, { maximumFractionDigits: 4 })
      : result.prediction;
    setHTML("pred-result-value", display);
    showEl("pred-result");
  } catch (err) {
    showError("pred-error", err.message);
  }
});
