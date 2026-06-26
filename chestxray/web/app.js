"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  body: document.body,
  dropzone: $("dropzone"),
  fileInput: $("fileInput"),
  dropzoneEmpty: $("dropzoneEmpty"),
  uploadHint: $("uploadHint"),
  dropHintSub: $("dropHintSub"),
  previewWrap: $("previewWrap"),
  previewImg: $("previewImg"),
  batchStrip: $("batchStrip"),
  analyzeBtn: $("analyzeBtn"),
  clearBtn: $("clearBtn"),
  fileName: $("fileName"),
  placeholder: $("placeholder"),
  results: $("results"),
  batchResults: $("batchResults"),
  errorBox: $("errorBox"),
  statusPill: $("statusPill"),
  statusText: $("statusText"),
  resultTools: $("resultTools"),
  copyBtn: $("copyBtn"),
  downloadBtn: $("downloadBtn"),
  pdfBtn: $("pdfBtn"),
  fhirBtn: $("fhirBtn"),
  exportAllBtn: $("exportAllBtn"),
  triageChip: $("triageChip"),
  autoAnalyze: $("autoAnalyze"),
  demoNormalBtn: $("demoNormalBtn"),
  demoPneuBtn: $("demoPneuBtn"),
  batchFilters: $("batchFilters"),
  exportBatchJsonBtn: $("exportBatchJsonBtn"),
  exportFlaggedBtn: $("exportFlaggedBtn"),
  oodBanner: $("oodBanner"),
  feedback: $("feedback"),
  fbYes: $("fbYes"),
  fbNo: $("fbNo"),
  feedbackThanks: $("feedbackThanks"),
  refreshHistory: $("refreshHistory"),
  historyTbody: $("historyTbody"),
  refreshReview: $("refreshReview"),
  reviewTbody: $("reviewTbody"),
  patientRef: $("patientRef"),
  studyDate: $("studyDate"),
  caseNotes: $("caseNotes"),
  createCaseBtn: $("createCaseBtn"),
  caseIdChip: $("caseIdChip"),
  triageBanner: $("triageBanner"),
  qualityBanner: $("qualityBanner"),
  verdict: $("verdict"),
  verdictLabel: $("verdictLabel"),
  verdictIcon: $("verdictIcon"),
  advisory: $("advisory"),
  confidenceValue: $("confidenceValue"),
  gaugeFill: $("gaugeFill"),
  threshold: $("threshold"),
  thresholdVal: $("thresholdVal"),
  opacity: $("opacity"),
  opacityVal: $("opacityVal"),
  camBase: $("camBase"),
  camHeat: $("camHeat"),
  probs: $("probs"),
  imgOriginal: $("imgOriginal"),
  imgHeatmap: $("imgHeatmap"),
  imgOverlay: $("imgOverlay"),
  resultMeta: $("resultMeta"),
  batchSummary: $("batchSummary"),
  batchProgress: $("batchProgress"),
  batchProgressFill: $("batchProgressFill"),
  batchProgressText: $("batchProgressText"),
  batchTbody: $("batchTbody"),
  exportCsvBtn: $("exportCsvBtn"),
  btnLabel: document.querySelector(".btn-label"),
  spinner: document.querySelector("#analyzeBtn .spinner"),
  lightbox: $("lightbox"),
  lightboxImg: $("lightboxImg"),
  lightboxClose: $("lightboxClose"),
  toast: $("toast"),
  themeToggle: $("themeToggle"),
  userChip: $("userChip"),
  userDropdown: $("userDropdown"),
  userAvatar: $("userAvatar"),
  userName: $("userName"),
  userRole: $("userRole"),
  openLoginBtn: $("openLoginBtn"),
  logoutBtn: $("logoutBtn"),
  loginModal: $("loginModal"),
  loginModalClose: $("loginModalClose"),
  loginForm: $("loginForm"),
  loginUser: $("loginUser"),
  loginPass: $("loginPass"),
  loginError: $("loginError"),
  loginSub: $("loginSub"),
  datasetNote: $("datasetNote"),
  heroModels: $("heroModels"),
};

const GAUGE_C = 327;
const LOW_CONF = 0.65;
const ICONS = {
  NORMAL: `<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>`,
  PNEUMONIA: `<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
};

let mode = "single";
let singleFile = null;
let batchFiles = [];
let modelMeta = null;
let currentProbs = null; // {NORMAL, PNEUMONIA}
let currentUncertainty = null;
let lastSingle = null; // for copy/report
let batchRows = [];
let activeCaseId = null;
let authEnabled = false;
let liveMetrics = null;
let batchFilter = "all";
const BATCH_CONCURRENCY = 4;

const DATASET_BENCHMARKS = {
  kaggle: { live: true, note: "Metrics from your trained model on the Kaggle Chest X-Ray Pneumonia dataset." },
  nih: {
    live: false,
    note: "Reference benchmarks from published ResNet studies on NIH ChestX-ray14 (multi-label). Not trained by this deployment.",
    metrics: { accuracy: 0.71, f1: 0.68, precision: 0.70, recall: 0.69, auc: 0.78 },
  },
  rsna: {
    live: false,
    note: "Reference benchmarks for RSNA Pneumonia Detection (literature). Not trained by this deployment.",
    metrics: { accuracy: 0.88, f1: 0.86, precision: 0.87, recall: 0.88, auc: 0.92 },
  },
  covid: {
    live: false,
    note: "Reference benchmarks for COVID-19 CXR classification (literature). Not trained by this deployment.",
    metrics: { accuracy: 0.91, f1: 0.90, precision: 0.89, recall: 0.92, auc: 0.94 },
  },
};

/* fetch wrapper that attaches an optional API key (set via localStorage). */
function apiFetch(url, options = {}) {
  const key = localStorage.getItem("pulmo-api-key");
  const token = localStorage.getItem("pulmo-jwt");
  const headers = new Headers(options.headers || {});
  if (key) headers.set("X-API-Key", key);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(url, { ...options, headers });
}

/* ---------- Productivity settings ---------- */
function initProductivitySettings() {
  const savedThreshold = localStorage.getItem("pulmo-threshold");
  if (savedThreshold && els.threshold) {
    els.threshold.value = savedThreshold;
    els.thresholdVal.textContent = `${savedThreshold}%`;
  }
  const savedAuto = localStorage.getItem("pulmo-auto-analyze");
  if (els.autoAnalyze && savedAuto !== null) {
    els.autoAnalyze.checked = savedAuto === "true";
  }
  els.threshold?.addEventListener("input", () => {
    localStorage.setItem("pulmo-threshold", els.threshold.value);
    applyThreshold();
  });
  els.autoAnalyze?.addEventListener("change", () => {
    localStorage.setItem("pulmo-auto-analyze", String(els.autoAnalyze.checked));
  });
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
      e.preventDefault();
      if (!els.analyzeBtn.disabled) analyze();
    }
    if (e.key === "Escape") clearAll();
  });
  document.querySelectorAll(".filter-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      batchFilter = btn.dataset.filter || "all";
      document.querySelectorAll(".filter-chip").forEach((b) => b.classList.toggle("active", b === btn));
      applyBatchFilter();
    });
  });
}

function isAutoAnalyze() {
  return els.autoAnalyze ? els.autoAnalyze.checked : false;
}

function setTriageChip(triage, reasons) {
  if (!els.triageChip) return;
  const t = triage || "routine";
  els.triageChip.className = `triage-chip ${t}`;
  els.triageChip.textContent = t.toUpperCase();
  els.triageChip.title = reasons?.length ? reasons.join("; ") : "No manual review required";
}

async function ensureCaseId() {
  if (activeCaseId) return activeCaseId;
  const patient_ref = els.patientRef?.value?.trim();
  if (!patient_ref) return null;
  try {
    const res = await apiFetch("/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_ref,
        study_date: els.studyDate?.value || null,
        notes: els.caseNotes?.value || "",
      }),
    });
    const data = await res.json();
    if (!res.ok) return null;
    activeCaseId = data.case_id;
    if (els.caseIdChip) {
      els.caseIdChip.hidden = false;
      els.caseIdChip.textContent = `Case: ${activeCaseId}`;
    }
    return activeCaseId;
  } catch {
    return null;
  }
}

async function loadDemoSample(label) {
  hideError();
  try {
    const res = await fetch(`/demo/sample?label=${label}`);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      return showError(data.detail || "Demo sample unavailable. Run setup-data first.");
    }
    const blob = await res.blob();
    const file = new File([blob], `demo_${label.toLowerCase()}.jpeg`, { type: blob.type || "image/jpeg" });
    setMode("single");
    handleFiles([file], { skipAuto: true });
    await ensureCaseId();
    analyzeSingle();
  } catch {
    showError("Could not load demo sample.");
  }
}

function isValidUpload(f) {
  if (f.type.startsWith("image/")) return true;
  return /\.dcm$/i.test(f.name);
}

/* ---------- Auth / user profile ---------- */
function initials(name) {
  const parts = String(name || "G").trim().split(/\s+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return (parts[0].slice(0, 2) || "G").toUpperCase();
}

function roleLabel(role) {
  if (role === "clinician") return "Clinician · MD";
  if (role === "admin") return "Administrator";
  if (role === "viewer") return "Viewer";
  return "Demo mode";
}

function setUserDisplay(name, role) {
  if (!els.userName) return;
  els.userName.textContent = name;
  els.userRole.textContent = roleLabel(role);
  if (els.userAvatar) els.userAvatar.textContent = initials(name);
  const loggedIn = role !== "guest";
  if (els.logoutBtn) els.logoutBtn.hidden = !loggedIn;
  if (els.openLoginBtn) els.openLoginBtn.hidden = loggedIn;
}

function loadStoredUser() {
  const raw = localStorage.getItem("pulmo-user");
  if (raw) {
    try {
      const u = JSON.parse(raw);
      setUserDisplay(u.name || "Guest", u.role || "guest");
      return;
    } catch {
      /* fall through */
    }
  }
  setUserDisplay("Guest", "guest");
}

function openLoginModal() {
  if (els.loginModal) els.loginModal.hidden = false;
  if (els.loginError) els.loginError.hidden = true;
  if (els.loginSub) {
    els.loginSub.textContent = authEnabled
      ? "Sign in to record analyses under your identity."
      : "Auth is disabled on this server — sign-in stores a demo profile locally.";
  }
}

function closeLoginModal() {
  if (els.loginModal) els.loginModal.hidden = true;
}

async function handleLogin(e) {
  e.preventDefault();
  const username = els.loginUser?.value?.trim();
  const password = els.loginPass?.value || "";
  if (!username) return;

  try {
    const res = await fetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Login failed");

    const displayName = username.charAt(0).toUpperCase() + username.slice(1);
    const role = data.role || "clinician";
    if (data.token) localStorage.setItem("pulmo-jwt", data.token);
    localStorage.setItem("pulmo-user", JSON.stringify({ name: displayName, role }));
    setUserDisplay(displayName, role);
    closeLoginModal();
    showToast(`Signed in as ${displayName}`);
  } catch (err) {
    if (els.loginError) {
      els.loginError.hidden = false;
      els.loginError.textContent = err.message || "Invalid credentials";
    }
  }
}

function logoutUser() {
  localStorage.removeItem("pulmo-jwt");
  localStorage.removeItem("pulmo-user");
  setUserDisplay("Guest", "guest");
  if (els.logoutBtn) els.logoutBtn.hidden = true;
  if (els.openLoginBtn) els.openLoginBtn.hidden = false;
  showToast("Signed out");
  if (els.userDropdown) els.userDropdown.hidden = true;
}

function initUserMenu() {
  loadStoredUser();
  els.userChip?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!els.userDropdown) return;
    const open = els.userDropdown.hidden;
    els.userDropdown.hidden = !open;
    els.userChip?.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", () => {
    if (els.userDropdown) els.userDropdown.hidden = true;
  });
  els.openLoginBtn?.addEventListener("click", () => {
    els.userDropdown.hidden = true;
    openLoginModal();
  });
  els.logoutBtn?.addEventListener("click", logoutUser);
  els.loginModalClose?.addEventListener("click", closeLoginModal);
  els.loginModal?.addEventListener("click", (e) => {
    if (e.target === els.loginModal) closeLoginModal();
  });
  els.loginForm?.addEventListener("submit", handleLogin);
}

/* ---------- Dataset benchmark tabs ---------- */
function applyDatasetTab(key) {
  document.querySelectorAll(".dataset-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.dataset === key);
  });
  const cfg = DATASET_BENCHMARKS[key] || DATASET_BENCHMARKS.kaggle;
  if (els.datasetNote) els.datasetNote.textContent = cfg.note;
  const warn = $("benchmarkWarn");
  if (warn) warn.hidden = key === "kaggle";

  if (key === "kaggle" && liveMetrics) {
    setMetric("accuracy", liveMetrics.accuracy);
    setMetric("f1", liveMetrics.f1);
    setMetric("precision", liveMetrics.precision);
    setMetric("recall", liveMetrics.recall);
    setMetric("auc", liveMetrics.auc, true);
    return;
  }
  if (key === "kaggle") {
    showTargetMetrics();
    return;
  }
  const m = cfg.metrics || {};
  setMetric("accuracy", m.accuracy);
  setMetric("f1", m.f1);
  setMetric("precision", m.precision);
  setMetric("recall", m.recall);
  setMetric("auc", m.auc, true);
}

function initDatasetTabs() {
  document.querySelectorAll(".dataset-tab").forEach((tab) => {
    tab.addEventListener("click", () => applyDatasetTab(tab.dataset.dataset));
  });
  $("switchKaggleTab")?.addEventListener("click", () => applyDatasetTab("kaggle"));
  applyDatasetTab("kaggle");
}

async function loadModelCount() {
  try {
    const res = await fetch("/models");
    if (res.ok) {
      const data = await res.json();
      const n = (data.models || []).length || 1;
      if (els.heroModels) els.heroModels.textContent = String(n);
    }
  } catch {
    /* ignore */
  }
}

/* ---------- Theme ---------- */
function initTheme() {
  const saved = localStorage.getItem("pulmo-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  setTheme(saved || (prefersDark ? "dark" : "light"));
}
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("pulmo-theme", theme);
  if (els.themeToggle) els.themeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
}
function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  setTheme(next);
}

/* ---------- Scroll reveal ---------- */
function initReveal() {
  const targets = document.querySelectorAll(
    ".section-head, .step-card, .feature-card, .metric-card, .tech-row, .faq-item, .hero-stats"
  );
  if (!("IntersectionObserver" in window)) {
    targets.forEach((t) => t.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  targets.forEach((t, i) => {
    t.classList.add("reveal");
    t.style.transitionDelay = `${(i % 4) * 80}ms`;
    io.observe(t);
  });
}

/* ---------- Model status ---------- */
async function checkStatus() {
  try {
    const res = await apiFetch("/metadata");
    if (res.ok) {
      modelMeta = await res.json();
      authEnabled = !!modelMeta.auth_enabled;
      setStatus("ready", `Model ready · ${modelMeta.device.toUpperCase()}`);
      loadStoredUser();
    } else {
      modelMeta = null;
      setStatus("offline", "No model loaded");
    }
  } catch {
    modelMeta = null;
    setStatus("offline", "Server offline");
  }
}
function setStatus(state, text) {
  els.statusPill.dataset.state = state;
  els.statusText.textContent = text;
}

/* ---------- Performance metrics ---------- */
async function loadMetrics() {
  try {
    const res = await fetch("/metrics");
    if (res.ok) {
      liveMetrics = await res.json();
      setMetric("accuracy", liveMetrics.accuracy);
      setMetric("f1", liveMetrics.f1);
      setMetric("precision", liveMetrics.precision);
      setMetric("recall", liveMetrics.recall);
      setMetric("auc", liveMetrics.auc, true);
      const lead = $("perfLead");
      if (lead) lead.textContent = "Live metrics from your most recent evaluation run.";
      if (typeof liveMetrics.accuracy === "number") {
        $("heroAcc").textContent = `${(liveMetrics.accuracy * 100).toFixed(1)}%`;
      }
      const active = document.querySelector(".dataset-tab.active");
      if (active?.dataset.dataset === "kaggle") applyDatasetTab("kaggle");
      loadErrorAnalysis();
    } else {
      showTargetMetrics();
    }
  } catch {
    showTargetMetrics();
  }
}
function setMetric(key, value, isAuc = false) {
  const el = document.querySelector(`.metric-val[data-k="${key}"]`);
  if (el && typeof value === "number") {
    el.textContent = isAuc ? value.toFixed(3) : `${(value * 100).toFixed(1)}%`;
  }
}
function showTargetMetrics() {
  const lead = $("perfLead");
  if (lead)
    lead.textContent =
      "No evaluation yet — showing typical targets. Train a model to see live results.";
  const targets = { accuracy: "~94%", f1: "~0.94", precision: "~94%", recall: "~95%", auc: "~0.98" };
  Object.entries(targets).forEach(([k, v]) => {
    const el = document.querySelector(`.metric-val[data-k="${k}"]`);
    if (el) el.textContent = v;
  });
  const errBox = $("errorAnalysis");
  if (errBox) errBox.hidden = true;
}

async function loadErrorAnalysis() {
  const box = $("errorAnalysis");
  const tbody = $("errorTable")?.querySelector("tbody");
  const lead = $("errorAnalysisLead");
  if (!box || !tbody) return;
  try {
    const res = await fetch("/metrics/errors");
    if (!res.ok) {
      box.hidden = true;
      return;
    }
    const data = await res.json();
    const rows = data.misclassified || [];
    tbody.innerHTML = "";
    rows.slice(0, 12).forEach((row) => {
      const tr = document.createElement("tr");
      const path = row.path || "";
      const short = path.split(/[/\\]/).slice(-2).join("/");
      tr.innerHTML = `<td>${row.true_label}</td><td>${row.predicted_label}</td><td>${(row.confidence * 100).toFixed(1)}%</td><td title="${path}">${short}</td>`;
      tbody.appendChild(tr);
    });
    if (lead) {
      lead.textContent = `${data.count || rows.length} misclassified on the test set (showing up to 12).`;
    }
    box.hidden = rows.length === 0;
  } catch {
    box.hidden = true;
  }
}

/* ---------- Mode ---------- */
function setMode(next) {
  mode = next;
  els.body.dataset.mode = next;
  document.querySelectorAll(".mode-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === next)
  );
  els.fileInput.multiple = next === "batch";
  els.uploadHint.textContent =
    next === "batch"
      ? "Drop multiple chest X-rays, or click to select several."
      : "Drag a chest X-ray here, or click to browse.";
  els.dropHintSub.textContent =
    next === "batch" ? "Select many PNG/JPG files" : "PNG, JPG up to ~10 MB";
  clearAll();
}

/* ---------- File handling ---------- */
function handleFiles(fileList, opts = {}) {
  const files = Array.from(fileList).filter(isValidUpload);
  if (!files.length) {
    showError("Please select valid image files (PNG, JPG, or DICOM .dcm).");
    return;
  }
  hideError();
  if (mode === "single") {
    singleFile = files[0];
    els.previewImg.src = URL.createObjectURL(singleFile);
    els.previewWrap.hidden = false;
    els.batchStrip.hidden = true;
    els.dropzoneEmpty.hidden = true;
    els.fileName.textContent = `${singleFile.name} · ${(singleFile.size / 1024).toFixed(0)} KB`;
    els.analyzeBtn.disabled = false;
    els.clearBtn.disabled = false;
    if (isAutoAnalyze() && !opts.skipAuto) {
      ensureCaseId().then(() => analyzeSingle());
    }
  } else {
    batchFiles = files;
    renderBatchStrip();
    els.batchStrip.hidden = false;
    els.previewWrap.hidden = true;
    els.dropzoneEmpty.hidden = true;
    els.fileName.textContent = `${files.length} file${files.length > 1 ? "s" : ""} selected`;
    els.analyzeBtn.disabled = false;
    els.clearBtn.disabled = false;
    if (isAutoAnalyze() && !opts.skipAuto) analyzeBatch();
  }
}

function renderBatchStrip() {
  els.batchStrip.innerHTML = "";
  const max = 11;
  batchFiles.slice(0, max).forEach((f) => {
    const img = document.createElement("img");
    img.className = "thumb";
    img.src = URL.createObjectURL(f);
    els.batchStrip.appendChild(img);
  });
  if (batchFiles.length > max) {
    const more = document.createElement("div");
    more.className = "more";
    more.textContent = `+${batchFiles.length - max}`;
    els.batchStrip.appendChild(more);
  }
}

function clearAll() {
  singleFile = null;
  batchFiles = [];
  els.fileInput.value = "";
  els.previewWrap.hidden = true;
  els.previewImg.removeAttribute("src");
  els.batchStrip.hidden = true;
  els.batchStrip.innerHTML = "";
  els.dropzoneEmpty.hidden = false;
  els.analyzeBtn.disabled = true;
  els.clearBtn.disabled = true;
  els.fileName.textContent = "";
  els.results.hidden = true;
  els.batchResults.hidden = true;
  if (els.batchFilters) els.batchFilters.hidden = true;
  els.resultTools.hidden = true;
  els.oodBanner.hidden = true;
  els.qualityBanner.hidden = true;
  els.triageBanner.hidden = true;
  setTriageChip("routine");
  els.placeholder.hidden = false;
  hideError();
}

/* ---------- Analyze dispatch ---------- */
async function analyze() {
  if (mode === "single") return analyzeSingle();
  return analyzeBatch();
}

/* ---------- Single ---------- */
async function analyzeSingle() {
  if (!singleFile) return;
  setLoading(true);
  hideError();
  await ensureCaseId();
  const form = new FormData();
  form.append("file", singleFile);
  if (activeCaseId) form.append("case_id", activeCaseId);
  const t0 = performance.now();
  try {
    const res = await apiFetch("/predict/analyze", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) return showError(data.detail || `Request failed (${res.status}).`);
    renderSingle(data, performance.now() - t0);
    loadHistory();
    loadReviewQueue();
  } catch {
    showError("Could not reach the server. Is it still running?");
  } finally {
    setLoading(false);
  }
}

function renderSingle(data, elapsedMs) {
  els.placeholder.hidden = true;
  els.batchResults.hidden = true;
  els.results.hidden = false;
  els.resultTools.hidden = false;

  currentProbs = data.probabilities;
  currentUncertainty = data.uncertainty || null;
  lastSingle = { file: singleFile ? singleFile.name : "", ...data, threshold: +els.threshold.value / 100 };

  // OOD input guard banner
  if (data.input_check && data.input_check.is_xray_like === false) {
    els.oodBanner.hidden = false;
    els.oodBanner.innerHTML = `⚠ ${data.input_check.reason || "Input may not be a chest X-ray."}`;
  } else {
    els.oodBanner.hidden = true;
  }

  // Image quality
  if (data.quality && data.quality.warnings && data.quality.warnings.length) {
    els.qualityBanner.hidden = false;
    els.qualityBanner.innerHTML = `📷 ${data.quality.warnings.join(" ")} (score ${(data.quality.score * 100).toFixed(0)}%)`;
  } else {
    els.qualityBanner.hidden = true;
  }

  // Clinical triage
  setTriageChip(data.triage, data.triage_reasons);
  if (data.triage && data.triage !== "routine") {
    els.triageBanner.hidden = false;
    const label = data.triage === "reject" ? "REJECT — do not trust result" : "REVIEW REQUIRED";
    els.triageBanner.className = `triage-banner triage-${data.triage}`;
    els.triageBanner.textContent = `${label}${data.triage_reasons?.length ? ": " + data.triage_reasons.join("; ") : ""}`;
  } else {
    els.triageBanner.hidden = true;
  }

  // reset feedback widget
  els.feedback.hidden = false;
  els.feedbackThanks.hidden = true;
  els.fbYes.disabled = false;
  els.fbNo.disabled = false;

  // probability bars (model output, fixed)
  els.probs.innerHTML = "";
  Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, value]) => {
      const cls = name === "NORMAL" ? "normal" : name === "PNEUMONIA" ? "pneumonia" : "other";
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-meta"><span class="name">${name}</span>
        <span class="val">${(value * 100).toFixed(1)}%</span></div>
        <div class="prob-track"><div class="prob-fill ${cls}"></div></div>`;
      els.probs.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".prob-fill").style.width = `${value * 100}%`;
      });
    });

  // grad-cam images + interactive blend
  els.camBase.src = data.images.original;
  els.camHeat.src = data.images.heatmap;
  els.imgOriginal.src = data.images.original;
  els.imgHeatmap.src = data.images.heatmap;
  els.imgOverlay.src = data.images.overlay;
  applyOpacity();

  // meta strip
  const device = modelMeta ? modelMeta.device.toUpperCase() : "—";
  let metaHtml = `
    <span class="chip"><b>device</b> ${device}</span>
    <span class="chip"><b>classes</b> ${Object.keys(data.probabilities).length}</span>
    <span class="chip"><b>latency</b> ${elapsedMs.toFixed(0)} ms</span>
    <span class="chip"><b>model</b> ResNet-50</span>`;
  if (currentUncertainty) {
    metaHtml += `<span class="chip"><b>entropy</b> ${currentUncertainty.entropy.toFixed(3)}</span>`;
  }
  if (data.triage) {
    metaHtml += `<span class="chip"><b>triage</b> ${data.triage.toUpperCase()}</span>`;
  }
  if (data.case_id) {
    metaHtml += `<span class="chip"><b>case</b> ${data.case_id}</span>`;
  }
  els.resultMeta.innerHTML = metaHtml;

  applyThreshold(); // sets verdict/gauge/advisory
}

function applyThreshold() {
  if (!currentProbs) return;
  const t = +els.threshold.value / 100;
  els.thresholdVal.textContent = `${els.threshold.value}%`;

  const pneu = currentProbs.PNEUMONIA ?? 0;
  const label = pneu >= t ? "PNEUMONIA" : "NORMAL";
  const confidence = currentProbs[label] ?? Math.max(...Object.values(currentProbs));

  els.verdict.dataset.label = label;
  els.verdictLabel.textContent = label;
  els.verdictIcon.innerHTML = ICONS[label] || "";

  requestAnimationFrame(() => {
    els.gaugeFill.style.strokeDashoffset = String(GAUGE_C * (1 - confidence));
  });
  els.confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;

  const maxProb = Math.max(...Object.values(currentProbs));
  const borderline = Math.abs(pneu - t) < 0.1;
  const abstain = currentUncertainty && currentUncertainty.abstain;
  if (abstain) {
    els.advisory.hidden = false;
    els.advisory.textContent = "⚠ Model uncertain — recommend radiologist review";
  } else if (maxProb < LOW_CONF || borderline) {
    els.advisory.hidden = false;
    els.advisory.textContent = borderline
      ? "⚠ Borderline call — recommend expert review"
      : "⚠ Low confidence — recommend expert review";
  } else {
    els.advisory.hidden = true;
  }
  if (lastSingle) lastSingle.threshold = t;
}

function applyOpacity() {
  els.camHeat.style.opacity = String(+els.opacity.value / 100);
  els.opacityVal.textContent = `${els.opacity.value}%`;
}

/* ---------- Batch (parallel) ---------- */
async function analyzeBatch() {
  if (!batchFiles.length) return;
  setLoading(true);
  hideError();
  await ensureCaseId();
  els.placeholder.hidden = true;
  els.results.hidden = true;
  els.resultTools.hidden = true;
  els.batchResults.hidden = false;
  if (els.batchFilters) els.batchFilters.hidden = false;
  batchRows = [];
  batchFilter = "all";
  document.querySelectorAll(".filter-chip").forEach((b) => b.classList.toggle("active", b.dataset.filter === "all"));

  els.batchTbody.innerHTML = "";
  batchFiles.forEach((f, i) => {
    const tr = document.createElement("tr");
    tr.id = `brow-${i}`;
    tr.dataset.triage = "pending";
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td><div class="file-cell"><img src="${URL.createObjectURL(f)}" alt=""/><span>${f.name}</span></div></td>
      <td><span class="badge pending">…</span></td>
      <td><span class="badge pending">pending</span></td>
      <td class="conf-cell"><span class="mini-val">—</span></td>`;
    els.batchTbody.appendChild(tr);
  });

  const total = batchFiles.length;
  els.batchProgress.hidden = false;
  updateProgress(0, total);

  let nextIndex = 0;
  let done = 0;

  async function processOne(i) {
    const f = batchFiles[i];
    const form = new FormData();
    form.append("file", f);
    if (activeCaseId) form.append("case_id", activeCaseId);
    try {
      const res = await apiFetch("/predict", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        markRowError(i, data.detail || `Error ${res.status}`);
      } else {
        batchRows.push({ file: f.name, ...data });
        fillRow(i, data);
      }
    } catch {
      markRowError(i, "request failed");
    }
    done += 1;
    updateProgress(done, total);
    updateSummary();
  }

  async function worker() {
    while (nextIndex < total) {
      const i = nextIndex++;
      await processOne(i);
    }
  }

  const workers = Math.min(BATCH_CONCURRENCY, total);
  await Promise.all(Array.from({ length: workers }, () => worker()));

  setLoading(false);
  applyBatchFilter();
  showToast(`Analyzed ${batchRows.length}/${total} · ${BATCH_CONCURRENCY} parallel`);
  loadHistory();
  loadReviewQueue();
}

function triageBadge(triage) {
  const t = triage || "routine";
  return `<span class="badge triage-${t}">${t.toUpperCase()}</span>`;
}

function fillRow(i, data) {
  const tr = $(`brow-${i}`);
  if (!tr) return;
  const cls = data.label === "PNEUMONIA" ? "pneumonia" : "normal";
  const pct = (data.confidence * 100).toFixed(1);
  const triage = data.triage || "routine";
  tr.dataset.triage = triage;
  tr.children[2].innerHTML = triageBadge(triage);
  const warn =
    data.input_check && data.input_check.is_xray_like === false
      ? ` <span class="flag-dot warn" title="May not be a chest X-ray"></span>`
      : "";
  tr.children[3].innerHTML = `<span class="badge ${cls}">${data.label}</span>${warn}`;
  tr.children[4].innerHTML = `
    <div class="mini-track"><div class="mini-fill" style="width:${pct}%"></div></div>
    <span class="mini-val">${pct}%</span>`;
}

function markRowError(i, msg) {
  const tr = $(`brow-${i}`);
  if (!tr) return;
  tr.dataset.triage = "error";
  tr.children[2].innerHTML = `<span class="badge pending">—</span>`;
  tr.children[3].innerHTML = `<span class="badge pending">error</span>`;
  tr.children[4].innerHTML = `<span class="mini-val" title="${msg}">—</span>`;
}

function applyBatchFilter() {
  document.querySelectorAll("#batchTbody tr").forEach((tr) => {
    const t = tr.dataset.triage || "routine";
    const show = batchFilter === "all" || t === batchFilter;
    tr.classList.toggle("filtered-out", !show);
  });
}

function updateProgress(done, total) {
  els.batchProgressFill.style.width = `${(done / total) * 100}%`;
  els.batchProgressText.textContent = `${done} / ${total}`;
  if (done === total) setTimeout(() => (els.batchProgress.hidden = true), 600);
}

function updateSummary() {
  const normal = batchRows.filter((r) => r.label === "NORMAL").length;
  const pneu = batchRows.filter((r) => r.label === "PNEUMONIA").length;
  const review = batchRows.filter((r) => r.triage === "review").length;
  const reject = batchRows.filter((r) => r.triage === "reject").length;
  els.batchSummary.innerHTML = `
    <div class="summary-card"><div class="num">${batchRows.length}</div><div class="lbl">Analyzed</div></div>
    <div class="summary-card normal"><div class="num">${normal}</div><div class="lbl">Normal</div></div>
    <div class="summary-card pneumonia"><div class="num">${pneu}</div><div class="lbl">Pneumonia</div></div>
    <div class="summary-card"><div class="num">${review}</div><div class="lbl">Review</div></div>
    <div class="summary-card"><div class="num">${reject}</div><div class="lbl">Reject</div></div>`;
}

function batchExportRows(flaggedOnly = false) {
  return flaggedOnly
    ? batchRows.filter((r) => r.triage === "review" || r.triage === "reject")
    : batchRows;
}

function exportCsv(flaggedOnly = false) {
  const rows = batchExportRows(flaggedOnly);
  if (!rows.length) return showToast(flaggedOnly ? "No flagged studies" : "Nothing to export");
  const header = ["file", "triage", "prediction", "confidence", "prob_normal", "prob_pneumonia", "study_id"];
  const lines = [header.join(",")];
  for (const r of rows) {
    lines.push(
      [
        `"${r.file}"`,
        r.triage || "routine",
        r.label,
        r.confidence.toFixed(4),
        (r.probabilities.NORMAL ?? 0).toFixed(4),
        (r.probabilities.PNEUMONIA ?? 0).toFixed(4),
        r.id || "",
      ].join(",")
    );
  }
  const name = flaggedOnly ? "pulmoscan_flagged.csv" : "pulmoscan_batch_results.csv";
  downloadBlob(lines.join("\n"), name, "text/csv");
  showToast("CSV exported");
}

function exportBatchJson(flaggedOnly = false) {
  const rows = batchExportRows(flaggedOnly);
  if (!rows.length) return showToast(flaggedOnly ? "No flagged studies" : "Nothing to export");
  const payload = {
    app: "PulmoScan",
    generated_at: new Date().toISOString(),
    case_id: activeCaseId,
    threshold: +els.threshold.value / 100,
    studies: rows,
    disclaimer: "Research/education only. Not a medical device.",
  };
  downloadBlob(JSON.stringify(payload, null, 2), flaggedOnly ? "pulmoscan_flagged.json" : "pulmoscan_batch.json", "application/json");
  showToast("JSON exported");
}

async function downloadFhir() {
  if (!lastSingle?.id) return showToast("Analyze an image first");
  const ref = encodeURIComponent(els.patientRef?.value?.trim() || "anonymous");
  try {
    const res = await apiFetch(`/studies/${lastSingle.id}/fhir?patient_ref=${ref}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "FHIR export failed");
    downloadBlob(JSON.stringify(data, null, 2), `pulmoscan_${lastSingle.id}_fhir.json`, "application/json");
    showToast("FHIR report downloaded");
  } catch (e) {
    showError(e.message || "Could not export FHIR");
  }
}

async function exportAllReports() {
  if (!lastSingle || !singleFile) return showToast("Analyze an image first");
  showToast("Exporting PDF, JSON, FHIR…");
  downloadReport();
  await downloadPdf();
  await downloadFhir();
}

/* ---------- PDF report ---------- */
async function downloadPdf() {
  if (!singleFile) return showToast("Analyze an image first");
  showToast("Generating PDF…");
  const form = new FormData();
  form.append("file", singleFile);
  form.append("threshold", String(+els.threshold.value / 100));
  try {
    const res = await apiFetch("/predict/report", { method: "POST", body: form });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      return showError(data.detail || `PDF failed (${res.status}).`);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "pulmoscan_report.pdf";
    a.click();
    URL.revokeObjectURL(url);
    showToast("PDF downloaded");
  } catch {
    showError("Could not generate PDF report.");
  }
}

/* ---------- Feedback ---------- */
async function sendFeedback(isCorrect) {
  if (!lastSingle) return;
  const predicted = els.verdictLabel.textContent;
  const classes = Object.keys(lastSingle.probabilities);
  const correct = isCorrect ? predicted : classes.find((c) => c !== predicted) || predicted;
  try {
    const res = await apiFetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: lastSingle.id,
        predicted_label: predicted,
        correct_label: correct,
      }),
    });
    if (!res.ok) throw new Error();
    els.fbYes.disabled = true;
    els.fbNo.disabled = true;
    els.feedbackThanks.hidden = false;
    showToast("Feedback recorded");
  } catch {
    showToast("Could not send feedback");
  }
}

/* ---------- History ---------- */
async function loadHistory() {
  try {
    const res = await apiFetch("/history?limit=25");
    if (!res.ok) return;
    const data = await res.json();
    renderHistory(data.items || []);
  } catch {
    /* ignore */
  }
}
function renderHistory(items) {
  if (!els.historyTbody) return;
  if (!items.length) {
    els.historyTbody.innerHTML = `<tr><td colspan="5" class="history-empty">No analyses yet.</td></tr>`;
    return;
  }
  els.historyTbody.innerHTML = items
    .map((it) => {
      const cls = it.label === "PNEUMONIA" ? "pneumonia" : "normal";
      const pct = typeof it.confidence === "number" ? `${(it.confidence * 100).toFixed(1)}%` : "—";
      const time = it.ts ? new Date(it.ts).toLocaleString() : "—";
      let flags = "";
      if (it.is_xray_like === false) flags += `<span class="flag-dot warn" title="OOD input"></span>`;
      if (it.abstain) flags += `<span class="flag-dot abstain" title="Abstained"></span>`;
      if (it.triage && it.triage !== "routine") flags += `<span class="flag-dot triage" title="${it.triage}"></span>`;
      return `<tr>
        <td><span class="mini-val">${time}</span></td>
        <td><div class="file-cell"><span>${it.file || "—"}</span></div></td>
        <td><span class="badge ${cls}">${it.label || "—"}</span></td>
        <td><span class="mini-val">${pct}</span></td>
        <td>${flags || "—"}</td>
      </tr>`;
    })
    .join("");
}

/* ---------- Cases ---------- */
async function createCase() {
  const patient_ref = els.patientRef?.value?.trim() || "anonymous";
  try {
    const res = await apiFetch("/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_ref,
        study_date: els.studyDate?.value || null,
        notes: els.caseNotes?.value || "",
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed");
    activeCaseId = data.case_id;
    els.caseIdChip.hidden = false;
    els.caseIdChip.textContent = `Case: ${activeCaseId}`;
    showToast("Case created");
  } catch (e) {
    showToast("Could not create case");
  }
}

/* ---------- Review queue ---------- */
async function loadReviewQueue() {
  try {
    const res = await apiFetch("/review/queue?limit=25");
    if (!res.ok) return;
    const data = await res.json();
    renderReviewQueue(data.items || []);
  } catch {
    /* ignore */
  }
}

function renderReviewQueue(items) {
  if (!els.reviewTbody) return;
  if (!items.length) {
    els.reviewTbody.innerHTML = `<tr><td colspan="6" class="history-empty">No pending reviews.</td></tr>`;
    return;
  }
  els.reviewTbody.innerHTML = items
    .map((it) => {
      const time = it.created_at ? new Date(it.created_at).toLocaleString() : "—";
      const cls = it.label === "PNEUMONIA" ? "pneumonia" : "normal";
      return `<tr>
        <td><span class="mini-val">${time}</span></td>
        <td><span class="mini-val">${it.study_id || "—"}</span></td>
        <td><span class="badge ${cls}">${it.label || "—"}</span></td>
        <td><span class="badge triage-${it.triage || "review"}">${(it.triage || "review").toUpperCase()}</span></td>
        <td><span class="mini-val">${it.reason || "—"}</span></td>
        <td>
          <button class="tool-btn resolve-btn" data-id="${it.review_id}" data-decision="agree">Agree</button>
          <button class="tool-btn resolve-btn" data-id="${it.review_id}" data-decision="disagree">Disagree</button>
        </td>
      </tr>`;
    })
    .join("");
  els.reviewTbody.querySelectorAll(".resolve-btn").forEach((btn) => {
    btn.addEventListener("click", () => resolveReview(btn.dataset.id, btn.dataset.decision));
  });
}

async function resolveReview(reviewId, decision) {
  try {
    const res = await apiFetch(`/review/${reviewId}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, notes: "" }),
    });
    if (!res.ok) throw new Error();
    showToast("Review resolved");
    loadReviewQueue();
  } catch {
    showToast("Could not resolve review");
  }
}

/* ---------- Copy / Report ---------- */
function copyResults() {
  if (!lastSingle) return;
  const payload = {
    file: lastSingle.file,
    prediction: els.verdictLabel.textContent,
    threshold: lastSingle.threshold,
    confidence: lastSingle.confidence,
    probabilities: lastSingle.probabilities,
  };
  navigator.clipboard
    .writeText(JSON.stringify(payload, null, 2))
    .then(() => showToast("Copied to clipboard"))
    .catch(() => showToast("Copy failed"));
}
function downloadReport() {
  if (!lastSingle) return;
  const report = {
    app: "PulmoScan",
    generated_at: new Date().toISOString(),
    file: lastSingle.file,
    study_id: lastSingle.id,
    case_id: lastSingle.case_id || activeCaseId,
    triage: lastSingle.triage,
    triage_reasons: lastSingle.triage_reasons,
    quality: lastSingle.quality,
    model: { device: modelMeta?.device ?? null, architecture: "ResNet-50" },
    prediction: els.verdictLabel.textContent,
    decision_threshold: lastSingle.threshold,
    confidence: lastSingle.confidence,
    probabilities: lastSingle.probabilities,
    uncertainty: lastSingle.uncertainty,
    disclaimer: "Research/education only. Not a medical device.",
  };
  downloadBlob(JSON.stringify(report, null, 2), "pulmoscan_report.json", "application/json");
  showToast("Report downloaded");
}
function downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/* ---------- UI helpers ---------- */
function setLoading(loading) {
  els.analyzeBtn.disabled = loading || (mode === "single" ? !singleFile : !batchFiles.length);
  els.clearBtn.disabled = loading;
  els.btnLabel.textContent = loading ? "Analyzing…" : "Analyze";
  els.spinner.hidden = !loading;
  els.dropzone.classList.toggle("scanning", loading && mode === "single");
}
function showError(msg) {
  els.errorBox.textContent = msg;
  els.errorBox.hidden = false;
}
function hideError() {
  els.errorBox.hidden = true;
  els.errorBox.textContent = "";
}
let toastTimer;
function showToast(msg) {
  els.toast.textContent = msg;
  els.toast.hidden = false;
  requestAnimationFrame(() => els.toast.classList.add("show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    els.toast.classList.remove("show");
    setTimeout(() => (els.toast.hidden = true), 250);
  }, 2200);
}

/* ---------- Lightbox ---------- */
function openLightbox(src) {
  els.lightboxImg.src = src;
  els.lightbox.hidden = false;
}
function closeLightbox() {
  els.lightbox.hidden = true;
  els.lightboxImg.removeAttribute("src");
}

/* ---------- Events ---------- */
document.querySelectorAll(".mode-btn").forEach((b) =>
  b.addEventListener("click", () => setMode(b.dataset.mode))
);

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    els.fileInput.click();
  }
});
els.fileInput.addEventListener("change", (e) => handleFiles(e.target.files));
["dragenter", "dragover"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("dragover");
  })
);
els.dropzone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));

els.analyzeBtn.addEventListener("click", analyze);
els.clearBtn.addEventListener("click", clearAll);
els.opacity.addEventListener("input", applyOpacity);
els.copyBtn.addEventListener("click", copyResults);
els.downloadBtn.addEventListener("click", downloadReport);
els.pdfBtn.addEventListener("click", downloadPdf);
els.fhirBtn?.addEventListener("click", downloadFhir);
els.exportAllBtn?.addEventListener("click", exportAllReports);
els.exportCsvBtn.addEventListener("click", () => exportCsv(false));
els.exportBatchJsonBtn?.addEventListener("click", () => exportBatchJson(false));
els.exportFlaggedBtn?.addEventListener("click", () => {
  exportCsv(true);
  exportBatchJson(true);
});
els.demoNormalBtn?.addEventListener("click", () => loadDemoSample("NORMAL"));
els.demoPneuBtn?.addEventListener("click", () => loadDemoSample("PNEUMONIA"));
els.fbYes.addEventListener("click", () => sendFeedback(true));
els.fbNo.addEventListener("click", () => sendFeedback(false));
els.refreshHistory.addEventListener("click", loadHistory);
els.refreshReview.addEventListener("click", loadReviewQueue);
els.createCaseBtn?.addEventListener("click", createCase);

document.querySelectorAll("figure[data-zoom]").forEach((fig) =>
  fig.addEventListener("click", () => {
    const img = fig.querySelector("img");
    if (img && img.src) openLightbox(img.src);
  })
);
els.lightboxClose.addEventListener("click", closeLightbox);
els.lightbox.addEventListener("click", (e) => {
  if (e.target === els.lightbox) closeLightbox();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !els.lightbox.hidden) closeLightbox();
});

els.themeToggle.addEventListener("click", toggleTheme);

initTheme();
initReveal();
initUserMenu();
initProductivitySettings();
initDatasetTabs();
checkStatus();
loadMetrics();
loadModelCount();
loadHistory();
loadReviewQueue();
setInterval(checkStatus, 15000);
