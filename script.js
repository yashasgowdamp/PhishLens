"use strict";

const pageTitles = {
  dashboard: ["Dashboard", "DASHBOARD"],
  email: ["Email Scanner", "EMAIL"],
  url: ["URL Scanner", "URL"],
  history: ["Detection History", "HISTORY"],
  reports: ["Analysis Reports", "REPORTS"]
};

const API_BASE_URL = window.PHISHLENS_API_BASE_URL || "http://127.0.0.1:5000";
const state = {
  page: "dashboard",
  records: [],
  query: "",
  mobileMenuOpen: false,
  historyOffset: 0,
  historyLimit: 50,
  historyTotal: 0,
  historyRequest: 0,
  historyTimer: null,
  dashboardRecords: [],
  dashboardRequest: 0,
  dashboardHistoryAvailable: null,
  reportRequest: 0,
  reportTotal: 0
};

const pageTitle = document.querySelector("#page-title");
const breadcrumb = document.querySelector("#breadcrumb");
const sidebar = document.querySelector("#sidebar");
const menuToggle = document.querySelector("#menu-toggle");
const historySearch = document.querySelector("#history-search");
const typeFilter = document.querySelector("#type-filter");
const riskFilter = document.querySelector("#risk-filter");
const historyMessage = document.querySelector("#history-message");
const historyPrevious = document.querySelector("#history-previous");
const historyNext = document.querySelector("#history-next");
const recordDialog = document.querySelector("#record-dialog");
const dialogContent = document.querySelector("#dialog-content");

document.querySelectorAll("[data-page]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    navigate(link.dataset.page);
  });
});

window.addEventListener("hashchange", () => {
  const requestedPage = window.location.hash.slice(1);
  showPage(pageTitles[requestedPage] ? requestedPage : "dashboard");
});

menuToggle.addEventListener("click", () => {
  state.mobileMenuOpen = !state.mobileMenuOpen;
  sidebar.classList.toggle("open", state.mobileMenuOpen);
  menuToggle.setAttribute("aria-expanded", String(state.mobileMenuOpen));
  menuToggle.setAttribute("aria-label", state.mobileMenuOpen ? "Close navigation" : "Open navigation");
});

historySearch.addEventListener("input", () => {
  state.query = historySearch.value.trim();
  state.historyOffset = 0;
  queueHistoryLoad();
});

typeFilter.addEventListener("change", () => {
  state.historyOffset = 0;
  loadHistory();
});
riskFilter.addEventListener("change", () => {
  state.historyOffset = 0;
  loadHistory();
});
historyPrevious.addEventListener("click", () => {
  state.historyOffset = Math.max(0, state.historyOffset - state.historyLimit);
  loadHistory();
});
historyNext.addEventListener("click", () => {
  if (state.historyOffset + state.historyLimit < state.historyTotal) {
    state.historyOffset += state.historyLimit;
    loadHistory();
  }
});
document.querySelector("#email-form").addEventListener("submit", analyzeEmail);
document.querySelector("#url-form").addEventListener("submit", analyzeUrl);
document.querySelector("#email-form").addEventListener("reset", () => {
  window.setTimeout(() => resetAssessment("email"), 0);
});
document.querySelector("#url-form").addEventListener("reset", () => {
  window.setTimeout(() => resetAssessment("url"), 0);
});
document.querySelector("#export-report").addEventListener("click", exportReport);
document.querySelector(".dialog-close").addEventListener("click", () => recordDialog.close());
recordDialog.addEventListener("click", (event) => {
  if (event.target === recordDialog) recordDialog.close();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.mobileMenuOpen) closeMobileMenu();
});

document.querySelector("#dashboard-records").addEventListener("click", handleRecordAction);
document.querySelector("#history-records").addEventListener("click", handleRecordAction);

function navigate(page) {
  if (!pageTitles[page]) return;
  if (window.location.hash === `#${page}`) {
    showPage(page);
  } else {
    window.location.hash = page;
  }
}

function showPage(page) {
  state.page = page;
  document.querySelectorAll("[data-view]").forEach((view) => {
    view.hidden = view.dataset.view !== page;
  });
  document.querySelectorAll(".nav-link").forEach((link) => {
    const active = link.dataset.page === page;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  pageTitle.textContent = pageTitles[page][0];
  breadcrumb.textContent = pageTitles[page][1];
  if (state.mobileMenuOpen) closeMobileMenu();
  if (page === "history") loadHistory();
  if (page === "dashboard") loadDashboard();
  if (page === "reports") loadReports();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function closeMobileMenu() {
  state.mobileMenuOpen = false;
  sidebar.classList.remove("open");
  menuToggle.setAttribute("aria-expanded", "false");
  menuToggle.setAttribute("aria-label", "Open navigation");
}

async function loadDashboard() {
  const requestId = ++state.dashboardRequest;
  const message = document.querySelector("#dashboard-message");
  message.textContent = "Loading shared dashboard statistics...";
  message.hidden = false;
  message.classList.add("form-message-info");

  const [statisticsResult, historyResult] = await Promise.all([
    getJson("/api/dashboard/stats").then((value) => ({ value }), (error) => ({ error })),
    getJson("/api/history?limit=4&offset=0&type=all&risk=all&search=")
      .then((value) => ({ value }), (error) => ({ error }))
  ]);
  if (requestId !== state.dashboardRequest) return;

  const errors = [];
  if (statisticsResult.error) {
    resetDashboardStatistics();
    errors.push(`Dashboard statistics unavailable: ${statisticsResult.error.message}`);
  } else {
    try {
      renderDashboardStatistics(statisticsResult.value.statistics);
    } catch (error) {
      resetDashboardStatistics();
      errors.push(error.message);
    }
  }

  if (historyResult.error) {
    state.dashboardRecords = [];
    state.dashboardHistoryAvailable = false;
    errors.push(`Recent shared history unavailable: ${historyResult.error.message}`);
  } else {
    try {
      if (!Array.isArray(historyResult.value.records)) {
        throw new Error("The Flask API returned an incomplete recent-history response.");
      }
      state.dashboardRecords = historyResult.value.records.map(mapHistoryRecord);
      state.dashboardHistoryAvailable = true;
    } catch (error) {
      state.dashboardRecords = [];
      state.dashboardHistoryAvailable = false;
      errors.push(error.message);
    }
  }
  renderDashboardRecords();

  message.textContent = errors.join(" ");
  message.hidden = errors.length === 0;
  message.classList.toggle("form-message-info", errors.length === 0);
}

function renderDashboardStatistics(statistics) {
  if (
    !statistics
    || !Number.isInteger(statistics.email_analyses)
    || !Number.isInteger(statistics.url_analyses)
    || !Number.isInteger(statistics.spam_emails)
    || !Number.isInteger(statistics.suspicious_urls)
    || !Number.isInteger(statistics.total_analyses)
    || !Number.isInteger(statistics.flagged_analyses)
    || typeof statistics.flagged_percentage !== "number"
    || !Array.isArray(statistics.activity_last_7_days)
    || statistics.activity_last_7_days.length !== 7
  ) {
    throw new Error("The Flask API returned incomplete dashboard statistics.");
  }

  document.querySelector("#dashboard-email-count").textContent =
    statistics.email_analyses.toLocaleString();
  document.querySelector("#dashboard-url-count").textContent =
    statistics.url_analyses.toLocaleString();
  document.querySelector("#dashboard-spam-count").textContent =
    statistics.spam_emails.toLocaleString();
  document.querySelector("#dashboard-suspicious-url-count").textContent =
    statistics.suspicious_urls.toLocaleString();
  document.querySelector("#dashboard-total-count").textContent =
    statistics.total_analyses.toLocaleString();
  document.querySelector("#dashboard-flagged-count").textContent =
    statistics.flagged_analyses.toLocaleString();
  document.querySelector("#dashboard-flagged-percent").textContent =
    `${statistics.flagged_percentage.toFixed(1)}%`;
  document.querySelector("#dashboard-flagged-donut").setAttribute(
    "aria-label",
    `${statistics.flagged_percentage.toFixed(1)} percent of shared analyses are medium or high risk`
  );
  document.querySelector("#dashboard-flagged-donut").style.setProperty(
    "--flagged-angle",
    `${Math.min(100, Math.max(0, statistics.flagged_percentage)) * 3.6}deg`
  );
  renderActivityChart(
    statistics.activity_last_7_days,
    document.querySelector("#dashboard-activity-chart")
  );
}

function resetDashboardStatistics() {
  [
    "#dashboard-email-count",
    "#dashboard-url-count",
    "#dashboard-spam-count",
    "#dashboard-suspicious-url-count",
    "#dashboard-total-count",
    "#dashboard-flagged-count",
    "#dashboard-flagged-percent"
  ].forEach((selector) => {
    document.querySelector(selector).textContent = "—";
  });
  const donut = document.querySelector("#dashboard-flagged-donut");
  donut.style.removeProperty("--flagged-angle");
  donut.setAttribute("aria-label", "Shared risk summary unavailable");
  clearActivityChart(document.querySelector("#dashboard-activity-chart"));
}

function renderActivityChart(activity, chart) {
  if (!activity.every((day) =>
    typeof day.date === "string"
    && /^\d{4}-\d{2}-\d{2}$/.test(day.date)
    && !Number.isNaN(Date.parse(`${day.date}T00:00:00Z`))
    && Number.isInteger(day.analyses)
    && Number.isInteger(day.flagged_analyses)
  )) {
    throw new Error("The Flask API returned invalid seven-day activity data.");
  }

  const totalValues = activity.map((day) => day.analyses);
  const flaggedValues = activity.map((day) => day.flagged_analyses);
  const maximum = Math.max(1, ...totalValues);
  const x = activity.map((_, index) => 44 + (698 * index) / Math.max(1, activity.length - 1));
  const y = (value) => 208 - (Math.min(value, maximum) / maximum) * 180;
  const totalPath = totalValues.map((value, index) =>
    `${index === 0 ? "M" : "L"}${x[index]} ${y(value)}`
  ).join(" ");
  const flaggedPath = flaggedValues.map((value, index) =>
    `${index === 0 ? "M" : "L"}${x[index]} ${y(value)}`
  ).join(" ");

  chart.querySelector(".activity-total-path").setAttribute("d", totalPath);
  chart.querySelector(".activity-fill-path").setAttribute("d", `${totalPath} L742 208 L44 208 Z`);
  chart.querySelector(".activity-flagged-path").setAttribute("d", flaggedPath);
  chart.querySelector(".activity-points").replaceChildren(
    ...totalValues.map((_, index) => {
      const point = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      point.setAttribute("cx", String(x[index]));
      point.setAttribute("cy", String(y(totalValues[index])));
      point.setAttribute("r", "3.5");
      return point;
    })
  );
  chart.querySelector(".activity-days").replaceChildren(
    ...activity.map((day, index) => {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", String(x[index] - 10));
      label.setAttribute("y", "234");
      label.textContent = new Intl.DateTimeFormat(undefined, {
        weekday: "short",
        timeZone: "UTC"
      }).format(new Date(`${day.date}T00:00:00Z`));
      return label;
    })
  );
  ["activity-y0", "activity-y1", "activity-y2", "activity-y3"].forEach((selector, index) => {
    chart.querySelector(`.${selector}`).textContent =
      Math.round((maximum * index) / 3).toLocaleString();
  });
  chart.setAttribute(
    "aria-label",
    `Shared analysis counts for the last seven UTC days. Maximum daily count: ${maximum}.`
  );
}

function clearActivityChart(chart) {
  [".activity-total-path", ".activity-fill-path", ".activity-flagged-path"].forEach((selector) => {
    chart.querySelector(selector).setAttribute("d", "");
  });
  chart.querySelector(".activity-points").replaceChildren();
  chart.querySelector(".activity-days").replaceChildren();
  [".activity-y0", ".activity-y1", ".activity-y2", ".activity-y3"].forEach((selector) => {
    chart.querySelector(selector).textContent = "—";
  });
}

function renderDashboardRecords() {
  const table = document.querySelector("#dashboard-records").closest("table");
  const empty = document.querySelector("#dashboard-recent-empty");
  renderRows(document.querySelector("#dashboard-records"), state.dashboardRecords);
  table.hidden = state.dashboardRecords.length === 0;
  empty.textContent = state.dashboardHistoryAvailable === false
    ? "Recent shared history is unavailable."
    : state.dashboardHistoryAvailable === null
      ? "Recent shared history has not been loaded."
      : "No shared analyses have been recorded yet.";
  empty.hidden = state.dashboardRecords.length !== 0;
}

async function loadReports() {
  const requestId = ++state.reportRequest;
  const message = document.querySelector("#report-message");
  const exportButton = document.querySelector("#export-report");
  message.textContent = "Loading shared report statistics...";
  message.hidden = false;
  message.classList.add("form-message-info");
  exportButton.disabled = true;

  try {
    const response = await getJson("/api/reports");
    if (requestId !== state.reportRequest) return;
    renderReportStatistics(response.reports);
    state.reportTotal = response.reports.total_analyses;
    message.textContent = state.reportTotal
      ? ""
      : "No shared analysis records are available to report or export yet.";
    message.hidden = state.reportTotal > 0;
    exportButton.disabled = state.reportTotal === 0;
  } catch (error) {
    if (requestId !== state.reportRequest) return;
    state.reportTotal = 0;
    clearReportStatistics();
    message.textContent = `Shared reports are unavailable: ${error.message}`;
    message.hidden = false;
    message.classList.remove("form-message-info");
    exportButton.disabled = true;
  }
}

function renderReportStatistics(reports) {
  const riskLevels = ["low", "medium", "high"];
  if (
    !reports
    || !Number.isInteger(reports.total_analyses)
    || !Number.isInteger(reports.email_analyses)
    || !Number.isInteger(reports.url_analyses)
    || !Number.isInteger(reports.spam_emails)
    || !Number.isInteger(reports.legitimate_emails)
    || !reports.risk_distribution
    || !riskLevels.every((level) =>
      Number.isInteger(reports.risk_distribution[level]?.count)
      && typeof reports.risk_distribution[level]?.percentage === "number"
    )
    || !Array.isArray(reports.activity_last_7_days)
    || reports.activity_last_7_days.length !== 7
  ) {
    throw new Error("The Flask API returned incomplete report statistics.");
  }

  document.querySelector("#report-total").textContent =
    reports.total_analyses.toLocaleString();
  document.querySelector("#report-emails").textContent =
    reports.email_analyses.toLocaleString();
  document.querySelector("#report-urls").textContent =
    reports.url_analyses.toLocaleString();
  document.querySelector("#report-email-classifications").textContent =
    `${reports.spam_emails.toLocaleString()} / ${reports.legitimate_emails.toLocaleString()}`;

  riskLevels.forEach((level) => {
    const distribution = reports.risk_distribution[level];
    document.querySelector(`#report-${level}-summary`).textContent =
      `${distribution.count.toLocaleString()} · ${distribution.percentage.toFixed(1)}%`;
    document.querySelector(`#report-${level}-bar`).style.width =
      `${Math.min(100, Math.max(0, distribution.percentage))}%`;
  });
  if (typeof reports.risk_score_notice === "string") {
    document.querySelector("#report-risk-notice").textContent = reports.risk_score_notice;
  }
  renderActivityChart(
    reports.activity_last_7_days,
    document.querySelector("#report-activity-chart")
  );
}

function clearReportStatistics() {
  [
    "#report-total",
    "#report-emails",
    "#report-urls",
    "#report-email-classifications",
    "#report-low-summary",
    "#report-medium-summary",
    "#report-high-summary"
  ].forEach((selector) => {
    document.querySelector(selector).textContent = "—";
  });
  ["#report-low-bar", "#report-medium-bar", "#report-high-bar"].forEach((selector) => {
    document.querySelector(selector).style.width = "0%";
  });
  clearActivityChart(document.querySelector("#report-activity-chart"));
}

async function exportReport() {
  const button = document.querySelector("#export-report");
  if (button.disabled) return;
  button.disabled = true;
  button.lastChild.textContent = "Preparing export...";
  setReportMessage("Fetching all shared analysis records...", true);

  try {
    const records = await fetchSharedRecordsForExport((count, total) => {
      setReportMessage(`Preparing export: ${count.toLocaleString()} of ${total.toLocaleString()} records loaded...`, true);
    });
    if (records.length === 0) {
      setReportMessage("There are no shared records to export.", false);
      return;
    }

    const rows = [
      ["Date and time (UTC)", "Type", "Email subject or URL", "Classification", "Risk level", "Risk score", "Warning reasons", "Indicators"],
      ...records.map((record) => [
        record.created_at,
        record.type === "email" ? "Email" : "URL",
        record.target,
        record.classification,
        getRiskLabel(record.risk_level),
        `${record.risk_score}/100`,
        record.warning_reasons.join("; "),
        record.indicators.join("; ")
      ])
    ];
    const csv = rows.map((row) => row.map(csvCell).join(",")).join("\r\n");
    const file = new Blob([`\uFEFF${csv}`], { type: "text/csv;charset=utf-8" });
    const downloadUrl = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = `phishlens-shared-report-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
    setReportMessage(`Exported ${records.length.toLocaleString()} shared analysis records.`, true);
  } catch (error) {
    setReportMessage(`Report export failed: ${error.message}`, false);
  } finally {
    button.lastChild.textContent = "Export Report";
    button.disabled = state.reportTotal === 0;
  }
}

async function fetchSharedRecordsForExport(onProgress) {
  const limit = 100;
  const maximumOffset = 10_000;
  const maximumRecords = maximumOffset + limit;
  let total = null;
  const records = [];

  for (let offset = 0; ; offset += limit) {
    if (offset > maximumOffset) {
      throw new Error("The API history export limit was reached; no partial file was downloaded.");
    }
    const response = await getJson(
      `/api/history?limit=${limit}&offset=${offset}&type=all&risk=all&search=`
    );
    if (
      !Number.isInteger(response.total)
      || !Array.isArray(response.records)
      || response.records.length > limit
    ) {
      throw new Error("The Flask API returned an incomplete shared-history page.");
    }
    if (total === null) {
      total = response.total;
      if (total > maximumRecords) {
        throw new Error("More than 10,100 records exceed the API export limit; no partial file was downloaded.");
      }
    } else if (response.total !== total) {
      throw new Error("Shared history changed during export; retry to avoid an incomplete report.");
    }

    const pageRecords = response.records.map(mapHistoryRecord);
    if (!pageRecords.length && records.length < total) {
      throw new Error("The API returned an incomplete history page; no partial file was downloaded.");
    }
    records.push(...pageRecords);
    onProgress(records.length, total);
    if (records.length >= total) break;
    if (pageRecords.length !== limit) {
      throw new Error("Shared history changed during export; retry to avoid an incomplete report.");
    }
  }

  const ids = new Set(records.map((record) => record.id));
  if (ids.size !== records.length) {
    throw new Error("Shared history changed during export; retry to avoid duplicate rows.");
  }
  return records;
}

function setReportMessage(message, isInfo) {
  const output = document.querySelector("#report-message");
  output.textContent = message;
  output.hidden = !message;
  output.classList.toggle("form-message-info", Boolean(isInfo));
}

function renderHistory(showEmptyMessage = true) {
  const count = document.querySelector("#history-count");
  const noResults = document.querySelector("#no-results");
  const table = document.querySelector("#history-records").closest("table");
  const pageNumber = Math.floor(state.historyOffset / state.historyLimit) + 1;
  const firstRecord = state.historyTotal ? state.historyOffset + 1 : 0;
  const lastRecord = state.historyOffset + state.records.length;

  renderRows(document.querySelector("#history-records"), state.records);
  count.textContent = `Showing ${firstRecord}–${lastRecord} of ${state.historyTotal} shared records`;
  noResults.textContent = state.historyTotal
    ? "No shared analysis records match these filters."
    : "No shared analysis records are available yet.";
  noResults.hidden = state.records.length !== 0 || !showEmptyMessage;
  table.hidden = state.records.length === 0;
  historyPrevious.disabled = state.historyOffset === 0;
  historyNext.disabled =
    state.historyOffset + state.records.length >= state.historyTotal
    || state.historyOffset >= 10_000;
  document.querySelector("#history-page-label").textContent = `Page ${pageNumber}`;
}

function queueHistoryLoad() {
  window.clearTimeout(state.historyTimer);
  state.historyTimer = window.setTimeout(loadHistory, 250);
}

async function loadHistory() {
  window.clearTimeout(state.historyTimer);
  const requestId = ++state.historyRequest;
  const params = new URLSearchParams({
    type: typeFilter.value,
    risk: riskFilter.value,
    search: state.query,
    limit: String(state.historyLimit),
    offset: String(state.historyOffset)
  });
  historyMessage.textContent = "Loading shared analysis history...";
  historyMessage.hidden = false;
  historyMessage.classList.add("form-message-info");

  try {
    const response = await getJson(`/api/history?${params.toString()}`);
    if (requestId !== state.historyRequest) return;
    if (!Array.isArray(response.records) || !Number.isInteger(response.total)) {
      throw new Error("The Flask API returned an incomplete history response.");
    }
    state.records = response.records.map(mapHistoryRecord);
    state.historyTotal = response.total;
    historyMessage.hidden = true;
    renderHistory();
  } catch (error) {
    if (requestId !== state.historyRequest) return;
    state.records = [];
    state.historyTotal = 0;
    historyMessage.textContent = error.message;
    historyMessage.hidden = false;
    historyMessage.classList.remove("form-message-info");
    renderHistory(false);
  }
}

function renderRows(tbody, records) {
  tbody.replaceChildren();
  records.forEach((record) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="date-cell">${escapeHtml(record.date)}</td>
      <td class="type-cell">${record.type === "email" ? "✉　Email" : "↗　URL"}</td>
      <td class="target-cell" title="${escapeHtml(record.target)}">${escapeHtml(record.target)}</td>
      <td class="classification">${escapeHtml(record.classification)}</td>
      <td><span class="risk-badge risk-${escapeHtml(record.risk_level)}">${escapeHtml(getRiskLabel(record.risk_level))}</span></td>
      <td class="score-cell">${escapeHtml(record.risk_score)}/100</td>
      <td><button class="detail-button" type="button" data-record-id="${escapeHtml(record.id)}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2.5 12s3.2-6 9.5-6 9.5 6 9.5 6-3.2 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/></svg>View details</button></td>`;
    tbody.appendChild(row);
  });
}

function handleRecordAction(event) {
  const button = event.target.closest("[data-record-id]");
  if (!button) return;
  const record = [...state.records, ...state.dashboardRecords]
    .find((item) => item.id === button.dataset.recordId);
  if (record) showRecordDetails(record);
}

function showRecordDetails(record) {
  const details = [
    ["Date and time", record.date],
    ["Type", record.type === "email" ? "Email" : "URL"],
    [record.type === "email" ? "Email subject" : "URL", record.target],
    ["Classification", record.classification],
    ["Risk level", getRiskLabel(record.risk_level)],
    ["Risk score", `${record.risk_score} / 100`],
    ["Detection reasons", record.warning_reasons.join(", ") || "No warning reasons reported."],
    ["Suspicious indicators", record.indicators.join(", ") || "None identified."]
  ];

  dialogContent.replaceChildren();
  const body = document.createElement("dl");
  body.className = "dialog-body";
  details.forEach(([label, value]) => {
    const item = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    if (label === "Risk score") description.className = "dialog-score";
    item.append(term, description);
    body.appendChild(item);
  });
  dialogContent.appendChild(body);
  if (typeof recordDialog.showModal === "function") recordDialog.showModal();
  else recordDialog.setAttribute("open", "");
}

async function analyzeEmail(event) {
  event.preventDefault();
  const form = event.currentTarget;
  if (!form.reportValidity()) return;

  const submitButton = form.querySelector('[type="submit"]');
  setSubmitting(form, submitButton, true, "email");
  try {
    const response = await postJson("/api/email/analyze", {
      sender: document.querySelector("#email-from").value.trim(),
      subject: document.querySelector("#email-subject").value.trim(),
      body: document.querySelector("#email-body").value.trim(),
      url: document.querySelector("#email-url").value.trim()
    });
    const record = mapAnalysisResponse(response, "email");
    updateAssessment("email", record);
    setFormMessage("email", persistenceMessage(response), true);
    if (response.persistence && response.persistence.saved) {
      state.historyOffset = 0;
      loadHistory();
    }
  } catch (error) {
    setFormMessage("email", error.message, false);
  } finally {
    setSubmitting(form, submitButton, false, "email");
  }
}

async function analyzeUrl(event) {
  event.preventDefault();
  const form = event.currentTarget;
  if (!form.reportValidity()) return;
  const submitButton = form.querySelector('[type="submit"]');
  setSubmitting(form, submitButton, true, "url");
  try {
    const response = await postJson("/api/url/analyze", {
      url: document.querySelector("#url-input").value.trim()
    });
    const record = mapAnalysisResponse(response, "url");
    updateAssessment("url", record);
    setFormMessage("url", persistenceMessage(response), true);
    if (response.persistence && response.persistence.saved) {
      state.historyOffset = 0;
      loadHistory();
    }
  } catch (error) {
    setFormMessage("url", error.message, false);
  } finally {
    setSubmitting(form, submitButton, false, "url");
  }
}

function updateAssessment(type, record) {
  const card = document.querySelector(`#${type}-assessment`);
  const riskLevel = record.risk_level;
  card.querySelector(".risk-score").textContent = String(record.risk_score);
  card.querySelector(".risk-gauge").dataset.level = riskLevel;
  const status = card.querySelector(".risk-status");
  status.textContent = getRiskLabel(riskLevel).toUpperCase();
  status.className = `tag tag-muted risk-status risk-status-${riskLevel}`;
  card.querySelector(".risk-summary").textContent = record.warning_reasons.join(". ") + ".";
  card.querySelector(".reasons").textContent = record.warning_reasons.join(" · ");
  card.querySelector(".patterns").textContent =
    "Pattern details are not separately reported by the analysis API.";
  card.querySelector(".indicators").textContent = record.indicators.length
    ? record.indicators.join(" · ")
    : "No suspicious indicators reported.";
}

function resetAssessment(type) {
  const card = document.querySelector(`#${type}-assessment`);
  card.querySelector(".risk-score").textContent = "—";
  card.querySelector(".risk-gauge").removeAttribute("data-level");
  const status = card.querySelector(".risk-status");
  status.textContent = "NOT ANALYSED";
  status.className = "tag tag-muted risk-status";
  card.querySelector(".risk-summary").textContent =
    "Submit an email or URL to generate an indicative risk assessment.";
  card.querySelector(".reasons").textContent = "No detection reasons available yet.";
  card.querySelector(".patterns").textContent = "Pattern details are not separately reported by the analysis API.";
  card.querySelector(".indicators").textContent = "No suspicious indicators reported.";
  setFormMessage(type, "");
}

async function postJson(path, payload) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  } catch {
    throw new Error("Cannot reach the Flask API. Start the backend and check its configured address.");
  }

  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error("The Flask API returned an unreadable response.");
  }
  if (!response.ok || result.status !== "success") {
    throw new Error(result.message || `The Flask API request failed (${response.status}).`);
  }
  return result;
}

function mapAnalysisResponse(response, type) {
  const allowedRiskLevels = ["low", "medium", "high"];
  if (
    response.analysis_type !== type
    || typeof response.target !== "string"
    || typeof response.classification !== "string"
    || !allowedRiskLevels.includes(response.risk_level)
    || !Number.isInteger(response.risk_score)
    || response.risk_score < 0
    || response.risk_score > 100
    || !Array.isArray(response.warning_reasons)
    || !Array.isArray(response.indicators)
  ) {
    throw new Error("The Flask API returned an incomplete analysis result.");
  }
  return {
    id: response.id || createId(),
    date: formatDate(response.created_at || new Date().toISOString()),
    type: response.analysis_type,
    target: response.target,
    classification: response.classification,
    risk_level: response.risk_level,
    risk_score: response.risk_score,
    warning_reasons: response.warning_reasons,
    indicators: response.indicators
  };
}

function mapHistoryRecord(record) {
  const allowedTypes = ["email", "url"];
  const allowedRiskLevels = ["low", "medium", "high"];
  if (
    typeof record.id !== "string"
    || !allowedTypes.includes(record.analysis_type)
    || typeof record.target !== "string"
    || typeof record.classification !== "string"
    || !allowedRiskLevels.includes(record.risk_level)
    || !Number.isInteger(record.risk_score)
    || record.risk_score < 0
    || record.risk_score > 100
    || typeof record.created_at !== "string"
    || !Array.isArray(record.warning_reasons)
    || !Array.isArray(record.indicators)
  ) {
    throw new Error("The Flask API returned a malformed history record.");
  }
  return {
    id: record.id,
    date: formatDate(record.created_at),
    created_at: record.created_at,
    type: record.analysis_type,
    target: record.target,
    classification: record.classification,
    risk_level: record.risk_level,
    risk_score: record.risk_score,
    warning_reasons: record.warning_reasons,
    indicators: record.indicators
  };
}

async function getJson(path) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`);
  } catch {
    throw new Error("Cannot reach the Flask API. Start the backend and check its configured address.");
  }

  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error("The Flask API returned an unreadable response.");
  }
  if (!response.ok || result.status !== "success") {
    throw new Error(result.message || `The Flask API request failed (${response.status}).`);
  }
  return result;
}

function persistenceMessage(response) {
  const persistence = response.persistence;
  if (persistence && persistence.saved) {
    return persistence.message || "Analysis saved to shared history.";
  }
  return persistence && persistence.message
    ? `Analysis completed but was not saved: ${persistence.message}`
    : "Analysis completed, but the API did not confirm that it was saved.";
}

function setSubmitting(form, button, isSubmitting, type) {
  form.setAttribute("aria-busy", String(isSubmitting));
  button.disabled = isSubmitting;
  const label = isSubmitting ? "Analysing..." : type === "email" ? "Analyse Email" : "Analyse URL";
  button.lastChild.textContent = label;
  if (isSubmitting) setFormMessage(type, "Sending analysis to the Flask API...", true);
}

function setFormMessage(type, message, isInfo) {
  const output = document.querySelector(`#${type}-message`);
  output.textContent = message;
  output.hidden = !message;
  output.classList.toggle("form-message-info", Boolean(isInfo));
}

function csvCell(value) {
  let text = String(value);
  if (/^[\t\r\n ]*[=+\-@]/.test(text)) text = `'${text}`;
  return `"${text.replace(/"/g, '""')}"`;
}

function getRiskLabel(level) {
  return `${level[0].toUpperCase()}${level.slice(1)} Risk`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function createId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `analysis-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[character]);
}

const initialPage = window.location.hash.slice(1);
showPage(pageTitles[initialPage] ? initialPage : "dashboard");
