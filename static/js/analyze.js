/* ==========================================================================
   Sentiment Lab — unified analyze page
   Vanilla JS, no framework, no charting library. Charts are hand-drawn SVG
   and CSS bars so the project keeps zero front-end dependencies.
   ========================================================================== */
(function () {
  "use strict";

  var CONF = window.SENTIMENT_LAB || {};
  var PAGE_SIZE = 25;

  var state = {
    source: "text",
    upload: null,      // { upload_id, filename, rows, columns }
    rows: [],
    columns: [],
    analysisColumns: [],
    textColumn: "comment",
    summary: null,
    fileId: null,
    meta: null,
    sort: { key: null, dir: 1 },
    page: 1,
    poller: null
  };

  /* ------------------------------------------------------------- helpers */
  function $(id) { return document.getElementById(id); }
  function all(sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); }

  function show(el, visible) {
    if (el) { el.hidden = !visible; }
  }

  function esc(value) {
    if (value === null || value === undefined) { return ""; }
    return String(value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function num(value, digits) {
    if (value === null || value === undefined || value === "" || isNaN(value)) {
      return "—";
    }
    return Number(value).toFixed(digits === undefined ? 2 : digits);
  }

  function signed(value, digits) {
    if (value === null || value === undefined || value === "" || isNaN(value)) {
      return "—";
    }
    var n = Number(value);
    return (n > 0 ? "+" : "") + n.toFixed(digits === undefined ? 2 : digits);
  }

  function int(value) {
    if (value === null || value === undefined || isNaN(value)) { return "—"; }
    return Number(value).toLocaleString();
  }

  function sentClass(label) {
    if (label === "POSITIVE") { return "is-pos"; }
    if (label === "NEGATIVE") { return "is-neg"; }
    if (label === "NEUTRAL") { return "is-neu"; }
    return "tag--muted";
  }

  function dateOnly(value) {
    if (!value) { return "—"; }
    var d = new Date(value);
    if (isNaN(d.getTime())) { return String(value).slice(0, 10); }
    return d.toISOString().slice(0, 10);
  }

  /* -------------------------------------------------------- feedback UI */
  function setError(message) {
    var box = $("alert");
    $("alert-text").textContent = message;
    show(box, Boolean(message));
    if (message) { box.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
  }

  function clearError() { setError(""); }

  function setBusy(button, busy, idleLabel) {
    if (!button) { return; }
    button.disabled = busy;
    if (busy) {
      button.dataset.idle = button.dataset.idle || button.textContent.trim();
      button.innerHTML = '<span class="spinner" aria-hidden="true"></span> Working…';
    } else {
      button.textContent = idleLabel || button.dataset.idle || "Analyze";
    }
  }

  function setProgress(stage, done, total, percent) {
    show($("progress"), true);
    $("progress-stage").textContent = stage || "Working…";
    $("progress-count").textContent =
      total ? int(done) + " / " + int(total) + " processed" : (done ? int(done) + " fetched" : "");
    var bar = $("progress-bar");
    if (percent === null || percent === undefined) {
      bar.classList.add("progress__bar--indeterminate");
      bar.style.width = "";
    } else {
      bar.classList.remove("progress__bar--indeterminate");
      bar.style.width = Math.max(2, percent) + "%";
    }
  }

  function hideProgress() { show($("progress"), false); }

  /* --------------------------------------------------------------- fetch */
  function postJSON(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(body || {})
    }).then(readJSON);
  }

  function readJSON(response) {
    return response.json().catch(function () {
      throw new Error("The server returned an unreadable response.");
    }).then(function (data) {
      if (!response.ok || data.success === false) {
        throw new Error(data.error || "Request failed (" + response.status + ").");
      }
      return data;
    });
  }

  function pollJob(jobId, onDone, onFail) {
    if (state.poller) { clearInterval(state.poller); }
    state.poller = setInterval(function () {
      fetch("/api/job/" + jobId, { headers: { "Accept": "application/json" } })
        .then(function (r) { return r.json(); })
        .then(function (job) {
          if (job.status === "error") {
            clearInterval(state.poller);
            onFail(new Error(job.error || "Analysis failed."));
            return;
          }
          setProgress(job.stage, job.done, job.total, job.percent);
          if (job.status === "done") {
            clearInterval(state.poller);
            onDone(job.result);
          }
        })
        .catch(function () {
          clearInterval(state.poller);
          onFail(new Error("Lost contact with the server while analyzing."));
        });
    }, 700);
  }

  /* ------------------------------------------------------------- sources */
  var HINTS = {
    text: "Single piece of text, analyzed instantly.",
    dataset: "CSV upload — your original columns are preserved.",
    youtube: "Official YouTube Data API v3, no HTML reading.",
    blog: "WordPress REST, structured data, or standard comment markup."
  };

  function selectSource(source) {
    state.source = source;
    all(".source-btn").forEach(function (btn) {
      var on = btn.dataset.source === source;
      btn.setAttribute("aria-selected", on ? "true" : "false");
      btn.tabIndex = on ? 0 : -1;
    });
    ["text", "dataset", "youtube", "blog"].forEach(function (name) {
      show($("panel-" + name), name === source);
    });
    $("input-hint").textContent = HINTS[source] || "";
    clearError();
    hideProgress();
    resetResults();
    history.replaceState(null, "", "/analyze?source=" + source);
  }

  function resetResults() {
    show($("results"), false);
    show($("results-empty"), true);
    ["single-card", "summary-card", "charts-card", "table-card"].forEach(function (id) {
      show($(id), false);
    });
    state.rows = [];
    state.summary = null;
    state.page = 1;
    state.sort = { key: null, dir: 1 };
  }

  /* ------------------------------------------------------ single result */
  function renderSingle(result) {
    show($("results"), true);
    show($("results-empty"), false);
    show($("single-card"), true);

    var badge = $("single-label");
    badge.textContent = result.sentiment;
    badge.className = "verdict__badge " + sentClass(result.sentiment);

    var isVader = result.compound !== null && result.compound !== undefined;
    $("single-score-label").textContent = isVader ? "VADER compound" : "Model confidence";
    $("single-score-value").textContent = isVader
      ? signed(result.compound, 2)
      : num(result.confidence, 2);

    $("single-engine").textContent = "Engine: " + result.model;

    $("single-meta").innerHTML = [
      '<span class="chip">Language · ' + esc(result.language) + "</span>",
      '<span class="chip">Model · ' + esc(result.model) + "</span>",
      '<span class="chip">Scores · ' +
        (isVader ? "VADER compound" : "model probabilities") + "</span>"
    ].join("");

    $("single-bars").innerHTML = [
      bar("Positive", result.positive, "var(--pos)"),
      bar("Neutral", result.neutral, "var(--neu)"),
      bar("Negative", result.negative, "var(--neg)")
    ].join("");

    var note = result.note || (isVader ? "" :
      "Scores are class probabilities from " + result.model +
      ", not VADER compound values.");
    $("single-note-text").textContent = note;
    show($("single-note"), Boolean(note));
  }

  function bar(label, value, color) {
    var pct = Math.round((Number(value) || 0) * 1000) / 10;
    return '<div><div class="bar__top"><span>' + esc(label) +
      '</span><span class="num">' + num(value, 3) + '</span></div>' +
      '<div class="bar__track"><div class="bar__fill" style="width:' + pct +
      "%;background:" + color + '"></div></div></div>';
  }

  /* ------------------------------------------------------------ dashboard */
  function renderDashboard(payload) {
    state.rows = payload.rows || [];
    state.columns = payload.columns || [];
    state.analysisColumns = payload.analysis_columns || [];
    state.textColumn = payload.text_column || "comment";
    state.summary = payload.summary;
    state.fileId = payload.file_id;
    state.meta = payload.meta || {};
    state.page = 1;
    state.sort = { key: null, dir: 1 };

    show($("results"), true);
    show($("results-empty"), false);
    show($("single-card"), false);
    show($("summary-card"), true);
    show($("charts-card"), true);
    show($("table-card"), true);

    renderKPIs(payload);
    renderDonut(payload.summary);
    renderScores(payload.summary);
    renderLanguages(payload.summary);
    buildLanguageFilter();
    configureFilters();
    renderTable();

    $("btn-download").href = "/download/" + payload.file_id;
    $("table-hint").textContent = payload.truncated
      ? "Showing the first " + int(payload.preview_row_limit) +
        " rows in the browser — the CSV download contains every row."
      : int(state.rows.length) + " rows";

    $("summary-card").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderKPIs(payload) {
    var s = payload.summary;
    var label = payload.source === "dataset" ? "Rows analyzed" : "Comments analyzed";
    var polarity = s.avg_polarity;

    var cards = [
      kpi(int(s.analyzed), label,
        s.skipped ? int(s.skipped) + " skipped (no usable text)" : "All rows scored"),
      kpi(s.positive_pct + "%", "Positive", int(s.positive) + " items", "pos"),
      kpi(s.neutral_pct + "%", "Neutral", int(s.neutral) + " items", "neu"),
      kpi(s.negative_pct + "%", "Negative", int(s.negative) + " items", "neg"),
      kpi(signed(polarity, 2), "Average sentiment", "Cross-engine polarity, −1 to +1")
    ];
    $("kpis").innerHTML = cards.join("");

    var context = [];
    if (payload.source === "youtube" && state.meta && state.meta.title) {
      context.push(state.meta.title);
      if (state.meta.channel) { context.push(state.meta.channel); }
    } else if (payload.source === "blog" && state.meta) {
      context.push(state.meta.title || state.meta.url);
      if (state.meta.method) { context.push("via " + state.meta.method); }
    } else if (payload.source === "dataset" && state.meta) {
      context.push(state.meta.filename || "dataset");
      context.push("column: " + state.textColumn);
    }
    $("summary-context").textContent = context.join(" · ");

    var notes = [];
    if (s.vader_rows && s.model_rows) {
      notes.push(int(s.vader_rows) + " English rows scored by VADER, " +
        int(s.model_rows) + " non-English rows by the multilingual model.");
    }
    var fallback = state.rows.some(function (r) {
      return String(r.model || "").indexOf("fallback") !== -1;
    });
    if (fallback) {
      notes.push("Some non-English rows fell back to VADER because the " +
        "multilingual model was unavailable — treat those as unreliable.");
    }
    $("summary-note-text").textContent = notes.join(" ");
    show($("summary-note"), notes.length > 0);
  }

  function kpi(value, label, sub, tone) {
    return '<div class="kpi' + (tone ? " kpi--" + tone : "") + '">' +
      '<div class="kpi__value">' + esc(value) + "</div>" +
      '<div class="kpi__label">' + esc(label) + "</div>" +
      '<div class="kpi__sub">' + esc(sub || "") + "</div></div>";
  }

  function renderDonut(s) {
    var total = s.positive + s.neutral + s.negative;
    var R = 54, C = 2 * Math.PI * R;
    var segments = [
      { name: "Positive", value: s.positive, pct: s.positive_pct, color: "var(--pos)" },
      { name: "Neutral", value: s.neutral, pct: s.neutral_pct, color: "var(--neu)" },
      { name: "Negative", value: s.negative, pct: s.negative_pct, color: "var(--neg)" }
    ];

    var offset = 0;
    var arcs = segments.map(function (seg) {
      var frac = total ? seg.value / total : 0;
      var len = frac * C;
      var arc = '<circle r="' + R + '" cx="70" cy="70" fill="none" stroke="' +
        seg.color + '" stroke-width="20" stroke-dasharray="' + len + " " +
        (C - len) + '" stroke-dashoffset="' + (-offset) +
        '" transform="rotate(-90 70 70)"><title>' + seg.name + ": " +
        seg.pct + "%</title></circle>";
      offset += len;
      return arc;
    }).join("");

    var svg = '<svg width="140" height="140" viewBox="0 0 140 140" role="img" ' +
      'aria-label="Sentiment distribution donut chart">' +
      '<circle r="' + R + '" cx="70" cy="70" fill="none" stroke="var(--border)" ' +
      'stroke-width="20"></circle>' + arcs +
      '<text x="70" y="66" text-anchor="middle" font-family="var(--font-mono)" ' +
      'font-size="21" font-weight="700" fill="var(--ink)">' + int(total) + "</text>" +
      '<text x="70" y="84" text-anchor="middle" font-size="10" ' +
      'fill="var(--ink-muted)" letter-spacing="1">SCORED</text></svg>';

    var legend = '<div class="legend">' + segments.map(function (seg) {
      return '<div class="legend__row"><span class="legend__swatch" style="background:' +
        seg.color + '"></span><span class="legend__name">' + seg.name +
        '</span><span class="num">' + seg.pct + "% · " + int(seg.value) +
        "</span></div>";
    }).join("") + (s.skipped ? '<div class="legend__row"><span class="legend__swatch" ' +
      'style="background:var(--border-strong)"></span><span class="legend__name">' +
      'Unscored</span><span class="num">' + int(s.skipped) + "</span></div>" : "") +
      "</div>";

    $("donut").innerHTML = svg + legend;
  }

  function renderScores(s) {
    var rows = [];
    if (s.avg_compound !== null && s.avg_compound !== undefined) {
      rows.push(scoreRow("Average VADER compound", signed(s.avg_compound, 3),
        "English rows only (" + int(s.vader_rows) + ")"));
    }
    if (s.avg_confidence !== null && s.avg_confidence !== undefined) {
      rows.push(scoreRow("Average model confidence", num(s.avg_confidence, 3),
        "Multilingual rows only (" + int(s.model_rows) + ")"));
    }
    rows.push(bar("Positive share", s.positive_pct / 100, "var(--pos)"));
    rows.push(bar("Neutral share", s.neutral_pct / 100, "var(--neu)"));
    rows.push(bar("Negative share", s.negative_pct / 100, "var(--neg)"));
    $("score-bars").innerHTML = rows.join("");

    var polarity = s.avg_polarity === null || s.avg_polarity === undefined
      ? 0 : Number(s.avg_polarity);
    var left = ((polarity + 1) / 2) * 100;
    $("gauge").innerHTML =
      '<div class="bar__top"><span>Average sentiment</span><span class="num">' +
      signed(s.avg_polarity, 2) + "</span></div>" +
      '<div class="gauge__track"><div class="gauge__needle" style="left:' +
      left + '%"></div></div>' +
      '<div class="gauge__scale"><span>−1.0 negative</span><span>0</span>' +
      "<span>+1.0 positive</span></div>";
  }

  function scoreRow(label, value, hint) {
    return '<div class="bar__top" style="margin-bottom:0"><span>' + esc(label) +
      '<br><span style="font-size:12px;color:var(--ink-faint)">' + esc(hint) +
      '</span></span><span class="num" style="font-size:19px;font-weight:700">' +
      esc(value) + "</span></div>";
  }

  function renderLanguages(s) {
    var langs = s.languages || {};
    var names = Object.keys(langs);
    if (!names.length) { $("language-breakdown").innerHTML = ""; return; }
    var total = names.reduce(function (acc, k) { return acc + langs[k]; }, 0);

    $("language-breakdown").innerHTML =
      '<h3 style="font-size:13px;letter-spacing:.07em;text-transform:uppercase;' +
      'color:var(--ink-muted);margin-bottom:12px;">Language mix</h3>' +
      '<div class="legend">' + names.map(function (name) {
        var pct = Math.round((langs[name] / total) * 1000) / 10;
        return '<div class="legend__row"><span class="legend__name">' + esc(name) +
          '</span><span class="num">' + pct + "% · " + int(langs[name]) +
          "</span></div>";
      }).join("") + "</div>";
  }

  /* ---------------------------------------------------------------- table */
  function columnsForSource() {
    var analysis = [
      { key: "language", label: "Language", type: "tag" },
      { key: "sentiment", label: "Sentiment", type: "sentiment" },
      { key: "compound", label: "Compound", type: "signed", hint: "VADER" },
      { key: "confidence", label: "Confidence", type: "num", hint: "Model" },
      { key: "positive", label: "Pos", type: "num" },
      { key: "neutral", label: "Neu", type: "num" },
      { key: "negative", label: "Neg", type: "num" },
      { key: "model", label: "Model", type: "model" }
    ];

    if (state.source === "youtube") {
      return [
        { key: "author", label: "Author", type: "nowrap" },
        { key: "comment", label: "Comment", type: "text" },
        { key: "published_at", label: "Published", type: "date" },
        { key: "like_count", label: "Likes", type: "int" }
      ].concat(analysis);
    }
    if (state.source === "blog") {
      return [
        { key: "author", label: "Author", type: "nowrap" },
        { key: "comment", label: "Comment", type: "text" },
        { key: "published_at", label: "Published", type: "date" }
      ].concat(analysis);
    }

    // Dataset — every original column is preserved, then analysis appended.
    var analysisKeys = state.analysisColumns;
    var original = state.columns.filter(function (c) {
      return analysisKeys.indexOf(c) === -1;
    }).map(function (c) {
      return {
        key: c,
        label: c,
        type: c === state.textColumn ? "text" : "plain"
      };
    });
    return original.concat(analysis);
  }

  function filteredRows() {
    var q = ($("f-search").value || "").trim().toLowerCase();
    var sentiment = $("f-sentiment").value;
    var language = $("f-language").value;
    var minLikes = parseInt($("f-likes").value, 10);
    var from = $("f-from").value ? new Date($("f-from").value) : null;
    var to = $("f-to").value ? new Date($("f-to").value + "T23:59:59") : null;

    var rows = state.rows.filter(function (row) {
      if (sentiment && row.sentiment !== sentiment) { return false; }
      if (language && row.language !== language) { return false; }
      if (!isNaN(minLikes) && Number(row.like_count || 0) < minLikes) { return false; }
      if (from || to) {
        var d = new Date(row.published_at);
        if (!isNaN(d.getTime())) {
          if (from && d < from) { return false; }
          if (to && d > to) { return false; }
        }
      }
      if (q) {
        var haystack = [row[state.textColumn], row.comment, row.author, row.text]
          .filter(Boolean).join(" ").toLowerCase();
        if (haystack.indexOf(q) === -1) { return false; }
      }
      return true;
    });

    if (state.sort.key) {
      var key = state.sort.key, dir = state.sort.dir;
      rows = rows.slice().sort(function (a, b) {
        var x = a[key], y = b[key];
        var nx = Number(x), ny = Number(y);
        if (!isNaN(nx) && !isNaN(ny) && x !== "" && y !== "" &&
            x !== null && y !== null) {
          return (nx - ny) * dir;
        }
        return String(x === null || x === undefined ? "" : x)
          .localeCompare(String(y === null || y === undefined ? "" : y)) * dir;
      });
    }
    return rows;
  }

  function renderTable() {
    var columns = columnsForSource();
    $("table-head").innerHTML = columns.map(function (col) {
      var sortState = state.sort.key === col.key
        ? (state.sort.dir === 1 ? "ascending" : "descending") : "none";
      return '<th scope="col" data-key="' + esc(col.key) + '" aria-sort="' +
        sortState + '" title="Sort by ' + esc(col.label) + '">' + esc(col.label) +
        (col.hint ? ' <span style="font-weight:400;text-transform:none;' +
          'letter-spacing:0">(' + esc(col.hint) + ")</span>" : "") + "</th>";
    }).join("");

    var rows = filteredRows();
    var pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    state.page = Math.min(state.page, pages);
    var slice = rows.slice((state.page - 1) * PAGE_SIZE, state.page * PAGE_SIZE);

    $("table-body").innerHTML = slice.length
      ? slice.map(function (row) {
          return "<tr>" + columns.map(function (col) {
            return cell(row, col);
          }).join("") + "</tr>";
        }).join("")
      : '<tr><td colspan="' + columns.length +
        '" style="padding:34px;text-align:center;color:var(--ink-muted)">' +
        "No rows match these filters.</td></tr>";

    updateScrollHint();
    $("table-count").textContent = int(rows.length) + " of " +
      int(state.rows.length) + " rows shown";
    $("page-label").textContent = state.page + " / " + pages;
    $("page-prev").disabled = state.page <= 1;
    $("page-next").disabled = state.page >= pages;
  }

  function updateScrollHint() {
    var scroller = $("table-scroll");
    var wrap = $("table-wrap");
    if (!scroller || !wrap) { return; }
    var atEnd = scroller.scrollLeft + scroller.clientWidth >=
      scroller.scrollWidth - 2;
    var overflows = scroller.scrollWidth > scroller.clientWidth + 2;
    wrap.classList.toggle("is-scrollable", overflows && !atEnd);
  }

  function shortModel(value) {
    if (!value) { return "—"; }
    var name = String(value);
    if (name.indexOf("VADER") === 0) { return name; }
    if (name === "none") { return "—"; }
    var tail = name.split("/").pop();          // drop the org prefix
    if (tail.length > 22) { tail = tail.slice(0, 21) + "…"; }
    return tail;
  }

  function cell(row, col) {
    var value = row[col.key];
    switch (col.type) {
      case "sentiment":
        return '<td class="cell--nowrap"><span class="tag ' + sentClass(value) +
          '">' + esc(value || "—") + "</span></td>";
      case "tag":
        return '<td class="cell--nowrap"><span class="tag tag--muted">' +
          esc(value || "—") + "</span></td>";
      case "model":
        // Hugging Face ids are long ("cardiffnlp/twitter-xlm-roberta-base-
        // sentiment"), so show a short label and keep the full id in the
        // tooltip. The exported CSV always carries the full id.
        return '<td class="cell--nowrap"><span class="tag tag--muted" title="' +
          esc(value || "") + '">' + esc(shortModel(value)) + "</span></td>";
      case "signed":
        return '<td class="cell--num">' + signed(value, 3) + "</td>";
      case "num":
        return '<td class="cell--num">' + num(value, 3) + "</td>";
      case "int":
        return '<td class="cell--num">' + int(value) + "</td>";
      case "date":
        return '<td class="cell--nowrap">' + esc(dateOnly(value)) + "</td>";
      case "nowrap":
        return '<td class="cell--nowrap">' + esc(value || "—") + "</td>";
      case "text":
        return (
          '<td class="cell--text"><div class="cell-body">' +
          esc(value || "") +
          "</div></td>"
        );
      default:
        return "<td>" + esc(value === null || value === undefined ? "" : value) + "</td>";
    }
  }

  function buildLanguageFilter() {
    var select = $("f-language");
    var seen = {};
    state.rows.forEach(function (row) {
      if (row.language) { seen[row.language] = true; }
    });
    select.innerHTML = '<option value="">All</option>' +
      Object.keys(seen).sort().map(function (name) {
        return '<option value="' + esc(name) + '">' + esc(name) + "</option>";
      }).join("");
  }

  function configureFilters() {
    var isYouTube = state.source === "youtube";
    var hasDates = state.rows.some(function (r) { return Boolean(r.published_at); });
    show($("f-likes-wrap"), isYouTube);
    show($("f-from-wrap"), hasDates);
    show($("f-to-wrap"), hasDates);
    ["f-search", "f-sentiment", "f-language", "f-likes", "f-from", "f-to"]
      .forEach(function (id) { $(id).value = ""; });
  }

  /* ----------------------------------------------------------- run: text */
  function runText() {
    var button = $("btn-text");
    var text = $("text-input").value;
    if (!text.trim()) { setError("Please enter some text to analyze."); return; }

    clearError();
    setBusy(button, true);
    setProgress("Analyzing…", 0, null, null);

    postJSON("/api/analyze/text", { text: text })
      .then(function (data) {
        hideProgress();
        renderSingle(data.result);
      })
      .catch(function (err) { hideProgress(); setError(err.message); })
      .finally(function () { setBusy(button, false, "Analyze Sentiment"); });
  }

  /* -------------------------------------------------------- run: dataset */
  function uploadDataset(file) {
    if (!file) { return; }
    if (!/\.csv$/i.test(file.name)) {
      setError("Only .csv files are supported.");
      return;
    }
    clearError();
    resetResults();
    setProgress("Reading columns…", 0, null, null);

    var form = new FormData();
    form.append("file", file);

    fetch("/api/dataset/inspect", { method: "POST", body: form })
      .then(readJSON)
      .then(function (data) {
        hideProgress();
        state.upload = data;
        $("file-name").textContent = data.filename + " · " + int(data.rows) + " rows";
        show($("file-pill"), true);
        show($("dataset-config"), true);

        var columnSelect = $("dataset-column");
        columnSelect.innerHTML = data.columns.map(function (col) {
          var label = col.name + " (" + int(col.filled) + " filled, ~" +
            col.avg_length + " chars)";
          return '<option value="' + esc(col.name) + '"' +
            (col.name === data.suggested_column ? " selected" : "") + ">" +
            esc(label) + "</option>";
        }).join("");

        $("dataset-column-help").textContent = data.suggested_column
          ? "Detected \u201c" + data.suggested_column +
            "\u201d as the text column — change it if that's wrong."
          : "No standard text column name was found. Pick the column that " +
            "holds the text.";

        var rowSelect = $("dataset-rows");
        rowSelect.innerHTML = ['<option value="all" selected>All rows (' +
          int(data.rows) + ")</option>"].concat(
          data.row_choices.map(function (n) {
            return '<option value="' + n + '">First ' + int(n) + " rows</option>";
          })).join("");
        $("dataset-rows-help").textContent =
          "Empty and invalid rows are kept and marked UNSCORED, never dropped.";
      })
      .catch(function (err) { hideProgress(); setError(err.message); });
  }

  function runDataset() {
    if (!state.upload) { setError("Please upload a CSV first."); return; }
    var button = $("btn-dataset");
    clearError();
    setBusy(button, true);
    setProgress("Queued…", 0, null, null);

    postJSON("/api/dataset/analyze", {
      upload_id: state.upload.upload_id,
      text_column: $("dataset-column").value,
      limit: $("dataset-rows").value
    })
      .then(function (data) {
        pollJob(data.job_id, function (result) {
          hideProgress();
          setBusy(button, false, "Analyze Dataset");
          renderDashboard(result);
        }, function (err) {
          hideProgress();
          setBusy(button, false, "Analyze Dataset");
          setError(err.message);
        });
      })
      .catch(function (err) {
        hideProgress();
        setBusy(button, false, "Analyze Dataset");
        setError(err.message);
      });
  }

  /* -------------------------------------------------------- run: youtube */
  function runYouTube() {
    var button = $("btn-youtube");
    var url = $("yt-url").value.trim();
    if (!url) { setError("Please paste a YouTube video URL."); return; }

    clearError();
    resetResults();
    setBusy(button, true);
    setProgress("Contacting YouTube…", 0, null, null);

    postJSON("/api/youtube/analyze", {
      url: url,
      limit: $("yt-limit").value,
      order: $("yt-order").value,
      include_replies: $("yt-replies").checked
    })
      .then(function (data) {
        pollJob(data.job_id, function (result) {
          hideProgress();
          setBusy(button, false, "Analyze YouTube Comments");
          renderDashboard(result);
        }, function (err) {
          hideProgress();
          setBusy(button, false, "Analyze YouTube Comments");
          setError(err.message);
        });
      })
      .catch(function (err) {
        hideProgress();
        setBusy(button, false, "Analyze YouTube Comments");
        setError(err.message);
      });
  }

  /* ----------------------------------------------------------- run: blog */
  function runBlog() {
    var button = $("btn-blog");
    var url = $("blog-url").value.trim();
    if (!url) { setError("Please paste a blog or article URL."); return; }

    clearError();
    resetResults();
    setBusy(button, true);
    setProgress("Checking robots.txt…", 0, null, null);

    postJSON("/api/blog/analyze", { url: url, limit: $("blog-limit").value })
      .then(function (data) {
        pollJob(data.job_id, function (result) {
          hideProgress();
          setBusy(button, false, "Analyze Blog Comments");
          renderDashboard(result);
        }, function (err) {
          hideProgress();
          setBusy(button, false, "Analyze Blog Comments");
          setError(err.message);
        });
      })
      .catch(function (err) {
        hideProgress();
        setBusy(button, false, "Analyze Blog Comments");
        setError(err.message);
      });
  }

  /* ---------------------------------------------------------------- init */
  function bind() {
    all(".source-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { selectSource(btn.dataset.source); });
      btn.addEventListener("keydown", function (event) {
        var order = ["text", "dataset", "youtube", "blog"];
        var i = order.indexOf(state.source);
        if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
          event.preventDefault();
          var next = order[(i + (event.key === "ArrowRight" ? 1 : 3)) % 4];
          selectSource(next);
          $("tab-" + next).focus();
        }
      });
    });

    $("btn-text").addEventListener("click", runText);
    $("text-input").addEventListener("keydown", function (event) {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { runText(); }
    });
    $("btn-text-sample").addEventListener("click", function () {
      var samples = [
        "I absolutely love this product! Best purchase of the year.",
        "\u092c\u0939\u0941\u0924 \u0905\u091b\u094d\u0925\u093e \u0935\u0940\u0921\u093f\u092f\u094b \u0939\u0948\u0964",
        "Bhai kya hi video hai \uD83D\uDE02",
        "The delivery was late and the box arrived damaged."
      ];
      $("text-input").value = samples[Math.floor(Math.random() * samples.length)];
      $("text-input").focus();
    });

    $("drop").addEventListener("click", function () { $("file-input").click(); });
    $("drop").addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        $("file-input").click();
      }
    });
    ["dragenter", "dragover"].forEach(function (name) {
      $("drop").addEventListener(name, function (event) {
        event.preventDefault();
        $("drop").classList.add("is-over");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      $("drop").addEventListener(name, function (event) {
        event.preventDefault();
        $("drop").classList.remove("is-over");
      });
    });
    $("drop").addEventListener("drop", function (event) {
      if (event.dataTransfer.files.length) {
        uploadDataset(event.dataTransfer.files[0]);
      }
    });
    $("file-input").addEventListener("change", function (event) {
      if (event.target.files.length) { uploadDataset(event.target.files[0]); }
    });
    $("btn-file-clear").addEventListener("click", function () {
      state.upload = null;
      $("file-input").value = "";
      show($("file-pill"), false);
      show($("dataset-config"), false);
      resetResults();
    });

    $("btn-dataset").addEventListener("click", runDataset);
    $("btn-youtube").addEventListener("click", runYouTube);
    $("btn-blog").addEventListener("click", runBlog);
    [$("yt-url"), $("blog-url")].forEach(function (input) {
      input.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
          event.preventDefault();
          (input.id === "yt-url" ? runYouTube : runBlog)();
        }
      });
    });

    ["f-search", "f-sentiment", "f-language", "f-likes", "f-from", "f-to"]
      .forEach(function (id) {
        $(id).addEventListener("input", function () {
          state.page = 1;
          renderTable();
        });
      });

    $("table-head").addEventListener("click", function (event) {
      var th = event.target.closest("th");
      if (!th) { return; }
      var key = th.dataset.key;
      state.sort = {
        key: key,
        dir: state.sort.key === key ? -state.sort.dir : 1
      };
      renderTable();
    });

    $("page-prev").addEventListener("click", function () {
      state.page = Math.max(1, state.page - 1);
      renderTable();
    });
    $("page-next").addEventListener("click", function () {
      state.page += 1;
      renderTable();
    });
  }

  function loadStatus() {
    fetch("/api/status").then(function (r) { return r.json(); }).then(function (s) {
      var dot = $("dot-multilingual");
      var chip = $("chip-multilingual");
      // ready   = packages installed AND model already in memory
      // on demand = packages installed, model loads on first non-English text
      // unavailable / disabled = it will fall back to VADER, and say so
      var label;
      if (s.multilingual_loaded) { label = "ready"; }
      else if (s.multilingual_ready) { label = "on demand"; }
      else if (s.multilingual_enabled) { label = "unavailable"; }
      else { label = "disabled"; }
      dot.className = "dot " + (s.multilingual_ready ? "dot--on" : "dot--off");
      chip.lastChild.textContent = "Multilingual · " + label;
      chip.title = s.multilingual_note ||
        (s.multilingual_model + " — loaded on first non-English text.");
    }).catch(function () { /* status is cosmetic */ });
  }

  document.addEventListener("DOMContentLoaded", function () {
    bind();
    var scroller = $("table-scroll");
    if (scroller) { scroller.addEventListener("scroll", updateScrollHint); }
    window.addEventListener("resize", updateScrollHint);
    selectSource(CONF.initialSource || "text");
    loadStatus();
  });
})();
