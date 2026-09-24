// Audiobook viewer: render the PDF with PDF.js, play the narration audio, and highlight
// the current word by looking up the timing JSON against audio.currentTime.
//
// Coordinate note (important): the timing.json bboxes come from PyMuPDF, whose coordinate
// origin is the TOP-LEFT of the page with y growing downward — exactly like an HTML canvas.
// A PDF.js canvas rendered at scale S has pixel size (pageWidthPoints * S). So the highlight
// rectangle in canvas pixels is simply bbox * S — no y-flip, no viewport transform needed.

import * as pdfjsLib from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.min.mjs";
pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.worker.min.mjs";

const els = {
  file: document.getElementById("file"),
  tts: document.getElementById("tts"),
  status: document.getElementById("status"),
  canvas: document.getElementById("canvas"),
  highlight: document.getElementById("highlight"),
  audio: document.getElementById("audio"),
  pagelabel: document.getElementById("pagelabel"),
  stage: document.getElementById("stage"),
};

const ctx = els.canvas.getContext("2d");

let state = {
  pdf: null,            // pdf.js document
  timing: null,         // parsed timing.json
  words: [],            // timing.words
  scale: 1,             // current render scale (canvas px per PDF point)
  renderedPage: -1,     // which page index is currently on the canvas
  activeIdx: -1,        // index of the currently highlighted word
};

function setStatus(msg) { els.status.textContent = msg; }

// Fit the page to the available width (max 900px), returning the render scale.
function computeScale(pageWidthPts) {
  const maxWidth = Math.min(window.innerWidth - 60, 900);
  return maxWidth / pageWidthPts;
}

async function renderPage(pageIndex) {
  if (!state.pdf || pageIndex === state.renderedPage) return;
  const page = await state.pdf.getPage(pageIndex + 1); // pdf.js is 1-based
  const meta = state.timing.pages[pageIndex];
  state.scale = computeScale(meta.width);

  const viewport = page.getViewport({ scale: state.scale });
  const dpr = window.devicePixelRatio || 1;
  els.canvas.width = Math.floor(viewport.width * dpr);
  els.canvas.height = Math.floor(viewport.height * dpr);
  els.canvas.style.width = `${viewport.width}px`;
  els.canvas.style.height = `${viewport.height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  await page.render({ canvasContext: ctx, viewport }).promise;
  state.renderedPage = pageIndex;
  els.pagelabel.textContent = `Page ${pageIndex + 1} / ${state.timing.pages.length}`;
}

// Position the highlight div over the given word (bbox in PDF points).
function positionHighlight(word) {
  if (!word || !word.bbox) { els.highlight.classList.remove("on"); return; }
  const s = state.scale;
  const [x0, y0, x1, y1] = word.bbox;
  els.highlight.style.left = `${x0 * s}px`;
  els.highlight.style.top = `${y0 * s}px`;
  els.highlight.style.width = `${(x1 - x0) * s}px`;
  els.highlight.style.height = `${(y1 - y0) * s}px`;
  els.highlight.classList.add("on");
}

// Binary search: last word whose start <= t.
function findWordIndex(t) {
  const w = state.words;
  let lo = 0, hi = w.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (w[mid].start <= t) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  // If we're past the current word's end and before the next start, keep showing it.
  return ans;
}

async function onTimeUpdate() {
  if (!state.words.length) return;
  const t = els.audio.currentTime;
  const idx = findWordIndex(t);
  if (idx < 0) { els.highlight.classList.remove("on"); return; }
  const word = state.words[idx];

  // Hide the highlight during gaps (e.g. between sentences / synthetic announcements).
  if (t > word.end + 0.05) { els.highlight.classList.remove("on"); }

  if (idx !== state.activeIdx || word.page !== state.renderedPage) {
    state.activeIdx = idx;
    await renderPage(word.page);
    positionHighlight(word);
  } else if (t <= word.end + 0.05) {
    positionHighlight(word);
  }
}

// Load a generated job from a set of URLs (timing / audio / pdf).
async function loadJob(timingUrl, audioUrl, pdfUrl) {
  setStatus("Loading timing…");
  state.timing = await (await fetch(timingUrl)).json();
  state.words = state.timing.words;
  state.renderedPage = -1;
  state.activeIdx = -1;

  setStatus("Loading PDF…");
  state.pdf = await pdfjsLib.getDocument(pdfUrl).promise;
  await renderPage(0);

  els.audio.src = audioUrl;
  setStatus(`Ready — ${state.words.length} words, ${state.timing.duration.toFixed(1)}s. Press play.`);
}

// Upload a PDF and have the backend build the audiobook.
async function processUpload(file) {
  setStatus("Uploading & processing… (first run may take a moment)");
  const form = new FormData();
  form.append("file", file);
  const url = `/api/process?tts=${encodeURIComponent(els.tts.value)}`;
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) { setStatus("Processing failed: " + (await res.text())); return; }
  const job = await res.json();
  await loadJob(job.timing_url, job.audio_url, job.pdf_url);
}

els.file.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) processUpload(file);
});

els.audio.addEventListener("timeupdate", onTimeUpdate);
window.addEventListener("resize", () => { state.renderedPage = -1; onTimeUpdate(); });

// If launched with ?src=/out (a pre-generated dir served statically), load it directly.
const params = new URLSearchParams(location.search);
const src = params.get("src");
if (src) {
  const base = src.replace(/\/$/, "");
  loadJob(`${base}/timing.json`, `${base}/audio.wav`, `${base}/source.pdf`)
    .catch((err) => setStatus("Could not load job: " + err.message));
}
