/* PDF Audiobook frontend.
 * Renders PDF pages with pdf.js, overlays highlight boxes from the manifest,
 * and drives highlighting from the audio element's currentTime.
 */
(function () {
  "use strict";

  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file");
  const statusEl = document.getElementById("status");
  const goBtn = document.getElementById("go");
  const playerBox = document.getElementById("player");
  const audio = document.getElementById("audio");
  const viewer = document.getElementById("viewer");

  let manifest = null;
  let segments = [];         // sorted by start
  let boxEls = new Map();    // segId -> [elements]
  let pageScales = [];       // per-page render scale
  let activeSegId = -1;

  function opt(id) { return document.getElementById(id); }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;
    goBtn.disabled = true;
    setStatus("Uploading and processing… (first run may take a moment)");
    viewer.innerHTML = "";
    boxEls.clear();

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    fd.append("read_figures", opt("read_figures").checked);
    fd.append("read_tables", opt("read_tables").checked);
    fd.append("read_captions", opt("read_captions").checked);
    fd.append("speak_placeholders", opt("speak_placeholders").checked);
    fd.append("tts_provider", opt("tts_provider").value);
    fd.append("ocr_provider", opt("ocr_provider").value);

    try {
      const res = await fetch("/api/process", { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      manifest = await res.json();
      await onProcessed();
    } catch (err) {
      setStatus("Error: " + err.message);
    } finally {
      goBtn.disabled = false;
    }
  });

  async function onProcessed() {
    segments = (manifest.segments || []).slice().sort((a, b) => a.start - b.start);
    const notes = manifest.notes && manifest.notes.length
      ? " • " + manifest.notes.length + " note(s)" : "";
    setStatus(
      `Done: ${manifest.pages.length} page(s), ${segments.length} segments, ` +
      `${manifest.audio_duration}s, TTS=${manifest.tts_provider}` +
      (manifest.ocr_used ? ", OCR used" : "") + notes
    );

    audio.src = manifest.audio_url;
    playerBox.hidden = false;

    await renderPdf(manifest.pdf_url);
    buildOverlays();
  }

  async function renderPdf(url) {
    const pdf = await pdfjsLib.getDocument(url).promise;
    const containerWidth = Math.min(viewer.clientWidth - 48, 900);
    pageScales = [];

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const unscaled = page.getViewport({ scale: 1 });
      const scale = containerWidth / unscaled.width;
      pageScales[i - 1] = scale;
      const viewport = page.getViewport({ scale });

      const wrap = document.createElement("div");
      wrap.className = "page-wrap";
      wrap.id = "page-" + (i - 1);
      wrap.style.width = viewport.width + "px";
      wrap.style.height = viewport.height + "px";

      const canvas = document.createElement("canvas");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const overlay = document.createElement("div");
      overlay.className = "overlay";
      overlay.id = "overlay-" + (i - 1);

      wrap.appendChild(canvas);
      wrap.appendChild(overlay);
      viewer.appendChild(wrap);

      await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
    }
  }

  function buildOverlays() {
    boxEls.clear();
    for (const seg of segments) {
      const overlay = document.getElementById("overlay-" + seg.page);
      if (!overlay) continue;
      const scale = pageScales[seg.page] || 1;
      const els = [];
      for (const bb of seg.bboxes) {
        const [x0, y0, x1, y1] = bb;
        const el = document.createElement("div");
        el.className = "hl " + seg.type;
        el.style.left = x0 * scale + "px";
        el.style.top = y0 * scale + "px";
        el.style.width = (x1 - x0) * scale + "px";
        el.style.height = (y1 - y0) * scale + "px";
        el.title = seg.text || seg.type;
        el.addEventListener("click", () => {
          if (seg.spoken) { audio.currentTime = seg.start + 0.01; audio.play(); }
        });
        overlay.appendChild(el);
        els.push(el);
      }
      boxEls.set(seg.id, els);
    }
  }

  audio.addEventListener("timeupdate", () => {
    const t = audio.currentTime;
    const seg = findSegmentAt(t);
    const id = seg ? seg.id : -1;
    if (id === activeSegId) return;
    // clear old
    if (boxEls.has(activeSegId)) boxEls.get(activeSegId).forEach(e => e.classList.remove("active"));
    activeSegId = id;
    if (seg && boxEls.has(id)) {
      boxEls.get(id).forEach(e => e.classList.add("active"));
      const first = boxEls.get(id)[0];
      if (first) first.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  });

  function findSegmentAt(t) {
    // Only spoken segments occupy time; binary-ish linear scan is fine.
    let match = null;
    for (const s of segments) {
      if (s.spoken && t >= s.start && t < s.end) { match = s; break; }
    }
    return match;
  }

  function setStatus(msg) { statusEl.textContent = msg; }
})();
