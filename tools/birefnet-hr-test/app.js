/* PhotoCraft BiRefNet HR v3 / FP32. Classic script: no local .mjs MIME dependency. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let source = null, raw = null, corrected = null, alpha = null, reference = null;
  let worker = null, ready = null, busy = false, loading = false, job = 0, loadId = 0;
  let sourceName = 'kep', clock = null, started = 0, metrics = null, appliedStrength = 0;
  let diagnosticReport = null;
  function status(message, value) { $('status').textContent = message; if (Number.isFinite(value)) $('progress').value = value; }
  function controls() {
    $('file').disabled = busy; $('run').disabled = busy || loading || !source || !navigator.gpu || !isSecureContext;
    $('cancel').disabled = !busy; $('referenceFile').disabled = busy || loading || !source;
    $('strength').disabled = busy || !raw || !worker; $('clearCache').disabled = busy;
    $('download').disabled = $('downloadMask').disabled = busy || !raw;
    $('downloadReport').disabled = !diagnosticReport;
  }
  function setBusy(value) {
    busy = value; clearInterval(clock);
    if (value) { started = performance.now(); $('timer').textContent = '0 mp'; clock = setInterval(() => { $('timer').textContent = `${Math.round((performance.now() - started) / 1000)} mp`; }, 1000); }
    else if (started) $('timer').textContent = `${((performance.now() - started) / 1000).toFixed(1)} mp összesen`;
    controls();
  }
  function draw(rgba, canvas) { canvas.width = source.width; canvas.height = source.height; canvas.getContext('2d').putImageData(new ImageData(rgba, source.width, source.height), 0, 0); }
  function grayMask(data) { const rgba = new Uint8ClampedArray(data.length * 4); for (let i = 0; i < data.length; i++) { rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = data[i]; rgba[i * 4 + 3] = 255; } return rgba; }
  function render() {
    const mode = $('resultMode').value, background = $('background').value;
    document.querySelectorAll('.output').forEach(stage => { stage.dataset.background = background; });
    $('resultTitle').textContent = mode === 'raw' ? 'BiRefNet · nyers eredmény' : `BiRefNet · peremtisztítás ${appliedStrength}%`;
    if (raw) draw(background === 'mask' ? grayMask(alpha) : mode === 'raw' ? raw : corrected, $('result'));
    if (reference) {
      if (background === 'mask') { const a = new Uint8ClampedArray(source.width * source.height); for (let i = 0; i < a.length; i++) a[i] = reference[i * 4 + 3]; draw(grayMask(a), $('reference')); }
      else draw(reference, $('reference'));
    }
  }
  function zoom() { document.querySelectorAll('.stage-inner').forEach(el => { el.style.width = !source || $('zoom').value === 'fit' ? '100%' : `${source.width * Number($('zoom').value)}px`; }); }
  function removeReference() { reference = null; $('referenceFile').value = ''; $('referenceCard').hidden = true; $('comparison').classList.remove('has-reference'); $('reference').width = $('reference').height = 1; }
  function stopWorker() { worker?.terminate(); worker = null; ready = null; }
  function fail(error) { status(`Hiba: ${String(error?.message ?? error)}`, 0); setBusy(false); }
  function showDiagnostics(report) {
    diagnosticReport = { ...report, browser: navigator.userAgent };
    $('diagnostics').textContent = JSON.stringify(diagnosticReport, null, 2);
    $('downloadReport').disabled = false;
  }
  async function getWorker() {
    if (ready) return ready;
    const instance = new Worker('./worker.js?v=3'); worker = instance;
    ready = new Promise((resolve, reject) => {
      const timer = setTimeout(() => { reject(new Error('A feldolgozó nem indult el. Ellenőrizd, hogy a worker.js is fel van töltve.')); }, 12000);
      instance.addEventListener('error', event => {
        clearTimeout(timer); reject(new Error(event.message || 'A worker.js nem tölthető be.'));
        if (worker === instance) { stopWorker(); fail(new Error('A feldolgozó leállt. Ellenőrizd a worker.js fájlt, majd indítsd újra.')); }
      });
      instance.addEventListener('message', ({ data }) => {
        if (worker !== instance) return;
        if (data.type === 'ready') { clearTimeout(timer); if (data.version !== '3.0.0') { reject(new Error('Eltérő verziójú feldolgozó töltődött be. A v3 csomag összes fájlját másold felül, majd frissíts Ctrl+Shift+R-rel.')); return; } resolve(instance); return; }
        if (data.id !== job) return;
        if (data.type === 'progress') status(data.message, data.value);
        else if (data.type === 'notice') { $('notice').hidden = false; $('notice').textContent = data.message; }
        else if (data.type === 'diagnostics') showDiagnostics(data.report);
        else if (data.type === 'metrics') { metrics = data; }
        else if (data.type === 'error') { $('diagnostics').textContent += `\nHiba: ${data.detail}`; stopWorker(); fail(new Error(data.message)); }
        else if (data.type === 'result') {
          if (!source || data.width !== source.width || data.height !== source.height || data.rgba?.byteLength !== source.width * source.height * 4 || data.alpha?.byteLength !== source.width * source.height) { stopWorker(); fail(new Error('Az eredmény mérete hibás.')); return; }
          if (data.raw) raw = new Uint8ClampedArray(data.raw);
          corrected = new Uint8ClampedArray(data.rgba); alpha = new Uint8ClampedArray(data.alpha); appliedStrength = Math.round(data.strength * 100);
          $('resultEmpty').hidden = true; render(); setBusy(false);
          status('Kész. Ellenőrizd a haj- és szőrszálakat fehér és fekete háttéren, 100–200%-os nézetben.', 1);
        }
      });
    });
    try { return await ready; } catch (error) { if (worker === instance) stopWorker(); throw error; }
  }
  async function readFile(file) {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 40 * 1024 * 1024) throw new Error('JPG, PNG vagy WebP képet válassz, legfeljebb 40 MB méretben.');
    let bitmap;
    try {
      bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
      if (bitmap.width > 8192 || bitmap.height > 8192 || bitmap.width * bitmap.height > 16_000_000) throw new Error('Legfeljebb 16 megapixeles, oldalanként 8192 pixeles kép használható.');
      const canvas = document.createElement('canvas'); canvas.width = bitmap.width; canvas.height = bitmap.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true }); ctx.drawImage(bitmap, 0, 0);
      return { width: canvas.width, height: canvas.height, data: ctx.getImageData(0, 0, canvas.width, canvas.height).data };
    } finally { bitmap?.close(); }
  }
  $('file').addEventListener('change', async () => {
    const file = $('file').files?.[0]; if (!file || busy) return;
    const version = ++loadId; job++; stopWorker(); loading = true; source = raw = corrected = alpha = null; metrics = diagnosticReport = null; removeReference();
    for (const id of ['original', 'result']) $(id).width = $(id).height = 1;
    $('originalEmpty').hidden = $('resultEmpty').hidden = false; $('notice').hidden = true; $('timer').textContent = ''; started = 0;
    $('strength').value = '0'; $('strengthValue').value = '0%'; $('resultMode').value = 'raw'; appliedStrength = 0;
    $('diagnostics').textContent = 'v3 · BiRefNet_HR-matting · 2048 × 2048 · FP32'; controls(); status('Kép betöltése…', 0);
    try {
      const image = await readFile(file); if (version !== loadId) return; source = image;
      sourceName = file.name.replace(/\.[^.]+$/, '').replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_') || 'kep';
      draw(source.data, $('original')); $('originalEmpty').hidden = true; $('dimensions').textContent = `${source.width} × ${source.height} px`;
      zoom(); status(navigator.gpu && isSecureContext ? 'A kép kész. Indítsd el a háttéreltávolítást.' : 'Az előnézet kész. A feldolgozáshoz HTTPS és WebGPU-képes asztali böngésző szükséges.', 0);
    } catch (error) { if (version === loadId) fail(error); }
    finally { if (version === loadId) { loading = false; controls(); } }
  });
  $('run').addEventListener('click', async () => {
    if (busy || loading || !source) return;
    const id = ++job;
    // A previous successful mask must not be exported as if a failed rerun had
    // succeeded. Preserve the source/reference, discard only derived results.
    raw = corrected = alpha = null; metrics = diagnosticReport = null;
    $('result').width = $('result').height = 1; $('resultEmpty').hidden = false;
    $('diagnostics').textContent = 'v3 · FP32 feldolgozás indul…';
    setBusy(true); $('notice').hidden = true;
    try {
      const active = await getWorker(); if (id !== job) return;
      const copy = source.data.slice();
      active.postMessage({ type: 'run', id, width: source.width, height: source.height, rgba: copy.buffer, strength: Number($('strength').value) / 100 }, [copy.buffer]);
    } catch (error) { if (id === job) fail(error); }
  });
  $('cancel').addEventListener('click', () => { if (!busy) return; job++; stopWorker(); setBusy(false); status('Megszakítva. Az eredeti kép megmaradt, újraindíthatod a feldolgozást.', 0); });
  $('strength').addEventListener('input', () => { $('strengthValue').value = `${$('strength').value}%`; });
  $('strength').addEventListener('change', () => {
    if (busy || !worker || !raw) return;
    $('resultMode').value = 'corrected'; setBusy(true); status('Szegélyszín tisztítása · a modell újrafuttatása nélkül…', .95);
    worker.postMessage({ type: 'compose', id: ++job, strength: Number($('strength').value) / 100 });
  });
  $('referenceFile').addEventListener('change', async () => {
    const file = $('referenceFile').files?.[0]; if (!file || busy || !source) return;
    const version = loadId;
    try {
      const image = await readFile(file); if (version !== loadId || !source) return;
      if (image.width !== source.width || image.height !== source.height) throw new Error(`Az összehasonlító kép legyen ugyanakkora, mint az eredeti: ${source.width} × ${source.height} px.`);
      reference = image.data; $('referenceCard').hidden = false; $('comparison').classList.add('has-reference'); zoom(); render(); status('Összehasonlító kép betöltve. A nézetek nagyítása és görgetése együtt változik.');
    } catch (error) { if (version === loadId) status(`Összehasonlítás: ${error.message}`); }
  });
  $('removeReference').addEventListener('click', removeReference);
  $('background').addEventListener('change', render); $('resultMode').addEventListener('change', render); $('zoom').addEventListener('change', zoom);
  let scrolling = false;
  document.querySelectorAll('.stage').forEach(stage => stage.addEventListener('scroll', () => {
    if (scrolling || $('zoom').value === 'fit') return; scrolling = true;
    document.querySelectorAll('.stage').forEach(other => { if (other !== stage) { other.scrollTop = stage.scrollTop; other.scrollLeft = stage.scrollLeft; } });
    requestAnimationFrame(() => { scrolling = false; });
  }));
  async function download(isMask) {
    if (!raw || busy) return;
    try {
      const mode = $('resultMode').value, canvas = document.createElement('canvas');
      draw(isMask ? grayMask(alpha) : mode === 'raw' ? raw : corrected, canvas);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png')); if (!blob) throw new Error('A PNG nem hozható létre.');
      const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = `${sourceName}-birefnet-hr-${isMask ? 'alfamaszk' : mode === 'raw' ? 'nyers' : 'tisztitott'}.png`; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (error) { fail(error); }
  }
  $('download').addEventListener('click', () => download(false)); $('downloadMask').addEventListener('click', () => download(true));
  $('downloadReport').addEventListener('click', () => {
    if (!diagnosticReport) return;
    const blob = new Blob([JSON.stringify(diagnosticReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = 'photocraft-birefnet-v3-hibajelentes.json'; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
  });
  $('clearCache').addEventListener('click', async () => { if (busy) return; try { await caches.delete('photocraft-birefnet-hr-v1'); await caches.delete('photocraft-birefnet-hr-fp32-v2'); status('Modellgyorsítótár törölve. A következő futás újra letölti az FP32 modellt.'); } catch { status('A böngésző nem engedi a gyorsítótár kezelését.'); } });
  window.addEventListener('pagehide', () => { stopWorker(); clearInterval(clock); });
  $('gpu').textContent = !isSecureContext ? 'HTTPS címen nyisd meg ezt az oldalt. Helyi teszthez localhost használható.' : !navigator.gpu ? 'WebGPU nem érhető el. Próbáld asztali Chrome, Edge vagy Opera böngészőben, bekapcsolt hardveres gyorsítással.' : 'v3 · FP32 · csökkentett pufferigény. A v2-ben már letöltött modell újra használható. Első letöltés: kb. 932 MB; a feldolgozás a saját videokártyádon fut.';
  controls(); document.documentElement.dataset.appReady = '1'; status('Válaszd ki az eredeti fotót.', 0);
})();
