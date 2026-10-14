import { buildTrimap, resizeMask, mask, validateSize, clamp } from './matting-core.mjs?v=5';
import { refineAlpha, readAlphaTensor, checkAbort } from './matting-engine.mjs?v=5';
import { composeCutout } from './edge-colors.mjs?v=5';

const $ = (id) => document.getElementById(id);
const VERSION = '5';
const BEN_REV = 'c552aa82688edce09f0ac9d2e31ad53d9d629010';
const MATTE_REV = '6bc1297f6140f055a227b6d2cfe8c093281f35d2';
let source = null, sourceName = 'kep', subjectAlpha = null, finalImage = null, finalAlpha = null;
let marks = null, busy = false, dirty = false, controller = null, loadVersion = 0;
let hfPromise = null, matteProcessor = null, matteModel = null;
let stroke = null, lastPoint = null, history = [];

$('build').textContent = `PhotoCraft · háttéreltávolítás v${VERSION}`;
$('gpu').textContent = navigator.gpu
  ? 'A feldolgozás a böngésződben fut. Első alkalommal kb. 320 MB modell töltődik le; a képet nem küldjük szerverre.'
  : 'A feldolgozáshoz WebGPU-képes Chrome, Edge vagy Opera és bekapcsolt hardveres gyorsítás szükséges.';

function status(text, progress) {
  $('status').textContent = text;
  if (Number.isFinite(progress)) $('progress').value = clamp(progress, 0, 1);
}
function controls() {
  $('file').disabled = busy;
  $('run').disabled = busy || !source || !navigator.gpu;
  $('download').disabled = busy || !finalImage || dirty;
  $('cancel').disabled = !busy || !controller || controller.signal.aborted;
  $('undo').disabled = busy || !history.length;
  $('clearMarks').disabled = busy || !marks?.some((v) => v !== 0);
  document.querySelectorAll('[data-setting]').forEach((el) => { el.disabled = busy; });
  $('overlay').style.pointerEvents = !busy && source && $('brush').value !== 'inspect' ? 'auto' : 'none';
}
function error(err) {
  console.error(err);
  status(`Hiba: ${err?.message ?? err}.`, 0);
}
function draw(raw, canvas) {
  canvas.width = raw.width; canvas.height = raw.height;
  let rgba = raw.data;
  if (raw.channels === 1) {
    rgba = new Uint8ClampedArray(raw.width * raw.height * 4);
    for (let i = 0; i < raw.data.length; i++) {
      rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = raw.data[i]; rgba[i * 4 + 3] = 255;
    }
  }
  canvas.getContext('2d').putImageData(new ImageData(rgba, raw.width, raw.height), 0, 0);
}
function clear(canvas) { canvas.width = canvas.height = 1; }
function renderFinal() {
  $('outputStage').dataset.background = $('background').value;
  if (finalImage) draw($('background').value === 'mask' ? finalAlpha : finalImage, $('final'));
}
function zoom() {
  for (const el of document.querySelectorAll('.stage-inner')) {
    el.style.width = $('zoom').value === 'fit' || !source ? '100%' : `${source.width * Number($('zoom').value)}px`;
  }
}

$('file').addEventListener('change', async () => {
  if (busy) return;
  const file = $('file').files?.[0]; if (!file) return;
  const version = ++loadVersion;
  source = subjectAlpha = finalImage = finalAlpha = marks = null;
  dirty = false; history = []; stroke = null; lastPoint = null;
  ['original', 'overlay', 'final', 'ben2', 'trimap'].forEach((id) => clear($(id)));
  controls(); status('Kép betöltése…', 0);
  let bitmap;
  try {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 40 * 1024 * 1024) {
      throw new Error('JPG, PNG vagy WebP képet válassz, legfeljebb 40 MB méretben');
    }
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    if (version !== loadVersion) return;
    validateSize(bitmap.width, bitmap.height);
    const c = $('original'); c.width = bitmap.width; c.height = bitmap.height;
    c.getContext('2d').drawImage(bitmap, 0, 0);
    source = { data: c.getContext('2d').getImageData(0, 0, c.width, c.height).data, width: c.width, height: c.height, channels: 4 };
    sourceName = file.name.replace(/\.[^.]+$/, '').replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_') || 'kep';
    marks = new Uint8Array(source.width * source.height);
    $('overlay').width = source.width; $('overlay').height = source.height;
    $('dimensions').textContent = `${source.width} × ${source.height} px`;
    status('A kép kész. Indítsd el a háttéreltávolítást.', 0);
    zoom();
  } catch (err) { if (version === loadVersion) error(err); }
  finally { bitmap?.close(); controls(); }
});

async function hf() {
  if (!hfPromise) hfPromise = import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1').catch((err) => { hfPromise = null; throw err; });
  return hfPromise;
}
function modelProgress(label, start, span) {
  return (p) => {
    if (p?.status === 'progress' && Number.isFinite(p.progress)) status(`${label} letöltése: ${Math.round(p.progress)}%`, start + span * p.progress / 100);
  };
}
async function findSubject(signal) {
  if (subjectAlpha) return subjectAlpha;
  const { pipeline, RawImage } = await hf();
  checkAbort(signal);
  // BEN2 and ViTMatte never occupy GPU model memory together.
  if (matteModel) { await matteModel.dispose(); matteModel = null; }
  status('Az alany felismerése – BEN2 betöltése…', .02);
  let segmenter;
  try {
    segmenter = await pipeline('image-segmentation', 'onnx-community/BEN2-ONNX', {
      device: 'webgpu', dtype: 'fp16', revision: BEN_REV,
      progress_callback: modelProgress('BEN2', .02, .22),
    });
    checkAbort(signal);
    // resize RETURNS a new image; the original must remain untouched.
    let analysis = new RawImage(source.data.slice(), source.width, source.height, 4);
    const scale = Math.min(1, 2048 / Math.max(source.width, source.height));
    if (scale < 1) analysis = await analysis.resize(Math.round(source.width * scale), Math.round(source.height * scale));
    status('Az alany felismerése…', .25);
    const output = await segmenter(analysis);
    checkAbort(signal);
    const result = output?.[0]?.mask;
    if (!result || result.channels !== 1) throw new Error('A BEN2 nem adott értelmezhető maszkot');
    const alpha = resizeMask(mask(result.data, result.width, result.height), source.width, source.height);
    subjectAlpha = alpha;
    draw(alpha, $('ben2'));
    return alpha;
  } finally { if (segmenter) await segmenter.dispose(); }
}
async function prepareMatte(signal) {
  const { AutoProcessor, VitMatteForImageMatting } = await hf();
  checkAbort(signal);
  if (!matteProcessor) matteProcessor = await AutoProcessor.from_pretrained('Xenova/vitmatte-small-composition-1k', { revision: MATTE_REV });
  if (!matteModel) {
    status('Haj- és szőrfinomító modell betöltése…', .32);
    matteModel = await VitMatteForImageMatting.from_pretrained('Xenova/vitmatte-small-composition-1k', {
      revision: MATTE_REV, device: 'webgpu', dtype: 'fp32', progress_callback: modelProgress('ViTMatte', .32, .16),
    });
  }
  checkAbort(signal);
}
async function inferTile(image, trimap) {
  const { RawImage } = await hf();
  let inputs, outputs;
  try {
    inputs = await matteProcessor(new RawImage(image.data, image.width, image.height, image.channels).rgb(), new RawImage(trimap.data, trimap.width, trimap.height, 1));
    outputs = await matteModel(inputs);
    return readAlphaTensor(outputs.alphas);
  } finally {
    const tensors = new Set([inputs?.pixel_values, ...Object.values(outputs ?? {})]);
    for (const tensor of tensors) tensor?.dispose?.();
  }
}
function number(id, fallback, min, max) {
  const raw = $(id).value, v = Number(raw);
  return raw.trim() && Number.isFinite(v) ? clamp(v, min, max) : fallback;
}

$('run').addEventListener('click', async () => {
  if (busy || !source || !navigator.gpu) return;
  busy = true; dirty = true; controller = new AbortController(); controls();
  const signal = controller.signal;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('Nem érhető el használható WebGPU-adapter. Kapcsold be a böngésző hardveres gyorsítását');
    checkAbort(signal);
    const alpha = await findSubject(signal);
    checkAbort(signal);
    const scale = Math.max(source.width, source.height) / 2048;
    const sourceAlpha = new Uint8Array(alpha.data.length);
    for (let i = 0; i < sourceAlpha.length; i++) sourceAlpha[i] = source.data[i * 4 + 3];
    const trimap = buildTrimap(alpha, {
      inner: Math.max(1, Math.round(number('inner', 12, 1, 32) * scale)),
      outer: Math.max(4, Math.round(number('outer', 32, 8, 128) * scale)),
      removeSpecks: $('removeSpecks').checked, marks, sourceAlpha,
    });
    if (!trimap.data.some((v) => v === 255)) throw new Error('Nincs biztos előtér. Jelölj meg zöld ecsettel egy megtartandó belső területet, majd futtasd újra');
    draw(trimap, $('trimap'));
    await prepareMatte(signal);
    status('Haj- és szőrperem finomítása eredeti felbontásban…', .5);
    const matte = await refineAlpha(source, trimap, inferTile, { signal, progress: (done, total) => status(`Finomítás: ${done} / ${total} képrész`, .5 + .43 * done / total) });
    checkAbort(signal);
    status('Háttérszín eltávolítása a félig átlátszó szélekből…', .95);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const result = composeCutout(source, matte, number('decontaminate', 80, 0, 100) / 100);
    checkAbort(signal);
    finalImage = result;
    const exportedAlpha = new Uint8ClampedArray(matte.data.length);
    for (let i = 0; i < exportedAlpha.length; i++) exportedAlpha[i] = result.data[i * 4 + 3];
    finalAlpha = mask(exportedAlpha, source.width, source.height);
    dirty = false; renderFinal();
    status('Kész. Ellenőrizd a peremeket világos és sötét háttéren; a PNG eredeti méretben tölthető le.', 1);
  } catch (err) {
    if (err.name === 'AbortError') status('Megszakítva. A beállításokat módosítva újraindíthatod.', 0);
    else {
      error(err);
      // A failed GPU session must not poison every subsequent retry.
      if (matteModel) { try { await matteModel.dispose(); } catch { /* Already lost device. */ } matteModel = null; }
    }
  } finally { busy = false; controller = null; controls(); }
});
$('cancel').addEventListener('click', () => {
  controller?.abort(); controls(); status('Leállítás a folyamatban lévő modellhívás befejezése után…');
});
$('download').addEventListener('click', async () => {
  if (!finalImage || dirty || busy) return;
  try {
    const downloadName = `${sourceName}-hatter-nelkul.png`;
    const canvas = document.createElement('canvas'); draw(finalImage, canvas);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('A PNG nem hozható létre');
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = downloadName; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (err) { error(err); }
});
$('background').addEventListener('change', renderFinal);
$('zoom').addEventListener('change', zoom);
$('brush').addEventListener('change', controls);
for (const id of ['inner', 'outer', 'removeSpecks', 'decontaminate']) $(id).addEventListener('change', () => {
  if (finalImage) { dirty = true; status('A beállítás változott. Futtasd újra a finomítást.'); controls(); }
});

// Explicit correction marks, only where the user paints. No semantic model can
// silently replace a whole body with a hair-only segment.
const colors = { 1: 'rgba(255,72,95,.65)', 2: 'rgba(67,218,146,.65)', 3: 'rgba(64,166,255,.65)' };
function point(event) {
  const rect = $('overlay').getBoundingClientRect();
  return { x: clamp((event.clientX - rect.left) * source.width / rect.width, 0, source.width - 1), y: clamp((event.clientY - rect.top) * source.height / rect.height, 0, source.height - 1) };
}
function dab(p, value, radius) {
  const x0 = Math.max(0, Math.floor(p.x - radius)), x1 = Math.min(source.width - 1, Math.ceil(p.x + radius));
  const y0 = Math.max(0, Math.floor(p.y - radius)), y1 = Math.min(source.height - 1, Math.ceil(p.y + radius));
  const ctx = $('overlay').getContext('2d');
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    if ((x - p.x) ** 2 + (y - p.y) ** 2 > radius ** 2) continue;
    const i = y * source.width + x;
    if (!stroke.has(i)) stroke.set(i, marks[i]);
    marks[i] = value;
  }
  ctx.globalCompositeOperation = 'destination-out'; ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fill();
  if (value) { ctx.globalCompositeOperation = 'source-over'; ctx.fillStyle = colors[value]; ctx.fill(); }
  ctx.globalCompositeOperation = 'source-over';
}
function paint(p) {
  const value = Number($('brush').value), radius = number('brushSize', 24, 2, 100) / 2;
  const start = lastPoint ?? p, steps = Math.max(1, Math.ceil(Math.hypot(p.x - start.x, p.y - start.y) / Math.max(1, radius / 2)));
  for (let i = 1; i <= steps; i++) dab({ x: start.x + (p.x - start.x) * i / steps, y: start.y + (p.y - start.y) * i / steps }, value, radius);
  lastPoint = p;
}
function redrawMarks() {
  const overlay = new Uint8ClampedArray(marks.length * 4), rgb = { 1: [255,72,95], 2: [67,218,146], 3: [64,166,255] };
  for (let i = 0; i < marks.length; i++) if (marks[i]) { overlay.set(rgb[marks[i]], i * 4); overlay[i * 4 + 3] = 166; }
  draw({ data: overlay, width: source.width, height: source.height, channels: 4 }, $('overlay'));
}
$('overlay').addEventListener('pointerdown', (e) => {
  if (busy || !source || $('brush').value === 'inspect' || e.button !== 0) return;
  e.preventDefault(); $('overlay').setPointerCapture(e.pointerId); stroke = new Map(); lastPoint = null; paint(point(e));
});
$('overlay').addEventListener('pointermove', (e) => { if (stroke && !busy) { e.preventDefault(); paint(point(e)); } });
function endStroke() {
  if (!stroke) return;
  history.push({ indexes: Uint32Array.from(stroke.keys()), values: Uint8Array.from(stroke.values()) });
  // Bound retained undo data to 8 MB; never retain full source images per stroke.
  while (history.length > 1 && history.reduce((n, s) => n + s.indexes.byteLength + s.values.byteLength, 0) > 8_000_000) history.shift();
  stroke = null; lastPoint = null; dirty = true; controls();
  status('Jelölés kész. Futtasd újra a finomítást az alkalmazásához.');
}
$('overlay').addEventListener('pointerup', endStroke);
$('overlay').addEventListener('pointercancel', endStroke);
$('undo').addEventListener('click', () => {
  const last = history.pop(); if (!last || busy) return;
  for (let i = 0; i < last.indexes.length; i++) marks[last.indexes[i]] = last.values[i];
  redrawMarks(); dirty = true; controls(); status('Az utolsó ecsetvonás visszavonva. Futtasd újra a finomítást.');
});
$('clearMarks').addEventListener('click', () => {
  if (busy || !marks) return;
  marks.fill(0); history = []; redrawMarks(); dirty = true; controls(); status('Jelölések törölve. Futtasd újra a finomítást.');
});
controls();
