import { MODEL_SIZE, validateSize, normalizeRgb, readModelAlpha, resizeAlpha, exportAlpha } from './core.mjs';
import { createEdgeCleaner } from './edge-colors.mjs';
import { adaptPinnedModel } from './onnx-compat.mjs';
import { gpuLimits, classifyGpuError } from './gpu-errors.mjs';

const MODEL_REV = '792518b9d4dfe9793891712677de25e554f948c6';
// Use the publisher's actual FP32 reference, not FP16 weights cast back to FP32.
const MODEL_URL = `https://huggingface.co/PinkPixel/birefnet-hr-matting-onnx/resolve/${MODEL_REV}/model.onnx`;
const MODEL_BYTES = 932150975;
const MODEL_HASH = 'a548a1f3307250766bef50b174026b3cfa39a695bc1e42ec131c14fe82797ffd';
const CACHE_NAME = 'photocraft-birefnet-hr-fp32-v2';
const RUNTIME_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.webgpu.min.mjs';
let source = null, matte = null, running = false, diagnostic = null, cleaner = null;
function progress(id, message, value) { if (diagnostic) diagnostic.stage = message; self.postMessage({ type: 'progress', id, message, value }); }
function report(id) { self.postMessage({ type: 'diagnostics', id, report: diagnostic }); }
async function modelBytes(id) {
  let cache = null, response = null;
  // The v1 FP16 download cannot be reused for FP32. Free its browser storage.
  try { await caches.delete('photocraft-birefnet-hr-v1'); } catch { /* Optional cache. */ }
  try { cache = await caches.open(CACHE_NAME); response = await cache.match(MODEL_URL); } catch { /* Cache is optional (private mode / quota). */ }
  const cached = !!response;
  if (!response) response = await fetch(MODEL_URL, { credentials: 'omit', referrerPolicy: 'no-referrer', signal: AbortSignal.timeout(600_000) });
  if (!response.ok || !response.body) throw new Error('A modell nem tölthető le. Ellenőrizd az internetkapcsolatot és a huggingface.co elérhetőségét.');
  const reader = response.body.getReader(), data = new Uint8Array(MODEL_BYTES);
  let offset = 0, last = 0;
  try {
    while (true) {
      let timer;
      const next = await Promise.race([reader.read(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('A modell letöltése megállt. Próbáld újra.')), 90_000); })]).finally(() => clearTimeout(timer));
      if (next.done) break;
      if (offset + next.value.length > data.length) throw new Error('A letöltött modell mérete eltér a várt változattól.');
      data.set(next.value, offset); offset += next.value.length;
      if (performance.now() - last > 150) {
        progress(id, `${cached ? 'FP32 modell a gyorsítótárból' : 'FP32 modell letöltése'}: ${Math.round(offset / 1e6)} / ${Math.round(MODEL_BYTES / 1e6)} MB`, .04 + .32 * offset / data.length); last = performance.now();
      }
    }
  } catch (error) { await reader.cancel().catch(() => {}); throw error; }
  if (offset !== data.length) { await cache?.delete(MODEL_URL); throw new Error('A modellletöltés hiányos. Indítsd újra.'); }
  progress(id, 'A modell épségének ellenőrzése…', .37);
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', data))].map(b => b.toString(16).padStart(2, '0')).join('');
  if (digest !== MODEL_HASH) { await cache?.delete(MODEL_URL); throw new Error('A modell ellenőrzőösszege hibás. Indítsd újra a letöltést.'); }
  if (!cached && cache) {
    try { await cache.put(MODEL_URL, new Response(data, { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(data.length) } })); }
    catch { self.postMessage({ type: 'notice', id, message: 'A modell nem fért a gyorsítótárba. Ez a feldolgozást nem akadályozza; legközelebb újra letöltjük.' }); }
  }
  return data;
}
function compose(id, strength, includeRaw) {
  if (!source || !matte) throw new Error('Előbb futtasd le a háttéreltávolítást.');
  if (!Number.isFinite(strength) || strength < 0 || strength > 1) throw new Error('Hibás peremtisztítási erősség.');
  if (!cleaner) cleaner = createEdgeCleaner(source, matte);
  const result = cleaner.compose(strength);
  if (diagnostic) { diagnostic.edgeCleanup = { ...cleaner.info, strength }; report(id); }
  const alpha = exportAlpha(result.data);
  const payload = { type: 'result', id, width: source.width, height: source.height, rgba: result.data.buffer, alpha: alpha.buffer, strength, cleanup: { ...cleaner.info } };
  const transfers = [payload.rgba, payload.alpha];
  if (includeRaw) { payload.raw = cleaner.compose(0).data.buffer; transfers.push(payload.raw); }
  self.postMessage(payload, transfers);
}
async function run(message) {
  const { id, width, height, rgba } = message;
  diagnostic = { appVersion: '4.0.0', model: 'BiRefNet_HR-matting', revision: MODEL_REV, file: 'model.onnx', sha256: MODEL_HASH, precision: 'FP32', runtime: 'onnxruntime-web 1.23.2', backend: 'WebGPU', graphOptimization: 'basic', memoryOptimization: 'project-before-resize-v3', inputSize: [MODEL_SIZE, MODEL_SIZE], sourceSize: [width, height] };
  report(id);
  validateSize(width, height);
  if (!(rgba instanceof ArrayBuffer) || rgba.byteLength !== width * height * 4) throw new Error('Hibás képadat érkezett.');
  source = { data: new Uint8ClampedArray(rgba), width, height, channels: 4 }; matte = null; cleaner = null;
  progress(id, 'Videokártya ellenőrzése…', .01);
  const adapter = await self.navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('Nem érhető el WebGPU. Asztali Chrome, Edge vagy Opera böngészőben, HTTPS alatt, bekapcsolt hardveres gyorsítással próbáld.');
  // FP32 does not require shader-f16; unsupported FP16 was not the only source
  // of invalid masks. Half-precision arithmetic can also overflow on some GPUs.
  const info = adapter.info;
  diagnostic.gpu = info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description } : null;
  diagnostic.shaderF16Available = adapter.features.has('shader-f16'); diagnostic.adapterLimits = gpuLimits(adapter.limits); report(id);
  const ort = await import(RUNTIME_URL);
  ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
  ort.env.webgpu.adapter = adapter; ort.env.webgpu.powerPreference = 'high-performance';
  ort.env.logLevel = 'error';
  let bytes = await modelBytes(id);
  progress(id, 'A HR modell memóriaigényének csökkentése…', .40);
  let compatible = adaptPinnedModel(bytes); bytes = null;
  let session, input, outputs;
  try {
    progress(id, 'HR modell betöltése a videokártyára…', .46);
    // Avoid extended/all fusions as well as FP16 arithmetic. Keep basic constant
    // folding; never silently replace NaN pixels or lower the model resolution.
    session = await ort.InferenceSession.create(compatible, { executionProviders: ['webgpu'], graphOptimizationLevel: 'basic', enableCpuMemArena: false, enableMemPattern: false });
    compatible = null;
    diagnostic.deviceLimits = gpuLimits(ort.env.webgpu.device?.limits); report(id);
    if (session.inputNames.join(',') !== 'image' || session.outputNames.join(',') !== 'alpha') throw new Error('Eltérő modellbemenet vagy -kimenet.');
    const original = new OffscreenCanvas(width, height), resized = new OffscreenCanvas(MODEL_SIZE, MODEL_SIZE);
    original.getContext('2d').putImageData(new ImageData(source.data, width, height), 0, 0);
    const ctx = resized.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    // Matches upstream's square Resize(2048,2048); output restored to original size.
    ctx.drawImage(original, 0, 0, MODEL_SIZE, MODEL_SIZE);
    input = new ort.Tensor('float32', normalizeRgb(ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data, MODEL_SIZE, MODEL_SIZE), [1, 3, MODEL_SIZE, MODEL_SIZE]);
    original.width = original.height = resized.width = resized.height = 1;
    progress(id, 'BiRefNet HR-matting dolgozik · FP32 · 2048 × 2048 px…', .60);
    const start = performance.now(); outputs = await session.run({ image: input });
    const seconds = (performance.now() - start) / 1000;
    diagnostic.seconds = seconds;
    diagnostic.output = { type: outputs.alpha?.type, dims: outputs.alpha?.dims, length: outputs.alpha?.data?.length };
    const floats = readModelAlpha(outputs.alpha);
    progress(id, 'Eredeti képméret és átlátszóság visszaállítása…', .92);
    matte = resizeAlpha(floats, MODEL_SIZE, MODEL_SIZE, width, height);
    diagnostic.result = 'valid'; report(id);
    self.postMessage({ type: 'metrics', id, seconds, model: 'BiRefNet_HR-matting', revision: MODEL_REV, size: MODEL_SIZE, precision: 'FP32', backend: 'WebGPU' });
  } finally {
    input?.dispose(); for (const output of Object.values(outputs ?? {})) output?.dispose();
    await session?.release();
  }
  progress(id, 'Átlátszó PNG előnézet készítése…', .97);
  compose(id, message.strength, true);
}
self.addEventListener('message', async ({ data }) => {
  if (running || !['run', 'compose'].includes(data?.type)) return;
  running = true;
  try { if (data.type === 'run') await run(data); else compose(data.id, data.strength, false); }
  catch (error) {
    const failure = classifyGpuError(error);
    if (diagnostic) { diagnostic.result = 'error'; diagnostic.error = { code: failure.code, message: failure.detail, buffer: failure.buffer ?? null, alphaStats: error?.alphaStats ?? null }; report(data.id); }
    self.postMessage({ type: 'error', id: data.id, message: failure.message, detail: failure.detail });
  } finally { running = false; }
});
self.postMessage({ type: 'ready', version: '4.0.0' });
