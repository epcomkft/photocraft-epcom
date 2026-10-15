// Shared pure operations. Soft alpha is never thresholded into a silhouette.
export const MODEL_SIZE = 2048;
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export function validateSize(width, height) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1 || width > 8192 || height > 8192 || width * height > 16_000_000) {
    throw new Error('Legfeljebb 16 megapixeles, oldalanként 8192 pixeles kép használható.');
  }
}
export function mask(data, width, height) {
  validateSize(width, height);
  if (!(data instanceof Uint8Array || data instanceof Uint8ClampedArray) || data.length !== width * height) throw new Error('Hibás alfamaszk.');
  return { data, width, height, channels: 1 };
}
export function normalizeRgb(rgba, width, height) {
  validateSize(width, height);
  if (rgba.length !== width * height * 4) throw new Error('Hibás bemeneti képadat.');
  const n = width * height, out = new Float32Array(n * 3);
  const mean = [.485, .456, .406], std = [.229, .224, .225];
  for (let i = 0; i < n; i++) {
    const alpha = rgba[i * 4 + 3] / 255;
    for (let c = 0; c < 3; c++) out[c * n + i] = ((rgba[i * 4 + c] / 255 * alpha + 1 - alpha) - mean[c]) / std[c];
  }
  return out;
}
export function readModelAlpha(tensor) {
  if (tensor?.type !== 'float32' || tensor.dims?.join(',') !== '1,1,2048,2048' || !(tensor.data instanceof Float32Array) || tensor.data.length !== MODEL_SIZE * MODEL_SIZE) {
    throw new Error('A HR modell kimenetének típusa vagy mérete hibás.');
  }
  // The pinned export ALREADY includes Sigmoid. Applying it twice destroys alpha.
  // Invalid output must never be changed to transparent/opaque pixels to hide a
  // failed GPU inference. Collect the entire failure, then reject atomically.
  const stats = { total: tensor.data.length, nan: 0, infinite: 0, outOfRange: 0, invalid: 0, min: null, max: null, firstInvalidIndex: null };
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < tensor.data.length; i++) {
    const v = tensor.data[i]; let bad = false;
    if (Number.isNaN(v)) { stats.nan++; bad = true; }
    else if (!Number.isFinite(v)) { stats.infinite++; bad = true; }
    else { min = Math.min(min, v); max = Math.max(max, v); if (v < -1e-6 || v > 1 + 1e-6) { stats.outOfRange++; bad = true; } }
    if (bad) { stats.invalid++; if (stats.firstInvalidIndex === null) stats.firstInvalidIndex = i; }
  }
  stats.min = Number.isFinite(min) ? min : null; stats.max = Number.isFinite(max) ? max : null;
  if (stats.invalid) {
    const error = new Error(`A GPU számítása érvénytelen alfamaszkot adott (${stats.invalid} / ${stats.total} hibás képpont). A hibajelentés a „Peremtisztítás és ellenőrzés” részből letölthető.`);
    error.code = 'MODEL_ALPHA_INVALID'; error.alphaStats = stats; throw error;
  }
  const out = new Float32Array(tensor.data.length);
  for (let i = 0; i < out.length; i++) out[i] = clamp(tensor.data[i], 0, 1);
  return out;
}
export function resizeAlpha(alpha, sw, sh, width, height) {
  validateSize(sw, sh); validateSize(width, height);
  if (!(alpha instanceof Float32Array) || alpha.length !== sw * sh) throw new Error('Hibás modellmaszk.');
  const out = new Uint8ClampedArray(width * height);
  for (let y = 0; y < height; y++) {
    const sy = clamp((y + .5) * sh / height - .5, 0, sh - 1), y0 = Math.floor(sy), y1 = Math.min(y0 + 1, sh - 1), fy = sy - y0;
    for (let x = 0; x < width; x++) {
      const sx = clamp((x + .5) * sw / width - .5, 0, sw - 1), x0 = Math.floor(sx), x1 = Math.min(x0 + 1, sw - 1), fx = sx - x0;
      out[y * width + x] = 255 * ((alpha[y0 * sw + x0] * (1 - fx) + alpha[y0 * sw + x1] * fx) * (1 - fy) + (alpha[y1 * sw + x0] * (1 - fx) + alpha[y1 * sw + x1] * fx) * fy);
    }
  }
  return mask(out, width, height);
}
export function exportAlpha(rgba) {
  if (rgba.length % 4) throw new Error('Hibás eredmény.');
  const out = new Uint8ClampedArray(rgba.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4 + 3];
  return out;
}
