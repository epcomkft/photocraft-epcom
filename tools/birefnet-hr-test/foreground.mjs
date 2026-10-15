// Fast Multi-Level Foreground Estimation, Germer et al. (2020).
// Adapted from PyMatting (c) 2020 PyMatting, MIT; see licenses/PyMatting-MIT.txt.
// Reference source blob: c117f291779fcf2c8b01997dbc84c2a0c5f5b699.
// A deterministic serial solve in a Web Worker. This estimates colours only:
// alpha is never thresholded, eroded, blurred or regenerated.

import { validateSize } from './core.mjs';

export function estimateForeground(source, alpha, width, height) {
  validateSize(width, height);
  if (!(source instanceof Uint8ClampedArray || source instanceof Uint8Array) || source.length !== width * height * 4 || !(alpha instanceof Uint8ClampedArray || alpha instanceof Uint8Array) || alpha.length !== width * height) throw new Error('Hibás szín- vagy alfamaszkadat.');
  const count = width * height;
  let fc = 0, bc = 0;
  let previousF = new Float32Array(3), previousB = new Float32Array(3);
  for (let i = 0; i < count; i++) {
    if (alpha[i] > 229) { fc++; for (let c = 0; c < 3; c++) previousF[c] += source[i * 4 + c] / 255; }
    if (alpha[i] < 26) { bc++; for (let c = 0; c < 3; c++) previousB[c] += source[i * 4 + c] / 255; }
  }
  // Without both classes no colour separation is identifiable. Keep original RGB.
  if (!fc || !bc) return null;
  for (let c = 0; c < 3; c++) { previousF[c] /= fc; previousB[c] /= bc; }
  let oldW = 1, oldH = 1;
  const levels = Math.max(1, Math.ceil(Math.log2(Math.max(width, height))));
  const neighbours = new Int32Array(4), weights = new Float64Array(4);
  for (let level = 0; level <= levels; level++) {
    const w = Math.round(width ** (level / levels)), h = Math.round(height ** (level / levels));
    const n = w * h, f = new Float32Array(n * 3), b = new Float32Array(n * 3);
    const a = new Float32Array(n), indices = new Uint32Array(n);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, original = Math.floor(y * height / h) * width + Math.floor(x * width / w);
      indices[i] = original * 4; a[i] = alpha[original] / 255;
      const old = (Math.floor(y * oldH / h) * oldW + Math.floor(x * oldW / w)) * 3;
      for (let c = 0; c < 3; c++) { f[i * 3 + c] = previousF[old + c]; b[i * 3 + c] = previousB[old + c]; }
    }
    // Only two colour fields for this level and its predecessor are live. Clear
    // the predecessor references before the expensive full-resolution passes.
    previousF = previousB = null;
    const iterations = w <= 32 && h <= 32 ? 10 : 2;
    for (let iter = 0; iter < iterations; iter++) {
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x, af = a[i], ab = 1 - af;
        neighbours[0] = x ? i - 1 : i; neighbours[1] = x + 1 < w ? i + 1 : i;
        neighbours[2] = y ? i - w : i; neighbours[3] = y + 1 < h ? i + w : i;
        let regularizer = 0;
        for (let k = 0; k < 4; k++) { const weight = 1e-5 + Math.abs(af - a[neighbours[k]]); weights[k] = weight; regularizer += weight; }
        const ff = af * af + regularizer, bb = ab * ab + regularizer, fb = af * ab;
        const determinant = ff * bb - fb * fb;
        if (!(determinant > 0) || !Number.isFinite(determinant)) throw new Error('A szegélyszín-becslés nem adott érvényes eredményt.');
        for (let c = 0; c < 3; c++) {
          const observed = source[indices[i] + c] / 255;
          let rhsF = af * observed, rhsB = ab * observed;
          for (let k = 0; k < 4; k++) { const q = neighbours[k] * 3 + c; rhsF += weights[k] * f[q]; rhsB += weights[k] * b[q]; }
          f[i * 3 + c] = Math.max(0, Math.min(1, (bb * rhsF - fb * rhsB) / determinant));
          b[i * 3 + c] = Math.max(0, Math.min(1, (ff * rhsB - fb * rhsF) / determinant));
        }
      }
    }
    previousF = f; previousB = b; oldW = w; oldH = h;
  }
  const result = new Uint8ClampedArray(count * 3);
  for (let i = 0; i < result.length; i++) result[i] = previousF[i] * 255;
  return result;
}
