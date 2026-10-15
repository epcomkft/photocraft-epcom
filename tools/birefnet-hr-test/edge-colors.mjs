import { clamp, mask } from './core.mjs';
import { estimateForeground } from './foreground.mjs';

function nearestSeeds(w, h, radius, seed) {
  const n = w * h, index = new Int32Array(n), distance = new Uint16Array(n);
  index.fill(-1); distance.fill(radius + 1);
  for (let i = 0; i < n; i++) if (seed(i)) { index[i] = i; distance[i] = 0; }
  const take = (i, j) => { const d = distance[j] + 1; if (d < distance[i] && d <= radius) { distance[i] = d; index[i] = index[j]; } };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = y * w + x; if (x) take(i, i - 1); if (y) take(i, i - w); }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) { const i = y * w + x; if (x + 1 < w) take(i, i + 1); if (y + 1 < h) take(i, i + w); }
  return { index, distance };
}

// A high model alpha can still contain background colour. Use a small patch
// safely inside the foreground, and correct only colours that fit its local
// foreground/background mixture. This changes RGB, never the matte boundary.
function removeResidualSpill(rgb, source, a, w, h) {
  const radius = clamp(Math.round(Math.max(w, h) / 32), 12, 96);
  const inset = clamp(Math.round(Math.max(w, h) / 400), 2, 8);
  const boundary = nearestSeeds(w, h, inset, i => a[i] < 254);
  const fg = nearestSeeds(w, h, radius, i => a[i] >= 254 && boundary.distance[i] >= inset).index;
  const bg = nearestSeeds(w, h, radius, i => a[i] <= 1).index;
  const f = new Float64Array(3), b = new Float64Array(3);
  const patch = (at, isForeground, out) => {
    out.fill(0); let count = 0;
    const x0 = at % w, y0 = Math.floor(at / w);
    for (let y = Math.max(0, y0 - 1); y <= Math.min(h - 1, y0 + 1); y++) {
      for (let x = Math.max(0, x0 - 1); x <= Math.min(w - 1, x0 + 1); x++) {
        const i = y * w + x;
        if (isForeground ? a[i] < 254 : a[i] > 1) continue;
        for (let c = 0; c < 3; c++) out[c] += source[i * 4 + c];
        count++;
      }
    }
    for (let c = 0; c < 3; c++) out[c] /= Math.max(count, 1);
  };
  for (let i = 0; i < a.length; i++) {
    if (!a[i] || a[i] === 255 || fg[i] < 0 || bg[i] < 0) continue;
    patch(fg[i], true, f); patch(bg[i], false, b);
    let contrast2 = 0, projection = 0;
    for (let c = 0; c < 3; c++) { const d = f[c] - b[c]; contrast2 += d * d; projection += (source[i * 4 + c] - b[c]) * d; }
    const contrast = Math.sqrt(contrast2);
    if (contrast < 24) continue;
    const coverage = clamp(projection / contrast2, 0, 1);
    let residual2 = 0;
    for (let c = 0; c < 3; c++) residual2 += (source[i * 4 + c] - (coverage * f[c] + (1 - coverage) * b[c])) ** 2;
    const fit = clamp(1 - Math.sqrt(residual2) / (contrast * .25 + 4), 0, 1);
    const confidence = fit * clamp((a[i] / 255 - coverage - .05) / .2, 0, 1) * clamp((contrast - 24) / 48, 0, 1);
    for (let c = 0; c < 3; c++) rgb[i * 3 + c] += confidence * (f[c] - rgb[i * 3 + c]);
  }
  return rgb;
}

export function createEdgeCleaner(source, matte) {
  const { width: w, height: h, data: a } = mask(matte.data, matte.width, matte.height);
  if (source.width !== w || source.height !== h || source.channels !== 4 || !(source.data instanceof Uint8ClampedArray || source.data instanceof Uint8Array) || source.data.length !== w * h * 4) throw new Error('A színkép és az alfamaszk mérete eltér.');
  let prepared = false, colors = null;
  const info = { method: 'multilevel-local-spill-v4', status: 'not-needed', preparations: 0, milliseconds: 0 };
  return {
    info,
    compose(strength = 0) {
      if (!Number.isFinite(strength) || strength < 0 || strength > 1) throw new Error('Érvénytelen szegélyszín-korrekció.');
      if (strength > 0 && !prepared) {
        const start = performance.now();
        let opaque = true, partial = false;
        for (let i = 0; i < a.length; i++) { if (source.data[i * 4 + 3] !== 255) opaque = false; if (a[i] > 0 && a[i] < 255) partial = true; }
        // Existing transparent PNG colours are already foreground colours; its
        // transparent RGB cannot be treated as a photographed background.
        if (!opaque) info.status = 'transparent-source';
        else if (!partial) info.status = 'no-soft-edge';
        else {
          colors = estimateForeground(source.data, a, w, h);
          if (colors) removeResidualSpill(colors, source.data, a, w, h);
          info.status = colors ? 'ready' : 'insufficient-context';
        }
        info.milliseconds = performance.now() - start; info.preparations++; prepared = true;
      }
      const out = new Uint8ClampedArray(source.data);
      for (let i = 0; i < a.length; i++) {
        const p = i * 4;
        out[p + 3] = a[i] / 255 * source.data[p + 3];
        if (!out[p + 3]) { out[p] = out[p + 1] = out[p + 2] = 0; continue; }
        // Opaque interiors are byte-identical, independent of strength.
        if (!colors || !strength || a[i] === 255) continue;
        for (let c = 0; c < 3; c++) out[p + c] = source.data[p + c] + strength * (colors[i * 3 + c] - source.data[p + c]);
      }
      return { data: out, width: w, height: h, channels: 4 };
    }
  };
}

export function composeCutout(source, matte, strength = 0) {
  return createEdgeCleaner(source, matte).compose(strength);
}
