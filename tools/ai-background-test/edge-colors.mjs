import { clamp, mask } from './matting-core.mjs?v=5';

// Bounded, two-pass Manhattan nearest-seed field. No full-image colour blur:
// unrelated subject/background pixels cannot bleed into opaque areas.
function nearest(alpha, w, h, foreground, radius) {
  const n = w * h, index = new Int32Array(n), distance = new Uint16Array(n);
  index.fill(-1); distance.fill(radius + 1);
  for (let i = 0; i < n; i++) {
    if (foreground ? alpha[i] >= 250 : alpha[i] <= 1) { index[i] = i; distance[i] = 0; }
  }
  const take = (i, j) => {
    const d = distance[j] + 1;
    if (d < distance[i] && d <= radius) { distance[i] = d; index[i] = index[j]; }
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (x) take(i, i - 1);
    if (y) take(i, i - w);
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x;
    if (x + 1 < w) take(i, i + 1);
    if (y + 1 < h) take(i, i + w);
  }
  return index;
}

// C = alpha * F + (1-alpha) * B. Estimate local B and regularise F towards a
// nearby opaque foreground colour; never divide tiny alphas without a prior.
// Only partially covered boundary pixels change. PNG RGB is straight, not premultiplied.
export function composeCutout(source, matte, strength = .8) {
  const { width: w, height: h, data: a } = mask(matte.data, matte.width, matte.height);
  if (source.width !== w || source.height !== h || source.channels !== 4 || source.data.length !== w * h * 4) {
    throw new Error('A színkép és az alfamaszk mérete eltér.');
  }
  if (!Number.isFinite(strength) || strength < 0 || strength > 1) throw new Error('Érvénytelen szegélyszín-korrekció.');
  const out = new Uint8ClampedArray(source.data);
  // Transparent input has no reliable background RGB. Preserve its colour/alpha.
  let opaqueSource = true;
  for (let i = 0; i < a.length; i++) if (source.data[i * 4 + 3] !== 255) { opaqueSource = false; break; }
  const radius = clamp(Math.round(Math.max(w, h) / 64), 8, 64);
  const bg = strength && opaqueSource ? nearest(a, w, h, false, radius) : null;
  const fg = bg ? nearest(a, w, h, true, radius) : null;
  for (let i = 0; i < a.length; i++) {
    const p = i * 4, alpha = a[i] / 255;
    out[p + 3] = alpha * source.data[p + 3];
    if (!out[p + 3]) { out[p] = out[p + 1] = out[p + 2] = 0; continue; }
    if (!bg || alpha >= .98 || bg[i] < 0 || fg[i] < 0) continue;
    const lambda = .02 * (1 - alpha) ** 2 + .001;
    for (let c = 0; c < 3; c++) {
      const observed = source.data[p + c], background = source.data[bg[i] * 4 + c];
      const prior = source.data[fg[i] * 4 + c];
      const estimate = clamp((alpha * (observed - (1 - alpha) * background) + lambda * prior) / (alpha * alpha + lambda), 0, 255);
      out[p + c] = observed + strength * (estimate - observed);
    }
  }
  return { data: out, width: w, height: h, channels: 4 };
}
