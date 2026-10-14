import { clamp, mask, validateSize } from './matting-core.mjs?v=5';

export const TILE = 512;
export const OVERLAP = 128;

export function tileStarts(length, tile = TILE, stride = TILE - OVERLAP) {
  if (!Number.isInteger(length) || length < 1 || !Number.isInteger(tile) || tile < 1 ||
      !Number.isInteger(stride) || stride < 1 || stride > tile) throw new Error('Hibás csempeméret.');
  if (length <= tile) return [0];
  const starts = [];
  for (let s = 0; s < length - tile; s += stride) starts.push(s);
  if (starts.at(-1) !== length - tile) starts.push(length - tile);
  return starts;
}

export function cropPad(raw, x, y, size = TILE) {
  const out = new Uint8ClampedArray(size * size * raw.channels);
  for (let ty = 0; ty < size; ty++) for (let tx = 0; tx < size; tx++) {
    const si = (Math.min(y + ty, raw.height - 1) * raw.width + Math.min(x + tx, raw.width - 1)) * raw.channels;
    const di = (ty * size + tx) * raw.channels;
    for (let c = 0; c < raw.channels; c++) out[di + c] = raw.data[si + c];
  }
  return { data: out, width: size, height: size, channels: raw.channels };
}

// No silent zeros, NaNs or squeezed wrong output shapes in an exported image.
export function readAlphaTensor(tensor, width = TILE, height = TILE) {
  const d = tensor?.dims, src = tensor?.data;
  if (!Array.isArray(d) || d.length !== 4 || d[0] !== 1 || d[1] !== 1 ||
      !Number.isSafeInteger(d[2]) || !Number.isSafeInteger(d[3]) ||
      d[2] < height || d[3] < width || d[2] > TILE || d[3] > TILE || src?.length !== d[2] * d[3]) {
    throw new Error('A ViTMatte váratlan méretű alfamaszkot adott vissza.');
  }
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const v = src[y * d[3] + x];
    if (!Number.isFinite(v)) throw new Error('A ViTMatte érvénytelen pixelértéket adott vissza.');
    out[y * width + x] = clamp(v, 0, 1);
  }
  return out;
}

function weight(pos, start, length) {
  const smooth = (v) => v * v * (3 - 2 * v);
  let value = 1;
  if (start > 0 && pos < OVERLAP) value *= smooth((pos + 1) / (OVERLAP + 1));
  if (start + TILE < length && pos >= TILE - OVERLAP) value *= smooth((TILE - pos) / (OVERLAP + 1));
  return value;
}

export function checkAbort(signal) {
  if (signal?.aborted) throw new DOMException('A feldolgozás megszakítva.', 'AbortError');
}

// The injectable inference callback is also used by the real ONNX CPU verification.
// Peak GPU input stays 512 x 512; accumulation stays float until the final PNG alpha.
export async function refineAlpha(image, trimap, infer, { signal, progress } = {}) {
  const { width: w, height: h } = image;
  validateSize(w, h);
  mask(trimap.data, trimap.width, trimap.height);
  if (trimap.width !== w || trimap.height !== h || ![3, 4].includes(image.channels) || image.data.length !== w * h * image.channels) {
    throw new Error('A kép és a trimap mérete vagy csatornaszáma eltér.');
  }
  const tiles = [];
  for (const y of tileStarts(h)) for (const x of tileStarts(w)) {
    let unknown = false;
    for (let yy = y; yy < Math.min(h, y + TILE) && !unknown; yy++) {
      for (let xx = x; xx < Math.min(w, x + TILE); xx++) {
        if (trimap.data[yy * w + xx] === 128) { unknown = true; break; }
      }
    }
    if (unknown) tiles.push([x, y]);
  }
  const accum = new Float32Array(w * h), weights = new Float32Array(w * h);
  let done = 0;
  for (const [x, y] of tiles) {
    checkAbort(signal);
    const prediction = await infer(cropPad(image, x, y), cropPad(trimap, x, y));
    checkAbort(signal);
    if (!(prediction instanceof Float32Array) || prediction.length !== TILE * TILE) throw new Error('Hibás csempeeredmény.');
    const aw = Math.min(TILE, w - x), ah = Math.min(TILE, h - y);
    for (let ty = 0; ty < ah; ty++) for (let tx = 0; tx < aw; tx++) {
      const i = (y + ty) * w + x + tx;
      if (trimap.data[i] !== 128) continue;
      const v = prediction[ty * TILE + tx];
      if (!Number.isFinite(v) || v < 0 || v > 1) throw new Error('Érvénytelen csempe-alfa.');
      const ww = Math.max(.001, weight(tx, x, w) * weight(ty, y, h));
      accum[i] += v * ww; weights[i] += ww;
    }
    progress?.(++done, tiles.length);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  checkAbort(signal);
  const out = new Uint8ClampedArray(w * h);
  for (let i = 0; i < out.length; i++) {
    if (trimap.data[i] === 128) {
      if (weights[i] <= 0) throw new Error('Egy bizonytalan képrész feldolgozása elmaradt.');
      out[i] = clamp(accum[i] / weights[i], 0, 1) * 255;
    } else out[i] = trimap.data[i];
  }
  return mask(out, w, h);
}
