// Pure pixel operations: browser and regression harness use exactly this code.
export const MAX_PIXELS = 16_000_000;
export const MAX_SIDE = 8192;
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export function validateSize(width, height) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) ||
      width < 1 || height < 1 || width > MAX_SIDE || height > MAX_SIDE ||
      width * height > MAX_PIXELS) {
    throw new Error("Legfeljebb 16 megapixeles, oldalanként 8192 px-es kép használható.");
  }
}
export function mask(data, width, height) {
  validateSize(width, height);
  if (!(data instanceof Uint8Array || data instanceof Uint8ClampedArray) || data.length !== width * height) {
    throw new Error("Hibás maszkméret vagy pixeladat.");
  }
  return { data, width, height, channels: 1 };
}

export function resizeMask(raw, width, height, smooth = true) {
  mask(raw.data, raw.width, raw.height);
  validateSize(width, height);
  const out = new Uint8ClampedArray(width * height);
  for (let y = 0; y < height; y++) {
    // Clamp the coordinate BEFORE deriving both neighbours, including at x/y=0.
    const sy = clamp((y + .5) * raw.height / height - .5, 0, raw.height - 1);
    const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, raw.height - 1), fy = sy - y0;
    for (let x = 0; x < width; x++) {
      const sx = clamp((x + .5) * raw.width / width - .5, 0, raw.width - 1);
      if (!smooth) { out[y * width + x] = raw.data[Math.round(sy) * raw.width + Math.round(sx)]; continue; }
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, raw.width - 1), fx = sx - x0;
      out[y * width + x] = (raw.data[y0 * raw.width + x0] * (1 - fx) + raw.data[y0 * raw.width + x1] * fx) * (1 - fy)
        + (raw.data[y1 * raw.width + x0] * (1 - fx) + raw.data[y1 * raw.width + x1] * fx) * fy;
    }
  }
  return mask(out, width, height);
}

// Retain every substantial subject, not just the largest. Small disconnected
// specks are not new subjects; nearby disconnected hairs remain in the unknown band.
export function subjectSupport(alpha, removeSpecks = true) {
  const { data, width: w, height: h } = mask(alpha.data, alpha.width, alpha.height);
  const n = w * h, support = new Uint8Array(n);
  for (let i = 0; i < n; i++) support[i] = data[i] >= 128 ? 1 : 0;
  if (!removeSpecks) return support;
  const visited = new Uint8Array(n), queue = new Int32Array(n);
  const minArea = Math.max(4, Math.floor(n * .0001));
  for (let seed = 0; seed < n; seed++) {
    if (!support[seed] || visited[seed]) continue;
    let head = 0, tail = 1; queue[0] = seed; visited[seed] = 1;
    while (head < tail) {
      const i = queue[head++], x = i % w, y = Math.floor(i / w);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
        const j = yy * w + xx;
        if (support[j] && !visited[j]) { visited[j] = 1; queue[tail++] = j; }
      }
    }
    if (tail < minArea) for (let j = 0; j < tail; j++) support[queue[j]] = 0;
  }
  return support;
}

export function buildTrimap(alpha, { inner = 12, outer = 32, removeSpecks = true, marks = null, sourceAlpha = null } = {}) {
  const { data, width: w, height: h } = mask(alpha.data, alpha.width, alpha.height);
  if (!Number.isFinite(inner) || !Number.isFinite(outer) || inner < 0 || outer < 1 || inner > 256 || outer > 512) {
    throw new Error("Érvénytelen peremméret.");
  }
  const support = subjectSupport(alpha, removeSpecks);
  const opaque = new Uint8Array(data.length);
  for (let i = 0; i < opaque.length; i++) opaque[i] = support[i] && data[i] >= 250 ? 1 : 0;
  const foreground = erodeBinary(opaque, w, h, Math.round(inner));
  const possible = dilateBinary(support, w, h, Math.round(outer));
  const out = new Uint8ClampedArray(data.length);
  for (let i = 0; i < out.length; i++) out[i] = !possible[i] ? 0 : foreground[i] ? 255 : 128;
  if (marks) {
    if (marks.length !== out.length) throw new Error("A javítójelölések mérete eltér a képtől.");
    for (let i = 0; i < out.length; i++) {
      if (marks[i] === 1) out[i] = 0;
      else if (marks[i] === 2) out[i] = 255;
      else if (marks[i] === 3) out[i] = 128;
    }
  }
  if (sourceAlpha) {
    if (sourceAlpha.length !== out.length) throw new Error("Hibás eredeti alfa.");
    for (let i = 0; i < out.length; i++) if (sourceAlpha[i] === 0) out[i] = 0;
  }
  return mask(out, w, h);
}

export function dilateBinary(src, w, h, radius) {
  // Kétmenetes csúszóablakos maximum: O(width*height), nem O(r²*pixel).
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);

  for (let y = 0; y < h; y++) {
    const deque = [];
    let head = 0;
    for (let x = 0; x < w; x++) {
      while (head < deque.length && deque[head] < x - radius) head++;
      const add = Math.min(w - 1, x + radius);
      if (x === 0) {
        for (let k = 0; k <= add; k++) {
          while (deque.length > head && src[y*w + deque[deque.length-1]] <= src[y*w + k]) deque.pop();
          deque.push(k);
        }
      } else {
        while (deque.length > head && src[y*w + deque[deque.length-1]] <= src[y*w + add]) deque.pop();
        deque.push(add);
      }
      tmp[y*w + x] = src[y*w + deque[head]];
    }
  }

  for (let x = 0; x < w; x++) {
    const deque = [];
    let head = 0;
    for (let y = 0; y < h; y++) {
      while (head < deque.length && deque[head] < y - radius) head++;
      const add = Math.min(h - 1, y + radius);
      if (y === 0) {
        for (let k = 0; k <= add; k++) {
          while (deque.length > head && tmp[deque[deque.length-1]*w + x] <= tmp[k*w + x]) deque.pop();
          deque.push(k);
        }
      } else {
        while (deque.length > head && tmp[deque[deque.length-1]*w + x] <= tmp[add*w + x]) deque.pop();
        deque.push(add);
      }
      out[y*w + x] = tmp[deque[head]*w + x];
    }
  }
  return out;
}

export function erodeBinary(src, w, h, radius) {
  if (radius <= 0) return new Uint8Array(src);
  const inv = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) inv[i] = src[i] ? 0 : 1;
  const expandedBg = dilateBinary(inv, w, h, radius);
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = expandedBg[i] ? 0 : 1;
  return out;
}

