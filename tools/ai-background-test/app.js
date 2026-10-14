import {
  pipeline,
  AutoProcessor,
  VitMatteForImageMatting,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { sam2Subject } from "./sam2.js";

const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const runButton = $("run");
const downloadButton = $("download");
const statusEl = $("status");
const progressEl = $("progress");
const gpuEl = $("gpu");

let sourceImage = null;
let finalImage = null;
let ben2Pipe = null;
let matteProcessor = null;
let matteModel = null;

const VITMATTE_TILE = 512;
const VITMATTE_OVERLAP = 96;

gpuEl.textContent = navigator.gpu
  ? "WebGPU: elérhető – GPU gyorsítás használható. ViTMatte 512 px-es csempékkel fut a GPU-memória védelmében."
  : "WebGPU: nem elérhető. Ehhez a teszthez Chrome/Edge + WebGPU szükséges.";

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  try {
    setStatus("Kép betöltése…", 0);
    sourceImage = await RawImage.fromBlob(file);
    sourceImage = await limitImage(sourceImage, readNumber("maxSide", 2048));
    drawRaw(sourceImage, $("original"));
    clearCanvas($("ben2"));
    clearCanvas($("sam2"));
    clearCanvas($("final"));
    clearCanvas($("trimap"));
    finalImage = null;
    downloadButton.disabled = true;
    runButton.disabled = !navigator.gpu;
    setStatus(
      `Kép kész: ${sourceImage.width} × ${sourceImage.height} px. Indítható az AI.`,
      0
    );
  } catch (err) {
    fail(err);
  }
});

runButton.addEventListener("click", async () => {
  if (!sourceImage) return;
  if (!navigator.gpu) {
    setStatus("WebGPU nem érhető el ebben a böngészőben.", 0);
    return;
  }

  runButton.disabled = true;
  downloadButton.disabled = true;
  finalImage = null;

  try {
    await ensureModels();

    setStatus("BEN2: alany felismerése és soft alpha maszk készítése…", 0.55);
    const benResult = await ben2Pipe(sourceImage);
    const benImage = Array.isArray(benResult) ? benResult[0] : benResult;
    drawRaw(benImage, $("ben2"));

    const alpha = alphaFromBen2(benImage);

    const seed = pickSubjectSeed(alpha);
    setStatus(`SAM2: fő alany kiválasztása (${Math.round(seed.x)}, ${Math.round(seed.y)})…`, 0.57);
    const sam = await sam2Subject(sourceImage, seed, (message, p) => setStatus(message, p));
    const samRaw = new RawImage(sam.data, sam.width, sam.height, 1);
    drawRaw(samRaw, $("sam2"));

    const inner = Math.max(0, Math.round(readNumber("samInner", 8)));
    const outer = Math.max(inner + 1, Math.round(readNumber("samOuter", 36)));

    setStatus("Trimap készítése: SAM2 szemantika + BEN2 finom részletek…", 0.69);
    const trimap = makeSamTrimap(alpha, samRaw, inner, outer);
    drawRaw(trimap, $("trimap"));

    setStatus("ViTMatte: haj/szőr perem finomítása 512 px-es csempékkel…", 0.72);
    const matte = await runViTMatteTiled(sourceImage, trimap, alpha, (done, total) => {
      const f = total > 0 ? done / total : 1;
      setStatus(`ViTMatte: haj/szőr finomítása – ${done}/${total} csempe`, 0.72 + 0.26 * f);
    });
    const cutout = sourceImage.clone().rgba();
    cutout.putAlpha(matte);
    finalImage = cutout;

    drawRaw(finalImage, $("final"));
    downloadButton.disabled = false;
    setStatus(`Kész. SAM2 IoU: ${sam.score.toFixed(3)}. A végső maszkot a ViTMatte finomította.`, 1);
  } catch (err) {
    fail(err);
  } finally {
    runButton.disabled = !sourceImage || !navigator.gpu;
  }
});

downloadButton.addEventListener("click", async () => {
  if (!finalImage) return;
  const canvas = rawToCanvas(finalImage);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ben2-vitmatte.png";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

async function ensureModels() {
  if (!ben2Pipe) {
    setStatus("BEN2 modell betöltése (~219 MB első alkalommal)…", 0.05);
    ben2Pipe = await pipeline(
      "background-removal",
      "onnx-community/BEN2-ONNX",
      {
        device: "webgpu",
        dtype: "fp16",
        progress_callback: modelProgress(0.05, 0.35, "BEN2"),
      }
    );
  }

  if (!matteProcessor) {
    setStatus("ViTMatte processor betöltése…", 0.36);
    matteProcessor = await AutoProcessor.from_pretrained(
      "Xenova/vitmatte-small-composition-1k",
      { progress_callback: modelProgress(0.36, 0.42, "ViTMatte processor") }
    );
  }

  if (!matteModel) {
    setStatus("ViTMatte modell betöltése (~100 MB első alkalommal)…", 0.43);
    matteModel = await VitMatteForImageMatting.from_pretrained(
      "Xenova/vitmatte-small-composition-1k",
      {
        device: "webgpu",
        dtype: "fp32",
        progress_callback: modelProgress(0.43, 0.53, "ViTMatte"),
      }
    );
  }
}

function modelProgress(from, to, name) {
  return (p) => {
    if (p?.status === "progress" && Number.isFinite(p.progress)) {
      const f = from + (to - from) * (p.progress / 100);
      setStatus(`${name} betöltése: ${Math.round(p.progress)}%`, f);
    }
  };
}

function alphaFromBen2(img) {
  if (img.channels === 4) {
    return img.split()[3];
  }
  if (img.channels === 1) {
    return img.clone();
  }
  throw new Error(`BEN2 váratlan kimenet: ${img.channels} csatorna.`);
}

function pickSubjectSeed(alpha) {
  const w = alpha.width;
  const h = alpha.height;
  const a = alpha.data;
  let sumW = 0;
  let sumX = 0;
  let sumY = 0;
  let threshold = 190;

  const accumulate = (t) => {
    sumW = sumX = sumY = 0;
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const v = a[row + x];
        if (v < t) continue;
        const weight = (v / 255) ** 4;
        sumW += weight;
        sumX += x * weight;
        sumY += y * weight;
      }
    }
  };

  accumulate(threshold);
  if (sumW < 1) {
    threshold = 100;
    accumulate(threshold);
  }
  if (sumW < 1) return { x: w / 2, y: h / 2 };

  const cx = sumX / sumW;
  const cy = sumY / sumW;

  // A súlyozott középpont könnyen háttérre eshet (pl. két kar között).
  // Keressük meg a közelében a legközelebbi nagy biztonságú BEN2 pixelt.
  let best = null;
  let bestD2 = Infinity;
  const search = Math.max(48, Math.round(Math.min(w, h) * 0.25));
  const x0 = Math.max(0, Math.floor(cx - search));
  const x1 = Math.min(w - 1, Math.ceil(cx + search));
  const y0 = Math.max(0, Math.floor(cy - search));
  const y1 = Math.min(h - 1, Math.ceil(cy + search));

  for (let y = y0; y <= y1; y += 2) {
    const row = y * w;
    for (let x = x0; x <= x1; x += 2) {
      if (a[row + x] < threshold) continue;
      const dx = x - cx;
      const dy = y - cy;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
        bestD2 = d2;
        best = { x, y };
      }
    }
  }

  return best ?? { x: cx, y: cy };
}

function makeSamTrimap(alpha, samMask, innerRadius, outerRadius) {
  const w = alpha.width;
  const h = alpha.height;
  const n = w * h;
  if (samMask.width !== w || samMask.height !== h) {
    throw new Error("SAM2 és BEN2 maszk mérete nem egyezik.");
  }

  const sam = new Uint8Array(n);
  for (let i = 0; i < n; i++) sam[i] = samMask.data[i] >= 128 ? 1 : 0;

  const sure = innerRadius > 0 ? erodeBinary(sam, w, h, innerRadius) : sam;
  const allowed = outerRadius > 0 ? dilateBinary(sam, w, h, outerRadius) : sam;
  const out = new Uint8ClampedArray(n);

  for (let i = 0; i < n; i++) {
    if (!allowed[i]) {
      // A SAM2 szerint ez nem a fő objektum: ne engedjük, hogy BEN2 háttérdarabokat
      // (szék, kanapé, növény stb.) végleges előtérnek zároljon.
      out[i] = 0;
    } else if (sure[i] && alpha.data[i] >= 32) {
      // A fő objektum belseje: biztos előtér.
      out[i] = 255;
    } else {
      // A SAM2 határa körüli széles sávot a ViTMatte döntse el.
      // Ebbe beleférnek a SAM által elvesztett haj- és szőrszálak is.
      out[i] = 128;
    }
  }

  return new RawImage(out, w, h, 1);
}

async function runViTMatteTiled(image, trimap, benAlpha, progress) {
  const w = image.width;
  const h = image.height;
  if (trimap.width !== w || trimap.height !== h || benAlpha.width !== w || benAlpha.height !== h) {
    throw new Error("A BEN2 maszk, trimap és eredeti kép mérete nem egyezik.");
  }

  const stride = VITMATTE_TILE - VITMATTE_OVERLAP;
  const xs = tileStarts(w, VITMATTE_TILE, stride);
  const ys = tileStarts(h, VITMATTE_TILE, stride);
  const tiles = [];

  for (const y of ys) {
    for (const x of xs) {
      if (tileHasUnknown(trimap, x, y, VITMATTE_TILE)) {
        tiles.push([x, y]);
      }
    }
  }

  const accum = new Float32Array(w * h);
  const weights = new Float32Array(w * h);
  const finalAlpha = new Uint8ClampedArray(w * h);

  // A biztos területek 0/255 maradnak. A bizonytalan terület addig BEN2 fallback.
  for (let i = 0; i < finalAlpha.length; i++) {
    const t = trimap.data[i];
    finalAlpha[i] = t === 0 ? 0 : t === 255 ? 255 : benAlpha.data[i];
  }

  if (tiles.length === 0) {
    progress?.(0, 0);
    return new RawImage(finalAlpha, w, h, 1);
  }

  let done = 0;
  for (const [startX, startY] of tiles) {
    const imageTile = cropPadRaw(image, startX, startY, VITMATTE_TILE);
    const trimapTile = cropPadRaw(trimap, startX, startY, VITMATTE_TILE);
    const inputs = await matteProcessor(imageTile, trimapTile);
    const { alphas } = await matteModel(inputs);
    const matteTile = alphaTensorToRaw(alphas, VITMATTE_TILE, VITMATTE_TILE);

    const actualW = Math.min(VITMATTE_TILE, w - startX);
    const actualH = Math.min(VITMATTE_TILE, h - startY);

    for (let ty = 0; ty < actualH; ty++) {
      const gy = startY + ty;
      for (let tx = 0; tx < actualW; tx++) {
        const gx = startX + tx;
        const gi = gy * w + gx;
        if (trimap.data[gi] !== 128) continue;

        const wx = overlapWeight(tx, startX, w, VITMATTE_TILE, VITMATTE_OVERLAP);
        const wy = overlapWeight(ty, startY, h, VITMATTE_TILE, VITMATTE_OVERLAP);
        const weight = Math.max(0.001, wx * wy);
        const alpha01 = matteTile.data[ty * VITMATTE_TILE + tx] / 255;
        accum[gi] += alpha01 * weight;
        weights[gi] += weight;
      }
    }

    done++;
    progress?.(done, tiles.length);
    await yieldToBrowser();
  }

  for (let i = 0; i < finalAlpha.length; i++) {
    if (trimap.data[i] === 128 && weights[i] > 0) {
      finalAlpha[i] = Math.round(clamp(accum[i] / weights[i], 0, 1) * 255);
    }
  }

  return new RawImage(finalAlpha, w, h, 1);
}

function tileStarts(length, tile, stride) {
  if (length <= tile) return [0];
  const starts = [];
  for (let s = 0; s < length - tile; s += stride) starts.push(s);
  const last = length - tile;
  if (starts[starts.length - 1] !== last) starts.push(last);
  return starts;
}

function tileHasUnknown(trimap, startX, startY, tile) {
  const maxX = Math.min(trimap.width, startX + tile);
  const maxY = Math.min(trimap.height, startY + tile);
  for (let y = startY; y < maxY; y++) {
    const row = y * trimap.width;
    for (let x = startX; x < maxX; x++) {
      if (trimap.data[row + x] === 128) return true;
    }
  }
  return false;
}

function cropPadRaw(raw, startX, startY, size) {
  const channels = raw.channels;
  const out = new Uint8ClampedArray(size * size * channels);
  const src = raw.data;

  for (let y = 0; y < size; y++) {
    const sy = clamp(startY + y, 0, raw.height - 1);
    for (let x = 0; x < size; x++) {
      const sx = clamp(startX + x, 0, raw.width - 1);
      const si = (sy * raw.width + sx) * channels;
      const di = (y * size + x) * channels;
      for (let c = 0; c < channels; c++) out[di + c] = src[si + c];
    }
  }

  return new RawImage(out, size, size, channels);
}

function overlapWeight(pos, globalStart, fullLength, tile, overlap) {
  let weight = 1;
  if (globalStart > 0 && pos < overlap) {
    weight *= smooth01((pos + 1) / (overlap + 1));
  }
  if (globalStart + tile < fullLength && pos >= tile - overlap) {
    weight *= smooth01((tile - pos) / (overlap + 1));
  }
  return weight;
}

function smooth01(t) {
  t = clamp(t, 0, 1);
  return t * t * (3 - 2 * t);
}

function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function dilateBinary(src, w, h, radius) {
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

function erodeBinary(src, w, h, radius) {
  if (radius <= 0) return new Uint8Array(src);
  const inv = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) inv[i] = src[i] ? 0 : 1;
  const expandedBg = dilateBinary(inv, w, h, radius);
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = expandedBg[i] ? 0 : 1;
  return out;
}

function alphaTensorToRaw(alphas, targetW, targetH) {
  const dims = alphas.dims;
  if (!Array.isArray(dims) || dims.length < 2) {
    throw new Error("ViTMatte alpha tensor mérete nem értelmezhető.");
  }
  const outH = dims[dims.length - 2];
  const outW = dims[dims.length - 1];
  const src = alphas.data;
  const out = new Uint8ClampedArray(targetW * targetH);

  const copyW = Math.min(targetW, outW);
  const copyH = Math.min(targetH, outH);
  for (let y = 0; y < copyH; y++) {
    for (let x = 0; x < copyW; x++) {
      const v = src[y * outW + x];
      out[y * targetW + x] = Math.round(clamp(v, 0, 1) * 255);
    }
  }
  return new RawImage(out, targetW, targetH, 1);
}

async function limitImage(img, maxSide) {
  maxSide = Math.max(512, Math.round(maxSide));
  const longest = Math.max(img.width, img.height);
  if (longest <= maxSide) return img;
  const scale = maxSide / longest;
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const copy = img.clone();
  await copy.resize(w, h);
  return copy;
}

function drawRaw(raw, canvas) {
  const c = rawToCanvas(raw);
  canvas.width = c.width;
  canvas.height = c.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(c, 0, 0);
}

function rawToCanvas(raw) {
  const canvas = document.createElement("canvas");
  canvas.width = raw.width;
  canvas.height = raw.height;
  const ctx = canvas.getContext("2d");
  const rgba = new Uint8ClampedArray(raw.width * raw.height * 4);

  if (raw.channels === 4) {
    rgba.set(raw.data);
  } else if (raw.channels === 3) {
    for (let i = 0, j = 0; i < raw.data.length; i += 3, j += 4) {
      rgba[j] = raw.data[i];
      rgba[j+1] = raw.data[i+1];
      rgba[j+2] = raw.data[i+2];
      rgba[j+3] = 255;
    }
  } else if (raw.channels === 1) {
    for (let i = 0, j = 0; i < raw.data.length; i++, j += 4) {
      const v = raw.data[i];
      rgba[j] = rgba[j+1] = rgba[j+2] = v;
      rgba[j+3] = 255;
    }
  } else {
    throw new Error(`Nem támogatott csatornaszám: ${raw.channels}`);
  }

  ctx.putImageData(new ImageData(rgba, raw.width, raw.height), 0, 0);
  return canvas;
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.width = 1;
  canvas.height = 1;
}

function readNumber(id, fallback) {
  const v = Number($(id).value);
  return Number.isFinite(v) ? v : fallback;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function setStatus(text, progress) {
  statusEl.textContent = text;
  if (Number.isFinite(progress)) progressEl.value = clamp(progress, 0, 1);
}

function fail(err) {
  console.error(err);
  setStatus(`Hiba: ${err?.message ?? err}`, 0);
}
