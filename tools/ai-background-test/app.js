import {
  pipeline,
  AutoProcessor,
  VitMatteForImageMatting,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

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

gpuEl.textContent = navigator.gpu
  ? "WebGPU: elérhető – GPU gyorsítás használható."
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
    const bgThreshold = clamp(readNumber("bgThreshold", 0.03), 0, 0.49);
    const fgThreshold = clamp(readNumber("fgThreshold", 0.97), 0.51, 1);
    const radius = Math.max(0, Math.round(readNumber("edgeRadius", 10)));

    setStatus("Trimap készítése a BEN2 maszkból…", 0.68);
    const trimap = makeTrimap(alpha, bgThreshold, fgThreshold, radius);
    drawRaw(trimap, $("trimap"));

    setStatus("ViTMatte: haj/szőr és félig áttetsző perem finomítása…", 0.75);
    const inputs = await matteProcessor(sourceImage, trimap);
    const { alphas } = await matteModel(inputs);

    const matte = alphaTensorToRaw(alphas, sourceImage.width, sourceImage.height);
    const cutout = sourceImage.clone().rgba();
    cutout.putAlpha(matte);
    finalImage = cutout;

    drawRaw(finalImage, $("final"));
    downloadButton.disabled = false;
    setStatus("Kész. A jobb alsó kép a BEN2 + ViTMatte eredmény.", 1);
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

function makeTrimap(alpha, bgThreshold, fgThreshold, radius) {
  const w = alpha.width;
  const h = alpha.height;
  const n = w * h;
  const a = alpha.data;
  const bg = Math.round(bgThreshold * 255);
  const fg = Math.round(fgThreshold * 255);

  const uncertain = new Uint8Array(n);
  const hard = new Uint8Array(n);

  for (let i = 0; i < n; i++) {
    const v = a[i];
    if (v <= bg) hard[i] = 0;
    else if (v >= fg) hard[i] = 255;
    else {
      hard[i] = v >= 128 ? 255 : 0;
      uncertain[i] = 1;
    }
  }

  // A tiszta 0/255 határ körül is legyen bizonytalan sáv, mert a BEN2
  // ott túl kemény kontúrt adhat akkor is, ha az alpha már telített.
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const c = hard[i];
      if (
        hard[i - 1] !== c || hard[i + 1] !== c ||
        hard[i - w] !== c || hard[i + w] !== c
      ) {
        uncertain[i] = 1;
      }
    }
  }

  const expanded = radius > 0 ? dilateBinary(uncertain, w, h, radius) : uncertain;
  const out = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i++) {
    out[i] = expanded[i] ? 128 : hard[i];
  }
  return new RawImage(out, w, h, 1);
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
