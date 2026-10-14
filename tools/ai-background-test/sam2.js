import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";

const MODEL_SIZE = 1024;
const MASK_INPUT_SIZE = 256;
const ENCODER_URL = "https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx/resolve/main/encoder.with_runtime_opt.ort";
const DECODER_URL = "https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx/resolve/main/decoder.onnx";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

let sessionsPromise = null;

export async function sam2Subject(raw, seed, onStatus) {
  if (!navigator.gpu) throw new Error("SAM2 WebGPU-t igényel.");
  const { encoder, decoder } = await getSessions(onStatus);

  onStatus?.("SAM2: kép előfeldolgozása…", 0.58);
  const imageTensor = preprocess(raw);

  onStatus?.("SAM2: fő alany kódolása…", 0.60);
  const encoded = await encoder.run({ image: imageTensor });

  const scale = MODEL_SIZE / Math.max(raw.width, raw.height);
  const sx = clamp(seed.x * scale, 0, MODEL_SIZE - 1);
  const sy = clamp(seed.y * scale, 0, MODEL_SIZE - 1);

  const point_coords = new ort.Tensor(
    "float32",
    new Float32Array([sx, sy, 0, 0]),
    [1, 2, 2],
  );
  const point_labels = new ort.Tensor(
    "float32",
    new Float32Array([1, -1]),
    [1, 2],
  );
  const mask_input = new ort.Tensor(
    "float32",
    new Float32Array(MASK_INPUT_SIZE * MASK_INPUT_SIZE),
    [1, 1, MASK_INPUT_SIZE, MASK_INPUT_SIZE],
  );
  const has_mask_input = new ort.Tensor(
    "float32",
    new Float32Array([0]),
    [1],
  );

  onStatus?.("SAM2: fő objektum kiválasztása…", 0.65);
  const decoded = await decoder.run({
    image_embed: encoded.image_embed,
    high_res_feats_0: encoded.high_res_feats_0,
    high_res_feats_1: encoded.high_res_feats_1,
    point_coords,
    point_labels,
    mask_input,
    has_mask_input,
  });

  const masks = decoded.masks;
  const scores = decoded.iou_predictions;
  const dims = masks.dims.map(Number);
  const maskH = dims[dims.length - 2];
  const maskW = dims[dims.length - 1];
  const numMasks = dims[dims.length - 3] ?? 1;
  const perMask = maskW * maskH;

  let best = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < numMasks; i++) {
    const score = Number(scores.data[i] ?? -Infinity);
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }

  const rawMasks = masks.data;
  const out = new Uint8ClampedArray(raw.width * raw.height);
  const paddedSide = Math.max(raw.width, raw.height);
  const sampleScaleX = maskW / paddedSide;
  const sampleScaleY = maskH / paddedSide;
  const offset = best * perMask;

  for (let y = 0; y < raw.height; y++) {
    const my = y * sampleScaleY;
    for (let x = 0; x < raw.width; x++) {
      const mx = x * sampleScaleX;
      const logit = bilinear(rawMasks, offset, maskW, maskH, mx, my);
      out[y * raw.width + x] = logit > 0 ? 255 : 0;
    }
  }

  return {
    data: out,
    width: raw.width,
    height: raw.height,
    score: bestScore,
    seed,
  };
}

async function getSessions(onStatus) {
  if (!sessionsPromise) {
    sessionsPromise = (async () => {
      onStatus?.("SAM2 Tiny encoder betöltése (~128 MB, első alkalommal)…", 0.53);
      const encoder = await ort.InferenceSession.create(ENCODER_URL, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "disabled",
      });

      onStatus?.("SAM2 decoder betöltése (~20 MB, első alkalommal)…", 0.56);
      const decoder = await ort.InferenceSession.create(DECODER_URL, {
        executionProviders: ["webgpu"],
      });

      return { encoder, decoder };
    })();
  }
  return sessionsPromise;
}

function preprocess(raw) {
  const src = document.createElement("canvas");
  src.width = raw.width;
  src.height = raw.height;
  const sctx = src.getContext("2d", { willReadFrequently: true });
  if (!sctx) throw new Error("SAM2: nem sikerült canvas contextet létrehozni.");

  const rgba = toRgba(raw);
  sctx.putImageData(new ImageData(rgba, raw.width, raw.height), 0, 0);

  const scale = MODEL_SIZE / Math.max(raw.width, raw.height);
  const newW = Math.max(1, Math.round(raw.width * scale));
  const newH = Math.max(1, Math.round(raw.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("SAM2: nem sikerült 1024px canvas contextet létrehozni.");

  ctx.clearRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, 0, 0, newW, newH);

  const pixels = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const count = MODEL_SIZE * MODEL_SIZE;
  const tensor = new Float32Array(3 * count);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < count; i++) {
    const j = i * 4;
    tensor[i] = (pixels[j] / 255 - mean[0]) / std[0];
    tensor[count + i] = (pixels[j + 1] / 255 - mean[1]) / std[1];
    tensor[count * 2 + i] = (pixels[j + 2] / 255 - mean[2]) / std[2];
  }

  return new ort.Tensor("float32", tensor, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function toRgba(raw) {
  const out = new Uint8ClampedArray(raw.width * raw.height * 4);
  if (raw.channels === 4) {
    out.set(raw.data);
    return out;
  }
  if (raw.channels === 3) {
    for (let i = 0, j = 0; i < raw.data.length; i += 3, j += 4) {
      out[j] = raw.data[i];
      out[j + 1] = raw.data[i + 1];
      out[j + 2] = raw.data[i + 2];
      out[j + 3] = 255;
    }
    return out;
  }
  if (raw.channels === 1) {
    for (let i = 0, j = 0; i < raw.data.length; i++, j += 4) {
      const v = raw.data[i];
      out[j] = out[j + 1] = out[j + 2] = v;
      out[j + 3] = 255;
    }
    return out;
  }
  throw new Error(`SAM2: nem támogatott csatornaszám: ${raw.channels}`);
}

function bilinear(data, offset, width, height, x, y) {
  const x0 = clamp(Math.floor(x), 0, width - 1);
  const y0 = clamp(Math.floor(y), 0, height - 1);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const fx = x - x0;
  const fy = y - y0;
  const v00 = Number(data[offset + y0 * width + x0]);
  const v10 = Number(data[offset + y0 * width + x1]);
  const v01 = Number(data[offset + y1 * width + x0]);
  const v11 = Number(data[offset + y1 * width + x1]);
  return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}
