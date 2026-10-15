// Execute the real generated worker control flow with a mocked model/runtime.
// This verifies failure handling and FP32 configuration, not GPU model quality.
import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';

async function exercise(values, { gpu = true, createError = null } = {}) {
  let handler, options, releases = 0, disposals = 0;
  const events = [], pixels = new Uint8ClampedArray(2048 * 2048 * 4).fill(255);
  class Canvas {
    getContext() { return { putImageData() {}, fillRect() {}, drawImage() {}, getImageData() { return { data: pixels }; } }; }
  }
  const tensor = { type: 'float32', dims: [1, 1, 2048, 2048], data: values, dispose() { disposals++; } };
  const testOrt = {
    env: { wasm: {}, webgpu: {} },
    Tensor: class { dispose() { disposals++; } },
    InferenceSession: { async create(bytes, config) {
      options = config; if (createError) throw createError;
      return { inputNames: ['image'], outputNames: ['alpha'], async run() { return { alpha: tensor }; }, async release() { releases++; } };
    } }
  };
  const self = { navigator: { gpu: gpu ? { async requestAdapter() { return { features: new Set(), info: { vendor: 'test' } }; } } : undefined },
    postMessage(event) { events.push(structuredClone(event)); },
    addEventListener(type, fn) { if (type === 'message') handler = fn; } };
  const context = vm.createContext({ self, testOrt, TextEncoder, TextDecoder, ArrayBuffer, Uint8Array, Uint8ClampedArray, Uint32Array, Float32Array, performance, setTimeout, clearTimeout, OffscreenCanvas: Canvas, ImageData: class {} });
  let worker = await readFile(new URL('../worker.js', import.meta.url), 'utf8');
  const importLine = 'const ort = await import(RUNTIME_URL);';
  assert.ok(worker.includes(importLine)); worker = worker.replace(importLine, 'const ort = testOrt;');
  vm.runInContext(worker, context);
  vm.runInContext('modelBytes = async () => new Uint8Array(1); adaptPinnedModel = bytes => bytes;', context);
  const configuration = vm.runInContext('({ MODEL_URL, MODEL_BYTES, MODEL_HASH, CACHE_NAME })', context);
  await handler({ data: { type: 'run', id: 7, width: 2, height: 1, rgba: new Uint8ClampedArray([40,50,60,255,10,20,30,128]).buffer, strength: 0 } });
  return { events, options, configuration, releases, disposals };
}

test('FP32 model, basic optimization and no shader-f16 dependency; valid soft alpha survives', async () => {
  const result = await exercise(new Float32Array(2048 * 2048).fill(.5));
  assert.ok(result.configuration.MODEL_URL.endsWith('/model.onnx'));
  assert.equal(result.configuration.MODEL_BYTES, 932150975);
  assert.equal(result.configuration.MODEL_HASH, 'a548a1f3307250766bef50b174026b3cfa39a695bc1e42ec131c14fe82797ffd');
  assert.equal(result.configuration.CACHE_NAME, 'photocraft-birefnet-hr-fp32-v2');
  assert.equal(result.options.graphOptimizationLevel, 'basic');
  const done = result.events.find(e => e.type === 'result');
  assert.ok(done); assert.deepEqual([...new Uint8ClampedArray(done.alpha)], [128, 64]);
  assert.equal(result.events.some(e => e.type === 'error'), false);
  assert.equal(result.events.find(e => e.type === 'metrics').precision, 'FP32');
  assert.equal(result.releases, 1); assert.equal(result.disposals, 2);
});

test('NaN, Infinity and out-of-range masks produce diagnostics and never an exported result', async () => {
  const values = new Float32Array(2048 * 2048).fill(.25);
  values[0] = NaN; values[4] = Infinity; values[8] = -Infinity; values[9] = -2; values[10] = 2;
  const result = await exercise(values);
  assert.equal(result.events.some(e => e.type === 'result'), false);
  assert.equal(result.events.some(e => e.type === 'metrics'), false);
  assert.match(result.events.find(e => e.type === 'error').message, /5 \/ 4194304/);
  const error = result.events.filter(e => e.type === 'diagnostics').at(-1).report.error;
  assert.equal(error.code, 'MODEL_ALPHA_INVALID');
  assert.deepEqual(error.alphaStats, { total: 4194304, nan: 1, infinite: 2, outOfRange: 2, invalid: 5, min: -2, max: 2, firstInvalidIndex: 0 });
  assert.equal(result.releases, 1); assert.equal(result.disposals, 2);
  assert.ok(!JSON.stringify(error).includes('rgba'));
});

test('missing WebGPU and session allocation failure leave no result and return a report', async () => {
  for (const config of [{ gpu: false }, { createError: new Error('allocation failed') }]) {
    const result = await exercise(new Float32Array(2048 * 2048), config);
    assert.equal(result.events.some(e => e.type === 'result'), false);
    assert.ok(result.events.some(e => e.type === 'error'));
    assert.equal(result.events.filter(e => e.type === 'diagnostics').at(-1).report.result, 'error');
  }
});
