// DOM/controller checks, not a substitute for a WebGPU/browser rendering test.
// Optional test dependencies: jsdom@26.1.0 and @napi-rs/canvas.
const { JSDOM } = require('jsdom');
const { createCanvas, loadImage, ImageData } = require('@napi-rs/canvas');
const vm = require('node:vm'), fs = require('node:fs/promises'), path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');

(async () => {
  const dom = new JSDOM(await fs.readFile(path.join(root, 'index.html'), 'utf8'));
  const { document } = dom.window, backing = new WeakMap(), blobs = new Map();
  const state = { subjectCalls: 0, tileCalls: 0, disposedInputs: 0, disposedOutputs: 0, downloads: [], segmentSize: null };
  function nativeCanvas(el) {
    let c = backing.get(el);
    if (!c) { c = createCanvas(el.width, el.height); backing.set(el, c); }
    if (c.width !== el.width) c.width = el.width;
    if (c.height !== el.height) c.height = el.height;
    return c;
  }
  dom.window.HTMLCanvasElement.prototype.getContext = function () { return nativeCanvas(this).getContext('2d'); };
  dom.window.HTMLCanvasElement.prototype.toBlob = function (callback) { callback(new Blob([nativeCanvas(this).toBuffer('image/png')], { type: 'image/png' })); };
  dom.window.HTMLCanvasElement.prototype.setPointerCapture = () => {};
  dom.window.HTMLCanvasElement.prototype.getBoundingClientRect = function () { return { left: 0, top: 0, width: this.width, height: this.height }; };
  dom.window.HTMLAnchorElement.prototype.click = function () { state.downloads.push({ name: this.download, blob: blobs.get(this.href) }); };
  const context = vm.createContext({ document, navigator: { gpu: { requestAdapter: async () => ({}) } },
    console, ImageData, Uint8ClampedArray, Uint8Array, Uint16Array, Int32Array, Uint32Array, Float32Array,
    DOMException, AbortController, Blob, Map, Set, Math, Number, Object, Array, Promise,
    setTimeout: (fn, ms) => { const t = setTimeout(fn, ms); if (ms > 1000) t.unref(); return t; },
    URL: { createObjectURL: (blob) => { const id = `blob:qa-${blobs.size}`; blobs.set(id, blob); return id; }, revokeObjectURL: (id) => blobs.delete(id) },
    createImageBitmap: async (file) => { const im = await loadImage(file.bytes); im.close = () => {}; return im; },
  });
  class RawImage {
    constructor(data, width, height, channels) { Object.assign(this, { data, width, height, channels }); }
    rgb() { return this; }
    async resize(width, height) { return new RawImage(new Uint8ClampedArray(width * height * this.channels), width, height, this.channels); }
  }
  const mock = {
    RawImage,
    pipeline: async () => {
      const fn = async (im) => {
        state.subjectCalls++; state.segmentSize = [im.width, im.height];
        const data = new Uint8ClampedArray(im.width * im.height);
        for (let y = 0; y < im.height; y++) for (let x = 0; x < im.width; x++) {
          data[y * im.width + x] = Math.hypot((x / im.width - .5) * 2, (y / im.height - .5) * 2) < .65 ? 254 : 0;
        }
        return [{ mask: new RawImage(data, im.width, im.height, 1) }];
      };
      fn.dispose = async () => { state.benDisposed = true; }; return fn;
    },
    AutoProcessor: { from_pretrained: async () => async (image, trimap) => ({ trimap, pixel_values: { dispose: () => state.disposedInputs++ } }) },
    VitMatteForImageMatting: { from_pretrained: async () => {
      const fn = async ({ trimap }) => {
        state.tileCalls++; await new Promise((r) => setTimeout(r, 5));
        const data = new Float32Array(512 * 512);
        for (let i = 0; i < data.length; i++) data[i] = trimap.data[i] / 255;
        return { alphas: { dims: [1,1,512,512], data, dispose: () => state.disposedOutputs++ } };
      }; fn.dispose = async () => {}; return fn;
    } },
  };
  const hf = new vm.SyntheticModule(Object.keys(mock), function () { for (const [key, value] of Object.entries(mock)) this.setExport(key, value); }, { context });
  await hf.link(() => {}); await hf.evaluate();
  const modules = new Map();
  async function moduleAt(filename) {
    filename = filename.split('?')[0];
    if (modules.has(filename)) return modules.get(filename);
    const mod = new vm.SourceTextModule(await fs.readFile(filename, 'utf8'), {
      context, identifier: filename,
      importModuleDynamically: async (specifier) => { assert.equal(specifier, 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1'); return hf; },
    });
    modules.set(filename, mod);
    await mod.link((specifier, referring) => moduleAt(path.resolve(path.dirname(referring.identifier), specifier)));
    return mod;
  }
  const app = await moduleAt(path.join(root, 'app.js')); await app.evaluate();
  const el = (id) => document.getElementById(id);
  const change = (id) => el(id).dispatchEvent(new dom.window.Event('change'));
  const wait = async (condition) => { for (let i = 0; i < 500; i++) { if (condition()) return; await new Promise((r) => setTimeout(r, 5)); } throw Error(`Timed out: ${el('status').textContent}`); };
  async function load(w, h) {
    const c = createCanvas(w, h), ctx = c.getContext('2d'); ctx.fillStyle = '#a68b6f'; ctx.fillRect(0, 0, w, h);
    const bytes = c.toBuffer('image/png');
    Object.defineProperty(el('file'), 'files', { configurable: true, value: [{ name: 'synthetic.png', type: 'image/png', size: bytes.length, bytes }] });
    change('file'); await wait(() => !el('run').disabled);
  }
  await load(600, 800); el('run').click();
  assert.equal(el('file').disabled, true);
  await wait(() => !el('download').disabled);
  assert.equal(state.subjectCalls, 1); assert.ok(state.benDisposed);
  assert.equal(state.tileCalls, state.disposedInputs); assert.equal(state.tileCalls, state.disposedOutputs);
  el('background').value = 'white'; change('background'); el('download').click(); await wait(() => state.downloads.length === 1);
  const exported = await loadImage(Buffer.from(await state.downloads[0].blob.arrayBuffer()));
  assert.deepEqual([exported.width, exported.height], [600, 800]);
  const c = createCanvas(600, 800); c.getContext('2d').drawImage(exported, 0, 0);
  assert.equal(c.getContext('2d').getImageData(0, 0, 1, 1).data[3], 0, 'white preview must not flatten PNG');
  assert.equal(state.downloads[0].name, 'synthetic-hatter-nelkul.png');
  el('brush').value = '1'; change('brush');
  function pointer(type) {
    const event = new dom.window.Event(type); Object.assign(event, { clientX: 300, clientY: 300, button: 0, pointerId: 1 }); el('overlay').dispatchEvent(event);
  }
  pointer('pointerdown'); pointer('pointerup');
  assert.equal(el('download').disabled, true); assert.equal(el('undo').disabled, false);
  el('undo').click(); el('run').click(); await wait(() => !el('download').disabled); assert.equal(state.subjectCalls, 1);
  el('run').click(); el('cancel').click(); await wait(() => !el('run').disabled); assert.equal(el('download').disabled, true);
  await load(3000, 300); el('run').click(); await wait(() => !el('download').disabled);
  assert.deepEqual(state.segmentSize, [2048, 205], 'analysis must use returned resized image');
  assert.equal(el('final').width, 3000); assert.equal(el('final').height, 300);
  console.log('PASS: DOM import, GPU guard, processing locks, tensor disposal, transparent PNG export, brush/undo, cached rerun, cancellation, analysis resize and original-size output.');
})().catch((err) => { console.error(err); process.exitCode = 1; });
