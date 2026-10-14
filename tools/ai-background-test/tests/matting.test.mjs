import test from 'node:test';
import assert from 'node:assert/strict';
import { mask, buildTrimap, resizeMask, subjectSupport, validateSize } from '../matting-core.mjs';
import { refineAlpha, readAlphaTensor, tileStarts, TILE } from '../matting-engine.mjs';
import { composeCutout } from '../edge-colors.mjs';
const filled = (w, h, value) => mask(new Uint8ClampedArray(w * h).fill(value), w, h);
const image = (w, h) => ({ data: new Uint8ClampedArray(w * h * 4).fill(255), width: w, height: h, channels: 4 });

// Regression: the old >=20 support threshold plus erosion permanently locked
// this weak background halo as alpha255, so ViTMatte could never remove it.
test('weak BEN2 halo is NEVER locked as opaque foreground', () => {
  const a = filled(100, 100, 0);
  for (let y = 10; y < 90; y++) for (let x = 10; x < 90; x++) a.data[y * 100 + x] = 22;
  for (let y = 30; y < 70; y++) for (let x = 30; x < 70; x++) a.data[y * 100 + x] = 254;
  const t = buildTrimap(a, { inner: 6, outer: 24 });
  assert.equal(t.data[50 * 100 + 20], 128);
  assert.equal(t.data[50 * 100 + 50], 255);
  assert.equal(t.data[0], 0);
  for (let i = 0; i < a.data.length; i++) if (a.data[i] < 250) assert.notEqual(t.data[i], 255);
});
test('hair gaps and semi-transparent strands remain unknown, not closed or binarised', () => {
  const a = filled(80, 80, 0);
  for (let y = 20; y < 60; y++) for (let x = 20; x < 60; x++) a.data[y * 80 + x] = 254;
  a.data[40 * 80 + 40] = 0;
  a.data[40 * 80 + 15] = 72;
  const t = buildTrimap(a, { inner: 4, outer: 12 });
  assert.equal(t.data[40 * 80 + 40], 128);
  assert.equal(t.data[40 * 80 + 15], 128);
});
test('speck filtering keeps multiple subjects and nearby thin detail in matting band', () => {
  const a = filled(200, 200, 0);
  for (let y = 20; y < 60; y++) for (let x = 20; x < 60; x++) a.data[y * 200 + x] = 254;
  for (let y = 100; y < 160; y++) for (let x = 100; x < 160; x++) a.data[y * 200 + x] = 254;
  a.data[1000] = 254; a.data[30 * 200 + 15] = 254;
  const t = buildTrimap(a, { inner: 2, outer: 8 });
  assert.equal(t.data[1000], 0);
  assert.equal(t.data[30 * 200 + 15], 128);
  assert.equal(t.data[130 * 200 + 130], 255);
  assert.equal(t.data[40 * 200 + 40], 255);
  assert.equal(subjectSupport(a, false)[1000], 1);
});
test('explicit corrections can restore or remove areas, but never resurrect source transparency', () => {
  const a = filled(20, 20, 254), marks = new Uint8Array(400), sourceAlpha = new Uint8Array(400).fill(255);
  marks[200] = 1; marks[201] = 2; marks[202] = 3; sourceAlpha[201] = 0;
  const t = buildTrimap(a, { marks, sourceAlpha, inner: 1, outer: 4 });
  assert.equal(t.data[200], 0); assert.equal(t.data[201], 0); assert.equal(t.data[202], 128);
});
test('bilinear resize clamps edge coordinates without leaking the next pixel', () => {
  const a = mask(new Uint8ClampedArray([0, 255, 255, 0]), 2, 2), r = resizeMask(a, 4, 4);
  assert.equal(r.data[0], 0); assert.equal(r.data[3], 255); assert.equal(r.data[12], 255); assert.equal(r.data[15], 0);
  assert.equal(r.data[1], 64);
  assert.deepEqual([...new Set(resizeMask(a, 7, 7, false).data)].sort(), [0, 255]);
});
test('pixel bounds and malformed masks reject before allocating', () => {
  assert.throws(() => validateSize(100000, 100000)); assert.throws(() => validateSize(NaN, 1));
  assert.throws(() => mask(new Uint8Array(2), 2, 2));
  assert.throws(() => buildTrimap(filled(4, 4, 0), { outer: Infinity }));
});
test('tile starts cover borders and use a single tile for small images', () => {
  assert.deepEqual(tileStarts(100), [0]); assert.deepEqual(tileStarts(513), [0, 1]);
  const starts = tileStarts(1365); assert.equal(starts.at(-1), 853);
  for (let x = 0; x < 1365; x++) assert.ok(starts.some((s) => x >= s && x < s + TILE));
  assert.throws(() => tileStarts(20, 512, 0));
});
test('tensor shape mismatch and NaN cannot silently produce empty/black output', () => {
  assert.throws(() => readAlphaTensor({ dims: [1, 1, 1, 1], data: new Float32Array([1]) }));
  assert.throws(() => readAlphaTensor({ dims: [2, 1, 512, 512], data: new Float32Array(2 * 512 * 512) }));
  const a = new Float32Array(TILE * TILE); a[100] = NaN;
  assert.throws(() => readAlphaTensor({ dims: [1, 1, TILE, TILE], data: a }));
});
test('overlap blending preserves soft alpha without seams; hard seeds stay exact', async () => {
  const w = 905, h = 537, im = image(w, h), trimap = filled(w, h, 128);
  trimap.data[0] = 0; trimap.data[1] = 255;
  let calls = 0;
  const alpha = await refineAlpha(im, trimap, async (tile, tm) => {
    assert.equal(tile.width, TILE); assert.equal(tm.height, TILE); calls++;
    return new Float32Array(TILE * TILE).fill(.371);
  });
  assert.equal(calls, 6); assert.equal(alpha.data[0], 0); assert.equal(alpha.data[1], 255);
  for (let i = 2; i < alpha.data.length; i++) assert.equal(alpha.data[i], 95);
});
test('small images pad safely; fully known regions do not call the model', async () => {
  const out = await refineAlpha(image(7, 3), filled(7, 3, 128), async (tile) => {
    assert.equal(tile.data.at(-1), 255); return new Float32Array(TILE * TILE).fill(.5);
  });
  assert.ok(out.data.every((v) => v === 128));
  const noCall = await refineAlpha(image(7, 3), filled(7, 3, 255), () => { throw new Error('must not run'); });
  assert.ok(noCall.data.every((v) => v === 255));
});
test('cancelled and incomplete/invalid inference never exports a fallback as success', async () => {
  const controller = new AbortController(); controller.abort();
  await assert.rejects(refineAlpha(image(20, 20), filled(20, 20, 128), () => {}, { signal: controller.signal }), { name: 'AbortError' });
  await assert.rejects(refineAlpha(image(20, 20), filled(20, 20, 128), async () => new Float32Array(4)));
  await assert.rejects(refineAlpha(image(20, 20), filled(20, 20, 128), async () => new Float32Array(TILE * TILE).fill(NaN)));
});
test('source alpha is multiplied, never overwritten; opaque colours stay exact', () => {
  const src = { width: 3, height: 1, channels: 4, data: new Uint8ClampedArray([50,60,70,0, 50,60,70,128, 50,60,70,255]) };
  const out = composeCutout(src, mask(new Uint8ClampedArray([255,128,255]), 3, 1), 1);
  assert.deepEqual([...out.data], [0,0,0,0, 50,60,70,64, 50,60,70,255]);
});
test('edge unmixing reduces white-background contamination without darkening opaque foreground', () => {
  const src = { width: 3, height: 1, channels: 4, data: new Uint8ClampedArray([255,255,255,255, 148,148,148,255, 40,40,40,255]) };
  const matte = mask(new Uint8ClampedArray([0,128,255]), 3, 1);
  const clean = composeCutout(src, matte, 1), unchanged = composeCutout(src, matte, 0);
  assert.ok(Math.abs(clean.data[4] - 40) <= 2);
  assert.equal(clean.data[8], 40); assert.equal(clean.data[7], 128); assert.equal(unchanged.data[4], 148);
});
