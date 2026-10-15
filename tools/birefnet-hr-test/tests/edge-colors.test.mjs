import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createEdgeCleaner, composeCutout } from '../edge-colors.mjs';
import { estimateForeground } from '../foreground.mjs';
import { exportAlpha, mask } from '../core.mjs';

function fixture({ foreground = [55, 34, 18], background = [200, 215, 225], overestimate = 0 } = {}) {
  const width = 80, height = 32, rgba = new Uint8ClampedArray(width * height * 4), a = new Uint8ClampedArray(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = y * width + x;
    const coverage = Math.max(0, Math.min(1, (x - 18) / 36));
    a[i] = 255 * (coverage > 0 && coverage < 1 ? Math.min(.98, coverage + overestimate) : coverage);
    for (let c = 0; c < 3; c++) rgba[4 * i + c] = coverage * foreground[c] + (1 - coverage) * background[c];
    rgba[4 * i + 3] = 255;
  }
  return { source: { data: rgba, width, height, channels: 4 }, matte: mask(a, width, height), foreground };
}
function error(data, matte, foreground) {
  let sum = 0, n = 0;
  for (let i = 0; i < matte.data.length; i++) if (matte.data[i] > 25 && matte.data[i] < 230) {
    for (let c = 0; c < 3; c++) { sum += Math.abs(data[4 * i + c] - foreground[c]); n++; }
  }
  return sum / n;
}

test('multilevel estimate agrees with the independent upstream CPU oracle', async () => {
  const oracle = JSON.parse(await readFile(new URL('./fixtures/foreground-oracle.json', import.meta.url), 'utf8'));
  for (const f of oracle.cases) {
    const result = estimateForeground(Uint8ClampedArray.from(f.rgba), Uint8ClampedArray.from(f.alpha), f.width, f.height);
    let maximum = 0, total = 0;
    for (let i = 0; i < result.length; i++) { const d = Math.abs(result[i] - f.foreground[i]); maximum = Math.max(maximum, d); total += d; }
    assert.ok(maximum <= 1, `reference max RGB difference ${maximum}`);
    assert.ok(total / result.length < .05);
  }
});

test('colour recovery improves known dark and white hair without changing alpha or interiors', () => {
  for (const options of [{}, { foreground: [245, 240, 235], background: [30, 45, 70] }]) {
    const { source, matte, foreground } = fixture(options);
    const cleaner = createEdgeCleaner(source, matte), raw = cleaner.compose(0), full = cleaner.compose(1);
    assert.ok(error(full.data, matte, foreground) < error(raw.data, matte, foreground) * .3);
    assert.deepEqual(exportAlpha(full.data), exportAlpha(raw.data));
    for (let i = 0; i < matte.data.length; i++) if (matte.data[i] === 255) assert.deepEqual(full.data.slice(i * 4, i * 4 + 4), source.data.slice(i * 4, i * 4 + 4));
  }
});

test('a partially opaque contaminated fringe is corrected while preserving thin coverage', () => {
  const { source, matte, foreground } = fixture({ overestimate: .25 });
  const cleaner = createEdgeCleaner(source, matte), raw = cleaner.compose(0), full = cleaner.compose(1);
  assert.ok(error(full.data, matte, foreground) < error(raw.data, matte, foreground) * .65);
  assert.deepEqual(exportAlpha(full.data), exportAlpha(raw.data));
});

test('10/20/30 strengths are distinct; cache is prepared once; zero restores raw exactly', () => {
  const { source, matte } = fixture(), cleaner = createEdgeCleaner(source, matte);
  const raw = cleaner.compose(0); assert.equal(cleaner.info.preparations, 0);
  const results = [.1, .2, .3].map(s => cleaner.compose(s).data);
  assert.notDeepEqual(results[0], results[1]); assert.notDeepEqual(results[1], results[2]);
  assert.equal(cleaner.info.preparations, 1);
  assert.deepEqual(cleaner.compose(0).data, raw.data);
  assert.deepEqual(source.data, fixture().source.data);
});

test('transparent inputs retain straight RGB and multiply original alpha', () => {
  const { source, matte } = fixture(); source.data[4 * 32 + 3] = 120;
  const cleaner = createEdgeCleaner(source, matte), raw = cleaner.compose(0), full = cleaner.compose(1);
  assert.deepEqual(full.data, raw.data); assert.equal(cleaner.info.status, 'transparent-source');
  assert.equal(full.data[4 * 32 + 3], Math.round(matte.data[32] / 255 * 120));
});

test('degenerate/no-context masks remain finite and unchanged; invalid parameters fail', () => {
  for (const value of [0, 128, 255]) {
    const source = { width: 1, height: 1, channels: 4, data: Uint8ClampedArray.from([50, 80, 100, 255]) };
    const m = mask(Uint8ClampedArray.of(value), 1, 1);
    assert.deepEqual(composeCutout(source, m, 1).data, composeCutout(source, m, 0).data);
    for (const s of [NaN, Infinity, -1, 1.1]) assert.throws(() => composeCutout(source, m, s));
  }
  assert.throws(() => estimateForeground(new Uint8Array(0), new Uint8Array(0), 4001, 4001));
  assert.throws(() => estimateForeground(new Uint8Array(2), new Uint8Array(1), 1, 1));
});
